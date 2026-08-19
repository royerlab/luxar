/**
 * WebGL-only blend-variant program warm-up.
 *
 * Headless Chromium often runs WebGL through SwiftShader, whose shader/program
 * link step is synchronous and can monopolize the main thread for seconds.
 * Luxar's Layers panel lets users switch blending modes at runtime, and some
 * mode transitions flip shader defines (`max`, `volumetric`, gsplat `normal`,
 * mesh `opaque` cutout). Without a warm-up pass, the first click onto each
 * variant pays that synchronous link cost on the interaction path.
 *
 * This manager pre-compiles every DISTINCT blend-variant program a live visual
 * material can reach, one compile per post-frame idle opportunity so visible
 * rendering and input get a turn before every link. Browsers without idle
 * callbacks use a conservative 50 ms delay. The warmed programs stay pinned
 * by keeper materials until the source material is replaced or disposed.
 *
 * The manager is configured only for classic `THREE.WebGLRenderer`; WebGPU/TSL
 * sessions do not use it.
 *
 * @module rendering/webgl-blend-warmup
 */

import * as THREE from 'three';
import { BLENDING_MODES, type BlendingMode } from '../types/blending';
import { log, Modules } from '../utils/log';

type WarmupNodeType = 'points' | 'lines' | 'gsplats' | 'mesh';

const BLEND_VARIANT_DEFINES = new Set([
  'LUXAR_MAX_RGB_CONTRIBUTION',
  'LUXAR_VOLUMETRIC',
  'LUXAR_NORMAL_PREMULT',
  'LUXAR_MESH_ALPHA_CUTOUT',
]);

type WarmupMaterial = THREE.Material & {
  readonly defines?: Record<string, unknown>;
  readonly vertexShader?: string;
  readonly fragmentShader?: string;
  applyBlendingMode(mode: BlendingMode): void;
};

interface WarmupConfig {
  enabled: boolean;
  renderer: THREE.WebGLRenderer | null;
  camera: THREE.Camera | null;
  targetScene: THREE.Scene | null;
}

interface WarmupTask {
  sourceMaterial: THREE.Material;
  keeperMaterial: THREE.Material;
  compileObject: THREE.Object3D;
}

interface TrackedObjectState {
  sourceMaterial: THREE.Material;
  removeHandler: () => void;
}

interface SourceWarmupState {
  fingerprint: string;
  keepers: Set<THREE.Material>;
  objects: Set<THREE.Object3D>;
  disposeHandler: () => void;
}

interface WarmCompletion {
  generation: number;
  resolve: () => void;
}

type WaitForWarmupTurn = () => Promise<void>;
type DeferActivation = (activate: () => void) => void;
type IdleCallbackScheduler = (callback: () => void, options?: { timeout: number }) => number;
type CompileOne = (
  renderer: THREE.WebGLRenderer,
  camera: THREE.Camera,
  targetScene: THREE.Scene,
  compileObject: THREE.Object3D
) => void;

const WARMUP_IDLE_TIMEOUT_MS = 250;
const WARMUP_FALLBACK_DELAY_MS = 50;

function waitForIdleOpportunity(resolve: () => void): void {
  const requestIdle = (
    globalThis as typeof globalThis & {
      requestIdleCallback?: IdleCallbackScheduler;
    }
  ).requestIdleCallback;

  if (typeof requestIdle === 'function') {
    requestIdle.call(globalThis, resolve, { timeout: WARMUP_IDLE_TIMEOUT_MS });
    return;
  }

  setTimeout(resolve, WARMUP_FALLBACK_DELAY_MS);
}

function defaultWaitForWarmupTurn(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame !== 'function') {
      waitForIdleOpportunity(resolve);
      return;
    }

    requestAnimationFrame(() => {
      waitForIdleOpportunity(resolve);
    });
  });
}

function defaultDeferActivation(activate: () => void): void {
  setTimeout(activate, 0);
}

function defaultCompileOne(
  renderer: THREE.WebGLRenderer,
  camera: THREE.Camera,
  targetScene: THREE.Scene,
  compileObject: THREE.Object3D
): void {
  const compileScene = new THREE.Scene();
  compileScene.add(compileObject);
  renderer.compile(compileScene, camera, targetScene);
  compileScene.remove(compileObject);
}

function isWarmupNodeType(value: unknown): value is WarmupNodeType {
  return value === 'points' || value === 'lines' || value === 'gsplats' || value === 'mesh';
}

function hasWarmupMaterialApi(
  material: THREE.Material | null | undefined
): material is WarmupMaterial {
  return (
    typeof (material as Partial<WarmupMaterial> | null | undefined)?.applyBlendingMode ===
    'function'
  );
}

function buildDefinesFingerprint(
  defines: Record<string, unknown> | undefined,
  includeBlendDefines: boolean
): string {
  if (!defines) return '';
  const parts = Object.entries(defines)
    .filter(([name]) => includeBlendDefines || !BLEND_VARIANT_DEFINES.has(name))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, value]) => `${name}=${String(value)}`);
  return parts.join('|');
}

function buildGeometryFingerprint(geometry: THREE.BufferGeometry | undefined): string {
  if (!geometry) return 'no-geometry';

  const attributes = Object.entries(geometry.attributes)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, attribute]) => {
      const attr = attribute as THREE.BufferAttribute & {
        readonly isInstancedBufferAttribute?: boolean;
      };
      return [
        name,
        attr.itemSize,
        attr.normalized ? 'n' : 'nn',
        attr.isInstancedBufferAttribute === true ? 'instanced' : 'plain',
      ].join(':');
    })
    .join(';');

  const morph = Object.entries(geometry.morphAttributes)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, values]) => `${name}:${values.length}`)
    .join(';');

  return [geometry.index ? 'indexed' : 'non-indexed', attributes, morph].join('::');
}

/**
 * Approximate the inputs that Three's WebGL program cache keys on.
 *
 * This is intentionally broader than the warm-up issue itself: if a material's
 * compile-time state changes for any reason other than the blend-mode defines
 * (e.g. colormap enablement, point/line/splat texture-width define, mesh
 * flat-normal variant), the service must rebuild that source material's keeper
 * set instead of pinning stale programs forever.
 */
export function buildBlendWarmupFingerprint(
  material: THREE.Material,
  object: THREE.Object3D,
  includeBlendDefines: boolean = true
): string {
  const shaderMaterial = material as THREE.Material & {
    readonly defines?: Record<string, unknown>;
    readonly vertexShader?: string;
    readonly fragmentShader?: string;
    readonly index0AttributeName?: string;
    readonly premultipliedAlpha?: boolean;
    readonly vertexColors?: boolean;
    readonly fog?: boolean;
  };
  const geometry = (object as THREE.Mesh).geometry as THREE.BufferGeometry | undefined;

  return [
    object.type,
    object instanceof THREE.InstancedMesh ? 'instanced' : 'non-instanced',
    object instanceof THREE.SkinnedMesh ? 'skinned' : 'non-skinned',
    object instanceof THREE.Points ? 'points' : 'not-points',
    object instanceof THREE.Line ? 'line' : 'not-line',
    material.type,
    material.side,
    shaderMaterial.premultipliedAlpha === true ? 'premult' : 'straight',
    shaderMaterial.vertexColors === true ? 'vertex-colors' : 'no-vertex-colors',
    shaderMaterial.fog === true ? 'fog' : 'no-fog',
    shaderMaterial.index0AttributeName ?? '',
    buildDefinesFingerprint(shaderMaterial.defines, includeBlendDefines),
    shaderMaterial.vertexShader ?? '',
    shaderMaterial.fragmentShader ?? '',
    shaderMaterial.customProgramCacheKey(),
    buildGeometryFingerprint(geometry),
  ].join('@@');
}

function hasRenderableContent(object: THREE.Object3D, nodeType: WarmupNodeType): boolean {
  const userData = object.userData as {
    visiblePointCount?: number;
    visibleSegmentCount?: number;
    visibleSplatCount?: number;
    visibleTriangleCount?: number;
  };

  switch (nodeType) {
    case 'points':
      return (userData.visiblePointCount ?? 0) > 0;
    case 'lines':
      return (userData.visibleSegmentCount ?? 0) > 0;
    case 'gsplats':
      return (userData.visibleSplatCount ?? 0) > 0;
    case 'mesh':
      if ((userData.visibleTriangleCount ?? 0) > 0) return true;
      return ((object as THREE.Mesh).geometry?.index?.count ?? 0) / 3 > 0;
  }
}

function createCompileObject(source: THREE.Object3D, material: THREE.Material): THREE.Object3D {
  const geometry = (source as THREE.Mesh | THREE.Line | THREE.Points).geometry;
  let compileObject: THREE.Object3D;

  if (source instanceof THREE.InstancedMesh) {
    const instanced = new THREE.InstancedMesh(geometry, material, source.count);
    if (source.instanceColor) instanced.instanceColor = source.instanceColor;
    compileObject = instanced;
  } else if (source instanceof THREE.Mesh) {
    compileObject = new THREE.Mesh(geometry, material);
  } else if (source instanceof THREE.LineSegments) {
    compileObject = new THREE.LineSegments(geometry, material);
  } else if (source instanceof THREE.LineLoop) {
    compileObject = new THREE.LineLoop(geometry, material);
  } else if (source instanceof THREE.Line) {
    compileObject = new THREE.Line(geometry, material);
  } else {
    compileObject = new THREE.Points(geometry, material);
  }

  compileObject.name = source.name;
  compileObject.layers.mask = source.layers.mask;
  compileObject.visible = true;
  compileObject.castShadow = source.castShadow;
  compileObject.receiveShadow = source.receiveShadow;
  compileObject.frustumCulled = false;
  compileObject.renderOrder = source.renderOrder;
  compileObject.matrix.copy(source.matrix);
  compileObject.matrixWorld.copy(source.matrixWorld);
  compileObject.matrixAutoUpdate = false;
  compileObject.matrixWorldAutoUpdate = false;
  compileObject.matrixWorldNeedsUpdate = false;
  return compileObject;
}

export class WebGLBlendWarmupManager {
  private enabled = false;
  private armed = false;
  private renderer: THREE.WebGLRenderer | null = null;
  private camera: THREE.Camera | null = null;
  private targetScene: THREE.Scene | null = null;
  private generation = 0;
  private activationToken = 0;
  private queue: WarmupTask[] = [];
  private objectStates = new WeakMap<THREE.Object3D, TrackedObjectState>();
  private sourceStates = new Map<THREE.Material, SourceWarmupState>();
  private warmCompletions = new Set<WarmCompletion>();
  private drainPromise: Promise<void> | null = null;

  constructor(
    private readonly waitForWarmupTurn: WaitForWarmupTurn = defaultWaitForWarmupTurn,
    private readonly compileOne: CompileOne = defaultCompileOne,
    private readonly deferActivation: DeferActivation = defaultDeferActivation
  ) {}

  configure(config: WarmupConfig): void {
    const nextEnabled =
      config.enabled &&
      config.renderer !== null &&
      config.camera !== null &&
      config.targetScene !== null;

    this.enabled = nextEnabled;
    this.renderer = config.renderer;
    this.camera = config.camera;
    this.targetScene = config.targetScene;
    this.clear();
  }

  warmScene(root: THREE.Object3D): Promise<void> {
    if (!this.enabled) return Promise.resolve();

    return new Promise((resolve) => {
      const completion = { generation: this.generation, resolve };
      this.warmCompletions.add(completion);
      const activationToken = ++this.activationToken;

      this.deferActivation(() => {
        if (
          activationToken !== this.activationToken ||
          completion.generation !== this.generation ||
          !this.enabled
        ) {
          this.settleWarmCompletion(completion);
          return;
        }

        this.armed = true;
        try {
          root.traverse((object) => {
            this.scheduleObject(object);
          });
        } catch (error) {
          log.warning(
            Modules.RENDERER,
            `Blend warm-up traversal skipped after an error: ${String(error)}`
          );
        }
        this.resolveWarmCompletionsIfIdle(completion.generation);
      });
    });
  }

  scheduleObject(object: THREE.Object3D): void {
    if (!this.enabled || !this.armed || !this.renderer || !this.camera || !this.targetScene) {
      return;
    }

    const nodeType = (object.userData as { nodeType?: unknown }).nodeType;
    if (!isWarmupNodeType(nodeType) || !hasRenderableContent(object, nodeType)) return;

    if (!(
      object instanceof THREE.Mesh ||
      object instanceof THREE.Line ||
      object instanceof THREE.Points
    )) {
      return;
    }

    const sourceMaterial = Array.isArray(object.material) ? null : object.material;
    if (!hasWarmupMaterialApi(sourceMaterial)) return;

    const previousObjectState = this.objectStates.get(object);
    if (previousObjectState && previousObjectState.sourceMaterial !== sourceMaterial) {
      this.detachObject(object, previousObjectState);
    }

    const fingerprint = buildBlendWarmupFingerprint(sourceMaterial, object, false);
    let existing = this.sourceStates.get(sourceMaterial);
    if (existing && existing.fingerprint !== fingerprint) {
      this.clearSource(sourceMaterial);
      existing = undefined;
    }

    if (!existing) {
      const keepers = new Set<THREE.Material>();
      const disposeHandler = () => {
        this.clearSource(sourceMaterial);
      };
      sourceMaterial.addEventListener('dispose', disposeHandler);
      existing = { fingerprint, keepers, objects: new Set(), disposeHandler };
      this.sourceStates.set(sourceMaterial, existing);

      const seenVariants = new Set<string>();

      for (const mode of BLENDING_MODES) {
        const keeper: THREE.Material = sourceMaterial.clone();
        if (!hasWarmupMaterialApi(keeper)) {
          keeper.dispose();
          continue;
        }

        keeper.applyBlendingMode(mode);
        const variantFingerprint = buildBlendWarmupFingerprint(keeper, object, true);
        if (seenVariants.has(variantFingerprint)) {
          keeper.dispose();
          continue;
        }

        seenVariants.add(variantFingerprint);
        keepers.add(keeper);
        this.queue.push({
          sourceMaterial,
          keeperMaterial: keeper,
          compileObject: createCompileObject(object, keeper),
        });
      }

      if (keepers.size === 0) {
        this.clearSource(sourceMaterial);
        return;
      }

      this.startDrain();
    }

    this.attachObject(object, sourceMaterial, existing);
  }

  clear(): void {
    this.armed = false;
    this.activationToken++;
    this.generation++;
    this.queue = [];
    for (const completion of Array.from(this.warmCompletions)) {
      this.settleWarmCompletion(completion);
    }
    for (const material of Array.from(this.sourceStates.keys())) {
      this.clearSource(material);
    }
  }

  private settleWarmCompletion(completion: WarmCompletion): void {
    if (!this.warmCompletions.delete(completion)) return;
    completion.resolve();
  }

  private resolveWarmCompletionsIfIdle(generation: number): void {
    if (this.drainPromise || this.queue.length > 0) return;
    for (const completion of Array.from(this.warmCompletions)) {
      if (completion.generation === generation) {
        this.settleWarmCompletion(completion);
      }
    }
  }

  private clearSource(sourceMaterial: THREE.Material): void {
    const state = this.sourceStates.get(sourceMaterial);
    if (!state) return;

    sourceMaterial.removeEventListener('dispose', state.disposeHandler);
    this.sourceStates.delete(sourceMaterial);
    this.queue = this.queue.filter((task) => task.sourceMaterial !== sourceMaterial);

    for (const object of state.objects) {
      const objectState = this.objectStates.get(object);
      if (objectState?.sourceMaterial === sourceMaterial) {
        object.removeEventListener('removed', objectState.removeHandler);
        this.objectStates.delete(object);
      }
    }
    state.objects.clear();

    for (const keeper of state.keepers) {
      keeper.dispose();
    }
    state.keepers.clear();
  }

  private attachObject(
    object: THREE.Object3D,
    sourceMaterial: THREE.Material,
    state: SourceWarmupState
  ): void {
    const existing = this.objectStates.get(object);
    if (existing?.sourceMaterial === sourceMaterial) return;

    const removeHandler = () => {
      this.releaseObject(object);
    };
    object.addEventListener('removed', removeHandler);
    this.objectStates.set(object, { sourceMaterial, removeHandler });
    state.objects.add(object);
  }

  private detachObject(object: THREE.Object3D, objectState: TrackedObjectState): void {
    object.removeEventListener('removed', objectState.removeHandler);
    this.objectStates.delete(object);

    const sourceState = this.sourceStates.get(objectState.sourceMaterial);
    if (!sourceState) return;

    sourceState.objects.delete(object);
    if (sourceState.objects.size === 0) {
      this.clearSource(objectState.sourceMaterial);
    }
  }

  private releaseObject(object: THREE.Object3D): void {
    const objectState = this.objectStates.get(object);
    if (!objectState) return;
    this.detachObject(object, objectState);
  }

  private startDrain(): void {
    if (this.drainPromise || !this.enabled) return;
    const generation = this.generation;
    this.drainPromise = this.drain(generation).finally(() => {
      this.drainPromise = null;
      if (this.enabled && this.armed && this.queue.length > 0) {
        this.startDrain();
      } else {
        this.resolveWarmCompletionsIfIdle(generation);
      }
    });
  }

  private async drain(generation: number): Promise<void> {
    while (generation === this.generation) {
      const task = this.queue.shift();
      if (!task) return;
      if (!this.renderer || !this.camera || !this.targetScene) return;

      const state = this.sourceStates.get(task.sourceMaterial);
      if (!state || !state.keepers.has(task.keeperMaterial)) {
        continue;
      }

      await this.waitForWarmupTurn();

      if (generation !== this.generation) return;
      if (!this.renderer || !this.camera || !this.targetScene) return;

      const currentState = this.sourceStates.get(task.sourceMaterial);
      if (!currentState || !currentState.keepers.has(task.keeperMaterial)) {
        continue;
      }

      try {
        this.compileOne(this.renderer, this.camera, this.targetScene, task.compileObject);
      } catch (error) {
        log.warning(
          Modules.RENDERER,
          `Blend warm-up compile skipped after a WebGL compile failure: ${String(error)}`
        );
      }
    }
  }
}

let defaultManager = new WebGLBlendWarmupManager();

export function configureBlendModeProgramWarmup(config: WarmupConfig): void {
  defaultManager.configure(config);
}

export function scheduleBlendModeProgramWarmupForObject(object: THREE.Object3D): void {
  defaultManager.scheduleObject(object);
}

export function warmSceneBlendModePrograms(root: THREE.Object3D): Promise<void> {
  return defaultManager.warmScene(root);
}

export function clearBlendModeProgramWarmup(): void {
  defaultManager.clear();
}

export function __resetBlendModeWarmupForTests(): void {
  defaultManager.clear();
  defaultManager = new WebGLBlendWarmupManager();
}
