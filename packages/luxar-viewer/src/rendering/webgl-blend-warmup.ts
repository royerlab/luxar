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

/**
 * What the session warms against. A null renderer/camera/scene, a WebGPU
 * surface, or `?no-blend-warmup` all resolve to a disabled manager.
 */
export interface WarmupConfig {
  enabled: boolean;
  renderer: THREE.WebGLRenderer | null;
  camera: THREE.Camera | null;
  targetScene: THREE.Scene | null;
}

/** One queued link: which keeper to compile, and the throwaway object to compile it on. */
export interface WarmupTask {
  sourceMaterial: THREE.Material;
  keeperMaterial: THREE.Material;
  compileObject: THREE.Object3D;
}

/** Per-object bookkeeping: the material it was warmed for, and its `removed` listener. */
export interface TrackedObjectState {
  sourceMaterial: THREE.Material;
  removeHandler: () => void;
}

/**
 * Per-source-material state: the compile-time fingerprint the keepers were built
 * against, the keepers pinning the programs, the objects sharing the material,
 * and the material's `dispose` listener.
 */
export interface SourceWarmupState {
  fingerprint: string;
  keepers: Set<THREE.Material>;
  objects: Set<THREE.Object3D>;
  disposeHandler: () => void;
  /**
   * Program variants this source USES (variant fingerprint → blend mode),
   * whether it owns the keeper for them or shares one compiled for an
   * identical material on another node. Released in `clearSource`.
   */
  variants: Map<string, BlendingMode>;
}

/** Counters for `getPerf().blendWarmup` and the unit tests. */
export interface BlendWarmupStats {
  /** Keeper compiles queued (one per program variant that had no live keeper). */
  queued: number;
  /** Keeper compiles that ran. */
  compiled: number;
  /** Blend modes of one source that collapsed onto an already-queued variant of the same source. */
  dedupedWithinSource: number;
  /**
   * Variants a source shares with a keeper compiled for an IDENTICAL material on
   * another node (same program) instead of queueing its own — the 100-node
   * benchmark scene queued 400 compiles for ~4 distinct programs before this.
   */
  dedupedAcrossSources: number;
  /** Shared variants re-queued under a surviving node after their owner was released. */
  ownershipTransfers: number;
}

/**
 * One outstanding `warmScene` caller. Settled by the drain going idle, by
 * {@link WebGLBlendWarmupManager.clear}, or by the readiness budget expiring —
 * whichever happens first.
 */
export interface WarmCompletion {
  generation: number;
  resolve: () => void;
  budgetTimer: ReturnType<typeof setTimeout> | null;
}

/** Yields until the next compile may run (a rendered frame, then an idle slot). */
export type WaitForWarmupTurn = () => Promise<void>;
/** Defers scene activation past the caller's own turn. */
export type DeferActivation = (activate: () => void) => void;
type IdleCallbackScheduler = (callback: () => void, options?: { timeout: number }) => number;
/** Links one keeper material by compiling `compileObject` against the live scene. */
export type CompileOne = (
  renderer: THREE.WebGLRenderer,
  camera: THREE.Camera,
  targetScene: THREE.Scene,
  compileObject: THREE.Object3D
) => void;

const WARMUP_IDLE_TIMEOUT_MS = 250;
const WARMUP_FALLBACK_DELAY_MS = 50;

/**
 * How long dataset readiness may wait on a warm-up before giving up on it.
 *
 * The queue is NOT a finite set snapshotted at activation: every geometry commit
 * schedules the node it just populated, so a progressively-loading partition
 * keeps handing the drain new work for as long as it streams. Awaiting "the
 * queue is empty" would therefore hold `loadDataset` — and with it
 * `window.__luxarDebug`, the `dataset-loaded` event and `switchDataset`'s
 * promise — until the whole scene settled. The same wait parks indefinitely in a
 * background tab, where `requestAnimationFrame` is never serviced and no turn
 * ever comes.
 *
 * So the wait is a budget, not a barrier: an ordinary scene drains well inside
 * it and is fully pinned before readiness, while anything pathological releases
 * readiness here and keeps warming in the background.
 */
const WARMUP_READINESS_BUDGET_MS = 5000;

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
 *
 * The shader SOURCE enters only as its length, not its text. Both comparisons
 * this fingerprint feeds stay inside one material lineage — a source material
 * against its own earlier state, and clones of that source against each other —
 * and neither can carry different shader text, which every wrapper fixes in its
 * constructor. Embedding it would make each call concatenate ~25 kB, and
 * `scheduleObject` runs on the geometry-commit path and once per affected leaf
 * per Layers-panel slider event, so that allocation lands squarely on the
 * interaction path this module exists to keep clear.
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
    `vs:${(shaderMaterial.vertexShader ?? '').length}`,
    `fs:${(shaderMaterial.fragmentShader ?? '').length}`,
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

/**
 * The warm-up service itself — one instance per session (the module keeps the
 * default one; the constructor's three injection points exist for tests).
 *
 * Lifecycle: {@link configure} once per renderer, {@link warmScene} once the
 * dataset is fully set up, {@link scheduleObject} for every node that gains or
 * changes compile-time state afterwards, {@link clear} on scene replacement,
 * context loss and disposal.
 */
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
  // Cross-node dedupe. A program variant is keyed by its FULL fingerprint
  // (material lineage + blend defines + geometry layout); every source material
  // that uses it is a "user", and exactly one user — the "owner" — holds the
  // keeper clone whose compile pins the program. Identical materials on many
  // nodes (the common case: one node factory config per geometry type) share
  // one keeper per variant instead of each queueing its own.
  private variantUsers = new Map<string, Set<THREE.Material>>();
  private variantOwner = new Map<string, THREE.Material>();
  private stats: BlendWarmupStats = {
    queued: 0,
    compiled: 0,
    dedupedWithinSource: 0,
    dedupedAcrossSources: 0,
    ownershipTransfers: 0,
  };

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

  /**
   * Arm the manager and warm every reachable variant under `root`.
   *
   * Resolves when the queue drains, or when the readiness budget
   * (`WARMUP_READINESS_BUDGET_MS`) elapses — whichever comes first. The budget
   * expiring is not a failure: the drain carries on afterwards, it just stops
   * holding the caller.
   */
  warmScene(root: THREE.Object3D): Promise<void> {
    if (!this.enabled) return Promise.resolve();

    return new Promise((resolve) => {
      const completion: WarmCompletion = {
        generation: this.generation,
        resolve,
        budgetTimer: null,
      };
      completion.budgetTimer = setTimeout(() => {
        completion.budgetTimer = null;
        if (!this.warmCompletions.has(completion)) return;
        log.info(
          Modules.RENDERER,
          `Blend warm-up still draining after ${WARMUP_READINESS_BUDGET_MS} ms ` +
            `(${this.queue.length} variants queued) — releasing readiness, warming continues`
        );
        this.settleWarmCompletion(completion);
      }, WARMUP_READINESS_BUDGET_MS);
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
      existing = {
        fingerprint,
        keepers,
        objects: new Set(),
        disposeHandler,
        variants: new Map(),
      };
      this.sourceStates.set(sourceMaterial, existing);
      this.enqueueVariantKeepers(sourceMaterial, object, existing);

      if (existing.variants.size === 0) {
        // No blend mode produced a warmable variant (material without the
        // warm-up API after cloning): nothing to pin, nothing to track.
        this.clearSource(sourceMaterial);
        return;
      }

      // A source whose every variant is already pinned by another node's
      // keeper queues nothing — but it still registers as a user (above) so
      // the keeper is handed over if that node goes away first.
      if (keepers.size > 0) this.startDrain();
    }

    this.attachObject(object, sourceMaterial, existing);
  }

  /**
   * Enumerate the blend-mode variants of `sourceMaterial` and queue a keeper
   * compile for each variant no live keeper pins yet. Two dedupe levels: modes
   * of THIS source that land on the same program (e.g. additive/luminous
   * differing only in blend state) collapse to one keeper, and a variant an
   * identical material on ANOTHER node already owns is shared, not re-queued.
   */
  private enqueueVariantKeepers(
    sourceMaterial: THREE.Material,
    object: THREE.Object3D,
    state: SourceWarmupState
  ): void {
    for (const mode of BLENDING_MODES) {
      const keeper: THREE.Material = sourceMaterial.clone();
      if (!hasWarmupMaterialApi(keeper)) {
        keeper.dispose();
        continue;
      }

      keeper.applyBlendingMode(mode);
      const variantFingerprint = buildBlendWarmupFingerprint(keeper, object, true);
      if (state.variants.has(variantFingerprint)) {
        keeper.dispose();
        this.stats.dedupedWithinSource += 1;
        continue;
      }
      state.variants.set(variantFingerprint, mode);
      this.registerVariantUser(variantFingerprint, sourceMaterial);

      const owner = this.variantOwner.get(variantFingerprint);
      if (owner && owner !== sourceMaterial && this.sourceStates.has(owner)) {
        keeper.dispose();
        this.stats.dedupedAcrossSources += 1;
        continue;
      }

      this.variantOwner.set(variantFingerprint, sourceMaterial);
      this.queueKeeper(sourceMaterial, object, keeper, state);
    }
  }

  private registerVariantUser(variantFingerprint: string, sourceMaterial: THREE.Material): void {
    let users = this.variantUsers.get(variantFingerprint);
    if (!users) {
      users = new Set();
      this.variantUsers.set(variantFingerprint, users);
    }
    users.add(sourceMaterial);
  }

  private queueKeeper(
    sourceMaterial: THREE.Material,
    object: THREE.Object3D,
    keeper: THREE.Material,
    state: SourceWarmupState
  ): void {
    state.keepers.add(keeper);
    this.queue.push({
      sourceMaterial,
      keeperMaterial: keeper,
      compileObject: createCompileObject(object, keeper),
    });
    this.stats.queued += 1;
  }

  /**
   * The owner of a shared variant went away while other nodes still use it:
   * hand the keeper to one of them (a fresh clone from that node's material,
   * queued for compile) so the program stays pinned for the survivors.
   */
  private adoptVariant(
    variantFingerprint: string,
    mode: BlendingMode,
    newOwner: THREE.Material
  ): void {
    const state = this.sourceStates.get(newOwner);
    const object = state ? state.objects.values().next().value : undefined;
    if (!state || !object) return;
    const keeper: THREE.Material = newOwner.clone();
    if (!hasWarmupMaterialApi(keeper)) {
      keeper.dispose();
      return;
    }
    keeper.applyBlendingMode(mode);
    this.variantOwner.set(variantFingerprint, newOwner);
    this.queueKeeper(newOwner, object, keeper, state);
    this.stats.ownershipTransfers += 1;
    this.startDrain();
  }

  /** Release `sourceMaterial`'s variant registrations, transferring owned ones. */
  private releaseVariants(sourceMaterial: THREE.Material, state: SourceWarmupState): void {
    for (const [variantFingerprint, mode] of state.variants) {
      const users = this.variantUsers.get(variantFingerprint);
      users?.delete(sourceMaterial);
      if (this.variantOwner.get(variantFingerprint) !== sourceMaterial) continue;
      this.variantOwner.delete(variantFingerprint);
      const heir = users?.values().next().value;
      if (heir) {
        this.adoptVariant(variantFingerprint, mode, heir);
      } else {
        this.variantUsers.delete(variantFingerprint);
      }
    }
    state.variants.clear();
  }

  /** Snapshot of the warm-up counters (copied; safe to hand to probes). */
  getStats(): BlendWarmupStats {
    return { ...this.stats };
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
    this.variantUsers.clear();
    this.variantOwner.clear();
  }

  private settleWarmCompletion(completion: WarmCompletion): void {
    if (!this.warmCompletions.delete(completion)) return;
    if (completion.budgetTimer !== null) {
      clearTimeout(completion.budgetTimer);
      completion.budgetTimer = null;
    }
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
    this.releaseVariants(sourceMaterial, state);
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
        this.stats.compiled += 1;
      } catch (error) {
        log.warning(
          Modules.RENDERER,
          `Blend warm-up compile skipped after a WebGL compile failure: ${String(error)}`
        );
      }
    }
  }
}

const defaultManager = new WebGLBlendWarmupManager();

/**
 * Point the session's warm-up at a renderer, camera and scene, and drop
 * whatever the previous configuration had pinned. Called once per renderer
 * setup; `enabled: false` (WebGPU, or `?no-blend-warmup`) makes every other
 * entry point below a no-op.
 */
export function configureBlendModeProgramWarmup(config: WarmupConfig): void {
  defaultManager.configure(config);
}

/**
 * Warm (or re-warm) the blend variants one node can reach.
 *
 * Called from the geometry-commit material sync and the Layers panel, i.e.
 * wherever a node gains content or changes compile-time state. Cheap and
 * idempotent while the node's fingerprint is unchanged; a no-op until
 * {@link warmSceneBlendModePrograms} has armed the session.
 */
export function scheduleBlendModeProgramWarmupForObject(object: THREE.Object3D): void {
  defaultManager.scheduleObject(object);
}

/**
 * Arm the session and warm everything already in the scene.
 *
 * Awaited at the end of a dataset load so the first Layers-panel blend switch
 * finds its program cached — bounded by the readiness budget, so a
 * still-streaming scene releases readiness and keeps warming behind it.
 */
export function warmSceneBlendModePrograms(root: THREE.Object3D): Promise<void> {
  return defaultManager.warmScene(root);
}

/**
 * Disarm the session, drop the queue and release every pinned program. Called
 * on scene replacement, WebGL context loss, and scene-manager disposal.
 */
export function clearBlendModeProgramWarmup(): void {
  defaultManager.clear();
}

/** Warm-up counters for `__luxarDebug.getPerf().blendWarmup` (see BlendWarmupStats). */
export function getBlendModeProgramWarmupStats(): BlendWarmupStats {
  return defaultManager.getStats();
}
