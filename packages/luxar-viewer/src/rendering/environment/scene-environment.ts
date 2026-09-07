/**
 * The scene environment — the ONE lighting input Luxar's otherwise light-free scene
 * has, built lazily for physical mesh materials.
 *
 * Luxar's four geometry types are emissive by design and the house mesh shader lights
 * itself from a fixed view-space key, so the scene has never held a light or an
 * environment map. A physically based material renders BLACK without one. Rather than
 * add light objects to the graph (spec `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3 / §5),
 * the viewer sets `scene.environment` from one of three sources, in this precedence:
 *
 * 1. **A baked map** attached to the store (`luxar env bake` / `env attach`) whose
 *    `scene_content_hash` matches the scene — the raw six-face capture, prefiltered by
 *    three at load in milliseconds. Zero live cost for a published scene.
 * 2. **The authored source** (`viewer_config.environment.source`): `scene` — an EXACT
 *    cube capture of the scene itself from the probe (`./cube-capture.ts`), so metals
 *    and glass reflect the data they sit in; re-captured on data commit, slice change
 *    and appearance change once the loader settles, never per frame (a fixed-probe
 *    cube map is view-independent, so camera motion is NOT a trigger); or `hdri` — an
 *    equirectangular image (`./hdri.ts`), with the room standing in until it arrives.
 * 3. **The room** — three's procedural `RoomEnvironment` prefiltered once: no asset, a
 *    neutral key, believable reflections, and what three's own examples light with.
 *
 * Two properties are load-bearing:
 *
 * - **Lazy.** Nothing is built until {@link SceneEnvironment.ensure} is called, and
 *   the only caller is the material manager's physical-material hook. A scene with no
 *   physical mesh keeps `scene.environment === null`, so it renders byte-identically
 *   to before this module existed — whatever its config says.
 * - **Invisible to house materials.** `scene.environment` is read only by three's
 *   lighting-model materials. Every Luxar material is a `ShaderMaterial` /
 *   `NodeMaterial` with its own fragment code that never samples an environment, so
 *   setting it changes nothing about points, lines, splats or house meshes — a
 *   property the unit tests assert rather than assume.
 *
 * The backend-specific pieces (PMREM generator, cube render target) are injected as
 * factories; the WebGPU ones are reached through `rendering/tsl/registry.ts` to keep the
 * lazy chunk lazy (see {@link createSceneEnvironment}).
 *
 * @module rendering/environment/scene-environment
 */

import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { MaterialBackend } from '../material-manager/factories';
import { requireTslMaterials } from '../tsl/slot';
import { log, Modules } from '../../utils/log';
import {
  DEFAULT_ENVIRONMENT_CONFIG,
  type BakedEnvironment,
  type EnvironmentConfig,
  type EnvironmentProbe,
} from '../../types/environment';
import { buildBakedCubeTexture } from './baked';
import { captureClipPlanes, captureSceneCube, type CubeTargetLike } from './cube-capture';
import { loadEquirectangularTexture, resolveEnvironmentUrl } from './hdri';
import { committedBoundingSphere, formatProbeSpec, resolveProbe } from './probe';

/**
 * The slice of a PMREM generator this module uses — the same on both backends, and
 * narrow enough that a unit test can hand in a stub without a GPU.
 */
export interface PmremGeneratorLike {
  fromScene(scene: THREE.Scene, sigma?: number): { texture: THREE.Texture; dispose(): void };
  dispose(): void;
}

/**
 * Blur applied when prefiltering the room. Three's `RoomEnvironment` examples use
 * `0.04`; it softens the room's hard box edges into the gentle gradients a reflection
 * on a curved surface should show, while leaving the area lights sharp enough to read
 * as highlights.
 */
export const ROOM_ENVIRONMENT_SIGMA = 0.04;

/**
 * How long after the last stale mark a live `scene` capture waits before running: a
 * burst of commits (a partition landing part by part) is one capture, not many.
 */
export const CAPTURE_DEBOUNCE_MS = 250;

/** What the environment is currently lit by. */
export type EnvironmentKind = 'none' | 'room' | 'scene' | 'baked' | 'hdri';

/** The backend-specific factories and scene the environment works on. */
export interface SceneEnvironmentDeps {
  scene: THREE.Scene;
  /** Either backend's renderer, handed to `CubeCamera.update`. */
  renderer: unknown;
  createGenerator: () => PmremGeneratorLike;
  /** A half-float cube render target of the given face size. */
  createCubeTarget: (resolution: number) => CubeTargetLike;
}

/**
 * What a live capture needs from the running app — attached by the init pipeline once
 * those pieces exist. Without it, `scene` falls back to the room.
 */
export interface CaptureRuntime {
  /** The scene root (`LuxarScene`), for probe resolution and physical-mesh hiding. */
  sceneRoot: () => THREE.Object3D | null;
  /** Push the cube camera's params to the material manager (fov 90°, square buffer). */
  pushCaptureCameraParams: (resolution: number) => void;
  /** Restore the main camera's params (the ordinary camera-materials push). */
  restoreCameraParams: () => void;
  /** Whether the loader has settled (no update sweep, load pass or LOD load in flight). */
  isSettled: () => boolean;
  /** Base URL of the store, for a store-relative `hdri` url. */
  baseUrl: () => string | undefined;
}

/** The result of one live capture, also what the bake reads back. */
export interface SceneCapture {
  target: CubeTargetLike;
  resolution: number;
  probe: EnvironmentProbe;
  probePosition: THREE.Vector3;
  hiddenPhysicalMeshes: number;
}

export class SceneEnvironment {
  private roomTarget: { texture: THREE.Texture; dispose(): void } | null = null;
  private captureTarget: CubeTargetLike | null = null;
  private bakedTexture: THREE.CubeTexture | null = null;
  private hdriTexture: THREE.Texture | null = null;
  private hdriLoading: string | null = null;

  private config: EnvironmentConfig = DEFAULT_ENVIRONMENT_CONFIG;
  private baked: BakedEnvironment | null = null;
  private runtime: CaptureRuntime | null = null;
  private captureProbeSpec: string | null = null;
  /** Set by the first `ensure()` — a physical material exists, so the scene needs light. */
  private wanted = false;
  private active: EnvironmentKind = 'none';
  private staleSince: number | null = null;
  /** How many live captures have run (a diagnostic the tests and the bake read). */
  captureCount = 0;

  constructor(private readonly deps: SceneEnvironmentDeps) {}

  /** Whether an environment has been built and assigned. */
  isReady(): boolean {
    return this.active !== 'none';
  }

  /** What the scene is currently lit by. */
  activeKind(): EnvironmentKind {
    return this.active;
  }

  /** The authored config in force (the default when the scene authored none). */
  getConfig(): EnvironmentConfig {
    return this.config;
  }

  /** Whether a valid baked map is attached (it wins over `config.source`). */
  hasBaked(): boolean {
    return this.baked !== null;
  }

  /**
   * Set the authored config (`viewer_config.environment`, validated by the zarr bridge).
   * `null` restores the default. Re-applies immediately if a physical material already
   * asked for light; otherwise it just waits for `ensure()`.
   */
  configure(config: EnvironmentConfig | null): void {
    this.config = config ?? DEFAULT_ENVIRONMENT_CONFIG;
    this.deps.scene.environmentIntensity = this.config.intensity;
    if (this.wanted) this.apply();
  }

  /** Clear dataset-owned state while preserving runtime wiring and the reusable room PMREM. */
  resetForDataset(): void {
    if (this.roomTarget && this.deps.scene.environment === this.roomTarget.texture) {
      this.deps.scene.environment = null;
    }
    this.wanted = false;
    this.baked = null;
    this.staleSince = null;
    this.hdriLoading = null;
    this.releaseCaptureTarget();
    if (this.bakedTexture) {
      if (this.deps.scene.environment === this.bakedTexture) this.deps.scene.environment = null;
      this.bakedTexture.dispose();
      this.bakedTexture = null;
    }
    if (this.hdriTexture) {
      if (this.deps.scene.environment === this.hdriTexture) this.deps.scene.environment = null;
      this.hdriTexture.dispose();
      this.hdriTexture = null;
    }
    this.active = 'none';
  }

  /** Attach (or clear) a baked map. A valid map takes precedence over any source. */
  setBaked(map: BakedEnvironment | null): void {
    this.baked = map;
    if (this.bakedTexture) {
      if (this.deps.scene.environment === this.bakedTexture) this.deps.scene.environment = null;
      this.bakedTexture.dispose();
      this.bakedTexture = null;
    }
    if (this.wanted) this.apply();
  }

  /** Give the environment what a live capture needs (the init pipeline does this once). */
  attachRuntime(runtime: CaptureRuntime | null): void {
    this.runtime = runtime;
    if (this.wanted && this.active === 'room' && this.config.source === 'scene') this.apply();
  }

  /**
   * A physical material now exists: make sure the scene is lit.
   *
   * Idempotent — every call after the first that changes nothing returns `false`.
   * Synchronous for the room and a live capture, so a physical mesh created this
   * frame is lit on its first draw.
   *
   * @returns `true` when this call built or replaced the environment.
   */
  ensure(): boolean {
    this.wanted = true;
    return this.apply();
  }

  /**
   * Something the capture depends on changed (a commit, a slice, an appearance edit).
   * Only meaningful while lit by a live `scene` capture; the next {@link tick} after
   * the debounce, with the loader settled, re-captures.
   */
  markStale(): void {
    if (this.active !== 'scene') return;
    this.staleSince = performance.now();
  }

  /**
   * Per-frame hook (the init pipeline registers it). Re-captures a stale live
   * environment once the debounce has passed and the loader is settled. Returns
   * whether a capture ran.
   */
  tick(now: number = performance.now()): boolean {
    if (this.staleSince === null || this.active !== 'scene' || !this.runtime) return false;
    if (now - this.staleSince < CAPTURE_DEBOUNCE_MS) return false;
    if (!this.runtime.isSettled()) return false;
    this.staleSince = null;
    return this.captureScene() !== null;
  }

  /**
   * Run one exact capture of the scene from the configured (or given) probe and make
   * it the environment. Returns `null` without a runtime. Also the bake's entry point:
   * the returned target is what `bake.ts` reads the six faces back from.
   */
  captureScene(override?: { probe?: EnvironmentProbe; resolution?: number }): SceneCapture | null {
    const runtime = this.runtime;
    if (!runtime) return null;
    const resolution = override?.resolution ?? this.config.resolution;
    const probe = override?.probe ?? this.config.probe;
    const root = runtime.sceneRoot();
    const probePosition = resolveProbe(probe, root, new THREE.Vector3());
    const bounds = committedBoundingSphere(root);
    const { near, far } = captureClipPlanes(probePosition, bounds);

    if (this.captureTarget && this.captureTarget.width !== resolution) {
      this.captureTarget.dispose();
      this.captureTarget = null;
    }
    this.captureTarget ??= this.deps.createCubeTarget(resolution);

    const start = performance.now();
    const target = this.captureTarget;
    const hidden = withCaptureCameraParams(runtime, resolution, () =>
      captureSceneCube({
        renderer: this.deps.renderer,
        scene: this.deps.scene,
        target,
        probe: probePosition,
        near,
        far,
        root,
      })
    );
    this.deps.scene.environment = this.captureTarget.texture;
    this.active = 'scene';
    this.captureProbeSpec = formatProbeSpec(probe);
    this.captureCount += 1;
    log.info(
      Modules.RENDERER,
      `Scene environment captured (${resolution}px cube at ${formatProbeSpec(probe)} → ` +
        `[${probePosition
          .toArray()
          .map((v) => v.toFixed(3))
          .join(', ')}], ` +
        `${hidden} physical mesh(es) hidden) in ${(performance.now() - start).toFixed(1)} ms`
    );
    return {
      target: this.captureTarget,
      resolution,
      probe,
      probePosition,
      hiddenPhysicalMeshes: hidden,
    };
  }

  /** Rebuild previously-created GPU resources after the renderer context is restored. */
  rebuild(): boolean {
    if (!this.isReady()) return false;
    const scene = this.deps.scene;
    const ours = [
      this.roomTarget?.texture,
      this.captureTarget?.texture,
      this.bakedTexture,
      this.hdriTexture,
    ];
    if (scene.environment && ours.includes(scene.environment)) scene.environment = null;
    this.roomTarget?.dispose();
    this.captureTarget?.dispose();
    this.bakedTexture?.dispose();
    this.hdriTexture?.dispose();
    this.roomTarget = null;
    this.captureTarget = null;
    this.captureProbeSpec = null;
    this.bakedTexture = null;
    this.hdriTexture = null;
    this.active = 'none';
    this.staleSince = null;
    const rebuilt = this.apply();
    this.markStale();
    return rebuilt;
  }

  /** Release everything and clear `scene.environment` if it is ours. */
  dispose(): void {
    const scene = this.deps.scene;
    const ours = [
      this.roomTarget?.texture,
      this.captureTarget?.texture,
      this.bakedTexture,
      this.hdriTexture,
    ];
    if (scene.environment && ours.includes(scene.environment)) scene.environment = null;
    this.roomTarget?.dispose();
    this.captureTarget?.dispose();
    this.bakedTexture?.dispose();
    this.hdriTexture?.dispose();
    this.roomTarget = null;
    this.captureTarget = null;
    this.captureProbeSpec = null;
    this.bakedTexture = null;
    this.hdriTexture = null;
    this.runtime = null;
    this.active = 'none';
    this.wanted = false;
    this.staleSince = null;
  }

  // ---------------------------------------------------------------------------

  /** Apply the precedence rule for the current state. Returns whether anything changed. */
  private apply(): boolean {
    if (this.baked) return this.applyBaked(this.baked);
    switch (this.config.source) {
      case 'scene':
        if (this.runtime) {
          if (this.captureMatchesConfig()) return false;
          // First light is an immediate capture — of whatever is resident now — and
          // every later commit re-captures through `markStale` / `tick`.
          return this.captureScene() !== null;
        }
        return this.applyRoom();
      case 'hdri':
        this.startHdri();
        // The room stands in until the image arrives (or fails).
        return this.hdriTexture ? this.applyHdri() : this.applyRoom();
      default:
        return this.applyRoom();
    }
  }

  private captureMatchesConfig(): boolean {
    return (
      this.active === 'scene' &&
      this.captureTarget?.width === this.config.resolution &&
      this.captureProbeSpec === formatProbeSpec(this.config.probe) &&
      this.deps.scene.environment === this.captureTarget.texture
    );
  }

  private applyRoom(): boolean {
    if (!this.roomTarget) {
      const start = performance.now();
      const generator = this.deps.createGenerator();
      const room = new RoomEnvironment();
      try {
        this.roomTarget = generator.fromScene(room, ROOM_ENVIRONMENT_SIGMA);
      } finally {
        // Both are scaffolding: the prefiltered target is the only thing kept.
        room.dispose();
        generator.dispose();
      }
      log.info(
        Modules.RENDERER,
        `Scene environment built (RoomEnvironment PMREM) in ${(performance.now() - start).toFixed(1)} ms`
      );
    }
    if (this.active === 'room' && this.deps.scene.environment === this.roomTarget.texture) {
      return false;
    }
    this.releaseCaptureTarget();
    this.deps.scene.environment = this.roomTarget.texture;
    this.active = 'room';
    this.staleSince = null;
    return true;
  }

  private applyBaked(map: BakedEnvironment): boolean {
    if (this.active === 'baked' && this.bakedTexture) return false;
    this.releaseCaptureTarget();
    this.bakedTexture ??= buildBakedCubeTexture(map);
    this.deps.scene.environment = this.bakedTexture;
    this.active = 'baked';
    this.staleSince = null;
    log.info(
      Modules.RENDERER,
      `Scene environment: using the baked ${map.resolution}px map (probe ${map.header.probe.spec})`
    );
    return true;
  }

  private startHdri(): void {
    const url = this.config.url;
    if (!url || this.hdriTexture || this.hdriLoading === url) return;
    this.hdriLoading = url;
    const resolved = resolveEnvironmentUrl(url, this.runtime?.baseUrl());
    void loadEquirectangularTexture(resolved).then(
      (texture) => {
        if (this.hdriLoading !== url) {
          texture.dispose();
          return;
        }
        this.hdriLoading = null;
        this.hdriTexture = texture;
        if (this.wanted && !this.baked && this.config.source === 'hdri') this.applyHdri();
      },
      (error: unknown) => {
        this.hdriLoading = null;
        log.warning(
          Modules.RENDERER,
          `Scene environment: could not load HDRI '${resolved}' (${String(error)}); keeping the room`
        );
      }
    );
  }

  private applyHdri(): boolean {
    if (!this.hdriTexture) return false;
    if (this.active === 'hdri') return false;
    this.releaseCaptureTarget();
    this.deps.scene.environment = this.hdriTexture;
    this.active = 'hdri';
    this.staleSince = null;
    log.info(Modules.RENDERER, `Scene environment: HDRI '${this.config.url}' applied`);
    return true;
  }

  private releaseCaptureTarget(): void {
    if (!this.captureTarget) return;
    if (this.deps.scene.environment === this.captureTarget.texture) {
      this.deps.scene.environment = null;
    }
    this.captureTarget.dispose();
    this.captureTarget = null;
    this.captureProbeSpec = null;
  }
}

/** Run `draw` with the cube camera's params pushed to the materials, restoring the main camera's after. */
function withCaptureCameraParams<T>(runtime: CaptureRuntime, resolution: number, draw: () => T): T {
  runtime.pushCaptureCameraParams(resolution);
  try {
    return draw();
  } finally {
    runtime.restoreCameraParams();
  }
}

/**
 * Build the environment for the active renderer.
 *
 * Dispatches on the material backend rather than on the renderer class: the WebGPU
 * renderer's PMREM generator and cube render target are different classes from
 * `three`'s and live in the lazy chunk, so they are fetched through the TSL registry —
 * which is loaded before any WebGPU material can exist, hence before any of this can
 * ever run on that backend.
 */
export function createSceneEnvironment(
  renderer: THREE.WebGLRenderer | object,
  backend: MaterialBackend,
  scene: THREE.Scene
): SceneEnvironment {
  const targetOptions = {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    colorSpace: THREE.LinearSRGBColorSpace,
    generateMipmaps: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: true,
  };
  const createGenerator = (): PmremGeneratorLike =>
    backend === 'tsl'
      ? (new (requireTslMaterials().environment.PMREMGenerator)(
          renderer as never
        ) as unknown as PmremGeneratorLike)
      : (new THREE.PMREMGenerator(renderer as THREE.WebGLRenderer) as PmremGeneratorLike);
  const createCubeTarget = (resolution: number): CubeTargetLike =>
    backend === 'tsl'
      ? (new (requireTslMaterials().environment.CubeRenderTarget)(
          resolution,
          targetOptions as never
        ) as unknown as CubeTargetLike)
      : (new THREE.WebGLCubeRenderTarget(resolution, targetOptions) as unknown as CubeTargetLike);
  return new SceneEnvironment({ scene, renderer, createGenerator, createCubeTarget });
}
