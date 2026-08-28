/**
 * Layer mode — render a Luxar scene inside a host application's own
 * Three.js renderer, camera, and scene graph.
 *
 * This is the headless sibling of `LuxarApp`. Where `LuxarApp` owns the
 * whole pipeline (renderer, camera, controls, post-processing, panels, input),
 * `LuxarLayer` owns **none** of it: the host keeps its renderer and camera, and
 * the layer contributes a `THREE.Group` plus the per-frame bookkeeping that
 * keeps Luxar's streaming, LOD selection, and depth sorting correct.
 *
 * The seam this rests on is already in the architecture: `SceneLoader.loadScene`
 * returns a plain `THREE.Group`, and `LODGroupRegistryDeps` / `configureDepthSort`
 * are defined purely in terms of injectable getters. Nothing in the data, cache,
 * LOD, or material path needs `SceneManager`.
 *
 * The minimal embed shape:
 *
 * ```ts
 * import { LuxarLayer } from '@royerlab/luxar-viewer';
 *
 * const layer = new LuxarLayer({
 *   renderer,                                  // host-owned
 *   getCamera: () => camera,                   // host-owned
 *   getViewportSize: () => renderer.getSize(new THREE.Vector2()),
 *   scene,                                     // host-owned
 * });
 * await layer.load('https://example.com/imaging.luxar.zarr');
 *
 * // in the host's render loop, BEFORE renderer.render():
 * layer.update();
 *
 * // on teardown:
 * await layer.dispose();
 * ```
 *
 * ## What the host is responsible for
 *
 * - Calling {@link LuxarLayer.update} once per frame, before its own render.
 * - Calling {@link LuxarLayer.resize} after a viewport or camera-projection
 *   change (the layer cannot observe the host's canvas).
 * - Draw order for its *own* geometry. Luxar stamps the configured
 *   `renderOrder` onto every Group it owns; host transparent groups should use
 *   explicit lower/higher values rather than rely on insertion order.
 * - Calling {@link LuxarLayer.handleContextLost} when WebGL context loss is
 *   reported, then {@link LuxarLayer.handleContextRestored} after rebuilding
 *   its own renderer / post-processing resources.
 *
 * ## A scene's post/camera/UI config does NOT apply in layer mode
 *
 * Everything a scene declares under `viewer_config` that is applied by
 * `ui/rendering-controls.ts` rather than by the node path is **silently inert**
 * here — `tone_mapping`, `exposure`, `global_gamma`, `global_offset`, the
 * `bloom_*` family, `background_color`, and the whole `camera` block. The layer
 * owns no post-processing, no camera, and no UI, so nothing consumes them. Only
 * per-node appearance (colormap, blending mode, opacity, absorption, the
 * intensity/offset window) travels with the geometry and takes effect.
 *
 * Tone mapping is the one worth calling out, because it is invisible from the
 * scene file and changes what the data looks like. Luxar tone-maps in the
 * mega-shader, a post-processing pass — `PostProcessingManager` even forces
 * `renderer.toneMapping = NoToneMapping` because of it — so it is not a
 * per-material setting that could be pushed onto the nodes. Measured in a host application: a scene
 * authored with ACES rendered pixel-identical to one authored with `None`.
 * (Measured, not asserted — no test in this repo covers it.)
 *
 * The consequence: **emissive geometry in a host with no tone mapping clips
 * flat**. `additive` blending sums contributions into a framebuffer that clamps
 * at 1.0, and normalising amplitudes fixes the per-splat scale, not the
 * accumulated one. Without a filmic rolloff, overlapping bright structure goes
 * to white with no gradient.
 *
 * A host that wants the rolloff has to tone-map itself
 * (`renderer.toneMapping = THREE.ACESFilmicToneMapping`, which an `OutputPass`
 * will pick up) — noting that this applies to the host's own geometry too.
 * Otherwise the only exposure controls are the scene's authored `opacity` and
 * {@link LuxarLayer.setExposure}, and `max` blending is the one mode that
 * cannot saturate at all.
 *
 * ## Limits (same as `LuxarApp`, and for the same reasons)
 *
 * One layer per page, and never alongside a `LuxarApp`. The scene-loader
 * manager, dimension manager, material manager, and worker pool are process
 * singletons; two owners would share and then corrupt each other's state.
 * The layer has no notifier UI; terminal archive failures are exposed through
 * {@link LuxarLayer.onDatasetFault} and {@link LuxarLayer.getDatasetFault} so
 * the host can surface them.
 *
 * @module core/layer/luxar-layer
 */

import * as THREE from 'three';

import {
  loadScene,
  updateSceneForDimensions,
  prefetchSceneForDimensions,
} from '../../data/zarr-loader';
import { SceneLoaderManager, getSceneLoader } from '../../data/scene-loader-manager';
import { LODGroupRegistry } from '../../scene/lod-group-registry';
import { sceneDimsManager, snapDiscreteValue } from '../../scene/scene-dims-manager';
import type { FadeableMaterial } from '../../scene/lod-fade';
import { materialManager } from '../../rendering/material-manager';
import { createKTX2TextureDecoder } from '../../rendering/ktx2-texture-decoder';
import { createRendererCapabilities, isWebGLRenderer } from '../../rendering/renderer-capabilities';
import {
  getGpuByteBudget,
  reduceGpuByteBudgetForContextLoss,
} from '../../rendering/gpu-byte-budget';
import {
  clearBlendModeProgramWarmup,
  configureBlendModeProgramWarmup,
  warmSceneBlendModePrograms,
} from '../../rendering/webgl-blend-warmup';
import {
  configureDepthSort,
  setDepthSortEnabled,
  evaluateDepthSortPerFrame,
  warmUpDepthSortWorker,
  releaseDepthSortNode,
  disposeDepthSort,
} from '../../rendering/depth-sort-coordinator';
import { disposeWorkerPool } from '../../workers/worker-pool';
import { applyModuleOverrides } from '../app/init/module-overrides';
import {
  getCameraFovRadians,
  isOrthographicCamera,
  getOrthoFrustumHeight,
  type LuxarCamera,
} from '../../utils/camera-utils';
import { config } from '../../config';
import { isLuxarMaterial } from '../../ui/layers/luxar-material';
import { clamp } from '../../utils/clamp';
import { log, Modules } from '../../utils/log';
import { markSceneResourcesDirtyForContextRestore } from '../../scene/scene-manager/render-pipeline/webgl-context-recovery';
import type { Renderer } from '../../rendering/renderer-capabilities';
import type { LoaderConfig } from '../../data/data-loader-types';
import type { EmbedderDimensions } from '../app/embedder/events';

/** Viewport size in CSS pixels. */
export interface ViewportSize {
  width: number;
  height: number;
}

export interface LuxarLayerOptions {
  /**
   * Host-owned renderer. The layer never constructs, resizes, clears, or
   * disposes it — it only reads capabilities and the drawing-buffer size.
   */
  renderer: Renderer;
  /**
   * Live camera accessor. A getter rather than a value because a host may
   * swap camera objects (a perspective/ortho toggle), and a captured
   * reference would keep projecting from the abandoned one.
   */
  getCamera: () => THREE.Camera;
  /** Live viewport accessor, in CSS pixels. Read on resize and per LOD evaluation. */
  getViewportSize: () => ViewportSize;
  /**
   * Host scene. The layer adds its root group on {@link LuxarLayer.load} and
   * removes it on {@link LuxarLayer.dispose}.
   */
  scene: THREE.Scene;
  /**
   * Called when new geometry commits outside the host's own interaction —
   * progressive refinement, lazy LOD loads, retries. Hosts with an on-demand
   * render loop must wire this or late commits will not repaint. Hosts that
   * render continuously can omit it.
   */
  requestRender?: () => void;
  /** Cache and prefetch flags forwarded to the data loader. */
  loaderConfig?: LoaderConfig;
  /** Override for bundlers that can't resolve `import.meta.url` asset URLs. */
  wasmPath?: string;
  /** Same, for the data worker. */
  workerPath?: string;
  /** Cross-fade adjacent replacement LOD levels. Default true. */
  lodFade?: boolean;
  /** Compensate incomplete stream ladders by committed energy. Default true. */
  lodEnergyComp?: boolean;
  /** Force the finest replacement LOD regardless of coverage. Default false. */
  lodFinest?: boolean;
  /** Worker-based back-to-front sorting for order-dependent geometry. Default true. */
  depthSort?: boolean;
  /**
   * `renderOrder` stamped onto every Group in the layer subtree. Three.js uses
   * the nearest Group's value as the primary transparent-sort key, so nested
   * scene / LOD / partition groups must agree. Default 10.
   */
  renderOrder?: number;
}

/**
 * The opacity a material was authored with.
 *
 * Read through the same `getOpacity()` surface the LOD fade uses rather than
 * `material.opacity`: Luxar materials carry exposure in their shader uniform,
 * while the Three.js field stays at its default. A material without that
 * surface is treated as fully exposed rather than silently dimmed.
 */
function readOpacity(mat: THREE.Material): number {
  return (mat as Partial<FadeableMaterial>).getOpacity?.() ?? 1;
}

const LOADER_ID = 'default';
const DEFAULT_RENDER_ORDER = 10;

/**
 * A Luxar scene rendered inside a host-owned Three.js pipeline.
 *
 * See the module docstring for the embed shape and the host's per-frame
 * responsibilities.
 */
export class LuxarLayer {
  private readonly options: LuxarLayerOptions;
  private readonly bufferSize = new THREE.Vector2();

  private rootGroup: THREE.Group | null = null;
  private disposed = false;
  /** Guards against overlapping `load()` calls — see {@link LuxarLayer.load}. */
  private inFlightLoad: Promise<THREE.Group> | null = null;
  private disposePromise: Promise<void> | null = null;
  // Placement survives across load(), so alignTo() and load() may be called in
  // either order (see alignTo).
  private pendingMatrix: THREE.Matrix4 | null = null;
  // Host exposure control (see setExposure). Keyed on the material so a scene
  // whose geometry streams in over time converges on one exposure, and so the
  // authored value is never lost to repeated scaling.
  private exposure = 1;
  private visible = true;
  private renderOrderDirty = false;
  private datasetFaultLoader: NonNullable<ReturnType<typeof getSceneLoader>> | null = null;
  private datasetFaultUnsubscribe: (() => void) | null = null;
  private readonly datasetFaultListeners = new Set<(error: Error) => void>();
  private readonly authoredOpacity = new WeakMap<THREE.Material, number>();
  private readonly appliedExposure = new WeakMap<THREE.Material, number>();
  // nD update coalescing. A scrubbing host outruns the loader by ~30x, so
  // callers arriving during an in-flight pass are all served by the NEXT pass
  // rather than each getting one of their own.
  private inFlightDimUpdate: Promise<void> | null = null;
  private dimWaiters: Array<() => void> = [];
  private dimDirty = false;

  constructor(options: LuxarLayerOptions) {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      throw new Error('LuxarLayer requires a browser environment (window/document unavailable).');
    }
    this.options = options;

    applyModuleOverrides({ wasmPath: options.wasmPath, workerPath: options.workerPath });

    // Materials must know the renderer's capabilities BEFORE any node is
    // built — the GLSL vs. TSL dispatch in the material factories branches on
    // them, and a node created first would get the wrong backend.
    materialManager.setCaps(createRendererCapabilities(options.renderer));
    this.resize();

    this.installLodRegistryFactory();
    this.installDepthSort();
  }

  /** The loaded scene root, or null before {@link load} resolves. */
  get root(): THREE.Group | null {
    return this.rootGroup;
  }

  /** Current terminal archive fault, or null before one occurs or after replacement/disposal. */
  getDatasetFault(): Error | null {
    return this.datasetFaultLoader?.archiveFault ?? null;
  }

  /**
   * Subscribe to the current dataset's one-shot terminal archive fault.
   * A fault already latched by the loaded dataset is replayed immediately.
   * Listener exceptions are logged and do not interrupt layer loading or other listeners.
   */
  onDatasetFault(listener: (error: Error) => void): () => void {
    this.assertLive();
    this.datasetFaultListeners.add(listener);
    const fault = this.getDatasetFault();
    if (fault) this.invokeDatasetFaultListener(listener, fault);
    return () => this.datasetFaultListeners.delete(listener);
  }

  /**
   * Load a `.luxar.zarr` scene and add it to the host scene.
   *
   * Resolves once the first slice has committed, so a caller that awaits this
   * can frame the camera on real bounds rather than an empty group.
   *
   * Calling it a second time is a **dataset switch**: the previous root is
   * detached (`loadScene` has already disposed its loader, so leaving it
   * attached would draw over disposed backing stores). Calling it again while a
   * load is still in flight **throws** — see the comment in the body.
   */
  async load(src: string): Promise<THREE.Group> {
    this.assertLive();
    // Concurrent loads corrupt each other: the second's `createLoaderAsync`
    // disposes the first's loader mid-flight, and whichever resolves LAST wins
    // the root slot — so the losing race can leave a group backed by a disposed
    // loader attached to the host scene. Sequential switches are fine (the old
    // root is detached below); overlapping ones are refused rather than
    // silently producing dead geometry.
    if (this.inFlightLoad) {
      throw new Error(
        'LuxarLayer.load() is already in progress; await it before loading another scene.'
      );
    }
    const pending = this.loadInner(src);
    this.inFlightLoad = pending;
    try {
      const root = await pending;
      this.installDatasetFaultLoader();
      return root;
    } finally {
      if (this.inFlightLoad === pending) this.inFlightLoad = null;
    }
  }

  private async loadInner(src: string): Promise<THREE.Group> {
    this.clearDatasetFaultLoader();
    let root: THREE.Group;
    try {
      root = await loadScene(src, this.options.loaderConfig, LOADER_ID);
    } catch (error) {
      // A dataset switch destroys the previous loader before fetching the new
      // scene. If that fetch fails, its old root is backed by dead resources.
      if (this.rootGroup) {
        this.detachRoot(this.rootGroup);
        this.rootGroup = null;
      }
      throw error;
    }
    // A dispose() that lands mid-load must not leave either the group or the
    // loader alive. dispose() waits for this cleanup before dropping globals.
    if (this.disposed) {
      this.detachRoot(root);
      await SceneLoaderManager.getInstance().destroyLoaderAsync(LOADER_ID);
      return root;
    }

    // A second load() is a dataset switch: `loadScene` has already disposed the
    // previous SceneLoader, so leaving the old group attached would keep the
    // host drawing geometry over disposed backing stores.
    if (this.rootGroup && this.rootGroup !== root) {
      this.detachRoot(this.rootGroup);
    }

    this.options.scene.add(root);
    this.rootGroup = root;
    root.visible = this.visible;
    this.applyRenderOrder();
    // A placement declared before the scene arrived applies now. Hosts derive
    // the matrix from their own metadata, which resolves on a schedule
    // unrelated to this fetch, so either order is legitimate.
    if (this.pendingMatrix) this.applyMatrix(this.pendingMatrix);

    // Dimension metadata has to resolve AFTER the group is attached (the
    // manager walks the scene) and BEFORE the first slice query, which reads
    // the displayed-dims set.
    sceneDimsManager.initFromScene(this.options.scene);

    const dims = sceneDimsManager.getDims();
    try {
      if (dims) await updateSceneForDimensions(dims, root, LOADER_ID);
    } catch (error) {
      this.detachRoot(root);
      this.rootGroup = null;
      throw error;
    }
    if (this.disposed) return root;
    this.applyRenderOrder();
    this.configureBlendWarmup();
    await warmSceneBlendModePrograms(root);

    log.info(Modules.LUXAR, `Layer loaded: ${src}`);
    return root;
  }

  /**
   * Per-frame bookkeeping. Call once per host frame, before the host renders.
   *
   * Cheap and self-gating: it early-outs before a scene is loaded, and both
   * inner evaluations early-out when nothing needs re-sorting or swapping.
   */
  update(): void {
    if (!this.rootGroup || this.disposed) return;
    // Order matters: depth sorting assigns the cross-node render order that a
    // LOD swap may then invalidate, so sorting runs first.
    evaluateDepthSortPerFrame();
    getSceneLoader(LOADER_ID)?.lodGroupRegistry?.evaluatePerFrame();
    // Scene / LOD / partition groups may attach lazily after load(). Three.js
    // replaces the transparent group-order key at every Group boundary, so a
    // newly attached default-zero group would otherwise nullify the option.
    if (this.renderOrderDirty) this.applyRenderOrder();
    // Geometry streams in and LOD swaps mint materials after setExposure() ran,
    // so a non-default exposure has to be re-asserted. Skipped entirely at the
    // authored exposure, which is the common case.
    if (this.exposure !== 1) this.applyExposure();
  }

  /**
   * Push the host's camera projection and drawing-buffer size into the
   * material manager. Call after a viewport resize, a DPR change, or a change
   * to the camera's FOV / ortho frustum.
   *
   * Does NOT push a near-cull distance. `SceneManager` derives one from its
   * dynamic scene-bounds cache and passes it as a fourth argument, which fades
   * geometry approaching the near plane; without it the shared near fade stays
   * at its default and elements pop instead. Wiring it here would mean
   * reproducing the bounds cache, so it is a known limitation rather than an
   * oversight — a host that cares can keep its own near plane clear of the
   * data.
   */
  resize(): void {
    if (this.disposed) return;
    // The public option is the broad `THREE.Camera` so any host camera is
    // accepted; the projection helpers want the narrower perspective/ortho
    // union, and a camera that is neither falls back to the default FOV.
    const camera = this.options.getCamera() as LuxarCamera;
    this.options.renderer.getDrawingBufferSize(this.bufferSize);
    if (isOrthographicCamera(camera)) {
      materialManager.updateCameraParams(getOrthoFrustumHeight(camera), this.bufferSize, true);
    } else {
      materialManager.updateCameraParams(getCameraFovRadians(camera), this.bufferSize, false);
    }
  }

  /** Dimension metadata for the loaded scene (cloned), or null if none. */
  getDimensions(): EmbedderDimensions | null {
    const dims = sceneDimsManager.getDims();
    if (!dims) return null;
    return {
      ndim: dims.ndim,
      displayed: [...dims.displayed],
      currentStep: [...dims.currentStep],
      metadata: sceneDimsManager.getDimensionMetadata().map((metadata) => ({
        ...metadata,
        ...(metadata.range
          ? { range: [metadata.range[0], metadata.range[1]] as [number, number] }
          : {}),
        ...(metadata.categories ? { categories: [...metadata.categories] } : {}),
      })),
      ranges: (sceneDimsManager.getDimensionRanges() ?? []).map(
        (range) => [range[0], range[1]] as [number, number]
      ),
    };
  }

  /**
   * Index of the dimension with this name (case-insensitive), or null.
   *
   * Convenience for the common host case of mapping its own timeline onto
   * whichever axis the scene calls "time".
   */
  findDimension(name: string): number | null {
    const names = sceneDimsManager.getDimensionNames();
    const index = names.findIndex((n) => n.toLowerCase() === name.toLowerCase());
    return index >= 0 ? index : null;
  }

  /**
   * Axis names in center-column order, or `[]` before load.
   *
   * A host aligning Luxar data to its own frame needs to know which column is
   * which. Producers disagree — a scene may name its axes `Z, Y, X` or `X, Y, Z`
   * for the same specimen, and one authored by `luxar gsplat convert` names them
   * `dim0…dimN` and says nothing at all. Getting it wrong renders a plausible,
   * silently transposed scene, so hosts should branch on these rather than
   * assume a column order.
   */
  getDimensionNames(): string[] {
    return sceneDimsManager.getDimensionNames();
  }

  /**
   * Move a non-displayed dimension (time, channel, z) to `value`.
   *
   * Updates are **coalesced**: while one slice is in flight, further calls
   * replace a single queued update rather than queueing each one. A host
   * scrubbing a slider at frame rate would otherwise issue tens of full view
   * updates per second, all but the last of them already stale.
   *
   * Resolves when the slice this call led to has committed. Callers driving
   * playback should NOT await it — see {@link prefetchDimensionValue}.
   *
   * A no-op before {@link load}: the dimension set comes from the scene, and
   * `initFromScene` would overwrite any pre-load value with the scene's own
   * defaults anyway. Hosts restoring a saved timepoint should apply it after
   * `load()` resolves.
   */
  async setDimensionValue(index: number, value: number): Promise<void> {
    this.assertLive();
    if (!this.rootGroup) return;

    // The dims manager always takes the new value, so the final position is
    // correct even when the intermediate slices are coalesced away.
    sceneDimsManager.setDimensionValue(index, value);

    const settled = new Promise<void>((resolve) => this.dimWaiters.push(resolve));
    this.dimDirty = true;
    if (!this.inFlightDimUpdate) this.startDimDrain();
    return settled;
  }

  private startDimDrain(): void {
    let resolveDrain!: () => void;
    this.inFlightDimUpdate = new Promise<void>((resolve) => {
      resolveDrain = resolve;
    });
    void this.drainDimUpdates()
      .catch((error) => {
        log.error(Modules.LUXAR, 'LuxarLayer dimension update failed', error);
      })
      .then(resolveDrain);
  }

  /**
   * Run view updates until no new dimension value has arrived.
   *
   * Each pass CLAIMS the waiters queued when it starts and resolves exactly
   * those once it commits, so every caller is released by the pass that
   * included its value — and callers that arrive mid-pass share the next one
   * instead of each triggering their own.
   */
  private async drainDimUpdates(): Promise<void> {
    try {
      while (this.dimDirty && !this.disposed && this.rootGroup) {
        this.dimDirty = false;
        const claimed = this.dimWaiters;
        this.dimWaiters = [];

        try {
          const dims = sceneDimsManager.getDims();
          if (dims) await updateSceneForDimensions(dims, this.rootGroup, LOADER_ID);
        } finally {
          for (const resolve of claimed) resolve();
        }
      }
    } finally {
      this.inFlightDimUpdate = null;
      // Never strand a caller: anyone queued during teardown (or after a
      // getDims() miss broke the loop) resolves rather than hanging forever.
      const stranded = this.dimWaiters;
      this.dimWaiters = [];
      for (const resolve of stranded) resolve();
    }
  }

  /**
   * Warm the cache for a dimension value the host is about to move to, without
   * moving the view. Fire-and-forget; superseded by the next foreground update.
   *
   * The playback pattern is `setDimensionValue(t)` (not awaited) plus
   * `prefetchDimensionValue(t + 1)` on the same tick, so the host's own
   * timeline never stalls waiting on streamed slices.
   */
  prefetchDimensionValue(index: number, value: number, budgetMs = 16): void {
    if (this.disposed || !this.rootGroup) return;
    const dims = sceneDimsManager.getDims();
    if (!dims || index < 0 || index >= dims.ndim || !Number.isFinite(value)) return;
    let predictedValue = value;
    let min = -Infinity;
    let max = Infinity;
    const range = sceneDimsManager.getDimensionRanges()?.[index];
    if (range) {
      [min, max] = range;
      predictedValue = clamp(predictedValue, min, max);
    }
    const metadata = dims.metadata?.[index];
    if (metadata?.discrete) {
      predictedValue = snapDiscreteValue(predictedValue, metadata.step || 1, min, max);
    }
    const predicted = {
      ...dims,
      currentStep: dims.currentStep.map((current, currentIndex) =>
        currentIndex === index ? predictedValue : current
      ),
    };
    prefetchSceneForDimensions(predicted, this.rootGroup, LOADER_ID, { budgetMs });
  }

  /** Resolve once no slice update is in flight. */
  async awaitDimensionUpdate(): Promise<void> {
    await this.inFlightDimUpdate;
  }

  /**
   * Show or hide the layer's geometry.
   *
   * Toggles `visible` on the root rather than detaching it, so caches and
   * in-flight fetches survive. Lazy LOD loads pause while hidden and resume on
   * the next `update()` after re-showing; under GPU-budget pressure, resident
   * levels in a hidden layer are evicted before visible ones.
   */
  setVisible(visible: boolean): void {
    this.visible = visible;
    if (this.rootGroup) this.rootGroup.visible = visible;
  }

  /** Whether the layer's geometry is currently visible. */
  isVisible(): boolean {
    return this.rootGroup?.visible ?? false;
  }

  /**
   * Scale the layer's exposure by `multiplier`, relative to what the scene was
   * authored with. 1 restores the authored appearance.
   *
   * For the three emissive geometry types this multiplies the `opacity`
   * uniform, which in additive blending is the amount each element contributes
   * to the accumulation rather than a coverage fraction. For a Mesh in its
   * default `opaque` mode, the same uniform is cutout coverage: a value below
   * `alphaCutoff` (0.5 by default) discards the surface, while a value above it
   * does not dim the surviving fragments. Author the mesh with `normal`
   * blending when this control should produce smooth surface transparency.
   *
   * A host needs this because a scene's authored exposure was tuned against
   * *some* post-processing chain, and the host's is a different one — a value
   * that reads well in Luxar's own viewer can land dim or blown out behind a
   * host's bloom and tone mapping, with nothing wrong with the data.
   *
   * Re-applied by {@link update} as geometry streams in, so nodes that arrive
   * later match the ones already on screen.
   */
  setExposure(multiplier: number): void {
    this.exposure = multiplier;
    this.applyExposure();
  }

  /** Current exposure multiplier; 1 means the authored appearance. */
  getExposure(): number {
    return this.exposure;
  }

  /**
   * Push {@link exposure} onto any material that has not received it yet.
   *
   * The authored opacity is captured per material the first time that material
   * is seen, so repeated calls compose against the original rather than
   * compounding. Materials arriving with a later chunk are picked up on the
   * next call.
   */
  private applyExposure(): void {
    const root = this.rootGroup;
    if (!root) return;
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      const mat = mesh.material;
      if (!mat || Array.isArray(mat) || !isLuxarMaterial(mat)) return;

      let base = this.authoredOpacity.get(mat);
      if (base === undefined) {
        // Prefer the fade's own snapshot over the live uniform. During an LOD
        // cross-fade `uOpacity` is `_lodFadeBase * product`, so a material first
        // seen mid-fade would otherwise cache a fractional value as its
        // "authored" one — permanently, in the WeakMap. That is the likely case
        // rather than the exotic one: `update()` re-asserts exposure on nodes
        // the moment they commit, which is exactly when they start fading in.
        base = (mesh.userData._lodFadeBase as number | undefined) ?? readOpacity(mat);
        this.authoredOpacity.set(mat, base);
      } else if (this.appliedExposure.get(mat) === this.exposure) {
        return; // already at this exposure
      }

      const next = base * this.exposure;
      // An in-flight LOD fade recomputes `_lodFadeBase x fadeProduct` every
      // frame, so writing the uniform here would be overwritten on the next
      // fade frame. Rebase the fade's snapshot instead — the same contract
      // `ui/layers/layer-apply.ts` follows for the Layers panel.
      if (mesh.userData._lodFadeBase != null) {
        mesh.userData._lodFadeBase = next;
      } else {
        mat.updateOpacity(next);
      }
      this.appliedExposure.set(mat, this.exposure);
    });
  }

  /** World-space bounds of the loaded scene, or null before load. */
  getBounds(): THREE.Box3 | null {
    if (!this.rootGroup) return null;
    return new THREE.Box3().setFromObject(this.rootGroup);
  }

  /**
   * Place the layer root in the host's world space.
   *
   * A host whose own data lives in a normalized or otherwise transformed frame
   * uses this to bring Luxar's data coordinates into it. Applied to the root's
   * matrix directly, so nothing downstream (LOD selection reads projected
   * screen area, which is transform-invariant) needs to know.
   *
   * Order-independent with respect to {@link load}: a matrix declared first is
   * remembered and applied when the scene arrives. The host's placement usually
   * comes from its own metadata, which resolves on a schedule unrelated to the
   * scene fetch, so requiring one order would make correctness a race.
   */
  alignTo(matrix: THREE.Matrix4): void {
    this.pendingMatrix = matrix.clone();
    if (this.rootGroup) this.applyMatrix(this.pendingMatrix);
  }

  private applyMatrix(matrix: THREE.Matrix4): void {
    const root = this.rootGroup;
    if (!root) return;
    root.matrixAutoUpdate = false;
    root.matrix.copy(matrix);
    root.updateMatrixWorld(true);
  }

  private applyRenderOrder(): void {
    const root = this.rootGroup;
    if (!root) return;
    const renderOrder = this.options.renderOrder ?? DEFAULT_RENDER_ORDER;
    root.traverse((object) => {
      if ((object as THREE.Group).isGroup) object.renderOrder = renderOrder;
    });
    this.renderOrderDirty = false;
  }

  private handleGeometryCommit(): void {
    this.renderOrderDirty = true;
    this.options.requestRender?.();
  }

  private detachRoot(root: THREE.Group): void {
    root.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      releaseDepthSortNode(object);
      object.geometry.dispose();
    });
    this.options.scene.remove(root);
  }

  private configureBlendWarmup(): void {
    const renderer = isWebGLRenderer(this.options.renderer) ? this.options.renderer : null;
    configureBlendModeProgramWarmup({
      enabled: renderer !== null,
      renderer,
      camera: this.options.getCamera(),
      targetScene: this.options.scene,
    });
  }

  /** Back off Luxar's GPU budget after a host WebGL context-loss event. */
  handleContextLost(): void {
    if (this.disposed) return;
    clearBlendModeProgramWarmup();
    reduceGpuByteBudgetForContextLoss();
  }

  /**
   * Rebuild Luxar-owned GPU resources after the host restores a WebGL context.
   * Call from the host's `webglcontextrestored` handler after resetting its
   * renderer and rebuilding any host-owned post-processing resources.
   */
  handleContextRestored(): void {
    if (this.disposed || !this.rootGroup) return;
    materialManager.rebuildAfterContextRestore();
    markSceneResourcesDirtyForContextRestore(this.rootGroup);
    this.resize();
    getSceneLoader(LOADER_ID)?.nodeFactory.rebuildAfterContextRestore(this.rootGroup);
    this.configureBlendWarmup();
    void warmSceneBlendModePrograms(this.rootGroup);
    this.options.requestRender?.();
  }

  /**
   * Tear down everything the layer owns: the scene group, the loader and its
   * caches, the data-worker pool, and the depth-sort worker.
   *
   * Awaits loader teardown for real, via `destroyAllAsync()`: every
   * prefetcher, caching store, and L0 cache is drained before the pools that
   * serve them are torn down. `disposeInstance()` alone is explicitly
   * fire-and-forget, so a host that awaits this would otherwise get a promise
   * that guarantees nothing and a subsequent `load()` could race the drain.
   * This is stronger than what `LuxarApp` does at shutdown, and deliberately
   * so — a host may remount repeatedly within one page lifetime.
   *
   * The host's renderer, camera, and scene are left untouched.
   */
  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposePromise = this.disposeInner();
    return this.disposePromise;
  }

  private async disposeInner(): Promise<void> {
    this.disposed = true;
    this.clearDatasetFaultLoader();
    this.datasetFaultListeners.clear();

    // Keep the process singletons alive until loadScene has either failed or
    // disposed the loader it just registered. React unmounts commonly land
    // here while load() is pending (and StrictMode does so deliberately).
    if (this.inFlightLoad) {
      try {
        await this.inFlightLoad;
      } catch {
        // The original load() caller still receives the rejection; teardown
        // must continue and destroy any loader registered before the failure.
      }
    }
    if (this.inFlightDimUpdate) await this.inFlightDimUpdate;

    if (this.rootGroup) {
      this.detachRoot(this.rootGroup);
      this.rootGroup = null;
    }
    this.pendingMatrix = null;
    configureBlendModeProgramWarmup({
      enabled: false,
      renderer: null,
      camera: null,
      targetScene: null,
    });

    // Same three-tier ordering LuxarApp uses: dimension state and the loader
    // (which holds worker references) before the pools that serve them, so no
    // in-flight call outlives its owner.
    const steps: Array<[string, () => void | Promise<void>]> = [
      ['sceneDims', () => sceneDimsManager.reset()],
      [
        'sceneLoaderManager',
        async () => {
          // Await the drain, THEN drop the singleton. `disposeInstance()` runs
          // `destroyAll()`, which fires disposes without awaiting them; by the
          // time it runs here the loaders map is already empty, so it is a
          // cheap no-op that just clears the instance.
          await SceneLoaderManager.getInstance().destroyAllAsync();
          SceneLoaderManager.disposeInstance();
        },
      ],
      ['materialManager', () => materialManager.dispose()],
      ['workerPool', () => disposeWorkerPool()],
      ['depthSort', () => disposeDepthSort()],
    ];
    for (const [label, step] of steps) {
      try {
        await step();
      } catch (error) {
        log.error(Modules.LUXAR, `LuxarLayer.dispose(): ${label} threw`, error);
      }
    }
  }

  private installLodRegistryFactory(): void {
    const { lodFade = true, lodEnergyComp = true, lodFinest = false } = this.options;
    SceneLoaderManager.getInstance().setLODGroupRegistryFactory(
      (owner) =>
        new LODGroupRegistry({
          getCamera: () => this.options.getCamera(),
          getViewportSize: () => this.options.getViewportSize(),
          // Empty, not [0, 1, 2], before dims resolve: the registry's
          // `displayDims.length < 2` early-return then skips evaluation, whereas
          // the plausible-looking default projects a 2D scene onto a phantom Z.
          // Matches `core/app/init/pipeline.ts`.
          getDisplayDims: () => sceneDimsManager.getDims()?.displayed ?? [],
          getResidentByteBudget: () => getGpuByteBudget(),
          // Both halves of the budget are required: `lod-eviction` bails on
          // `!getResidentBytes`, so supplying only the budget makes it
          // decorative and no cold level is ever demoted — an unbounded VRAM
          // climb over a long host session, with nothing to see until it fails.
          getResidentBytes: () => owner.gpuBufferPool?.getResidentBytes() ?? 0,
          getViewVersion: () => owner.currentViewVersion,
          getCrossFadeEnabled: () => lodFade,
          getEnergyCompEnabled: () => lodEnergyComp,
          getForceFinestLOD: () => lodFinest,
          registerMaterial: (material) => materialManager.register(material),
          requestRender: () => this.handleGeometryCommit(),
        })
    );
    SceneLoaderManager.getInstance().setRequestRender(() => this.handleGeometryCommit());
    SceneLoaderManager.getInstance().setKTX2TextureDecoder(
      createKTX2TextureDecoder(this.options.renderer)
    );
  }

  private clearDatasetFaultLoader(): void {
    this.datasetFaultUnsubscribe?.();
    this.datasetFaultUnsubscribe = null;
    this.datasetFaultLoader = null;
  }

  private installDatasetFaultLoader(): void {
    if (this.disposed) return;
    const sceneLoader = getSceneLoader(LOADER_ID);
    this.datasetFaultLoader = sceneLoader;
    if (sceneLoader) {
      this.datasetFaultUnsubscribe = sceneLoader.onArchiveFault(
        (error) => this.notifyDatasetFault(error),
        { replayCurrent: true }
      );
    }
  }

  private notifyDatasetFault(error: Error): void {
    for (const listener of [...this.datasetFaultListeners]) {
      this.invokeDatasetFaultListener(listener, error);
    }
  }

  private invokeDatasetFaultListener(listener: (error: Error) => void, error: Error): void {
    try {
      listener(error);
    } catch (listenerError) {
      log.warning(Modules.LUXAR, 'LuxarLayer dataset fault listener threw:', listenerError);
    }
  }

  private installDepthSort(): void {
    // The module-level flag is what actually gates registration, the per-frame
    // cross-node renderOrder pass, and worker spawn. Returning early without
    // clearing it leaves `depthSort: false` doing nothing at all.
    // AND with the build config the same way the app does, so a config that
    // ships depth sorting off is honoured in layer mode too. No behavioural
    // difference while the config default is true — it bites only if that flips.
    const enabled = config.depthSort.enabled && this.options.depthSort !== false;
    setDepthSortEnabled(enabled);
    if (!enabled) return;
    configureDepthSort({
      getCamera: () => this.options.getCamera(),
      requestRender: () => this.options.requestRender?.(),
      requestReprocess: () => {
        void getSceneLoader(LOADER_ID)?.updateView({});
      },
      isLoadInProgress: () => getSceneLoader(LOADER_ID)?.isUpdateInProgress() ?? false,
      getProfiler: () => SceneLoaderManager.getInstance().getProfiler(),
      getDisplayDims: () => sceneDimsManager.getDims()?.displayed ?? null,
    });
    // Spawn the sort worker while the page is idle. Deferring it to the first
    // order-dependent commit puts its initialize() reply on a main thread that
    // is saturated decoding the scene, where it can miss its deadline.
    warmUpDepthSortWorker();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('LuxarLayer has been disposed.');
  }
}
