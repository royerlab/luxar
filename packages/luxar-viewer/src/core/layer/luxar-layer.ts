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
 * returns a plain `THREE.Group`, and `LODGroupRegistryDeps` /
 * `DepthSortCoordinator.configure` are defined purely in terms of injectable
 * getters. Nothing in the data, cache, LOD, or material path needs `SceneManager`.
 * The layer owns its own `DepthSortCoordinator`, so it sorts against the host's
 * camera even when a `LuxarApp` or another layer shares the page.
 *
 * The minimal embed shape:
 *
 * ```ts
 * import { LuxarLayer } from '@luxar/viewer';
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
 * - Scene environment sharing. If the host leaves `scene.environment` unset, the
 *   first Luxar physical mesh installs a prefiltered `RoomEnvironment` there, which
 *   can also affect the host's own lighting-model materials. A host-supplied
 *   environment is preserved, and an environment owned by the layer is released
 *   by {@link LuxarLayer.dispose}.
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
 * ## Several hosts on one page
 *
 * A layer may share the page with a `LuxarApp` and with other layers: it owns
 * its own loader manager, dimension state, material manager, depth-sort
 * coordinator and blend warm-up, and holds a lease on the page's shared worker
 * pools. Two layers in ONE THREE.Scene draw as whole layers in `renderOrder`
 * order rather than interleaving by depth (the second one warns). Every host on
 * the page must render through the same backend. See the module README.
 *
 * The layer has no notifier UI; archive failures are exposed through
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
import type { SceneLoader } from '../../data/scene-loader';
import { SceneLoaderManager } from '../../data/scene-loader-manager';
import { LODGroupRegistry } from '../../scene/lod-group-registry';
import { SceneDimsManager, snapDiscreteValue } from '../../scene/scene-dims-manager';
import { rebaseLodFade, type FadeableMaterial } from '../../scene/lod-fade';
import { MaterialManager } from '../../rendering/material-manager';
import { resolveMaterialBackend } from '../../rendering/material-manager/factories';
import {
  createSceneEnvironment,
  type SceneEnvironment,
} from '../../rendering/environment/scene-environment';
import { createKTX2TextureDecoder } from '../../rendering/ktx2-texture-decoder';
import {
  createRendererCapabilities,
  isWebGLRenderer,
  type RendererCapabilities,
} from '../../rendering/renderer-capabilities';
import {
  getGpuByteBudget,
  initializeGpuByteBudget,
  reduceGpuByteBudgetForContextLoss,
} from '../../rendering/gpu-byte-budget';
import {
  WebGLBlendWarmupManager,
  registerBlendWarmupManager,
  unregisterBlendWarmupManager,
} from '../../rendering/webgl-blend-warmup';
import { DepthSortCoordinator, releaseDepthSortNode } from '../../rendering/depth-sort-coordinator';
import { releaseWorkerPool, retainWorkerPool } from '../../workers/worker-pool';
import type { DatasetFaultPayload } from '../app/embedder/events';
import { applyModuleOverrides } from '../app/init/module-overrides';
import { getInputProfile } from '../../utils/input-capabilities';
import { config } from '../../config';
import { isLuxarMaterial } from '../../ui/layers/luxar-material';
import { clamp } from '../../utils/clamp';
import { log, Modules } from '../../utils/log';
import { markSceneResourcesDirtyForContextRestore } from '../../scene/scene-manager/render-pipeline/webgl-context-recovery';
import { configureRendererBackend } from '../../scene/scene-manager/render-pipeline/renderer-setup';
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
   * progressive refinement, lazy LOD loads, retries — and from
   * {@link LuxarLayer.update} when a frame's sort / LOD evaluation changed what
   * is drawn (a fade advances one step per frame). Hosts with an on-demand
   * render loop must wire this or late commits will not repaint and fades
   * freeze. Hosts that render continuously can omit it.
   */
  requestRender?: () => void;
  /** Cache and prefetch flags forwarded to the data loader. */
  loaderConfig?: LoaderConfig;
  /**
   * Session-wide GPU geometry budget in bytes. `null` auto-sizes from device
   * memory, measured heap, and device class; `0` disables byte-budget eviction,
   * and a positive value pins the budget. Defaults to
   * `config.dataLoading.performance.gpuPoolMaxBytes`.
   */
  gpuPoolMaxBytes?: number | null;
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
  /**
   * Replacement-LOD bias in screen-area units. Non-finite or non-positive
   * values are treated as the neutral `1`. Default 1.
   */
  lodBias?: number;
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
 * The live layers drawing into each host scene. Two layers in ONE scene is a
 * supported but limited configuration (see {@link warnIfSceneShared}).
 */
const layersByScene = new WeakMap<THREE.Scene, Set<LuxarLayer>>();

/**
 * Say, once per extra layer, what two layers in one scene do and do not get.
 *
 * Each layer owns its depth-sort coordinator, so each assigns cross-node
 * `renderOrder` ranks over its OWN nodes only; and the layer stamps its
 * `renderOrder` option onto every Group it owns, which three compares BEFORE any
 * per-mesh rank. So the two layers' transparent geometry never interleaves by
 * depth: the whole layer with the lower `renderOrder` draws first, and with
 * equal values the order between them is whatever three's group sort yields.
 * Interleaving two independently streamed, independently sorted datasets would
 * need one ordering domain spanning both hosts — deliberately not built. A
 * second layer is not refused, because a distinct `renderOrder` is a coherent,
 * stated order (e.g. an annotation layer always over a volume).
 */
function warnIfSceneShared(scene: THREE.Scene, layer: LuxarLayer, renderOrder: number): void {
  let layers = layersByScene.get(scene);
  if (!layers) {
    layers = new Set();
    layersByScene.set(scene, layers);
  }
  if (layers.size > 0) {
    log.warning(
      Modules.LUXAR,
      `${layers.size + 1} LuxarLayers share one THREE.Scene. Each depth-sorts only its own ` +
        'nodes, so their transparent geometry does not interleave by depth: the layer with the ' +
        'lower `renderOrder` option draws first as a whole, and with equal values (this one ' +
        `uses ${renderOrder}) the order between them is undefined. Give each layer a distinct ` +
        '`renderOrder`, or render them in separate scenes.'
    );
  }
  layers.add(layer);
}

/**
 * A Luxar scene rendered inside a host-owned Three.js pipeline.
 *
 * See the module docstring for the embed shape and the host's per-frame
 * responsibilities.
 */
export class LuxarLayer {
  private readonly options: LuxarLayerOptions;
  private readonly capabilities: RendererCapabilities;
  private readonly bufferSize = new THREE.Vector2();

  private rootGroup: THREE.Group | null = null;
  /**
   * This layer's own depth-sort coordinator: it tracks only this layer's
   * nodes, sorts them against `options.getCamera()` and wakes only
   * `options.requestRender` — so a layer sharing a page with a LuxarApp or
   * another layer never sorts against, or wakes, someone else's view. The
   * SortWorker itself is shared page-wide.
   */
  private readonly depthSort = new DepthSortCoordinator();
  /**
   * This layer's OWN material manager, loader manager and dimension state.
   * Never the LuxarApp's (`materialManager` / `SceneLoaderManager.getInstance()`
   * / `sceneDimsManager`): a layer sharing the page with an app or another
   * layer builds its materials against its own renderer capabilities, pushes
   * camera parameters only to them, registers its loader, resolves its
   * dimensions, and tears all of it down without touching theirs.
   */
  private readonly materials = new MaterialManager();
  private readonly sceneLoaders = new SceneLoaderManager({ materials: this.materials });
  private readonly dims = new SceneDimsManager();
  /**
   * This layer's blend-program warm-up, against the HOST renderer and scene.
   * Registered so node commits in the host scene are routed here rather than
   * to the LuxarApp's warm-up (which targets the app's own renderer).
   */
  private readonly blendWarmup = new WebGLBlendWarmupManager();
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
  private datasetFaultLoader: SceneLoader | null = null;
  private datasetFaultSrc: string | null = null;
  private datasetFaultUnsubscribe: (() => void) | null = null;
  private readonly datasetFaultListeners = new Set<(payload: DatasetFaultPayload) => void>();
  private readonly authoredOpacity = new WeakMap<THREE.Material, number>();
  private readonly appliedExposure = new WeakMap<THREE.Material, number>();
  // nD update coalescing. A scrubbing host outruns the loader by ~30x, so
  // callers arriving during an in-flight pass are all served by the NEXT pass
  // rather than each getting one of their own.
  private inFlightDimUpdate: Promise<void> | null = null;
  private dimWaiters: Array<() => void> = [];
  private dimDirty = false;
  private environment: SceneEnvironment | null = null;
  private unsubscribeEnvironment: (() => void) | null = null;

  constructor(options: LuxarLayerOptions) {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      throw new Error('LuxarLayer requires a browser environment (window/document unavailable).');
    }
    this.options = options;

    applyModuleOverrides({ wasmPath: options.wasmPath, workerPath: options.workerPath });
    retainWorkerPool(this);
    registerBlendWarmupManager(this.blendWarmup);
    warnIfSceneShared(options.scene, this, options.renderOrder ?? DEFAULT_RENDER_ORDER);
    initializeGpuByteBudget(options.gpuPoolMaxBytes);

    // Materials must know the renderer's capabilities BEFORE any node is
    // built — the GLSL vs. TSL dispatch in the material factories branches on
    // them, and a node created first would get the wrong backend.
    this.capabilities = createRendererCapabilities(options.renderer);
    this.materials.setCaps(this.capabilities);
    // The same per-backend switches the app sets for its own renderer: left at
    // their classic-WebGL defaults, a host WebGPU renderer stalls large sorts
    // after their first slice and never evicts stale RenderObjects.
    configureRendererBackend(options.renderer, this.capabilities);
    this.resize();

    this.installLodRegistryFactory();
    // The layer's loaders' commits report to the layer's coordinator.
    this.sceneLoaders.setDepthSortCoordinator(this.depthSort);
    this.installDepthSort();
  }

  /** The loaded scene root, or null before {@link load} resolves. */
  get root(): THREE.Group | null {
    return this.rootGroup;
  }

  /** Current archive fault, or null before one occurs, once a new load() begins, or after disposal. */
  getDatasetFault(): DatasetFaultPayload | null {
    const error = this.datasetFaultLoader?.archiveFault;
    if (!error || !this.datasetFaultSrc) return null;
    return { src: this.datasetFaultSrc, error };
  }

  /**
   * Subscribe to archive faults from the current dataset.
   * A fault already latched by the loaded dataset is replayed immediately.
   * Listener exceptions are logged and do not interrupt layer loading or other listeners.
   */
  onDatasetFault(listener: (payload: DatasetFaultPayload) => void): () => void {
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
    this.setupEnvironment();
    const pending = this.loadInner(src);
    this.inFlightLoad = pending;
    try {
      const root = await pending;
      this.installDatasetFaultLoader(src);
      return root;
    } finally {
      if (this.inFlightLoad === pending) this.inFlightLoad = null;
    }
  }

  private async loadInner(src: string): Promise<THREE.Group> {
    this.clearDatasetFaultLoader();
    let root: THREE.Group;
    try {
      root = await loadScene(this.sceneLoaders, src, this.options.loaderConfig, LOADER_ID);
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
      await this.sceneLoaders.destroyLoaderAsync(LOADER_ID);
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

    // Dimension metadata has to resolve BEFORE the first slice query, which
    // reads the displayed-dims set. Resolved from THIS layer's root, not the
    // host scene: a host scene holding two layers would otherwise hand the
    // second one the first one's dimension set.
    this.dims.initFromScene(root);

    const dims = this.dims.getDims();
    try {
      if (dims) await updateSceneForDimensions(this.sceneLoaders, dims, root, LOADER_ID);
    } catch (error) {
      this.detachRoot(root);
      this.rootGroup = null;
      throw error;
    }
    if (this.disposed) return root;
    this.applyRenderOrder();
    this.configureBlendWarmup();
    await this.blendWarmup.warmScene(root);

    log.info(Modules.LUXAR, `Layer loaded: ${src}`);
    return root;
  }

  /**
   * Per-frame bookkeeping. Call once per host frame, before the host renders.
   *
   * Cheap and self-gating: it early-outs before a scene is loaded, and both
   * inner evaluations early-out when nothing needs re-sorting or swapping.
   *
   * Returns whether this frame's evaluation changed what is drawn — a
   * cross-node render order or ordering-buffer slot, a LOD level shown or
   * hidden, a cross-fade / energy-compensation opacity step, a partition part
   * culled or restored. Those are multi-frame motions (a fade advances one step
   * per evaluation), so on a change the layer also calls `requestRender`: a
   * host that renders on demand then keeps ticking until the motion settles,
   * and a host that renders continuously can ignore both.
   */
  update(): boolean {
    if (!this.rootGroup || this.disposed) return false;
    // Order matters: depth sorting assigns the cross-node render order that a
    // LOD swap may then invalidate, so sorting runs first.
    const sortChanged = this.depthSort.evaluatePerFrame();
    const lodChanged = this.evaluateLodFrame();
    // Scene / LOD / partition groups may attach lazily after load(). Three.js
    // replaces the transparent group-order key at every Group boundary, so a
    // newly attached default-zero group would otherwise nullify the option.
    if (this.renderOrderDirty) this.applyRenderOrder();
    // Geometry streams in and LOD swaps mint materials after setExposure() ran,
    // so a non-default exposure has to be re-asserted. Skipped entirely at the
    // authored exposure, which is the common case.
    if (this.exposure !== 1) this.applyExposure();
    const changed = sortChanged || lodChanged;
    if (changed) this.options.requestRender?.();
    return changed;
  }

  /**
   * Run the LOD selector for this frame and report whether the drawn state
   * changed. `takeDrawnStateChanged()` is taken every frame (as the app's
   * `'lod-group-selector'` callback does) so the flag never carries over.
   */
  private evaluateLodFrame(): boolean {
    const registry = this.sceneLoaders.getLoader(LOADER_ID)?.lodGroupRegistry;
    if (!registry) return false;
    const { levelChanged, cullChanged } = registry.evaluatePerFrame();
    return registry.takeDrawnStateChanged() || levelChanged || cullChanged;
  }

  /**
   * Push the host's drawing-buffer size and pixel ratio into the material
   * manager. Call after a viewport resize or a DPR change. An FOV / ortho-zoom
   * change or a perspective/orthographic swap needs no call: every shader reads
   * its projection terms — the ortho test included — from the camera's
   * projection matrix per draw, which also
   * makes a camera that is neither perspective nor orthographic (a plain
   * `THREE.Camera` with its own matrix) render at the right size.
   *
   * Does NOT push a near-cull distance. `SceneManager` derives one from its
   * dynamic scene-bounds cache and passes it as a third argument, which fades
   * geometry approaching the near plane; without it the shared near fade stays
   * at its default and elements pop instead. Wiring it here would mean
   * reproducing the bounds cache, so it is a known limitation rather than an
   * oversight — a host that cares can keep its own near plane clear of the
   * data.
   */
  resize(): void {
    if (this.disposed) return;
    // Any host camera works: every projection term — the ortho test
    // included — is read in shader from the camera's own projection matrix.
    this.options.renderer.getDrawingBufferSize(this.bufferSize);
    const viewportHeight = this.options.getViewportSize().height;
    const pixelRatio =
      viewportHeight > 0
        ? this.bufferSize.y / viewportHeight
        : this.options.renderer.getPixelRatio();
    this.materials.updateCameraParams(this.bufferSize, undefined, pixelRatio);
  }

  /** Dimension metadata for the loaded scene (cloned), or null if none. */
  getDimensions(): EmbedderDimensions | null {
    const dims = this.dims.getDims();
    if (!dims) return null;
    return {
      ndim: dims.ndim,
      displayed: [...dims.displayed],
      currentStep: [...dims.currentStep],
      metadata: this.dims.getDimensionMetadata().map((metadata) => ({
        ...metadata,
        ...(metadata.range
          ? { range: [metadata.range[0], metadata.range[1]] as [number, number] }
          : {}),
        ...(metadata.categories ? { categories: [...metadata.categories] } : {}),
      })),
      ranges: (this.dims.getDimensionRanges() ?? []).map(
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
    const names = this.dims.getDimensionNames();
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
    return this.dims.getDimensionNames();
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
    this.dims.setDimensionValue(index, value);

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
          const dims = this.dims.getDims();
          if (dims)
            await updateSceneForDimensions(this.sceneLoaders, dims, this.rootGroup, LOADER_ID);
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
    const dims = this.dims.getDims();
    if (!dims || index < 0 || index >= dims.ndim || !Number.isFinite(value)) return;
    let predictedValue = value;
    let min = -Infinity;
    let max = Infinity;
    const range = this.dims.getDimensionRanges()?.[index];
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
    prefetchSceneForDimensions(this.sceneLoaders, predicted, this.rootGroup, LOADER_ID, {
      budgetMs,
    });
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
      // A live LOD fade holds the uniform at `_lodFadeBase x fadeProduct`, so
      // writing `next` here would be overwritten (or, on a density-thinned
      // node, drop its compensation). Rebase the fade instead — the same
      // contract `ui/layers/layer-apply.ts` follows for the Layers panel.
      if (!rebaseLodFade(mesh, mat, next)) mat.updateOpacity(next);
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
    this.blendWarmup.configure({
      enabled: renderer !== null && getInputProfile().deviceClass !== 'mobile',
      renderer,
      camera: this.options.getCamera(),
      targetScene: this.options.scene,
    });
  }

  /** Back off Luxar's GPU budget after a host WebGL context-loss event. */
  handleContextLost(): void {
    if (this.disposed) return;
    this.blendWarmup.clear();
    reduceGpuByteBudgetForContextLoss();
  }

  /**
   * Rebuild Luxar-owned GPU resources after the host restores a WebGL context.
   * Call from the host's `webglcontextrestored` handler after resetting its
   * renderer and rebuilding any host-owned post-processing resources.
   */
  handleContextRestored(): void {
    if (this.disposed || !this.rootGroup) return;
    this.materials.rebuildAfterContextRestore();
    try {
      if (this.environment?.isReady()) this.environment.rebuild();
    } catch (error) {
      log.warning(Modules.LUXAR, 'Failed to rebuild scene environment', error);
    }
    markSceneResourcesDirtyForContextRestore(this.rootGroup);
    this.resize();
    this.sceneLoaders.getLoader(LOADER_ID)?.nodeFactory.rebuildAfterContextRestore(this.rootGroup);
    this.configureBlendWarmup();
    void this.blendWarmup.warmScene(this.rootGroup);
    this.options.requestRender?.();
  }

  /**
   * Tear down everything the layer owns: the scene group, its loader manager
   * and the loader's caches, its materials and dimension state, its depth-sort
   * coordinator and blend warm-up — and release its lease on the page-wide
   * data-worker pool and SortWorker, which terminate only when no other host
   * (a LuxarApp, another layer) still uses them.
   *
   * Awaits loader teardown for real, via `destroyAllAsync()`: every
   * prefetcher, caching store, and L0 cache is drained before the pools that
   * serve them are torn down. The manager's `dispose()` alone is explicitly
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

    // Keep the layer's loader manager and pool lease alive until loadScene has either failed or
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
    this.blendWarmup.configure({
      enabled: false,
      renderer: null,
      camera: null,
      targetScene: null,
    });
    unregisterBlendWarmupManager(this.blendWarmup);
    layersByScene.get(this.options.scene)?.delete(this);

    // Same three-tier ordering LuxarApp uses: dimension state and the loader
    // (which holds worker references) before the pools that serve them, so no
    // in-flight call outlives its owner.
    const steps: Array<[string, () => void | Promise<void>]> = [
      ['sceneDims', () => this.dims.reset()],
      [
        'sceneLoaderManager',
        async () => {
          // Await the drain, THEN release the manager. `dispose()` runs
          // `destroyAll()`, which fires disposes without awaiting them; by the
          // time it runs here the loaders map is already empty, so it only
          // releases the KTX2 decoder. The LuxarApp's manager is untouched.
          await this.sceneLoaders.destroyAllAsync();
          this.sceneLoaders.dispose();
        },
      ],
      [
        'sceneEnvironment',
        () => {
          this.unsubscribeEnvironment?.();
          this.unsubscribeEnvironment = null;
          this.environment?.dispose();
          this.environment = null;
        },
      ],
      ['materialManager', () => this.materials.dispose()],
      // Terminates the shared pool only when no other host still uses it.
      ['workerPool', () => releaseWorkerPool(this)],
      ['depthSort', () => this.depthSort.dispose()],
    ];
    for (const [label, step] of steps) {
      try {
        await step();
      } catch (error) {
        log.error(Modules.LUXAR, `LuxarLayer.dispose(): ${label} threw`, error);
      }
    }
  }

  private setupEnvironment(): void {
    if (this.environment || this.options.scene.environment !== null) return;
    this.environment = createSceneEnvironment(
      this.options.renderer,
      resolveMaterialBackend(this.capabilities),
      this.options.scene
    );
    this.unsubscribeEnvironment = this.materials.onPhysicalMaterialCreated(() => {
      try {
        this.environment?.ensure();
      } catch (error) {
        log.warning(Modules.LUXAR, 'Failed to build scene environment', error);
      }
    });
  }

  private installLodRegistryFactory(): void {
    const { lodFade = true, lodEnergyComp = true, lodFinest = false, lodBias = 1 } = this.options;
    this.sceneLoaders.setLODGroupRegistryFactory(
      (owner) =>
        new LODGroupRegistry({
          getCamera: () => this.options.getCamera(),
          getViewportSize: () => this.options.getViewportSize(),
          // Empty, not [0, 1, 2], before dims resolve: the registry's
          // `displayDims.length < 2` early-return then skips evaluation, whereas
          // the plausible-looking default projects a 2D scene onto a phantom Z.
          // Matches `core/app/init/pipeline.ts`.
          getDisplayDims: () => this.dims.getDims()?.displayed ?? [],
          hasArchiveFault: () => owner.archiveFault !== null,
          hasNetworkFailureUnder: (path) => owner.hasNetworkFailureUnder(path),
          requestReprocess: (paths) => owner.requestReprocess(paths),
          // A view PASS in flight or queued — not a refinement hold (see the
          // app pipeline's identical wiring).
          isUpdateInProgress: () => owner.isLoadPassInProgress(),
          getCommittedViewState: () => owner.committedViewState,
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
          getLodBias: () => lodBias,
          registerMaterial: (material) => this.materials.register(material),
          requestRender: () => this.handleGeometryCommit(),
        })
    );
    this.sceneLoaders.setRequestRender(() => this.handleGeometryCommit());
    this.sceneLoaders.setKTX2TextureDecoder(createKTX2TextureDecoder(this.options.renderer));
  }

  private clearDatasetFaultLoader(): void {
    this.datasetFaultUnsubscribe?.();
    this.datasetFaultUnsubscribe = null;
    this.datasetFaultLoader = null;
    this.datasetFaultSrc = null;
  }

  private installDatasetFaultLoader(src: string): void {
    this.clearDatasetFaultLoader();
    if (this.disposed) return;
    const sceneLoader = this.sceneLoaders.getLoader(LOADER_ID);
    this.datasetFaultLoader = sceneLoader;
    this.datasetFaultSrc = sceneLoader ? src : null;
    if (sceneLoader) {
      this.datasetFaultUnsubscribe = sceneLoader.onArchiveFault(
        (error) => this.notifyDatasetFault(error),
        { replayCurrent: true }
      );
    }
  }

  private notifyDatasetFault(error: Error): void {
    const src = this.datasetFaultSrc;
    if (!src) return;
    const payload = { src, error };
    for (const listener of [...this.datasetFaultListeners]) {
      this.invokeDatasetFaultListener(listener, payload);
    }
  }

  private invokeDatasetFaultListener(
    listener: (payload: DatasetFaultPayload) => void,
    payload: DatasetFaultPayload
  ): void {
    try {
      listener(payload);
    } catch (listenerError) {
      log.warning(Modules.LUXAR, 'LuxarLayer dataset fault listener threw:', listenerError);
    }
  }

  private installDepthSort(): void {
    // The coordinator's flag is what actually gates registration, the per-frame
    // cross-node renderOrder pass, and worker spawn. Returning early without
    // clearing it leaves `depthSort: false` doing nothing at all.
    // AND with the build config the same way the app does, so a config that
    // ships depth sorting off is honoured in layer mode too. No behavioural
    // difference while the config default is true — it bites only if that flips.
    const enabled = config.depthSort.enabled && this.options.depthSort !== false;
    this.depthSort.setEnabled(enabled);
    if (!enabled) return;
    this.depthSort.configure({
      getCamera: () => this.options.getCamera(),
      requestRender: () => this.options.requestRender?.(),
      requestReprocess: () => {
        void this.sceneLoaders.getLoader(LOADER_ID)?.updateView({});
      },
      isLoadInProgress: () => this.sceneLoaders.getLoader(LOADER_ID)?.isUpdateInProgress() ?? false,
      getProfiler: () => this.sceneLoaders.getProfiler(),
      getDisplayDims: () => this.dims.getDims()?.displayed ?? null,
    });
    // Spawn the sort worker while the page is idle. Deferring it to the first
    // order-dependent commit puts its initialize() reply on a main thread that
    // is saturated decoding the scene, where it can miss its deadline.
    this.depthSort.warmUp();
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('LuxarLayer has been disposed.');
  }
}
