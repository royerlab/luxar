/**
 * Unified scene loader that orchestrates the loading of complete Luxar scenes.
 *
 * This loader handles the entire scene graph, using spatial index-based
 * loading for all points nodes and managing the THREE.js scene construction.
 */

import * as zarr from './zarr';
import * as THREE from 'three';
import { getWorkerPool } from '../workers/worker-pool';
import { normalizeURL } from './scene-loader/url-normalization';
import { setupCaches } from './scene-loader/cache-setup';
import { wireMonitorAfterLoad } from './scene-loader/monitor-wiring';
import { applyEffectiveAttrs as applyEffectiveAttrsHelper } from './scene-loader/effective-attrs';
import {
  getCacheStats as getCacheStatsHelper,
  listCachedDatasets as listCachedDatasetsHelper,
  clearL0Cache as clearL0CacheHelper,
  clearL1Cache as clearL1CacheHelper,
  clearL2Cache as clearL2CacheHelper,
  clearAllCaches as clearAllCachesHelper,
  type CacheStatsSnapshot,
} from './scene-loader/cache-api';
import {
  processLinesData as processLinesDataHelper,
  commitLinesGeometry as commitLinesGeometryHelper,
  type StagedLinesCommit,
} from './scene-loader/process/data-processor-lines';
import {
  processGSplatsData as processGSplatsDataHelper,
  commitGSplatsGeometry as commitGSplatsGeometryHelper,
  type StagedGSplatsCommit,
} from './scene-loader/process/data-processor-gsplats';
import {
  createPointsLoader as createPointsLoaderHelper,
  createLinesLoader as createLinesLoaderHelper,
  createGSplatsLoader as createGSplatsLoaderHelper,
  createProgressiveGSplatsLoader as createProgressiveGSplatsLoaderHelper,
  type LoaderFactoryDeps,
} from './scene-loader/loader-factory';
import { commitPointsGeometry as commitPointsGeometryHelper } from './scene-loader/commit/commit-points-geometry';
import { ViewStateQueue } from './scene-loader/view-state-queue';
import { runGSplatsRefinement } from './gsplats/lod-refinement';
import {
  loadAndStage as pointsLoadAndStage,
  label as pointsLabel,
  type PointsHandlerCtx,
} from './points/handler';
import {
  loadAndStage as linesLoadAndStage,
  label as linesLabel,
  type LinesHandlerCtx,
} from './lines/handler';
import {
  loadAndStage as gsplatsLoadAndStage,
  label as gsplatsLabel,
  type GSplatsHandlerCtx,
} from './gsplats/handler';

export type { StagedLinesCommit } from './scene-loader/process/data-processor-lines';
export type { StagedGSplatsCommit } from './scene-loader/process/data-processor-gsplats';
import {
  DataLoader,
  ViewState,
  SceneNode,
  LoaderConfig,
  LoadedPointsData,
} from './data-loader-types';
import type { LoaderMonitor } from '../types/data-monitor-types';
import { ZarrSceneAttrs } from '../types/zarr';
import type {
  SceneLoaderMonitorPort,
  SceneLoaderMonitorFactory,
} from './scene-loader-monitor-port';
import { ArrayRefRegistry } from './array-decoder/decoder';
import { log, Modules, LogEmoji } from '../utils/log';
import { config as appConfig } from '../config';
import { MultiLevelCachingStore, DecompressedChunkCache } from '../cache';
import { disposeCustomColormapTextures } from '../rendering/colormap-textures';
import type { PointsMetadata } from '../types/points';
import type {
  LinesMetadata,
  LinesDataLoader,
  LinesViewState,
  LoadedLinesData,
} from '../types/lines';
import { isLinesUserData } from '../types/lines';
import type {
  GSplatsMetadata,
  GSplatsDataLoader,
  GSplatsUserData,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../types/gsplats';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { invertNdTransformForQuery, computeWorldNdTransform } from './transforms/nd-transform';
import { NodeFactory } from '../rendering/node-factory';
import { UpdateProfiler, type UpdateSession } from '../profiling/update-profiler';
import { LoaderRegistry } from './scene-loader/loader-registry';
import { loadOverlayConfigs } from './loaders/overlay-loader';
import { notifier } from '../utils/notifier';

/** Check if an object has any own properties (avoids Object.keys() allocation). */
import {
  hasOwnProperties,
  getOrComputeExtendedTolerance,
  validateExtendDims,
} from './scene-loader/extend-tolerance';

// ============================================================================
// Staged commit types for atomic geometry updates
// ============================================================================
// During dimension animation, all nodes must update in the same render frame
// to prevent flickering. These types hold processed data between the async
// load/process stage and the synchronous commit stage.

// StagedPointsCommit type moved to ./points/handler (step 7 of the
// god-object refactor). Imported above; re-export skipped because the
// type is internal to scene-loader + data-processor wiring.

/**
 * Classification of a per-node load failure. The actual policy in
 * `loadLeafNode` (around `:1285-1307`) is partial-scene resilience:
 * every classified kind — including `Unexpected` — is logged, toasted
 * (severity varies by kind), and the leaf returns `null` so its
 * siblings can still render. Nothing rethrows from a classified
 * `LoaderError`. The `kind` field drives the user-visible severity
 * and message, not control flow.
 */
import {
  LoaderError,
  classifyLoaderError,
  loadLeafNode as loadLeafNodeHelper,
} from './scene-loader/nodes/load-leaf-error-dispatch';
import { initializeSceneDimensions as initializeSceneDimensionsHelper } from './scene-loader/nodes/initialize-scene-dimensions';
import { buildSceneGraph as buildSceneGraphHelper } from './scene-loader/nodes/build-scene-graph';

/**
 * Main scene loader that handles the complete loading pipeline.
 *
 * Features:
 * - Spatial index-based loading for efficient nD queries
 * - Hierarchical scene graph construction
 * - Transform and rendering attribute inheritance
 * - Dimension metadata management
 * - Memory-efficient loading with proper caching
 *
 * **Lifecycle: one-shot.** Each `loadScene()` call disposes prior
 * loaders + caches and nulls the monitor reference. Reusing a single
 * `SceneLoader` instance across two `loadScene()` calls is unsupported
 * and will leave the second load with a null monitor reference. Use
 * `SceneLoaderManager.createLoader()` (the canonical entry point in
 * `data/zarr-loader.ts`), which constructs a fresh loader per load —
 * the SceneLoaderManager handles the destroy/recreate dance for you.
 */
export class SceneLoader {
  private _zarrStore: zarr.Readable | null = null;

  /** Public accessor for the zarr store (needed by LabelLoader). */
  get zarrStore(): zarr.Readable | null {
    return this._zarrStore;
  }
  private cachingStore: MultiLevelCachingStore | null = null;
  // L0 decompressed chunk cache - caches decoded zarr chunks to avoid Blosc decompression
  private l0Cache: DecompressedChunkCache | null = null;
  private registry = new LoaderRegistry();

  // Delegate to registry for backwards compatibility within this class
  private get loaders() {
    return this.registry.loaders;
  }
  private get linesLoaders() {
    return this.registry.linesLoaders;
  }
  private get gsplatLoaders() {
    return this.registry.gsplatLoaders;
  }
  private get failedLoaders() {
    return this.registry.failedLoaders;
  }

  private viewState: ViewState;
  private config: LoaderConfig;
  private rootGroup: THREE.Group | null = null;
  /**
   * Optional monitor port — populated at construction by the factory
   * passed in via `SceneLoaderManager.setMonitorFactory`. Null when no
   * UI is wired up (tests, embedders), in which case all monitor calls
   * become no-ops at the call sites.
   */
  private monitor: SceneLoaderMonitorPort | null = null;
  private arrayRefRegistry: ArrayRefRegistry;

  // GPU buffer pool for geometry reuse, integrated into
  // updatePointsGeometry/updateLinesGeometry/updateGSplatsGeometry. Enabled
  // via config.dataLoading.performance.useGPUBufferPool.
  private _gpuBufferPool: GPUBufferPool | null = null;
  public readonly nodeFactory = new NodeFactory();

  // Update profiler for timing scene updates (optional, provided by SceneLoaderManager)
  private profiler: UpdateProfiler | null = null;

  // Serialized update queue: prevents concurrent updateView calls from corrupting shared buffers
  // When a new update arrives while one is in progress, we store the latest and process it after
  private _updateInProgress = false;
  private _updateVersion = 0; // For logging/debugging
  private _sceneGraph: SceneNode | null = null;

  /**
   * View-state queue: owns `_pendingViewState` (set/take/has + drain)
   * and the per-loader previous view-state map used by predictive
   * prefetch (S6). Extracted in step 5 of the god-object refactor —
   * see ./scene-loader/view-state-queue.ts.
   *
   * The pending-state slot is overwritten on every queued update, so a
   * burst of view changes during an in-flight retry collapses to a
   * single drained call (latest-wins). The per-node prev-state map is
   * reset on dataset switch (loadScene) and dispose; skipped paths
   * forget their snapshot so the next non-skip update re-baselines.
   */
  private viewStateQueue = new ViewStateQueue();

  /**
   * Per-dataset AbortController. Created on every `loadScene` and
   * aborted at the START of the next `loadScene` (and on `dispose`)
   * so worker tasks queued by the previous dataset settle
   * immediately instead of running to completion against a
   * superseded scene. The signal is registered with the WorkerPool
   * via `setAbortSignal`. WASM execution itself cannot be cancelled,
   * but the orphan results are discarded — see {@link WorkerAbortError}.
   */
  private _datasetAbortController: AbortController | null = null;

  /** Public accessor for the scene graph built during loadScene(). */
  get sceneGraph(): SceneNode | null {
    return this._sceneGraph;
  }

  // ============================================================
  // Cache surface
  //
  // SceneLoader owns the L0 (decompressed-chunk) cache and the L1/L2
  // MultiLevelCachingStore. These methods expose a typed, public API that
  // tools (debug interface, embedders) can call without reaching into
  // private fields. Each method gracefully no-ops if the cache layer is
  // unavailable (e.g. when `noCache` is set in LoaderConfig).
  // ============================================================

  /** Snapshot of all cache levels (L0, L1, L2) for debug and embed tooling. */
  getCacheStats(): CacheStatsSnapshot {
    return getCacheStatsHelper(this.l0Cache, this.cachingStore);
  }

  /** List datasets currently held by the L1/L2 caching store. */
  async listCachedDatasets(): Promise<Awaited<ReturnType<MultiLevelCachingStore['listDatasets']>>> {
    return listCachedDatasetsHelper(this.cachingStore);
  }

  /** True if the L1/L2 caching store is configured for this loader. */
  get hasCachingStore(): boolean {
    return this.cachingStore !== null;
  }

  /** Clear the in-memory L0 decompressed-chunk cache. No-op if absent. */
  clearL0Cache(): void {
    clearL0CacheHelper(this.l0Cache);
  }

  /** Clear the in-memory L1 metadata/chunk cache. No-op if absent. */
  clearL1Cache(): void {
    clearL1CacheHelper(this.cachingStore);
  }

  /** Clear the persistent L2 OPFS cache. No-op if absent. */
  async clearL2Cache(): Promise<void> {
    await clearL2CacheHelper(this.cachingStore);
  }

  /** Clear all cache levels (L0 + L1 + L2). */
  async clearAllCaches(): Promise<void> {
    await clearAllCachesHelper(this.l0Cache, this.cachingStore);
  }

  /**
   * Return a node-attrs record with rendering attributes replaced by the
   * effective values composed along the scene-graph ancestry (root → leaf).
   * This implements the hierarchical composition described in the Python
   * core/SPECIFICATIONS.md: opacity/gamma/intensity multiply, offset adds,
   * blending_mode uses the nearest ancestor's choice.
   *
   * If the scene graph is unavailable, falls back to the node's raw attrs.
   */
  private applyEffectiveAttrs(node: SceneNode): SceneNode['attrs'] {
    return applyEffectiveAttrsHelper(this._sceneGraph, node);
  }

  constructor(
    config: LoaderConfig = {},
    id?: string,
    profiler?: UpdateProfiler,
    monitorFactory?: SceneLoaderMonitorFactory | null
  ) {
    this.profiler = profiler ?? null;
    this.config = config;
    this.viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [],
      tolerance: [],
    };
    this.arrayRefRegistry = new ArrayRefRegistry();

    // GPU buffer pool requires Float32Array data; the geometry-update path
    // falls back to the standard route for Uint8/Uint16 attributes.
    //
    // B.5: defensively guard against double-init. The current call path
    // is sequential (constructor only), so this branch runs once today —
    // but matching the OnceInit pattern used elsewhere prevents a future
    // re-init refactor from silently leaking the previous pool.
    if (appConfig.dataLoading.performance.useGPUBufferPool && !this._gpuBufferPool) {
      this._gpuBufferPool = new GPUBufferPool(
        appConfig.dataLoading.performance.gpuPoolMaxSize,
        appConfig.dataLoading.performance.gpuPoolEvictionFrames,
        appConfig.dataLoading.performance.gpuPoolEvictBatchSize,
        appConfig.dataLoading.performance.gpuPoolMaxBytes
      );
      const mb = (appConfig.dataLoading.performance.gpuPoolMaxBytes / 1024 / 1024).toFixed(0);
      log.info(
        Modules.GPU_BUFFER_POOL,
        `GPU buffer pool enabled (max size: ${appConfig.dataLoading.performance.gpuPoolMaxSize}, ` +
          `byte budget: ${mb} MB, ` +
          `eviction: ${appConfig.dataLoading.performance.gpuPoolEvictionFrames} frames, ` +
          `batch cap: ${appConfig.dataLoading.performance.gpuPoolEvictBatchSize})`
      );
    }

    // Resolve the monitor port through the injected factory. The
    // factory owns the UI-side singleton (DataMonitorManager) — we
    // only see the SceneLoaderMonitorPort surface so the data → ui
    // layer rule stays clean.
    if (typeof document !== 'undefined' && config.enableMonitor !== false && monitorFactory) {
      const monitorId = id ? `${id}-monitor` : 'default';
      this.monitor = monitorFactory(monitorId);
    }
  }

  /**
   * Load a complete scene from a Zarr store using chunk-based spatial indexing.
   *
   * Orchestrates the loading of hierarchical scene graphs, managing spatial indices,
   * attribute inheritance, and dimension metadata. Supports both points and lines
   * with automatic fallback for datasets without spatial ordering.
   *
   * The loading process:
   * 1. Opens Zarr store with optional two-level caching (L1 memory + L2 OPFS)
   * 2. Loads scene metadata and initializes dimensions
   * 3. Recursively constructs THREE.js scene graph from Zarr group hierarchy
   * 4. Creates spatial index loaders for efficient nD queries
   * 5. Connects loaders to data monitor for debugging
   *
   * @param url - Complete URL to the Zarr store. Can be:
   *              - HTTP URL: 'https://example.com/data.zarr'
   *              - Local path: '/path/to/data.zarr'
   *              - With query params: 'https://example.com/data.zarr?no-cache'
   *
   * @returns Promise resolving to a THREE.Group containing the complete scene graph.
   *          The group's userData contains:
   *          - sceneDimensions: Dimension metadata if available
   *          - bounds: AABB of all points
   *          - nodeCount: Total number of leaf nodes
   *
   * @throws {Error} If the Zarr store cannot be opened or is invalid
   * @throws {Error} If consolidated metadata (.zmetadata) is malformed
   * @throws {Error} If required arrays (positions) are missing from point nodes
   *
   * @example
   * ```typescript
   * // Load a scene from HTTP URL
   * const scene = await sceneLoader.loadScene('https://example.com/data.zarr');
   * threeScene.add(scene);
   * console.log(`Loaded ${scene.children.length} top-level nodes`);
   * ```
   *
   * @example
   * ```typescript
   * // Load with error handling
   * try {
   *   const scene = await sceneLoader.loadScene(url);
   *   if (scene.children.length === 0) {
   *     console.warn('Scene is empty');
   *   }
   * } catch (error) {
   *   console.error('Failed to load scene:', error);
   *   // Fallback to default visualization
   * }
   * ```
   *
   * @example
   * ```typescript
   * // Access scene metadata after loading
   * const scene = await sceneLoader.loadScene(url);
   * const dims = scene.userData.sceneDimensions;
   * if (dims) {
   *   console.log(`${dims.length}D dataset:`, dims.map(d => d.name).join(', '));
   * }
   * ```
   *
   * @see {@link MultiLevelCachingStore} for caching implementation
   * @see SPECIFICATIONS.md - Section 4 for complete scene loading protocol
   */
  async loadScene(url: string): Promise<THREE.Group> {
    log.custom(LogEmoji.SCENE, Modules.SCENE_LOADER, `Loading scene from ${url}`);

    // Clear any existing loaders from monitor before loading new scene
    this.monitor?.disconnectAllLoaders();

    // Abort any in-flight worker tasks queued by the previous dataset.
    // Doing this BEFORE `dispose()` settles already-racing
    // `runWithTimeout` callers immediately so they unwind without
    // waiting for the worker tasks to complete — the worker keeps
    // executing the WASM kernels to completion (no WASM cancellation),
    // but the results are dropped.
    if (this._datasetAbortController) {
      this._datasetAbortController.abort();
      this._datasetAbortController = null;
    }
    getWorkerPool().setAbortSignal(undefined);

    // Dispose of any existing loaders. Awaited so the previous caching
    // store fully drains (prefetcher tear-down, OPFS metadata flush,
    // validation cancellation) before we construct the next one — without
    // this, rapid dataset switches let an old store's writes land after
    // the new store starts initialising.
    if (this.loaders.size > 0) {
      await this.dispose();
    }

    // Fresh abort source for THIS dataset; wire into the worker pool so
    // every subsequent `runWithTimeout` races against it.
    this._datasetAbortController = new AbortController();
    getWorkerPool().setAbortSignal(this._datasetAbortController.signal);

    // S6: reset per-loader prefetch predictor state. Without this,
    // the first updateView on a new dataset would extrapolate from
    // the prior dataset's slicePosition, producing wild prefetch
    // targets.
    this.viewStateQueue.clearPrev();

    const cacheResult = await setupCaches(this.normalizeURL(url), {
      noCache: this.config.noCache,
      cacheDebug: this.config.cacheDebug,
      clearCache: this.config.clearCache,
      noPrefetch: this.config.noPrefetch,
      prefetchDebug: this.config.prefetchDebug,
    });
    this.l0Cache = cacheResult.l0Cache;
    this.cachingStore = cacheResult.cachingStore;
    this._zarrStore = (await zarr.openStore(cacheResult.rawStore)) as zarr.Readable;

    // Create root THREE.js group
    this.rootGroup = new THREE.Group();
    this.rootGroup.name = 'LuxarScene';

    // Load scene metadata
    const rootLoc = zarr.root(this._zarrStore);
    const rootZarrGroup = await zarr.open(rootLoc, { kind: 'group' });
    const sceneAttrs = rootZarrGroup.attrs as ZarrSceneAttrs;

    // Initialize scene dimensions - CRITICAL for extend_to_all feature
    if (sceneAttrs?.scene_dimensions) {
      this.initializeSceneDimensions(sceneAttrs.scene_dimensions);
      this.rootGroup.userData.sceneDimensions = sceneAttrs.scene_dimensions;

      // Log dimension initialization status for debugging
      const ndim = this.viewState.dimensions?.length ?? 0;
      if (ndim > 0) {
        log.success(
          Modules.SCENE_LOADER,
          `Scene dimensions initialized: ${ndim} dimensions, ` +
            `displayed=[${this.viewState.displayDims.join(', ')}]`
        );
      }

      // Surface a user-facing toast when the scene exceeds the WASM
      // 16-dim ceiling — the worker auto-falls-back to TS, which is
      // correct but slower, and silent fallback can confuse users
      // wondering why interaction feels sluggish.
      if (ndim > 16) {
        notifier.toast(
          `Scene has ${ndim} dimensions — WASM acceleration limited to 16D, using TypeScript fallback. ` +
            'Consider reducing dimensions for better performance.',
          5000
        );
      }
    } else {
      log.warning(
        Modules.SCENE_LOADER,
        'No scene_dimensions found in scene metadata. extend_to_all features will not work.'
      );
    }

    // Extract viewer_config if present (Python API scene defaults)
    if (sceneAttrs?.viewer_config) {
      this.rootGroup.userData.viewerConfig = sceneAttrs.viewer_config;
      log.info(
        Modules.SCENE_LOADER,
        `Viewer config found in zarr: ${Object.keys(sceneAttrs.viewer_config).join(', ')}`
      );
    }

    // Store scene-level position bounds (from Python compiler)
    // These bounds represent the full dataset extent, available immediately without loading points
    if (sceneAttrs?.position_bounds) {
      this.rootGroup.userData.positionBounds = sceneAttrs.position_bounds;
      log.info(
        Modules.SCENE_LOADER,
        `Scene bounds loaded: min=[${sceneAttrs.position_bounds.min.join(', ')}], ` +
          `max=[${sceneAttrs.position_bounds.max.join(', ')}]`
      );
    }

    // Build scene graph
    const sceneGraph = await this.buildSceneGraph(rootLoc, sceneAttrs);
    this._sceneGraph = sceneGraph;

    // Load points
    await this.loadSceneNodes(sceneGraph, this.rootGroup, rootLoc);

    // Load overlay configs (screen-space annotations)
    const overlayConfigs = await loadOverlayConfigs(this._zarrStore, rootLoc);
    if (overlayConfigs.length > 0) {
      this.rootGroup.userData.overlayConfigs = overlayConfigs;
      // Store base URL for image fetching
      this.rootGroup.userData.zarrBaseUrl = this.normalizeURL(url);
    }

    // Post-load monitor-tab provider wiring (extracted to
    // scene-loader/monitor-wiring.ts).
    wireMonitorAfterLoad({
      monitor: this.monitor,
      cachingStore: this.cachingStore,
      l0Cache: this.l0Cache,
      cacheTelemetryState: cacheResult.telemetryState,
      gpuBufferPool: this._gpuBufferPool,
      profiler: this.profiler,
      loaders: this.loaders,
      linesLoaders: this.linesLoaders,
      gsplatLoaders: this.gsplatLoaders,
      sceneGraph,
      updateVisibleCounts: () => this.updateVisibleCountsInMonitor(),
    });

    log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');

    // Schedule progressive GSplats LOD refinement after initial load.
    // loadGSplats() loads LOD 0 for each progressive loader, but LODs 1-N
    // are only loaded by the refinement loop. Without this trigger, higher
    // LODs would not load until the first updateView() call (user interaction).
    const needsPostLoadRefinement = [...this.gsplatLoaders.values()].some(
      (l) => l.hasMoreLODs === true
    );
    if (needsPostLoadRefinement) {
      log.info(
        Modules.SCENE_LOADER,
        'Scheduling post-load GSplats LOD refinement (higher LODs pending)'
      );
      // Hold the serialization lock during refinement so any updateView() calls
      // queue as _pendingViewState (which naturally cancels the refinement loop)
      this._updateInProgress = true;
      this.scheduleGSplatsRefinement();
    }

    return this.rootGroup;
  }

  /**
   * Derive a per-node view state from `this.viewState`, folding in
   * `extend_to_all` tolerance overrides and the inverse `nd_transform`
   * for the node's path.
   *
   * The single source of truth for query-state derivation: both the
   * main update path and `retryFailedLoader` go through this method so
   * a retry cannot load a different query region than a fresh update
   * would. The pre-extraction retry skipped both adjustments, which
   * could produce a "successful" retry rendering incorrect data.
   *
   * Returns `{ skip: 'extend_to_all' }` if the node's `extend_to_all`
   * dims fully cover all non-displayed dims (the work is a no-op),
   * otherwise the derived view state.
   *
   * @param path  Scene-graph path of the node, used to compose the
   *              world `nd_transform` from this node up to the root.
   * @param attrs Node attrs (only `extend_to_all` is read here).
   * @param opts.applyPartialExtendTolerance
   *              When `true`, partial `extend_to_all` coverage triggers
   *              a tolerance override via `getOrComputeExtendedTolerance`.
   *              Points/GSplats: true. Lines: false because line bounds
   *              already encode their non-displayed spatial extent.
   * @param opts.extendedToleranceCache
   *              Optional cross-node cache for the partial-extend
   *              tolerance array. Main update path passes one cache per
   *              cycle; retry passes nothing (fresh).
   */
  private deriveNodeViewState(
    path: string,
    attrs: { extend_to_all?: string[] } | undefined,
    opts: {
      applyPartialExtendTolerance: boolean;
      extendedToleranceCache?: Map<string, number[]>;
    }
  ): { skip: 'extend_to_all' } | { skip: false; viewState: ViewState } {
    const extendDims: string[] = attrs?.extend_to_all ?? [];

    let derived: ViewState = {
      displayDims: this.viewState.displayDims,
      slicePosition: this.viewState.slicePosition,
      tolerance: this.viewState.tolerance,
      dimensions: this.viewState.dimensions,
    };

    // Step 1: full-extend skip check.
    if (extendDims.length > 0 && this.viewState.dimensions) {
      const dims = this.viewState.dimensions;
      validateExtendDims(extendDims, dims);
      const nonDisplayedDims = dims
        .filter((_: { name?: string }, idx: number) => !this.viewState.displayDims.includes(idx))
        .map((d: { name?: string }) => d.name)
        .filter((name: string | undefined): name is string => !!name);

      const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
        extendDims.includes(dimName)
      );
      if (isFullyExtended) {
        return { skip: 'extend_to_all' };
      }

      // Step 2: partial-extend tolerance override (Points + GSplats only).
      if (opts.applyPartialExtendTolerance) {
        const tolerance = getOrComputeExtendedTolerance(
          this.viewState.tolerance,
          extendDims,
          this.viewState.dimensions,
          opts.extendedToleranceCache ?? new Map<string, number[]>()
        );
        derived = { ...derived, tolerance };
      }
    }

    // Step 3: nd_transform inverse for world→local query mapping.
    if (this._sceneGraph && derived.dimensions) {
      const worldNdT = computeWorldNdTransform(this._sceneGraph, path);
      if (hasOwnProperties(worldNdT)) {
        const dimNames = derived.dimensions.map((d: { name?: string }) => d.name ?? '');
        const inverted = invertNdTransformForQuery(
          derived.slicePosition,
          derived.tolerance,
          worldNdT,
          dimNames,
          derived.displayDims
        );
        derived = { ...derived, ...inverted };
      }
    }

    return { skip: false, viewState: derived };
  }

  /**
   * Shared scaffolding for the per-geometry update loops. The
   * Points / Lines / GSplats branches in updateView differ only in
   * the type-specific work (deriveNodeViewState, call
   * loader.updateView, post-process, setMetadata, return staged) —
   * the surrounding try/catch with failedLoaders bookkeeping +
   * retryCount tracking, profiler dispatch, and Promise.all are all
   * identical and live here.
   *
   * @param loaders   The map of (path → loader) for one geometry type.
   * @param loaderType Human-readable type for profiler label + error log.
   * @param updateFn  Per-loader work; returns staged commit data or
   *                  null when there's nothing to commit.
   */
  private async runLoaderUpdates<TLoader, TStaged>(
    loaders: Map<string, TLoader>,
    loaderType: 'Points' | 'Lines' | 'GSplats',
    updateFn: (path: string, loader: TLoader, session: UpdateSession) => Promise<TStaged | null>
  ): Promise<Array<{ staged: TStaged | null; session: UpdateSession }>> {
    const noopSession: UpdateSession = {
      begin: () => noopSession,
      end: () => {},
      setMetadata: () => {},
      markSkipped: () => {},
    };

    const tasks = Array.from(loaders.entries()).map(async ([path, loader]) => {
      // Open a top-level session per node and keep it alive across the
      // atomic commit stage so the per-node "Update Buffers" child entry
      // nests under this session. The caller is responsible for calling
      // session.end() once the commit has run.
      const session = this.profiler
        ? this.profiler.beginTopLevel(`${loaderType} (${path})`)
        : noopSession;
      try {
        const staged = await updateFn(path, loader, session);
        return { staged, session };
      } catch (error) {
        // Predictive prefetch is keyed by the previous successful
        // derived view-state for this path. If the demand update
        // fails, discard that baseline so the next success
        // re-baselines instead of extrapolating across a stale/error
        // gap and warming irrelevant chunks.
        this.viewStateQueue.forgetPath(path);

        const errorInfo = this.failedLoaders.get(path);
        const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
        this.failedLoaders.set(path, {
          error: error as Error,
          timestamp: Date.now(),
          retryCount,
        });
        const lcType = loaderType === 'Points' ? '' : `${loaderType.toLowerCase()} `;
        log.error(
          Modules.SCENE_LOADER,
          `Failed to update ${lcType}${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
        );
        return { staged: null, session };
      }
    });
    return Promise.all(tasks);
  }

  /**
   * Update all points and lines for a new view state.
   *
   * Uses serialized execution to prevent race conditions: only one update runs at a time.
   * If a new update arrives while one is in progress, it's queued as "pending" and processed
   * after the current update completes. Only the LATEST pending state is kept (older ones
   * are discarded), ensuring eventual convergence without starvation.
   */
  async updateView(viewState: Partial<ViewState>): Promise<void> {
    // SERIALIZATION: If an update is already in progress, queue this one and return
    if (this._updateInProgress) {
      // Store the latest pending state (supersedes any previous pending state).
      // G.3: log the supersede when a previous pending was already queued so
      // rapid slider drags surface as "v5 superseded v4, in flight v3"
      // rather than three identical "Update queued" lines.
      const supersededPrevious = this.viewStateQueue.hasPending();
      this.viewStateQueue.setPending(viewState);
      const newVersion = this._updateVersion + 1;
      if (supersededPrevious) {
        log.info(
          Modules.SCENE_LOADER,
          `Update queued (v${newVersion}) - supersedes previous pending; in-flight v${this._updateVersion}`
        );
      } else {
        log.info(
          Modules.SCENE_LOADER,
          `Update queued (v${newVersion}) - in-flight v${this._updateVersion}`
        );
      }
      return;
    }

    // Mark update as in progress
    this._updateInProgress = true;
    this._updateVersion++;
    const currentVersion = this._updateVersion;

    try {
      // CRITICAL: Deep copy arrays to prevent mutation during async operations
      // The spread operator only does shallow copy - arrays must be explicitly copied
      this.viewState = {
        ...this.viewState,
        ...viewState,
        // Always copy arrays to prevent external mutation affecting in-flight updates
        displayDims: viewState.displayDims
          ? [...viewState.displayDims]
          : [...this.viewState.displayDims],
        slicePosition: viewState.slicePosition
          ? [...viewState.slicePosition]
          : [...this.viewState.slicePosition],
        tolerance: viewState.tolerance ? [...viewState.tolerance] : [...this.viewState.tolerance],
      };

      const totalLoaders = this.loaders.size + this.linesLoaders.size + this.gsplatLoaders.size;
      log.update(
        Modules.SCENE_LOADER,
        `Updating view v${currentVersion} for ${totalLoaders} loaders`
      );

      // Start profiling update cycle
      this.profiler?.beginUpdate();

      // Cache for extended tolerance arrays — nodes with the same extend_to_all
      // dimensions share a single tolerance array instead of each copying their own.
      const extendedToleranceCache = new Map<string, number[]>();

      // ================================================================
      // Stage 1: Load + Process all nodes in parallel (async)
      // Each callback returns staged commit data WITHOUT mutating geometry,
      // so all nodes are ready before any geometry changes. Shared
      // scaffolding (try/catch + failedLoaders bookkeeping + profiler
      // dispatch) lives in runLoaderUpdates; each branch below contains
      // only the type-specific work.
      // ================================================================

      // The three inline branches collapse into thin wrappers around
      // the per-type handlers in data/{points,lines,gsplats}/handler.ts
      // (step 7 of the god-object refactor). Each handler owns the
      // per-type variations — extend_to_all opts, processX call,
      // metadata shape, version-gated log — so this orchestration site
      // just builds the ctx and dispatches.
      const pointsCtx: PointsHandlerCtx = {
        rootGroup: this.rootGroup,
        viewStateQueue: this.viewStateQueue,
        clearFailure: (path) => this.failedLoaders.delete(path),
        currentVersion,
        extendedToleranceCache,
        deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      };
      const linesCtx: LinesHandlerCtx = {
        rootGroup: this.rootGroup,
        viewStateQueue: this.viewStateQueue,
        clearFailure: (path) => this.failedLoaders.delete(path),
        currentVersion,
        updateVersion: this._updateVersion,
        deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      };
      const gsplatsCtx: GSplatsHandlerCtx = {
        rootGroup: this.rootGroup,
        viewStateQueue: this.viewStateQueue,
        clearFailure: (path) => this.failedLoaders.delete(path),
        currentVersion,
        updateVersion: this._updateVersion,
        extendedToleranceCache,
        deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      };

      const pointsTask = this.runLoaderUpdates(this.loaders, pointsLabel, (path, loader, session) =>
        pointsLoadAndStage(path, loader, session, pointsCtx)
      );

      const linesTask = this.runLoaderUpdates(
        this.linesLoaders,
        linesLabel,
        (path, loader, session) => linesLoadAndStage(path, loader, session, linesCtx)
      );

      const gsplatsTask = this.runLoaderUpdates(
        this.gsplatLoaders,
        gsplatsLabel,
        (path, loader, session) => gsplatsLoadAndStage(path, loader, session, gsplatsCtx)
      );

      // Wait for ALL loaders to complete (load + process)
      const [pointsStaged, linesStaged, gsplatsStaged] = await Promise.all([
        pointsTask,
        linesTask,
        gsplatsTask,
      ]);

      // S6: predictive prefetch now lives inside each loader-task
      // branch (Points / Lines / GSplats) and uses the per-node
      // derived view-state computed by deriveNodeViewState. The
      // global dispatch site previously here over-prefetched
      // extend_to_all-skipped nodes.

      // ================================================================
      // Stage 2: Atomic commit — ALL geometry mutations in one sync block.
      // Since JS is single-threaded, no requestAnimationFrame can fire
      // during this block, so all meshes update in the same rendered frame.
      // ================================================================

      // Advance GPU buffer pool frame counter once per update cycle
      // (not per-acquire) so eviction timing reflects actual frames
      if (this._gpuBufferPool) {
        this._gpuBufferPool.beginFrame();
      }

      // Commits run inside each per-node session so the GPU-upload step
      // ("Update Buffers") shows up under Points/Lines/GSplats in the
      // Performance tab. Always end the session afterwards — including
      // the staged === null case (loader failed or marked skipped) so
      // every opened session is closed exactly once.
      //
      // Belt-and-braces: if a commit throws synchronously, the
      // remaining iterations and the later geometry-type loops never
      // run, leaving their sessions un-ended. The outer `finally`
      // sweeps every staged session afterwards. `SessionImpl.end()`
      // is idempotent (no-ops on already-ended sessions), so this is
      // safe to overlay on the per-iteration end() calls that record
      // accurate per-node timings on the happy path.
      try {
        for (const { staged, session } of pointsStaged) {
          try {
            if (staged) this.updatePointsGeometry(staged.path, staged.data, session);
          } finally {
            session.end();
          }
        }
        for (const { staged, session } of linesStaged) {
          try {
            if (staged) this.commitLinesGeometry(staged, session);
          } finally {
            session.end();
          }
        }
        for (const { staged, session } of gsplatsStaged) {
          try {
            if (staged) this.commitGSplatsGeometry(staged, session);
          } finally {
            session.end();
          }
        }
      } finally {
        for (const { session } of pointsStaged) session.end();
        for (const { session } of linesStaged) session.end();
        for (const { session } of gsplatsStaged) session.end();
      }

      // Invalidate cached pick buffer after geometry changes
      if (pointsStaged.length > 0 || linesStaged.length > 0 || gsplatsStaged.length > 0) {
        this.nodeFactory.markPickingDirty();
      }

      // Update monitor with total visible segments across all lines nodes
      this.updateVisibleCountsInMonitor();

      // Warn user if any loaders failed
      if (this.failedLoaders.size > 0) {
        const failedPaths = Array.from(this.failedLoaders.keys()).join(', ');
        log.warning(
          Modules.SCENE_LOADER,
          `⚠️ ${this.failedLoaders.size} loader(s) failed: ${failedPaths}`
        );
        log.warning(
          Modules.SCENE_LOADER,
          `Some data could not be loaded. Failed loaders: ${failedPaths}. ` +
            'Check console output for details. Data may be incomplete.'
        );
      }
    } finally {
      // End profiling update cycle (always, even if errors)
      this.profiler?.endUpdate();

      // SERIALIZATION: Process pending update if one was queued
      // CRITICAL: Keep _updateInProgress = true until the rAF callback fires!
      // This prevents new slider events from starting updates during the yield.
      const pendingState = this.viewStateQueue.takePending();
      if (pendingState !== null) {

        // Yield to render loop: ensure at least one frame is painted before next update
        // This prevents the "updates faster than renders" problem that causes black screen
        if (typeof requestAnimationFrame !== 'undefined') {
          requestAnimationFrame(() => {
            // Release the lock right before starting the next update
            // Any slider events during the yield were queued (because lock was held)
            this._updateInProgress = false;
            this.updateView(pendingState);
          });
        } else {
          // Fallback for non-browser environments (e.g., tests)
          this._updateInProgress = false;
          this.updateView(pendingState);
        }
      } else {
        // No pending update — check if progressive GSplats loaders need refinement
        const needsRefinement = [...this.gsplatLoaders.values()].some(
          (l) => l.hasMoreLODs === true
        );

        if (needsRefinement) {
          // Keep _updateInProgress = true during refinement so slider/animation
          // events queue as _pendingViewState (which naturally cancels refinement)
          log.info(
            Modules.SCENE_LOADER,
            'Scheduling GSplats LOD refinement (hasMoreLODs=true after update)'
          );
          this.scheduleGSplatsRefinement();
        } else {
          // No pending update, no refinement needed - release the lock now
          this._updateInProgress = false;
        }
      }
    }
  }

  /**
   * Schedule progressive GSplats LOD refinement.
   *
   * Thin wrapper around `runGSplatsRefinement` in
   * `data/gsplats/lod-refinement.ts` (extracted in step 8 of the
   * god-object refactor). The full timing semantics — rAF yield per
   * pass, cancellation hand-off on pending view-state, lock release on
   * normal completion — live in the extracted module.
   */
  private async scheduleGSplatsRefinement(): Promise<void> {
    return runGSplatsRefinement({
      rootGroup: this.rootGroup,
      viewStateQueue: this.viewStateQueue,
      gsplatLoaders: this.gsplatLoaders,
      deriveNodeViewState: (path, attrs, opts) => this.deriveNodeViewState(path, attrs, opts),
      processGSplats: (path, data, viewState) => this.processGSplatsData(path, data, viewState),
      commitGSplats: (staged) => this.commitGSplatsGeometry(staged),
      updateVisibleCountsInMonitor: () => this.updateVisibleCountsInMonitor(),
      releaseLock: () => {
        this._updateInProgress = false;
      },
      retriggerUpdate: (pendingState) => {
        if (typeof requestAnimationFrame !== 'undefined') {
          requestAnimationFrame(() => {
            this._updateInProgress = false;
            this.updateView(pendingState);
          });
        } else {
          this._updateInProgress = false;
          this.updateView(pendingState);
        }
      },
    });
  }

  /**
   * Aggregate visible counts from all lines and gsplats meshes and update monitor.
   * This should be called after view updates to report accurate visible counts.
   */
  private updateVisibleCountsInMonitor(): void {
    if (!this.rootGroup || !this.monitor) return;

    let totalVisibleSegments = 0;
    let totalVisibleSplats = 0;

    // Traverse all objects in the scene graph
    this.rootGroup.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        if (isLinesUserData(object.userData)) {
          totalVisibleSegments += object.userData.visibleSegmentCount ?? 0;
        } else if (object.userData?.nodeType === 'gsplats') {
          totalVisibleSplats += (object.userData as GSplatsUserData).visibleSplatCount ?? 0;
        }
      }
    });

    this.monitor.updateVisibleSegments(totalVisibleSegments);
    this.monitor.updateVisibleSplats(totalVisibleSplats);
  }

  /**
   * Process lines data: compute tolerance, project to 3D (async).
   * Returns staged commit data without mutating any mesh geometry.
   *
   * Implementation lives in `scene-loader/process/data-processor-lines.ts`; this
   * method is a thin delegate so the pipeline can be tested in isolation
   * without instantiating a SceneLoader.
   */
  private async processLinesData(
    path: string,
    data: LoadedLinesData,
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<StagedLinesCommit | null> {
    return processLinesDataHelper(
      path,
      data,
      viewState,
      this.rootGroup,
      this._updateVersion,
      session
    );
  }

  /**
   * Commit lines geometry to GPU buffers (synchronous).
   * Called as part of the atomic commit stage — no async operations allowed.
   * The optional `session` is forwarded so the helper can record an
   * "Update Buffers" child entry under the per-node profiler session
   * (parity with Points and GSplats).
   */
  private commitLinesGeometry(staged: StagedLinesCommit, session?: UpdateSession): void {
    commitLinesGeometryHelper(staged, this.rootGroup, this._gpuBufferPool, session);
  }

  /**
   * Process gsplats data: project nD to 3D, pack Cholesky factors (async).
   * Returns staged commit data without mutating any mesh geometry.
   *
   * Implementation lives in `scene-loader/process/data-processor-gsplats.ts`.
   */
  private async processGSplatsData(
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<StagedGSplatsCommit | null> {
    return processGSplatsDataHelper(
      path,
      data,
      viewState,
      this.rootGroup,
      this._updateVersion,
      session
    );
  }

  /**
   * Commit gsplats geometry to GPU buffers (synchronous).
   * Called as part of the atomic commit stage — no async operations allowed.
   * The optional `session` is forwarded so the helper can record an
   * "Update Buffers" child entry under the per-node profiler session
   * (parity with Points and Lines).
   */
  private commitGSplatsGeometry(staged: StagedGSplatsCommit, session?: UpdateSession): void {
    commitGSplatsGeometryHelper(staged, this.rootGroup, this._gpuBufferPool, session);
  }

  /**
   * Build the scene graph structure. Implementation lives in
   * `scene-loader/nodes/build-scene-graph.ts`.
   */
  private async buildSceneGraph(
    rootLoc: zarr.Location<zarr.Readable>,
    rootAttrs: ZarrSceneAttrs
  ): Promise<SceneNode> {
    return buildSceneGraphHelper(rootLoc, rootAttrs, this._zarrStore);
  }

  /**
   * Load all nodes in the scene graph. Each leaf node is wrapped in a
   * per-node try/catch so a single failing node does not abort loading
   * sibling nodes — the user gets a partial scene plus a per-failure
   * log entry instead of an empty scene with no actionable signal.
   *
   * `LoaderError` thrown from `loadX` is dispatched on `kind`:
   * - Network → log warning + skip (transient, retry path will handle)
   * - Decode / Validation → log error + toast (real data problem)
   * - Unexpected → log error + toast (programmer bug; doesn't re-throw
   *   because we still want sibling nodes to render)
   */
  private async loadSceneNodes(
    node: SceneNode,
    parentThree: THREE.Object3D,
    parentLoc: zarr.Location<zarr.Readable>
  ): Promise<void> {
    if (node.type === 'points') {
      // loadPoints attaches its own placeholder to parentThree before
      // fetching data; no caller-side `if (points) add(points)` is needed.
      // The placeholder stays in the scene even on failure so retry can
      // populate it.
      await this.loadLeafNode(() => this.loadPoints(node, parentThree, parentLoc), node.path);
    } else if (node.type === 'lines') {
      await this.loadLeafNode(() => this.loadLines(node, parentThree, parentLoc), node.path);
    } else if (node.type === 'gsplats') {
      await this.loadLeafNode(() => this.loadGSplats(node, parentThree, parentLoc), node.path);
    } else if (node.children) {
      // Create group and recurse
      const group = new THREE.Group();
      group.name = node.path;

      // Apply transform if present
      if (node.attrs.transform) {
        this.nodeFactory.applyTransform(group, node.attrs.transform);
      }

      parentThree.add(group);

      // Load children
      for (const child of node.children) {
        const childLoc = parentLoc.resolve(child.path.slice(1));
        await this.loadSceneNodes(child, group, childLoc);
      }
    }
  }

  /**
   * Run a leaf-node loader, dispatching {@link LoaderError} by kind so
   * one bad node doesn't sink the whole scene. Implementation lives in
   * `scene-loader/build/load-leaf-error-dispatch.ts`.
   */
  private async loadLeafNode<T extends THREE.Object3D>(
    load: () => Promise<T | null>,
    path: string
  ): Promise<T | null> {
    return loadLeafNodeHelper(load, path);
  }

  /**
   * Load a single points node
   */
  private async loadPoints(
    node: SceneNode,
    parentThree: THREE.Object3D,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Mesh | null> {
    log.custom('📍', Modules.SCENE_LOADER, `Loading points: ${node.path}`);
    log.info(Modules.SCENE_LOADER, `  Has spatial index: ${node.hasSpatialIndex}`);
    log.info(Modules.SCENE_LOADER, `  Total points: ${node.attrs.n_points || 'unknown'}`);

    // Create appropriate loader
    const loader = this.createLoader(node, loc);

    // Store loader for updates (route through the registry's
    // register* methods rather than mutating its internal map).
    this.registry.registerPointsLoader(node.path, loader);

    // Construct + attach an empty placeholder before fetching data, so an
    // initial-load failure leaves a recoverable scene state.
    // commit-points-geometry finds the placeholder by name and populates
    // it once data arrives (initial fetch or future retry/update); the
    // 0-points → N-points transition naturally takes the "different size"
    // branch in commitPointsGeometry. retryFailedLoader() reads the
    // placeholder's `userData.attrs` to derive the retry view state.
    const attrs = this.applyEffectiveAttrs(node) as unknown as PointsMetadata;
    const placeholder = this.nodeFactory.createEmptyPointsNode(node.path, attrs, loader);
    parentThree.add(placeholder);

    try {
      // Load points data
      log.info(Modules.SCENE_LOADER, 'Initial ViewState for loading:');
      log.info(Modules.SCENE_LOADER, `  displayDims: [${this.viewState.displayDims.join(', ')}]`);
      log.info(
        Modules.SCENE_LOADER,
        `  slicePosition: [${this.viewState.slicePosition.join(', ')}]`
      );
      log.info(Modules.SCENE_LOADER, `  tolerance: [${this.viewState.tolerance.join(', ')}]`);

      // Route initial load through deriveNodeViewState (same helper as
      // the main update path and retry) so initial / update / retry can
      // never silently load different query regions. Initial load doesn't
      // apply the full-extend skip — we still want to construct the THREE
      // node so future slice changes can populate it; the skip return only
      // happens on update/retry where there's an existing node to leave
      // alone.
      const derived = this.deriveNodeViewState(node.path, node.attrs, {
        applyPartialExtendTolerance: true,
      });
      let pointsViewState: ViewState;
      if (derived.skip) {
        // Full-extend on initial load: behave as if extend_to_all
        // weren't set (load with the base view state) so the empty
        // node still gets constructed.
        pointsViewState = this.viewState;
      } else {
        pointsViewState = derived.viewState;
      }

      const data = await loader.loadPoints(pointsViewState);

      // Log if no initial points are visible (this is normal for nD slicing)
      if (data.pointCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible points for ${node.path} - object created for future updates`
        );
      }

      // Commit data into the placeholder via the same path future
      // updateView() / retry calls use. Unifies initial-load and update
      // through one geometry-commit code path.
      this.updatePointsGeometry(node.path, data);

      log.success(Modules.SCENE_LOADER, `Loaded ${data.pointCount} points for ${node.path}`);

      return placeholder;
    } catch (error) {
      // Record the failure so `retryFailedLoader(path)` can target this
      // node. The placeholder stays attached to the scene (added before
      // this try/catch), so retry can populate it.
      this.registry.recordFailure(node.path, error as Error);
      throw new LoaderError(classifyLoaderError(error), node.path, error);
    }
  }

  /**
   * Load a single lines node
   */
  private async loadLines(
    node: SceneNode,
    parentThree: THREE.Object3D,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Mesh | null> {
    log.custom('📐', Modules.SCENE_LOADER, `Loading lines: ${node.path}`);

    const attrs = node.attrs as unknown as LinesMetadata;
    log.info(Modules.SCENE_LOADER, `  Segments: ${attrs.n_segments || 'unknown'}`);
    log.info(Modules.SCENE_LOADER, `  Vertices: ${attrs.n_vertices || 'unknown'}`);

    // Create lines loader
    const loader = this.createLinesLoader(node, loc);

    // Store loader for updates (route through registry).
    this.registry.registerLinesLoader(node.path, loader);

    // Construct + attach empty placeholder before fetching.
    // processLinesData / commitLinesGeometry look up the mesh by name
    // and populate it on success; on failure the placeholder remains
    // for retry to target. Same path is used by every future update.
    const placeholder = this.nodeFactory.createEmptyLinesNode(
      node.path,
      this.applyEffectiveAttrs(node),
      attrs,
      loader
    );
    parentThree.add(placeholder);

    try {
      // Lines path does not apply the partial-extend tolerance override
      // during the data fetch (only during clipping below), so
      // applyPartialExtendTolerance=false. This call still validates
      // extend_to_all dim names and applies the inverse nd_transform.
      const derivedLines = this.deriveNodeViewState(node.path, attrs, {
        applyPartialExtendTolerance: false,
      });
      const linesViewState: LinesViewState = derivedLines.skip
        ? this.viewState
        : derivedLines.viewState;

      const data = await loader.loadLines(linesViewState);

      if (data.segmentCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible segments for ${node.path} - object created for future updates`
        );
      }

      // Project + commit through the same helpers used by every update
      // and retry. processLinesData reads the placeholder's userData
      // (extend_to_all etc.) and finds the mesh by name; commit step
      // writes into the existing geometry.
      const staged = await this.processLinesData(node.path, data, linesViewState);
      if (staged) this.commitLinesGeometry(staged);

      log.success(Modules.SCENE_LOADER, `Loaded ${data.segmentCount} segments for ${node.path}`);

      return placeholder;
    } catch (error) {
      // See loadPoints catch — same record-failure-then-throw shape.
      this.registry.recordFailure(node.path, error as Error);
      throw new LoaderError(classifyLoaderError(error), node.path, error);
    }
  }

  /**
   * Create a lines loader for a node
   */
  /** Build the per-call dependency snapshot for the loader factory. */
  private factoryDeps(): LoaderFactoryDeps {
    return {
      zarrStore: this._zarrStore!,
      arrayRefRegistry: this.arrayRefRegistry,
      profiler: this.profiler,
      l0Cache: this.l0Cache,
      cachingStore: this.cachingStore,
    };
  }

  private createLinesLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): LinesDataLoader {
    const loader = createLinesLoaderHelper(node, loc, this.factoryDeps());
    this.connectLoaderToMonitor(node.path, loader);
    return loader;
  }

  /**
   * Load a single gsplats node
   */
  private async loadGSplats(
    node: SceneNode,
    parentThree: THREE.Object3D,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Mesh | null> {
    const attrs = node.attrs as unknown as GSplatsMetadata;
    // v2.0 surfaces the default substitutive level's additive sub-LOD count
    // as `n_additive_sublods_default` on the splats group attrs. Progressive
    // loading kicks in when that count > 1.
    const nAdditive = attrs.n_additive_sublods_default ?? 0;
    const defaultSub = attrs.default_substitutive ?? 0;
    log.custom('🔮', Modules.SCENE_LOADER, `Loading gsplats: ${node.path}`);
    log.info(
      Modules.SCENE_LOADER,
      `  Splats: ${(nAdditive > 1 ? (attrs.n_splats_total ?? attrs.n_splats) : attrs.n_splats)?.toLocaleString() || 'unknown'}`
    );
    log.info(Modules.SCENE_LOADER, `  Dimensions: ${attrs.ndim || 'unknown'}D`);
    if ((attrs.n_substitutive ?? 1) > 1) {
      log.info(
        Modules.SCENE_LOADER,
        `  Substitutive levels: ${attrs.n_substitutive} (rendering default level ${defaultSub})`
      );
    }
    if (nAdditive > 1) {
      log.info(
        Modules.SCENE_LOADER,
        `  Additive sub-LODs: ${nAdditive} (progressive loading enabled)`
      );
    }

    // Create gsplats loader — progressive for multi-additive, standard otherwise
    const loader =
      nAdditive > 1
        ? await this.createProgressiveGSplatsLoader(node, loc, nAdditive, defaultSub)
        : this.createGSplatsLoader(node, loc);

    // Store loader for updates (route through registry).
    this.registry.registerGSplatsLoader(node.path, loader);

    // Empty placeholder + same-flow commit. See loadPoints/loadLines
    // for the rationale.
    const placeholder = this.nodeFactory.createEmptyGSplatsNode(
      node.path,
      this.applyEffectiveAttrs(node),
      attrs,
      loader
    );
    parentThree.add(placeholder);

    try {
      // GSplats path mirrors Points: applyPartialExtendTolerance=true so
      // tolerance overrides + nd_transform inversion both happen up front.
      const derivedGSplats = this.deriveNodeViewState(node.path, node.attrs, {
        applyPartialExtendTolerance: true,
      });
      const gsplatsViewState: GSplatsViewState = derivedGSplats.skip
        ? {
            displayDims: this.viewState.displayDims,
            slicePosition: this.viewState.slicePosition,
            tolerance: this.viewState.tolerance,
            dimensions: this.viewState.dimensions,
          }
        : derivedGSplats.viewState;

      // Load gsplats data
      const data = await loader.loadGSplats(gsplatsViewState);

      if (data.splatCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible gsplats for ${node.path} - object created for future updates`
        );
      }

      // Process + commit through the same helpers used by every update
      // and retry. Helpers find the placeholder by name and read its
      // userData for truncate/attrs.
      const staged = await this.processGSplatsData(node.path, data, gsplatsViewState);
      if (staged) this.commitGSplatsGeometry(staged);

      log.success(
        Modules.SCENE_LOADER,
        `Loaded ${data.splatCount.toLocaleString()} gsplats for ${node.path}`
      );

      return placeholder;
    } catch (error) {
      // See loadPoints catch.
      this.registry.recordFailure(node.path, error as Error);
      throw new LoaderError(classifyLoaderError(error), node.path, error);
    }
  }

  /** Create a single-LOD gsplats loader for a node. */
  private createGSplatsLoader(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): GSplatsDataLoader {
    const loader = createGSplatsLoaderHelper(node, loc, this.factoryDeps());
    this.connectLoaderToMonitor(node.path, loader);
    return loader;
  }

  /**
   * Create a progressive gsplats loader for a multi-LOD node. The
   * parent's effective rendering attrs are composed up the scene-graph
   * ancestry here (not in the helper) so the LOD synthetic nodes see
   * ancestor opacity/intensity/etc.
   */
  private async createProgressiveGSplatsLoader(
    node: SceneNode,
    _loc: zarr.Location<zarr.Readable>,
    nAdditive: number,
    defaultSub: number
  ): Promise<GSplatsDataLoader> {
    const loader = await createProgressiveGSplatsLoaderHelper(
      node,
      nAdditive,
      defaultSub,
      this.applyEffectiveAttrs(node),
      this.factoryDeps()
    );
    this.connectLoaderToMonitor(node.path, loader);
    return loader;
  }

  /**
   * Create the points spatial index loader for a node and connect it to
   * the data monitor (the monitor wiring stays here because it touches
   * SceneManager-only state — the factory only constructs the loader).
   */
  private createLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): DataLoader {
    const loader = createPointsLoaderHelper(node, loc, this.factoryDeps());
    this.connectLoaderToMonitor(node.path, loader);
    return loader;
  }

  /**
   * Connect a loader to the data-loading monitor when the monitor is active
   * and the loader implements the {@link LoaderMonitor} surface (lines and
   * gsplats expose the surface as optional methods, points always defines
   * them). Same wiring is used for all three geometry types.
   */
  private connectLoaderToMonitor(
    path: string,
    loader: DataLoader | LinesDataLoader | GSplatsDataLoader
  ): void {
    const monitor = this.monitor;
    if (!monitor) return;

    const candidate = loader as Partial<LoaderMonitor>;
    if (
      typeof candidate.addEventListener === 'function' &&
      typeof candidate.removeEventListener === 'function' &&
      typeof candidate.getMetrics === 'function' &&
      typeof candidate.getActiveQueries === 'function'
    ) {
      monitor.connectLoader(path, candidate as LoaderMonitor);
    }
  }

  /**
   * Update geometry for a specific points node.
   *
   * Implementation lives in `scene-loader/geometry-commit-handler.ts`.
   */
  private updatePointsGeometry(
    path: string,
    data: LoadedPointsData,
    session?: UpdateSession
  ): void {
    commitPointsGeometryHelper(
      path,
      data,
      this.rootGroup,
      this._gpuBufferPool,
      this.nodeFactory,
      session
    );
  }

  /**
   * Initialize scene dimensions from metadata. Implementation lives in
   * `scene-loader/nodes/initialize-scene-dimensions.ts`; null return
   * means validation failed and the existing viewState stays.
   */
  private initializeSceneDimensions(sceneDims: unknown): void {
    const next = initializeSceneDimensionsHelper(sceneDims);
    if (next) this.viewState = next;
  }

  /**
   * Normalize URL for zarr store access. Guards `window` so this works
   * in non-DOM contexts (tests, embed-in-Worker scenarios). Absolute URLs
   * ignore the origin entirely; the fallback only matters for relative
   * paths.
   */
  private normalizeURL(url: string): string {
    const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
    return normalizeURL(url, origin);
  }

  /**
   * Show the monitor UI
   */
  showMonitor(): void {
    this.monitor?.show();
  }

  /**
   * Hide the monitor UI
   */
  hideMonitor(): void {
    this.monitor?.hide();
  }

  /**
   * Toggle the monitor UI
   */
  toggleMonitor(): void {
    this.monitor?.toggle();
  }

  /**
   * Get information about failed loaders
   * @returns Map of loader paths to error information
   */
  getFailedLoaders(): ReadonlyMap<string, { error: Error; timestamp: number; retryCount: number }> {
    return this.failedLoaders;
  }

  /**
   * Check if there are any failed loaders
   */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
  }

  /**
   * Clear failed loader tracking
   * Useful for retry operations or after user acknowledges errors
   */
  clearFailures(): void {
    this.registry.clearAllFailures();
  }

  /**
   * Retry loading a specific failed loader.
   *
   * Re-triggers the update for a failed loader using the current view
   * state. Useful for recovering from transient network errors or after
   * connectivity is restored.
   *
   * @param path - The path of the failed loader to retry
   * @returns Promise resolving to true if retry succeeded, false if failed or not found
   *
   * @example
   * ```typescript
   * // Retry a specific loader after network recovery
   * const success = await sceneLoader.retryFailedLoader('/points/cloud1');
   * if (success) {
   *   console.log('Loader recovered successfully');
   * }
   * ```
   */
  async retryFailedLoader(path: string): Promise<boolean> {
    // Check if this path is actually in failed loaders
    if (!this.failedLoaders.has(path)) {
      log.warning(Modules.SCENE_LOADER, `Path "${path}" is not in failed loaders list`);
      return false;
    }

    // Serialize against the main update path. retry calls
    // `loader.updateView(...)` directly, which would race with
    // updateView()'s own per-loader call for the same path — concurrent
    // zarr fetches and concurrent commits to the same THREE object
    // produce inconsistent state. Both halves are required: refusing
    // when an update is already active AND taking the lock so an
    // updateView starting AFTER retry begins can't race.
    if (this._updateInProgress) {
      log.info(
        Modules.SCENE_LOADER,
        `Retry of ${path} deferred — main update in progress; try again after the update settles`
      );
      return false;
    }

    this._updateInProgress = true;
    try {
      return await this._retryFailedLoaderUnlocked(path);
    } finally {
      this._updateInProgress = false;
      this.viewStateQueue.drain((state) => this.updateView(state));
    }
  }

  /**
   * Internal retry body without the `_updateInProgress` lock dance.
   * Used by both `retryFailedLoader` (which takes the lock once) and
   * `retryAllFailedLoaders` (which takes the lock once and runs
   * multiple retries inside it).
   */
  private async _retryFailedLoaderUnlocked(path: string): Promise<boolean> {
    if (!this.failedLoaders.has(path)) return false;

    log.info(Modules.SCENE_LOADER, `Retrying failed loader: ${path}`);

    // Determine which loader type this path belongs to
    const pointsLoader = this.loaders.get(path);
    const linesLoader = this.linesLoaders.get(path);
    const gsplatsLoader = this.gsplatLoaders.get(path);

    try {
      // Look up the per-node attrs so retry applies the same
      // extend_to_all / nd_transform adjustments as the main update path.
      // Passing a raw view state here silently renders an incorrect query
      // region for transformed or extended nodes.
      const obj = this.rootGroup?.getObjectByName(path) as
        | THREE.Object3D
        | THREE.Mesh
        | THREE.Points
        | undefined;
      const attrs = obj?.userData?.attrs as { extend_to_all?: string[] } | undefined;

      // Defensive guard — only clear `failedLoaders` if the named object
      // still exists in the scene. The placeholder model should make
      // commit always succeed when retry runs in normal conditions, but
      // a scene reload or programmatic node removal between failure and
      // retry could leave us fetching data that has nowhere to land.
      // Without this guard, retry would falsely report success ("data
      // fetched + commit silently no-op'd") and clear the failure,
      // hiding the broken state from `hasFailures()`.
      const verifyAndClear = (kind: string): boolean => {
        if (!this.rootGroup?.getObjectByName(path)) {
          log.warning(
            Modules.SCENE_LOADER,
            `Retry of ${path} fetched data but no scene object exists; not clearing failure`
          );
          return false;
        }
        this.failedLoaders.delete(path);
        log.success(Modules.SCENE_LOADER, `Successfully retried ${kind} loader: ${path}`);
        return true;
      };

      if (pointsLoader) {
        const derived = this.deriveNodeViewState(path, attrs, {
          applyPartialExtendTolerance: true,
        });
        // Mirror loadPoints() initial-load fallback. When derived.skip
        // is true (extend_to_all fully covers), the placeholder still
        // needs data committed — skipping the load and clearing
        // failedLoaders would falsely report success against an empty
        // placeholder.
        const pointsViewState = derived.skip ? this.viewState : derived.viewState;
        const points = await pointsLoader.updateView(pointsViewState);
        if (points) this.updatePointsGeometry(path, points);
        return verifyAndClear('points');
      } else if (linesLoader) {
        const derived = this.deriveNodeViewState(path, attrs, {
          applyPartialExtendTolerance: false,
        });
        // See Points branch.
        const linesViewState: LinesViewState = derived.skip ? this.viewState : derived.viewState;
        const data = await linesLoader.updateView(linesViewState);
        if (data) {
          const staged = await this.processLinesData(path, data, linesViewState);
          if (staged) this.commitLinesGeometry(staged);
        }
        return verifyAndClear('lines');
      } else if (gsplatsLoader) {
        const derived = this.deriveNodeViewState(path, attrs, {
          applyPartialExtendTolerance: true,
        });
        // Mirror loadGSplats() initial-load fallback shape (explicit
        // object spread to match LinesViewState/GSplatsViewState).
        const gsplatsViewState: GSplatsViewState = derived.skip
          ? {
              displayDims: this.viewState.displayDims,
              slicePosition: this.viewState.slicePosition,
              tolerance: this.viewState.tolerance,
              dimensions: this.viewState.dimensions,
            }
          : derived.viewState;
        const data = await gsplatsLoader.updateView(gsplatsViewState);
        if (data) {
          const staged = await this.processGSplatsData(path, data, gsplatsViewState);
          if (staged) this.commitGSplatsGeometry(staged);
        }
        return verifyAndClear('gsplats');
      } else {
        // Loader not found - it may have been disposed
        log.warning(Modules.SCENE_LOADER, `No loader found for path: ${path}`);
        this.failedLoaders.delete(path); // Clean up stale entry
        return false;
      }
    } catch (error) {
      // Update error tracking with new attempt
      const errorInfo = this.failedLoaders.get(path);
      const retryCount = errorInfo ? errorInfo.retryCount + 1 : 1;
      this.failedLoaders.set(path, {
        error: error as Error,
        timestamp: Date.now(),
        retryCount,
      });
      log.error(
        Modules.SCENE_LOADER,
        `Retry failed for ${path} (attempt ${retryCount}): ${(error as Error).message}`
      );
      return false;
    }
  }

  /**
   * Retry all failed loaders. Useful for batch recovery after network
   * connectivity is restored.
   *
   * @returns Promise resolving to an object with succeeded and failed path arrays
   *
   * @example
   * ```typescript
   * // Retry all failed loaders after network recovery
   * const result = await sceneLoader.retryAllFailedLoaders();
   * console.log(`Recovered: ${result.succeeded.length}, Still failing: ${result.failed.length}`);
   * ```
   */
  async retryAllFailedLoaders(): Promise<{ succeeded: string[]; failed: string[] }> {
    const failedPaths = Array.from(this.failedLoaders.keys());

    if (failedPaths.length === 0) {
      log.info(Modules.SCENE_LOADER, 'No failed loaders to retry');
      return { succeeded: [], failed: [] };
    }

    // Same serialization as retryFailedLoader: take the lock once around
    // the parallel batch and call the unlocked retry helper for each
    // path, so siblings in the same batch don't trigger the lock-refusal
    // branch.
    if (this._updateInProgress) {
      log.info(
        Modules.SCENE_LOADER,
        'Retry-all deferred — main update in progress; try again after the update settles'
      );
      return { succeeded: [], failed: failedPaths };
    }

    log.info(Modules.SCENE_LOADER, `Retrying ${failedPaths.length} failed loader(s)`);

    this._updateInProgress = true;
    try {
      const succeeded: string[] = [];
      const failed: string[] = [];

      const results = await Promise.all(
        failedPaths.map(async (path) => {
          const success = await this._retryFailedLoaderUnlocked(path);
          return { path, success };
        })
      );

      for (const { path, success } of results) {
        if (success) {
          succeeded.push(path);
        } else {
          failed.push(path);
        }
      }

      log.info(
        Modules.SCENE_LOADER,
        `Retry complete: ${succeeded.length} succeeded, ${failed.length} still failing`
      );

      return { succeeded, failed };
    } finally {
      this._updateInProgress = false;
      this.viewStateQueue.drain((state) => this.updateView(state));
    }
  }

  // _drainPendingViewState + _dispatchPerLoaderPrefetch moved to
  // ./scene-loader/view-state-queue (step 5 of the god-object refactor).

  /**
   * Dispose of all resources.
   *
   * Async because the caching store dispose path drains the prefetcher,
   * cancels in-flight validation, and flushes OPFS metadata — work that
   * a dataset switch should wait for before constructing the next
   * loader. Existing sync callers (the `beforeunload` path,
   * `SceneLoaderManager.destroyLoader/destroyAll`) still work; the
   * returned promise just unwinds in the background. Callers that need
   * deterministic teardown should await this method or use
   * `SceneLoaderManager.destroyLoaderAsync` (added in a follow-up
   * commit).
   *
   * B.7 — worker pool policy: Web Workers used for projection/decoding
   * live in a MODULE-LEVEL singleton (`workers/worker-pool.ts:
   * getWorkerPool`), not per-SceneLoader. Dataset switches deliberately
   * do NOT terminate workers — the pool is bounded, and tearing it down
   * per switch would force a fresh worker spin-up on the next load
   * (10s of ms of WASM re-init on each cycle). Workers are terminated
   * only at app shutdown via `disposeWorkerPool()` in `core/app.ts`,
   * which is the right scope for that lifecycle.
   */
  async dispose(): Promise<void> {
    // Abort the dataset-scoped signal first so any in-flight worker
    // `runWithTimeout` callers settle immediately instead of waiting
    // for their tasks to complete (WASM tasks themselves keep running
    // but their results are discarded). Clear the pool's reference
    // afterwards so future workers don't get an already-aborted
    // signal from this disposed loader.
    if (this._datasetAbortController) {
      this._datasetAbortController.abort();
      this._datasetAbortController = null;
    }
    getWorkerPool().setAbortSignal(undefined);

    // Dispose all geometry loaders via registry
    this.registry.disposeAll();

    // dispose GPU buffer pool. Without this, the pool retains
    // active+pooled InstancedBufferGeometry references after a dataset
    // switch — at million-element scale this can leak hundreds of MB
    // of GPU memory until the page is refreshed. The pool's internal
    // dispose() is idempotent.
    if (this._gpuBufferPool) {
      try {
        this._gpuBufferPool.dispose();
      } catch (error) {
        log.warning(Modules.SCENE_LOADER, 'GPU buffer pool disposal failed', error);
      }
      this._gpuBufferPool = null;
    }

    // Dispose caching store (flushes L2 metadata, clears L1).
    // Awaited so a dataset switch sees the previous L2 fully drained
    // before the next caching store is constructed.
    if (this.cachingStore) {
      try {
        await this.cachingStore.dispose();
      } catch (error) {
        log.warning(Modules.SCENE_LOADER, 'Caching store disposal failed', error);
      }
      this.cachingStore = null;
    }

    // Clear L0 decompressed chunk cache
    if (this.l0Cache) {
      const stats = this.l0Cache.getStats();
      log.info(
        Modules.SCENE_LOADER,
        `L0 cache stats at dispose: ${stats.count} chunks, ${(stats.size / 1024 / 1024).toFixed(1)}MB, ` +
          `hit rate: ${(stats.hitRate * 100).toFixed(1)}%`
      );
      this.l0Cache.clear();
      this.l0Cache = null;
    }

    this._zarrStore = null;
    this.rootGroup = null;
    this._sceneGraph = null;

    // S6: clear per-loader prefetch predictor state on dispose so a
    // reused SceneLoader doesn't extrapolate from a prior dataset.
    this.viewStateQueue.clearPrev();

    // Dispose dataset-scoped custom colormap LUTs. The custom-LUT cache
    // is keyed by content hash and shared across all scenes, but entries
    // from an unloaded dataset have no value and would accumulate in a
    // long-lived app that swaps many unique LUTs. Built-ins survive
    // because they're shared with all scenes and cheap to keep.
    try {
      disposeCustomColormapTextures();
    } catch (error) {
      log.warning(Modules.SCENE_LOADER, 'Custom colormap disposal failed', error);
    }

    // Tell the monitor to drop its scene-loader-bound closures (cache
    // stats, L0 cache, GPU buffer pool, accumulators, profiler) before
    // we release our reference. Without this, the monitor outlives the
    // loader with closures that capture our nulled-out fields and NPE
    // on the next stats poll. The monitor's lifecycle itself is owned
    // by core/app.ts via DataMonitorManager.
    this.monitor?.disconnectAllLoaders();
    this.monitor = null;
  }
}
