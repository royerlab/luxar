/**
 * Unified scene loader that orchestrates the loading of complete Luxar scenes.
 *
 * This loader handles the entire scene graph, using spatial index-based
 * loading for all points nodes and managing the THREE.js scene construction.
 */

import * as zarr from 'zarrita';
import type { Readable } from '@zarrita/storage';
import * as THREE from 'three';
import { normalizeURL } from './scene-loader/url-normalization';
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
  projectLinesTo3DUsingWorker as projectLinesTo3DUsingWorkerHelper,
  type StagedLinesCommit,
} from './scene-loader/data-processor-lines';
import {
  processGSplatsData as processGSplatsDataHelper,
  commitGSplatsGeometry as commitGSplatsGeometryHelper,
  projectGSplatsTo3DUsingWorker as projectGSplatsTo3DUsingWorkerHelper,
  type StagedGSplatsCommit,
} from './scene-loader/data-processor-gsplats';
import {
  createPointsLoader as createPointsLoaderHelper,
  createLinesLoader as createLinesLoaderHelper,
  createGSplatsLoader as createGSplatsLoaderHelper,
  createProgressiveGSplatsLoader as createProgressiveGSplatsLoaderHelper,
  type LoaderFactoryDeps,
} from './scene-loader/loader-factory';
import { commitPointsGeometry as commitPointsGeometryHelper } from './scene-loader/geometry-commit-handler';

export type { StagedLinesCommit } from './scene-loader/data-processor-lines';
export type { StagedGSplatsCommit } from './scene-loader/data-processor-gsplats';
import { buildInstanceBuffers } from './lines/lines-spatial-index-loader';
import {
  DataLoader,
  ViewState,
  SceneNode,
  LoaderConfig,
  LoadedPointsData,
} from './data-loader-types';
import type { SceneGraphNode } from '../types/data-monitor-types';
import { ZarrSceneAttrs, ZarrNodeAttrs, hasContentsMethod } from '../types/zarr';
import { DataMonitorManager } from '../ui/monitors/data-monitor-manager';
import { ArrayRefRegistry } from './utils/array-decoder';
import {
  ViewStateManager,
  type SceneDimensions,
  type DimensionMetadata,
} from './view-state-manager';
import { log, Modules, LogEmoji } from '../utils/log';
import { config as appConfig } from '../config';
import { MultiLevelCachingStore, ChunkPrefetcher, DecompressedChunkCache } from '../cache';
import type { PointsMetadata } from '../types/points';
import type {
  LinesMetadata,
  LinesDataLoader,
  LoadedLinesData,
  ProcessedLinesData,
} from '../types/lines';
import { isLinesUserData } from '../types/lines';
import type {
  GSplatsMetadata,
  GSplatsDataLoader,
  GSplatsUserData,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../types/gsplats';
import { processGSplats } from './gsplats/gsplats-processor';
import { packCholeskyForShader } from '../rendering/gsplat-geometry';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { invertNdTransformForQuery, computeWorldNdTransform } from './transforms/nd-transform';
import { NodeFactory } from '../rendering/node-factory';
import { UpdateProfiler, type UpdateSession } from '../profiling/update-profiler';
import {
  getAggregatedPointsAccumulatorStats,
  getAggregatedLinesAccumulatorStats,
  getAggregatedGSplatsAccumulatorStats,
} from './utils/stats-aggregator';
import { LoaderRegistry } from './loaders/loader-registry';
import { computeTolerance } from './utils/tolerance-computer';
import { loadOverlayConfigs } from './loaders/overlay-loader';
import { notifier } from '../utils/notifier';

/** Check if an object has any own properties (avoids Object.keys() allocation). */
function hasOwnProperties(obj: Record<string, unknown>): boolean {
  for (const k in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) return true;
  }
  return false;
}

/**
 * Get or compute an extended tolerance array for extend_to_all dimensions.
 * Nodes with the same extendDims share a single cached array.
 */
function getOrComputeExtendedTolerance(
  baseTolerance: readonly number[],
  extendDims: string[],
  dimensionMetadata: Array<{ name?: string }>,
  cache: Map<string, number[]>
): number[] {
  validateExtendDims(extendDims, dimensionMetadata);
  const key = extendDims.slice().sort().join(',');
  let cached = cache.get(key);
  if (cached) return cached;
  cached = [...baseTolerance];
  for (const dimName of extendDims) {
    const dimIndex = dimensionMetadata.findIndex((d) => d.name === dimName);
    if (dimIndex >= 0 && dimIndex < cached.length) {
      cached[dimIndex] = 1e10;
    }
  }
  cache.set(key, cached);
  return cached;
}

function validateExtendDims(
  extendDims: string[],
  dimensionMetadata: Array<{ name?: string }>
): void {
  const validNames = new Set(
    dimensionMetadata.map((dim) => dim.name).filter((name): name is string => !!name)
  );
  const invalid = extendDims.filter((dimName) => !validNames.has(dimName));
  if (invalid.length > 0) {
    throw new Error(
      `Invalid extend_to_all dimension(s): ${invalid.join(', ')}. ` +
        `Valid dimensions: ${Array.from(validNames).join(', ')}`
    );
  }
}

function isSceneDimensions(value: unknown): value is SceneDimensions {
  if (!value || typeof value !== 'object') return false;
  return Array.isArray((value as { dimensions?: unknown }).dimensions);
}

// ============================================================================
// Staged commit types for atomic geometry updates
// ============================================================================
// During dimension animation, all nodes must update in the same render frame
// to prevent flickering. These types hold processed data between the async
// load+process phase and the synchronous commit phase.

/** Staged points data ready for GPU commit */
interface StagedPointsCommit {
  path: string;
  data: LoadedPointsData;
}

/**
 * Main scene loader that handles the complete loading pipeline.
 *
 * Features:
 * - Spatial index-based loading for efficient nD queries
 * - Hierarchical scene graph construction
 * - Transform and rendering attribute inheritance
 * - Dimension metadata management
 * - Memory-efficient loading with proper caching
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
  private monitorId: string | null = null;
  private arrayRefRegistry: ArrayRefRegistry;

  // Phase 4: GPU buffer pool for geometry reuse ✅ INTEGRATED
  // Integrated into updatePointsGeometry/updateLinesGeometry/updateGSplatsGeometry
  // Enabled via config.dataLoading.performance.useGPUBufferPool
  private _gpuBufferPool: GPUBufferPool | null = null;
  public readonly nodeFactory = new NodeFactory();

  // Update profiler for timing scene updates (optional, provided by SceneLoaderManager)
  private profiler: UpdateProfiler | null = null;

  // Serialized update queue: prevents concurrent updateView calls from corrupting shared buffers
  // When a new update arrives while one is in progress, we store the latest and process it after
  private _updateInProgress = false;
  private _pendingViewState: Partial<ViewState> | null = null;
  private _updateVersion = 0; // For logging/debugging
  private _sceneGraph: SceneNode | null = null;

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

  /**
   * Snapshot of all cache levels (L0, L1, L2) in the form historically
   * exposed by `__luxarDebug.cache.getStats()`.
   */
  getCacheStats(): CacheStatsSnapshot {
    return getCacheStatsHelper(this.l0Cache, this.cachingStore);
  }

  /** List datasets currently held by the L1/L2 caching store. */
  async listCachedDatasets(): Promise<
    Awaited<ReturnType<MultiLevelCachingStore['listDatasets']>>
  > {
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

  constructor(config: LoaderConfig = {}, id?: string, profiler?: UpdateProfiler) {
    this.profiler = profiler ?? null;
    this.config = config;
    this.viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [],
      tolerance: [],
    };
    this.arrayRefRegistry = new ArrayRefRegistry();

    // Phase 4: Initialize GPU buffer pool if enabled
    // Fully integrated into updatePointsGeometry/updateLinesGeometry/updateGSplatsGeometry
    // Note: Requires Float32Array data; falls back to standard path for Uint8/Uint16
    if (appConfig.dataLoading.performance.useGPUBufferPool) {
      this._gpuBufferPool = new GPUBufferPool(
        appConfig.dataLoading.performance.gpuPoolMaxSize,
        appConfig.dataLoading.performance.gpuPoolEvictionFrames,
        appConfig.dataLoading.performance.gpuPoolEvictBatchSize
      );
      log.info(
        Modules.GPU_BUFFER_POOL,
        `GPU buffer pool enabled (max size: ${appConfig.dataLoading.performance.gpuPoolMaxSize}, ` +
          `eviction: ${appConfig.dataLoading.performance.gpuPoolEvictionFrames} frames, ` +
          `batch cap: ${appConfig.dataLoading.performance.gpuPoolEvictBatchSize})`
      );
    }

    // Use the DataMonitorManager to get or create a monitor
    if (typeof document !== 'undefined' && config.enableMonitor !== false) {
      const monitorManager = DataMonitorManager.getInstance();
      const monitorId = id ? `${id}-monitor` : 'default';

      // Only create if it doesn't exist
      if (!monitorManager.hasMonitor(monitorId)) {
        monitorManager.createMonitor(monitorId, document.body);
      }
      this.monitorId = monitorId;
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
    if (this.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
      if (monitor) {
        monitor.disconnectAllLoaders();
      }
    }

    // Dispose of any existing loaders
    if (this.loaders.size > 0) {
      this.dispose();
    }

    // Cache flags (`?no-cache`, `?cache-debug`, `?clear-cache`, `?no-prefetch`,
    // `?prefetch-debug`) are routed via LoaderConfig from main.ts; SceneLoader
    // does not consult window.location directly.
    const noCache = this.config.noCache ?? false;
    const cacheDebug = this.config.cacheDebug ?? false;
    const clearCache = this.config.clearCache ?? false;
    const noPrefetch = this.config.noPrefetch ?? false;
    const prefetchDebug = this.config.prefetchDebug ?? false;

    if (appConfig.cache.l0Enabled && !noCache) {
      this.l0Cache = new DecompressedChunkCache({
        maxSize: appConfig.cache.l0MaxSizeMB * 1024 * 1024,
        debug: cacheDebug || appConfig.cache.debug,
      });

      // Clear L0 if ?clear-cache is set (matches L1/L2 behavior)
      if (clearCache) {
        this.l0Cache.clear();
        log.info(Modules.SCENE_LOADER, 'L0 cache cleared via ?clear-cache URL parameter');
      }

      log.info(
        Modules.SCENE_LOADER,
        `L0 decompressed chunk cache enabled (max size: ${appConfig.cache.l0MaxSizeMB}MB)`
      );
    } else {
      this.l0Cache = null;
      if (noCache) {
        log.info(Modules.SCENE_LOADER, 'L0 cache disabled via ?no-cache URL parameter');
      }
    }

    // Open zarr store with caching
    let rawStore: Readable;
    if (appConfig.cache.enabled && !noCache) {
      const cachingStore = new MultiLevelCachingStore(this.normalizeURL(url), {
        l1MaxSize: appConfig.cache.l1MaxSizeMB * 1024 * 1024,
        l2MaxSize: appConfig.cache.l2MaxSizeMB * 1024 * 1024,
        debug: cacheDebug || appConfig.cache.debug,
        noCache,
        clearCache,
      });
      await cachingStore.init();

      // Attach prefetcher to enable transparent adjacent chunk prefetching
      const prefetcher = new ChunkPrefetcher(cachingStore, {
        maxConcurrent: 4,
        enabled: !noPrefetch,
        debug: prefetchDebug,
      });
      cachingStore.setPrefetcher(prefetcher);

      // Register L0 invalidation: when L1/L2 are cleared (e.g., content hash change),
      // also clear the L0 decompressed chunk cache to prevent stale data.
      if (this.l0Cache) {
        const l0 = this.l0Cache;
        cachingStore.onInvalidate(() => {
          l0.clear();
          log.info(Modules.SCENE_LOADER, 'L0 cache cleared due to L1/L2 invalidation');
        });
      }

      rawStore = cachingStore;
      this.cachingStore = cachingStore;
    } else {
      rawStore = new zarr.FetchStore(this.normalizeURL(url));
    }
    this._zarrStore = (await zarr.tryWithConsolidated(rawStore)) as zarr.Readable;

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
      const ndim = this.viewState.dimensions?.metadata?.length ?? 0;
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

    // Force update the monitor UI after all loaders are connected
    // This ensures the UI shows the correct state even if no events have fired yet
    if (this.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
      if (monitor) {
        // Connect cache stats provider for L1/L2 cache monitoring
        if (this.cachingStore) {
          monitor.setCacheStatsProvider(this.cachingStore);
        }

        // Connect L0 decompressed chunk cache provider for Cache tab
        if (this.l0Cache) {
          monitor.setL0CacheProvider({
            getStats: () => this.l0Cache!.getStats(),
            clear: () => this.l0Cache!.clear(),
          });
        }

        // Connect GPU buffer pool provider for Memory tab
        if (this._gpuBufferPool) {
          monitor.setGPUBufferPoolProvider(this._gpuBufferPool);
        }

        // Connect accumulator providers for Memory tab (aggregate stats across all loaders)
        monitor.setAccumulatorProvider('points', {
          getStats: () => getAggregatedPointsAccumulatorStats(this.loaders),
        });
        monitor.setAccumulatorProvider('lines', {
          getStats: () => getAggregatedLinesAccumulatorStats(this.linesLoaders),
        });
        monitor.setAccumulatorProvider('gsplats', {
          getStats: () => getAggregatedGSplatsAccumulatorStats(this.gsplatLoaders),
        });

        // Connect profiler for Performance tab timing display
        if (this.profiler) {
          monitor.setProfiler(this.profiler);
        }

        // Send scene graph to monitor for display
        const sceneGraphRoot = this.convertToSceneGraphNode(sceneGraph);
        monitor.setSceneGraph(sceneGraphRoot);

        // Update visible segments count (initial load)
        this.updateVisibleCountsInMonitor();

        monitor.forceUpdate();
      }
    }

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
      // Store the latest pending state (supersedes any previous pending state)
      this._pendingViewState = viewState;
      log.info(
        Modules.SCENE_LOADER,
        `Update queued (v${this._updateVersion + 1}) - another update in progress`
      );
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

      // Noop session for when no profiler is available
      const noopSession: UpdateSession = {
        begin: () => noopSession,
        end: () => {},
        setMetadata: () => {},
        markSkipped: () => {},
      };

      // ================================================================
      // Phase 1: Load + Process all nodes in parallel (async)
      // Each callback returns staged commit data WITHOUT mutating geometry.
      // This ensures all nodes are ready before any geometry changes.
      // ================================================================

      // Load points data (no processing needed — data is used directly)
      const pointsLoaders = Array.from(this.loaders.entries()).map(async ([path, loader]) => {
        const updateFn = async (session: UpdateSession): Promise<StagedPointsCommit | null> => {
          try {
            // Get points object to check extend_to_all attribute
            const pointsObj = this.rootGroup?.getObjectByName(path) as THREE.Points | undefined;
            const attrs = pointsObj?.userData?.attrs as { extend_to_all?: string[] } | undefined;
            const extendDims: string[] = attrs?.extend_to_all || [];

            // Check if we can skip this update (extend_to_all optimization)
            if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
              const dims = this.viewState.dimensions.metadata;
              validateExtendDims(extendDims, dims);
              const nonDisplayedDims = dims
                .filter(
                  (_: { name?: string }, idx: number) => !this.viewState.displayDims.includes(idx)
                )
                .map((d: { name?: string }) => d.name)
                .filter((name: string | undefined): name is string => !!name);

              const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
                extendDims.includes(dimName)
              );

              if (isFullyExtended) {
                log.info(
                  Modules.SCENE_LOADER,
                  `Skipping update for ${path} - all non-displayed dims are extended`
                );
                session.markSkipped('extend_to_all');
                return null;
              }
            }

            // Build viewState with tolerance override for extend_to_all dimensions
            let pointsViewState = this.viewState;
            if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
              const tolerance = getOrComputeExtendedTolerance(
                this.viewState.tolerance,
                extendDims,
                this.viewState.dimensions.metadata,
                extendedToleranceCache
              );
              pointsViewState = { ...this.viewState, tolerance };
            }

            // Apply nd_transform inverse: convert world query to local coordinates
            // Uses composed world nd_transform (inherits from parent groups)
            if (this._sceneGraph && pointsViewState.dimensions?.metadata) {
              const worldNdT = computeWorldNdTransform(this._sceneGraph, path);
              if (hasOwnProperties(worldNdT)) {
                const dimNames = pointsViewState.dimensions.metadata.map(
                  (d: { name?: string }) => d.name ?? ''
                );
                const inverted = invertNdTransformForQuery(
                  pointsViewState.slicePosition,
                  pointsViewState.tolerance,
                  worldNdT,
                  dimNames,
                  pointsViewState.displayDims
                );
                pointsViewState = {
                  ...pointsViewState,
                  slicePosition: inverted.slicePosition,
                  tolerance: inverted.tolerance,
                };
              }
            }

            const points = await loader.updateView(pointsViewState, session);
            if (points) {
              if (currentVersion <= 1) {
                log.info(
                  Modules.SCENE_LOADER,
                  `[GEOM] v${currentVersion} points ${path}: ${points.pointCount} visible`
                );
              }
              session.setMetadata({ points: points.metadata.loadedPoints });
              this.failedLoaders.delete(path);
              return { path, data: points };
            }
            this.failedLoaders.delete(path);
            return null;
          } catch (error) {
            const errorInfo = this.failedLoaders.get(path);
            const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
            this.failedLoaders.set(path, {
              error: error as Error,
              timestamp: Date.now(),
              retryCount,
            });
            log.error(
              Modules.SCENE_LOADER,
              `Failed to update ${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
            );
            return null;
          }
        };

        if (this.profiler) {
          return this.profiler.timeTopLevel(`Points (${path})`, updateFn);
        } else {
          return updateFn(noopSession);
        }
      });

      // Load + process lines data (includes async worker projection)
      const linesLoaders = Array.from(this.linesLoaders.entries()).map(async ([path, loader]) => {
        const updateFn = async (session: UpdateSession): Promise<StagedLinesCommit | null> => {
          try {
            const mesh = this.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
            const attrs = mesh?.userData?.attrs as { extend_to_all?: string[] } | undefined;
            const extendDims: string[] = attrs?.extend_to_all || [];

            // Check if we can skip this update (extend_to_all optimization)
            if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
              const dims = this.viewState.dimensions.metadata;
              validateExtendDims(extendDims, dims);
              const nonDisplayedDims = dims
                .filter(
                  (_: { name?: string }, idx: number) => !this.viewState.displayDims.includes(idx)
                )
                .map((d: { name?: string }) => d.name)
                .filter((name: string | undefined): name is string => !!name);

              const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
                extendDims.includes(dimName)
              );

              if (isFullyExtended) {
                log.info(
                  Modules.SCENE_LOADER,
                  `Skipping update for ${path} - all non-displayed dims are extended`
                );
                session.markSkipped('extend_to_all');
                return null;
              }
            }

            let linesViewState: {
              displayDims: readonly number[];
              slicePosition: readonly number[];
              tolerance: readonly number[];
              dimensions?: import('../types/dims').DimensionMetadata[];
            } = {
              displayDims: this.viewState.displayDims,
              slicePosition: this.viewState.slicePosition,
              tolerance: this.viewState.tolerance,
              dimensions: this.viewState.dimensions?.metadata,
            };

            // Apply nd_transform inverse for lines query
            // Uses composed world nd_transform (inherits from parent groups)
            if (this._sceneGraph && linesViewState.dimensions) {
              const worldNdT = computeWorldNdTransform(this._sceneGraph, path);
              if (hasOwnProperties(worldNdT)) {
                const dimNames = linesViewState.dimensions.map(
                  (d: { name?: string }) => d.name ?? ''
                );
                const inverted = invertNdTransformForQuery(
                  linesViewState.slicePosition,
                  linesViewState.tolerance,
                  worldNdT,
                  dimNames,
                  linesViewState.displayDims
                );
                linesViewState = { ...linesViewState, ...inverted };
              }
            }

            const data = await loader.updateView(linesViewState, session);
            if (data) {
              if (currentVersion <= 1) {
                log.info(
                  Modules.SCENE_LOADER,
                  `[GEOM] v${currentVersion} lines ${path}: ${data.segmentCount} loaded`
                );
              }
              const staged = await this.processLinesData(path, data, linesViewState, session);
              session.setMetadata({ segments: data.segments ? data.segments.length / 2 : 0 });
              this.failedLoaders.delete(path);
              return staged;
            }
            this.failedLoaders.delete(path);
            return null;
          } catch (error) {
            const errorInfo = this.failedLoaders.get(path);
            const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
            this.failedLoaders.set(path, {
              error: error as Error,
              timestamp: Date.now(),
              retryCount,
            });
            log.error(
              Modules.SCENE_LOADER,
              `Failed to update lines ${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
            );
            return null;
          }
        };

        if (this.profiler) {
          return this.profiler.timeTopLevel(`Lines (${path})`, updateFn);
        } else {
          return updateFn(noopSession);
        }
      });

      // Load + process gsplats data (includes async worker projection + Cholesky packing)
      const gsplatsLoaders = Array.from(this.gsplatLoaders.entries()).map(
        async ([path, loader]) => {
          const updateFn = async (session: UpdateSession): Promise<StagedGSplatsCommit | null> => {
            try {
              const mesh = this.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
              const attrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
              const extendDims: string[] = attrs?.extend_to_all || [];

              // Check if we can skip this update (extend_to_all optimization)
              if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
                const dims = this.viewState.dimensions.metadata;
                validateExtendDims(extendDims, dims);
                const nonDisplayedDims = dims
                  .filter(
                    (_: { name?: string }, idx: number) => !this.viewState.displayDims.includes(idx)
                  )
                  .map((d: { name?: string }) => d.name)
                  .filter((name: string | undefined): name is string => !!name);

                const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
                  extendDims.includes(dimName)
                );

                if (isFullyExtended) {
                  log.info(
                    Modules.SCENE_LOADER,
                    `Skipping gsplats update for ${path} - all non-displayed dims are extended`
                  );
                  session.markSkipped('extend_to_all');
                  return null;
                }
              }

              // Build viewState with tolerance override for extend_to_all dimensions
              let gsplatsViewState: GSplatsViewState = {
                displayDims: this.viewState.displayDims,
                slicePosition: this.viewState.slicePosition,
                tolerance: this.viewState.tolerance,
                dimensions: this.viewState.dimensions?.metadata,
              };
              if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
                const tolerance = getOrComputeExtendedTolerance(
                  this.viewState.tolerance,
                  extendDims,
                  this.viewState.dimensions.metadata,
                  extendedToleranceCache
                );
                gsplatsViewState = { ...gsplatsViewState, tolerance };
              }

              // Apply nd_transform inverse for gsplats query
              // Uses composed world nd_transform (inherits from parent groups)
              if (this._sceneGraph && gsplatsViewState.dimensions) {
                const worldNdT = computeWorldNdTransform(this._sceneGraph, path);
                if (hasOwnProperties(worldNdT)) {
                  const dimNames = gsplatsViewState.dimensions.map(
                    (d: { name?: string }) => d.name ?? ''
                  );
                  const inverted = invertNdTransformForQuery(
                    gsplatsViewState.slicePosition,
                    gsplatsViewState.tolerance,
                    worldNdT,
                    dimNames,
                    gsplatsViewState.displayDims
                  );
                  gsplatsViewState = { ...gsplatsViewState, ...inverted };
                }
              }

              const data = await loader.updateView(gsplatsViewState, session);
              if (data) {
                const staged = await this.processGSplatsData(path, data, gsplatsViewState, session);
                session.setMetadata({ splats: data.splatCount });
                this.failedLoaders.delete(path);
                return staged;
              }
              this.failedLoaders.delete(path);
              return null;
            } catch (error) {
              const errorInfo = this.failedLoaders.get(path);
              const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
              this.failedLoaders.set(path, {
                error: error as Error,
                timestamp: Date.now(),
                retryCount,
              });
              log.error(
                Modules.SCENE_LOADER,
                `Failed to update gsplats ${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
              );
              return null;
            }
          };

          if (this.profiler) {
            return this.profiler.timeTopLevel(`GSplats (${path})`, updateFn);
          } else {
            return updateFn(noopSession);
          }
        }
      );

      // Wait for ALL loaders to complete (load + process)
      const [pointsStaged, linesStaged, gsplatsStaged] = await Promise.all([
        Promise.all(pointsLoaders),
        Promise.all(linesLoaders),
        Promise.all(gsplatsLoaders),
      ]);

      // ================================================================
      // Phase 2: Atomic commit — ALL geometry mutations in one sync block
      // Since JS is single-threaded, no requestAnimationFrame can fire
      // during this block. All meshes update in the same rendered frame.
      // ================================================================

      // Advance GPU buffer pool frame counter once per update cycle
      // (not per-acquire) so eviction timing reflects actual frames
      if (this._gpuBufferPool) {
        this._gpuBufferPool.beginFrame();
      }

      for (const staged of pointsStaged) {
        if (staged) this.updatePointsGeometry(staged.path, staged.data);
      }
      for (const staged of linesStaged) {
        if (staged) this.commitLinesGeometry(staged);
      }
      for (const staged of gsplatsStaged) {
        if (staged) this.commitGSplatsGeometry(staged);
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
      if (this._pendingViewState !== null) {
        const pendingState = this._pendingViewState;
        this._pendingViewState = null;

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
   * Runs after the main updateView() commits LOD 0, loading additional LODs
   * one pass at a time with a rAF yield between each pass (so each LOD level
   * is painted as a separate frame, giving visible progressive refinement).
   *
   * Cancellation: if _pendingViewState is set (user navigated), the loop
   * aborts and drains the pending state via the normal serialization path.
   *
   * IMPORTANT: This method does NOT go through the updateView() entry point
   * (which has the serialization lock). It directly calls loader.updateView()
   * + process + commit for GSplats loaders only.
   */
  private async scheduleGSplatsRefinement(): Promise<void> {
    // Track whether cancellation has taken ownership of the lock
    let lockHandedOff = false;
    try {
      while (true) {
        // Yield to let browser paint the current LOD level
        await new Promise<void>((resolve) => {
          if (typeof requestAnimationFrame !== 'undefined') {
            requestAnimationFrame(() => resolve());
          } else {
            resolve(); // Test environment: proceed immediately
          }
        });

        // Check cancellation: did the user navigate?
        if (this._pendingViewState !== null) {
          const pendingState = this._pendingViewState;
          this._pendingViewState = null;

          // Drain pending state via the normal path.
          // The rAF branch keeps the lock until the callback fires,
          // so we must not release it in finally.
          lockHandedOff = true;
          if (typeof requestAnimationFrame !== 'undefined') {
            requestAnimationFrame(() => {
              this._updateInProgress = false;
              this.updateView(pendingState);
            });
          } else {
            this._updateInProgress = false;
            this.updateView(pendingState);
          }
          return;
        }

        // Load next LOD level for each progressive loader
        for (const [path, loader] of this.gsplatLoaders) {
          if (loader.hasMoreLODs !== true) continue;

          try {
            // Build the gsplats view state (same as main update)
            const mesh = this.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
            const nodeAttrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
            const extendDims: string[] = nodeAttrs?.extend_to_all || [];

            let gsplatsViewState: GSplatsViewState = {
              displayDims: this.viewState.displayDims,
              slicePosition: this.viewState.slicePosition,
              tolerance: this.viewState.tolerance,
              dimensions: this.viewState.dimensions?.metadata,
            };

            if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
              const tolerance = [...this.viewState.tolerance];
              for (const dimName of extendDims) {
                const dimIndex = this.viewState.dimensions.metadata.findIndex(
                  (d: { name?: string }) => d.name === dimName
                );
                if (dimIndex >= 0 && dimIndex < tolerance.length) {
                  tolerance[dimIndex] = 1e10;
                }
              }
              gsplatsViewState = { ...gsplatsViewState, tolerance };
            }

            // Apply nd_transform inverse
            if (this._sceneGraph && gsplatsViewState.dimensions) {
              const worldNdT = computeWorldNdTransform(this._sceneGraph, path);
              if (hasOwnProperties(worldNdT)) {
                const dimNames = gsplatsViewState.dimensions.map(
                  (d: { name?: string }) => d.name ?? ''
                );
                const inverted = invertNdTransformForQuery(
                  gsplatsViewState.slicePosition,
                  gsplatsViewState.tolerance,
                  worldNdT,
                  dimNames,
                  gsplatsViewState.displayDims
                );
                gsplatsViewState = { ...gsplatsViewState, ...inverted };
              }
            }

            const data = await loader.updateView(gsplatsViewState);
            if (data) {
              const staged = await this.processGSplatsData(path, data, gsplatsViewState);
              if (staged) this.commitGSplatsGeometry(staged);
            }
          } catch (error) {
            log.error(
              Modules.SCENE_LOADER,
              `GSplats refinement failed for ${path}: ${(error as Error).message}`
            );
          }
        }

        // Update monitor after refinement commit
        this.updateVisibleCountsInMonitor();

        // Check if any progressive loaders still have more LODs after this pass
        const anyMore = [...this.gsplatLoaders.values()].some((l) => l.hasMoreLODs === true);
        if (!anyMore) break; // All LODs loaded
      }
    } finally {
      // Release the lock unless cancellation handed it off to a rAF callback
      if (!lockHandedOff) {
        this._updateInProgress = false;
      }
    }
  }

  /**
   * Aggregate visible counts from all lines and gsplats meshes and update monitor.
   * This should be called after view updates to report accurate visible counts.
   */
  private updateVisibleCountsInMonitor(): void {
    if (!this.rootGroup || !this.monitorId) return;

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

    // Update the monitor
    const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
    if (monitor) {
      monitor.updateVisibleSegments(totalVisibleSegments);
      monitor.updateVisibleSplats(totalVisibleSplats);
    }
  }

  /**
   * Process lines data: compute tolerance, project to 3D (async).
   * Returns staged commit data without mutating any mesh geometry.
   *
   * Implementation lives in `scene-loader/data-processor-lines.ts`; this
   * method is a thin delegate so the pipeline can be tested in isolation
   * without instantiating a SceneLoader.
   */
  private async processLinesData(
    path: string,
    data: LoadedLinesData,
    viewState: {
      displayDims: readonly number[];
      slicePosition: readonly number[];
      dimensions?: DimensionMetadata[];
    },
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
   * Called as part of the atomic commit phase — no async operations allowed.
   */
  private commitLinesGeometry(staged: StagedLinesCommit): void {
    commitLinesGeometryHelper(staged, this.rootGroup, this._gpuBufferPool);
  }

  /**
   * Process gsplats data: project nD to 3D, pack Cholesky factors (async).
   * Returns staged commit data without mutating any mesh geometry.
   *
   * Implementation lives in `scene-loader/data-processor-gsplats.ts`.
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
   * Called as part of the atomic commit phase — no async operations allowed.
   */
  private commitGSplatsGeometry(staged: StagedGSplatsCommit): void {
    commitGSplatsGeometryHelper(staged, this.rootGroup, this._gpuBufferPool);
  }

  /**
   * Build the scene graph structure
   */
  private async buildSceneGraph(
    rootLoc: zarr.Location<zarr.Readable>,
    rootAttrs: ZarrSceneAttrs
  ): Promise<SceneNode> {
    // Enumerate all groups in the store
    const listing = await this.enumerateStore();

    // Build hierarchical structure
    const root: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: rootAttrs,
      hasSpatialIndex: false,
      children: [],
    };

    // Build node map
    const nodeMap = new Map<string, SceneNode>();
    nodeMap.set('/', root);

    // Sort by path depth to ensure parents are created before children
    const sortedPaths = listing
      .filter((e) => e.kind === 'group' && e.path !== '/')
      .sort((a, b) => a.path.split('/').length - b.path.split('/').length);

    for (const entry of sortedPaths) {
      // Skip overlays group — screen-space overlays are not part of the 3D scene graph
      if (entry.path === '/overlays' || entry.path.startsWith('/overlays/')) {
        continue;
      }

      const loc = rootLoc.resolve(entry.path.slice(1)); // Remove leading /
      const group = await zarr.open(loc, { kind: 'group' });
      const attrs = group.attrs as ZarrNodeAttrs;

      // We no longer check for spatial index here - PointSpatialIndexLoader handles it
      const node: SceneNode = {
        path: entry.path,
        type: attrs?.type || 'group',
        attrs: attrs || {},
        hasSpatialIndex: false, // Will be determined by the loader
        children: [],
      };

      // Log if extend_to_all is present
      if (attrs?.extend_to_all) {
        log.data(
          Modules.SCENE_LOADER,
          `Node ${entry.path} has extend_to_all: ${attrs.extend_to_all.join(', ')}`
        );
      }

      // Find parent and add as child
      const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
      const parent = nodeMap.get(parentPath);
      if (parent) {
        parent.children = parent.children || [];
        parent.children.push(node);
      }

      nodeMap.set(entry.path, node);
    }

    return root;
  }

  /**
   * Load all nodes in the scene graph
   */
  private async loadSceneNodes(
    node: SceneNode,
    parentThree: THREE.Object3D,
    parentLoc: zarr.Location<zarr.Readable>
  ): Promise<void> {
    if (node.type === 'points') {
      // Load points
      const points = await this.loadPoints(node, parentLoc);
      if (points) {
        parentThree.add(points);
      }
    } else if (node.type === 'lines') {
      // Load lines
      const lines = await this.loadLines(node, parentLoc);
      if (lines) {
        parentThree.add(lines);
      }
    } else if (node.type === 'gsplats') {
      // Load gsplats
      const gsplats = await this.loadGSplats(node, parentLoc);
      if (gsplats) {
        parentThree.add(gsplats);
      }
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
   * Load a single points node
   */
  private async loadPoints(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Points | null> {
    log.custom('📍', Modules.SCENE_LOADER, `Loading points: ${node.path}`);
    log.info(Modules.SCENE_LOADER, `  Has spatial index: ${node.hasSpatialIndex}`);
    log.info(Modules.SCENE_LOADER, `  Total points: ${node.attrs.n_points || 'unknown'}`);

    // Create appropriate loader
    const loader = this.createLoader(node, loc);

    // Store loader for updates (route through the registry's
    // register* methods rather than mutating its internal map).
    this.registry.registerPointsLoader(node.path, loader);

    try {
      // Load points data
      log.info(Modules.SCENE_LOADER, 'Initial ViewState for loading:');
      log.info(Modules.SCENE_LOADER, `  displayDims: [${this.viewState.displayDims.join(', ')}]`);
      log.info(
        Modules.SCENE_LOADER,
        `  slicePosition: [${this.viewState.slicePosition.join(', ')}]`
      );
      log.info(Modules.SCENE_LOADER, `  tolerance: [${this.viewState.tolerance.join(', ')}]`);

      // Build viewState with tolerance override for extend_to_all dimensions
      const extendDims: string[] = (node.attrs.extend_to_all as string[]) || [];
      let pointsViewState = this.viewState;
      if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
        // Create modified tolerance array with infinite tolerance for extended dims
        const tolerance = [...this.viewState.tolerance];
        for (const dimName of extendDims) {
          const dimIndex = this.viewState.dimensions.metadata.findIndex(
            (d: { name?: string }) => d.name === dimName
          );
          if (dimIndex >= 0 && dimIndex < tolerance.length) {
            tolerance[dimIndex] = 1e10; // Effectively infinite tolerance
            log.info(
              Modules.SCENE_LOADER,
              `  extend_to_all: setting tolerance[${dimIndex}] (${dimName}) to infinity`
            );
          }
        }
        pointsViewState = { ...this.viewState, tolerance };
      }

      // Apply nd_transform inverse for initial load (same as update path)
      if (this._sceneGraph && pointsViewState.dimensions?.metadata) {
        const worldNdT = computeWorldNdTransform(this._sceneGraph, node.path);
        if (hasOwnProperties(worldNdT)) {
          const dimNames = pointsViewState.dimensions.metadata.map(
            (d: { name?: string }) => d.name ?? ''
          );
          const inverted = invertNdTransformForQuery(
            pointsViewState.slicePosition,
            pointsViewState.tolerance,
            worldNdT,
            dimNames,
            pointsViewState.displayDims
          );
          pointsViewState = {
            ...pointsViewState,
            slicePosition: inverted.slicePosition,
            tolerance: inverted.tolerance,
          };
        }
      }

      const data = await loader.loadPoints(pointsViewState);

      // Log if no initial points are visible (this is normal for nD slicing)
      if (data.pointCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible points for ${node.path} - object created for future updates`
        );
      }

      const attrs = this.applyEffectiveAttrs(node) as unknown as PointsMetadata;
      const points = this.nodeFactory.createPointsNode(node.path, attrs, data, loader);

      log.success(Modules.SCENE_LOADER, `Loaded ${data.pointCount} points for ${node.path}`);

      return points;
    } catch (error) {
      // Improved error logging - extract message from error object
      const errorMessage =
        error instanceof Error
          ? error.message
          : typeof error === 'string'
            ? error
            : JSON.stringify(error);
      log.error(Modules.SCENE_LOADER, `Failed to load ${node.path}: ${errorMessage}`);
      // Also log stack trace for debugging
      if (error instanceof Error && error.stack) {
        log.error(Modules.SCENE_LOADER, `Stack trace for ${node.path}`, error.stack);
      }
      return null;
    }
  }

  /**
   * Load a single lines node
   */
  private async loadLines(
    node: SceneNode,
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

    try {
      // Load lines data
      let linesViewState: {
        displayDims: readonly number[];
        slicePosition: readonly number[];
        tolerance: readonly number[];
        dimensions?: import('../types/dims').DimensionMetadata[];
      } = {
        displayDims: this.viewState.displayDims,
        slicePosition: this.viewState.slicePosition,
        tolerance: this.viewState.tolerance,
        dimensions: this.viewState.dimensions?.metadata,
      };

      // Apply nd_transform inverse for initial load
      if (this._sceneGraph && linesViewState.dimensions) {
        const worldNdT = computeWorldNdTransform(this._sceneGraph, node.path);
        if (hasOwnProperties(worldNdT)) {
          const dimNames = linesViewState.dimensions.map((d: { name?: string }) => d.name ?? '');
          const inverted = invertNdTransformForQuery(
            linesViewState.slicePosition,
            linesViewState.tolerance,
            worldNdT,
            dimNames,
            linesViewState.displayDims
          );
          linesViewState = { ...linesViewState, ...inverted };
        }
      }

      const data = await loader.loadLines(linesViewState);

      if (data.segmentCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible segments for ${node.path} - object created for future updates`
        );
      }

      // Build instance buffers with nD clipping
      let tolerance = computeTolerance(
        'lines',
        linesViewState.displayDims,
        attrs.ndim || 3,
        linesViewState.dimensions
      );

      // CRITICAL: For extend_to_all dimensions, set tolerance to infinity
      // This ensures segments aren't clipped when navigating through extended dimensions
      const extendDims: string[] = attrs.extend_to_all || [];
      if (extendDims.length > 0 && linesViewState.dimensions) {
        tolerance = [...tolerance]; // Make a copy to avoid mutating shared array
        for (const dimName of extendDims) {
          const dimIndex = linesViewState.dimensions.findIndex(
            (d: { name?: string }) => d.name === dimName
          );
          if (dimIndex >= 0 && dimIndex < tolerance.length) {
            tolerance[dimIndex] = 1e10; // Effectively infinite tolerance
          }
        }
      }

      // Build instance buffers (use worker for larger datasets)
      const useWorkerProjection =
        appConfig.dataLoading.performance.useWebWorkers && data.segmentCount > 1000;

      let processed: ProcessedLinesData;
      if (useWorkerProjection) {
        processed = await projectLinesTo3DUsingWorkerHelper(
          data,
          linesViewState,
          tolerance,
          this._updateVersion
        );
      } else {
        processed = buildInstanceBuffers(
          data,
          linesViewState.slicePosition,
          tolerance,
          linesViewState.displayDims
        );
      }

      const mesh = this.nodeFactory.createLinesNode(
        node.path,
        this.applyEffectiveAttrs(node),
        attrs,
        processed,
        loader
      );

      log.success(
        Modules.SCENE_LOADER,
        `Loaded ${processed.segmentCount} segments for ${node.path}`
      );

      return mesh;
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : typeof error === 'string'
            ? error
            : JSON.stringify(error);
      log.error(Modules.SCENE_LOADER, `Failed to load lines ${node.path}: ${errorMessage}`);
      if (error instanceof Error && error.stack) {
        log.error(Modules.SCENE_LOADER, `Stack trace for ${node.path}`, error.stack);
      }
      return null;
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
    return createLinesLoaderHelper(node, loc, this.factoryDeps());
  }

  /**
   * Load a single gsplats node
   */
  private async loadGSplats(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Mesh | null> {
    const attrs = node.attrs as unknown as GSplatsMetadata;
    const nLods = attrs.n_lods ?? 0;
    log.custom('🔮', Modules.SCENE_LOADER, `Loading gsplats: ${node.path}`);
    log.info(
      Modules.SCENE_LOADER,
      `  Splats: ${(nLods > 1 ? (attrs.n_splats_total ?? attrs.n_splats) : attrs.n_splats)?.toLocaleString() || 'unknown'}`
    );
    log.info(Modules.SCENE_LOADER, `  Dimensions: ${attrs.ndim || 'unknown'}D`);
    if (nLods > 1) {
      log.info(Modules.SCENE_LOADER, `  LODs: ${nLods} (progressive loading enabled)`);
    }

    // Create gsplats loader — progressive for multi-LOD, standard for single-LOD
    const loader =
      nLods > 1
        ? await this.createProgressiveGSplatsLoader(node, loc, nLods)
        : this.createGSplatsLoader(node, loc);

    // Store loader for updates (route through registry).
    this.registry.registerGSplatsLoader(node.path, loader);

    try {
      // Build gsplats view state with tolerance override for extend_to_all dimensions
      const extendDims: string[] = (node.attrs.extend_to_all as string[]) || [];
      let gsplatsViewState: GSplatsViewState = {
        displayDims: this.viewState.displayDims,
        slicePosition: this.viewState.slicePosition,
        tolerance: this.viewState.tolerance,
        dimensions: this.viewState.dimensions?.metadata,
      };
      if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
        const tolerance = [...this.viewState.tolerance];
        for (const dimName of extendDims) {
          const dimIndex = this.viewState.dimensions.metadata.findIndex(
            (d: { name?: string }) => d.name === dimName
          );
          if (dimIndex >= 0 && dimIndex < tolerance.length) {
            tolerance[dimIndex] = 1e10;
          }
        }
        gsplatsViewState = { ...gsplatsViewState, tolerance };
      }

      // Apply nd_transform inverse for initial load
      if (this._sceneGraph && gsplatsViewState.dimensions) {
        const worldNdT = computeWorldNdTransform(this._sceneGraph, node.path);
        if (hasOwnProperties(worldNdT)) {
          const dimNames = gsplatsViewState.dimensions.map((d: { name?: string }) => d.name ?? '');
          const inverted = invertNdTransformForQuery(
            gsplatsViewState.slicePosition,
            gsplatsViewState.tolerance,
            worldNdT,
            dimNames,
            gsplatsViewState.displayDims
          );
          gsplatsViewState = { ...gsplatsViewState, ...inverted };
        }
      }

      // Load gsplats data
      const data = await loader.loadGSplats(gsplatsViewState);

      if (data.splatCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible gsplats for ${node.path} - object created for future updates`
        );
      }

      // Process nD data to 3D for rendering
      // Strategy: use worker for larger nD datasets
      const useWorkerProjection =
        appConfig.dataLoading.performance.useWebWorkers && data.splatCount > 1000 && data.ndim > 3;

      // Read truncation radius from zarr metadata for nD attenuation
      const truncate = (attrs.truncation_radius as number | undefined) ?? 3.0;

      let processed: ReturnType<typeof processGSplats>;
      if (useWorkerProjection) {
        processed = await projectGSplatsTo3DUsingWorkerHelper(
          data,
          gsplatsViewState,
          truncate,
          this._updateVersion
        );
      } else {
        processed = processGSplats(data, gsplatsViewState, truncate);
      }

      // Pack Cholesky factors for shader
      const { cholesky01, cholesky23, cholesky45 } = packCholeskyForShader(
        processed.choleskyFactors3D,
        processed.splatCount
      );

      const meshConfig = {
        centers: processed.centers3D,
        cholesky01,
        cholesky23,
        cholesky45,
        amplitudes: processed.amplitudes,
        colors: processed.colors,
        splatCount: processed.splatCount,
      };
      const mesh = this.nodeFactory.createGSplatsNode(
        node.path,
        this.applyEffectiveAttrs(node),
        attrs,
        meshConfig,
        loader
      );

      log.success(
        Modules.SCENE_LOADER,
        `Loaded ${processed.splatCount.toLocaleString()} gsplats for ${node.path}`
      );

      return mesh;
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : typeof error === 'string'
            ? error
            : JSON.stringify(error);
      log.error(Modules.SCENE_LOADER, `Failed to load gsplats ${node.path}: ${errorMessage}`);
      if (error instanceof Error && error.stack) {
        log.error(Modules.SCENE_LOADER, `Stack trace for ${node.path}`, error.stack);
      }
      return null;
    }
  }

  /** Create a single-LOD gsplats loader for a node. */
  private createGSplatsLoader(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): GSplatsDataLoader {
    return createGSplatsLoaderHelper(node, loc, this.factoryDeps());
  }

  /**
   * Create a progressive gsplats loader for a multi-LOD node. The
   * parent's effective rendering attrs are composed up the scene-graph
   * ancestry here (not in the helper) so the LOD synthetic nodes see
   * ancestor opacity/intensity/etc.
   */
  private createProgressiveGSplatsLoader(
    node: SceneNode,
    _loc: zarr.Location<zarr.Readable>,
    nLods: number
  ): Promise<GSplatsDataLoader> {
    return createProgressiveGSplatsLoaderHelper(
      node,
      nLods,
      this.applyEffectiveAttrs(node),
      this.factoryDeps()
    );
  }

  /**
   * Create the points spatial index loader for a node and connect it to
   * the data monitor (the monitor wiring stays here because it touches
   * SceneManager-only state — the factory only constructs the loader).
   */
  private createLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): DataLoader {
    const loader = createPointsLoaderHelper(node, loc, this.config, this.factoryDeps());

    if (this.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
      if (monitor) {
        monitor.connectLoader(node.path, loader);
      }
    }

    return loader;
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
   * Initialize scene dimensions from metadata using ViewStateManager
   */
  private initializeSceneDimensions(sceneDims: unknown): void {
    // Validate sceneDims structure
    if (!isSceneDimensions(sceneDims)) {
      log.warning(Modules.SCENE_LOADER, 'Invalid scene_dimensions format, skipping');
      return;
    }

    // Validate dimensions using ViewStateManager
    const validation = ViewStateManager.validateDimensions(sceneDims.dimensions);

    // Log validation results
    const displayedCount = sceneDims.dimensions.filter((d) => d.display === true).length;
    ViewStateManager.logValidationResults(validation, sceneDims.dimensions.length, displayedCount);

    // Stop if validation failed with errors
    if (!validation.isValid) {
      log.error(Modules.SCENE_LOADER, 'Scene dimensions validation failed, cannot initialize');
      return;
    }

    // Initialize ViewState using ViewStateManager
    this.viewState = ViewStateManager.initializeFromDimensions(sceneDims);
  }

  /**
   * Convert internal SceneNode to SceneGraphNode for monitor display
   */
  private convertToSceneGraphNode(node: SceneNode): SceneGraphNode {
    // Get display name from path
    const name =
      node.path === '/' ? 'Scene' : node.path.split('/').filter(Boolean).pop() || node.path;

    // Determine node type for display
    const type = node.type as 'scene' | 'group' | 'points' | 'lines' | 'gsplats' | 'mesh';

    // Build the graph node
    const graphNode: SceneGraphNode = {
      path: node.path,
      name,
      type: type === 'scene' || !type ? 'scene' : type,
      children: [],
      hasSpatialIndex: node.hasSpatialIndex,
    };

    // Add type-specific stats
    if (node.type === 'points') {
      graphNode.pointCount = node.attrs.n_points;
    } else if (node.type === 'lines') {
      graphNode.segmentCount = node.attrs.n_segments as number | undefined;
      graphNode.vertexCount = node.attrs.n_vertices as number | undefined;
    } else if (node.type === 'gsplats') {
      graphNode.splatCount = node.attrs.n_splats as number | undefined;
    }

    // Convert children recursively
    if (node.children) {
      graphNode.children = node.children.map((child) => this.convertToSceneGraphNode(child));
    }

    return graphNode;
  }

  /**
   * Enumerate all groups and arrays in the store
   */
  private async enumerateStore(): Promise<Array<{ path: string; kind: string }>> {
    if (!this._zarrStore) return [];

    // Try to use consolidated metadata
    if (hasContentsMethod(this._zarrStore)) {
      const contents = await this._zarrStore.contents();
      log.custom('📋', Modules.SCENE_LOADER, `Found ${contents.length} items in store`);
      return contents;
    }

    // Fallback enumeration
    log.warning(Modules.SCENE_LOADER, 'Store does not support contents(), using fallback');
    return [{ path: '/', kind: 'group' }];
  }

  /**
   * Normalize URL for zarr store access
   */
  private normalizeURL(url: string): string {
    return normalizeURL(url, window.location.origin);
  }

  /**
   * Show the monitor UI
   */
  showMonitor(): void {
    if (this.monitorId) {
      DataMonitorManager.getInstance().showMonitor(this.monitorId);
    }
  }

  /**
   * Hide the monitor UI
   */
  hideMonitor(): void {
    if (this.monitorId) {
      DataMonitorManager.getInstance().hideMonitor(this.monitorId);
    }
  }

  /**
   * Toggle the monitor UI
   */
  toggleMonitor(): void {
    if (this.monitorId) {
      DataMonitorManager.getInstance().toggleMonitor(this.monitorId);
    }
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
   * This method re-triggers the update for a previously failed loader,
   * using the current view state. Useful for recovering from transient
   * network errors or after connectivity is restored.
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

    log.info(Modules.SCENE_LOADER, `Retrying failed loader: ${path}`);

    // Determine which loader type this path belongs to
    const pointsLoader = this.loaders.get(path);
    const linesLoader = this.linesLoaders.get(path);
    const gsplatsLoader = this.gsplatLoaders.get(path);

    try {
      if (pointsLoader) {
        // Retry points loader
        const points = await pointsLoader.updateView(this.viewState);
        if (points) {
          this.updatePointsGeometry(path, points);
        }
        this.failedLoaders.delete(path);
        log.success(Modules.SCENE_LOADER, `Successfully retried points loader: ${path}`);
        return true;
      } else if (linesLoader) {
        // Retry lines loader
        const linesViewState = {
          displayDims: this.viewState.displayDims,
          slicePosition: this.viewState.slicePosition,
          tolerance: this.viewState.tolerance,
          dimensions: this.viewState.dimensions?.metadata,
        };
        const data = await linesLoader.updateView(linesViewState);
        if (data) {
          const staged = await this.processLinesData(path, data, linesViewState);
          if (staged) this.commitLinesGeometry(staged);
        }
        this.failedLoaders.delete(path);
        log.success(Modules.SCENE_LOADER, `Successfully retried lines loader: ${path}`);
        return true;
      } else if (gsplatsLoader) {
        // Retry gsplats loader
        const gsplatsViewState: GSplatsViewState = {
          displayDims: this.viewState.displayDims,
          slicePosition: this.viewState.slicePosition,
          tolerance: this.viewState.tolerance,
          dimensions: this.viewState.dimensions?.metadata,
        };
        const data = await gsplatsLoader.updateView(gsplatsViewState);
        if (data) {
          const staged = await this.processGSplatsData(path, data, gsplatsViewState);
          if (staged) this.commitGSplatsGeometry(staged);
        }
        this.failedLoaders.delete(path);
        log.success(Modules.SCENE_LOADER, `Successfully retried gsplats loader: ${path}`);
        return true;
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
   * Retry all failed loaders.
   *
   * This method attempts to re-load all loaders that previously failed.
   * Useful for batch recovery after network connectivity is restored.
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

    log.info(Modules.SCENE_LOADER, `Retrying ${failedPaths.length} failed loader(s)`);

    const succeeded: string[] = [];
    const failed: string[] = [];

    // Retry all in parallel for efficiency
    const results = await Promise.all(
      failedPaths.map(async (path) => {
        const success = await this.retryFailedLoader(path);
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
  }

  /**
   * Dispose of all resources
   */
  dispose(): void {
    // Dispose all geometry loaders via registry
    this.registry.disposeAll();

    // Dispose caching store (flushes L2 metadata, clears L1)
    if (this.cachingStore) {
      this.cachingStore.dispose().catch((error) => {
        log.warning(Modules.SCENE_LOADER, 'Caching store disposal failed', error);
      });
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

    // Note: We don't dispose the monitor here as it's managed by DataMonitorManager
    // The monitor can be reused by other SceneLoader instances
    this.monitorId = null;
  }
}
