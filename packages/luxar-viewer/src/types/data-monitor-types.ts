/**
 * Type definitions for the new Data Loading Monitor architecture.
 *
 * This module defines the interfaces and types for the event-driven
 * monitoring system that tracks spatial index queries, cache performance,
 * and loading statistics in real-time.
 */

/**
 * Geometry-neutral index range for monitor events and query tracking.
 * `PointRange` / `SegmentRange` / `SplatRange` (the per-geometry query
 * types) are structurally identical, so every loader's ranges assign here
 * directly — no casts.
 */
export interface ElementRange {
  /** Starting index (inclusive) */
  start: number;
  /** Ending index (exclusive) */
  end: number;
}

/**
 * Event types emitted by data loaders
 */
export type MonitorEventType =
  | 'query' // Spatial index or range query initiated
  | 'load' // Data loaded from source
  | 'cache-hit' // Data found in cache
  | 'cache-miss' // Data not in cache, needs loading
  | 'evict' // Data evicted from cache
  | 'error' // Loading or query error
  | 'prefetch'; // Prefetch operation

/**
 * Loader types in the system
 */
export type LoaderType = 'point-spatial-index' | 'lines-spatial-index' | 'gsplats-spatial-index';

/**
 * Event emitted by data loaders for monitoring
 */
export interface MonitorEvent {
  type: MonitorEventType;
  loader: LoaderType;
  timestamp: number;
  data: {
    path?: string;
    arrayName?: string;
    ranges?: ElementRange[];
    /** Element count for the event (points / vertices-or-segments / splats). */
    elements?: number;
    cells?: number;
    latency?: number;
    memory?: number;
    cacheKey?: string;
    error?: string;
    // Spatial-specific data
    queryPosition?: readonly number[];
    queryTolerance?: readonly number[];
    gridBounds?: { min: readonly number[]; max: readonly number[] };
  };
}

/**
 * Listener function for monitor events
 */
export type MonitorEventListener = (event: MonitorEvent) => void;

/**
 * Interface for objects that can be monitored
 */
export interface LoaderMonitor {
  addEventListener(listener: MonitorEventListener): void;
  removeEventListener(listener: MonitorEventListener): void;
  getMetrics(): LoaderMetrics;
  getActiveQueries(): QueryInfo[];
}

/**
 * Metrics for a specific loader
 */
export interface LoaderMetrics {
  type: LoaderType;
  path: string;
  // Basic counters
  queries: number;
  loads: number;
  evictions: number;
  errors: number;
  // Performance metrics
  elementsLoaded: number; // Cumulative (for throughput calculation)
  bytesLoaded: number;
  // Dataset info
  visibleElements: number; // Currently visible/rendered points (non-cumulative)
  avgQueryTime: number;
  avgLoadTime: number;
  // Memory usage
  memoryUsed: number;
  memoryLimit: number;
  // Spatial index specific metrics
  spatialIndex?: SpatialIndexMetrics;
  // Performance optimization metrics
  optimization?: OptimizationMetrics;
}

/**
 * Performance optimization metrics — accumulator pooling, worker
 * offload, WASM acceleration, and GPU buffer pool.
 */
export interface OptimizationMetrics {
  // Accumulator stats
  accumulator?: {
    enabled: boolean;
    capacity: number;
    allocations: number;
    growthEvents: number;
    memoryMB: number;
  };
  // Worker stats
  worker?: {
    enabled: boolean;
    queriesOffloaded: number;
    fallbackCount: number;
  };
  // WASM stats
  wasm?: {
    loaded: boolean;
    queriesAccelerated: number;
  };
  // GPU Buffer Pool stats
  gpuPool?: {
    enabled: boolean;
    allocations: number;
    reuses: number;
    evictions: number;
    capacityGrowths: number;
    activeBuffers: number;
    pooledBuffers: number;
    reuseRate: number; // Calculated: reuses / (allocations + reuses)
  };
}

/**
 * Spatial index specific metrics
 */
export interface SpatialIndexMetrics {
  gridShape: number[];
  gridOrigin: number[];
  cellSize: number[];
  occupiedCells: number;
  totalCells: number;
  avgCellsPerQuery: number;
  avgElementsPerCell: number;
  queryEfficiency: number; // Points loaded / points in query region
  lastQueryBounds?: { min: number[]; max: number[] };
  rangesInCache: number; // Number of cached range queries
}

/**
 * Information about an active or recent query
 */
export interface QueryInfo {
  id: string;
  loader: LoaderType;
  path: string;
  startTime: number;
  endTime?: number;
  status: 'pending' | 'loading' | 'complete' | 'error';
  cells?: number;
  /** Element count the query matched (points / segments / splats). */
  elements?: number;
  ranges?: ElementRange[];
  fromCache?: boolean;
  error?: string;
}

/**
 * Recommendation for performance improvement
 */
export interface Recommendation {
  id: string;
  severity: 'info' | 'warning' | 'error';
  category: 'performance' | 'memory' | 'configuration';
  title: string;
  message: string;
  suggestion?: string;
  metric?: string;
  value?: number;
  threshold?: number;
}

/**
 * Monitor display configuration
 */
export interface MonitorConfig {
  // Display settings
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  theme: 'dark' | 'light' | 'auto';
  defaultView: 'compact' | 'detailed' | 'debug';

  // Update settings
  updateInterval: number; // ms between UI updates
  maxEvents: number; // Maximum events to keep in history

  // Feature flags
  showSpatialGrid: boolean;
  showTimeline: boolean;
  showRecommendations: boolean;
  autoExpand: boolean; // Auto-expand on warnings

  // Performance
  enableProfiling: boolean;
  sampleRate: number; // Sample 1 in N events for profiling
}

/**
 * Aggregated statistics across all loaders
 */
export interface GlobalStats {
  totalLoaders: number;
  activeSpatialLoaders: number;
  totalPoints: number; // Cumulative points loaded (for throughput)
  // Resident memory across all loaders — sum of each loader's `memoryUsed`
  // (current accumulator allocation, set in the spatial-index loaders'
  // recordLoadMetrics). Drives the compact badge's memory figure.
  totalMemory: number;
  // Dataset metrics - Points
  datasetSize: number; // Total points in all datasets
  visibleElements: number; // Currently visible/rendered points
  // Dataset metrics - Lines
  datasetSegments: number; // Total segments in all line datasets
  visibleSegments: number; // Currently visible/rendered segments (for lines, typically equals total)
  // Dataset metrics - GSplats
  datasetSplats: number; // Total splats in all gsplats datasets
  visibleSplats: number; // Currently visible/rendered splats
  totalQueries: number;
  totalLoads: number;
  avgQueryTime: number;
  queriesPerSecond: number;
  recommendations: Recommendation[];
}

/**
 * UI Component state
 */
export interface MonitorUIState {
  isVisible: boolean;
  isExpanded: boolean;
  activeTab: 'overview' | 'cache' | 'memory' | 'performance' | 'insights';
  selectedLoader?: string;
  timeRange: number; // Seconds of history to show
  spatialViewDimensions?: [number, number]; // Which 2D slice to show (for future use)
}

/**
 * Explicit cache telemetry state — distinguishes the three
 * "not-enabled" variants from each other so the UI can render an
 * accurate disabled-reason. A `?no-cache` URL run produces no
 * provider, so without this state the cache tab would default to
 * `enabled` and mislead the user.
 *
 *   - `enabled`           : caching is on AND providers are wired.
 *   - `disabled-no-cache` : caching turned off via `?no-cache`.
 *   - `disabled-config`   : turned off via app config.
 *   - `not-wired`         : caching is on but providers haven't been
 *                           wired yet (e.g. mid-scene-transition).
 *
 * Lives in `types/` so the data layer (cache-setup.ts) and the UI
 * layer (ui/data-loading-monitor/metrics/cache.ts) can both reference
 * it without crossing the layer boundary.
 */
export type CacheTelemetryState =
  | { kind: 'enabled' }
  | { kind: 'disabled-no-cache' }
  | { kind: 'disabled-config' }
  | { kind: 'not-wired' };

/**
 * Cache metrics for detailed analytics
 */
export interface CacheMetrics {
  totalCacheMemory: number;
  memoryLimit: number;
  memoryPercent: number;
  totalEntries: number;
  totalAccesses: number;
  /**
   * L1-only hit rate. Computed as `l1.hits / (l1.hits + l1.misses)`.
   * Kept for dashboard compatibility.
   */
  recentHitRate: number;
  /**
   * Effective demand hit-rate across all cache tiers:
   * `(l0Hits + l1Hits + l2Hits) / (l0Hits + l1Hits + l2Hits + networkRequests)`.
   * Undefined when neither L0 provider nor `demand` counters are wired
   * (lets the UI distinguish "not wired up" from "0% hit rate").
   */
  effectiveDemandHitRate?: number;
  /** Total evictions accumulated across loaders. */
  evictionsTotal: number;
  avgEntrySize: number;
  reuseRatio: number;
  hitsPerSecond: number;
  missesPerSecond: number;
  avgAccessTime: number;
  queriesPerSec: number;
  loadsPerSec: number;
  bandwidth: number;
  /** L1 memory cache breakdown (optional, only when CacheStatsProvider connected) */
  l1?: {
    size: number;
    count: number;
    hits: number;
    misses: number;
    evictions: number;
  };
  /** L2 OPFS cache breakdown (optional, only when CacheStatsProvider connected) */
  l2?: {
    size: number;
    count: number;
    /** Successful gets (= L2 hits). */
    reads: number;
    writes: number;
    /** Failed gets (file not present, size mismatch, I/O error). */
    misses: number;
    /**
     * R3: surface OPFS health counters so the cache tab can render
     * them inline (rather than only signalling them via the
     * `cache-errors-detected` / `quota-constrained` badges). Each is
     * optional — providers that don't expose them simply omit the
     * field and the UI degrades to a "no errors" indicator.
     */
    quotaWriteSkipped?: number;
    writeFailures?: number;
    corruptedEntries?: number;
    metadataParseFailures?: number;
  };
  /** L0 decompressed chunk cache breakdown (optional, only when L0 cache connected) */
  l0?: {
    size: number;
    count: number;
    hits: number;
    misses: number;
    evictions: number;
    hitRate: number;
    /** Resolved (heap-aware) byte budget. */
    maxSize?: number;
  };
  /** SliceCache ("S-cache") breakdown (optional, only when the SliceCache is connected) */
  slice?: {
    size: number;
    count: number;
    hits: number;
    misses: number;
    evictions: number;
    hitRate: number;
    /** Eviction-induced misses (working-set-over-budget / cyclic-playback thrash). */
    thrashMisses?: number;
    /** Resolved (heap-aware) byte budget — varies by device heap. */
    maxSize?: number;
  };
  /**
   * Whether caching is enabled. Derived from
   * `telemetryState.kind === 'enabled'`. New code should read
   * `telemetryState` directly to distinguish the three not-enabled
   * variants from each other.
   */
  enabled?: boolean;
  /**
   * Explicit cache telemetry state. Distinguishes "disabled-no-cache"
   * (URL flag), "disabled-config" (app config / isEnabled false),
   * and "not-wired" (provider absent during scene transition or
   * before cache setup completes) from each other.
   */
  telemetryState?: CacheTelemetryState;
  /** Network I/O stats (optional, only when CacheStatsProvider connected) */
  network?: {
    bytesTransferred: number;
    requestCount: number;
    bandwidth: number;
    /**
     * Cumulative bytes delivered to demand callers across all cache
     * tiers (L1 + L2 + network). Optional — providers predating the
     * field omit it. Drives the Overview "DATA LOADED" card so it
     * stays informative on warm/cache-served reloads.
     */
    totalBytesServed?: number;
    /** Count of demand reads served across all tiers. */
    totalRequestsServed?: number;
  };
  /**
   * Status badges for the cache tab. Derived in the aggregator from
   * telemetry state + provider presence + cache health. UI renders
   * each as a small chip; consumers reading metrics programmatically
   * (e.g. debug snapshots, E2E tests) can also assert on them.
   */
  status?: CacheStatusBadge[];
  /**
   * Cache health snapshot mirroring MultiLevelCachingStore.getStats().health.
   * Optional because some providers may not surface it.
   */
  health?: {
    validationMode?: 'content-hash' | 'zattrs-hash' | 'ttl' | 'none';
    lastValidatedAt?: number | null;
    unvalidatedExternalDataset?: boolean;
    /**
     * S2: `true` when OPFS L2 storage is operational or caching is
     * disabled (no L2 expected). `false` when L2 was expected but
     * could not be initialised — drives the `opfs-unavailable` badge.
     */
    opfsAvailable?: boolean;
  };
}

/**
 * UI status badges surfaced in the cache tab. Derived in
 * `aggregateCacheMetrics`. Each badge corresponds to a different
 * "operational state" the user might need to know about — not all are
 * mutually exclusive (e.g. cache-enabled + unvalidated-external can
 * coexist, surfaced as two badges).
 */
export type CacheStatusBadge =
  | 'cache-enabled'
  | 'no-cache'
  | 'disabled-config'
  | 'opfs-unavailable'
  | 'quota-constrained'
  | 'unvalidated-external-dataset'
  | 'cache-errors-detected'
  | 'provider-missing';

/**
 * Performance timeline data point
 *
 * @internal — reserved extension shape; no current consumer.
 */
export interface TimelinePoint {
  timestamp: number;
  queryTime?: number;
  loadTime?: number;
  cacheHitRate?: number;
  memoryUsed?: number;
  elementsLoaded?: number;
  loaderType?: LoaderType;
  event?: MonitorEventType;
}

/**
 * Spatial grid cell state for visualization
 *
 * @internal — reserved extension shape; no current consumer.
 */
export interface GridCellState {
  x: number;
  y: number;
  z?: number;
  isOccupied: boolean;
  isCached: boolean;
  isLoading: boolean;
  isQueried: boolean;
  points: number;
  lastAccess?: number;
}

/**
 * Node type for scene graph display
 */
export type SceneGraphNodeType = 'scene' | 'group' | 'points' | 'lines' | 'gsplats' | 'mesh';

/**
 * Scene graph node for UI display.
 * Simplified version of SceneNode from data-loader-types.ts.
 */
export interface SceneGraphNode {
  /** Path in the zarr store */
  path: string;
  /** Display name (last component of path or 'Scene') */
  name: string;
  /** Node type */
  type: SceneGraphNodeType;
  /** Number of points (for points nodes) */
  pointCount?: number;
  /** Number of visible points after nD slicing (for points nodes) */
  visiblePointCount?: number;
  /** Number of segments (for lines nodes) */
  segmentCount?: number;
  /** Number of visible segments after nD slicing (for lines nodes) */
  visibleSegmentCount?: number;
  /** Number of vertices (for lines nodes) */
  vertexCount?: number;
  /** Number of splats (for gsplats nodes) */
  splatCount?: number;
  /** Number of visible splats after nD slicing (for gsplats nodes) */
  visibleSplatCount?: number;
  /**
   * Specialized-group discriminant, set when the underlying scene-graph
   * node is a `kind=lod` (substitutive LOD) or `kind=partition` (BSP)
   * `Group`. Drives the tree's kind badge + icon. Mirrors `LayerInfo.kind`
   * in `ui/layers/layer-state.ts`.
   */
  kind?: 'lod' | 'partition';
  /**
   * Resolved geometry `display_type` (points / lines / gsplats) for a
   * specialized group — the type the user logically sees the group as.
   * Absent for plain groups and leaves (use `type` there).
   */
  displayType?: 'points' | 'lines' | 'gsplats';
  /** For `kind=lod` groups: number of substitutive levels (child count). */
  lodGroupChildCount?: number;
  /** For `kind=partition` groups: number of BSP parts (child count). */
  partCount?: number;
  /**
   * For additive-LOD leaves (`n_additive_sublods > 1`): the total number of
   * additive sublods. A structural marker (the `additive_<i>` subgroups are
   * hidden from the scene graph) so the tree can always render a "LOD x/N"
   * progress slot; the live `loaded`/`refining` values come from
   * {@link LODProgressProvider}.
   */
  additiveSublods?: number;
  /** Whether this node has a spatial index */
  hasSpatialIndex?: boolean;
  /** Child nodes */
  children: SceneGraphNode[];
  /** UI state: whether node is expanded in tree view */
  isExpanded?: boolean;
}

/**
 * Discriminates the three LOD/partition loading shapes a scene-graph node
 * can take, for the live progress surfaced by {@link LODProgressProvider}:
 *
 *   - `lod`        : substitutive `kind=lod` group — K mutually-exclusive
 *                    levels, one rendered at a time (`LODGroupRegistry`).
 *   - `additive`   : a single node with `n_additive_sublods > 1`,
 *                    progressively refined by the refinement loop.
 *   - `partition`  : `kind=partition` BSP group — N disjoint parts, all
 *                    rendered (per-part frustum culling).
 */
export type LODNodeKind = 'lod' | 'additive' | 'partition';

/**
 * Live, per-node LOD / progressive-refinement / cache-residency state,
 * polled by the monitor each tick. Keyed by scene-graph path. All fields
 * beyond `kind` are optional — only the ones relevant to a node's kind are
 * populated.
 */
export interface LODProgressState {
  kind: LODNodeKind;
  /** substitutive: total number of levels (child count). */
  levelCount?: number;
  /** substitutive: currently active level index (0-based, coarsest-first). */
  activeLevel?: number;
  /** substitutive: selector mode — `'auto'` or a locked level label. */
  selector?: string;
  /** additive: number of LOD levels loaded so far. */
  loaded?: number;
  /** additive: total LOD levels available. */
  total?: number;
  /** additive: more LODs pending — refinement loop still running. */
  refining?: boolean;
  /**
   * additive: whether the most recent streamed load was fully
   * cache-resident (drives a "cached" vs "streaming" indicator). Mirrors
   * `ResidencyAccumulator.allResident`.
   */
  lastAllResident?: boolean;
  /**
   * additive: committed energy fraction e(k) ∈ [0, 1] of the loaded ladder
   * prefix (from the build-time `energy_fraction_cum` quality stamps —
   * energy-ordered streaming front-loads the visually important elements).
   * Undefined on unstamped (legacy) datasets.
   */
  energy?: number;
  /** partition: number of BSP parts. */
  partCount?: number;
}

/**
 * Provides a snapshot of live LOD / refinement / residency state keyed by
 * scene-graph path. Injected into the monitor via
 * `SceneLoaderMonitorPort.setLODProgressProvider` and polled on each tick.
 * Implemented in the data layer over the loader maps + `LODGroupRegistry`.
 */
export interface LODProgressProvider {
  getLODStates(): Map<string, LODProgressState>;
}

/**
 * Scene graph state for monitor
 */
export interface SceneGraphState {
  /** Root node of the scene graph */
  root: SceneGraphNode | null;
  /** Total number of nodes */
  totalNodes: number;
  /** Number of points nodes */
  pointsNodes: number;
  /** Number of lines nodes */
  linesNodes: number;
  /** Number of gsplats nodes */
  gsplatsNodes: number;
  /** Total points across all nodes */
  totalPoints: number;
  /** Currently visible points (after nD clipping / progressive LOD) */
  visibleElements: number;
  /** Total segments across all lines */
  totalSegments: number;
  /** Currently visible segments (after nD clipping) */
  visibleSegments: number;
  /** Total splats across all gsplats nodes */
  totalSplats: number;
  /** Currently visible splats (after nD clipping) */
  visibleSplats: number;
}

/**
 * Interface for objects that provide cache statistics.
 * Used for loose coupling between MultiLevelCachingStore and DataLoadingMonitor.
 */
export interface CacheStatsProvider {
  /** Get current cache statistics */
  getStats(): {
    l1: {
      metadataSize: number;
      chunksSize: number;
      metadataCount: number;
      chunksCount: number;
      hits: number;
      misses: number;
      evictions: number;
      /** Resolved (heap-aware) L1 byte budget. */
      maxSize?: number;
    };
    l2: {
      size: number;
      count: number;
      reads: number;
      writes: number;
      misses: number;
      /** Fixed OPFS/disk byte budget. */
      maxSize?: number;
      /**
       * Optional OPFS health counters; provider stubs may omit them.
       */
      oversizedWriteSkipped?: number;
      quotaWriteSkipped?: number;
      evictions?: number;
      writeFailures?: number;
      corruptedEntries?: number;
      metadataParseFailures?: number;
      orphanedFilesRemoved?: number;
    };
    network: {
      bytesTransferred: number;
      requestCount: number;
      bandwidth: number;
      /**
       * Cumulative bytes delivered to demand callers across all tiers
       * (L1 + L2 + network). Optional so provider stubs may omit it.
       */
      totalBytesServed?: number;
      totalRequestsServed?: number;
    };
    /**
     * Per-tier demand-hit counters from the multi-level caching
     * store. Each demand request increments exactly one of l1Hits /
     * l2Hits / networkRequests. Optional so providers that don't
     * report it still typecheck.
     */
    demand?: {
      l1Hits: number;
      l2Hits: number;
      networkRequests: number;
    };
    /** Optional cache validation health. */
    health?: {
      validationMode: 'content-hash' | 'zattrs-hash' | 'ttl' | 'none';
      lastValidatedAt: number | null;
      unvalidatedExternalDataset: boolean;
      /**
       * S2: `true` when OPFS L2 storage is operational or caching is
       * disabled (no L2 expected). `false` only when L2 was expected
       * but could not be initialised. Older providers omit the field
       * entirely; consumers treat the absent case as "unknown / assume
       * available".
       */
      opfsAvailable?: boolean;
    };
  };
  /** Clear L1 memory cache */
  clearL1(): void;
  /** Clear L2 OPFS cache */
  clearL2(): Promise<void>;
  /** Clear all caches (L1 + L2) */
  clearAll(): Promise<void>;
  /** Check if caching is enabled */
  isEnabled(): boolean;
}

// ============================================================================
// Memory metrics contracts
// ============================================================================
//
// These contracts are *cross-layer* — the data layer's
// `SceneLoaderMonitorPort` exposes provider methods that produce them,
// and the UI layer renders them. Keeping them in `types/` (the
// foundational layer) lets both ends reference precise types instead of
// `unknown` without making the UI template module a contracts module.

/**
 * Memory metrics for GPU buffer pool (per-type).
 */
export interface GPUPoolTypeStats {
  allocations: number;
  reuses: number;
  evictions: number;
  activeBuffers: number;
  pooledBuffers: number;
}

/**
 * Memory metrics for GPU buffer pool (aggregated + per-type breakdown).
 */
export interface GPUPoolStats {
  allocations: number;
  reuses: number;
  evictions: number;
  capacityGrowths: number;
  activeBuffers: number;
  pooledBuffers: number;
  byType: {
    points: GPUPoolTypeStats;
    lines: GPUPoolTypeStats;
    gsplats: GPUPoolTypeStats;
  };
}

/**
 * Memory metrics for data accumulators.
 */
export interface AccumulatorStats {
  capacity: number;
  allocations: number;
  growthEvents: number;
  memoryMB: number;
}

/**
 * Combined memory metrics for the Memory tab.
 */
export interface MemoryMetrics {
  gpuPool: GPUPoolStats | null;
  accumulators: {
    points: AccumulatorStats | null;
    lines: AccumulatorStats | null;
    gsplats: AccumulatorStats | null;
  };
}
