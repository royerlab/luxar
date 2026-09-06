/**
 * Type definitions for the new Data Loading Monitor architecture.
 *
 * This module defines the interfaces and types for the event-driven
 * monitoring system that tracks spatial index queries, cache performance,
 * and loading statistics in real-time.
 */

import type { CacheValidationMode } from '../cache/types';
import type { GeometryTypeName, NodeKind, NodeTypeName } from './format-contract';

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
 * Event types emitted by data loaders.
 *
 * These three are the only types any loader actually emits: `query`
 * (the spatial-index loaders), `load` / `error` (the spatial facade).
 * The cache tiers report their own hit/miss/eviction counters directly
 * through the `CacheStatsProvider` snapshot rather than as monitor
 * events, so there is no `cache-hit` / `cache-miss` / `evict` /
 * `prefetch` event.
 */
export type MonitorEventType =
  | 'query' // Spatial index or range query initiated
  | 'load' // Data loaded from source
  | 'error'; // Loading or query error

/**
 * Loader types in the system — one per geometry type, named after the
 * loading STRATEGY rather than the geometry, because that is what the
 * telemetry means: the three `*-spatial-index` loaders answer per-slice
 * range queries, while `mesh-whole-node` fetches its node once and serves
 * every later view from memory. So a mesh loader reports `loads` and
 * `bytesLoaded` but no `queries` / `spatialIndex` — the honest shape, not a
 * gap (see `MeshWholeNodeLoader`).
 */
export type LoaderType =
  'point-spatial-index' | 'lines-spatial-index' | 'gsplats-spatial-index' | 'mesh-whole-node';

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
    /** Element count for the event (points / vertices-or-segments / splats / triangles). */
    elements?: number;
    cells?: number;
    latency?: number;
    memory?: number;
    error?: string;
    // Spatial-specific data
    queryPosition?: readonly number[];
    queryTolerance?: readonly number[];
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
  errors: number;
  // Performance metrics
  elementsLoaded: number; // Cumulative (for throughput calculation)
  bytesLoaded: number;
  // Dataset info
  visibleElements: number; // Currently visible elements: points / segments / splats / triangles (non-cumulative)
  avgQueryTime: number;
  avgLoadTime: number;
  // Memory usage
  memoryUsed: number;
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
  occupiedCells: number;
  totalCells: number;
  avgCellsPerQuery: number;
  avgElementsPerCell: number;
  queryEfficiency: number; // Points loaded / points in query region
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
  /** Element count the query matched (points / segments / splats / triangles). */
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

  // Update settings
  updateInterval: number; // ms between UI updates
  maxEvents: number; // Maximum events to keep in history

  // Feature flags
  showRecommendations: boolean;
  autoExpand: boolean; // Auto-expand on warnings

  // Performance
  enableProfiling: boolean;
}

/**
 * Aggregated statistics across all loaders
 */
export interface GlobalStats {
  totalLoaders: number;
  activeSpatialLoaders: number;
  totalElementsLoaded: number; // Cumulative elements loaded across all loaders (points/vertices/splats/triangles — throughput)
  // Resident memory across all loaders — sum of each loader's `memoryUsed`
  // (current accumulator allocation, set in the spatial-index loaders'
  // recordLoadMetrics). Drives the compact badge's memory figure.
  totalMemory: number;
  // Dataset metrics — one named pair per geometry type, each labelled with
  // that type's own element noun (points / segments / splats / triangles).
  // Named rather than keyed by type because each pair is rendered with its
  // own label, unit noun and DOM id; `metrics/global-stats.ts` is the single
  // place the kind-keyed `SceneGraphState` counters are projected onto them.
  // Dataset metrics - Points
  datasetSize: number; // Total points in all datasets
  visiblePoints: number; // Currently visible/rendered points (per-geometry quartet with visibleSegments / visibleSplats / visibleTriangles)
  // Dataset metrics - Lines
  datasetSegments: number; // Total segments in all line datasets
  visibleSegments: number; // Currently visible/rendered segments (for lines, typically equals total)
  // Dataset metrics - GSplats
  datasetSplats: number; // Total splats in all gsplats datasets
  visibleSplats: number; // Currently visible/rendered splats
  // Dataset metrics - Mesh
  datasetTriangles: number; // Total triangles (faces) in all mesh datasets
  visibleTriangles: number; // Currently visible/rendered triangles — triangles the active nD slice indexes
  /** Elements omitted by renderer capacity clamps across visible texture-backed nodes. */
  droppedElements: number;
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
  /**
   * Effective demand hit-rate across all cache tiers:
   * `(l0Hits + l1Hits + l2Hits) / (l0Hits + l1Hits + l2Hits + networkRequests)`.
   * Undefined when neither L0 provider nor `demand` counters are wired
   * (lets the UI distinguish "not wired up" from "0% hit rate").
   */
  effectiveDemandHitRate?: number;
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
    /** Number of ladder entries whose cached prefix is complete. */
    fullLadderCount?: number;
    /** Cached ladder counts keyed by `storedDepth/totalDepth`. */
    ladderDepthHistogram?: Record<string, number>;
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
    validationMode?: CacheValidationMode;
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
 * Node type for scene graph display, single-sourced from the cross-language
 * format contract (format-contract/contract.yaml → `node_types`).
 *
 * This was `NodeTypeName | 'mesh'` while `mesh` was a viewer-only forward
 * declaration; `mesh` is now in the contract, so the local extension is gone and
 * this is a plain alias. Keep it an alias rather than re-widening: a display type
 * the writer cannot emit has nothing to display.
 */
export type SceneGraphNodeType = NodeTypeName;

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
  /**
   * Number of triangles (for mesh nodes).
   *
   * Faces rather than vertices, because this trio counts the DRAWN PRIMITIVE per
   * type — note `lines` is counted by `segmentCount`, not `vertexCount`, for the
   * same reason. (The Python `Mesh.n_elements` counts vertices instead, since
   * there the primary element is whatever the per-element attribute arrays are
   * indexed by. The two conventions answer different questions.)
   *
   * Populated from the node's `n_faces` attr by `scene-graph-converter.ts` now that
   * the mesh loader has landed. Still optional: `elementCountOf` treats a missing
   * count as 0, so a store that omits the attr degrades to zero rather than NaN.
   */
  faceCount?: number;
  /** Number of visible splats after nD slicing (for gsplats nodes) */
  visibleSplatCount?: number;
  /**
   * Number of visible triangles after nD slicing (for mesh nodes) — the fourth
   * member of the visible-count family above.
   *
   * A mesh loads WHOLE, so this is not a streaming residency figure like its
   * siblings: it is how many of the node's faces the active nD slab indexes,
   * which is what `commit-mesh-geometry` puts in the index buffer. Pushed per
   * path by `monitor/visible-counts.ts` and merged in by
   * `SceneGraphModel.syncVisibleCountsIntoTree`.
   */
  visibleFaceCount?: number;
  /**
   * Specialized-group discriminant, set when the underlying scene-graph
   * node is a `kind=lod` (substitutive LOD) or `kind=partition` (BSP)
   * `Group`. Drives the tree's kind badge + icon. Mirrors `LayerInfo.kind`
   * in `ui/layers/layer-state.ts`. `NodeKind` is single-sourced from the
   * cross-language format contract (format-contract/contract.yaml).
   */
  kind?: NodeKind;
  /**
   * Resolved geometry `display_type` for a specialized group — the type the user
   * logically sees the group as. Absent for plain groups and leaves (use `type`
   * there).
   *
   * Spelled out rather than `GeometryTypeName`: this is the LOD/partition-CAPABLE
   * subset of the vocabulary — the UNION of `LODGroupMetadata.display_type` and
   * `PartitionGroupMetadata.display_type`, since one field carries both kinds. So
   * do not widen it when a geometry type is added — declare that type's `lod` /
   * `partition` capabilities in `types/geometry-capabilities`, and extend this
   * union only if one of them comes out `true`. `'mesh'` is here on both counts:
   * its `lod` and `partition` flags are now both true.
   */
  displayType?: 'points' | 'lines' | 'gsplats' | 'mesh';
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
 *
 * The on-disk `lod`/`partition` kinds come from the format contract's
 * `NodeKind`; `additive` is a viewer-only progressive-loading shape.
 */
export type LODNodeKind = NodeKind | 'additive';

/**
 * Live, per-node LOD / progressive-refinement / cache-residency state,
 * polled by the monitor each tick. Keyed by scene-graph path. All fields
 * beyond `kind` are optional — only the ones relevant to a node's kind are
 * populated.
 */
/** Why a progressive loader's next rung is held back (see `LODProgressState.held`). */
export type RefinementHoldReason = 'density' | 'budget';

export interface LODProgressState {
  kind: LODNodeKind;
  /** substitutive: total number of levels (child count). */
  levelCount?: number;
  /** substitutive: currently active level index (0-based, coarsest-first). */
  activeLevel?: number;
  /** substitutive: selector mode — `'auto'` or a locked level label. */
  selector?: string;
  /**
   * additive: rungs actually committed on screen; falls back to the loader's
   * cursor when no `committedLODCount` stamp exists.
   */
  loaded?: number;
  /** additive: total LOD levels available. */
  total?: number;
  /** additive: more LODs pending — refinement loop still running. */
  refining?: boolean;
  /**
   * additive: why the next rung is HELD rather than streaming, when it is.
   * `'density'`: the density guard's rung gate — at the current framing the
   * node already projects more elements per pixel than its cap; loads once
   * the camera moves in. `'budget'`: the residency ceiling — loading it would
   * exceed the in-memory budget; nothing more loads until memory frees.
   * Absent while a pending rung is genuinely downloading (or nothing is pending).
   */
  held?: RefinementHoldReason;
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
 * Live cross-node draw-order state for one data mesh, read off the THREE
 * material + object (`material.transparent`, `material.depthWrite`,
 * `mesh.renderOrder`). Surfaced per node in the scene-graph tree so a
 * compositing-order bug (a backdrop drawn after the content in front of it)
 * is visible without a renderer capture. Pure observability.
 */
export interface NodeDrawOrder {
  /** `'transparent'` (in the sorted set) or `'opaque'` (drawn depth-first). */
  bucket: 'opaque' | 'transparent';
  /** Whether the mesh writes depth (`material.depthWrite`). */
  depthWrite: boolean;
  /** Resolved `mesh.renderOrder` (compared ascending → lowest drawn first). */
  renderOrder: number;
  /**
   * Authored cross-layer draw order, when the layer states one
   * (`docs/guides/specs/LAYER_ORDER_SPEC.md`); `undefined` when it does
   * not. Unlike `renderOrder` this is camera-INDEPENDENT — it says *why* a
   * layer sits where it does rather than where it happens to sit this frame —
   * which is exactly what is hard to diagnose from the resolved integer alone.
   */
  layerOrder?: number;
}

/**
 * Provides a snapshot of live per-mesh draw-order state keyed by scene-graph
 * path (mesh `name`). Injected via `SceneLoaderMonitorPort.setDrawOrderProvider`
 * and polled on each tick — `renderOrder` is camera-dependent, so a live read
 * per tick keeps the panel honest as the view orbits. Implemented in the data
 * layer over the live THREE root group.
 */
export interface DrawOrderProvider {
  getDrawOrderStates(): Map<string, NodeDrawOrder>;
}

/**
 * Live projected-density state of one drawable node, as measured by the
 * density guard (`scene/projected-density.ts`). Distinct from the LOD chip:
 * that one says how much of the node is RESIDENT, this one says how much of
 * the resident data the shader actually DRAWS this frame. Pure observability.
 */
export interface NodeDensityState {
  /** Fraction of the resident elements drawn (1 = all; 1/2, 1/4, … when thinned). */
  keep: number;
  /** Resident elements per drawing-buffer pixel of the node's projected footprint (0 off-screen). */
  elementsPerPixel: number;
  /** Whether the node's blend mode sums energy — only such nodes are ever thinned. */
  blendable: boolean;
  onScreen: boolean;
}

/**
 * Per-path density snapshot for the scene-graph tree's lattice-glyph `1/K` density chip.
 * App-scoped (the guard outlives any one scene); wired by the init pipeline.
 */
export interface DensityProvider {
  getDensityStates(): Map<string, NodeDensityState>;
}

/**
 * Per-geometry-type counters, one entry per {@link GeometryTypeName}.
 *
 * Keyed by the contract vocabulary rather than written out as
 * `points…`/`lines…`/`gsplats…` triplets, so adding a geometry type extends
 * every counter at once (and fails to compile until the producers supply it)
 * instead of needing a field, an accumulator and a reader per counter.
 *
 * The *element* nouns (points / segments / splats) deliberately do NOT live
 * here: they belong to the display layer, which keeps per-type named fields on
 * {@link DataLoadingStats} because each is rendered with its own label and DOM
 * id. This shape is the aggregation model; the nouns are presentation.
 */
export type GeometryCounters = Record<GeometryTypeName, number>;

/**
 * Scene graph state for monitor
 */
export interface SceneGraphState {
  /** Root node of the scene graph */
  root: SceneGraphNode | null;
  /** Total number of nodes */
  totalNodes: number;
  /** Number of scene-graph nodes of each geometry type */
  nodesByType: GeometryCounters;
  /** Total elements of each geometry type across all nodes */
  totalByType: GeometryCounters;
  /** Currently visible elements per type (after nD clipping / progressive LOD) */
  visibleByType: GeometryCounters;
  /** Elements omitted by renderer capacity clamps across visible texture-backed nodes. */
  droppedElements: number;
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
      validationMode: CacheValidationMode;
      lastValidatedAt: number | null;
      unvalidatedExternalDataset: boolean;
      /**
       * S2: `true` when OPFS L2 storage is operational, or when the tier
       * was deliberately skipped (`?no-cache` / `?no-opfs` — no L2
       * expected). `false` only for UNREQUESTED degradation: L2 was
       * expected but could not be initialised, or the OPFS circuit
       * breaker disabled it after repeated timeouts. Older providers omit
       * the field entirely; consumers treat the absent case as "unknown /
       * assume available".
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
 * Geometry types that own GPU-pool buffers and element accumulators.
 *
 * Deliberately **not** `GeometryTypeName`. This is the subset of geometry types
 * whose data goes through the instanced-quad element-texture path, so it is what
 * keys {@link GPUPoolStats.byType} and {@link MemoryMetrics.accumulators}. A
 * geometry type that renders some other way has no entry in either record, and
 * iterating the full geometry vocabulary over them would index a key that does
 * not exist.
 *
 * The loops that walk those records iterate this const, so adding a key to one
 * record without adding it here (or vice versa) is a compile error rather than a
 * silently short iteration.
 */
export const POOLED_GEOMETRY_TYPES = ['points', 'lines', 'gsplats'] as const;
export type PooledGeometryType = (typeof POOLED_GEOMETRY_TYPES)[number];

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
  byType: Record<PooledGeometryType, GPUPoolTypeStats>;
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
  accumulators: Record<PooledGeometryType, AccumulatorStats | null>;
}

/**
 * What the monitor needs from one element accumulator. Registered per
 * {@link PooledGeometryType} via `DataLoadingMonitor.setAccumulatorProvider`.
 */
export interface AccumulatorProvider {
  getStats(): AccumulatorStats;
}
