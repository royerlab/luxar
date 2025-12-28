/**
 * Point spatial index-based data loader for efficient nD points loading.
 *
 * This loader uses the point spatial index built by the Python compiler to load
 * only the points that are visible in the current nD slice. It ensures
 * all attributes are loaded with the same point ranges for proper alignment.
 */

import * as zarr from 'zarrita';
import { get, slice } from 'zarrita';
import * as THREE from 'three';
import { log, Modules, LogEmoji } from '../utils/log';
import {
  DataLoader,
  ViewState,
  PointsData,
  LoaderConfig,
  PointRange,
  SceneNode,
  PositionArray,
  ColorArray,
  ScalarArray,
} from './data-loader-types';
import {
  loadChunkSpatialIndex,
  queryChunksForView,
  chunkIndicesToRanges,
  mergePointRanges,
  type ChunkSpatialIndex,
} from './chunk-spatial-index';
import {
  calculateEffectiveRadii,
  calculateSpatialQueryTolerance,
  shouldApplyEffectiveRadius,
  type EffectiveRadiusConfig,
} from './effective-radius-calculator';
import type {
  MonitorEvent,
  MonitorEventListener,
  LoaderMonitor,
  LoaderMetrics,
  QueryInfo,
} from '../ui/data-monitor-types';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from './array-decoder';
import type { ZarrSceneAttrs } from '../types/zarr';
import { PointsDataAccumulator } from './data-accumulator';
import { config as appConfig } from '../config';
import { getWorkerPool } from '../workers/worker-pool';

/**
 * Loader implementation that uses spatial indices for efficient nD queries.
 *
 * Key features:
 * - Queries spatial index to find visible point ranges
 * - Loads all attributes with identical ranges (fixes alignment bug)
 * - Projects nD points to 3D display space
 * - Real-time monitoring and performance tracking
 */
export class PointSpatialIndexLoader implements DataLoader, LoaderMonitor {
  private chunkIndex: ChunkSpatialIndex | null = null;
  // Total points count for datasets without chunk-based index (simple fallback)
  private totalPointsNoIndex: number = 0;
  private _effectiveRadiusConfig: EffectiveRadiusConfig | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private initPromise: Promise<void> | null = null;
  private initLock = false;
  private arrays: {
    positions?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    radii?: zarr.Array<zarr.DataType, zarr.Readable>;
    sharpness?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

  // Array decoder for handling encoded arrays
  private decoder: ArrayDecoder;

  // Data accumulator for object pooling (Phase 1 optimization)
  private _accumulator: PointsDataAccumulator | null = null;

  // Monitoring
  private eventListeners = new Set<MonitorEventListener>();
  private metrics: LoaderMetrics;
  private activeQueries = new Map<string, QueryInfo>();
  private lastQueryCells = 0;
  private zarrStore: zarr.Readable | null = null;

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    _config: LoaderConfig = {},
    refRegistry?: ArrayRefRegistry,
    zarrStore?: zarr.Readable
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.decoder = new ArrayDecoder(refRegistry || new ArrayRefRegistry());
    this.zarrStore = zarrStore || null;

    // Initialize metrics
    this.metrics = {
      type: 'point-spatial-index',
      path: node.path,
      queries: 0,
      loads: 0,
      evictions: 0,
      errors: 0,
      pointsLoaded: 0,
      bytesLoaded: 0,
      datasetSize: 0, // Will be set after index is loaded
      visiblePoints: 0, // Updated on each query
      avgQueryTime: 0,
      avgLoadTime: 0,
      memoryUsed: 0,
      memoryLimit: 0,
    };
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    // Load chunk-based spatial index (NEW: replaces grid-based index)
    try {
      this.chunkIndex = await loadChunkSpatialIndex(this.zarrLocation, this.node.attrs);

      if (!this.chunkIndex) {
        // For 3D datasets without Morton ordering, fall back to loading all points
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `No chunk-based index for ${this.node.path} - will load all points`
        );

        // For 3D datasets, we need to get the actual number of points from the positions array
        // Open the positions array to get its shape
        try {
          const positionsArray = await zarr.open(this.zarrLocation.resolve('positions'), {
            kind: 'array',
          });
          // The shape is [n_points, 3] for 3D data
          this.totalPointsNoIndex = positionsArray.shape[0];

          // For encoded arrays (like array_ref), shape may be [0, k] but original_shape has the real count
          if (this.totalPointsNoIndex === 0) {
            const attrs = positionsArray.attrs as unknown as ArrayMetadata;
            if (attrs.encoding?.original_shape && attrs.encoding.original_shape.length > 0) {
              this.totalPointsNoIndex = attrs.encoding.original_shape[0];
              log.query(
                Modules.SPATIAL_INDEX_LOADER,
                `Detected ${this.totalPointsNoIndex} points from encoding.original_shape for ${this.node.path}`
              );
            }
          }

          if (this.totalPointsNoIndex > 0) {
            log.query(
              Modules.SPATIAL_INDEX_LOADER,
              `Detected ${this.totalPointsNoIndex} points in 3D dataset`
            );
          }

          // Check if this is an nD dataset (ndim > 3) without spatial indexing
          // This is inefficient because ALL points must be loaded for each view
          const ndim = positionsArray.shape[1] || 3;
          if (ndim > 3) {
            log.warning(
              Modules.SPATIAL_INDEX_LOADER,
              `⚠️ nD dataset (${ndim}D) without spatial index for ${this.node.path}. ` +
                `This is inefficient: ALL ${this.totalPointsNoIndex} points will be loaded for every view. ` +
                'Enable spatial ordering in Python with enable_spatial_ordering=True for efficient nD slicing.'
            );
          }
        } catch {
          log.warning(
            Modules.SPATIAL_INDEX_LOADER,
            `Could not determine point count for ${this.node.path}, using 0`
          );
          this.totalPointsNoIndex = 0;
        }
      } else {
        // Chunk index loaded successfully
        log.query(
          Modules.SPATIAL_INDEX_LOADER,
          `Chunk index loaded: ${this.chunkIndex.metadata.total_chunks} chunks, ${this.chunkIndex.metadata.total_points} points`
        );
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `  Ordering: ${this.chunkIndex.metadata.ordering}, ${this.chunkIndex.metadata.ndim}D space`
        );
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `  Ordering dims: [${this.chunkIndex.metadata.ordering_dims.join(', ')}], Slice dims: [${this.chunkIndex.metadata.slice_dims.join(', ')}]`
        );
      }

      // Load spatial extension configuration from scene_dimensions (derived from root attributes)
      // This determines which dimensions points physically extend through vs categorical dimensions
      const spatialExtendDims = await this.loadSpatialExtendDimsFromSceneDimensions();
      if (spatialExtendDims) {
        this._effectiveRadiusConfig = {
          spatialExtendDims: spatialExtendDims,
          maxRadius: this.node.attrs.max_radius || appConfig.dataLoading.spatial.defaultMaxRadius,
        };

        // Log which dimensions are spatial
        const spatialDims = spatialExtendDims
          .map((isSpatial: boolean, idx: number) => (isSpatial ? idx : null))
          .filter((idx: number | null) => idx !== null);

        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Spatial extension enabled for dimensions: [${spatialDims.join(', ')}]`
        );
      }
    } catch (error) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Failed to load spatial index for ${this.node.path}:`,
        error
      );
      throw error;
    }

    // Open arrays for later access
    try {
      this.arrays.positions = await zarr.open(this.zarrLocation.resolve('positions'), {
        kind: 'array',
      });
    } catch (e) {
      log.error(Modules.SPATIAL_INDEX_LOADER, 'Failed to open positions array:', e);
      throw e;
    }

    // Try to open optional arrays - these may not exist and that's OK
    try {
      this.arrays.colors = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
    } catch (e: any) {
      // Colors are optional - only log if it's not a 404
      if (!e.message?.includes('404') && !e.message?.includes('Not Found')) {
        log.info(Modules.SPATIAL_INDEX_LOADER, 'No colors array found (using default colors)');
      }
    }

    try {
      this.arrays.radii = await zarr.open(this.zarrLocation.resolve('radii'), { kind: 'array' });
    } catch (e: any) {
      // Radii are optional - only log if it's not a 404
      if (!e.message?.includes('404') && !e.message?.includes('Not Found')) {
        log.info(Modules.SPATIAL_INDEX_LOADER, 'No radii array found (using default radii)');
      }
    }

    try {
      this.arrays.sharpness = await zarr.open(this.zarrLocation.resolve('sharpness'), {
        kind: 'array',
      });
    } catch (e: any) {
      // Sharpness is optional - only log if it's not a 404
      if (!e.message?.includes('404') && !e.message?.includes('Not Found')) {
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          'No sharpness array found (using default sharpness)'
        );
      }
    }

    // Initialize data accumulator for object pooling (Phase 1 optimization)
    // NOTE: Infrastructure-only for Phase 1. Full hot path integration deferred to Phase 2.
    // See src/data/DATA_ACCUMULATOR_STATUS.md for details.
    if (appConfig.dataLoading.performance.useAccumulators) {
      const totalPoints = this.chunkIndex?.metadata.total_points || this.totalPointsNoIndex || 0;
      const ndim = this.chunkIndex?.metadata.ndim || this.arrays.positions?.shape[1] || 3;
      const initialCapacity = Math.min(
        appConfig.dataLoading.performance.initialAccumulatorCapacity,
        Math.max(1024, Math.ceil(totalPoints / 10)) // At least 1024, or ~10% of total
      );

      this._accumulator = new PointsDataAccumulator(initialCapacity, ndim, totalPoints);

      if (appConfig.dataLoading.performance.enablePerformanceMonitoring) {
        const stats = this._accumulator.getStats();
        log.info(
          Modules.DATA_ACCUMULATOR,
          `Initialized PointsDataAccumulator for ${this.node.path}: ` +
            `capacity=${stats.capacity}, ndim=${ndim}, totalPoints=${totalPoints} ` +
            '(infrastructure-only, hot path integration in Phase 2)'
        );
      }
    }
  }

  /**
   * Load points data for the given view state
   */
  async loadPoints(viewState: ViewState): Promise<PointsData> {
    const startTime = Date.now();
    const queryId = `${this.node.path}-${startTime}`;

    try {
      // Prevent race conditions during initialization with atomic check-and-set
      if (!this.initPromise && !this.initLock) {
        this.initLock = true;
        this.initPromise = this.initialize().finally(() => {
          this.initLock = false;
        });
      }

      // Wait for initialization to complete
      if (this.initPromise) {
        await this.initPromise;
      }

      // Check if loader is properly initialized (chunk index OR fallback with total points count)
      if (!this.chunkIndex && this.totalPointsNoIndex === 0) {
        // Zero points is still valid - it just means we have no points to load
        log.warning(
          Modules.SPATIAL_INDEX_LOADER,
          `Loader initialized with no chunk index and 0 points for ${this.node.path}`
        );
      }

      if (!this.arrays.positions) {
        throw new Error('Loader not properly initialized: positions array not loaded');
      }

      // Query spatial index for visible ranges (Phase 2: async for worker support)
      const ranges = await this.queryVisibleRanges(viewState);

      // Emit query event
      const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
      this.lastQueryCells = ranges.length;
      this.metrics.queries++;
      this.metrics.visiblePoints = totalPoints; // Track current visible points (non-cumulative)

      this.emitEvent({
        type: 'query',
        loader: 'point-spatial-index',
        timestamp: Date.now(),
        data: {
          path: this.node.path,
          ranges,
          cells: ranges.length,
          points: totalPoints,
          queryPosition: viewState.slicePosition,
          queryTolerance: viewState.tolerance,
        },
      });

      // Track active query
      this.activeQueries.set(queryId, {
        id: queryId,
        loader: 'point-spatial-index',
        path: this.node.path,
        startTime,
        status: 'loading',
        cells: ranges.length,
        points: totalPoints,
        ranges,
      });

      if (ranges.length === 0) {
        // No visible points - return empty dataset
        this.activeQueries.delete(queryId);
        return this.createEmptyPoints(viewState);
      }

      // Load all arrays with the SAME ranges (critical for alignment!)
      // Load sequentially to prevent browser resource exhaustion (ERR_INSUFFICIENT_RESOURCES)
      // This is especially important for large datasets with many chunks
      log.info(
        LogEmoji.LOAD,
        Modules.SPATIAL_INDEX_LOADER,
        `Loading positions for ${ranges.length} ranges`
      );
      const positions = await this.loadRanges('positions', ranges);

      const colors = this.arrays.colors
        ? (log.info(
            LogEmoji.LOAD,
            Modules.SPATIAL_INDEX_LOADER,
            `Loading colors for ${ranges.length} ranges`
          ),
          await this.loadRanges('colors', ranges))
        : null;

      const radii = this.arrays.radii
        ? (log.info(
            LogEmoji.LOAD,
            Modules.SPATIAL_INDEX_LOADER,
            `Loading radii for ${ranges.length} ranges`
          ),
          await this.loadRanges('radii', ranges))
        : null;

      const sharpness = this.arrays.sharpness
        ? (log.info(
            LogEmoji.LOAD,
            Modules.SPATIAL_INDEX_LOADER,
            `Loading sharpness for ${ranges.length} ranges`
          ),
          await this.loadRanges('sharpness', ranges))
        : null;

      // Update query status
      const query = this.activeQueries.get(queryId);
      if (query) {
        query.status = 'complete';
        query.endTime = Date.now();
      }

      // Update metrics
      const queryTime = Date.now() - startTime;
      this.metrics.avgQueryTime =
        (this.metrics.avgQueryTime * (this.metrics.queries - 1) + queryTime) / this.metrics.queries;

      // Phase 1 Deep Integration: Prepare accumulator buffers if enabled
      let targetBuffers: {
        positions3D: Float32Array;
        colors: ColorArray;
        radii: ScalarArray;
        sharpness: ScalarArray;
      } | null = null;

      if (this._accumulator && appConfig.dataLoading.performance.useAccumulators) {
        const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

        // Ensure capacity and initialize types
        this._accumulator.ensureCapacity(totalPoints);

        // Initialize types if not already done
        if (!this._accumulator['types']) {
          this._accumulator.fill(0, {
            positions: new Float32Array(3),
            colors: colors
              ? (colors.subarray(0, Math.min(3, colors.length)) as ColorArray)
              : undefined,
            radii: radii
              ? (radii.subarray(0, Math.min(1, radii.length)) as ScalarArray)
              : undefined,
            sharpness: sharpness
              ? (sharpness.subarray(0, Math.min(1, sharpness.length)) as ScalarArray)
              : undefined,
          });
        }

        // Get references to accumulator buffers (ZERO allocations!)
        targetBuffers = {
          positions3D: this._accumulator['positionBuffer'] as Float32Array,
          colors: this._accumulator['colorBuffer'] as ColorArray,
          radii: this._accumulator['radiiBuffer'] as ScalarArray,
          sharpness: this._accumulator['sharpnessBuffer'] as ScalarArray,
        };

        // Copy source colors/sharpness to accumulator buffers (needed for filtering later)
        if (colors) {
          if (colors instanceof Uint8Array && targetBuffers.colors instanceof Uint8Array) {
            (targetBuffers.colors as Uint8Array).set(colors);
          } else if (colors instanceof Uint16Array && targetBuffers.colors instanceof Uint16Array) {
            (targetBuffers.colors as Uint16Array).set(colors);
          } else if (
            colors instanceof Float32Array &&
            targetBuffers.colors instanceof Float32Array
          ) {
            (targetBuffers.colors as Float32Array).set(colors as Float32Array);
          }
        }

        if (sharpness) {
          if (sharpness instanceof Uint8Array && targetBuffers.sharpness instanceof Uint8Array) {
            (targetBuffers.sharpness as Uint8Array).set(sharpness);
          } else if (
            sharpness instanceof Float32Array &&
            targetBuffers.sharpness instanceof Float32Array
          ) {
            (targetBuffers.sharpness as Float32Array).set(sharpness as Float32Array);
          }
        }
      }

      // Project to 3D display space (ZERO allocations when targetBuffers provided!)
      const result = this.projectTo3D(
        positions,
        colors,
        radii,
        sharpness,
        viewState,
        ranges,
        targetBuffers
      );

      // Clean up completed query
      this.activeQueries.delete(queryId);

      return result;
    } catch (error) {
      // Handle errors
      this.metrics.errors++;

      // Update query status
      const query = this.activeQueries.get(queryId);
      if (query) {
        query.status = 'error';
        query.error = String(error);
      }

      this.emitEvent({
        type: 'error',
        loader: 'point-spatial-index',
        timestamp: Date.now(),
        data: {
          path: this.node.path,
          error: String(error),
        },
      });

      this.activeQueries.delete(queryId);
      throw error;
    }
  }

  /**
   * Update view for new position (more efficient than full reload)
   */
  async updateView(viewState: ViewState): Promise<PointsData> {
    // For now, just reload everything
    // TODO: Implement incremental updates
    return this.loadPoints(viewState);
  }

  /**
   * Query spatial index for visible point ranges
   * Phase 2: Made async to support worker-based queries
   */
  private async queryVisibleRanges(viewState: ViewState): Promise<PointRange[]> {
    // Check if this node has extend_to_all dimensions
    const extendDims = this.node.attrs.extend_to_all || [];

    if (extendDims.length > 0) {
      // DEFENSIVE CHECK: Warn if dimensions not available for extend_to_all
      if (!viewState.dimensions?.metadata || viewState.dimensions.metadata.length === 0) {
        log.warning(
          Modules.SPATIAL_INDEX_LOADER,
          `extend_to_all=[${extendDims.join(', ')}] specified for ${this.node.path} but ` +
            'viewState.dimensions.metadata is undefined. extend_to_all will not work. ' +
            'Ensure scene dimensions are initialized before loading nodes.'
        );
      }

      // Check if we're navigating through an extended dimension
      const currentNonDisplayedDims =
        viewState.dimensions?.metadata
          ?.filter((_meta, idx) => !viewState.displayDims.includes(idx))
          ?.map((meta) => meta.name)
          ?.filter((name) => name) || [];

      const isExtending = extendDims.some((edim) => currentNonDisplayedDims.includes(edim));

      if (isExtending) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SPATIAL_INDEX_LOADER,
          `Extending ${this.node.path} visibility across: ${extendDims.join(', ')}`
        );
        // Return all points for extended dimensions
        const totalPoints =
          this.node.attrs.n_points ||
          this.chunkIndex?.metadata.total_points ||
          this.totalPointsNoIndex ||
          0;
        return [{ start: 0, end: totalPoints }];
      }
    }

    const { slicePosition, tolerance } = viewState;

    // Use max radius from node attributes if available
    const maxRadius = this.node.attrs.max_radius || appConfig.dataLoading.spatial.defaultMaxRadius;

    // Get full dimension count from index metadata or default to 3D
    const fullDim = this.chunkIndex?.metadata.ndim || 3;

    // Build tolerance array based on spatial extension configuration
    let queryTolerance: number[];

    if (this._effectiveRadiusConfig) {
      // Use spatial-aware tolerance calculation
      queryTolerance = calculateSpatialQueryTolerance(
        viewState,
        this._effectiveRadiusConfig,
        fullDim
      );

      // Debug logging for tolerance values
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Query tolerance (with discrete awareness): [${queryTolerance.map((t) => t.toFixed(3)).join(', ')}]`
      );
    } else {
      // Fallback to old behavior - uniform tolerance for all non-displayed dims
      queryTolerance = new Array(fullDim).fill(0);
      const displayedDims = viewState.displayDims;

      for (let d = 0; d < fullDim; d++) {
        if (displayedDims.includes(d)) {
          // Displayed dimensions need INFINITE tolerance for chunk-based queries
          // We want to see ALL points regardless of their position in displayed dims
          queryTolerance[d] = 1e10;
        } else {
          // Non-displayed dims use explicit tolerance or maxRadius
          queryTolerance[d] = tolerance[d] || maxRadius;
        }
      }
    }

    // Fill in missing slice positions with defaults
    const querySlicePos = new Array(fullDim).fill(0);
    for (let d = 0; d < fullDim && d < slicePosition.length; d++) {
      querySlicePos[d] = slicePosition[d] ?? 0;
    }

    log.query(Modules.SPATIAL_INDEX_LOADER, 'Querying spatial index:');
    log.info(Modules.SPATIAL_INDEX_LOADER, `  Full dimensions: ${fullDim}`);
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `  Query position: [${querySlicePos.map((p) => p.toFixed(2)).join(', ')}]`
    );

    // Query chunk-based spatial index
    let ranges: PointRange[];

    if (this.chunkIndex) {
      // Use chunk-based queries (worker or main thread based on config)
      let chunkIndices: number[];

      if (appConfig.dataLoading.performance.useWebWorkers) {
        // Phase 2: Use worker for spatial queries (offloads CPU work)
        try {
          const worker = await getWorkerPool().getWorker();
          const result = await worker.querySpatialIndex({
            chunkBounds: this.chunkIndex.chunkBounds,
            slicePosition: new Float32Array(querySlicePos),
            tolerance: new Float32Array(queryTolerance),
            numChunks: this.chunkIndex.metadata.total_chunks,
            ndim: fullDim,
          });
          chunkIndices = Array.from(result);
        } catch (error) {
          log.error(Modules.SPATIAL_INDEX_LOADER, 'Worker query failed, using main thread:', error);
          // Fallback to main thread
          chunkIndices = queryChunksForView(this.chunkIndex, querySlicePos, queryTolerance);
        }
      } else {
        // Main thread query
        chunkIndices = queryChunksForView(this.chunkIndex, querySlicePos, queryTolerance);
      }

      ranges = chunkIndicesToRanges(
        chunkIndices,
        this.chunkIndex.metadata.chunk_size,
        this.chunkIndex.metadata.total_points
      );

      const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
      log.query(
        Modules.SPATIAL_INDEX_LOADER,
        `Chunk query: ${chunkIndices.length} chunks → ${ranges.length} ranges → ${totalPoints} points`
      );
    } else {
      // No chunk index - load all points (fallback for 3D datasets without Morton ordering)
      const totalPoints: number = (this.node.attrs.n_points ||
        this.totalPointsNoIndex ||
        0) as number;
      ranges = [{ start: 0, end: totalPoints }];
      log.query(Modules.SPATIAL_INDEX_LOADER, `No index: loading all ${totalPoints} points`);
    }

    // Merge adjacent ranges for more efficient loading
    const merged = mergePointRanges(ranges);

    if (merged.length !== ranges.length) {
      const totalPoints = merged.reduce((sum, r) => sum + (r.end - r.start), 0);
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Merged ${ranges.length} ranges → ${merged.length} continuous ranges (${totalPoints} points)`
      );
    }

    return merged;
  }

  /**
   * Load data for specific point ranges
   */
  /**
   * Allocate output buffer for decoded data.
   * For encoded arrays, always use Float32Array (decoder output).
   * For direct arrays, match the source dtype.
   */
  private allocateOutputBuffer(
    totalElements: number,
    isEncoded: boolean,
    dtype: string
  ): Float32Array | Uint8Array | Uint16Array | Float16Array {
    // CRITICAL: For encoded arrays, ALWAYS use Float32Array because the decoder
    // always returns Float32Array (it dequantizes uint8/uint16 to float32).
    if (isEncoded) {
      return new Float32Array(totalElements);
    }

    // For direct arrays, preserve native type
    if (dtype === 'uint8' || dtype === '|u1') {
      return new Uint8Array(totalElements);
    }
    if (dtype === 'uint16' || dtype === '<u2' || dtype === '>u2') {
      return new Uint16Array(totalElements);
    }
    if (dtype === 'float16' || dtype === '<f2' || dtype === '>f2') {
      // Float16 with fallback
      if (typeof (globalThis as any).Float16Array !== 'undefined') {
        return new (globalThis as any).Float16Array(totalElements);
      }
      log.warning(Modules.SPATIAL_INDEX_LOADER, 'Float16Array not supported, using Float32Array');
    }
    return new Float32Array(totalElements);
  }

  /**
   * Load broadcasted array (single value replicated to all points).
   */
  private async loadBroadcastedRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    arrayName: string,
    output: Float32Array,
    totalPoints: number,
    actualElementsPerPoint: number
  ): Promise<void> {
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Broadcasted array ${arrayName}: replicating single value to ${totalPoints} points`
    );

    const fullData = await get(array);
    const broadcastValue = fullData.data as Float32Array | Uint8Array | Uint16Array;

    // Replicate to all points
    for (let i = 0; i < totalPoints; i++) {
      for (let j = 0; j < actualElementsPerPoint; j++) {
        output[i * actualElementsPerPoint + j] = broadcastValue[j] || broadcastValue[0];
      }
    }
  }

  /**
   * Load quantized array ranges and dequantize.
   */
  private async loadQuantizedRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    arrayName: string,
    attrs: ArrayMetadata,
    ranges: PointRange[],
    output: Float32Array,
    totalPoints: number
  ): Promise<number> {
    const quantMetadata = ArrayDecoder.getQuantizationMetadata(attrs);
    if (!quantMetadata) {
      throw new Error(`Quantization metadata missing for ${arrayName}`);
    }

    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Quantized range loading: ${arrayName}, ${ranges.length} ranges (${ArrayDecoder.getEncodingMode(attrs)}, ${totalPoints} values)`
    );

    const shape = array.shape;

    // PARALLEL FETCH: Load all chunks simultaneously (I/O parallelism)
    const chunkDataPromises = ranges.map((range) => {
      const sliceSpec: zarr.Slice[] =
        shape.length === 2
          ? [slice(range.start, range.end), slice(null)]
          : [slice(range.start, range.end)];
      return get(array, sliceSpec);
    });

    const chunks = await Promise.all(chunkDataPromises);

    // SEQUENTIAL DECODE: Process chunks in order (maintains correct destOffset)
    let destOffset = 0;

    for (const chunkData of chunks) {
      const quantizedData = chunkData.data as Uint8Array | Uint16Array;
      const dequantized = this.decoder.dequantizeRange(quantizedData, quantMetadata);

      output.set(dequantized, destOffset);
      destOffset += dequantized.length;
    }

    return destOffset;
  }

  /**
   * Load LUT-encoded array ranges and decode with lookup table.
   */
  private async loadLUTRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    arrayName: string,
    attrs: ArrayMetadata,
    ranges: PointRange[],
    output: Float32Array,
    totalPoints: number,
    totalElements: number
  ): Promise<number> {
    const lutMetadata = ArrayDecoder.getLUTMetadata(attrs);
    if (!lutMetadata) {
      throw new Error(`LUT metadata missing for ${arrayName}`);
    }

    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `LUT range loading: ${arrayName}, ${ranges.length} ranges (${totalPoints} indices → ${totalElements} elements)`
    );

    const shape = array.shape;

    // PARALLEL FETCH: Load all index chunks simultaneously (I/O parallelism)
    const chunkDataPromises = ranges.map((range) => {
      const sliceSpec: zarr.Slice[] =
        shape.length === 2
          ? [slice(range.start, range.end), slice(null)]
          : [slice(range.start, range.end)];
      return get(array, sliceSpec);
    });

    const chunks = await Promise.all(chunkDataPromises);

    // SEQUENTIAL DECODE: Process chunks in order (maintains correct destOffset)
    let destOffset = 0;

    for (const chunkData of chunks) {
      const indices = chunkData.data as Float32Array | Uint8Array | Uint16Array;
      const decoded = this.decoder.decodeLUTIndices(indices, lutMetadata);

      output.set(decoded, destOffset);
      destOffset += decoded.length;
    }

    return destOffset;
  }

  /**
   * Load array reference by resolving target and applying appropriate strategy.
   */
  private async loadArrayRefRanges(
    _array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    arrayName: string,
    attrs: ArrayMetadata,
    ranges: PointRange[],
    output: Float32Array,
    totalPoints: number,
    actualElementsPerPoint: number
  ): Promise<number> {
    const targetPath = attrs.encoding!.target!;
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Array ref: ${arrayName} → ${targetPath} (optimized range loading)`
    );

    // Resolve target array
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    const zarrRootLoc = zarr.root(storeToUse);
    const targetLoc = zarrRootLoc.resolve(targetPath);
    const targetArray = await zarr.open(targetLoc, { kind: 'array' });
    const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

    let destOffset = 0;

    // Dispatch based on target encoding
    if (ArrayDecoder.isQuantizedEncoding(targetAttrs)) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Array ref target is quantized (${ArrayDecoder.getEncodingMode(targetAttrs)})`
      );
      destOffset = await this.loadQuantizedRanges(
        targetArray,
        arrayName,
        targetAttrs,
        ranges,
        output,
        totalPoints
      );
    } else if (ArrayDecoder.isLUTEncoded(targetAttrs)) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Array ref target is LUT (${ArrayDecoder.getEncodingMode(targetAttrs)})`
      );
      destOffset = await this.loadLUTRanges(
        targetArray,
        arrayName,
        targetAttrs,
        ranges,
        output,
        totalPoints,
        totalPoints * actualElementsPerPoint
      );
    } else if (ArrayDecoder.isBroadcasted(targetAttrs)) {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'Array ref target is broadcasted');
      await this.loadBroadcastedRanges(
        targetArray,
        arrayName,
        output,
        totalPoints,
        actualElementsPerPoint
      );
      destOffset = output.length;
    } else {
      // Direct or unknown encoding
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Array ref target is direct/unknown (${ArrayDecoder.getEncodingMode(targetAttrs)}) - full decode`
      );

      const decoded = await this.decoder.decode(
        targetArray,
        targetAttrs,
        totalPoints * actualElementsPerPoint,
        zarrRootLoc
      );

      for (const range of ranges) {
        const rangeSize = (range.end - range.start) * actualElementsPerPoint;
        const srcOffset = range.start * actualElementsPerPoint;
        output.set(decoded.subarray(srcOffset, srcOffset + rangeSize), destOffset);
        destOffset += rangeSize;
      }
    }

    return destOffset;
  }

  /**
   * Load encoded array (generic fallback path).
   */
  private async loadGenericEncodedRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    arrayName: string,
    attrs: ArrayMetadata,
    ranges: PointRange[],
    output: Float32Array,
    totalElements: number,
    actualElementsPerPoint: number
  ): Promise<number> {
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `Decoding ${arrayName} (${ArrayDecoder.getEncodingMode(attrs)} - generic path)`
    );

    const storeToUse = this.zarrStore || this.zarrLocation.store;
    const zarrRootLoc = zarr.root(storeToUse);
    const decoded = await this.decoder.decode(array, attrs, totalElements, zarrRootLoc);

    let destOffset = 0;
    for (const range of ranges) {
      const rangeSize = (range.end - range.start) * actualElementsPerPoint;
      const srcOffset = range.start * actualElementsPerPoint;
      output.set(decoded.subarray(srcOffset, srcOffset + rangeSize), destOffset);
      destOffset += rangeSize;
    }

    return destOffset;
  }

  /**
   * Load direct (non-encoded) array ranges.
   */
  private async loadDirectRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    ranges: PointRange[],
    output: Float32Array | Uint8Array | Uint16Array | Float16Array
  ): Promise<number> {
    const shape = array.shape;

    // PARALLEL FETCH: Load all chunks simultaneously (I/O parallelism)
    const chunkDataPromises = ranges.map((range) => {
      const sliceSpec: zarr.Slice[] =
        shape.length === 2
          ? [slice(range.start, range.end), slice(null)]
          : [slice(range.start, range.end)];
      return get(array, sliceSpec);
    });

    const chunks = await Promise.all(chunkDataPromises);

    // SEQUENTIAL WRITE: Process chunks in order (maintains correct destOffset)
    let destOffset = 0;

    for (const chunkData of chunks) {
      const data = chunkData.data as Float32Array | Uint8Array | Uint16Array | Float16Array;

      output.set(data, destOffset);
      destOffset += data.length;
    }

    return destOffset;
  }

  /**
   * Update metrics and emit load event after successful load.
   */
  private recordLoadMetrics(arrayName: string, totalPoints: number, output: ArrayBufferView): void {
    const loadTime =
      Date.now() - (this.activeQueries.values().next().value?.startTime || Date.now());
    const bytes = output.byteLength;

    this.metrics.loads++;
    this.metrics.pointsLoaded += totalPoints;
    this.metrics.bytesLoaded += bytes;
    this.metrics.avgLoadTime =
      (this.metrics.avgLoadTime * (this.metrics.loads - 1) + loadTime) / this.metrics.loads;

    this.emitEvent({
      type: 'load',
      loader: 'point-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        arrayName,
        points: totalPoints,
        memory: bytes,
        latency: loadTime,
      },
    });
  }

  /**
   * Load array data for specified point ranges.
   *
   * This method handles multiple encoding strategies:
   * - Broadcasted: Load once, replicate
   * - Quantized: Load ranges, dequantize
   * - LUT: Load indices, decode with lookup table
   * - Array Reference: Resolve target, dispatch based on target encoding
   * - Generic Encoded: Decode full array, extract ranges
   * - Direct: Load ranges directly
   *
   * @param arrayName - Name of the array to load (positions, colors, radii, sharpness)
   * @param ranges - Point ranges to load
   * @returns Typed array with loaded data, or null if array doesn't exist
   */
  private async loadRanges(
    arrayName: string,
    ranges: PointRange[]
  ): Promise<Float32Array | Uint8Array | Uint16Array | Float16Array | null> {
    const array = this.arrays[arrayName as keyof typeof this.arrays];
    if (!array) return null;

    log.load(Modules.SPATIAL_INDEX_LOADER, `Loading ${arrayName} for ${ranges.length} ranges`);

    // Analyze array metadata
    const attrs = array.attrs as unknown as ArrayMetadata;
    const isEncoded = ArrayDecoder.isEncoded(attrs);
    const shape = array.shape;
    const elementsPerPoint = shape.length === 2 ? shape[1] : 1;
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    // Determine actual elements per point (may differ for encoded arrays)
    const actualElementsPerPoint =
      isEncoded && attrs.encoding?.original_shape?.[1]
        ? attrs.encoding.original_shape[1]
        : elementsPerPoint;

    const totalElements = totalPoints * actualElementsPerPoint;

    // Sanity check
    if (totalElements > 1000000000 || totalElements < 0) {
      throw new Error(
        `Invalid array size: ${totalElements} (points=${totalPoints}, elements/point=${actualElementsPerPoint})`
      );
    }

    // Allocate output buffer
    const output = this.allocateOutputBuffer(totalElements, isEncoded, array.dtype);

    // Dispatch to appropriate loading strategy
    const isLUTEncoded = ArrayDecoder.isLUTEncoded(attrs);
    const isBroadcasted = ArrayDecoder.isBroadcasted(attrs);
    const isQuantized = ArrayDecoder.isQuantizedEncoding(attrs);
    const isArrayRef = ArrayDecoder.isArrayRef(attrs);

    // Dispatch to appropriate loading strategy based on encoding type
    if (isBroadcasted) {
      await this.loadBroadcastedRanges(
        array,
        arrayName,
        output as Float32Array,
        totalPoints,
        actualElementsPerPoint
      );
    } else if (isQuantized) {
      await this.loadQuantizedRanges(
        array,
        arrayName,
        attrs,
        ranges,
        output as Float32Array,
        totalPoints
      );
    } else if (isLUTEncoded) {
      await this.loadLUTRanges(
        array,
        arrayName,
        attrs,
        ranges,
        output as Float32Array,
        totalPoints,
        totalElements
      );
    } else if (isArrayRef) {
      await this.loadArrayRefRanges(
        array,
        arrayName,
        attrs,
        ranges,
        output as Float32Array,
        totalPoints,
        actualElementsPerPoint
      );
    } else if (isEncoded) {
      await this.loadGenericEncodedRanges(
        array,
        arrayName,
        attrs,
        ranges,
        output as Float32Array,
        totalElements,
        actualElementsPerPoint
      );
    } else {
      await this.loadDirectRanges(array, ranges, output);
    }

    // Update metrics and emit load event
    this.recordLoadMetrics(arrayName, totalPoints, output);

    // Return in native format for efficiency - geometry can handle both
    return output;
  }

  /**
   * Project nD points to 3D display space
   *
   * @param targetBuffers - Optional accumulator buffers for zero-allocation operation
   *                        When provided, writes directly to buffers (deep integration)
   *                        When null, allocates new arrays (fallback path)
   */
  private projectTo3D(
    positions: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    colors: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    radii: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    sharpness: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    viewState: ViewState,
    ranges: PointRange[],
    targetBuffers?: {
      positions3D: Float32Array;
      colors: ColorArray;
      radii: ScalarArray;
      sharpness: ScalarArray;
    } | null
  ): PointsData {
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    if (!positions) {
      throw new Error('Positions data is required for points');
    }

    // CRITICAL: Calculate ndim from actual positions array, not chunk index metadata.
    const ndim =
      totalPoints > 0
        ? Math.round(positions.length / totalPoints)
        : this.chunkIndex?.metadata.ndim || 3;

    // Validate the calculation
    if (totalPoints > 0 && positions.length !== totalPoints * ndim) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Position data size mismatch: ${positions.length} elements for ${totalPoints} points ` +
          `doesn't divide evenly (calculated ndim=${ndim}). This may indicate encoding metadata issues.`
      );
    }

    let numPoints = totalPoints;

    // Phase 1 Deep Integration: Use target buffers if provided (ZERO allocations!)
    const { displayDims } = viewState;

    // Use target buffer or allocate new (zero-allocation when targetBuffers provided)
    let positions3D = targetBuffers ? targetBuffers.positions3D : new Float32Array(numPoints * 3);

    // Extract 3D positions from nD data (write directly to buffer)
    for (let i = 0; i < numPoints; i++) {
      // Extract displayed dimensions
      for (let j = 0; j < Math.min(3, displayDims.length); j++) {
        const dimIdx = displayDims[j];
        positions3D[i * 3 + j] = positions[i * ndim + dimIdx];
      }
      // Fill remaining with zeros
      for (let j = displayDims.length; j < 3; j++) {
        positions3D[i * 3 + j] = 0;
      }
    }

    // Calculate bounds
    const bounds = new THREE.Box3();
    const point = new THREE.Vector3();
    for (let i = 0; i < numPoints; i++) {
      point.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
      bounds.expandByPoint(point);
    }

    // Calculate effective radii if configuration exists and radii are provided
    let finalRadii: Float32Array | Uint8Array | undefined;
    let usedEffectiveRadius = false;

    if (radii) {
      // Use target buffer or allocate (zero-allocation when targetBuffers provided)
      if (targetBuffers && targetBuffers.radii) {
        // Deep integration: Write directly to accumulator radii buffer
        if (radii instanceof Uint8Array && targetBuffers.radii instanceof Uint8Array) {
          (targetBuffers.radii as Uint8Array).set(radii);
          finalRadii = targetBuffers.radii as Uint8Array;
        } else if (radii instanceof Float32Array && targetBuffers.radii instanceof Float32Array) {
          (targetBuffers.radii as Float32Array).set(radii as Float32Array);
          finalRadii = targetBuffers.radii as Float32Array;
        } else {
          // Type mismatch (rare): fallback to conversion
          const float32Radii = radii instanceof Float32Array ? radii : new Float32Array(radii);
          (targetBuffers.radii as Float32Array).set(float32Radii);
          finalRadii = targetBuffers.radii as Float32Array;
        }
      } else {
        // Fallback: Allocate if needed
        finalRadii = radii instanceof Float32Array ? radii : new Float32Array(radii);
      }

      // Normalize uint8 radii to world units before effective radius calculation
      let effectiveRadiusConfig = this._effectiveRadiusConfig;
      if (finalRadii instanceof Uint8Array) {
        // Convert Uint8 to Float32 for effective radius calculation
        const float32Radii = new Float32Array(finalRadii.length);
        for (let i = 0; i < finalRadii.length; i++) {
          float32Radii[i] = finalRadii[i] / 255.0;
        }
        // Write to target buffer or use temp array
        if (targetBuffers && targetBuffers.radii instanceof Float32Array) {
          (targetBuffers.radii as Float32Array).set(float32Radii);
          finalRadii = targetBuffers.radii as Float32Array;
        } else {
          finalRadii = float32Radii;
        }

        // Scale max_radius for effective radius calculation
        if (effectiveRadiusConfig) {
          effectiveRadiusConfig = {
            ...effectiveRadiusConfig,
            maxRadius: effectiveRadiusConfig.maxRadius / 255.0,
          };
        }
      }

      if (effectiveRadiusConfig && finalRadii instanceof Float32Array) {
        // Check if we should apply effective radius
        if (shouldApplyEffectiveRadius(effectiveRadiusConfig, viewState.displayDims, true)) {
          const effectiveRadii = calculateEffectiveRadii(
            positions,
            finalRadii,
            viewState,
            effectiveRadiusConfig,
            ndim
          );

          // Write result to target buffer (if using) or replace
          if (targetBuffers && targetBuffers.radii instanceof Float32Array) {
            (targetBuffers.radii as Float32Array).set(effectiveRadii);
            finalRadii = targetBuffers.radii as Float32Array;
          } else {
            finalRadii = effectiveRadii;
          }
          usedEffectiveRadius = true;
        }
      }
    }

    // Filter out zero-radius points to avoid sending them to GPU
    // This significantly improves performance for nD slicing
    // IMPORTANT: Only filter when we actually calculated effective radii
    if (usedEffectiveRadius && finalRadii) {
      const threshold = 0.0001; // Small threshold for floating point precision
      const validIndices: number[] = [];

      // Find indices of points with non-zero radius
      for (let i = 0; i < numPoints; i++) {
        if (finalRadii[i] > threshold) {
          validIndices.push(i);
        }
      }

      const filteredCount = validIndices.length;

      // Only filter if we're actually removing points AND we have valid points left
      if (filteredCount < numPoints && filteredCount > 0) {
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Filtering out ${numPoints - filteredCount} zero-radius points (keeping ${filteredCount})`
        );

        if (targetBuffers) {
          // Phase 1 Deep Integration: IN-PLACE compaction (ZERO allocations!)
          // Compact valid points to beginning of target buffers
          let writeIdx = 0;
          for (let i = 0; i < validIndices.length; i++) {
            const readIdx = validIndices[i];

            // Only copy if read index != write index (avoid redundant copy)
            if (writeIdx !== readIdx) {
              // Compact positions
              positions3D[writeIdx * 3] = positions3D[readIdx * 3];
              positions3D[writeIdx * 3 + 1] = positions3D[readIdx * 3 + 1];
              positions3D[writeIdx * 3 + 2] = positions3D[readIdx * 3 + 2];

              // Compact radii
              if (finalRadii) {
                if (finalRadii instanceof Float32Array) {
                  (finalRadii as Float32Array)[writeIdx] = (finalRadii as Float32Array)[readIdx];
                } else {
                  (finalRadii as Uint8Array)[writeIdx] = (finalRadii as Uint8Array)[readIdx];
                }
              }

              // Compact colors (type-preserving)
              if (colors && targetBuffers.colors) {
                if (colors instanceof Uint8Array && targetBuffers.colors instanceof Uint8Array) {
                  const cb = targetBuffers.colors as Uint8Array;
                  cb[writeIdx * 3] = cb[readIdx * 3];
                  cb[writeIdx * 3 + 1] = cb[readIdx * 3 + 1];
                  cb[writeIdx * 3 + 2] = cb[readIdx * 3 + 2];
                } else if (
                  colors instanceof Uint16Array &&
                  targetBuffers.colors instanceof Uint16Array
                ) {
                  const cb = targetBuffers.colors as Uint16Array;
                  cb[writeIdx * 3] = cb[readIdx * 3];
                  cb[writeIdx * 3 + 1] = cb[readIdx * 3 + 1];
                  cb[writeIdx * 3 + 2] = cb[readIdx * 3 + 2];
                } else if (
                  colors instanceof Float32Array &&
                  targetBuffers.colors instanceof Float32Array
                ) {
                  const cb = targetBuffers.colors as Float32Array;
                  cb[writeIdx * 3] = cb[readIdx * 3];
                  cb[writeIdx * 3 + 1] = cb[readIdx * 3 + 1];
                  cb[writeIdx * 3 + 2] = cb[readIdx * 3 + 2];
                }
              }

              // Compact sharpness (type-preserving)
              if (sharpness && targetBuffers.sharpness) {
                if (
                  sharpness instanceof Uint8Array &&
                  targetBuffers.sharpness instanceof Uint8Array
                ) {
                  (targetBuffers.sharpness as Uint8Array)[writeIdx] = (
                    targetBuffers.sharpness as Uint8Array
                  )[readIdx];
                } else if (
                  sharpness instanceof Float32Array &&
                  targetBuffers.sharpness instanceof Float32Array
                ) {
                  (targetBuffers.sharpness as Float32Array)[writeIdx] = (
                    targetBuffers.sharpness as Float32Array
                  )[readIdx];
                }
              }
            }

            writeIdx++;
          }

          // Update count to filtered count (arrays already compacted in-place!)
          numPoints = filteredCount;
        } else {
          // Fallback: Create filtered arrays (allocations when accumulator disabled)
          const filteredPositions3D = new Float32Array(filteredCount * 3);
          const filteredRadii = new Float32Array(filteredCount);

          // Filter colors if present
          let filteredColors: Float32Array | Uint8Array | Uint16Array | undefined;
          if (colors) {
            if (colors instanceof Float32Array) {
              filteredColors = new Float32Array(filteredCount * 3);
            } else if (colors instanceof Uint8Array) {
              filteredColors = new Uint8Array(filteredCount * 3);
            } else if (colors instanceof Uint16Array) {
              filteredColors = new Uint16Array(filteredCount * 3);
            }
          }

          // Filter sharpness if present
          let filteredSharpness: Float32Array | Uint8Array | Uint16Array | undefined;
          if (sharpness) {
            if (sharpness instanceof Float32Array) {
              filteredSharpness = new Float32Array(filteredCount);
            } else if (sharpness instanceof Uint8Array) {
              filteredSharpness = new Uint8Array(filteredCount);
            } else if (sharpness instanceof Uint16Array) {
              filteredSharpness = new Uint16Array(filteredCount);
            }
          }

          // Copy only valid points
          for (let i = 0; i < filteredCount; i++) {
            const srcIdx = validIndices[i];

            // Copy position (3 components)
            filteredPositions3D[i * 3] = positions3D[srcIdx * 3];
            filteredPositions3D[i * 3 + 1] = positions3D[srcIdx * 3 + 1];
            filteredPositions3D[i * 3 + 2] = positions3D[srcIdx * 3 + 2];

            // Copy radius
            filteredRadii[i] = finalRadii![srcIdx];

            // Copy colors if present (3 components)
            if (colors && filteredColors) {
              filteredColors[i * 3] = colors[srcIdx * 3];
              filteredColors[i * 3 + 1] = colors[srcIdx * 3 + 1];
              filteredColors[i * 3 + 2] = colors[srcIdx * 3 + 2];
            }

            // Copy sharpness if present
            if (sharpness && filteredSharpness) {
              filteredSharpness[i] = sharpness[srcIdx];
            }
          }

          // Replace arrays with filtered versions
          positions3D = filteredPositions3D;
          finalRadii = filteredRadii;
          colors = filteredColors || colors;
          sharpness = filteredSharpness || sharpness;

          // Update point count
          numPoints = filteredCount;
        }

        // Recalculate bounds for filtered points only
        bounds.makeEmpty();
        for (let i = 0; i < filteredCount; i++) {
          point.set(positions3D[i * 3], positions3D[i * 3 + 1], positions3D[i * 3 + 2]);
          bounds.expandByPoint(point);
        }
      } else if (filteredCount === 0) {
        // All points were filtered out - this is correct behavior for points outside the hyperplane!
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `All ${numPoints} points have zero effective radius - no points visible at this slice`
        );
        // Return empty points - this is the correct behavior
        return this.createEmptyPoints(viewState);
      }
    }

    // Phase 1 Deep Integration: Return from accumulator when using target buffers
    if (targetBuffers && this._accumulator) {
      // Data is already in accumulator buffers (written directly during processing)
      // Just update metadata and return (ZERO allocations!)
      this._accumulator.updateMetadata({
        bounds,
        usedSpatialIndex: true,
      });

      // Return from accumulator (subarrays are views into accumulator buffers)
      return this._accumulator.getData(numPoints);
    }

    // Fallback: Create new PointsData object (when accumulator disabled)
    const dtypes = {
      positions: this.node.attrs.position_dtype as string | undefined,
      colors: this.node.attrs.color_dtype as string | undefined,
      radii: this.node.attrs.radius_dtype as string | undefined,
      sharpness: this.node.attrs.sharpness_dtype as string | undefined,
    };

    return {
      positions: positions3D as PositionArray,
      colors: colors as ColorArray | undefined,
      radii: finalRadii as ScalarArray | undefined,
      sharpness: sharpness as ScalarArray | undefined,
      metadata: {
        totalPoints: this.node.attrs.n_points || totalPoints,
        loadedPoints: numPoints,
        bounds,
        ndim,
        usedSpatialIndex: true,
        usedEffectiveRadius,
        dtypes,
      },
    };
  }

  /**
   * Create empty points when no points are visible
   */
  private createEmptyPoints(viewState: ViewState): PointsData {
    // Note: viewState parameter kept for future use when we might need
    // dimension-aware empty points (e.g., different ndim based on view)
    void viewState; // Explicitly mark as intentionally unused for now

    // Get dtype metadata from node attributes
    const dtypes = {
      positions: this.node.attrs.position_dtype as string | undefined,
      colors: this.node.attrs.color_dtype as string | undefined,
      radii: this.node.attrs.radius_dtype as string | undefined,
      sharpness: this.node.attrs.sharpness_dtype as string | undefined,
    };

    return {
      positions: new Float32Array(0) as PositionArray,
      metadata: {
        totalPoints: this.node.attrs.n_points || 0,
        loadedPoints: 0,
        bounds: new THREE.Box3(),
        ndim: this.chunkIndex?.metadata.ndim || 3,
        usedSpatialIndex: true,
        dtypes,
      },
    };
  }

  // LoaderMonitor implementation

  /**
   * Add event listener
   */
  addEventListener(listener: MonitorEventListener): void {
    this.eventListeners.add(listener);
  }

  /**
   * Remove event listener
   */
  removeEventListener(listener: MonitorEventListener): void {
    this.eventListeners.delete(listener);
  }

  /**
   * Get current metrics
   */
  getMetrics(): LoaderMetrics {
    // Update spatial index metrics if available
    if (this.chunkIndex) {
      // NEW: Chunk-based index metrics
      const totalChunks = this.chunkIndex.metadata.total_chunks;
      const avgChunksPerQuery =
        this.metrics.queries > 0 ? this.lastQueryCells / this.metrics.queries : 0;

      this.metrics.spatialIndex = {
        // Chunk-based index metrics
        gridShape: [totalChunks], // Use total chunks as "grid" size
        gridOrigin: [0],
        cellSize: [this.chunkIndex.metadata.chunk_size],
        occupiedCells: totalChunks, // All chunks are "occupied"
        totalCells: totalChunks,
        avgCellsPerQuery: avgChunksPerQuery,
        avgPointsPerCell: totalChunks > 0 ? this.metrics.pointsLoaded / totalChunks : 0,
        queryEfficiency: avgChunksPerQuery / Math.max(totalChunks, 1),
        rangesInCache: 0, // L0 RangeCache removed
      };

      // Set dataset size from chunk index metadata
      this.metrics.datasetSize = this.chunkIndex.metadata.total_points;
    } else if (this.totalPointsNoIndex > 0) {
      // No chunk index but we have point count from the positions array
      this.metrics.datasetSize = this.totalPointsNoIndex;
    }

    return { ...this.metrics };
  }

  /**
   * Get active queries
   */
  getActiveQueries(): QueryInfo[] {
    return Array.from(this.activeQueries.values());
  }

  /**
   * Emit event to listeners
   */
  private emitEvent(event: MonitorEvent): void {
    for (const listener of this.eventListeners) {
      try {
        listener(event);
      } catch (error) {
        log.error(Modules.SPATIAL_INDEX_LOADER, 'Error in event listener:', error);
      }
    }
  }

  /**
   * Load spatial_extend_dims from scene_dimensions stored in root zarr attributes.
   *
   * This derives the spatial extension flags from the scene_dimensions which is
   * the single source of truth for dimension configuration. Each dimension has
   * a `spatial` flag indicating whether points physically extend through it.
   *
   * For displayed dimensions, we always treat them as spatial (they're in the view plane).
   * For non-displayed dimensions:
   *   - spatial=true: Points extend as hyperspheres (use effective radius)
   *   - spatial=false: Categorical/discrete dimension (exact match required)
   *
   * @returns Array of booleans indicating spatial extension per dimension, or null if unavailable
   */
  private async loadSpatialExtendDimsFromSceneDimensions(): Promise<boolean[] | null> {
    if (!this.zarrStore) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        'No zarr store available - cannot derive spatial_extend_dims'
      );
      return null;
    }

    try {
      // Open the root group to get scene_dimensions from root attributes
      const rootGroup = await zarr.open(this.zarrStore, { kind: 'group' });
      const rootAttrs = rootGroup.attrs as unknown as ZarrSceneAttrs;

      if (!rootAttrs?.scene_dimensions?.dimensions) {
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          'No scene_dimensions in root attributes - spatial filtering disabled'
        );
        return null;
      }

      const dimensions = rootAttrs.scene_dimensions.dimensions;

      // Derive spatial_extend_dims: for each dimension, check its spatial flag
      // Default behavior: displayed dimensions are always spatial (they're the viewing plane)
      // Non-displayed dimensions use their explicit spatial flag (defaulting to false if unset)
      const spatialExtendDims = dimensions.map((dim) => {
        if (dim.display) {
          // Displayed dimensions are always treated as spatial (points are in the plane)
          return true;
        }
        // Non-displayed: use explicit spatial flag, default false (categorical)
        return dim.spatial ?? false;
      });

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Derived spatial_extend_dims from scene_dimensions: [${spatialExtendDims.join(', ')}]`
      );

      return spatialExtendDims;
    } catch (error) {
      log.warning(
        Modules.SPATIAL_INDEX_LOADER,
        'Failed to load scene_dimensions from root:',
        error
      );
      return null;
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.chunkIndex = null;
    this.arrays = {};
    this.initPromise = null;
    this.eventListeners.clear();

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
  }
}
