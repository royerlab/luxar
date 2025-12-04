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
  CacheStats,
  PointRange,
  SceneNode,
  PositionArray,
  ColorArray,
  ScalarArray,
} from './data-loader-types';
import { RangeCache } from './range-cache';
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
import { config } from '../config';
import type {
  MonitorEvent,
  MonitorEventListener,
  LoaderMonitor,
  LoaderMetrics,
  QueryInfo,
} from '../ui/data-monitor-types';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from './array-decoder';
import type { ZarrSceneAttrs } from '../types/zarr';

/**
 * Loader implementation that uses spatial indices for efficient nD queries.
 *
 * Key features:
 * - Queries spatial index to find visible point ranges
 * - Loads all attributes with identical ranges (fixes alignment bug)
 * - Caches data at the range level for efficiency
 * - Projects nD points to 3D display space
 * - Real-time monitoring and performance tracking
 */
export class PointSpatialIndexLoader implements DataLoader, LoaderMonitor {
  private cache: RangeCache;
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

  // Monitoring
  private eventListeners = new Set<MonitorEventListener>();
  private metrics: LoaderMetrics;
  private activeQueries = new Map<string, QueryInfo>();
  private lastQueryCells = 0;
  private zarrStore: zarr.Readable | null = null;

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    config: LoaderConfig = {},
    refRegistry?: ArrayRefRegistry,
    zarrStore?: zarr.Readable
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.cache = new RangeCache(config);
    this.decoder = new ArrayDecoder(refRegistry || new ArrayRefRegistry());
    this.zarrStore = zarrStore || null;

    // Initialize metrics
    this.metrics = {
      type: 'point-spatial-index',
      path: node.path,
      queries: 0,
      loads: 0,
      cacheHits: 0,
      cacheMisses: 0,
      evictions: 0,
      errors: 0,
      pointsLoaded: 0,
      bytesLoaded: 0,
      avgQueryTime: 0,
      avgLoadTime: 0,
      cacheHitRate: 0,
      memoryUsed: 0,
      memoryLimit: this.cache.getMemoryInfo().max,
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
          // The shape is [num_points, 3] for 3D data
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
          maxRadius: this.node.attrs.max_radius || config.dataLoading.spatial.defaultMaxRadius,
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

      // Query spatial index for visible ranges
      const ranges = this.queryVisibleRanges(viewState);

      // Emit query event
      const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
      this.lastQueryCells = ranges.length;
      this.metrics.queries++;

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

      // Project to 3D display space
      const result = this.projectTo3D(positions, colors, radii, sharpness, viewState, ranges);

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
   */
  private queryVisibleRanges(viewState: ViewState): PointRange[] {
    // Check if this node has broadcast dimensions
    const broadcastDims = this.node.attrs.broadcast_dims || [];

    if (broadcastDims.length > 0) {
      // Check if we're navigating through a broadcast dimension
      const currentNonDisplayedDims =
        viewState.dimensions?.metadata
          ?.filter((_meta, idx) => !viewState.displayDims.includes(idx))
          ?.map((meta) => meta.name)
          ?.filter((name) => name) || [];

      const isBroadcasting = broadcastDims.some((bdim) => currentNonDisplayedDims.includes(bdim));

      if (isBroadcasting) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SPATIAL_INDEX_LOADER,
          `Broadcasting ${this.node.path} across: ${broadcastDims.join(', ')}`
        );
        // Return all points for broadcast dimensions
        const totalPoints =
          this.node.attrs.num_points ||
          this.chunkIndex?.metadata.total_points ||
          this.totalPointsNoIndex ||
          0;
        return [{ start: 0, end: totalPoints }];
      }
    }

    const { slicePosition, tolerance } = viewState;

    // Use max radius from node attributes if available
    const maxRadius = this.node.attrs.max_radius || config.dataLoading.spatial.defaultMaxRadius;

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
      // Use chunk-based queries
      const chunkIndices = queryChunksForView(this.chunkIndex, querySlicePos, queryTolerance);
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
        this.node.attrs.num_points ||
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
  private async loadRanges(
    arrayName: string,
    ranges: PointRange[]
  ): Promise<Float32Array | Uint8Array | Uint16Array | Float16Array | null> {
    const array = this.arrays[arrayName as keyof typeof this.arrays];
    if (!array) return null;

    const arrayPath = `${this.node.path}/${arrayName}`;

    // Check cache first
    const cached = this.cache.get(arrayPath, ranges);
    if (cached) {
      log.custom(LogEmoji.CACHE, Modules.SPATIAL_INDEX_LOADER, `Cache hit for ${arrayName}`);
      this.metrics.cacheHits++;
      this.updateCacheHitRate();

      this.emitEvent({
        type: 'cache-hit',
        loader: 'point-spatial-index',
        timestamp: Date.now(),
        data: {
          path: this.node.path,
          arrayName,
          cacheKey: arrayPath,
        },
      });

      return cached;
    }

    log.load(Modules.SPATIAL_INDEX_LOADER, `Loading ${arrayName} for ${ranges.length} ranges`);
    this.metrics.cacheMisses++;
    this.updateCacheHitRate();

    this.emitEvent({
      type: 'cache-miss',
      loader: 'point-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        arrayName,
        ranges,
      },
    });

    // Check if array is encoded (requires full array load + decode)
    // IMPORTANT: Check this FIRST before trying to calculate sizes from local array shape
    const attrs = array.attrs as unknown as ArrayMetadata;
    const isEncoded = ArrayDecoder.isEncoded(attrs);

    // Determine dimensions per element from the array shape
    // NOTE: For array_ref encoded arrays, the local shape may be misleading
    const shape = array.shape;
    const elementsPerPoint = shape.length === 2 ? shape[1] : 1;

    // Calculate total size needed from ranges
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    // For encoded arrays (especially array_ref), don't pre-allocate based on local shape
    // The decoder will return the correct size and we'll extract ranges afterward
    const dtype = array.dtype;
    let output: Float32Array | Uint8Array | Uint16Array | Float16Array;

    // Determine the actual elements per point from encoding metadata for encoded arrays
    const actualElementsPerPoint =
      isEncoded && attrs.encoding?.original_shape?.[1]
        ? attrs.encoding.original_shape[1]
        : elementsPerPoint;

    const totalElements = totalPoints * actualElementsPerPoint;

    // Sanity check for array allocation size
    if (totalElements > 1000000000 || totalElements < 0) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Invalid totalElements calculation: ${totalElements} ` +
          `(totalPoints=${totalPoints}, elementsPerPoint=${actualElementsPerPoint}). ` +
          `Array shape: ${JSON.stringify(shape)}, isEncoded: ${isEncoded}`
      );
      throw new Error(
        `Invalid array size: ${totalElements}. This may indicate corrupted metadata.`
      );
    }

    if (dtype === 'uint8' || (dtype as string) === '|u1') {
      output = new Uint8Array(totalElements);
    } else if (dtype === 'uint16' || (dtype as string) === '<u2' || (dtype as string) === '>u2') {
      output = new Uint16Array(totalElements);
    } else if (
      (dtype as string) === 'float16' ||
      (dtype as string) === '<f2' ||
      (dtype as string) === '>f2'
    ) {
      // Float16Array is supported in modern browsers (2024+)
      // Use runtime detection for compatibility
      if (typeof (globalThis as any).Float16Array !== 'undefined') {
        output = new (globalThis as any).Float16Array(totalElements);
      } else {
        // Fallback to Float32 for older browsers
        log.warning(
          Modules.SPATIAL_INDEX_LOADER,
          'Float16Array not supported, using Float32Array as fallback'
        );
        output = new Float32Array(totalElements);
      }
    } else {
      output = new Float32Array(totalElements);
    }

    let destOffset = 0; // Declare here for both branches

    if (isEncoded) {
      // Encoded arrays: Load full array once and decode
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Decoding ${arrayName} (${ArrayDecoder.getEncodingMode(attrs)} mode)`
      );

      // Get root location for array_ref resolution
      // The decoder needs the zarr root to resolve target paths like "GroupA/Node1/positions"
      // Use the passed store (from SceneLoader) if available, otherwise fall back to location's store
      const storeToUse = this.zarrStore || this.zarrLocation.store;
      const zarrRootLoc = zarr.root(storeToUse);

      const decoded = await this.decoder.decode(array, attrs, totalElements, zarrRootLoc);

      // Extract ranges from decoded full array
      for (const range of ranges) {
        const rangeSize = (range.end - range.start) * elementsPerPoint;
        const srcOffset = range.start * elementsPerPoint;

        // Copy from decoded array
        (output as Float32Array).set(
          decoded.subarray(srcOffset, srcOffset + rangeSize),
          destOffset
        );
        destOffset += rangeSize;
      }
    } else {
      // Direct arrays: Load only needed ranges
      for (const range of ranges) {
        // Build slice specification
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)] // [points, dims]
            : [slice(range.start, range.end)]; // [points]

        // Load data from zarr
        const chunkData = await get(array, sliceSpec);
        const data = chunkData.data as Float32Array | Uint8Array | Uint16Array | Float16Array;

        // Copy to output buffer
        output.set(data, destOffset);
        destOffset += data.length;
      }
    }

    // Cache the result (only Float32Array for now to save memory)
    if (output instanceof Float32Array) {
      this.cache.set(arrayPath, ranges, output);
    }

    // Update metrics and emit load event
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

    // Return in native format for efficiency - geometry can handle both
    return output;
  }

  /**
   * Project nD points to 3D display space
   */
  private projectTo3D(
    positions: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    colors: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    radii: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    sharpness: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    viewState: ViewState,
    ranges: PointRange[]
  ): PointsData {
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    if (!positions) {
      throw new Error('Positions data is required for points');
    }

    // Get dimensionality from positions array - use full dimensions, not just indexed ones
    const ndim = this.chunkIndex?.metadata.ndim || 3;
    let numPoints = positions.length / ndim;

    if (numPoints !== totalPoints) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Point count mismatch: expected ${totalPoints}, got ${numPoints}`
      );
    }

    // Extract 3D positions from nD data
    const { displayDims } = viewState;
    let positions3D = new Float32Array(numPoints * 3);

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
    let finalRadii =
      radii instanceof Float32Array ? radii : radii ? new Float32Array(radii) : undefined;
    let usedEffectiveRadius = false;

    // Normalize uint8 radii to world units before effective radius calculation
    let effectiveRadiusConfig = this._effectiveRadiusConfig;
    if (finalRadii && radii instanceof Uint8Array) {
      // uint8 radii are stored as 0-255, need to scale to 0-1 (or actual world units)
      // This matches the radiusScale = 1.0 / 255.0 used in the shader
      for (let i = 0; i < finalRadii.length; i++) {
        finalRadii[i] = finalRadii[i] / 255.0;
      }
      // Also need to scale max_radius for effective radius calculation
      // Create a copy of the config to avoid modifying the original
      if (effectiveRadiusConfig) {
        effectiveRadiusConfig = {
          ...effectiveRadiusConfig,
          maxRadius: effectiveRadiusConfig.maxRadius / 255.0,
        };
      }
    }

    if (finalRadii && effectiveRadiusConfig) {
      // Check if we should apply effective radius
      if (shouldApplyEffectiveRadius(effectiveRadiusConfig, viewState.displayDims, true)) {
        finalRadii = calculateEffectiveRadii(
          positions, // Original nD positions (any typed array)
          finalRadii, // Now in world units
          viewState,
          effectiveRadiusConfig, // Config with scaled maxRadius
          ndim
        );
        usedEffectiveRadius = true;

        // Log statistics
        const origRadii =
          radii instanceof Float32Array ? radii : radii ? new Float32Array(radii) : null;
        if (origRadii && finalRadii) {
          let changed = 0;
          for (let i = 0; i < numPoints; i++) {
            if (Math.abs(finalRadii[i] - origRadii[i]) > 0.01) {
              changed++;
            }
          }
          if (changed > 0) {
            log.info(
              Modules.SPATIAL_INDEX_LOADER,
              `Effective radii: ${changed}/${numPoints} points changed`
            );
          }
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

        // Create filtered arrays
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
          filteredRadii[i] = finalRadii[srcIdx];

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

    // Get dtype metadata from node attributes
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
        totalPoints: this.node.attrs.num_points || totalPoints,
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
        totalPoints: this.node.attrs.num_points || 0,
        loadedPoints: 0,
        bounds: new THREE.Box3(),
        ndim: this.chunkIndex?.metadata.ndim || 3,
        usedSpatialIndex: true,
        dtypes,
      },
    };
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): CacheStats {
    return this.cache.getStats();
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
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
        rangesInCache: this.cache.getStats().numEntries,
      };
    }

    // Update memory usage
    this.metrics.memoryUsed = this.cache.getStats().totalMemory;

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
   * Update cache hit rate metric
   */
  private updateCacheHitRate(): void {
    const total = this.metrics.cacheHits + this.metrics.cacheMisses;
    if (total > 0) {
      this.metrics.cacheHitRate = (this.metrics.cacheHits / total) * 100;
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
    this.clearCache();
    this.chunkIndex = null;
    this.arrays = {};
    this.initPromise = null;
    this.eventListeners.clear();
  }
}
