/**
 * Spatial index-based data loader for efficient nD point cloud loading.
 *
 * This loader uses the spatial index built by the Python compiler to load
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
  PointCloudData,
  LoaderConfig,
  CacheStats,
  PointRange,
  SceneNode,
} from './data-loader-types';
import { RangeCache } from './range-cache';
import {
  loadSpatialIndex,
  querySpatialIndex,
  mergePointRanges,
  type SpatialIndex,
} from './spatial-index';
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
export class SpatialIndexLoader implements DataLoader, LoaderMonitor {
  private cache: RangeCache;
  private spatialIndex: SpatialIndex | null = null;
  private _effectiveRadiusConfig: EffectiveRadiusConfig | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private initPromise: Promise<void> | null = null;
  private arrays: {
    positions?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    radii?: zarr.Array<zarr.DataType, zarr.Readable>;
    sharpness?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

  // Monitoring
  private eventListeners = new Set<MonitorEventListener>();
  private metrics: LoaderMetrics;
  private activeQueries = new Map<string, QueryInfo>();
  private lastQueryCells = 0;

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    config: LoaderConfig = {}
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.cache = new RangeCache(config);

    // Initialize metrics
    this.metrics = {
      type: 'spatial-index',
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
      memoryLimit: config.maxMemoryMB ? config.maxMemoryMB * 1024 * 1024 : 500 * 1024 * 1024,
    };
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    // Load spatial index
    try {
      this.spatialIndex = await loadSpatialIndex(this.zarrLocation);

      if (!this.spatialIndex) {
        // For 3D datasets where all dimensions are displayed, create a dummy index
        // that will return all points
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `No spatial index found for ${this.node.path}, assuming 3D dataset`
        );

        // For 3D datasets, we need to get the actual number of points from the positions array
        // Open the positions array to get its shape
        let totalPoints = 0;
        try {
          const positionsArray = await zarr.open(this.zarrLocation.resolve('positions'), {
            kind: 'array',
          });
          // The shape is [num_points, 3] for 3D data
          totalPoints = positionsArray.shape[0];
          log.query(Modules.SPATIAL_INDEX_LOADER, `Detected ${totalPoints} points in 3D dataset`);
        } catch {
          log.warning(
            Modules.SPATIAL_INDEX_LOADER,
            `Could not determine point count for ${this.node.path}, using 0`
          );
          totalPoints = 0;
        }

        // Create a minimal valid spatial index that returns all points
        this.spatialIndex = {
          metadata: {
            grid_shape: [],
            grid_origin: [],
            cell_size: [],
            num_occupied: 0,
            dimensions: 0,
            indexed_dimensions: [],
            full_dimensions: 3,
            displayed_dimensions: [0, 1, 2],
            total_points: totalPoints,
            total_cells: 0,
            build_version: '0.1.0',
          },
          occupiedCells: new Uint32Array(0),
          cellRanges: new BigUint64Array(0),
        };
      } else if (this.spatialIndex.metadata.num_occupied > 0) {
        log.query(
          Modules.SPATIAL_INDEX_LOADER,
          `Initialized with ${this.spatialIndex.metadata.num_occupied} occupied cells`
        );
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `  Grid shape: [${this.spatialIndex.metadata.grid_shape.join(', ')}]`
        );
        // Detailed metadata only in development
        // @ts-ignore - process.env might not be available in all environments
        if (typeof process !== 'undefined' && process.env?.NODE_ENV === 'development') {
          log.info(
            Modules.SPATIAL_INDEX_LOADER,
            `  Indexed dimensions: ${this.spatialIndex.metadata.dimensions} (indices: [${this.spatialIndex.metadata.indexed_dimensions?.join(', ')}])`
          );
        }
      }

      // Load spatial extension configuration from node attributes
      if (
        this.node.attrs.spatial_extend_dims &&
        Array.isArray(this.node.attrs.spatial_extend_dims)
      ) {
        const spatialExtendDims = this.node.attrs.spatial_extend_dims as boolean[];

        // Note: discreteDims loading removed - non-spatial dimensions are always discrete
        // This is enforced on the Python side, so we don't need to check it here

        this._effectiveRadiusConfig = {
          spatialExtendDims: spatialExtendDims,
          maxRadius: this.node.attrs.max_radius || 0.1,
          // discreteDims is deprecated - kept for backwards compatibility only
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
   * Load point cloud data for the given view state
   */
  async loadPointCloud(viewState: ViewState): Promise<PointCloudData> {
    const startTime = Date.now();
    const queryId = `${this.node.path}-${startTime}`;

    try {
      // Prevent race conditions during initialization
      if (!this.initPromise) {
        this.initPromise = this.initialize();
      }
      await this.initPromise;

      if (!this.spatialIndex || !this.arrays.positions) {
        throw new Error('Loader not properly initialized');
      }

      // Query spatial index for visible ranges
      const ranges = this.queryVisibleRanges(viewState);

      // Emit query event
      const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
      this.lastQueryCells = ranges.length;
      this.metrics.queries++;

      this.emitEvent({
        type: 'query',
        loader: 'spatial-index',
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
        loader: 'spatial-index',
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
        return this.createEmptyPointCloud(viewState);
      }

      // Load all arrays with the SAME ranges (critical for alignment!)
      const [positions, colors, radii, sharpness] = await Promise.all([
        this.loadRanges('positions', ranges),
        this.arrays.colors ? this.loadRanges('colors', ranges) : null,
        this.arrays.radii ? this.loadRanges('radii', ranges) : null,
        this.arrays.sharpness ? this.loadRanges('sharpness', ranges) : null,
      ]);

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
        loader: 'spatial-index',
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
  async updateView(viewState: ViewState): Promise<PointCloudData> {
    // For now, just reload everything
    // TODO: Implement incremental updates
    return this.loadPointCloud(viewState);
  }

  /**
   * Query spatial index for visible point ranges
   */
  private queryVisibleRanges(viewState: ViewState): PointRange[] {
    if (!this.spatialIndex) {
      return [];
    }

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
          this.spatialIndex.metadata.total_points ||
          Number.MAX_SAFE_INTEGER;
        return [{ start: 0, end: totalPoints }];
      }
    }

    const { slicePosition, tolerance } = viewState;

    // Use max radius from node attributes if available
    const maxRadius = this.node.attrs.max_radius || 0.1;

    // The spatial index now only contains non-displayed dimensions
    // We need to build the query arrays using the full dimension count
    const fullDim =
      this.spatialIndex.metadata.full_dimensions || this.spatialIndex.metadata.dimensions;

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
      const displayedDims =
        this.spatialIndex.metadata.displayed_dimensions || viewState.displayDims;

      for (let d = 0; d < fullDim; d++) {
        if (!displayedDims.includes(d)) {
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
    // Debug logging - commented out for production
    // log.info(Modules.SPATIAL_INDEX_LOADER, `  Indexed dimensions: [${indexedDims.join(', ')}]`);
    // log.info(Modules.SPATIAL_INDEX_LOADER, `  Displayed dimensions: [${displayedDims.join(', ')}]`);
    // log.info(Modules.SPATIAL_INDEX_LOADER, `  ViewState.slicePosition (raw): [${slicePosition.join(', ')}]`);
    // log.info(Modules.SPATIAL_INDEX_LOADER, `  ViewState.tolerance (raw): [${tolerance.join(', ')}]`);
    // log.info(Modules.SPATIAL_INDEX_LOADER, `  Query Position: [${querySlicePos.map((v) => v.toFixed(1)).join(', ')}]`);
    // log.info(Modules.SPATIAL_INDEX_LOADER,
    //   `  Query Tolerance: [${queryTolerance.map((v) => (v > 1000 ? '∞' : v.toFixed(1))).join(', ')}]`
    // );
    // log.info(Modules.SPATIAL_INDEX_LOADER, `  Max radius: ${maxRadius}`);

    // Query spatial index
    const ranges = querySpatialIndex(this.spatialIndex, querySlicePos, queryTolerance);

    // Merge adjacent ranges for more efficient loading
    const merged = mergePointRanges(ranges);

    const totalPoints = merged.reduce((sum, r) => sum + (r.end - r.start), 0);
    log.query(
      Modules.SPATIAL_INDEX_LOADER,
      `Query result: ${ranges.length} cells → ${merged.length} ranges → ${totalPoints} points`
    );

    return merged;
  }

  /**
   * Load data for specific point ranges
   */
  private async loadRanges(
    arrayName: string,
    ranges: PointRange[]
  ): Promise<Float32Array | Uint8Array | null> {
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
        loader: 'spatial-index',
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
      loader: 'spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        arrayName,
        ranges,
      },
    });

    // Determine dimensions per element
    const shape = array.shape;
    const elementsPerPoint = shape.length === 2 ? shape[1] : 1;

    // Calculate total size needed
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const totalElements = totalPoints * elementsPerPoint;

    // Allocate output buffer - keep native type for efficiency
    const dtype = array.dtype;
    let output: Float32Array | Uint8Array;

    if (dtype === 'uint8' || (dtype as string) === '|u1') {
      output = new Uint8Array(totalElements);
    } else {
      output = new Float32Array(totalElements);
    }

    // Load each range
    let destOffset = 0;
    for (const range of ranges) {
      // Build slice specification
      const sliceSpec: zarr.Slice[] =
        shape.length === 2
          ? [slice(range.start, range.end), slice(null)] // [points, dims]
          : [slice(range.start, range.end)]; // [points]

      // Load data from zarr
      const chunkData = await get(array, sliceSpec);
      const data = chunkData.data as Float32Array | Uint8Array;

      // Copy to output buffer
      output.set(data, destOffset);
      destOffset += data.length;
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
      loader: 'spatial-index',
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
    positions: Float32Array | Uint8Array | null,
    colors: Float32Array | Uint8Array | null,
    radii: Float32Array | Uint8Array | null,
    sharpness: Float32Array | Uint8Array | null,
    viewState: ViewState,
    ranges: PointRange[]
  ): PointCloudData {
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    if (!positions) {
      throw new Error('Positions data is required for point cloud');
    }

    // Get dimensionality from positions array - use full dimensions, not just indexed ones
    const ndim =
      this.spatialIndex?.metadata.full_dimensions || this.spatialIndex?.metadata.dimensions || 3;
    const numPoints = positions.length / ndim;

    if (numPoints !== totalPoints) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Point count mismatch: expected ${totalPoints}, got ${numPoints}`
      );
    }

    // Extract 3D positions from nD data
    const { displayDims } = viewState;
    const positions3D = new Float32Array(numPoints * 3);

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

    if (finalRadii && this._effectiveRadiusConfig && positions instanceof Float32Array) {
      // Check if we should apply effective radius
      if (shouldApplyEffectiveRadius(this._effectiveRadiusConfig, viewState.displayDims, true)) {
        finalRadii = calculateEffectiveRadii(
          positions, // Original nD positions
          finalRadii,
          viewState,
          this._effectiveRadiusConfig,
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

    return {
      positions: positions3D,
      colors:
        colors instanceof Float32Array ? colors : colors ? new Float32Array(colors) : undefined,
      radii: finalRadii,
      sharpness:
        sharpness instanceof Float32Array
          ? sharpness
          : sharpness
            ? new Float32Array(sharpness)
            : undefined,
      metadata: {
        totalPoints: this.node.attrs.num_points || totalPoints,
        loadedPoints: numPoints,
        bounds,
        ndim,
        usedSpatialIndex: true,
        usedEffectiveRadius,
      },
    };
  }

  /**
   * Create empty point cloud when no points are visible
   */
  private createEmptyPointCloud(viewState: ViewState): PointCloudData {
    // Note: viewState parameter kept for future use when we might need
    // dimension-aware empty point clouds (e.g., different ndim based on view)
    void viewState; // Explicitly mark as intentionally unused for now

    return {
      positions: new Float32Array(0),
      metadata: {
        totalPoints: this.node.attrs.num_points || 0,
        loadedPoints: 0,
        bounds: new THREE.Box3(),
        ndim: this.spatialIndex?.metadata.dimensions || 3,
        usedSpatialIndex: true,
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
    if (this.spatialIndex) {
      const totalCells = this.spatialIndex.metadata.total_cells || 1;
      const occupiedCells = this.spatialIndex.metadata.num_occupied || 0;

      this.metrics.spatialIndex = {
        gridShape: this.spatialIndex.metadata.grid_shape,
        gridOrigin: this.spatialIndex.metadata.grid_origin,
        cellSize: this.spatialIndex.metadata.cell_size,
        occupiedCells,
        totalCells,
        avgCellsPerQuery: this.metrics.queries > 0 ? this.lastQueryCells / this.metrics.queries : 0,
        avgPointsPerCell: occupiedCells > 0 ? this.metrics.pointsLoaded / occupiedCells : 0,
        queryEfficiency: 0.8, // TODO: Calculate actual efficiency
        cellsInCache: this.cache.getStats().numEntries,
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
   * Clean up resources
   */
  dispose(): void {
    this.clearCache();
    this.spatialIndex = null;
    this.arrays = {};
    this.initPromise = null;
    this.eventListeners.clear();
  }
}
