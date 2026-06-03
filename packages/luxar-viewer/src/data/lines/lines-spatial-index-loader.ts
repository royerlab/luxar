/**
 * Lines spatial index-based data loader for efficient nD lines loading.
 *
 * This loader uses segment-first loading:
 * 1. Query segment chunks and load visible segments
 * 2. Derive required vertex chunks from segment indices and load vertices
 *
 * It also handles nD slicing with endpoint clipping for segments that
 * partially intersect the visible slice.
 *
 * @module data/lines-spatial-index-loader
 */

import * as zarr from '../zarr';
import { get, slice } from '../zarr';
import { log, Modules } from '../../utils/log';
import type {
  LinesMetadata,
  LoadedLinesData,
  LinesDataLoader,
  LinesViewState,
  SegmentRange,
} from '../../types/lines';
import type { SceneNode, PointRange } from '../data-loader-types';
import { ArrayRefRegistry, type ArrayMetadata } from '../array-decoder/decoder';
import {
  RangeLoader,
  SpatialQueryBuilder,
  mergeRanges,
  type LoadRange,
  getExpectedColorType,
  loadColorRanges,
  computeLoadLatency,
  recordLoadEvent,
  LoaderEventEmitter,
  OnceInit,
  warnExtendToAllNoDimensions,
  announceExtendToAllOnce,
} from '../loaders';
import type {
  LoaderMetrics,
  MonitorEvent,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { LinesDataAccumulator, type AccumulatorStats } from '../accumulators/lines';
import { config as appConfig } from '../../config';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import { DecompressedChunkCache } from '../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../cache/residency-probe';
import { ChunkPrefetcher } from '../../cache/chunk-prefetcher';
import {
  type LinesDualChunkIndex,
  loadLinesDualChunkIndex,
  registerLinesArrayBounds,
  computeVertexRangesFromIndices,
} from './chunk-index-loader';
import { createEmptyLinesData } from './projection';

/**
 * Lines data loader using spatial indices for efficient nD queries.
 *
 * Key features:
 * - Segment-first loading: segments → vertices
 * - Index remapping from global to local indices
 * - nD endpoint clipping for partial segment visibility
 * - Per-vertex attribute interpolation for clipped segments
 */
export class LinesSpatialIndexLoader implements LinesDataLoader {
  private chunkIndex: LinesDualChunkIndex | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private _onceInit = new OnceInit();
  private rangeLoader: RangeLoader;
  private zarrStore: zarr.Readable | null = null;

  // Data accumulator for object pooling.
  private _accumulator: LinesDataAccumulator | null = null;

  // L0 decompressed chunk cache (optional, avoids Blosc decompression on repeat access)
  private l0Cache: DecompressedChunkCache | null = null;

  // Active cache-residency probe for the in-flight demand load (see
  // updateViewWithResidency); null at all other times.
  private _activeProbe: ResidencyAccumulator | null = null;

  // Chunk prefetcher (optional, for registering array bounds to suppress 404s)
  private prefetcher: ChunkPrefetcher | null = null;

  // Suppress detail logs after first successful view update
  private _initialLoadDone = false;

  // LoaderMonitor surface — same shape as the points and gsplats facades.
  private readonly events = new LoaderEventEmitter();
  private readonly metrics: LoaderMetrics;
  private readonly activeQueries = new Map<string, QueryInfo>();
  private nextQueryId = 0;

  private arrays: {
    vertices?: zarr.Array<zarr.DataType, zarr.Readable>;
    segments?: zarr.Array<zarr.DataType, zarr.Readable>;
    widths?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    sharpness?: zarr.Array<zarr.DataType, zarr.Readable>;
    /** optional per-vertex scalars for colormap lookup. */
    scalars?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    refRegistry?: ArrayRefRegistry,
    zarrStore?: zarr.Readable,
    profiler?: UpdateProfiler,
    l0Cache?: DecompressedChunkCache,
    prefetcher?: ChunkPrefetcher
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.rangeLoader = new RangeLoader(refRegistry || new ArrayRefRegistry());
    this.zarrStore = zarrStore || null;
    this.l0Cache = l0Cache || null;
    this.prefetcher = prefetcher || null;
    // profiler parameter kept for API compatibility; session is passed directly to methods
    void profiler;

    this.metrics = {
      type: 'lines-spatial-index',
      path: node.path,
      queries: 0,
      loads: 0,
      evictions: 0,
      errors: 0,
      pointsLoaded: 0, // Shared loader metric; counts vertices for lines.
      bytesLoaded: 0,
      visiblePoints: 0, // counts visible vertices for lines
      avgQueryTime: 0,
      avgLoadTime: 0,
      memoryUsed: 0,
      memoryLimit: 0,
    };
  }

  /**
   * Thin wrapper around the shared `registerLinesArrayBounds` helper so
   * the call sites read more naturally than passing the prefetcher and
   * node path on every call.
   */
  private registerBounds(arrayName: string, array: zarr.Array<zarr.DataType, zarr.Readable>): void {
    registerLinesArrayBounds(this.prefetcher, this.node.path, arrayName, array);
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    const attrs = this.node.attrs as unknown as LinesMetadata;

    // Load dual spatial index (vertex + segment chunk bounds)
    try {
      this.chunkIndex = await this.loadDualChunkBounds(attrs);

      if (!this.chunkIndex) {
        log.info(
          Modules.LINES_LOADER,
          `No spatial index for Lines ${this.node.path} - will load all data`
        );
      } else {
        log.query(
          Modules.LINES_LOADER,
          `Lines index loaded: ${this.chunkIndex.vertexChunkCount} vertex chunks, ${this.chunkIndex.segmentIndex.chunkCount} segment chunks`
        );
      }
    } catch (error) {
      log.error(
        Modules.LINES_LOADER,
        `Failed to load Lines spatial index for ${this.node.path}:`,
        error
      );
      throw error;
    }

    // Open arrays for later access
    try {
      let verticesArray = await zarr.open(this.zarrLocation.resolve('vertices'), {
        kind: 'array',
      });
      let segmentsArray = await zarr.open(this.zarrLocation.resolve('segments'), {
        kind: 'array',
      });
      // Register array bounds with prefetcher for upper-bounds checking
      this.registerBounds('vertices', verticesArray);
      this.registerBounds('segments', segmentsArray);
      // Wrap with L0 cache if enabled (caches decoded chunks to avoid Blosc decompression)
      if (this.l0Cache) {
        verticesArray = wrapWithCache(
          verticesArray,
          this.l0Cache,
          `${this.node.path}/vertices`,
          () => this._activeProbe
        );
        segmentsArray = wrapWithCache(
          segmentsArray,
          this.l0Cache,
          `${this.node.path}/segments`,
          () => this._activeProbe
        );
      }
      this.arrays.vertices = verticesArray;
      this.arrays.segments = segmentsArray;
    } catch (e) {
      log.error(Modules.LINES_LOADER, 'Failed to open required Lines arrays:', e);
      throw e;
    }

    // Try to open optional arrays
    try {
      let widthsArray = await zarr.open(this.zarrLocation.resolve('widths'), { kind: 'array' });
      this.registerBounds('widths', widthsArray);
      if (this.l0Cache) {
        widthsArray = wrapWithCache(
          widthsArray,
          this.l0Cache,
          `${this.node.path}/widths`,
          () => this._activeProbe
        );
      }
      this.arrays.widths = widthsArray;
    } catch {
      log.info(Modules.LINES_LOADER, 'No widths array found (using default width)');
    }

    try {
      let colorsArray = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
      this.registerBounds('colors', colorsArray);
      if (this.l0Cache) {
        colorsArray = wrapWithCache(
          colorsArray,
          this.l0Cache,
          `${this.node.path}/colors`,
          () => this._activeProbe
        );
      }
      this.arrays.colors = colorsArray;
    } catch {
      log.info(Modules.LINES_LOADER, 'No colors array found (using default color)');
    }

    try {
      let sharpnessArray = await zarr.open(this.zarrLocation.resolve('sharpnesses'), {
        kind: 'array',
      });
      this.registerBounds('sharpnesses', sharpnessArray);
      if (this.l0Cache) {
        sharpnessArray = wrapWithCache(
          sharpnessArray,
          this.l0Cache,
          `${this.node.path}/sharpnesses`,
          () => this._activeProbe
        );
      }
      this.arrays.sharpness = sharpnessArray;
    } catch {
      log.info(Modules.LINES_LOADER, 'No sharpnesses array found (using default sharpness)');
    }

    // Open optional `scalars` zarr array when the node declares
    // has_scalars=true. Mirrors the Points loader pattern.
    const linesAttrs = this.node.attrs as { has_scalars?: boolean };
    if (linesAttrs?.has_scalars) {
      try {
        let scalarsArray = await zarr.open(this.zarrLocation.resolve('scalars'), {
          kind: 'array',
        });
        this.registerBounds('scalars', scalarsArray);
        if (this.l0Cache) {
          scalarsArray = wrapWithCache(
            scalarsArray,
            this.l0Cache,
            `${this.node.path}/scalars`,
            () => this._activeProbe
          );
        }
        this.arrays.scalars = scalarsArray;
      } catch (e: unknown) {
        log.info(
          Modules.LINES_LOADER,
          `has_scalars=true but failed to open scalars array — colormap mode disabled. ${e instanceof Error ? e.message : ''}`
        );
      }
    }

    // Initialize data accumulator for object pooling. The hot path
    // (loadLines below) reuses this accumulator's segment + vertex
    // buffers across updates when `useAccumulators` is true.
    if (appConfig.dataLoading.performance.useAccumulators) {
      const totalSegments = attrs.n_segments || 0;
      const totalVertices = attrs.n_vertices || 0;
      const ndim = attrs.ndim || this.arrays.vertices?.shape[1] || 3;

      // Estimate initial capacity (at least 1024 segments, or ~10% of total)
      const initialSegmentCap = Math.min(
        appConfig.dataLoading.performance.initialAccumulatorCapacity,
        Math.max(512, Math.ceil(totalSegments / 10))
      );
      const initialVertexCap = Math.min(
        appConfig.dataLoading.performance.initialAccumulatorCapacity,
        Math.max(1024, Math.ceil(totalVertices / 10))
      );

      this._accumulator = new LinesDataAccumulator(initialVertexCap, initialSegmentCap, ndim);

      if (appConfig.dataLoading.performance.enablePerformanceMonitoring) {
        const stats = this._accumulator.getStats();
        log.info(
          Modules.DATA_ACCUMULATOR,
          `Initialized LinesDataAccumulator for ${this.node.path}: ` +
            `segmentCap=${stats.capacity}, vertexCap=${initialVertexCap}, ndim=${ndim}`
        );
      }
    }
  }

  /**
   * Load lines data for the given view state
   * @param viewState - Current view state
   * @param session - Optional profiler session for nested timing
   */
  async loadLines(viewState: LinesViewState, session?: UpdateSession): Promise<LoadedLinesData> {
    const startTime = Date.now();
    const queryId = `${this.node.path}-${startTime}-${this.nextQueryId++}`;

    try {
      const result = await this.loadLinesInternal(viewState, session, queryId, startTime);
      this.finishQueryTracking(queryId, startTime, 'complete');
      return result;
    } catch (err) {
      this.metrics.errors += 1;
      this.finishQueryTracking(queryId, startTime, 'error');
      // Emit a monitor 'error' event so event-driven dashboards /
      // timelines see the failure. Mirrors the Points loader semantics.
      this.emitEvent({
        type: 'error',
        loader: 'lines-spatial-index',
        timestamp: Date.now(),
        data: {
          path: this.node.path,
          error: String(err),
        },
      });
      throw err;
    }
  }

  private async loadLinesInternal(
    viewState: LinesViewState,
    session: UpdateSession | undefined,
    queryId: string,
    startTime: number
  ): Promise<LoadedLinesData> {
    await this._onceInit.ensure(() => this.initialize());

    if (!this.arrays.vertices || !this.arrays.segments) {
      throw new Error('[LinesLoader] Loader not properly initialized');
    }

    const attrs = this.node.attrs as unknown as LinesMetadata;

    // Stage 1: Query segment chunks and load segments (async to allow
    // worker offload).
    let segmentRanges: SegmentRange[];
    if (session) {
      const querySession = session.begin('Spatial Query');
      try {
        segmentRanges = await this.queryVisibleSegmentRanges(viewState);
      } finally {
        querySession.end();
      }
    } else {
      segmentRanges = await this.queryVisibleSegmentRanges(viewState);
    }

    // Begin query tracking now that we know the segment ranges.
    const totalSegmentsRequested = segmentRanges.reduce((sum, r) => sum + (r.end - r.start), 0);
    this.metrics.queries += 1;
    this.activeQueries.set(queryId, {
      id: queryId,
      loader: 'lines-spatial-index',
      path: this.node.path,
      startTime,
      status: 'loading',
      cells: segmentRanges.length,
      points: totalSegmentsRequested,
      ranges: segmentRanges as unknown as PointRange[],
    });
    this.emitEvent({
      type: 'query',
      loader: 'lines-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        ranges: segmentRanges as unknown as PointRange[],
        cells: segmentRanges.length,
        points: totalSegmentsRequested,
        queryPosition: viewState.slicePosition,
        queryTolerance: viewState.tolerance,
      },
    });

    if (segmentRanges.length === 0) {
      log.info(Modules.LINES_LOADER, 'No visible segments - returning empty lines data');
      return createEmptyLinesData(attrs);
    }

    // Load segment indices
    if (!this._initialLoadDone) {
      log.load(Modules.LINES_LOADER, `Loading segments for ${segmentRanges.length} ranges`);
    }
    let segmentData: Uint32Array;
    if (session) {
      const loadSegSession = session.begin('Load Segments');
      try {
        segmentData = await this.loadSegmentRanges(segmentRanges);
      } finally {
        loadSegSession.end();
      }
    } else {
      segmentData = await this.loadSegmentRanges(segmentRanges);
    }

    // Collect unique vertex indices from loaded segments
    const uniqueVertexIndices = new Set<number>();
    for (let i = 0; i < segmentData.length; i++) {
      uniqueVertexIndices.add(segmentData[i]);
    }

    // Stage 2: Load required vertices
    const sortedIndices = Array.from(uniqueVertexIndices).sort((a, b) => a - b);
    const vertexRanges = computeVertexRangesFromIndices(sortedIndices);
    const mergedVertexRanges = mergeRanges(vertexRanges);

    // DIAGNOSTIC: Show vertex index distribution
    const minIdx = sortedIndices[0];
    const maxIdx = sortedIndices[sortedIndices.length - 1];
    const indexSpan = maxIdx - minIdx + 1;
    const efficiency = sortedIndices.length / indexSpan;

    if (!this._initialLoadDone) {
      log.load(
        Modules.LINES_LOADER,
        `Loading vertices for ${mergedVertexRanges.length} ranges (${sortedIndices.length} unique vertices)`
      );
      log.info(
        Modules.LINES_LOADER,
        `  Vertex index range: [${minIdx} - ${maxIdx}], span=${indexSpan}, efficiency=${(efficiency * 100).toFixed(1)}%`
      );
      if (mergedVertexRanges.length <= 10) {
        log.info(
          Modules.LINES_LOADER,
          `  Ranges: ${mergedVertexRanges.map((r) => `[${r.start}-${r.end})`).join(', ')}`
        );
      } else {
        const first5 = mergedVertexRanges
          .slice(0, 5)
          .map((r) => `[${r.start}-${r.end})`)
          .join(', ');
        const last5 = mergedVertexRanges
          .slice(-5)
          .map((r) => `[${r.start}-${r.end})`)
          .join(', ');
        log.info(Modules.LINES_LOADER, `  First 5 ranges: ${first5}`);
        log.info(Modules.LINES_LOADER, `  Last 5 ranges: ${last5}`);
      }
    }

    // Load directly to the accumulator when enabled (zero allocations).
    // ensureCapacity is called with the ACTUAL segment count — relying
    // on a ratio estimate truncates the buffer for line geometries with
    // a near-1:1 segment-to-vertex ratio.
    const useAccumulator = this._accumulator && appConfig.dataLoading.performance.useAccumulators;

    if (useAccumulator && this._accumulator) {
      // Ensure capacity FIRST using ACTUAL counts (not estimated!)
      if (segmentData.length % 2 !== 0) {
        log.warning(
          Modules.LINES_LOADER,
          `Segment data length ${segmentData.length} is not even — truncating to floor`
        );
      }
      const segmentCount = Math.floor(segmentData.length / 2);
      const actualVertexCount = sortedIndices.length; // ACTUAL count from unique indices!
      // FIXED: Pass BOTH vertex count AND segment count to avoid buffer truncation
      // For particle tracks, N vertices → N-1 segments (ratio ~1:1, not 1.5:1)
      this._accumulator.ensureCapacity(actualVertexCount, segmentCount);

      // Initialize accumulator types based on array metadata (must be done BEFORE loading!)
      // This ensures colorBuffer has the correct type (Uint8/Uint16/Float32)
      // IMPORTANT: For encoded arrays, use encoding.original_dtype NOT the zarr array dtype!
      // The zarr array dtype is the quantized format (e.g., uint8), but loadColorRanges
      // returns the original dtype from encoding metadata (e.g., float32 for HDR colors).
      if (this.arrays.colors) {
        const colorAttrs = this.arrays.colors.attrs as unknown as ArrayMetadata;
        const originalDtype = colorAttrs?.encoding?.original_dtype;
        const colorDtype = originalDtype || String(this.arrays.colors.dtype);
        const colorType = getExpectedColorType(colorDtype);
        // Create a small typed array to initialize accumulator types
        const sampleColors =
          colorType === 'Uint8Array'
            ? new Uint8Array(3)
            : colorType === 'Uint16Array'
              ? new Uint16Array(3)
              : new Float32Array(3);
        this._accumulator.fill(0, 0, {
          positions: new Float32Array(attrs.ndim),
          segments: new Uint32Array(2),
          widths: new Float32Array(1),
          colors: sampleColors,
          sharpness: this.arrays.sharpness ? new Float32Array(1) : undefined,
        });
      }

      // Get direct buffer references (now colorBuffer has correct type!)
      const vertexBuffer = this._accumulator.getVertexBuffer();
      const segmentBuffer = this._accumulator.getSegmentBuffer();
      const widthBuffer = this._accumulator.getWidthBuffer();
      const colorBuffer = this.arrays.colors ? this._accumulator.getColorBuffer() : null;
      const sharpnessBuffer = this.arrays.sharpness ? this._accumulator.getSharpnessBuffer() : null;
      // per-vertex scalar buffer when available.
      const scalarBuffer = this.arrays.scalars ? this._accumulator.getScalarBuffer() : null;

      // Profile vertex loading (accumulator path)
      const loadVertSession = session?.begin('Load Vertices');
      try {
        // Load directly into accumulator buffers (ZERO intermediate allocations!)
        await this.loadVertexRanges('vertices', mergedVertexRanges, attrs.ndim, vertexBuffer);

        if (this.arrays.widths) {
          await this.loadVertexRanges('widths', mergedVertexRanges, 1, widthBuffer);
        } else {
          // Create default widths directly in buffer
          widthBuffer.fill(1.0, 0, sortedIndices.length);
        }

        // Load colors if present
        // NOTE: For encoded arrays, loadColorRanges uses Float32 intermediate buffer
        // then converts to original dtype - it may return a different buffer than targetBuffer.
        // We load without target buffer and copy the result to ensure correctness.
        if (colorBuffer) {
          const loadedColors = await this.loadColorRanges(mergedVertexRanges);
          // Copy loaded colors to accumulator's colorBuffer
          // Note: TypedArray.set() handles type conversion automatically
          colorBuffer.set(loadedColors as ArrayLike<number>);
        }

        if (sharpnessBuffer) {
          await this.loadVertexRanges('sharpness', mergedVertexRanges, 1, sharpnessBuffer);
        }

        // Load per-vertex scalars directly into the accumulator buffer.
        // The accumulator may hold a Uint8Array scalar buffer once the
        // first fill() observes Uint8 input, but the spatial-index path
        // hits this branch before any fill() and therefore sees the
        // constructor's default Float32Array. The `loadVertexRanges` API
        // is Float32-only by design; routing Uint8 zarr scalars through it
        // would require a typed-buffer variant.
        if (scalarBuffer) {
          await this.loadVertexRanges(
            'scalars',
            mergedVertexRanges,
            1,
            scalarBuffer as Float32Array
          );
          // Flip the hasScalars flag — direct buffer writes bypass fill().
          this._accumulator.markScalarsLoaded();
        }
      } finally {
        loadVertSession?.end();
      }

      const remapSession = session?.begin('Index Remap');
      let vertexCount: number;
      try {
        // Build global → local index mapping
        const vertexIndexMap = new Map<number, number>();
        let localIdx = 0;
        for (const range of mergedVertexRanges) {
          for (let i = range.start; i < range.end; i++) {
            vertexIndexMap.set(i, localIdx++);
          }
        }

        // Remap segment indices directly in accumulator buffer (ZERO allocation!)
        for (let i = 0; i < segmentData.length; i++) {
          const localIndex = vertexIndexMap.get(segmentData[i]);
          if (localIndex === undefined) {
            throw new Error(`Vertex index ${segmentData[i]} not found in loaded data`);
          }
          segmentBuffer[i] = localIndex;
        }

        vertexCount = vertexIndexMap.size;
      } finally {
        remapSession?.end();
      }

      // Return from accumulator (subarrays, zero copy!)
      return this._accumulator.getData(segmentCount, vertexCount);
    }

    // Fallback: Load to separate arrays (allocations when accumulator disabled)
    let vertexPositions: Float32Array;
    let widths: Float32Array;
    let colors: Float32Array | Uint8Array | Uint16Array | null = null;
    let sharpness: Float32Array | null = null;
    // optional per-vertex scalars for colormap lookup. Optional +
    // undefined matches `LoadedLinesData.scalars?: ScalarArray`.
    let scalars: Float32Array | undefined;

    if (session) {
      const loadVertSession = session.begin('Load Vertices');
      try {
        vertexPositions = await this.loadVertexRanges('vertices', mergedVertexRanges, attrs.ndim);
        widths = this.arrays.widths
          ? await this.loadVertexRanges('widths', mergedVertexRanges, 1)
          : this.createDefaultWidths(sortedIndices.length);
        // Use multi-type loadColorRanges for colors
        colors = this.arrays.colors ? await this.loadColorRanges(mergedVertexRanges) : null;
        sharpness = this.arrays.sharpness
          ? await this.loadVertexRanges('sharpness', mergedVertexRanges, 1)
          : null;
        scalars = this.arrays.scalars
          ? await this.loadVertexRanges('scalars', mergedVertexRanges, 1)
          : undefined;
      } finally {
        loadVertSession.end();
      }
    } else {
      vertexPositions = await this.loadVertexRanges('vertices', mergedVertexRanges, attrs.ndim);
      widths = this.arrays.widths
        ? await this.loadVertexRanges('widths', mergedVertexRanges, 1)
        : this.createDefaultWidths(sortedIndices.length);
      // Use multi-type loadColorRanges for colors
      colors = this.arrays.colors ? await this.loadColorRanges(mergedVertexRanges) : null;
      sharpness = this.arrays.sharpness
        ? await this.loadVertexRanges('sharpness', mergedVertexRanges, 1)
        : null;
      scalars = this.arrays.scalars
        ? await this.loadVertexRanges('scalars', mergedVertexRanges, 1)
        : undefined;
    }

    const remapSession = session?.begin('Index Remap');
    let remappedSegments: Uint32Array;
    let vertexIndexMapSize: number;
    try {
      // Build global → local index mapping
      const vertexIndexMap = new Map<number, number>();
      let localIdx = 0;
      for (const range of mergedVertexRanges) {
        for (let i = range.start; i < range.end; i++) {
          vertexIndexMap.set(i, localIdx++);
        }
      }

      // Remap segment indices to local space
      remappedSegments = new Uint32Array(segmentData.length);
      for (let i = 0; i < segmentData.length; i++) {
        const localIndex = vertexIndexMap.get(segmentData[i]);
        if (localIndex === undefined) {
          throw new Error(`Vertex index ${segmentData[i]} not found in loaded data`);
        }
        remappedSegments[i] = localIndex;
      }
      vertexIndexMapSize = vertexIndexMap.size;
    } finally {
      remapSession?.end();
    }

    // Fallback: Return new object
    return {
      positions: vertexPositions,
      segments: remappedSegments,
      widths,
      colors,
      sharpness,
      // scalars now flow through the loader fallback path.
      scalars,
      segmentCount: Math.floor(segmentData.length / 2),
      vertexCount: vertexIndexMapSize,
      ndim: attrs.ndim,
    };
  }

  /**
   * Update view for new position.
   * Note: extend_to_all optimization is handled at scene-loader level,
   * which skips calling this method entirely for fully-extended nodes.
   * @param viewState - Current view state
   * @param session - Optional profiler session for nested timing
   */
  async updateView(viewState: LinesViewState, session?: UpdateSession): Promise<LoadedLinesData> {
    const result = await this.loadLines(viewState, session);
    if (!this._initialLoadDone) {
      this._initialLoadDone = true;
      this.rangeLoader.setVerbose(false);
    }
    return result;
  }

  /**
   * Like {@link updateView} but also reports whether the load was served
   * entirely from cache (see PointsSpatialIndexLoader.updateViewWithResidency).
   */
  async updateViewWithResidency(
    viewState: LinesViewState,
    session?: UpdateSession
  ): Promise<{ data: LoadedLinesData; allResident: boolean }> {
    const probe = new ResidencyAccumulator();
    this._activeProbe = probe;
    try {
      const data = await this.updateView(viewState, session);
      return { data, allResident: probe.allResident };
    } finally {
      this._activeProbe = null;
    }
  }

  /**
   * Warm the cache for the given viewState without producing geometry.
   *
   * Mirrors the gsplats / points `prefetchChunks(viewState)` so a
   * dimension-animation hook can prefetch the next slice while the
   * current frame renders. Lines need both the segment-level bounds
   * (segments + widths + colors + sharpness keyed on segment range)
   * and the vertex-level bounds (vertices keyed on the
   * segment→vertex projection); this implementation keeps it simple
   * by warming each available array over the queried segment ranges
   * — the segment→vertex remap happens at demand time and the L0/L1
   * cache absorbs the extra reads.
   */
  async prefetchChunks(viewState: LinesViewState): Promise<void> {
    await this._onceInit.ensure(() => this.initialize());

    if (!this.arrays.segments) return;

    let ranges: SegmentRange[];
    try {
      ranges = await this.queryVisibleSegmentRanges(viewState);
    } catch {
      return;
    }
    if (ranges.length === 0) return;

    const arrays = [
      this.arrays.vertices,
      this.arrays.segments,
      this.arrays.widths,
      this.arrays.colors,
      this.arrays.sharpness,
    ].filter((a): a is zarr.Array<zarr.DataType, zarr.Readable> => a != null);

    const fetches: Promise<unknown>[] = [];
    for (const array of arrays) {
      const shape = array.shape;
      for (const range of ranges) {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];
        fetches.push(get(array, sliceSpec));
      }
    }
    await Promise.all(fetches);
  }

  /**
   * Probe both `vertex_chunk_bounds` and `segment_chunk_bounds` from zarr.
   *
   * Implementation lives in `lines/chunk-index-loader.ts`. The thin
   * wrapper here exists for symmetry with the points facade, which
   * follows the same pattern.
   */
  private async loadDualChunkBounds(attrs: LinesMetadata): Promise<LinesDualChunkIndex | null> {
    return loadLinesDualChunkIndex(this.zarrLocation, attrs);
  }

  /**
   * Query visible segment ranges based on view state.
   *
   * Delegates the chunk-bounds AABB scan and range coalescing to the canonical
   * `SpatialQueryBuilder` with `geometryType: 'lines'`. Returns a load-all
   * range when no spatial index is available.
   */
  private async queryVisibleSegmentRanges(viewState: LinesViewState): Promise<SegmentRange[]> {
    const attrs = this.node.attrs as unknown as LinesMetadata;
    const extendDims: string[] = this.node.attrs.extend_to_all || [];

    warnExtendToAllNoDimensions({
      extendDims,
      hasResolvedDimensions: !!viewState.dimensions && viewState.dimensions.length > 0,
      nodePath: this.node.path,
      logModule: Modules.LINES_LOADER,
    });

    if (!this.chunkIndex) {
      return [{ start: 0, end: attrs.n_segments }];
    }

    if (!this._initialLoadDone) {
      announceExtendToAllOnce({
        extendDims,
        nodePath: this.node.path,
        logModule: Modules.LINES_LOADER,
      });
    }

    const ranges = await new SpatialQueryBuilder(this.chunkIndex.segmentIndex, viewState, {
      geometryType: 'lines',
      totalElements: attrs.n_segments,
      chunkSize: attrs.segment_ordering!.chunk_size,
      extendDims,
      logModule: Modules.LINES_LOADER,
    }).execute();

    return ranges;
  }

  /**
   * Load segment index data for given ranges
   */
  private async loadSegmentRanges(ranges: SegmentRange[]): Promise<Uint32Array> {
    if (!this.arrays.segments) {
      throw new Error('[LinesLoader] Segments array not initialized');
    }

    // Calculate total segments to load
    const totalSegments = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const output = new Uint32Array(totalSegments * 2);

    let destOffset = 0;

    for (const range of ranges) {
      const sliceSpec = [slice(range.start, range.end), slice(null)] as zarr.Slice[];
      const data = await get(this.arrays.segments, sliceSpec);
      // Data can be typed array or ArrayBuffer-like, handle both
      const segmentData =
        data.data instanceof Uint32Array
          ? data.data
          : new Uint32Array(data.data as unknown as ArrayBufferLike);

      output.set(segmentData, destOffset);
      destOffset += segmentData.length;
    }

    this.recordLoadMetrics('segments', totalSegments, output);
    return output;
  }

  /**
   * Load vertex ranges with optional target buffer for zero-allocation operation.
   *
   * Uses RangeLoader for unified encoding dispatch (broadcasted, quantized, lut, direct).
   * Array references are handled specially (need zarrStore access to resolve target).
   */
  private async loadVertexRanges(
    arrayName: string,
    ranges: SegmentRange[],
    elementsPerVertex: number,
    targetBuffer?: Float32Array
  ): Promise<Float32Array> {
    const array = this.arrays[arrayName as keyof typeof this.arrays];
    if (!array) {
      throw new Error(`Array ${arrayName} not initialized`);
    }

    const totalVertices = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const totalElements = totalVertices * elementsPerVertex;

    // Use target buffer or allocate (ZERO allocation when targetBuffer provided!)
    const output = targetBuffer ? targetBuffer : new Float32Array(totalElements);
    const attrs = array.attrs as unknown as ArrayMetadata;

    // Helper resolves array_ref via zarrStore when needed and delegates to
    // RangeLoader.loadRanges otherwise. The hint elementsPerVertex is only
    // used for the non-ref path; ref targets recompute from their own shape.
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    await this.rangeLoader.loadRangesResolvingRef(
      array,
      attrs,
      ranges as LoadRange[],
      output,
      totalVertices,
      elementsPerVertex,
      storeToUse,
      `Lines:${arrayName}`
    );

    this.recordLoadMetrics(arrayName, totalVertices, output);
    return output;
  }

  /**
   * Load color ranges with multi-type support (preserves original_dtype)
   *
   * This method handles the full encoding/decoding pipeline for colors,
   * including original_dtype restoration for encoded arrays.
   */
  private async loadColorRanges(
    ranges: SegmentRange[],
    targetBuffer?: Float32Array | Uint8Array | Uint16Array
  ): Promise<Float32Array | Uint8Array | Uint16Array> {
    const array = this.arrays.colors;
    if (!array) {
      throw new Error('[LinesLoader] Colors array not initialized');
    }
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    const totalVertices = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const output = await loadColorRanges(
      array,
      ranges,
      this.rangeLoader,
      storeToUse,
      'Lines',
      targetBuffer
    );
    this.recordLoadMetrics('colors', totalVertices, output);
    return output;
  }

  /**
   * Create default widths array (all 1.0)
   */
  private createDefaultWidths(count: number): Float32Array {
    const widths = new Float32Array(count);
    widths.fill(1.0);
    return widths;
  }

  /**
   * Get accumulator stats for memory monitoring
   */
  getAccumulatorStats(): AccumulatorStats | null {
    return this._accumulator?.getStats() ?? null;
  }

  // ────────────────────────────────────────────────────────────────────
  // LoaderMonitor surface — same shape as the points and gsplats facades.
  // ────────────────────────────────────────────────────────────────────

  addEventListener(listener: MonitorEventListener): void {
    this.events.add(listener);
  }

  removeEventListener(listener: MonitorEventListener): void {
    this.events.remove(listener);
  }

  getMetrics(): LoaderMetrics {
    return { ...this.metrics };
  }

  getActiveQueries(): QueryInfo[] {
    return Array.from(this.activeQueries.values());
  }

  /**
   * Update metrics + emit a 'load' event. Mirrors the points facade's
   * recordLoadMetrics shape: items is the per-attribute item count (vertices
   * or segments depending on the array), output supplies the bytes.
   */
  private recordLoadMetrics(arrayName: string, items: number, output: ArrayBufferView): void {
    const queryStart = this.activeQueries.values().next().value?.startTime;
    const loadTime = computeLoadLatency(queryStart);
    const bytes = output.byteLength;

    recordLoadEvent(this.metrics, items, bytes, loadTime);

    // Resident memory = current accumulator allocation (MB → bytes). Assignment
    // (not +=): memoryUsed is a live footprint that grows/shrinks with the pool,
    // unlike the cumulative bytesLoaded counter updated above.
    this.metrics.memoryUsed = Math.round((this.getAccumulatorStats()?.memoryMB ?? 0) * 1024 * 1024);

    this.emitEvent({
      type: 'load',
      loader: 'lines-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        arrayName,
        points: items,
        memory: bytes,
        latency: loadTime,
      },
    });
  }

  private emitEvent(event: MonitorEvent): void {
    this.events.emit(event);
  }

  /**
   * Close out a tracked query: update the rolling avgQueryTime, mark the
   * QueryInfo as complete or errored, and drop it from `activeQueries`.
   */
  private finishQueryTracking(
    queryId: string,
    startTime: number,
    status: 'complete' | 'error'
  ): void {
    const query = this.activeQueries.get(queryId);
    if (query) {
      query.status = status;
      query.endTime = Date.now();
      this.activeQueries.delete(queryId);
    }
    const queryTime = Date.now() - startTime;
    if (this.metrics.queries > 0) {
      this.metrics.avgQueryTime =
        (this.metrics.avgQueryTime * (this.metrics.queries - 1) + queryTime) / this.metrics.queries;
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.chunkIndex = null;
    this.arrays = {};
    this._onceInit.reset();
    this.events.clear();
    this.activeQueries.clear();

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
  }
}
