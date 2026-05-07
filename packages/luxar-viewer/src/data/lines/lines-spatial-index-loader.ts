/**
 * Lines spatial index-based data loader for efficient nD lines loading.
 *
 * This loader implements two-phase loading:
 * 1. Query segment chunks and load visible segments
 * 2. Derive required vertex chunks from segment indices and load vertices
 *
 * It also handles nD slicing with endpoint clipping for segments that
 * partially intersect the visible slice.
 *
 * @module data/lines-spatial-index-loader
 */

import * as zarr from 'zarrita';
import { get, slice } from 'zarrita';
import { log, Modules } from '../../utils/log';
import type {
  LinesMetadata,
  LoadedLinesData,
  LinesDataLoader,
  LinesViewState,
  SegmentRange,
} from '../../types/lines';
import type { SceneNode } from '../data-loader-types';
import { ArrayRefRegistry, type ArrayMetadata } from '../utils/array-decoder';
import { fetchChunkBoundsArray } from '../loaders/chunk-bounds-loader';
import {
  RangeLoader,
  SpatialQueryBuilder,
  mergeRanges,
  type ChunkSpatialIndex,
  type LoadRange,
} from '../loaders';
import { getExpectedColorType, loadColorRanges } from '../loaders/color-attribute-utils';
import { OnceInit } from '../loaders/once-init';
import {
  warnExtendToAllNoDimensions,
  announceExtendToAllOnce,
} from '../loaders/extend-to-all-preflight';
import { LinesDataAccumulator, type AccumulatorStats } from '../utils/data-accumulator';
import { config as appConfig } from '../../config';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import { DecompressedChunkCache, wrapWithCache, ChunkPrefetcher } from '../../cache';

/**
 * Internal index shape: the lines loader carries both vertex and segment
 * chunk bounds, but the chunk-bounds spatial query only consults the segment
 * side. The vertex side is loaded so the loader can compute byte counts and
 * (in the future) re-enable vertex-driven prefetching, but is not used in
 * the current query path.
 */
interface LinesDualChunkIndex {
  segmentIndex: ChunkSpatialIndex;
  vertexChunkBounds: Float32Array;
  vertexChunkCount: number;
}

/**
 * Compute contiguous vertex ranges from a sorted list of vertex indices.
 *
 * Used after loading segment data: each segment references two vertex
 * indices, and we batch the unique sorted indices into runs of consecutive
 * integers so zarr loading touches the minimum number of chunks.
 *
 * This is genuinely lines-specific (operates on per-segment vertex indices,
 * not on chunk bounds) and is therefore not in the canonical
 * `loaders/spatial-query-builder` API.
 */
function computeVertexRangesFromIndices(sortedIndices: number[]): SegmentRange[] {
  if (sortedIndices.length === 0) return [];

  const ranges: SegmentRange[] = [];
  let rangeStart = sortedIndices[0];
  let rangeEnd = sortedIndices[0] + 1;

  for (let i = 1; i < sortedIndices.length; i++) {
    const idx = sortedIndices[i];
    if (idx === rangeEnd) {
      rangeEnd++;
    } else {
      ranges.push({ start: rangeStart, end: rangeEnd });
      rangeStart = idx;
      rangeEnd = idx + 1;
    }
  }
  ranges.push({ start: rangeStart, end: rangeEnd });

  return ranges;
}

/**
 * Lines data loader using spatial indices for efficient nD queries.
 *
 * Key features:
 * - Two-phase loading: segments → vertices
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

  // Data accumulator for object pooling (Phase 1 optimization)
  private _accumulator: LinesDataAccumulator | null = null;

  // L0 decompressed chunk cache (optional, avoids Blosc decompression on repeat access)
  private l0Cache: DecompressedChunkCache | null = null;

  // Chunk prefetcher (optional, for registering array bounds to suppress 404s)
  private prefetcher: ChunkPrefetcher | null = null;

  // Suppress detail logs after first successful view update
  private _initialLoadDone = false;

  private arrays: {
    vertices?: zarr.Array<zarr.DataType, zarr.Readable>;
    segments?: zarr.Array<zarr.DataType, zarr.Readable>;
    widths?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    sharpness?: zarr.Array<zarr.DataType, zarr.Readable>;
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
  }

  /** Register array shape with the prefetcher for upper-bounds checking. */
  private registerBounds(arrayName: string, array: zarr.Array<zarr.DataType, zarr.Readable>): void {
    if (!this.prefetcher) return;
    const path = `${this.node.path.startsWith('/') ? this.node.path.slice(1) : this.node.path}/${arrayName}`;
    this.prefetcher.registerArrayBounds(path, array.shape, array.chunks);
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
        verticesArray = wrapWithCache(verticesArray, this.l0Cache, `${this.node.path}/vertices`);
        segmentsArray = wrapWithCache(segmentsArray, this.l0Cache, `${this.node.path}/segments`);
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
        widthsArray = wrapWithCache(widthsArray, this.l0Cache, `${this.node.path}/widths`);
      }
      this.arrays.widths = widthsArray;
    } catch {
      log.info(Modules.LINES_LOADER, 'No widths array found (using default width)');
    }

    try {
      let colorsArray = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
      this.registerBounds('colors', colorsArray);
      if (this.l0Cache) {
        colorsArray = wrapWithCache(colorsArray, this.l0Cache, `${this.node.path}/colors`);
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
          `${this.node.path}/sharpnesses`
        );
      }
      this.arrays.sharpness = sharpnessArray;
    } catch {
      log.info(Modules.LINES_LOADER, 'No sharpnesses array found (using default sharpness)');
    }

    // Initialize data accumulator for object pooling (Phase 1 optimization)
    // NOTE: Infrastructure-only for Phase 1. Full hot path integration deferred to Phase 2.
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
            `segmentCap=${stats.capacity}, vertexCap=${initialVertexCap}, ndim=${ndim} ` +
            '(infrastructure-only, hot path integration in Phase 2)'
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
    await this._onceInit.ensure(() => this.initialize());

    if (!this.arrays.vertices || !this.arrays.segments) {
      throw new Error('Lines loader not properly initialized');
    }

    const attrs = this.node.attrs as unknown as LinesMetadata;

    // Phase 1: Query segment chunks and load segments (Phase 2: async for worker support)
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

    if (segmentRanges.length === 0) {
      log.info(Modules.LINES_LOADER, 'No visible segments - returning empty lines data');
      return this.createEmptyLinesData(attrs);
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

    // Phase 2: Load required vertices
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

    // Phase 1 DEEP Integration: Load directly to accumulator if enabled (ZERO allocations!)
    // BUG FIXED: Segment capacity was estimated incorrectly (1.5:1 ratio instead of ~1:1 for particle tracks)
    // Now passing actual segment count to ensureCapacity() to avoid buffer truncation.
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
      } finally {
        loadVertSession?.end();
      }

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

      const vertexCount = vertexIndexMap.size;

      // Return from accumulator (subarrays, zero copy!)
      return this._accumulator.getData(segmentCount, vertexCount);
    }

    // Fallback: Load to separate arrays (allocations when accumulator disabled)
    let vertexPositions: Float32Array;
    let widths: Float32Array;
    let colors: Float32Array | Uint8Array | Uint16Array | null = null;
    let sharpness: Float32Array | null = null;

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
    }

    // Build global → local index mapping
    const vertexIndexMap = new Map<number, number>();
    let localIdx = 0;
    for (const range of mergedVertexRanges) {
      for (let i = range.start; i < range.end; i++) {
        vertexIndexMap.set(i, localIdx++);
      }
    }

    // Remap segment indices to local space
    const remappedSegments = new Uint32Array(segmentData.length);
    for (let i = 0; i < segmentData.length; i++) {
      const localIndex = vertexIndexMap.get(segmentData[i]);
      if (localIndex === undefined) {
        throw new Error(`Vertex index ${segmentData[i]} not found in loaded data`);
      }
      remappedSegments[i] = localIndex;
    }

    // Fallback: Return new object
    return {
      positions: vertexPositions,
      segments: remappedSegments,
      widths,
      colors,
      sharpness,
      segmentCount: Math.floor(segmentData.length / 2),
      vertexCount: vertexIndexMap.size,
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
   * Probe both `vertex_chunk_bounds` and `segment_chunk_bounds` from zarr.
   *
   * Returns null when spatial ordering is disabled, expected for small/non-
   * ordered datasets (graceful fallback to full load).
   */
  private async loadDualChunkBounds(attrs: LinesMetadata): Promise<LinesDualChunkIndex | null> {
    if (attrs.ordering === 'none' || !attrs.vertex_ordering || !attrs.segment_ordering) {
      log.info(
        Modules.LINES_LOADER,
        `Lines node has no spatial ordering (ordering=${attrs.ordering})`
      );
      return null;
    }

    const vertexResult = await fetchChunkBoundsArray(
      this.zarrLocation,
      'vertex_chunk_bounds',
      Modules.LINES_LOADER,
      'No chunk bounds found - Lines dataset has no spatial indexing'
    );
    if (!vertexResult) return null;

    const segmentResult = await fetchChunkBoundsArray(
      this.zarrLocation,
      'segment_chunk_bounds',
      Modules.LINES_LOADER,
      'No chunk bounds found - Lines dataset has no spatial indexing'
    );
    if (!segmentResult) return null;

    const vertexChunkBounds = vertexResult.data;
    const segmentChunkBounds = segmentResult.data;

    const vertexChunkCount = Math.ceil(attrs.n_vertices / attrs.vertex_ordering.chunk_size);
    const segmentChunkCount = Math.ceil(attrs.n_segments / attrs.segment_ordering.chunk_size);

    const expectedVertexSize = vertexChunkCount * attrs.ndim * 2;
    const expectedSegmentSize = segmentChunkCount * attrs.ndim * 2;
    if (vertexChunkBounds.length !== expectedVertexSize) {
      log.warning(
        Modules.LINES_LOADER,
        `Vertex bounds size mismatch: got ${vertexChunkBounds.length}, expected ${expectedVertexSize}`
      );
    }
    if (segmentChunkBounds.length !== expectedSegmentSize) {
      log.warning(
        Modules.LINES_LOADER,
        `Segment bounds size mismatch: got ${segmentChunkBounds.length}, expected ${expectedSegmentSize}`
      );
    }

    return {
      segmentIndex: {
        chunkBounds: segmentChunkBounds,
        chunkCount: segmentChunkCount,
        metadata: { ndim: attrs.ndim, chunk_size: attrs.segment_ordering.chunk_size },
      },
      vertexChunkBounds,
      vertexChunkCount,
    };
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
      throw new Error('Segments array not initialized');
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
      throw new Error('Colors array not initialized');
    }
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    return loadColorRanges(array, ranges, this.rangeLoader, storeToUse, 'Lines', targetBuffer);
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
   * Create empty lines data
   */
  private createEmptyLinesData(attrs: LinesMetadata): LoadedLinesData {
    return {
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      segmentCount: 0,
      vertexCount: 0,
      ndim: attrs.ndim,
    };
  }

  /**
   * Get accumulator stats for memory monitoring
   */
  getAccumulatorStats(): AccumulatorStats | null {
    return this._accumulator?.getStats() ?? null;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.chunkIndex = null;
    this.arrays = {};
    this._onceInit.reset();

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
  }
}

