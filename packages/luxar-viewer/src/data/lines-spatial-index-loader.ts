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
import { log, Modules, LogEmoji } from '../utils/log';
import {
  loadLinesChunkSpatialIndex,
  querySegmentChunksForView,
  computeLinesTolerance,
  segmentChunkIndicesToRanges,
  mergeRanges,
  computeVertexRangesFromIndices,
} from './lines-chunk-spatial-index';
import type {
  LinesMetadata,
  LinesChunkSpatialIndex,
  LoadedLinesData,
  ProcessedLinesData,
  LinesDataLoader,
  LinesViewState,
  ClippedSegment,
  SegmentRange,
} from '../types/lines';
import type { SceneNode } from './data-loader-types';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from './array-decoder';
import { RangeLoader, type LoadRange } from './loaders';
import { LinesDataAccumulator, type AccumulatorStats } from './data-accumulator';
import { config as appConfig } from '../config';
import { getWorkerPool } from '../workers/worker-pool';
import type { UpdateProfiler, UpdateSession } from '../profiling/update-profiler';
import { initWasm, getFallback } from '../wasm';
import type { WasmModule } from '../wasm/types';
import { DecompressedChunkCache, wrapWithCache } from '../cache';

// ============================================================================
// WASM Module Caching for Hot Path Optimization
// ============================================================================

/** Cached WASM module instance (lazily initialized) */
let wasmModuleCache: WasmModule | null = null;
let wasmInitPromise: Promise<WasmModule> | null = null;

/**
 * Get the WASM module, initializing if necessary.
 * Uses caching to avoid repeated initialization overhead.
 */
async function getWasmModule(): Promise<WasmModule> {
  if (wasmModuleCache) {
    return wasmModuleCache;
  }

  if (!wasmInitPromise) {
    wasmInitPromise = initWasm().then((module) => {
      wasmModuleCache = module;
      return module;
    });
  }

  return wasmInitPromise;
}

/**
 * Get the WASM module synchronously (returns fallback if not yet initialized).
 * Used in hot paths where async is not desirable.
 */
function getWasmModuleSync(): WasmModule {
  if (wasmModuleCache) {
    return wasmModuleCache;
  }
  // Return fallback if WASM not yet loaded
  return getFallback();
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
  private chunkIndex: LinesChunkSpatialIndex | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private initPromise: Promise<void> | null = null;
  private initLock = false;
  private rangeLoader: RangeLoader;
  private zarrStore: zarr.Readable | null = null;

  // Data accumulator for object pooling (Phase 1 optimization)
  private _accumulator: LinesDataAccumulator | null = null;

  // L0 decompressed chunk cache (optional, avoids Blosc decompression on repeat access)
  private l0Cache: DecompressedChunkCache | null = null;

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
    l0Cache?: DecompressedChunkCache
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.rangeLoader = new RangeLoader(refRegistry || new ArrayRefRegistry());
    this.zarrStore = zarrStore || null;
    this.l0Cache = l0Cache || null;
    // profiler parameter kept for API compatibility; session is passed directly to methods
    void profiler;
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    const attrs = this.node.attrs as unknown as LinesMetadata;

    // Load dual spatial index
    try {
      this.chunkIndex = await loadLinesChunkSpatialIndex(this.zarrLocation, attrs);

      if (!this.chunkIndex) {
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `No spatial index for Lines ${this.node.path} - will load all data`
        );
      } else {
        log.query(
          Modules.SPATIAL_INDEX_LOADER,
          `Lines index loaded: ${this.chunkIndex.vertexChunkCount} vertex chunks, ${this.chunkIndex.segmentChunkCount} segment chunks`
        );
      }
    } catch (error) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
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
      // Wrap with L0 cache if enabled (caches decoded chunks to avoid Blosc decompression)
      if (this.l0Cache) {
        verticesArray = wrapWithCache(verticesArray, this.l0Cache, `${this.node.path}/vertices`);
        segmentsArray = wrapWithCache(segmentsArray, this.l0Cache, `${this.node.path}/segments`);
      }
      this.arrays.vertices = verticesArray;
      this.arrays.segments = segmentsArray;
    } catch (e) {
      log.error(Modules.SPATIAL_INDEX_LOADER, 'Failed to open required Lines arrays:', e);
      throw e;
    }

    // Try to open optional arrays
    try {
      let widthsArray = await zarr.open(this.zarrLocation.resolve('widths'), { kind: 'array' });
      if (this.l0Cache) {
        widthsArray = wrapWithCache(widthsArray, this.l0Cache, `${this.node.path}/widths`);
      }
      this.arrays.widths = widthsArray;
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No widths array found (using default width)');
    }

    try {
      let colorsArray = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
      if (this.l0Cache) {
        colorsArray = wrapWithCache(colorsArray, this.l0Cache, `${this.node.path}/colors`);
      }
      this.arrays.colors = colorsArray;
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No colors array found (using default color)');
    }

    try {
      let sharpnessArray = await zarr.open(this.zarrLocation.resolve('sharpness'), {
        kind: 'array',
      });
      if (this.l0Cache) {
        sharpnessArray = wrapWithCache(sharpnessArray, this.l0Cache, `${this.node.path}/sharpness`);
      }
      this.arrays.sharpness = sharpnessArray;
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No sharpness array found (using default sharpness)');
    }

    // Initialize data accumulator for object pooling (Phase 1 optimization)
    // NOTE: Infrastructure-only for Phase 1. Full hot path integration deferred to Phase 2.
    // See src/data/DATA_ACCUMULATOR_STATUS.md for details.
    if (appConfig.dataLoading.performance.useAccumulators) {
      const metadata = attrs as LinesMetadata;
      const totalSegments = this.chunkIndex?.metadata.n_segments || metadata.n_segments || 0;
      const totalVertices = this.chunkIndex?.metadata.n_vertices || metadata.n_vertices || 0;
      const ndim = this.chunkIndex?.metadata.ndim || this.arrays.vertices?.shape[1] || 3;

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
    // Prevent race conditions during initialization
    if (!this.initPromise && !this.initLock) {
      this.initLock = true;
      this.initPromise = this.initialize().finally(() => {
        this.initLock = false;
      });
    }

    if (this.initPromise) {
      await this.initPromise;
    }

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
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No visible segments - returning empty lines data');
      return this.createEmptyLinesData(attrs);
    }

    // Load segment indices
    log.info(
      LogEmoji.LOAD,
      Modules.SPATIAL_INDEX_LOADER,
      `Loading segments for ${segmentRanges.length} ranges`
    );
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

    log.info(
      LogEmoji.LOAD,
      Modules.SPATIAL_INDEX_LOADER,
      `Loading vertices for ${mergedVertexRanges.length} ranges (${sortedIndices.length} unique vertices)`
    );
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `  Vertex index range: [${minIdx} - ${maxIdx}], span=${indexSpan}, efficiency=${(efficiency * 100).toFixed(1)}%`
    );
    if (mergedVertexRanges.length <= 10) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
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
      log.info(Modules.SPATIAL_INDEX_LOADER, `  First 5 ranges: ${first5}`);
      log.info(Modules.SPATIAL_INDEX_LOADER, `  Last 5 ranges: ${last5}`);
    }

    // Phase 1 DEEP Integration: Load directly to accumulator if enabled (ZERO allocations!)
    // BUG FIXED: Segment capacity was estimated incorrectly (1.5:1 ratio instead of ~1:1 for particle tracks)
    // Now passing actual segment count to ensureCapacity() to avoid buffer truncation.
    const useAccumulator = this._accumulator && appConfig.dataLoading.performance.useAccumulators;

    if (useAccumulator && this._accumulator) {
      // Ensure capacity FIRST using ACTUAL counts (not estimated!)
      const segmentCount = segmentData.length / 2;
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
        const colorType = this.getExpectedColorType(colorDtype);
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
      const vertexBuffer = this._accumulator['vertexBuffer'] as Float32Array;
      const segmentBuffer = this._accumulator['segmentBuffer'] as Uint32Array;
      const widthBuffer = this._accumulator['widthBuffer'] as Float32Array;
      const colorBuffer = this.arrays.colors
        ? (this._accumulator['colorBuffer'] as Float32Array | Uint8Array | Uint16Array)
        : null;
      const sharpnessBuffer = this.arrays.sharpness
        ? (this._accumulator['sharpnessBuffer'] as Float32Array)
        : null;

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
      segmentCount: segmentData.length / 2,
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
    return this.loadLines(viewState, session);
  }

  /**
   * Query visible segment ranges based on view state
   */
  private async queryVisibleSegmentRanges(viewState: LinesViewState): Promise<SegmentRange[]> {
    const attrs = this.node.attrs as unknown as LinesMetadata;

    // Check if this node has extend_to_all dimensions
    const extendDims: string[] = this.node.attrs.extend_to_all || [];

    if (extendDims.length > 0) {
      // DEFENSIVE CHECK: Warn if dimensions not available for extend_to_all
      if (!viewState.dimensions || viewState.dimensions.length === 0) {
        log.warning(
          Modules.SPATIAL_INDEX_LOADER,
          `extend_to_all=[${extendDims.join(', ')}] specified for ${this.node.path} but ` +
            'viewState.dimensions is undefined. extend_to_all will not work. ' +
            'Ensure scene dimensions are initialized before loading nodes.'
        );
      }

      // Check if we're navigating through an extended dimension
      // LinesViewState.dimensions is DimensionMetadata[] directly
      const currentNonDisplayedDims: string[] =
        viewState.dimensions
          ?.filter((_meta: { name?: string }, idx: number) => !viewState.displayDims.includes(idx))
          ?.map((meta: { name?: string }) => meta.name)
          ?.filter((name: string | undefined): name is string => !!name) || [];

      const isExtending = extendDims.some((edim: string) => currentNonDisplayedDims.includes(edim));

      if (isExtending) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SPATIAL_INDEX_LOADER,
          `Extending ${this.node.path} visibility across: ${extendDims.join(', ')}`
        );
        // Return all segments for extended dimensions
        return [{ start: 0, end: attrs.n_segments }];
      }
    }

    if (!this.chunkIndex) {
      // No spatial index - load all segments
      return [{ start: 0, end: attrs.n_segments }];
    }

    // Compute tolerance for queries
    const tolerance = viewState.dimensions
      ? computeLinesTolerance(viewState.dimensions, viewState.displayDims)
      : new Array(attrs.ndim).fill(0).map((_, i) => (viewState.displayDims.includes(i) ? 1e10 : 0));

    // Ensure slicePosition has correct length
    const slicePosition = new Array(attrs.ndim).fill(0);
    for (let i = 0; i < Math.min(viewState.slicePosition.length, attrs.ndim); i++) {
      slicePosition[i] = viewState.slicePosition[i] ?? 0;
    }

    // Query segment chunks (worker or main thread based on config)
    let chunkIndices: number[];

    if (appConfig.dataLoading.performance.useWebWorkers) {
      // Phase 2: Use worker for spatial queries
      try {
        const worker = await getWorkerPool().getWorker();
        const result = await worker.querySpatialIndex({
          chunkBounds: this.chunkIndex.segmentChunkBounds,
          slicePosition: new Float32Array(slicePosition),
          tolerance: new Float32Array(tolerance),
          numChunks: this.chunkIndex.segmentChunkCount,
          ndim: attrs.ndim,
        });
        chunkIndices = Array.from(result);
      } catch (error) {
        log.error(Modules.SPATIAL_INDEX_LOADER, 'Worker query failed, using main thread:', error);
        // Fallback to main thread
        chunkIndices = querySegmentChunksForView(this.chunkIndex, slicePosition, tolerance);
      }
    } else {
      // Main thread query
      chunkIndices = querySegmentChunksForView(this.chunkIndex, slicePosition, tolerance);
    }

    if (chunkIndices.length === 0) {
      return [];
    }

    // Convert to ranges and merge
    const ranges = segmentChunkIndicesToRanges(
      chunkIndices,
      this.chunkIndex.metadata.segment_ordering!.chunk_size,
      attrs.n_segments
    );

    return mergeRanges(ranges);
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

    // Check for array_ref - needs special handling (zarrStore access to resolve target)
    const attrs = array.attrs as unknown as ArrayMetadata;
    const isArrayRef = ArrayDecoder.isArrayRef(attrs);

    if (isArrayRef) {
      // Array reference: Resolve target and use RangeLoader for target
      const targetPath = attrs.encoding!.target!;
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Lines: Array ref: ${arrayName} → ${targetPath} (using RangeLoader)`
      );

      const storeToUse = this.zarrStore || this.zarrLocation.store;
      const zarrRootLoc = zarr.root(storeToUse);
      const targetLoc = zarrRootLoc.resolve(targetPath);
      const targetArray = await zarr.open(targetLoc, { kind: 'array' });
      const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

      // Determine actual elements per vertex from target array shape
      const targetShape = targetArray.shape;
      const actualElementsPerVertex = targetShape.length === 2 ? targetShape[1] : 1;

      // Use RangeLoader for target
      const encoding = RangeLoader.detectEncoding(targetAttrs);
      log.info(Modules.SPATIAL_INDEX_LOADER, `Lines: Array ref target encoding: ${encoding}`);

      await this.rangeLoader.loadRanges(
        targetArray,
        targetAttrs,
        ranges as LoadRange[],
        output,
        totalVertices,
        actualElementsPerVertex
      );

      return output;
    }

    // Use RangeLoader for all other encodings (broadcasted, quantized, lut, direct)
    // Determine actual elements per vertex from array shape
    const shape = array.shape;
    const actualElementsPerVertex = shape.length === 2 ? shape[1] : 1;

    await this.rangeLoader.loadRanges(
      array,
      attrs,
      ranges as LoadRange[],
      output,
      totalVertices,
      actualElementsPerVertex
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

    const totalVertices = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const totalElements = totalVertices * 3; // RGB

    const attrs = array.attrs as unknown as ArrayMetadata;
    const isEncoded =
      ArrayDecoder.isQuantizedEncoding(attrs) ||
      ArrayDecoder.isLUTEncoded(attrs) ||
      ArrayDecoder.isBroadcasted(attrs);
    const isArrayRef = ArrayDecoder.isArrayRef(attrs);

    // For direct (unencoded) arrays, preserve native type
    if (!isEncoded && !isArrayRef) {
      const dtype = String(array.dtype);

      // Use target buffer if provided and type matches, otherwise allocate
      const expectedType = this.getExpectedColorType(dtype);
      const output =
        targetBuffer && this.colorBufferTypeMatches(targetBuffer, expectedType)
          ? targetBuffer
          : this.allocateColorBuffer(totalElements, false, dtype);

      // Load directly with type preservation
      await this.loadDirectColorRanges(array, ranges, output);
      return output;
    }

    // For encoded arrays or array_ref, decode to Float32Array then restore original_dtype
    const decodedFloat32 =
      targetBuffer instanceof Float32Array ? targetBuffer : new Float32Array(totalElements);

    if (isArrayRef) {
      // Resolve array_ref and load target
      const targetPath = attrs.encoding!.target!;
      const storeToUse = this.zarrStore || this.zarrLocation.store;
      const zarrRootLoc = zarr.root(storeToUse);
      const targetLoc = zarrRootLoc.resolve(targetPath);
      const targetArray = await zarr.open(targetLoc, { kind: 'array' });
      const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

      await this.rangeLoader.loadRanges(
        targetArray,
        targetAttrs,
        ranges as LoadRange[],
        decodedFloat32,
        totalVertices,
        3
      );

      // Restore original_dtype from array_ref encoding
      return this.restoreOriginalDtype(
        decodedFloat32,
        attrs.encoding?.original_dtype,
        totalElements
      );
    }

    // Use RangeLoader for encoded arrays
    const shape = array.shape;
    const actualElementsPerVertex = shape.length === 2 ? shape[1] : 1;

    await this.rangeLoader.loadRanges(
      array,
      attrs,
      ranges as LoadRange[],
      decodedFloat32,
      totalVertices,
      actualElementsPerVertex
    );

    // Restore original_dtype for encoded arrays
    return this.restoreOriginalDtype(decodedFloat32, attrs.encoding?.original_dtype, totalElements);
  }

  /**
   * Allocate color buffer based on dtype
   */
  private allocateColorBuffer(
    totalElements: number,
    isEncoded: boolean,
    dtype: string
  ): Float32Array | Uint8Array | Uint16Array {
    if (isEncoded) {
      return new Float32Array(totalElements);
    }
    if (dtype === 'uint8' || dtype === '|u1' || dtype === '<u1' || dtype === '>u1') {
      return new Uint8Array(totalElements);
    }
    if (dtype === 'uint16' || dtype === '|u2' || dtype === '<u2' || dtype === '>u2') {
      return new Uint16Array(totalElements);
    }
    return new Float32Array(totalElements);
  }

  /**
   * Get expected color buffer type from dtype string
   */
  private getExpectedColorType(dtype: string): 'Float32Array' | 'Uint8Array' | 'Uint16Array' {
    if (dtype === 'uint8' || dtype === '|u1' || dtype === '<u1' || dtype === '>u1') {
      return 'Uint8Array';
    }
    if (dtype === 'uint16' || dtype === '|u2' || dtype === '<u2' || dtype === '>u2') {
      return 'Uint16Array';
    }
    return 'Float32Array';
  }

  /**
   * Check if target buffer type matches expected type
   */
  private colorBufferTypeMatches(
    buffer: Float32Array | Uint8Array | Uint16Array,
    expectedType: 'Float32Array' | 'Uint8Array' | 'Uint16Array'
  ): boolean {
    if (expectedType === 'Uint8Array') return buffer instanceof Uint8Array;
    if (expectedType === 'Uint16Array') return buffer instanceof Uint16Array;
    return buffer instanceof Float32Array;
  }

  /**
   * Load direct (unencoded) color ranges with type preservation
   */
  private async loadDirectColorRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    ranges: SegmentRange[],
    output: Float32Array | Uint8Array | Uint16Array
  ): Promise<void> {
    let destOffset = 0;
    const shape = array.shape;

    for (const range of ranges) {
      const sliceSpec: zarr.Slice[] =
        shape.length === 2
          ? [slice(range.start, range.end), slice(null)]
          : [slice(range.start, range.end)];

      const chunkData = await get(array, sliceSpec);
      const data = chunkData.data;

      // Copy data preserving type (no conversion!)
      if (output instanceof Float32Array && data instanceof Float32Array) {
        output.set(data, destOffset);
      } else if (output instanceof Uint8Array && data instanceof Uint8Array) {
        output.set(data, destOffset);
      } else if (output instanceof Uint16Array && data instanceof Uint16Array) {
        output.set(data, destOffset);
      } else {
        // Fallback: convert values (not buffer reinterpretation!)
        const float32Data =
          data instanceof Float32Array ? data : new Float32Array(data as ArrayLike<number>);
        (output as Float32Array).set(float32Data, destOffset);
      }

      destOffset += (range.end - range.start) * 3;
    }
  }

  /**
   * Restore original dtype for encoded arrays
   */
  private restoreOriginalDtype(
    decodedFloat32: Float32Array,
    originalDtype: string | undefined,
    totalElements: number
  ): Float32Array | Uint8Array | Uint16Array {
    if (!originalDtype) {
      return decodedFloat32;
    }

    if (
      originalDtype === 'uint8' ||
      originalDtype === '|u1' ||
      originalDtype === '<u1' ||
      originalDtype === '>u1'
    ) {
      const uint8Output = new Uint8Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        uint8Output[i] = Math.round(Math.max(0, Math.min(255, decodedFloat32[i])));
      }
      return uint8Output;
    }

    if (
      originalDtype === 'uint16' ||
      originalDtype === '|u2' ||
      originalDtype === '<u2' ||
      originalDtype === '>u2'
    ) {
      const uint16Output = new Uint16Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        uint16Output[i] = Math.round(Math.max(0, Math.min(65535, decodedFloat32[i])));
      }
      return uint16Output;
    }

    // For float32/float64 or unspecified, keep as Float32Array
    return decodedFloat32;
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
    this.initPromise = null;

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
  }
}

// ============================================================================
// nD Clipping and Instance Buffer Construction
// ============================================================================

/**
 * Clip a segment to the current nD slice.
 *
 * Handles 5 visibility cases:
 * A: Both endpoints IN slice → Render full segment
 * B: P1 IN, P2 OUT → Clip P2 to slice boundary
 * C: P1 OUT, P2 IN → Clip P1 to slice boundary
 * D: Both OUT, opposite sides → Clip both (segment crosses slice)
 * E: Both OUT, same side → Don't render (segment misses slice)
 *
 * @param p1 - Start vertex (nD)
 * @param p2 - End vertex (nD)
 * @param slicePosition - Current slice position in nD
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display [d0, d1, d2]
 * @returns Clipped segment with interpolation parameters
 */
export function clipSegmentToSlice(
  p1: number[],
  p2: number[],
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ClippedSegment {
  let t1 = 0.0; // Parameter at start
  let t2 = 1.0; // Parameter at end

  for (let dim = 0; dim < p1.length; dim++) {
    if (displayDims.includes(dim)) continue; // Skip displayed dimensions

    const tol = tolerance[dim];
    const sliceMin = slicePosition[dim] - tol;
    const sliceMax = slicePosition[dim] + tol;

    const v1 = p1[dim];
    const v2 = p2[dim];

    // Classify endpoints relative to slice
    const p1In = v1 >= sliceMin && v1 <= sliceMax;
    const p2In = v2 >= sliceMin && v2 <= sliceMax;

    if (p1In && p2In) {
      // Both in - no clipping needed for this dimension
      continue;
    }

    if (!p1In && !p2In) {
      // Both out - check if on same side (Case E)
      if ((v1 < sliceMin && v2 < sliceMin) || (v1 > sliceMax && v2 > sliceMax)) {
        return { p1: [], p2: [], t1: 0, t2: 0, visible: false };
      }
      // Opposite sides - clip both (Case D)
    }

    // Compute intersection parameters
    const dv = v2 - v1;
    if (Math.abs(dv) < 1e-10) continue; // Parallel to slice

    // t where line crosses sliceMin and sliceMax
    const tMin = (sliceMin - v1) / dv;
    const tMax = (sliceMax - v1) / dv;

    // Clip t1 (entry) and t2 (exit) to valid range
    if (dv > 0) {
      // Moving from low to high
      t1 = Math.max(t1, tMin);
      t2 = Math.min(t2, tMax);
    } else {
      // Moving from high to low
      t1 = Math.max(t1, tMax);
      t2 = Math.min(t2, tMin);
    }

    if (t1 >= t2) {
      return { p1: [], p2: [], t1: 0, t2: 0, visible: false }; // No valid range
    }
  }

  // Interpolate clipped positions (in full nD space)
  const clippedP1 = p1.map((v, i) => v + t1 * (p2[i] - v));
  const clippedP2 = p1.map((v, i) => v + t2 * (p2[i] - v));

  // Project to 3D display space
  const display1 = displayDims.map((d) => clippedP1[d]);
  const display2 = displayDims.map((d) => clippedP2[d]);

  // Pad to 3D if fewer than 3 display dims
  while (display1.length < 3) display1.push(0);
  while (display2.length < 3) display2.push(0);

  return { p1: display1, p2: display2, t1, t2, visible: true };
}

/**
 * Linear interpolation between two values.
 */
export function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

/**
 * Linear interpolation between two 3-component vectors.
 */
export function lerpVec3(a: number[], b: number[], t: number): number[] {
  return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
}

/**
 * Calculate 3D Euclidean distance.
 */
export function distance3D(a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Build GPU instance buffers from loaded lines data.
 *
 * Transforms per-vertex data into per-segment instance attributes:
 * - Clips segments to nD slice
 * - Interpolates attributes for clipped endpoints
 * - Calculates 3D segment lengths
 * - Tracks which endpoints were clipped
 *
 * @param loadedData - Raw lines data from loader
 * @param slicePosition - Current position in nD space
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display
 * @returns Processed data ready for GPU
 */
export function buildInstanceBuffers(
  loadedData: LoadedLinesData,
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ProcessedLinesData {
  const { positions, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;

  // DEBUG: Log input data for particle tracks investigation
  if (segmentCount > 10000) {
    console.log('[DEBUG buildInstanceBuffers] Input:');
    console.log('  segmentCount:', segmentCount);
    console.log(
      '  positions.length:',
      positions.length,
      '(should be',
      segmentCount * 2 * ndim,
      ')'
    );
    console.log('  segments.length:', segments.length, '(should be', segmentCount * 2, ')');
    console.log('  ndim:', ndim);
    console.log('  slicePosition:', slicePosition);
    console.log('  tolerance:', tolerance);
    console.log('  displayDims:', displayDims);
    console.log('  First vertex:', Array.from(positions.slice(0, ndim)));
    console.log('  First segment indices:', segments[0], segments[1]);
  }

  // Pre-allocate output arrays (may be smaller after clipping)
  const maxSegments = segmentCount;
  const startPositions = new Float32Array(maxSegments * 3);
  const endPositions = new Float32Array(maxSegments * 3);
  const startColors = new Float32Array(maxSegments * 3);
  const endColors = new Float32Array(maxSegments * 3);
  const startWidths = new Float32Array(maxSegments);
  const endWidths = new Float32Array(maxSegments);
  const startSharpness = new Float32Array(maxSegments);
  const endSharpness = new Float32Array(maxSegments);
  const segmentLengths = new Float32Array(maxSegments);
  const startClipped = new Uint8Array(maxSegments);
  const endClipped = new Uint8Array(maxSegments);

  let outIdx = 0;
  let firstClippedReason = null;

  for (let i = 0; i < segmentCount; i++) {
    // Get vertex indices (local space)
    const v0 = segments[i * 2];
    const v1 = segments[i * 2 + 1];

    // Extract nD positions
    const p1 = Array.from(positions.slice(v0 * ndim, (v0 + 1) * ndim));
    const p2 = Array.from(positions.slice(v1 * ndim, (v1 + 1) * ndim));

    // DEBUG: Log first few segments for particle tracks
    if (segmentCount > 10000 && i < 3) {
      console.log(`  Segment ${i}: v0=${v0}, v1=${v1}, p1=[${p1}], p2=[${p2}]`);
    }

    // Clip to slice
    const clipped = clipSegmentToSlice(p1, p2, slicePosition, tolerance, displayDims);
    if (!clipped.visible) {
      if (segmentCount > 10000 && !firstClippedReason) {
        firstClippedReason = `Segment ${i} clipped: p1=[${p1}], p2=[${p2}]`;
      }
      continue;
    }

    // Write 3D positions
    startPositions.set(clipped.p1, outIdx * 3);
    endPositions.set(clipped.p2, outIdx * 3);

    // Interpolate and write colors
    const c0 = colors ? Array.from(colors.slice(v0 * 3, (v0 + 1) * 3)) : [1, 1, 1];
    const c1 = colors ? Array.from(colors.slice(v1 * 3, (v1 + 1) * 3)) : [1, 1, 1];
    const startC = lerpVec3(c0, c1, clipped.t1);
    const endC = lerpVec3(c0, c1, clipped.t2);
    startColors.set(startC, outIdx * 3);
    endColors.set(endC, outIdx * 3);

    // Interpolate widths
    const w0 = widths[v0];
    const w1 = widths[v1];
    startWidths[outIdx] = lerp(w0, w1, clipped.t1);
    endWidths[outIdx] = lerp(w0, w1, clipped.t2);

    // Interpolate sharpness (default 1.0 if not present)
    const s0 = sharpness ? sharpness[v0] : 1.0;
    const s1 = sharpness ? sharpness[v1] : 1.0;
    startSharpness[outIdx] = lerp(s0, s1, clipped.t1);
    endSharpness[outIdx] = lerp(s0, s1, clipped.t2);

    // Calculate 3D segment length
    segmentLengths[outIdx] = distance3D(clipped.p1, clipped.p2);

    // Track clipping for cap factor adjustment
    startClipped[outIdx] = clipped.t1 > 0 ? 1 : 0;
    endClipped[outIdx] = clipped.t2 < 1 ? 1 : 0;

    outIdx++;
  }

  // DEBUG: Log results for particle tracks
  if (segmentCount > 10000) {
    console.log('[DEBUG buildInstanceBuffers] Output:');
    console.log('  Input segments:', segmentCount, '→ Output segments:', outIdx);
    console.log('  Clipped rate:', ((1 - outIdx / segmentCount) * 100).toFixed(1), '%');
    if (firstClippedReason) {
      console.log('  First clip reason:', firstClippedReason);
    }
    if (outIdx > 0) {
      console.log('  First output segment:', {
        p1: Array.from(startPositions.slice(0, 3)),
        p2: Array.from(endPositions.slice(0, 3)),
      });
    }
  }

  // Trim arrays to actual size
  return {
    startPositions: startPositions.slice(0, outIdx * 3),
    endPositions: endPositions.slice(0, outIdx * 3),
    startColors: startColors.slice(0, outIdx * 3),
    endColors: endColors.slice(0, outIdx * 3),
    startWidths: startWidths.slice(0, outIdx),
    endWidths: endWidths.slice(0, outIdx),
    startSharpness: startSharpness.slice(0, outIdx),
    endSharpness: endSharpness.slice(0, outIdx),
    segmentLengths: segmentLengths.slice(0, outIdx),
    startClipped: startClipped.slice(0, outIdx),
    endClipped: endClipped.slice(0, outIdx),
    segmentCount: outIdx,
  };
}

/**
 * WASM-accelerated version of buildInstanceBuffers.
 *
 * Uses batch WASM functions for significantly faster nD clipping:
 * - clip_segments_batch: Process all segments at once
 * - interpolate_clipped_positions: Batch position interpolation
 * - interpolate_scalars_batch: Batch width/sharpness interpolation
 * - interpolate_colors_batch: Batch color interpolation
 * - calculate_segment_lengths: Batch length calculation
 * - mark_clipped_endpoints: Batch endpoint marking
 *
 * @param loadedData - Raw lines data from loader
 * @param slicePosition - Current position in nD space
 * @param tolerance - Per-dimension tolerance
 * @param displayDims - Which dimensions to display
 * @returns Processed data ready for GPU
 */
export function buildInstanceBuffersWASM(
  loadedData: LoadedLinesData,
  slicePosition: number[],
  tolerance: number[],
  displayDims: number[]
): ProcessedLinesData {
  const { positions, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;

  // Get WASM module (uses cached instance or fallback)
  const wasm = getWasmModuleSync();

  // Convert inputs to typed arrays for WASM
  const slicePosF32 = new Float32Array(slicePosition);
  const toleranceF32 = new Float32Array(tolerance);
  const displayDimsU32 = new Uint32Array(displayDims);

  // Phase 1: Batch clip all segments
  const visibility = new Uint8Array(segmentCount);
  const t1Params = new Float32Array(segmentCount);
  const t2Params = new Float32Array(segmentCount);

  const visibleCount = wasm.clip_segments_batch(
    positions,
    segments,
    slicePosF32,
    toleranceF32,
    displayDimsU32,
    ndim,
    segmentCount,
    visibility,
    t1Params,
    t2Params
  );

  // Early exit if no visible segments
  if (visibleCount === 0) {
    return {
      startPositions: new Float32Array(0),
      endPositions: new Float32Array(0),
      startColors: new Float32Array(0),
      endColors: new Float32Array(0),
      startWidths: new Float32Array(0),
      endWidths: new Float32Array(0),
      startSharpness: new Float32Array(0),
      endSharpness: new Float32Array(0),
      segmentLengths: new Float32Array(0),
      startClipped: new Uint8Array(0),
      endClipped: new Uint8Array(0),
      segmentCount: 0,
    };
  }

  // Phase 2: Allocate output buffers for visible segments
  const startPositions = new Float32Array(visibleCount * 3);
  const endPositions = new Float32Array(visibleCount * 3);
  const startColors = new Float32Array(visibleCount * 3);
  const endColors = new Float32Array(visibleCount * 3);
  const startWidths = new Float32Array(visibleCount);
  const endWidths = new Float32Array(visibleCount);
  const startSharpness = new Float32Array(visibleCount);
  const endSharpness = new Float32Array(visibleCount);
  const segmentLengths = new Float32Array(visibleCount);
  const startClipped = new Uint8Array(visibleCount);
  const endClipped = new Uint8Array(visibleCount);

  // Phase 3: Interpolate clipped positions to 3D
  wasm.interpolate_clipped_positions(
    positions,
    segments,
    visibility,
    t1Params,
    t2Params,
    displayDimsU32,
    ndim,
    segmentCount,
    startPositions,
    endPositions
  );

  // Phase 4: Interpolate colors
  if (colors) {
    // WASM expects Float32Array, convert and normalize if needed
    let colorsF32: Float32Array;
    if (colors instanceof Float32Array) {
      colorsF32 = colors;
    } else {
      // Normalize Uint8 (0-255) or Uint16 (0-65535) to Float32 (0-1)
      colorsF32 = new Float32Array(colors.length);
      const normFactor = colors instanceof Uint8Array ? 1 / 255 : 1 / 65535;
      for (let i = 0; i < colors.length; i++) {
        colorsF32[i] = colors[i] * normFactor;
      }
    }
    wasm.interpolate_colors_batch(
      colorsF32,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startColors,
      endColors
    );
  } else {
    // Default white color
    startColors.fill(1.0);
    endColors.fill(1.0);
  }

  // Phase 5: Interpolate widths
  wasm.interpolate_scalars_batch(
    widths,
    segments,
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startWidths,
    endWidths
  );

  // Phase 6: Interpolate sharpness
  if (sharpness) {
    wasm.interpolate_scalars_batch(
      sharpness,
      segments,
      visibility,
      t1Params,
      t2Params,
      segmentCount,
      startSharpness,
      endSharpness
    );
  } else {
    // Default sharpness of 1.0
    startSharpness.fill(1.0);
    endSharpness.fill(1.0);
  }

  // Phase 7: Calculate segment lengths
  wasm.calculate_segment_lengths(startPositions, endPositions, visibleCount, segmentLengths);

  // Phase 8: Mark clipped endpoints
  wasm.mark_clipped_endpoints(
    visibility,
    t1Params,
    t2Params,
    segmentCount,
    startClipped,
    endClipped
  );

  return {
    startPositions,
    endPositions,
    startColors,
    endColors,
    startWidths,
    endWidths,
    startSharpness,
    endSharpness,
    segmentLengths,
    startClipped,
    endClipped,
    segmentCount: visibleCount,
  };
}

/**
 * Initialize WASM module for hot path optimization.
 * Call this early in application startup to ensure WASM is ready when needed.
 */
export async function initLinesWASM(): Promise<void> {
  await getWasmModule();
  log.info(Modules.SPATIAL_INDEX_LOADER, 'WASM module initialized for lines clipping');
}
