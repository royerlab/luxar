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
  private decoder: ArrayDecoder;
  private zarrStore: zarr.Readable | null = null;

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
    zarrStore?: zarr.Readable
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.decoder = new ArrayDecoder(refRegistry || new ArrayRefRegistry());
    this.zarrStore = zarrStore || null;
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
      this.arrays.vertices = await zarr.open(this.zarrLocation.resolve('vertices'), {
        kind: 'array',
      });
      this.arrays.segments = await zarr.open(this.zarrLocation.resolve('segments'), {
        kind: 'array',
      });
    } catch (e) {
      log.error(Modules.SPATIAL_INDEX_LOADER, 'Failed to open required Lines arrays:', e);
      throw e;
    }

    // Try to open optional arrays
    try {
      this.arrays.widths = await zarr.open(this.zarrLocation.resolve('widths'), { kind: 'array' });
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No widths array found (using default width)');
    }

    try {
      this.arrays.colors = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No colors array found (using default color)');
    }

    try {
      this.arrays.sharpness = await zarr.open(this.zarrLocation.resolve('sharpness'), {
        kind: 'array',
      });
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No sharpness array found (using default sharpness)');
    }
  }

  /**
   * Load lines data for the given view state
   */
  async loadLines(viewState: LinesViewState): Promise<LoadedLinesData> {
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

    // Phase 1: Query segment chunks and load segments
    const segmentRanges = this.queryVisibleSegmentRanges(viewState);

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
    const segmentData = await this.loadSegmentRanges(segmentRanges);

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

    // Load vertex data
    const vertices = await this.loadVertexRanges('vertices', mergedVertexRanges, attrs.ndim);
    const widths = this.arrays.widths
      ? await this.loadVertexRanges('widths', mergedVertexRanges, 1)
      : this.createDefaultWidths(sortedIndices.length);
    const colors = this.arrays.colors
      ? await this.loadVertexRanges('colors', mergedVertexRanges, 3)
      : null;
    const sharpness = this.arrays.sharpness
      ? await this.loadVertexRanges('sharpness', mergedVertexRanges, 1)
      : null;

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

    return {
      vertices,
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
   */
  async updateView(viewState: LinesViewState): Promise<LoadedLinesData> {
    return this.loadLines(viewState);
  }

  /**
   * Query visible segment ranges based on view state
   */
  private queryVisibleSegmentRanges(viewState: LinesViewState): SegmentRange[] {
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

    // Query segment chunks
    const chunkIndices = querySegmentChunksForView(this.chunkIndex, slicePosition, tolerance);

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
   * Load vertex attribute data for given ranges with optimized encoding handling.
   *
   * This method handles different encoding types efficiently:
   * - Broadcasted: Load single value, replicate to all vertices
   * - Quantized: Load only needed ranges of quantized data, dequantize those
   * - LUT: Load only needed ranges of indices, decode with LUT from metadata
   * - Array ref: Resolve target, apply optimized loading based on target encoding
   * - Direct: Load ranges directly from zarr
   *
   * CRITICAL: For encoded arrays, we load ONLY the needed ranges, not the full array!
   * This provides massive performance improvement for large animated datasets.
   */
  private async loadVertexRanges(
    arrayName: string,
    ranges: SegmentRange[],
    elementsPerVertex: number
  ): Promise<Float32Array> {
    const array = this.arrays[arrayName as keyof typeof this.arrays];
    if (!array) {
      throw new Error(`Array ${arrayName} not initialized`);
    }

    const totalVertices = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const totalElements = totalVertices * elementsPerVertex;

    // Check for different encoding types
    const attrs = array.attrs as unknown as ArrayMetadata;
    const isBroadcasted = ArrayDecoder.isBroadcasted(attrs);
    const isQuantized = ArrayDecoder.isQuantizedEncoding(attrs);
    const isLUTEncoded = ArrayDecoder.isLUTEncoded(attrs);
    const isArrayRef = ArrayDecoder.isArrayRef(attrs);

    // Determine actual elements per vertex from array shape
    const shape = array.shape;
    const actualElementsPerVertex = shape.length === 2 ? shape[1] : 1;

    const output = new Float32Array(totalElements);
    let destOffset = 0;

    if (isBroadcasted) {
      // Broadcasted encoding: Load single value and replicate to all vertices
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Lines: Broadcasted array ${arrayName}: replicating single value to ${totalVertices} vertices`
      );

      const fullData = await get(array);
      const broadcastValue = fullData.data as Float32Array | Uint8Array;

      for (let i = 0; i < totalVertices; i++) {
        for (let j = 0; j < actualElementsPerVertex; j++) {
          output[i * actualElementsPerVertex + j] = broadcastValue[j] || broadcastValue[0];
        }
      }
    } else if (isQuantized) {
      // Quantized encoding: Load only needed ranges of quantized data, then dequantize
      const quantMetadata = ArrayDecoder.getQuantizationMetadata(attrs);
      if (!quantMetadata) {
        throw new Error(
          `Lines: Quantization metadata missing for ${arrayName}. This should not happen if isQuantizedEncoding returned true.`
        );
      }

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Lines: Quantized range loading: ${arrayName} (${ArrayDecoder.getEncodingMode(attrs)} mode, loading ${totalVertices} values)`
      );

      // Load only the needed ranges of quantized data (NOT the full array!)
      for (const range of ranges) {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];

        const chunkData = await get(array, sliceSpec);
        const quantizedData = chunkData.data as Float32Array | Uint8Array | Uint16Array;

        const dequantized = this.decoder.dequantizeRange(quantizedData, quantMetadata);

        output.set(dequantized, destOffset);
        destOffset += dequantized.length;
      }
    } else if (isLUTEncoded) {
      // LUT encoding: Load only the range of indices, then decode with LUT from metadata
      const lutMetadata = ArrayDecoder.getLUTMetadata(attrs);
      if (!lutMetadata) {
        throw new Error(
          `Lines: LUT metadata missing for ${arrayName}. This should not happen if isLUTEncoded returned true.`
        );
      }

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Lines: LUT range loading: ${arrayName} (loading ${totalVertices} indices, decoding to ${totalElements} elements)`
      );

      // Load only the needed ranges of indices (NOT the full array!)
      for (const range of ranges) {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];

        const chunkData = await get(array, sliceSpec);
        const indices = chunkData.data as Float32Array | Uint8Array | Uint16Array;

        const decoded = this.decoder.decodeLUTIndices(indices, lutMetadata);

        output.set(decoded, destOffset);
        destOffset += decoded.length;
      }
    } else if (isArrayRef) {
      // Array reference: Resolve target and apply optimized range loading based on target encoding
      const targetPath = attrs.encoding!.target!;

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Lines: Array ref: ${arrayName} → ${targetPath} (resolving with optimized range loading)`
      );

      const storeToUse = this.zarrStore || this.zarrLocation.store;
      const zarrRootLoc = zarr.root(storeToUse);

      const targetLoc = zarrRootLoc.resolve(targetPath);
      const targetArray = await zarr.open(targetLoc, { kind: 'array' });
      const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

      if (ArrayDecoder.isQuantizedEncoding(targetAttrs)) {
        // Target is quantized → use quantized range loading!
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Lines: Array ref target is quantized (${ArrayDecoder.getEncodingMode(targetAttrs)})`
        );

        const quantMeta = ArrayDecoder.getQuantizationMetadata(targetAttrs);
        if (!quantMeta) {
          throw new Error(
            `Lines: Quantization metadata missing for array_ref target: ${targetPath}`
          );
        }

        for (const range of ranges) {
          const sliceSpec: zarr.Slice[] =
            targetArray.shape.length === 2
              ? [slice(range.start, range.end), slice(null)]
              : [slice(range.start, range.end)];

          const quantizedData = await get(targetArray, sliceSpec);
          const dequantized = this.decoder.dequantizeRange(
            quantizedData.data as Uint8Array | Uint16Array,
            quantMeta
          );

          output.set(dequantized, destOffset);
          destOffset += dequantized.length;
        }
      } else if (ArrayDecoder.isLUTEncoded(targetAttrs)) {
        // Target is LUT → use LUT range loading!
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Lines: Array ref target is LUT (${ArrayDecoder.getEncodingMode(targetAttrs)})`
        );

        const lutMetadata = ArrayDecoder.getLUTMetadata(targetAttrs);
        if (!lutMetadata) {
          throw new Error(`Lines: LUT metadata missing for array_ref target: ${targetPath}`);
        }

        for (const range of ranges) {
          const sliceSpec: zarr.Slice[] =
            targetArray.shape.length === 2
              ? [slice(range.start, range.end), slice(null)]
              : [slice(range.start, range.end)];

          const chunkData = await get(targetArray, sliceSpec);
          const decoded = this.decoder.decodeLUTIndices(
            chunkData.data as Float32Array | Uint8Array | Uint16Array,
            lutMetadata
          );

          output.set(decoded, destOffset);
          destOffset += decoded.length;
        }
      } else if (ArrayDecoder.isBroadcasted(targetAttrs)) {
        // Target is broadcasted → load once, replicate!
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          'Lines: Array ref target is broadcasted (loading single value)'
        );

        const fullData = await get(targetArray);
        const broadcastValue = fullData.data as Float32Array | Uint8Array | Uint16Array;

        for (let i = 0; i < totalVertices; i++) {
          for (let j = 0; j < actualElementsPerVertex; j++) {
            output[i * actualElementsPerVertex + j] = broadcastValue[j] || broadcastValue[0];
          }
        }
      } else {
        // Target is direct or unknown encoding → decode full target, extract ranges
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Lines: Array ref target is direct/unknown (${ArrayDecoder.getEncodingMode(targetAttrs)}) - using full decode`
        );

        const decoded = await this.decoder.decode(
          targetArray,
          targetAttrs,
          totalElements,
          zarrRootLoc
        );

        for (const range of ranges) {
          const rangeSize = (range.end - range.start) * actualElementsPerVertex;
          const srcOffset = range.start * actualElementsPerVertex;

          output.set(decoded.subarray(srcOffset, srcOffset + rangeSize), destOffset);
          destOffset += rangeSize;
        }
      }
    } else {
      // Direct arrays: load ranges directly from zarr
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Lines: Direct array loading: ${arrayName} (${totalVertices} vertices)`
      );

      for (const range of ranges) {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];

        const data = await get(array, sliceSpec);
        const floatData =
          data.data instanceof Float32Array
            ? data.data
            : new Float32Array(data.data as unknown as ArrayBufferLike);

        output.set(floatData, destOffset);
        destOffset += floatData.length;
      }
    }

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
   * Create empty lines data
   */
  private createEmptyLinesData(attrs: LinesMetadata): LoadedLinesData {
    return {
      vertices: new Float32Array(0),
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
   * Clean up resources
   */
  dispose(): void {
    this.chunkIndex = null;
    this.arrays = {};
    this.initPromise = null;
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
  const { vertices, segments, widths, colors, sharpness, ndim, segmentCount } = loadedData;

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

  for (let i = 0; i < segmentCount; i++) {
    // Get vertex indices (local space)
    const v0 = segments[i * 2];
    const v1 = segments[i * 2 + 1];

    // Extract nD positions
    const p1 = Array.from(vertices.slice(v0 * ndim, (v0 + 1) * ndim));
    const p2 = Array.from(vertices.slice(v1 * ndim, (v1 + 1) * ndim));

    // Clip to slice
    const clipped = clipSegmentToSlice(p1, p2, slicePosition, tolerance, displayDims);
    if (!clipped.visible) continue;

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
