/**
 * RangeLoader - Unified range-based array loading with encoding dispatch
 *
 * This module extracts the duplicated encoding logic from all spatial index loaders
 * (points, lines, gsplats) into a single reusable component.
 *
 * Features:
 * - Automatic encoding detection (broadcasted, quantized, LUT, array_ref, direct)
 * - Worker dispatch for CPU-intensive decoding
 * - Main thread fallback on worker failure
 * - Consistent threshold logic across all data types
 *
 * @module data/loaders/range-loader
 */

import * as zarr from 'zarrita';
import { get, slice } from 'zarrita';
import { log, Modules } from '../../utils/log';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from '../array-decoder';

/**
 * Range specification for loading array subsets
 */
export interface LoadRange {
  start: number;
  end: number;
}

/**
 * Configuration for range loading
 */
export interface RangeLoaderConfig {
  /** Minimum elements before using workers (default: 1000) */
  workerThreshold?: number;
  /** Log module for debug output */
  logModule?: string;
}

/**
 * Result of encoding detection
 */
export type EncodingType = 'broadcasted' | 'quantized' | 'lut' | 'array_ref' | 'direct';

/**
 * RangeLoader - Unified encoding dispatch for range-based loading
 *
 * Replaces duplicated loadBroadcastedRanges, loadQuantizedRanges, loadLUTRanges
 * methods across all spatial index loaders.
 */
export class RangeLoader {
  private decoder: ArrayDecoder;
  private config: Required<RangeLoaderConfig>;

  constructor(refRegistry: ArrayRefRegistry, config: RangeLoaderConfig = {}) {
    this.decoder = new ArrayDecoder(refRegistry);
    this.config = {
      workerThreshold: config.workerThreshold ?? 1000,
      logModule: config.logModule ?? Modules.SPATIAL_INDEX_LOADER,
    };
  }

  /**
   * Detect encoding type from array metadata
   */
  static detectEncoding(attrs: ArrayMetadata | undefined): EncodingType {
    if (!attrs?.encoding?.name) return 'direct';

    const enc = attrs.encoding;

    // Priority order (must match Python spec)
    if (enc.name === 'broadcasted') return 'broadcasted';
    if (enc.name === 'array_ref' && enc.target) return 'array_ref';
    if (enc.name?.startsWith('lut') && enc.lut) return 'lut';
    if (
      enc.name?.startsWith('log_scalar') ||
      enc.name?.includes('uint') ||
      enc.bounds ||
      (enc.min !== undefined && enc.max !== undefined)
    ) {
      return 'quantized';
    }

    return 'direct';
  }

  /**
   * Load array ranges with automatic encoding dispatch
   *
   * This is the main entry point that replaces the duplicated loadRanges methods
   * in each spatial index loader.
   *
   * @param array - Zarr array handle
   * @param attrs - Array metadata with encoding info
   * @param ranges - Ranges to load
   * @param output - Pre-allocated output buffer (for zero-allocation)
   * @param totalElements - Total elements expected
   * @param elementsPerItem - Elements per item (e.g., 3 for positions, 1 for radii)
   * @returns Number of elements written to output
   */
  async loadRanges(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    attrs: ArrayMetadata | undefined,
    ranges: LoadRange[],
    output: Float32Array,
    totalElements: number,
    elementsPerItem: number = 1
  ): Promise<number> {
    const encoding = RangeLoader.detectEncoding(attrs);

    switch (encoding) {
      case 'broadcasted':
        await this.loadBroadcasted(array, attrs!, output, totalElements, elementsPerItem);
        return totalElements * elementsPerItem;

      case 'quantized':
        return this.loadQuantized(array, attrs!, ranges, output);

      case 'lut':
        return this.loadLUT(array, attrs!, ranges, output);

      case 'array_ref':
        // Array ref needs special handling - resolve target and recurse
        return this.loadArrayRef(array, attrs!, ranges, output, totalElements, elementsPerItem);

      case 'direct':
      default:
        return this.loadDirect(array, ranges, output);
    }
  }

  /**
   * Load broadcasted array (single value replicated to all elements)
   */
  private async loadBroadcasted(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    _attrs: ArrayMetadata,
    output: Float32Array,
    totalElements: number,
    elementsPerItem: number
  ): Promise<void> {
    log.info(this.config.logModule, `Broadcasted: replicating to ${totalElements} elements`);

    // Fetch single value (cached via TwoLevelCachingStore)
    const fullData = await get(array);
    const broadcastValue = fullData.data as Float32Array | Uint8Array | Uint16Array;

    // Convert to Float32Array if needed
    const valueAsFloat32 =
      broadcastValue instanceof Float32Array ? broadcastValue : new Float32Array(broadcastValue);

    // Replicate on main thread (simple operation, not worth worker overhead)
    for (let i = 0; i < totalElements; i++) {
      for (let j = 0; j < elementsPerItem; j++) {
        output[i * elementsPerItem + j] = valueAsFloat32[j] ?? valueAsFloat32[0];
      }
    }
  }

  /**
   * Load quantized array ranges and dequantize
   *
   * OPTIMIZATION: Fetches all chunks in parallel, then decodes sequentially
   */
  private async loadQuantized(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    attrs: ArrayMetadata,
    ranges: LoadRange[],
    output: Float32Array
  ): Promise<number> {
    const quantMetadata = ArrayDecoder.getQuantizationMetadata(attrs);
    if (!quantMetadata) {
      throw new Error('Quantization metadata missing');
    }

    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    log.info(
      this.config.logModule,
      `Quantized: ${totalPoints} values, ${ranges.length} ranges (${ArrayDecoder.getEncodingMode(attrs)})`
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
    // NOTE: Decoding stays on main thread - workers only handle spatial queries
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
   * Load LUT-encoded array ranges and decode
   *
   * OPTIMIZATION: Fetches all chunks in parallel, then decodes sequentially
   */
  private async loadLUT(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    attrs: ArrayMetadata,
    ranges: LoadRange[],
    output: Float32Array
  ): Promise<number> {
    const lutMetadata = ArrayDecoder.getLUTMetadata(attrs);
    if (!lutMetadata) {
      throw new Error('LUT metadata missing');
    }

    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    log.info(
      this.config.logModule,
      `LUT: ${totalPoints} indices, ${ranges.length} ranges, k=${lutMetadata.k}, mode=${lutMetadata.lutMode}`
    );

    // NOTE: flatLUT not needed - decoder handles LUT internally

    // PARALLEL FETCH: Load all index chunks simultaneously (I/O parallelism)
    const chunkDataPromises = ranges.map((range) => {
      const sliceSpec: zarr.Slice[] = [slice(range.start, range.end)];
      return get(array, sliceSpec);
    });

    const chunks = await Promise.all(chunkDataPromises);

    // SEQUENTIAL DECODE: Process chunks in order (maintains correct destOffset)
    // NOTE: Decoding stays on main thread - workers only handle spatial queries
    let destOffset = 0;

    for (const chunkData of chunks) {
      const indices = chunkData.data as Uint8Array | Uint16Array;
      const decoded = this.decoder.decodeLUTIndices(indices, lutMetadata);

      output.set(decoded, destOffset);
      destOffset += decoded.length;
    }

    return destOffset;
  }

  /**
   * Load array reference (resolve target and recurse)
   */
  private async loadArrayRef(
    _array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    attrs: ArrayMetadata,
    _ranges: LoadRange[],
    _output: Float32Array,
    _totalElements: number,
    _elementsPerItem: number
  ): Promise<number> {
    const enc = attrs.encoding!;

    log.info(this.config.logModule, `Array ref: target=${enc.target}, hash=${enc.hash}`);

    // For now, array_ref handling requires the full ArrayDecoder flow
    // This is a placeholder - in practice, array_refs should be resolved
    // at initialization time and the target array used directly
    throw new Error(
      'Array reference range loading not yet implemented. ' +
        'Resolve array_ref at initialization and use target array directly.'
    );
  }

  /**
   * Load direct (unencoded) array ranges
   *
   * OPTIMIZATION: Fetches all chunks in parallel, then writes sequentially
   */
  private async loadDirect(
    array: zarr.Array<zarr.DataType, zarr.FetchStore>,
    ranges: LoadRange[],
    output: Float32Array
  ): Promise<number> {
    log.info(this.config.logModule, `Direct: loading ${ranges.length} ranges in parallel`);

    const shape = array.shape;

    // PARALLEL FETCH: Load all chunks simultaneously
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
      const data = chunkData.data;

      // Convert to Float32Array if needed (with value conversion, not buffer reinterpretation)
      let float32Data: Float32Array;
      if (data instanceof Float32Array) {
        float32Data = data;
      } else {
        // Convert from other types (uint8, uint16, etc.) to float32 with proper value conversion
        float32Data = new Float32Array(data as ArrayLike<number>);
      }

      output.set(float32Data, destOffset);
      destOffset += float32Data.length;
    }

    return destOffset;
  }

  /**
   * Get the underlying ArrayDecoder for full-array operations
   */
  getDecoder(): ArrayDecoder {
    return this.decoder;
  }
}

/**
 * Singleton RangeLoader instance for shared use
 */
let sharedRangeLoader: RangeLoader | null = null;
let sharedRefRegistry: ArrayRefRegistry | null = null;

/**
 * Get shared RangeLoader instance
 */
export function getSharedRangeLoader(): RangeLoader {
  if (!sharedRangeLoader) {
    sharedRefRegistry = new ArrayRefRegistry();
    sharedRangeLoader = new RangeLoader(sharedRefRegistry);
  }
  return sharedRangeLoader;
}

/**
 * Get shared ArrayRefRegistry
 */
export function getSharedRefRegistry(): ArrayRefRegistry {
  if (!sharedRefRegistry) {
    sharedRefRegistry = new ArrayRefRegistry();
  }
  return sharedRefRegistry;
}

/**
 * Reset shared instances (for testing)
 */
export function resetSharedRangeLoader(): void {
  sharedRangeLoader = null;
  sharedRefRegistry = null;
}
