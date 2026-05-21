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

import * as zarr from '../zarr';
import { get, slice } from '../zarr';
import { log, Modules } from '../../utils/log';
import { config as appConfig } from '../../config';
import { getWorkerPool } from '../../workers/worker-pool';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from '../array-decoder/decoder';
import type { LoadRange } from './base-types';

export type { LoadRange };

/**
 * Configuration for range loading
 */
export interface RangeLoaderConfig {
  /** Minimum elements before using workers (default: 1000) */
  workerThreshold?: number;
  /** Log module for debug output */
  logModule?: string;
}

type RangeNumericArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigUint64Array
  | BigInt64Array;

function numericArrayToFloat32(data: RangeNumericArray): Float32Array {
  if (data instanceof Float32Array) return data;
  if (typeof BigUint64Array !== 'undefined' && data instanceof BigUint64Array) {
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) result[i] = Number(data[i]);
    return result;
  }
  if (typeof BigInt64Array !== 'undefined' && data instanceof BigInt64Array) {
    const result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) result[i] = Number(data[i]);
    return result;
  }
  return new Float32Array(data as ArrayLike<number>);
}

function firstAxisRangeSlice(shape: readonly number[], range: LoadRange): zarr.Slice[] {
  return [slice(range.start, range.end), ...shape.slice(1).map(() => slice(null))];
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
  private _verbose = true;

  constructor(refRegistry: ArrayRefRegistry, config: RangeLoaderConfig = {}) {
    this.decoder = new ArrayDecoder(refRegistry);
    this.config = {
      workerThreshold: config.workerThreshold ?? 1000,
      logModule: config.logModule ?? Modules.SPATIAL_INDEX_LOADER,
    };
  }

  /** Suppress detail logs after initial load */
  setVerbose(verbose: boolean): void {
    this._verbose = verbose;
  }

  /**
   * Detect encoding type from array metadata
   */
  static detectEncoding(attrs: ArrayMetadata | undefined): EncodingType {
    if (!attrs?.encoding) return 'direct';

    const enc = attrs.encoding;
    ArrayDecoder.validateEncodingMetadata(enc);

    // Priority order (must match Python spec)
    if (enc.name === 'broadcasted') return 'broadcasted';
    if (enc.name === 'array_ref') return 'array_ref';
    if (ArrayDecoder.isLUTEncodingName(enc.name)) return 'lut';
    if (ArrayDecoder.isQuantizedEncoding(attrs)) return 'quantized';
    if (ArrayDecoder.isDirectEncodingName(enc.name)) return 'direct';

    throw new Error(`Unknown encoding name: ${enc.name}`);
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
    array: zarr.Array<zarr.DataType, zarr.Readable>,
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
   * Like {@link loadRanges} but transparently resolves `array_ref` encodings
   * by opening the target array and delegating to `loadRanges` against it.
   *
   * `loadRanges` itself rejects unresolved refs as a defensive check — every
   * caller of `loadRanges` directly should pre-resolve. This method is the
   * standard entry point for the spatial-index loaders, which all need the
   * same resolve-or-passthrough behavior. Pass the original `zarrStore` so
   * the target path can be resolved relative to the dataset root.
   *
   * The target's true `elementsPerItem` is recomputed from its shape (the
   * caller's `elementsPerItem` is used only when no ref is in play). Logs
   * the redirect once at INFO when verbose, then again from the underlying
   * encoding-specific loader.
   *
   * @param array - The directly-attached array (may be an array_ref).
   * @param attrs - That array's metadata.
   * @param ranges - Ranges to load.
   * @param output - Pre-allocated output buffer.
   * @param totalElements - Item count (e.g. number of points/splats).
   * @param elementsPerItem - Used only when no ref is present.
   * @param zarrStore - Store used to resolve the target path on a ref.
   * @param logPrefix - Optional caller tag for the "Array ref → target" line
   *   (e.g. "Points", "Lines"). Falls back to "RangeLoader" when omitted.
   */
  async loadRangesResolvingRef(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata | undefined,
    ranges: LoadRange[],
    output: Float32Array,
    totalElements: number,
    elementsPerItem: number,
    zarrStore: zarr.Readable,
    logPrefix?: string
  ): Promise<number> {
    if (attrs && ArrayDecoder.isArrayRef(attrs)) {
      const targetPath = attrs.encoding!.target!;
      if (this._verbose) {
        log.info(this.config.logModule, `${logPrefix ?? 'RangeLoader'}: Array ref → ${targetPath}`);
      }

      const targetLoc = zarr.root(zarrStore).resolve(targetPath);
      const targetArray = await zarr.open(targetLoc, { kind: 'array' });
      const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

      // The target's per-item element count is whatever the target array
      // says — the caller's hint applies only to the unresolved direct case.
      const targetShape = targetArray.shape;
      const targetElementsPerItem =
        targetShape.length > 1
          ? targetShape.slice(1).reduce((product, value) => product * value, 1)
          : 1;

      return this.loadRanges(
        targetArray as zarr.Array<zarr.DataType, zarr.Readable>,
        targetAttrs,
        ranges,
        output,
        totalElements,
        targetElementsPerItem
      );
    }

    return this.loadRanges(array, attrs, ranges, output, totalElements, elementsPerItem);
  }

  /**
   * Load broadcasted array (single value replicated to all elements)
   */
  private async loadBroadcasted(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata,
    output: Float32Array,
    totalElements: number,
    elementsPerItem: number
  ): Promise<void> {
    const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
    const arrayName = attrs.encoding?.name || 'broadcasted';

    if (this._verbose) {
      log.info(
        this.config.logModule,
        `Broadcasted: replicating to ${totalElements} elements (worker=${useWorkers})`
      );
    }

    // Fetch single value (cached via MultiLevelCachingStore)
    const fullData = await get(array);
    const broadcastValue = fullData.data as Float32Array | Uint8Array | Uint16Array;

    // Convert to Float32Array if needed
    const valueAsFloat32 =
      broadcastValue instanceof Float32Array ? broadcastValue : new Float32Array(broadcastValue);

    if (useWorkers && totalElements > this.config.workerThreshold) {
      try {
        const decoded = await getWorkerPool().runWithTimeout('decodeBroadcasted', 'decode', (api) =>
          api.decodeBroadcasted({
            value: valueAsFloat32,
            numPoints: totalElements,
            elementsPerPoint: elementsPerItem,
          })
        );
        output.set(decoded);
        return;
      } catch (error) {
        if (error instanceof Error && error.name === 'WorkerAbortError') {
          throw error;
        }
        log.warning(
          this.config.logModule,
          `Worker broadcast failed for ${arrayName}, falling back to main thread:`,
          error
        );
        // Fall through to main thread
      }
    }

    // Main thread replication
    for (let i = 0; i < totalElements; i++) {
      for (let j = 0; j < elementsPerItem; j++) {
        output[i * elementsPerItem + j] = valueAsFloat32[j] ?? valueAsFloat32[0];
      }
    }
  }

  /**
   * Load quantized array ranges and dequantize
   */
  private async loadQuantized(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata,
    ranges: LoadRange[],
    output: Float32Array
  ): Promise<number> {
    // Pass actual zarr dtype to avoid attrs.dtype bug (Python encoder doesn't write it)
    const zarrDtype = String(array.dtype);
    const quantMetadata = ArrayDecoder.getQuantizationMetadata(attrs, zarrDtype);
    if (!quantMetadata) {
      throw new Error('[RangeLoader] Quantization metadata missing');
    }

    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
    // Decide once whether to use workers based on total points across all ranges
    const shouldUseWorkers = useWorkers && totalPoints > this.config.workerThreshold;

    if (this._verbose) {
      log.info(
        this.config.logModule,
        `Quantized: ${totalPoints} values (${ArrayDecoder.getEncodingMode(attrs)}, dtype=${quantMetadata.dtype}, worker=${shouldUseWorkers})`
      );
    }

    let destOffset = 0;
    const shape = array.shape;

    for (const range of ranges) {
      const sliceSpec = firstAxisRangeSlice(shape, range);

      // Main thread fetches (cached via MultiLevelCachingStore)
      const chunkData = await get(array, sliceSpec);
      const quantizedData = chunkData.data as Uint8Array | Uint16Array;

      let dequantized: Float32Array;

      if (shouldUseWorkers) {
        try {
          if (quantMetadata.isLogSpace) {
            dequantized = await getWorkerPool().runWithTimeout('decodeLogScalar', 'decode', (api) =>
              api.decodeLogScalar({
                data: quantizedData,
                maxLog: quantMetadata.bounds[1],
                dtype: quantMetadata.dtype, // Already normalized by getQuantizationMetadata
              })
            );
          } else {
            dequantized = await getWorkerPool().runWithTimeout('decodeQuantized', 'decode', (api) =>
              api.decodeQuantized({
                data: quantizedData,
                bounds: quantMetadata.bounds,
                dtype: quantMetadata.dtype, // Already normalized by getQuantizationMetadata
              })
            );
          }
        } catch (error) {
          if (error instanceof Error && error.name === 'WorkerAbortError') {
            throw error;
          }
          log.warning(
            this.config.logModule,
            'Worker decoding failed, falling back to main thread:',
            error
          );
          dequantized = this.decoder.dequantizeRange(quantizedData, quantMetadata);
        }
      } else {
        dequantized = this.decoder.dequantizeRange(quantizedData, quantMetadata);
      }

      output.set(dequantized, destOffset);
      destOffset += dequantized.length;
    }

    return destOffset;
  }

  /**
   * Load LUT-encoded array ranges and decode
   */
  private async loadLUT(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata,
    ranges: LoadRange[],
    output: Float32Array
  ): Promise<number> {
    const lutMetadata = ArrayDecoder.getLUTMetadata(attrs);
    if (!lutMetadata) {
      throw new Error('[RangeLoader] LUT metadata missing');
    }

    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const useWorkers = appConfig.dataLoading.performance.useWebWorkers;
    // Decide once whether to use workers based on total points across all ranges
    const shouldUseWorkers = useWorkers && totalPoints > this.config.workerThreshold;

    if (this._verbose) {
      log.info(
        this.config.logModule,
        `LUT: ${totalPoints} indices, k=${lutMetadata.k}, mode=${lutMetadata.lutMode} (worker=${shouldUseWorkers})`
      );
    }

    // Flatten LUT once (shared across all ranges)
    const flatLUT: number[] = Array.isArray(lutMetadata.lut[0])
      ? (lutMetadata.lut as number[][]).flat()
      : (lutMetadata.lut as number[]);

    let destOffset = 0;
    const shape = array.shape;

    for (const range of ranges) {
      // Handle both 1D and 2D LUT-encoded arrays:
      // - 1D: row mode with one index per row (e.g., colors [N])
      // - 2D: scalar mode with one index per element (e.g., cholesky_factors [N, K])
      const sliceSpec = firstAxisRangeSlice(shape, range);

      // Main thread fetches indices
      const chunkData = await get(array, sliceSpec);
      const indices = chunkData.data as Uint8Array | Uint16Array;

      let decoded: Float32Array;

      if (shouldUseWorkers) {
        try {
          decoded = await getWorkerPool().runWithTimeout('decodeLUT', 'decode', (api) =>
            api.decodeLUT({
              indices,
              lut: flatLUT,
              k: lutMetadata.k,
              lutMode: lutMetadata.lutMode as 'row' | 'scalar',
            })
          );
        } catch (error) {
          if (error instanceof Error && error.name === 'WorkerAbortError') {
            throw error;
          }
          log.warning(
            this.config.logModule,
            'Worker LUT decode failed, falling back to main thread:',
            error
          );
          decoded = this.decoder.decodeLUTIndices(indices, lutMetadata);
        }
      } else {
        decoded = this.decoder.decodeLUTIndices(indices, lutMetadata);
      }

      output.set(decoded, destOffset);
      destOffset += decoded.length;
    }

    return destOffset;
  }

  /**
   * Load array reference (resolve target and recurse).
   *
   * This method should never be reached in practice. All spatial index loaders
   * (points-spatial-index-loader, lines-spatial-index-loader, gsplats-spatial-index-loader)
   * check for array_ref encoding via `ArrayDecoder.isArrayRef(attrs)` BEFORE calling
   * RangeLoader, and resolve the target array themselves using the zarrStore.
   *
   * If this is reached, it indicates a code path that bypasses the spatial index
   * loaders' array_ref resolution. Check the call stack to find the missing resolution.
   */
  private async loadArrayRef(
    _array: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata,
    _ranges: LoadRange[],
    _output: Float32Array,
    _totalElements: number,
    _elementsPerItem: number
  ): Promise<number> {
    const enc = attrs.encoding!;

    if (this._verbose) {
      log.info(this.config.logModule, `Array ref: target=${enc.target}, hash=${enc.hash}`);
    }

    // Array refs are resolved by spatial index loaders before reaching RangeLoader.
    // See isArrayRef checks in points-spatial-index-loader.ts, lines-spatial-index-loader.ts,
    // and gsplats-spatial-index-loader.ts. If this error is thrown, a new code path is
    // calling RangeLoader.loadRanges() without first resolving the array_ref.
    throw new Error(
      'Array reference encountered in RangeLoader but not pre-resolved. ' +
        'target=' +
        enc.target +
        ', hash=' +
        enc.hash +
        '. ' +
        'Array refs must be resolved by the spatial index loader before calling RangeLoader.'
    );
  }

  /**
   * Load direct (unencoded) array ranges
   */
  private async loadDirect(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    ranges: LoadRange[],
    output: Float32Array
  ): Promise<number> {
    if (this._verbose) {
      log.info(this.config.logModule, `Direct: loading ${ranges.length} ranges`);
    }

    let destOffset = 0;
    const shape = array.shape;

    for (const range of ranges) {
      const sliceSpec = firstAxisRangeSlice(shape, range);

      const chunkData = await get(array, sliceSpec);
      const data = chunkData.data;

      // Convert from typed storage (uint8/uint16/uint32/uint64/etc.) to
      // float32 with value conversion, not buffer reinterpretation.
      const float32Data = numericArrayToFloat32(data as RangeNumericArray);

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
 * Singleton RangeLoader instance for shared use.
 * Uses a single ArrayRefRegistry so all loaders share the same ref cache.
 */
let sharedRangeLoader: RangeLoader | null = null;
let sharedRefRegistry: ArrayRefRegistry | null = null;

/**
 * Get shared RangeLoader instance.
 *
 * @param registry - Optional ArrayRefRegistry to use. If provided on first call,
 *                   it becomes the shared registry. Subsequent calls ignore this
 *                   parameter (singleton is already created). Pass a registry when
 *                   you need the RangeLoader to share a registry with other components.
 */
export function getSharedRangeLoader(registry?: ArrayRefRegistry): RangeLoader {
  if (!sharedRangeLoader) {
    sharedRefRegistry = registry ?? new ArrayRefRegistry();
    sharedRangeLoader = new RangeLoader(sharedRefRegistry);
  }
  return sharedRangeLoader;
}

/**
 * Get shared ArrayRefRegistry (creates one if needed).
 * Prefer passing a registry to getSharedRangeLoader() instead of using this directly.
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
