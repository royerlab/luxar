/**
 * RangeLoader - Unified encoding dispatch for range-based array loading
 *
 * Single entry point shared by the points, lines, and gsplats spatial-index
 * loaders. Each encoding's body lives in a sibling file under
 * `./range-loader/`; this class is the dispatcher + context builder.
 *
 * @module data/loaders/spatial-query/range-loader
 */

import * as zarr from '../../zarr';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from '../../array-decoder/decoder';
import type { LoadRange } from '../base-types';
import {
  DEFAULT_RANGE_LOADER_CONFIG,
  type RangeLoaderConfig,
  type ResolvedRangeLoaderConfig,
  type EncodingType,
} from './range-loader/encoding-types';
import { detectEncoding } from './range-loader/detect-encoding';
import { loadBroadcasted } from './range-loader/broadcasted';
import { loadQuantized } from './range-loader/quantized';
import { loadLUT } from './range-loader/lut';
import { loadDirect } from './range-loader/direct';
import { loadArrayRef } from './range-loader/array-ref';
import { resolveArrayRef } from './range-loader/ref-resolution';

export type { LoadRange, RangeLoaderConfig, EncodingType };

export class RangeLoader {
  private decoder: ArrayDecoder;
  private config: ResolvedRangeLoaderConfig;
  private _verbose = true;

  constructor(refRegistry: ArrayRefRegistry, config: RangeLoaderConfig = {}) {
    this.decoder = new ArrayDecoder(refRegistry);
    this.config = { ...DEFAULT_RANGE_LOADER_CONFIG, ...config };
  }

  /** Suppress detail logs after initial load */
  setVerbose(verbose: boolean): void {
    this._verbose = verbose;
  }

  /** Detect encoding type from array metadata */
  static detectEncoding(attrs: ArrayMetadata | undefined): EncodingType {
    return detectEncoding(attrs);
  }

  /**
   * Load array ranges with automatic encoding dispatch. The main entry point;
   * replaces the duplicated loadRanges methods that lived in each spatial
   * index loader before the unification.
   */
  async loadRanges(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: ArrayMetadata | undefined,
    ranges: LoadRange[],
    output: Float32Array,
    totalElements: number,
    elementsPerItem: number = 1
  ): Promise<number> {
    const ctx = { config: this.config, verbose: this._verbose };
    switch (detectEncoding(attrs)) {
      case 'broadcasted':
        await loadBroadcasted(ctx, array, attrs!, output, totalElements, elementsPerItem);
        return totalElements * elementsPerItem;
      case 'quantized':
        return loadQuantized({ ...ctx, decoder: this.decoder }, array, attrs!, ranges, output);
      case 'lut':
        return loadLUT({ ...ctx, decoder: this.decoder }, array, attrs!, ranges, output);
      case 'array_ref':
        return loadArrayRef(ctx, attrs!);
      case 'direct':
      default:
        return loadDirect(ctx, array, ranges, output);
    }
  }

  /**
   * Like {@link loadRanges} but transparently resolves `array_ref` encodings
   * by opening the target array and delegating against it. The standard
   * entry point for spatial-index loaders.
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
    const ctx = { config: this.config, verbose: this._verbose };
    const resolved = await resolveArrayRef(ctx, attrs, zarrStore, logPrefix);
    return resolved
      ? this.loadRanges(
          resolved.array,
          resolved.attrs,
          ranges,
          output,
          totalElements,
          resolved.elementsPerItem
        )
      : this.loadRanges(array, attrs, ranges, output, totalElements, elementsPerItem);
  }

  /** Underlying ArrayDecoder for full-array operations */
  getDecoder(): ArrayDecoder {
    return this.decoder;
  }
}
