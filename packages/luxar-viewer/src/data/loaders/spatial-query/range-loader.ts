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
  type DirectOutputBuffer,
  type RangeLoaderConfig,
  type ResolvedRangeLoaderConfig,
  type EncodingType,
} from './range-loader/encoding-types';
import { detectEncoding } from './range-loader/detect-encoding';
import { loadBroadcasted } from './range-loader/broadcasted';
import { loadQuantized } from './range-loader/quantized';
import { loadLUT } from './range-loader/lut';
import { loadPerChannel } from './range-loader/perchannel';
import { loadDirect } from './range-loader/direct';
import { loadArrayRef } from './range-loader/array-ref';
import { RefTargetMemo, type RefTargetWrapper } from './range-loader/ref-resolution';
import { isVerboseLogging } from '../../../utils/log';

export type { LoadRange, RangeLoaderConfig, EncodingType, RefTargetWrapper };

export class RangeLoader {
  private decoder: ArrayDecoder;
  private config: ResolvedRangeLoaderConfig;
  private _verbose = true;
  // Source of the owning loader's abort signal. Resolved once at
  // the top of each loadRanges call and forwarded into the worker-decode
  // calls so a superseded update's LUT/quantized/broadcasted decode bails.
  private _getSignal?: () => AbortSignal | null;
  // Resolved array_ref targets, opened once per (store, target path) and
  // wrapped through the owning loader's L0 wrapper (see setRefTargetWrapper).
  private readonly refTargets = new RefTargetMemo();

  constructor(refRegistry: ArrayRefRegistry, config: RangeLoaderConfig = {}) {
    this.decoder = new ArrayDecoder(refRegistry);
    this.config = { ...DEFAULT_RANGE_LOADER_CONFIG, ...config };
  }

  /** Suppress detail logs after initial load */
  setVerbose(verbose: boolean): void {
    this._verbose = verbose;
  }

  /**
   * Whether this load prints its per-array decode detail: on a loader's first
   * load, and only under `?verboseLog` — every partition part is a loader, and
   * a slice step can bring several into the slice for the first time.
   */
  private get verbose(): boolean {
    return this._verbose && isVerboseLogging();
  }

  /**
   * Wire the owning loader's abort signal source. A per-call signal wins when
   * present; an initial build uses the loader lifetime signal, so disposal
   * also settles its worker decodes (see WorkerPool.runWithTimeout).
   */
  setSignalSource(getSignal: () => AbortSignal | null): void {
    this._getSignal = getSignal;
  }

  /**
   * Wire how resolved array_ref targets are wrapped — the owning loader passes
   * its L0 proxy (`wrapWithCache` with its probe/signal hooks) and the L0
   * cache's `generation` as the epoch, so target reads are L0-cached like the
   * loader's own arrays and re-opened after an L0 clear. `null` (or never
   * calling this) reads targets unwrapped; they are still opened only once.
   */
  setRefTargetWrapper(wrapper: RefTargetWrapper | null): void {
    this.refTargets.setWrapper(wrapper);
  }

  /**
   * Read direct (unencoded) ranges into a caller-allocated, natively-typed
   * `output` buffer — the single entry point for direct reads that branch
   * BEFORE encoding dispatch (Points non-color attributes, the direct/raw-RGB
   * color path, Lines segments). Preserves `output`'s dtype (see `loadDirect`)
   * and sources the per-update abort signal internally, so callers never
   * thread a signal themselves.
   */
  async loadDirectTyped(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    ranges: LoadRange[],
    output: DirectOutputBuffer
  ): Promise<number> {
    const ctx = { config: this.config, verbose: this.verbose, signal: this._getSignal?.() };
    return loadDirect(ctx, array, ranges, output);
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
    const ctx = { config: this.config, verbose: this.verbose, signal: this._getSignal?.() };
    switch (detectEncoding(attrs)) {
      case 'broadcasted':
        await loadBroadcasted(ctx, array, attrs!, output, totalElements, elementsPerItem);
        return totalElements * elementsPerItem;
      case 'quantized':
        return loadQuantized({ ...ctx, decoder: this.decoder }, array, attrs!, ranges, output);
      case 'lut':
        return loadLUT({ ...ctx, decoder: this.decoder }, array, attrs!, ranges, output);
      case 'perchannel':
        return loadPerChannel(ctx, array, attrs!, ranges, output, elementsPerItem);
      case 'array_ref':
        return loadArrayRef(ctx, attrs!);
      case 'direct':
      default:
        return loadDirect(ctx, array, ranges, output);
    }
  }

  /**
   * Like {@link loadRanges} but transparently resolves `array_ref` encodings
   * by delegating against the target array — opened (and L0-wrapped, see
   * {@link setRefTargetWrapper}) once per (store, target path), not per call.
   * The standard entry point for spatial-index loaders.
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
    const ctx = { config: this.config, verbose: this.verbose };
    const resolved = await this.refTargets.resolve(ctx, attrs, zarrStore, logPrefix);
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
