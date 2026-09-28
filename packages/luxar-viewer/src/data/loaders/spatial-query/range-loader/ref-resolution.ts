import * as zarr from '../../../zarr';
import { log } from '../../../../utils/log';
import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import type { ResolvedRangeLoaderConfig } from './encoding-types';

export interface RefResolutionCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
}

export interface ResolvedRefTarget {
  array: zarr.Array<zarr.DataType, zarr.Readable>;
  attrs: ArrayMetadata;
  /** Per-item element count computed from the target array's shape. */
  elementsPerItem: number;
}

/**
 * How a loader wraps a resolved array_ref target — normally with its L0 proxy
 * (`wrapWithCache(array, l0, targetPath, hooks)`), so target reads hit the
 * decompressed-chunk cache exactly like the loader's own attribute arrays.
 * Injected (rather than imported) so this module stays free of the cache
 * module's config-reading static initializer.
 */
export interface RefTargetWrapper {
  /** Wrap the freshly opened target; `targetPath` is the store-absolute path. */
  wrap(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    targetPath: string
  ): zarr.Array<zarr.DataType, zarr.Readable>;
  /**
   * Changes whenever memoised targets must be re-opened — the L0 cache's
   * `generation`, bumped by `clear()` on invalidation / dispose.
   */
  epoch(): number;
}

interface MemoEntry {
  epoch: number;
  promise: Promise<ResolvedRefTarget>;
}

/**
 * Memo of resolved array_ref targets: one open + one wrap per (store, target
 * path), reused across updates. Before it, every update re-ran `zarr.open` on
 * the target and read it through an UNWRAPPED array, so each revisit re-fetched
 * and re-decoded the target's chunks. Invalidation: a new dataset is a new store
 * object (a `WeakMap` miss), and an L0 clear moves the wrapper's epoch. A failed
 * open is not memoised.
 */
export class RefTargetMemo {
  private byStore = new WeakMap<object, Map<string, MemoEntry>>();
  private wrapper: RefTargetWrapper | null = null;

  /** Install (or remove) the wrapper; drops everything memoised so far. */
  setWrapper(wrapper: RefTargetWrapper | null): void {
    this.wrapper = wrapper;
    this.byStore = new WeakMap();
  }

  /**
   * If `attrs` is an array_ref, return its (memoised) target and shape-derived
   * metadata. Returns `null` when no resolution is needed (caller should keep
   * the original array/attrs).
   */
  resolve(
    ctx: RefResolutionCtx,
    attrs: ArrayMetadata | undefined,
    zarrStore: zarr.Readable,
    logPrefix?: string
  ): Promise<ResolvedRefTarget | null> {
    if (!attrs || !ArrayDecoder.isArrayRef(attrs)) return Promise.resolve(null);
    const targetPath = attrs.encoding!.target!;
    const epoch = this.wrapper?.epoch() ?? 0;
    let perStore = this.byStore.get(zarrStore);
    if (!perStore) {
      perStore = new Map();
      this.byStore.set(zarrStore, perStore);
    }
    const memo = perStore.get(targetPath);
    if (memo && memo.epoch === epoch) return memo.promise;

    const promise = openRefTarget(ctx, targetPath, zarrStore, this.wrapper, logPrefix);
    const entry: MemoEntry = { epoch, promise };
    perStore.set(targetPath, entry);
    const store = perStore;
    promise.catch(() => {
      if (store.get(targetPath) === entry) store.delete(targetPath);
    });
    return promise;
  }
}

/** Store-absolute form of an array_ref target, used as the L0 array path. */
function absolutePath(targetPath: string): string {
  return targetPath.startsWith('/') ? targetPath : `/${targetPath}`;
}

async function openRefTarget(
  ctx: RefResolutionCtx,
  targetPath: string,
  zarrStore: zarr.Readable,
  wrapper: RefTargetWrapper | null,
  logPrefix?: string
): Promise<ResolvedRefTarget> {
  if (ctx.verbose) {
    log.info(ctx.config.logModule, `${logPrefix ?? 'RangeLoader'}: Array ref → ${targetPath}`);
  }

  const targetLoc = zarr.root(zarrStore).resolve(targetPath);
  const targetArray = (await zarr.open(targetLoc, { kind: 'array' })) as zarr.Array<
    zarr.DataType,
    zarr.Readable
  >;
  const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;
  const targetShape = targetArray.shape;
  const elementsPerItem =
    targetShape.length > 1
      ? targetShape.slice(1).reduce((product, value) => product * value, 1)
      : 1;

  return {
    array: wrapper ? wrapper.wrap(targetArray, absolutePath(targetPath)) : targetArray,
    attrs: targetAttrs,
    elementsPerItem,
  };
}

/**
 * If `attrs` is an array_ref, open the target array and return its shape-
 * derived metadata. Returns `null` when no resolution is needed (caller
 * should keep the original array/attrs). Un-memoised and unwrapped — prefer a
 * {@link RefTargetMemo} (what `RangeLoader.loadRangesResolvingRef` uses).
 */
export async function resolveArrayRef(
  ctx: RefResolutionCtx,
  attrs: ArrayMetadata | undefined,
  zarrStore: zarr.Readable,
  logPrefix?: string
): Promise<ResolvedRefTarget | null> {
  if (!attrs || !ArrayDecoder.isArrayRef(attrs)) return null;
  return openRefTarget(ctx, attrs.encoding!.target!, zarrStore, null, logPrefix);
}
