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
 * If `attrs` is an array_ref, open the target array and return its shape-
 * derived metadata. Returns `null` when no resolution is needed (caller
 * should keep the original array/attrs).
 */
export async function resolveArrayRef(
  ctx: RefResolutionCtx,
  attrs: ArrayMetadata | undefined,
  zarrStore: zarr.Readable,
  logPrefix?: string
): Promise<ResolvedRefTarget | null> {
  if (!attrs || !ArrayDecoder.isArrayRef(attrs)) return null;

  const targetPath = attrs.encoding!.target!;
  if (ctx.verbose) {
    log.info(ctx.config.logModule, `${logPrefix ?? 'RangeLoader'}: Array ref → ${targetPath}`);
  }

  const targetLoc = zarr.root(zarrStore).resolve(targetPath);
  const targetArray = await zarr.open(targetLoc, { kind: 'array' });
  const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;
  const targetShape = targetArray.shape;
  const elementsPerItem =
    targetShape.length > 1
      ? targetShape.slice(1).reduce((product, value) => product * value, 1)
      : 1;

  return {
    array: targetArray as zarr.Array<zarr.DataType, zarr.Readable>,
    attrs: targetAttrs,
    elementsPerItem,
  };
}
