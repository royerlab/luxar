import { log } from '../../../../utils/log';
import type { ArrayMetadata } from '../../../array-decoder/decoder';
import type { ResolvedRangeLoaderConfig } from './encoding-types';

export interface ArrayRefCtx {
  config: ResolvedRangeLoaderConfig;
  verbose: boolean;
}

/**
 * Array refs must be resolved by spatial-index loaders before reaching
 * RangeLoader. Reaching this function indicates a code path that bypasses
 * the resolution. See `isArrayRef` checks in points/lines/gsplats spatial-
 * index loaders.
 */
export function loadArrayRef(ctx: ArrayRefCtx, attrs: ArrayMetadata): never {
  const enc = attrs.encoding!;

  if (ctx.verbose) {
    log.info(ctx.config.logModule, `Array ref: target=${enc.target}, hash=${enc.hash}`);
  }

  throw new Error(
    'Array reference encountered in RangeLoader but not pre-resolved. ' +
      `target=${enc.target}, hash=${enc.hash}. ` +
      'Array refs must be resolved by the spatial index loader before calling RangeLoader.'
  );
}
