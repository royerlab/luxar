/**
 * Shared preflight logging for the spatial-index loaders' query path.
 *
 * Each loader's queryVisible*Ranges method opens with the same two log
 * statements:
 *
 *   1. {@link warnExtendToAllNoDimensions} — fires UNCONDITIONALLY at the
 *      top of the query when `extend_to_all` is configured but the caller
 *      hasn't supplied resolved dimensions in the view state. The query
 *      proceeds anyway, but extend_to_all will not actually work until
 *      the scene dimensions are initialized.
 *
 *   2. {@link announceExtendToAllOnce} — fires only on the FIRST load and
 *      only AFTER the loader confirmed it has a usable chunk index. This
 *      is purely a one-time UX hint: surface the broadcast emoji so the
 *      console shows that this node is configured for extend_to_all.
 *
 * Splitting into two functions matches the existing call structure of
 * the loaders (warning → chunkIndex early-return → broadcast) so that
 * the broadcast doesn't fire on the load-all fallback path. Centralizing
 * keeps the wording, emoji, and "first-load only" semantics in one place.
 *
 * The "resolved dimensions" predicate differs slightly across loaders
 * (gsplats / lines check `dimensions.length`, points checks
 * `dimensions.metadata.length`), so callers pass it as a boolean.
 *
 * @module data/loaders/extend-to-all-preflight
 */

import { log, Modules, LogEmoji } from '../../utils/log';

type LogModule = (typeof Modules)[keyof typeof Modules];

/**
 * Warn (only) when extend_to_all is configured but the view state is
 * missing resolved dimensions. Silent when extendDims is empty or when
 * dimensions are present.
 */
export function warnExtendToAllNoDimensions(opts: {
  extendDims: string[];
  hasResolvedDimensions: boolean;
  nodePath: string;
  logModule: LogModule;
}): void {
  const { extendDims, hasResolvedDimensions, nodePath, logModule } = opts;
  if (extendDims.length === 0) return;
  if (hasResolvedDimensions) return;

  log.warning(
    logModule,
    `extend_to_all=[${extendDims.join(', ')}] specified for ${nodePath} but ` +
      'view state has no resolved scene dimensions. extend_to_all will not work. ' +
      'Ensure scene dimensions are initialized before loading nodes.'
  );
}

/**
 * Announce that this node is configured for `extend_to_all`. Intended to
 * fire exactly once per loader (gated by the caller's first-load flag).
 * Silent when extendDims is empty.
 */
export function announceExtendToAllOnce(opts: {
  extendDims: string[];
  nodePath: string;
  logModule: LogModule;
}): void {
  const { extendDims, nodePath, logModule } = opts;
  if (extendDims.length === 0) return;

  log.custom(
    LogEmoji.BROADCAST,
    logModule,
    `${nodePath} configured with extend_to_all: ${extendDims.join(', ')}`
  );
}
