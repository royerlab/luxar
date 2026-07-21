/**
 * Shared view-state equality for the progressive (additive-LOD) loaders.
 *
 * The points / lines / gsplats progressive loaders each compared view
 * states with three byte-identical local copies (their `*ViewState`
 * types are all aliases of the same {@link ViewState}); the copies were
 * a documented drift risk — any new query-affecting field would have to
 * be added to all three or the memoized-noop skip silently serves stale
 * data for the geometry that missed it. One implementation lives here,
 * next to its siblings (`concat-helpers.ts`, `streaming-policy.ts`).
 *
 * @module data/loaders/progressive/view-state-equal
 */

import type { ViewState } from '../../data-loader-types';

/**
 * Element-wise view-state equality (query-affecting fields only).
 *
 * INVARIANT: this equality is the linchpin of the no-op commit skip AND
 * the append fast path's generation reset. When it reports equal AND no
 * new LODs loaded, the loader returns the MEMOIZED concatenation — same
 * object reference — and the commit pipeline treats reference equality
 * as content equality (`mesh.userData.committedData === data`); when it
 * reports unequal, the loader bumps its reset generation, which drops
 * the prefix lineage so the append gate full-rewrites. Any new
 * query-affecting field added to {@link ViewState} MUST be compared
 * here, or the skip serves stale data. Per-pass directives that are NOT
 * part of the query (e.g. `frameBudgetMs`, `prefetch`) must NOT be
 * compared — they would defeat the memoization. If a geometry ever
 * specializes its `*ViewState` alias (today all three are plain
 * aliases of {@link ViewState}) with its own query-affecting field,
 * that loader must fork this comparison — the shared version would
 * silently ignore the new field.
 */
export function viewStatesEqual(a: ViewState, b: ViewState): boolean {
  if (a.displayDims.length !== b.displayDims.length) return false;
  for (let i = 0; i < a.displayDims.length; i++) {
    if (a.displayDims[i] !== b.displayDims[i]) return false;
  }
  if (a.slicePosition.length !== b.slicePosition.length) return false;
  for (let i = 0; i < a.slicePosition.length; i++) {
    if (a.slicePosition[i] !== b.slicePosition[i]) return false;
  }
  if (a.tolerance.length !== b.tolerance.length) return false;
  for (let i = 0; i < a.tolerance.length; i++) {
    if (a.tolerance[i] !== b.tolerance[i]) return false;
  }
  // Dimensions metadata: reference equality first, JSON compare for the
  // rare content change (e.g. ranges populated after the first load).
  if (a.dimensions !== b.dimensions) {
    if (!a.dimensions || !b.dimensions) return false;
    if (JSON.stringify(a.dimensions) !== JSON.stringify(b.dimensions)) return false;
  }
  return true;
}
