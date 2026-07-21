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
import type { DimensionMetadata } from '../../../types/dims';

/**
 * Canonical QUERY-DETERMINANT projection of the dimensions metadata,
 * mirroring the slice-cache key (`buildSliceViewSig`): per dimension,
 * only the fields that can change which elements a fixed
 * `displayDims`/`slicePosition`/`tolerance` query loads or how they
 * project —
 * - every dim: `name` (drives `extend_to_all` tolerance matching);
 * - non-displayed dims additionally: `discrete`/`spatial` (tolerance
 *   membership role), `step` (discrete half-cell gate), `cyclic`
 *   (wrap behavior). Displayed dims are driven entirely by the
 *   separately-compared position/tolerance arrays; their `step` etc.
 *   feed keyboard navigation, not the query.
 *
 * Deliberately EXCLUDED: `range`, `display`, `unit`, `scale`,
 * `description` — display/navigation metadata the cache determinant
 * also ignores. This projection (fixed array shape, not raw
 * `JSON.stringify` of the objects) exists because the scene REBUILDS
 * the dimensions metadata right after the first data load —
 * dropping the `range: null` key, deriving `step: null → 1` on
 * displayed dims, and reordering object keys. Comparing the raw JSON
 * made every progressive loader reset its generation once per
 * dataset load, discarding the ladder prefix (re-streamed from warm
 * cache) and the append-fast-path lineage for a query-identical view.
 */
function dimsQuerySig(view: ViewState): string {
  const displayed = new Set(view.displayDims);
  return JSON.stringify(
    (view.dimensions ?? []).map((m: DimensionMetadata | undefined, i: number) =>
      displayed.has(i)
        ? [m?.name ?? null]
        : [
            m?.name ?? null,
            m?.discrete === true,
            m?.spatial === true,
            m?.step ?? null,
            m?.cyclic === true,
          ]
    )
  );
}

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
  // Dimensions metadata: reference equality first, then a compare of the
  // canonical QUERY-DETERMINANT projection (see dimsQuerySig — display/
  // navigation fields and object key order deliberately don't matter).
  if (a.dimensions !== b.dimensions) {
    if (!a.dimensions || !b.dimensions) return false;
    if (a.dimensions.length !== b.dimensions.length) return false;
    if (dimsQuerySig(a) !== dimsQuerySig(b)) return false;
  }
  return true;
}
