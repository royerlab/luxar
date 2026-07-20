/**
 * Prefix-lineage tracking for the append fast path (depth-sorting Phase 4
 * Stage 2), shared by all three geometries (Points, Lines, GSplats).
 *
 * A progressive-LOD commit that adds detail EXTENDS the previous commit: the
 * loader appends LOD levels in order, so the concatenated input of a k-level
 * commit is a byte-identical PREFIX of the (k+1)-level concatenation. Because
 * every geometry's nD→3D projection is per-element-independent and
 * order-preserving, under an unchanged view state the (k+1)-level projection's
 * first `prevCount` visible outputs are byte-identical to the previous
 * commit's entire output. The commit layer can then write & upload only the
 * appended suffix instead of the whole buffer.
 *
 * To take that fast path the commit layer must know that the new concat result
 * genuinely extends the object currently on the GPU. We record it as a
 * forward-chained PARENT reference — each memoized concat result points at the
 * immediately-preceding SAME-GENERATION result — and check it by pure identity
 * against {@link getCommittedData}. A match proves three things at once:
 * - same generation (a view change bumps the loader's reset generation, so the
 *   post-reset k=1 concat has no parent → append is rejected → full rewrite);
 * - a genuine extension (not an unrelated reload);
 * - the GPU currently holds exactly that parent's projection (any intervening
 *   unrelated commit would have replaced `committedData`).
 *
 * Stored in a module-owned {@link WeakMap} keyed on the concat-result object,
 * NOT as a field on the data:
 * - The data object is sometimes SHARED (the single-LOD concat returns the raw
 *   LOD object directly) or DEEP-CLONED (slice-cache ladder restore), so a
 *   mutable field would alias or be lost; identity-keyed WeakMap entries are
 *   immune. A restored-from-cache clone simply has no entry → no append on the
 *   first post-restore commit, which is safe.
 * - Keeps lineage out of the cached payload and lets entries be garbage
 *   collected with their result objects.
 *
 * Lives in `types/` (the bottom layer) so the loaders (`data/points/`,
 * `data/lines/`, `data/gsplats/`) and the commit pipeline
 * (`data/scene-loader/`) share one definition. Companion of
 * {@link module:types/committed-data}.
 *
 * @module types/prefix-lineage
 */

/** result object → the same-generation concat result it extends. */
const prefixParents = new WeakMap<object, object>();

/**
 * Record that `child` (a freshly memoized concat result) extends `parent`
 * (the immediately-preceding same-generation concat result). Passing a null
 * parent — the first level after a reset, which extends nothing — clears any
 * stale entry so the append gate rejects it.
 */
export function setPrefixParent(child: object, parent: object | null): void {
  if (parent) {
    prefixParents.set(child, parent);
  } else {
    prefixParents.delete(child);
  }
}

/**
 * The concat result `child` extends, or `undefined` when it extends nothing:
 * a view change (new generation), the first loaded level, or a
 * restored-from-cache clone. The append gate treats `undefined` as "not an
 * append" and forces a full rewrite.
 */
export function getPrefixParent(child: object): object | undefined {
  return prefixParents.get(child);
}
