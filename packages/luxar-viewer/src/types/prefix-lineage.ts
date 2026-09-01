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
 *   LOD object directly, but only when that object's picking index space —
 *   `elementIds` for Points, `ranges` for GSplats, `vertexRangeBounds` for
 *   Lines — is already the one the node's label CSR is keyed by, which the
 *   concat must otherwise strip off a shallow copy. That holds when the object
 *   publishes no index space at all, and — for a Points ladder whose parent
 *   carries the union CSR (#1439) — also when it publishes one, since a lone
 *   retained payload spans a prefix beginning at offset 0 of that union space)
 *   or DEEP-CLONED
 *   (slice-cache ladder restore), so a
 *   mutable field would alias or be lost; identity-keyed WeakMap entries are
 *   immune. A restored-from-cache clone simply has no entry → no append on the
 *   first post-restore commit, which is safe.
 * - Keeps lineage out of the cached payload and lets entries be garbage
 *   collected with their result objects.
 *
 * RETENTION CONTRACT — the chain is capped at depth 1 and consumed on commit.
 * A WeakMap holds its VALUE strongly while the key is reachable, so a naive
 * forward chain (C_n→C_{n-1}→…→C_1) pins every intermediate concat of a
 * generation for as long as the newest is alive — and `committedData` keeps
 * the newest alive indefinitely on a static view. For an n-level ladder that
 * retains ~(n−1)/2 × the final CPU arrays (hundreds of MB on 10M-element
 * datasets). Two measures bound it:
 * - {@link setPrefixParent} DELETES the parent's own entry when linking a new
 *   child (the grandparent link was only needed for the parent's own gate
 *   check, which has either already run or — for a still-in-flight commit —
 *   safely degrades to a full rewrite).
 * - The commit layer clears the committed data's entry right after the gate
 *   consumes it (`setPrefixParent(data, null)`), releasing the parent
 *   as soon as the append/full-write decision is made. A retry after a
 *   throwing write therefore full-rewrites, which is the safe direction.
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
 *
 * Linking a child also DELETES the parent's own entry, capping the chain at
 * depth 1 (see the module retention contract). The parent's entry was only
 * needed for the parent's own commit-gate check; if that commit is still in
 * flight when the next concat supersedes it, its gate reads `undefined` and
 * takes the (safe) full-rewrite path.
 */
export function setPrefixParent(child: object, parent: object | null): void {
  if (parent) {
    prefixParents.set(child, parent);
    if (parent !== child) prefixParents.delete(parent);
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

/**
 * Release `data`'s lineage entry when the pass that created it did NOT commit.
 *
 * The retention contract above has two halves, and only one of them was
 * implemented. The commit layer's `setPrefixParent(data, null)` is
 * unconditional and correct — but a refinement pass does not always reach a
 * commit. It skips one when processing yields nothing to stage, and when a
 * superseding view-state aborts mid-flight, which happens constantly during
 * ordinary camera motion. On those exits the entry survived, and because the
 * newest concat is reachable indefinitely through `committedData` and the
 * loader's memo, so did the parent it pins.
 *
 * That parent is the PREVIOUS CUMULATIVE, which is what makes it expensive:
 * its size is `(passes - 1) / passes` of the final payload, so it grows with
 * ladder depth and saturates at a whole redundant copy. Measured in-process on
 * an equal-count ladder climbed one rung per pass, the retained fraction runs
 * 0.0 / 0.5 / 0.75 / 0.875 at 2 / 4 / 8 / 16 rungs — reproducing the depth
 * curve seen on real gsplat nodes (#2426). The two-rung case retains nothing
 * for the same reason: it completes in a single pass, so there is no previous
 * cumulative to pin. Those fractions come from an abort-path probe that never
 * commits; committing browser runs retain the same depth curve before and after
 * this release, so they must not be read as the settled deep-rung saving.
 *
 * A no-op when the pass committed (the commit already cleared it) and when
 * `data` is absent. Safe to over-call: clearing an entry that is already gone
 * costs a WeakMap miss, and the worst consequence of clearing one that was
 * still wanted is a full rewrite instead of an append.
 *
 * Mesh does not stamp lineage at all, so it neither needs nor uses this.
 */
export function releaseLineageIfUncommitted(
  data: object | null | undefined,
  committed: boolean
): void {
  if (!committed && data) setPrefixParent(data, null);
}
