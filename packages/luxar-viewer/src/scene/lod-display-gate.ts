/**
 * The **never-downgrade display gate** for the `LODGroupRegistry` selector —
 * pure policy over commit-time stamps, no THREE import (structural node
 * shapes only), unit-testable in isolation like its sibling
 * `lod-freshness.ts` (which supplies the leaf-stamp primitives).
 *
 * A lazy substitutive level flips `ready` after its FIRST additive chunk
 * commits, so an ungated swap to a fresh-but-still-streaming level pops
 * displayed quality down to chunk-1 (on zoom in, zoom out, or after a scrub
 * settles) and climbs back over the following passes.
 * {@link shouldHoldPreviousDisplay} holds the previously-displayed level
 * while the streaming aspiration is strictly worse than what is shown, and
 * releases on ladder completion, committed-count crossover, ladder failure,
 * or the previous level losing freshness — never blocking a swap when
 * nothing better is on screen (fast first paint is preserved).
 *
 * **Committed state only.** The gate reads the per-mesh commit stamps
 * (`visible*Count`, `loadedViewVersion`, `committedLadderComplete` — written
 * by the commit helpers, see `data/scene-loader/commit/stamp-view-version.ts`),
 * never the loaders' live `hasMoreLODs` getters. Live getters flip the moment
 * the final LOD's fetch resolves, frames before its processing + commit land;
 * the stamps flip in the same synchronous call that writes the final count,
 * so the gate can never see "complete" paired with a stale partial count.
 * (The registry's refinement-kick logic deliberately keeps using the live
 * getter — kicking MORE loading wants live loader state; displaying wants
 * committed state.)
 *
 * **Nested-group levels.** A lod_group child can be a whole subtree (e.g. the
 * `overview` recipe's fine `kind=partition` branch, or hand-authored
 * lod-of-lod nestings) rather than a single stamped leaf. Such a child has no
 * direct stamps; {@link subtreeDisplayProgress} folds its *visible* stamped
 * leaves into one aggregate (count sum / all-complete / all-fresh), which the
 * gate consumes exactly like a leaf's stamps. The walk respects descendant
 * `visible` flags — an inner lod_group keeps toggling its own levels while
 * the outer placeholder is hidden, so the aggregate counts precisely what
 * would render — but ignores the ROOT's own flag (the registry itself hides
 * the placeholder of a not-displayed child).
 *
 * @module scene/lod-display-gate
 */

import {
  countFromUserData,
  isFresh,
  isReady,
  visibleElementCount,
  type FreshnessChild,
} from './lod-freshness';

/**
 * Structural node shape for {@link subtreeDisplayProgress}: the subset of
 * `THREE.Object3D` the walk reads. Every field optional, so a bare
 * `FreshnessChild['object']` (leaf stubs in tests) is assignable too.
 */
export interface ProgressNode {
  visible?: boolean;
  children?: readonly ProgressNode[];
  userData?: FreshnessChild['object']['userData'];
}

/**
 * Aggregate committed display state of a subtree's visible stamped leaves.
 * `nodeType` is the shared leaf geometry type, or `'mixed'` when the subtree
 * blends types (counts across types are not comparable — the gate refuses).
 */
export interface SubtreeDisplayProgress {
  /** Sum of the visible stamped leaves' committed element counts. */
  count: number;
  /** True iff NO visible leaf carries `committedLadderComplete: false`. */
  complete: boolean;
  /**
   * True iff every visible stamped leaf is stamped for `version` (vacuously
   * true when `version` is null — freshness untracked).
   */
  fresh: boolean;
  /** Shared leaf type (`points` / `lines` / `gsplats`) or `'mixed'`. */
  nodeType: string;
}

/** Mutable fold state for the recursive walk (module-internal). */
interface ProgressAccumulator {
  count: number;
  complete: boolean;
  fresh: boolean;
  nodeType: string;
  mixed: boolean;
  any: boolean;
}

function foldProgress(
  node: ProgressNode,
  isRoot: boolean,
  version: number | null,
  acc: ProgressAccumulator
): void {
  // Respect descendant visibility (inner lod_groups toggle their own levels);
  // ignore the root's own flag (the registry hides the placeholder itself).
  if (!isRoot && node.visible === false) return;

  const ud = node.userData;
  // A stamped leaf contributes; unstamped nodes (groups, never-committed
  // placeholders) contribute nothing and never block completeness/freshness.
  // Reads the count straight off ``ud`` (no throwaway wrapper) — this recurses
  // over every descendant on the per-frame hot path during a hold.
  const count = countFromUserData(ud);
  if (count != null && ud) {
    acc.any = true;
    acc.count += count;
    if (ud.committedLadderComplete === false) acc.complete = false;
    if (version != null && ud.loadedViewVersion !== version) acc.fresh = false;
    const t = ud.nodeType ?? '';
    if (!acc.nodeType) acc.nodeType = t;
    else if (acc.nodeType !== t) acc.mixed = true;
  }

  const children = node.children;
  if (children) {
    for (const child of children) foldProgress(child, false, version, acc);
  }
}

/**
 * Fold the visible stamped leaves under `root` into one
 * {@link SubtreeDisplayProgress}, or `null` when the subtree contains no
 * stamped leaf at all (nothing committed yet / not a geometry subtree) — the
 * gate treats `null` as "unknown" and refuses to hold, preserving today's
 * behavior for shapes it cannot reason about.
 *
 * Arbitrary nesting depth is supported by construction (partition-of-leaves,
 * lod-of-partition, lod-of-lod, ...): every shape bottoms out in stamped
 * leaves, and inner lod_groups' own visibility toggling makes the fold count
 * exactly the levels that would render.
 */
export function subtreeDisplayProgress(
  root: ProgressNode,
  version: number | null
): SubtreeDisplayProgress | null {
  const acc: ProgressAccumulator = {
    count: 0,
    complete: true,
    fresh: true,
    nodeType: '',
    mixed: false,
    any: false,
  };
  foldProgress(root, true, version, acc);
  if (!acc.any) return null;
  return {
    count: acc.count,
    complete: acc.complete,
    fresh: acc.fresh,
    nodeType: acc.mixed ? 'mixed' : acc.nodeType,
  };
}

/**
 * The aspiration-side shape for {@link shouldHoldPreviousDisplay}: a
 * `FreshnessChild` plus the one lazy-lifecycle field the hold decision reads
 * (structurally compatible with the registry's `LODGroupChild`).
 */
export interface HoldCandidate extends FreshnessChild {
  /** Set by the load thunk when the last ``ensureLoaded`` pass failed. */
  failed?: boolean;
}

/**
 * One side of the hold comparison, normalized: a leaf child reads its own
 * commit stamps; a group child (no direct count, has children) reads the
 * subtree aggregate. `null` = unknown → the gate refuses to hold.
 */
interface SideProgress {
  count: number;
  complete: boolean;
  nodeType: string;
  /** Fresh for `version` (ready-based when untracked) — staleness beats quality. */
  freshForHold: boolean;
}

function sideProgress(child: FreshnessChild, version: number | null): SideProgress | null {
  const direct = visibleElementCount(child);
  if (direct != null) {
    const ud = child.object.userData!;
    return {
      count: direct,
      complete: ud.committedLadderComplete !== false,
      nodeType: ud.nodeType ?? '',
      freshForHold: version == null ? isReady(child) : isFresh(child, version),
    };
  }
  // Group / untracked child: aggregate over the subtree, if there is one.
  // FreshnessChild's declared object shape omits children/visible; the real
  // registry children hold THREE.Object3D, which carries both (ProgressNode
  // is all-optional, so leaf test stubs remain valid inputs too).
  const object = child.object as ProgressNode;
  if (!object.children || object.children.length === 0) return null;
  const aggregate = subtreeDisplayProgress(object, version);
  if (!aggregate) return null;
  return {
    count: aggregate.count,
    complete: aggregate.complete,
    nodeType: aggregate.nodeType,
    freshForHold: isReady(child) && aggregate.fresh,
  };
}

/**
 * The never-downgrade display gate: should the registry keep the
 * previously-displayed level (`prev`) on screen instead of swapping to the
 * `aspiration`, because the aspiration's additive ladder(s) are still
 * streaming and its committed geometry is strictly worse than what is shown?
 *
 * Holds only while ALL of these are true:
 *   - the aspiration is **streaming**: its committed ladder stamp — direct
 *     for a leaf, aggregated over visible leaves for a group subtree — says
 *     incomplete. A missing stamp reads as complete (single-LOD levels;
 *     never-committed levels aren't ``ready``, so the gate isn't reached).
 *   - the aspiration is not **failed** — a failing ladder degrades to the
 *     ungated behavior (show the partial aspiration; the failure cooldown
 *     retries) instead of pinning `prev` behind a possibly-permanent failure.
 *   - `prev` is fresh for `version` (aggregate freshness for a group prev;
 *     merely ready when `version` is ``null``): staleness always beats
 *     quality — a stale `prev` shows the wrong slice and must not be held.
 *   - both committed element counts are known, comparable (same leaf
 *     geometry type on both sides — splat counts vs segment counts are
 *     meaningless to compare, and `'mixed'` subtrees are never comparable),
 *     `prev`'s is non-zero, and the aspiration's is strictly below it.
 *
 * Releases (returns ``false``) on ladder completion (commit landed), count
 * crossover (the aspiration caught up — the rest of its ladder then streams
 * *visibly*), failure, a stale/empty/unknown `prev`, or no `prev` at all —
 * so a group with nothing better on screen always swaps immediately (fast
 * first paint).
 */
export function shouldHoldPreviousDisplay(
  aspiration: HoldCandidate,
  prev: FreshnessChild | undefined,
  version: number | null
): boolean {
  if (!prev) return false;
  const asp = sideProgress(aspiration, version);
  if (!asp || asp.complete) return false;
  if (aspiration.failed) return false;
  const prevProgress = sideProgress(prev, version);
  if (!prevProgress || !prevProgress.freshForHold) return false;
  if (prevProgress.count <= 0) return false;
  if (asp.nodeType !== prevProgress.nodeType || asp.nodeType === 'mixed') return false;
  return asp.count < prevProgress.count;
}
