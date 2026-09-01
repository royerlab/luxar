/**
 * Pass-start rollback for the progressive loaders (GSplats, Points, Lines,
 * and the Mesh reveal ladder).
 *
 * A progressive loader's `updateView` appends each streamed level to
 * `loadedLODs` as soon as the level's data arrives — BEFORE the caller has
 * concatenated, projected and committed it. That ordering is deliberate (the
 * concat at the end of the same call reads the array), but it means the
 * ladder cursor advances on work that has not yet been proven committable.
 *
 * When the commit then fails, the consequences compound rather than back off:
 *
 *  - the retry resumes from the ADVANCED cursor, so it loads the *next* level
 *    and attempts a strictly LARGER allocation than the one that just failed —
 *    escalation, not backoff, which is exactly wrong when the failure was
 *    `RangeError: Array buffer allocation failed`;
 *  - once the cursor reaches `nLods`, `hasMoreLODs` goes false. The refinement
 *    loop then drops the loader from `anyHasMoreLODs`, the
 *    `RefinementFailureTracker` never reaches its cap, no "giving up" toast
 *    fires, and the node is stranded PERMANENTLY at its last successfully
 *    committed rung while the data-loading monitor reports the cursor as
 *    "LOD n/n ~100%".
 *
 * Observed on the hosted `cosmicflows_laniakea` demo (#2426): basins reported
 * a complete ladder while rendering a coarse prefix, and the retries that were
 * supposed to recover the node were the thing driving it further out of
 * memory.
 *
 * The fix is to make a failed pass leave the loader exactly as the pass found
 * it, so a retry re-attempts the SAME prefix. Each loader records the level
 * count at pass start and exposes a `rollbackToPassStart()` that its
 * refinement wrapper calls from its catch. The DECISION lives here as a pure
 * function so the four loaders cannot drift — the same reason
 * `streaming-policy.ts` owns the streaming discipline — while the mutation
 * stays in the loader, which is the only thing that can touch its private
 * fields.
 *
 * COUNTS HERE ARE LOGICAL LEVELS, NEVER PAYLOAD ENTRIES. Every loader now FOLDS
 * its ladder — the rungs are concatenated into one cumulative payload and the
 * parts released — so `loadedLODs.length` becomes 1 while the logical count is
 * still 7. That is why each loader carries a separate `_loadedLODCount`, and
 * why {@link LadderRollbackState} takes the two counts apart rather than
 * inferring either from an array length. Feeding a payload count into a logical
 * field is silently wrong: no throw, just a memo gate evaluated against the
 * wrong number.
 *
 * The invariant that catches every variant of that mistake, and the one a
 * loader's tests should assert after any rollback:
 *
 *     _concatCache === null || _concatCache.lodCount <= <logical level count>
 *
 * A memo describing more levels than the loader holds is the whole hazard: the
 * loaders select their lineage parent on generation and level count, so an
 * over-long memo can stamp a prefix-lineage claim on an object it does not
 * extend, and the commit layer's append gate will then write a suffix over a
 * wrong prefix. Silent wrong render, no error raised.
 *
 * @module data/loaders/progressive/pass-rollback
 */

import { log, Modules } from '../../../utils/log';

export interface LadderRollbackState {
  /** Logical levels represented after the failed pass. */
  loadedLevelCount: number;
  /** Logical-level watermark captured before the failed pass appended work. */
  levelsAtPassStart: number;
  /** Logical level count covered by the concat memo, or `null` without one. */
  concatCacheLodCount: number | null;
  /** Separately retained payload objects after the failed pass. */
  retainedPayloadCount: number;
  /** Retained-payload watermark captured before the failed pass appended work. */
  payloadsAtPassStart: number;
  /** Whether the pass began from a full, not-yet-committed cache restore. */
  restoredFullLadderAtPassStart: boolean;
  /** Total logical levels in the ladder. */
  totalLevelCount: number;
}

/** What a loader should do to undo a failed pass. */
export type LadderRollbackPlan =
  | { action: 'none'; dropped: 0 }
  | { action: 'retry-folded-pass'; dropped: 0 }
  | { action: 'unwind-restored-full'; dropped: number }
  | {
      action: 'truncate';
      keep: number;
      dropped: number;
      /**
       * Whether the memoized concat must be discarded. True exactly when the
       * memo covers MORE logical levels than survive, in which case it
       * describes a prefix the loader no longer holds. A memo at or below
       * `keep` is still a faithful concatenation of retained levels, so it is
       * deliberately kept — the commit layer's append fast path gates on pure
       * IDENTITY between the memo and `committedData`, and dropping a valid
       * memo would downgrade the next commit to a full rewrite.
       */
      invalidateConcatCache: boolean;
    };

/**
 * Decide how far to unwind a ladder after a pass failed.
 *
 * Total, and defensive about its inputs: `levelsAtPassStart` is clamped into
 * `[0, loadedLevelCount]` so a stale or never-initialised value can only ever
 * under-drop (keep more levels than the pass started with) rather than discard
 * levels the pass did not add. Under-dropping degrades to today's behaviour;
 * over-dropping would throw away committed work.
 *
 * The returned action also accounts for folded payloads and restored full
 * snapshots, so every geometry makes the same retry/truncate/unwind choice.
 * `loadedLevelCount` / `levelsAtPassStart` track logical ladder progress;
 * `retainedPayloadCount` / `payloadsAtPassStart` separately reveal whether
 * the failed pass's appends still exist as individual payloads or were folded
 * into a cumulative payload before the failure surfaced.
 */
export function planLadderRollback(state: LadderRollbackState): LadderRollbackPlan {
  const loaded = Math.max(0, state.loadedLevelCount);
  const keep = Math.min(Math.max(0, state.levelsAtPassStart), loaded);
  const dropped = loaded - keep;
  if (dropped === 0) {
    return loaded === state.totalLevelCount
      ? { action: 'retry-folded-pass', dropped: 0 }
      : { action: 'none', dropped: 0 };
  }
  if (state.restoredFullLadderAtPassStart) {
    return { action: 'unwind-restored-full', dropped };
  }
  const retainedPayloadsAdded = state.retainedPayloadCount - state.payloadsAtPassStart;
  if (retainedPayloadsAdded < dropped) {
    return { action: 'retry-folded-pass', dropped: 0 };
  }
  return {
    action: 'truncate',
    keep,
    dropped,
    invalidateConcatCache: state.concatCacheLodCount !== null && state.concatCacheLodCount > keep,
  };
}

/** Run an optional loader rollback without replacing the original failure. */
export function tryRollbackToPassStart(loader: { rollbackToPassStart?: () => number }): number {
  try {
    return loader.rollbackToPassStart?.() ?? 0;
  } catch (error) {
    log.warning(Modules.SCENE_LOADER, 'Progressive loader rollback failed', error);
    return 0;
  }
}
