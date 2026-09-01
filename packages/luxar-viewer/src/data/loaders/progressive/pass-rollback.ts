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
 * @module data/loaders/progressive/pass-rollback
 */

import { log, Modules } from '../../../utils/log';

export interface LadderRollbackState {
  loadedLevelCount: number;
  levelsAtPassStart: number;
  concatCacheLodCount: number | null;
  retainedPayloadCount: number;
  payloadsAtPassStart: number;
  restoredFullLadderAtPassStart: boolean;
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
