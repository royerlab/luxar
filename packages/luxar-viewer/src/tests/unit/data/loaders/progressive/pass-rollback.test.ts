/**
 * Unit tests for the shared pass-start rollback plan
 * (`data/loaders/progressive/pass-rollback`).
 *
 * The plan is what stops a failed commit from escalating: without it a retry
 * resumes from the ADVANCED ladder cursor and allocates more than the attempt
 * that just failed, and on reaching the last rung flips `hasMoreLODs` false so
 * the node is stranded silently. See #2426.
 */

import { afterEach, describe, it, expect, vi } from 'vitest';
import {
  planLadderRollback,
  tryRollbackToPassStart,
} from '../../../../../data/loaders/progressive/pass-rollback';
import { log, Modules } from '../../../../../utils/log';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('planLadderRollback', () => {
  it('drops exactly the levels the failed pass appended', () => {
    // Pass started at 2, streamed to 6, then the commit threw.
    expect(planLadderRollback(6, 2, 6)).toEqual({
      keep: 2,
      dropped: 4,
      invalidateConcatCache: true,
    });
  });

  it('is a no-op when the pass appended nothing', () => {
    // The throw came from the fetch, before any level landed.
    expect(planLadderRollback(3, 3, 3)).toEqual({
      keep: 3,
      dropped: 0,
      invalidateConcatCache: false,
    });
  });

  it('keeps a memo that covers only surviving levels', () => {
    // The memo is from the last SUCCESSFUL commit at level 2. It still
    // describes levels the loader holds, and the commit layer's append fast
    // path gates on its identity against `committedData` — dropping it would
    // needlessly downgrade the next commit to a full rewrite.
    expect(planLadderRollback(5, 2, 2).invalidateConcatCache).toBe(false);
  });

  it('discards a memo that covers levels being dropped', () => {
    expect(planLadderRollback(5, 2, 3).invalidateConcatCache).toBe(true);
  });

  it('treats an absent memo as nothing to invalidate', () => {
    expect(planLadderRollback(5, 2, null).invalidateConcatCache).toBe(false);
  });

  it('unwinds a full ladder back to its restored prefix', () => {
    // The Laniakea shape: a 7-rung ladder that reached 7/7 across escalating
    // retries. Rolling back to the pass-start prefix is what keeps
    // `hasMoreLODs` true so the failure cap can actually be reached.
    expect(planLadderRollback(7, 4, 7)).toEqual({
      keep: 4,
      dropped: 3,
      invalidateConcatCache: true,
    });
  });

  describe('is defensive rather than destructive about bad watermarks', () => {
    it('never drops levels the pass did not add (watermark above the count)', () => {
      // A stale watermark must under-drop, not discard committed work.
      expect(planLadderRollback(3, 9, 3)).toEqual({
        keep: 3,
        dropped: 0,
        invalidateConcatCache: false,
      });
    });

    it('clamps a negative watermark to zero', () => {
      expect(planLadderRollback(4, -2, 4)).toEqual({
        keep: 0,
        dropped: 4,
        invalidateConcatCache: true,
      });
    });

    it('handles an empty ladder', () => {
      expect(planLadderRollback(0, 0, null)).toEqual({
        keep: 0,
        dropped: 0,
        invalidateConcatCache: false,
      });
    });
  });
});

describe('tryRollbackToPassStart', () => {
  it('logs a rollback failure without replacing the original failure path', () => {
    const rollbackError = new Error('rollback failed');
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});

    expect(
      tryRollbackToPassStart({
        rollbackToPassStart: () => {
          throw rollbackError;
        },
      })
    ).toBe(0);
    expect(warning).toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      'Progressive loader rollback failed',
      rollbackError
    );
  });
});
