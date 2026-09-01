/** Unit tests for the shared progressive pass rollback decision. */

import { afterEach, describe, it, expect, vi } from 'vitest';
import {
  planLadderRollback,
  tryRollbackToPassStart,
  type LadderRollbackState,
} from '../../../../../data/loaders/progressive/pass-rollback';
import { log, Modules } from '../../../../../utils/log';

afterEach(() => {
  vi.restoreAllMocks();
});

const baseState: LadderRollbackState = {
  loadedLevelCount: 6,
  levelsAtPassStart: 2,
  concatCacheLodCount: 6,
  retainedPayloadCount: 6,
  payloadsAtPassStart: 2,
  restoredFullLadderAtPassStart: false,
  totalLevelCount: 8,
};

describe('planLadderRollback', () => {
  it('truncates intact payloads appended by the failed pass', () => {
    expect(planLadderRollback(baseState)).toEqual({
      action: 'truncate',
      keep: 2,
      dropped: 4,
      invalidateConcatCache: true,
    });
  });

  it('keeps a memo that covers only surviving levels', () => {
    expect(planLadderRollback({ ...baseState, concatCacheLodCount: 2 })).toEqual({
      action: 'truncate',
      keep: 2,
      dropped: 4,
      invalidateConcatCache: false,
    });
  });

  it('retries a folded pass when no intact payload can be removed', () => {
    expect(
      planLadderRollback({
        ...baseState,
        loadedLevelCount: 6,
        levelsAtPassStart: 4,
        retainedPayloadCount: 1,
        payloadsAtPassStart: 1,
      })
    ).toEqual({ action: 'retry-folded-pass', dropped: 0 });
  });

  it('keeps a full no-append pass schedulable for retry', () => {
    expect(
      planLadderRollback({
        ...baseState,
        loadedLevelCount: 8,
        levelsAtPassStart: 8,
        retainedPayloadCount: 1,
        payloadsAtPassStart: 1,
      })
    ).toEqual({ action: 'retry-folded-pass', dropped: 0 });
  });

  it('fully unwinds an uncommitted restored full snapshot', () => {
    expect(
      planLadderRollback({
        ...baseState,
        restoredFullLadderAtPassStart: true,
      })
    ).toEqual({ action: 'unwind-restored-full', dropped: 4 });
  });

  it('does nothing when an incomplete pass appended nothing', () => {
    expect(
      planLadderRollback({
        ...baseState,
        loadedLevelCount: 3,
        levelsAtPassStart: 3,
        retainedPayloadCount: 1,
        payloadsAtPassStart: 1,
      })
    ).toEqual({ action: 'none', dropped: 0 });
  });

  it('clamps bad watermarks without dropping earlier work', () => {
    expect(
      planLadderRollback({
        ...baseState,
        loadedLevelCount: 3,
        levelsAtPassStart: 9,
        retainedPayloadCount: 3,
        payloadsAtPassStart: 3,
      })
    ).toEqual({ action: 'none', dropped: 0 });
  });

  it('clamps a negative watermark to zero', () => {
    expect(
      planLadderRollback({
        ...baseState,
        loadedLevelCount: 4,
        levelsAtPassStart: -2,
        retainedPayloadCount: 4,
        payloadsAtPassStart: 0,
      })
    ).toEqual({
      action: 'truncate',
      keep: 0,
      dropped: 4,
      invalidateConcatCache: true,
    });
  });

  it('handles an empty ladder', () => {
    expect(
      planLadderRollback({
        ...baseState,
        loadedLevelCount: 0,
        levelsAtPassStart: 0,
        retainedPayloadCount: 0,
        payloadsAtPassStart: 0,
        totalLevelCount: 8,
      })
    ).toEqual({ action: 'none', dropped: 0 });
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
