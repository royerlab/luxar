/**
 * Unit tests for the pure LOD freshness + settle helpers (no THREE / camera).
 */

import { describe, expect, it } from 'vitest';

import {
  coarsestFreshIndex,
  coarsestFreshNonEmptyIndex,
  isFresh,
  isReady,
  SettleTracker,
  shouldHoldPreviousDisplay,
  visibleElementCount,
  type FreshnessChild,
  type HoldCandidate,
} from '../../../scene/lod-freshness';

function child(
  nodeType: string | undefined,
  loadedViewVersion: number | undefined,
  ready: boolean | undefined = undefined
): FreshnessChild {
  return { ready, object: { userData: nodeType ? { nodeType, loadedViewVersion } : undefined } };
}

describe('isReady', () => {
  it('treats absent/true as ready and only false as not ready', () => {
    expect(isReady({})).toBe(true);
    expect(isReady({ ready: true })).toBe(true);
    expect(isReady({ ready: false })).toBe(false);
  });
});

describe('isFresh', () => {
  it('is false for a not-ready or missing child', () => {
    expect(isFresh(undefined, 1)).toBe(false);
    expect(isFresh(child('gsplats', 1, false), 1)).toBe(false);
  });

  it('version-checks each stamped leaf type (gsplats / points / lines)', () => {
    for (const t of ['gsplats', 'points', 'lines']) {
      expect(isFresh(child(t, 2), 2)).toBe(true); // stamp matches current version
      expect(isFresh(child(t, 1), 2)).toBe(false); // stale stamp
      expect(isFresh(child(t, undefined), 2)).toBe(false); // never committed
    }
  });

  it('treats a ready non-leaf child (nested group / unstamped) as always fresh', () => {
    expect(isFresh(child('group', undefined), 99)).toBe(true);
    expect(isFresh({ ready: true, object: { userData: undefined } }, 99)).toBe(true);
  });
});

describe('coarsestFreshIndex', () => {
  it('returns the coarsest (lowest-index) fresh child', () => {
    const children = [child('gsplats', 1), child('gsplats', 2), child('gsplats', 2)];
    // coarse stale@1, mid+fine fresh@2 → coarsest fresh is index 1.
    expect(coarsestFreshIndex(children, 2)).toBe(1);
  });

  it('returns -1 when no child is fresh (caller falls back to coarsest ready)', () => {
    const children = [child('gsplats', 1), child('gsplats', 1)];
    expect(coarsestFreshIndex(children, 2)).toBe(-1);
  });
});

describe('visibleElementCount', () => {
  function stamped(nodeType: string, counts: Record<string, number>): FreshnessChild {
    return { ready: true, object: { userData: { nodeType, loadedViewVersion: 1, ...counts } } };
  }

  it('reads the per-type commit stamp for the three leaf types', () => {
    expect(visibleElementCount(stamped('points', { visiblePointCount: 7 }))).toBe(7);
    expect(visibleElementCount(stamped('lines', { visibleSegmentCount: 3 }))).toBe(3);
    expect(visibleElementCount(stamped('gsplats', { visibleSplatCount: 0 }))).toBe(0);
  });

  it('returns null for untracked children (groups / unstamped leaves)', () => {
    expect(visibleElementCount(child('group', 1))).toBeNull();
    expect(visibleElementCount({ ready: true, object: { userData: undefined } })).toBeNull();
    // A tracked leaf WITHOUT the count stamp (not yet committed) is untracked
    // too — the guard must only act on a KNOWN-empty level.
    expect(visibleElementCount(child('gsplats', 1))).toBeNull();
  });
});

describe('coarsestFreshNonEmptyIndex', () => {
  function level(loadedViewVersion: number, visibleSplatCount?: number): FreshnessChild {
    return {
      ready: true,
      object: { userData: { nodeType: 'gsplats', loadedViewVersion, visibleSplatCount } },
    };
  }

  it('skips fresh-but-empty levels and returns the coarsest fresh non-empty one', () => {
    // coarse fresh with 100 splats, fine fresh with 0 → index 0.
    expect(coarsestFreshNonEmptyIndex([level(2, 100), level(2, 0)], 2)).toBe(0);
  });

  it('skips stale levels even when non-empty', () => {
    // coarse stale@1 (non-empty), fine fresh@2 (non-empty) → index 1.
    expect(coarsestFreshNonEmptyIndex([level(1, 100), level(2, 50)], 2)).toBe(1);
  });

  it('accepts untracked counts (null) — only KNOWN-empty is skipped', () => {
    const untracked: FreshnessChild = {
      ready: true,
      object: { userData: { nodeType: 'group' } },
    };
    expect(coarsestFreshNonEmptyIndex([untracked, level(2, 0)], 2)).toBe(0);
  });

  it('returns -1 when every fresh level is empty (genuinely empty slice)', () => {
    expect(coarsestFreshNonEmptyIndex([level(2, 0), level(2, 0)], 2)).toBe(-1);
  });
});

describe('shouldHoldPreviousDisplay (never-downgrade display gate)', () => {
  /** A gsplats level with a committed count + version stamp. */
  function shown(visibleSplatCount: number, loadedViewVersion = 2): FreshnessChild {
    return {
      ready: true,
      object: { userData: { nodeType: 'gsplats', loadedViewVersion, visibleSplatCount } },
    };
  }

  /** A streaming gsplats aspiration (ladder incomplete unless overridden). */
  function streamingAsp(
    visibleSplatCount: number,
    overrides: Partial<HoldCandidate> = {}
  ): HoldCandidate {
    return {
      ready: true,
      hasMoreLODs: () => true,
      object: { userData: { nodeType: 'gsplats', loadedViewVersion: 2, visibleSplatCount } },
      ...overrides,
    };
  }

  it('holds while the aspiration ladder is incomplete and its count is below prev', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100), 2)).toBe(true);
  });

  it('releases on ladder completion (hasMoreLODs false AND no pass in flight)', () => {
    const asp = streamingAsp(50, { hasMoreLODs: () => false, loading: false });
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(false);
  });

  it('still holds when hasMoreLODs is false but the final pass is mid-flight (premature-release regression)', () => {
    // hasMoreLODs flips false the moment the final LOD's FETCH resolves —
    // several frames before its processing + commit land. Releasing on it
    // alone re-shows the partial level for exactly those frames.
    const asp = streamingAsp(10, { hasMoreLODs: () => false, loading: true });
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(true);
  });

  it('releases at the committed-count crossover (aspiration caught up)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(100), shown(100), 2)).toBe(false);
    expect(shouldHoldPreviousDisplay(streamingAsp(150), shown(100), 2)).toBe(false);
  });

  it('releases when the aspiration ladder failed (degrade to ungated behavior)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10, { failed: true }), shown(100), 2)).toBe(
      false
    );
  });

  it('never holds a stale prev (staleness beats quality)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100, 1), 2)).toBe(false);
  });

  it('never holds an empty or count-untracked prev (fast first paint)', () => {
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(0), 2)).toBe(false);
    const untrackedPrev: FreshnessChild = {
      ready: true,
      object: { userData: { nodeType: 'gsplats', loadedViewVersion: 2 } },
    };
    expect(shouldHoldPreviousDisplay(streamingAsp(10), untrackedPrev, 2)).toBe(false);
    expect(shouldHoldPreviousDisplay(streamingAsp(10), undefined, 2)).toBe(false);
  });

  it('does not hold against an aspiration with an untracked count', () => {
    const asp = streamingAsp(0);
    delete asp.object.userData!.visibleSplatCount;
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(false);
  });

  it('does not hold for a single-LOD aspiration (no hasMoreLODs, not loading)', () => {
    const asp = streamingAsp(10, { hasMoreLODs: undefined });
    expect(shouldHoldPreviousDisplay(asp, shown(100), 2)).toBe(false);
  });

  it('does not compare counts across geometry types (mixed-type ladder)', () => {
    const linesPrev: FreshnessChild = {
      ready: true,
      object: { userData: { nodeType: 'lines', loadedViewVersion: 2, visibleSegmentCount: 100 } },
    };
    expect(shouldHoldPreviousDisplay(streamingAsp(10), linesPrev, 2)).toBe(false);
  });

  it('falls back to readiness for prev when freshness is untracked (version null)', () => {
    // No version wiring: a READY prev can be held...
    expect(shouldHoldPreviousDisplay(streamingAsp(10), shown(100), null)).toBe(true);
    // ...a not-ready prev cannot.
    const notReady = shown(100);
    notReady.ready = false;
    expect(shouldHoldPreviousDisplay(streamingAsp(10), notReady, null)).toBe(false);
  });
});

describe('SettleTracker', () => {
  it('reports settled only after the version has held steady for >= settleTicks', () => {
    const t = new SettleTracker();
    t.observe(5, 0); // version first seen at tick 0
    expect(t.isSettled(3, 8)).toBe(false); // only 3 ticks stable
    expect(t.isSettled(8, 8)).toBe(true); // 8 ticks stable → settled
  });

  it('resets the settle clock when the version changes', () => {
    const t = new SettleTracker();
    t.observe(5, 0);
    expect(t.isSettled(10, 8)).toBe(true);
    t.observe(6, 10); // scrubbed to a new version at tick 10
    expect(t.isSettled(12, 8)).toBe(false); // clock reset → not settled yet
    expect(t.isSettled(18, 8)).toBe(true); // stable again for 8 ticks
  });
});
