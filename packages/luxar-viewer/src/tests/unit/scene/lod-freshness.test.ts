/**
 * Unit tests for the pure LOD freshness + settle helpers (no THREE / camera).
 */

import { describe, expect, it } from 'vitest';

import {
  coarsestFreshIndex,
  isFresh,
  isReady,
  SettleTracker,
  visibleElementCount,
  type FreshnessChild,
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

  // Every LOD-capable leaf type, which is what `isFreshnessTracked` resolves to
  // (`supportsLod`) — mesh included, since it became a legal `kind=lod` level.
  it('version-checks each stamped leaf type (gsplats / points / lines / mesh)', () => {
    for (const t of ['gsplats', 'points', 'lines', 'mesh']) {
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

  it('reads the per-type commit stamp for every leaf type', () => {
    expect(visibleElementCount(stamped('points', { visiblePointCount: 7 }))).toBe(7);
    expect(visibleElementCount(stamped('lines', { visibleSegmentCount: 3 }))).toBe(3);
    expect(visibleElementCount(stamped('gsplats', { visibleSplatCount: 0 }))).toBe(0);
    expect(visibleElementCount(stamped('mesh', { visibleTriangleCount: 5 }))).toBe(5);
  });

  it('counts mesh TRIANGLES, not its vertices', () => {
    // A mesh commits both counts, and they answer different questions. This one
    // is "did the level commit anything DRAWABLE", so a level with vertices but
    // no surviving triangle must read as empty — the empty-level display guard
    // has to redirect away from it. (The ladder's `coverage_fraction` thresholds
    // are derived from VERTEX counts instead, which is the separate question of
    // how much detail a level carries. Reading the wrong one here would let a
    // triangle-less level be displayed because it still had vertices.)
    expect(
      visibleElementCount(stamped('mesh', { visibleTriangleCount: 0, visibleVertexCount: 12 }))
    ).toBe(0);
  });

  it('returns null for a mesh level that has not committed yet', () => {
    // Deferred LOD levels attach a placeholder with no counts at all, and a
    // never-committed level must not be mistaken for a known-empty one.
    expect(visibleElementCount(child('mesh', 1))).toBeNull();
  });

  it('returns null for untracked children (groups / unstamped leaves)', () => {
    expect(visibleElementCount(child('group', 1))).toBeNull();
    expect(visibleElementCount({ ready: true, object: { userData: undefined } })).toBeNull();
    // A tracked leaf WITHOUT the count stamp (not yet committed) is untracked
    // too — the guard must only act on a KNOWN-empty level.
    expect(visibleElementCount(child('gsplats', 1))).toBeNull();
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
