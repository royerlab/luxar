/**
 * Unit tests for the shared SliceCache loader helpers
 * (data/loaders/progressive/slice-cache-helper.ts): prefix-ladder
 * restore/store semantics, UPGRADE-IF-LONGER (never downgrade), byte
 * accounting on upgrade, and the non-counting `peek` used by the
 * upgrade check.
 *
 * The loader-side integration (restore→resume, budget-gated prefix
 * stores) is pinned in the three progressive-loader test files
 * (three-geometry symmetry); this file pins the helper contracts
 * directly.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SliceCache } from '../../../../../cache/slice-cache';
import {
  restoreLadder,
  storeLadder,
  buildSliceViewSig,
  measureLodBytes,
} from '../../../../../data/loaders/progressive/slice-cache-helper';

interface FakeLod {
  positions: Float32Array;
  count: number;
}

function makeLod(n: number): FakeLod {
  return { positions: new Float32Array(n).fill(0.5), count: n };
}

const view = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 7],
  tolerance: [0, 0, 0, 0.25],
};

const PATH = '/node';
const N_LODS = 3;

describe('slice-cache-helper — prefix ladders', () => {
  let sc: SliceCache;

  beforeEach(() => {
    sc = new SliceCache({ maxSize: 1024 * 1024 });
  });

  it('stores and restores a PREFIX ladder (length < nLods)', () => {
    storeLadder(sc, PATH, view, [makeLod(10)]);
    const restored = restoreLadder<FakeLod>(sc, PATH, view, N_LODS);
    expect(restored).not.toBeNull();
    expect(restored!.length).toBe(1);
    expect(restored![0].count).toBe(10);
  });

  it('still restores a FULL ladder', () => {
    storeLadder(sc, PATH, view, [makeLod(10), makeLod(5), makeLod(2)]);
    const restored = restoreLadder<FakeLod>(sc, PATH, view, N_LODS);
    expect(restored!.length).toBe(3);
  });

  it('upgrade-if-longer: a longer snapshot replaces, with exact byte accounting', () => {
    storeLadder(sc, PATH, view, [makeLod(10)]);
    const key = SliceCache.makeKey(PATH, buildSliceViewSig(view));
    expect((sc.peek(key)!.payload as unknown[]).length).toBe(1);

    const longer = [makeLod(10), makeLod(5)];
    storeLadder(sc, PATH, view, longer);
    expect((sc.peek(key)!.payload as unknown[]).length).toBe(2);
    // LRUCache.set replaces the old entry's bytes exactly (subtract + add).
    expect(sc.getStats().size).toBe(measureLodBytes(longer));
    expect(sc.getStats().count).toBe(1);
  });

  it('never downgrades: a shorter snapshot leaves the longer entry intact', () => {
    storeLadder(sc, PATH, view, [makeLod(10), makeLod(5), makeLod(2)]);
    const key = SliceCache.makeKey(PATH, buildSliceViewSig(view));

    storeLadder(sc, PATH, view, [makeLod(10)]);
    expect((sc.peek(key)!.payload as unknown[]).length).toBe(3);
  });

  it('equal length is a no-op (no pointless re-clone)', () => {
    storeLadder(sc, PATH, view, [makeLod(10)]);
    const key = SliceCache.makeKey(PATH, buildSliceViewSig(view));
    const before = sc.peek(key)!.payload;
    storeLadder(sc, PATH, view, [makeLod(10)]);
    expect(sc.peek(key)!.payload).toBe(before); // same entry, not replaced
  });

  it('empty ladders are never stored and never restored', () => {
    storeLadder(sc, PATH, view, []);
    expect(sc.getStats().count).toBe(0);
    expect(restoreLadder<FakeLod>(sc, PATH, view, N_LODS)).toBeNull();
  });

  it('peek does not perturb hit/miss statistics or count as an access', () => {
    storeLadder(sc, PATH, view, [makeLod(10)]);
    const key = SliceCache.makeKey(PATH, buildSliceViewSig(view));
    const before = sc.getStats();

    sc.peek(key);
    sc.peek('missing-key');

    const after = sc.getStats();
    expect(after.hits).toBe(before.hits);
    expect(after.misses).toBe(before.misses);
    expect(after.hitRate).toBe(before.hitRate);
  });

  it('storeLadder itself never perturbs hit/miss stats (its length check must use peek, not get)', () => {
    // The upgrade-if-longer check reads the existing entry on EVERY store —
    // if it used the counting get() instead of peek(), each playback tick
    // would inflate the hit count and corrupt the monitor's hit-rate.
    storeLadder(sc, PATH, view, [makeLod(10)]);
    const before = sc.getStats();

    storeLadder(sc, PATH, view, [makeLod(10)]); // equal-length no-op (reads entry)
    storeLadder(sc, PATH, view, [makeLod(10), makeLod(5)]); // upgrade (reads entry)

    const after = sc.getStats();
    expect(after.hits).toBe(before.hits);
    expect(after.misses).toBe(before.misses);
  });

  it('restore returns the cache payload; storing deep-clones (cache never aliases loader arrays)', () => {
    const original = [makeLod(10)];
    storeLadder(sc, PATH, view, original);
    // Store deep-cloned: mutating the ORIGINAL cannot corrupt the entry.
    original[0].positions.fill(999);
    const restored = restoreLadder<FakeLod>(sc, PATH, view, N_LODS)!;
    expect(restored[0].positions[0]).toBeCloseTo(0.5);
  });
});
