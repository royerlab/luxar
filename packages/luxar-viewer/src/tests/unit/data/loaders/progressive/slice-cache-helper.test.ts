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

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SliceCache } from '../../../../../cache/slice-cache';
import { log } from '../../../../../utils/log';
import {
  restoreLadder,
  restoreLadderSnapshot,
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

  it('stamps stored prefix depth against the node total for cache stats', () => {
    storeLadder(sc, PATH, view, [makeLod(10), makeLod(5)], { totalLODCount: 4 });
    expect(sc.getStats().fullLadderCount).toBe(0);
    expect(sc.getStats().ladderDepthHistogram).toEqual({ '2/4': 1 });
  });

  it('restores a logical depth that is larger than the folded payload count', () => {
    storeLadder(sc, PATH, view, [makeLod(17)], { ladderDepth: 3, totalLODCount: 3 });

    const restored = restoreLadderSnapshot<FakeLod>(sc, PATH, view, N_LODS);
    expect(restored).toEqual({ lods: [expect.objectContaining({ count: 17 })], depth: 3 });
    expect(sc.getStats().ladderDepthHistogram).toEqual({ '3/3': 1 });
  });

  it('uses logical depth for upgrade-if-longer when payload count stays folded', () => {
    storeLadder(sc, PATH, view, [makeLod(10)], { ladderDepth: 2, totalLODCount: 3 });
    const key = SliceCache.makeKey(PATH, buildSliceViewSig(view));
    const before = sc.peek(key)!.payload;

    storeLadder(sc, PATH, view, [makeLod(15)], { ladderDepth: 3, totalLODCount: 3 });

    expect(sc.peek(key)!.payload).not.toBe(before);
    expect(sc.peek(key)!.ladderDepth).toBe(3);
  });

  it('does not classify a plain one-result store as an additive ladder', () => {
    storeLadder(sc, PATH, view, [makeLod(10)]);

    expect(sc.getStats().fullLadderCount).toBe(0);
    expect(sc.getStats().ladderDepthHistogram).toEqual({});
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

  it('deep-copies and bills an elementIds map through a store/restore round trip', () => {
    // `cloneLodSnapshot` / `measureLodBytes` walk own typed-array props
    // generically, so the points slot → on-disk map (issue #1421) survives a
    // scrub-back restore for free. Pin it: the restored map must be a DISTINCT
    // buffer (the source is a view into the reused accumulator) with equal
    // contents, and its bytes must be billed.
    const lod = makeLod(4) as FakeLod & { elementIds: Uint32Array };
    lod.elementIds = new Uint32Array([2048, 2049, 6144, 6145]);
    storeLadder(sc, PATH, view, [lod]);

    const restored = restoreLadder<typeof lod>(sc, PATH, view, N_LODS);
    const map = restored![0].elementIds;
    expect(map).toBeInstanceOf(Uint32Array);
    expect(map).not.toBe(lod.elementIds); // deep copy, not an alias
    expect(Array.from(map)).toEqual([2048, 2049, 6144, 6145]);

    // Billed: 4 float32 positions + 4 uint32 ids = 32 bytes.
    expect(measureLodBytes([lod])).toBe(4 * 4 + 4 * 4);
    expect(sc.getStats().size).toBe(measureLodBytes([lod]));
  });

  // The lines `vertexRangeBounds` counterpart (issue #1424) deliberately does
  // NOT live here: these helpers walk typed arrays by SHAPE, never by name, so
  // a second inline fake payload would be provably equivalent to the
  // `elementIds` one above and could not fence the flat-`Uint32Array` design
  // decision it exists for. It is pinned against a real loader payload in
  // `tests/unit/data/lines/spatial-index-loader.test.ts` instead.

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

  it('forwards the scan hint to SliceCache.set (scan-resistant eviction opt-in)', () => {
    const setSpy = vi.spyOn(sc, 'set');
    storeLadder(sc, PATH, view, [makeLod(10)], { scan: true });
    expect(setSpy.mock.calls[0][2]).toEqual({ scan: true });

    // Longer ladder (passes upgrade-if-longer) without the opt: no scan flag.
    storeLadder(sc, PATH, view, [makeLod(10), makeLod(5)]);
    expect(setSpy.mock.calls[1][2]).toEqual({});
    setSpy.mockRestore();
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

  // Regression (deep-double-check): with EVERY dimension displayed there is
  // exactly one possible slice — it is never re-queried, so caching it can
  // never hit and would only pin a deep clone (up to the whole S-cache
  // budget, e.g. a plain 3D scene's full ladder) for nothing.
  it('skips store AND restore for views with no hidden dimensions', () => {
    const allDisplayed = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [0, 0, 0],
    };
    storeLadder(sc, PATH, allDisplayed, [makeLod(10)]);
    expect(sc.getStats().count).toBe(0); // nothing stored
    expect(restoreLadder<FakeLod>(sc, PATH, allDisplayed, N_LODS)).toBeNull();
  });

  // Regression (deep-double-check): a stored ladder LONGER than the current
  // nLods (stale after a ladder reconfiguration) must be rejected, not
  // restored — the loader would index past its own level table.
  it('restoreLadder rejects an entry longer than nLods (stale-longer-ladder guard)', () => {
    storeLadder(sc, PATH, view, [makeLod(8), makeLod(6), makeLod(4), makeLod(2)]); // 4 levels
    expect(restoreLadder<FakeLod>(sc, PATH, view, 3)).toBeNull();
    // Same entry serves fine when nLods covers it.
    expect(restoreLadder<FakeLod>(sc, PATH, view, 4)).not.toBeNull();
  });
});

// The SliceCache key is a CANONICAL projection: it MUST discriminate every
// field that changes the DECODED SET (slicePosition, displayDims, a
// non-displayed continuous tolerance, a non-displayed discrete step) and MUST
// ignore fields that don't (the query-irrelevant discrete ride-along
// tolerance, displayed-dim metadata, JSON property order) so the same slice
// keys identically no matter which viewState builder produced it. Getting the
// "ignore" half wrong is the timelapse-nav regression this fixes: the nav and
// init builders emit different discrete tolerance / step / key order, so a raw
// serialization stored each timepoint under two keys and never hit on revisit.
describe('buildSliceViewSig — canonical key', () => {
  // 4D: three displayed spatial dims + one non-displayed discrete (time) dim.
  const base = {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 7],
    tolerance: [0, 0, 0, 0.5],
    dimensions: [
      { name: 'Z', discrete: false, cyclic: false, step: 1, spatial: true },
      { name: 'Y', discrete: false, cyclic: false, step: 1, spatial: true },
      { name: 'X', discrete: false, cyclic: false, step: 1, spatial: true },
      { name: 'Time', discrete: true, cyclic: false, step: 1, spatial: false },
    ],
  };

  it('THE FIX: the init-path and nav-path viewStates of the same slice key identically', () => {
    // Nav builder (dims-to-view-state.ts): discrete tolerance 0.5, step 1.
    const nav = base;
    // Init builder (view-state-manager.ts): discrete tolerance 0, displayed
    // dims carry an absent step, and the metadata objects are built in a
    // different property order. None of that changes which elements load.
    const init = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 7],
      tolerance: [0, 0, 0, 0],
      dimensions: [
        { name: 'Z', spatial: true, step: undefined, discrete: false, cyclic: false },
        { name: 'Y', spatial: true, step: undefined, discrete: false, cyclic: false },
        { name: 'X', spatial: true, step: undefined, discrete: false, cyclic: false },
        { name: 'Time', spatial: false, step: 1, discrete: true, cyclic: false },
      ],
    };
    expect(buildSliceViewSig(init)).toBe(buildSliceViewSig(nav));
  });

  it('differs when slicePosition differs', () => {
    expect(buildSliceViewSig(base)).not.toBe(
      buildSliceViewSig({ ...base, slicePosition: [0, 0, 0, 8] })
    );
  });

  it('differs when displayDims differ', () => {
    expect(buildSliceViewSig(base)).not.toBe(
      buildSliceViewSig({ ...base, displayDims: [1, 2, 3] })
    );
  });

  it('differs when a non-displayed discrete step differs (fetch reach = 0.25×step)', () => {
    const steppier = {
      ...base,
      dimensions: base.dimensions.map((d) => (d.discrete ? { ...d, step: 2 } : d)),
    };
    expect(buildSliceViewSig(base)).not.toBe(buildSliceViewSig(steppier));
  });

  it('IGNORES the query-irrelevant discrete ride-along tolerance (0 vs 0.5)', () => {
    // The real query recomputes the discrete reach from step; the tolerance
    // array only rides along, and the two builders disagree on its value.
    expect(buildSliceViewSig(base)).toBe(buildSliceViewSig({ ...base, tolerance: [0, 0, 0, 0] }));
  });

  it('differs when a non-displayed CONTINUOUS dim tolerance differs (extent selects the set)', () => {
    const cont = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 50],
      tolerance: [0, 0, 0, 100],
      dimensions: [
        { spatial: true },
        { spatial: true },
        { spatial: true },
        { name: 'w', discrete: false, spatial: false },
      ],
    };
    expect(buildSliceViewSig(cont)).not.toBe(
      buildSliceViewSig({ ...cont, tolerance: [0, 0, 0, 200] })
    );
  });

  it('is stable for equal views, with undefined dimensions serialized deterministically', () => {
    const noDims = { ...base, dimensions: undefined };
    expect(buildSliceViewSig(noDims)).toBe(buildSliceViewSig({ ...noDims }));
    expect(buildSliceViewSig(base)).toBe(buildSliceViewSig({ ...base }));
  });
});

describe('slice-cache-helper — oversized ladder (partial-prefix caching)', () => {
  // makeLod(n) retains 4n bytes (a Float32Array(n)). A full ladder that
  // exceeds the budget must NOT be dropped wholesale (that slice would
  // re-decode forever); the largest coarse-first prefix that fits is cached.
  it('caches the coarse prefix that fits instead of dropping the whole ladder', () => {
    const sc = new SliceCache({ maxSize: 500 }); // holds 1×400B level, not 2
    const ovPath = '/oversized-node';
    const full = [makeLod(100), makeLod(100), makeLod(100)]; // 3 × 400B = 1200B
    storeLadder(sc, ovPath, view, full);
    const restored = restoreLadder<FakeLod>(sc, ovPath, view, N_LODS);
    expect(restored).not.toBeNull();
    expect(restored!.length).toBe(1); // only the coarsest level fit
    expect(restored![0].count).toBe(100);
  });

  it('caches an unfolded coarse prefix when ladderDepth equals the payload count', () => {
    const sc = new SliceCache({ maxSize: 500 });
    const ovPath = '/oversized-unfolded-node';
    const full = [makeLod(100), makeLod(100), makeLod(100)];
    storeLadder(sc, ovPath, view, full, { ladderDepth: full.length, totalLODCount: N_LODS });

    const restored = restoreLadderSnapshot<FakeLod>(sc, ovPath, view, N_LODS);
    expect(restored?.lods).toHaveLength(1);
    expect(restored?.lods[0].count).toBe(100);
    expect(restored?.depth).toBe(1);
  });

  it('drops nothing to the cache when even the coarsest level exceeds the budget', () => {
    const sc = new SliceCache({ maxSize: 100 }); // < a single 400B level
    const ovPath = '/tiny-budget-node';
    storeLadder(sc, ovPath, view, [makeLod(100)]);
    expect(restoreLadder<FakeLod>(sc, ovPath, view, N_LODS)).toBeNull();
  });

  it('does not trim a folded payload when the trimmed logical depth is unknowable', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
    const sc = new SliceCache({ maxSize: 500 });
    const ovPath = '/folded-oversized-node';
    try {
      storeLadder(sc, ovPath, view, [makeLod(50)], { ladderDepth: 1, totalLODCount: 3 });

      storeLadder(sc, ovPath, view, [makeLod(100), makeLod(100)], {
        ladderDepth: 3,
        totalLODCount: 3,
      });

      const restored = restoreLadderSnapshot<FakeLod>(sc, ovPath, view, N_LODS);
      expect(restored?.depth).toBe(1);
      expect(restored?.lods[0].count).toBe(50);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('warns once when it has to trim an oversized ladder', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
    try {
      const sc = new SliceCache({ maxSize: 500 });
      const ovPath = '/warn-node';
      storeLadder(sc, ovPath, view, [makeLod(100), makeLod(100), makeLod(100)]);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('dedups the oversized warning per NODE PATH, not per view (a playback sweep warns once)', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
    try {
      const sc = new SliceCache({ maxSize: 500 });
      const p = '/sweep-node';
      const oversized = [makeLod(100), makeLod(100), makeLod(100)];
      // Two DISTINCT views (slice positions) of the same node, both oversized.
      storeLadder(sc, p, { ...view, slicePosition: [0, 0, 0, 7] }, oversized);
      storeLadder(sc, p, { ...view, slicePosition: [0, 0, 0, 8] }, oversized);
      storeLadder(sc, p, { ...view, slicePosition: [0, 0, 0, 9] }, oversized);
      expect(warnSpy).toHaveBeenCalledTimes(1); // per-path → one warning for the node
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('dedups the warning per key but re-warns after clear() (instance-scoped, not a module-global leak)', () => {
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
    try {
      const sc = new SliceCache({ maxSize: 500 });
      const p = '/reclear-node';
      const oversized = () => storeLadder(sc, p, view, [makeLod(100), makeLod(100), makeLod(100)]);
      oversized();
      expect(warnSpy).toHaveBeenCalledTimes(1); // first trim warns
      oversized();
      expect(warnSpy).toHaveBeenCalledTimes(1); // same key → deduped, no spam
      // A dataset switch clears the cache; the warn-dedup must reset with it so
      // the next dataset can warn afresh (pre-fix: a module-global Set that
      // clear() never touched kept the count pinned at 1 forever).
      sc.clear();
      oversized();
      expect(warnSpy).toHaveBeenCalledTimes(2);
    } finally {
      warnSpy.mockRestore();
    }
  });
});
