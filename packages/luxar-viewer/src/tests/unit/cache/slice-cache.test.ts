import { describe, it, expect } from 'vitest';
import { SliceCache } from '../../../cache/slice-cache';
import { perfCounters } from '../../../profiling/perf-counters';

/** A payload of `bytes` bytes (a single typed array) for size-accounting tests. */
function entry(bytes: number, tag = 'x') {
  return { payload: { tag, buf: new Uint8Array(bytes) }, bytes };
}

describe('SliceCache', () => {
  it('stores and retrieves an opaque payload by key', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const key = SliceCache.makeKey('/neuromast', 'view-8');
    const e = entry(64, 't8');
    c.set(key, e);
    expect(c.has(key)).toBe(true);
    const got = c.get(key);
    expect(got).toBe(e);
    expect((got!.payload as { tag: string }).tag).toBe('t8');
  });

  it('returns undefined and counts a miss for an unknown key', () => {
    const c = new SliceCache({ maxSize: 1024 });
    expect(c.get(SliceCache.makeKey('/n', 'v'))).toBeUndefined();
    expect(c.getStats().misses).toBe(1);
    expect(c.getStats().hits).toBe(0);
  });

  it('tracks hit/miss/hitRate in stats', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'v');
    c.set(k, entry(10));
    c.get(k); // hit
    c.get(k); // hit
    c.get(SliceCache.makeKey('/n', 'other')); // miss
    const s = c.getStats();
    expect(s.hits).toBe(2);
    expect(s.misses).toBe(1);
    expect(s.count).toBe(1);
    expect(s.hitRate).toBeCloseTo(2 / 3, 5);
  });

  it('evicts least-recently-used entries to stay within the byte budget', () => {
    // Budget fits ~2 entries of 400 bytes; a 3rd forces an eviction.
    const c = new SliceCache({ maxSize: 1000 });
    const a = SliceCache.makeKey('/n', 'a');
    const b = SliceCache.makeKey('/n', 'b');
    const d = SliceCache.makeKey('/n', 'd');
    c.set(a, entry(400, 'a'));
    c.set(b, entry(400, 'b'));
    c.get(a); // touch 'a' so 'b' becomes LRU
    c.set(d, entry(400, 'd')); // 1200 > 1000 → evict LRU ('b')
    expect(c.has(a)).toBe(true);
    expect(c.has(d)).toBe(true);
    expect(c.has(b)).toBe(false);
    expect(c.getStats().evictions).toBe(1);
    expect(c.getStats().size).toBeLessThanOrEqual(1000);
  });

  it('rejects an entry larger than the whole budget without corrupting state', () => {
    const c = new SliceCache({ maxSize: 100 });
    c.set(SliceCache.makeKey('/n', 'big'), entry(200));
    expect(c.has(SliceCache.makeKey('/n', 'big'))).toBe(false);
    expect(c.getStats().count).toBe(0);
    expect(c.getStats().size).toBe(0);
  });

  it('clear() empties the cache and resets counters', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'v');
    c.set(k, entry(10));
    c.get(k);
    c.clear();
    expect(c.has(k)).toBe(false);
    const s = c.getStats();
    expect(s.count).toBe(0);
    expect(s.size).toBe(0);
    expect(s.hits).toBe(0);
    expect(s.misses).toBe(0);
  });

  it('delete() removes one entry without turning a later miss into cache thrash', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const key = SliceCache.makeKey('/n', 'v');
    c.set(key, { ...entry(10), ladderDepth: 2, totalLadderDepth: 2 });

    expect(c.delete(key)).toBe(true);
    expect(c.delete(key)).toBe(false);
    expect(c.getStats().count).toBe(0);
    expect(c.getStats().ladderDepthHistogram).toEqual({});
    expect(c.get(key)).toBeUndefined();
    expect(c.getStats().thrashMisses).toBe(0);
  });

  it('makeKey namespaces by node path so identical views on different nodes never collide', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k1 = SliceCache.makeKey('/a', 'view');
    const k2 = SliceCache.makeKey('/b', 'view');
    expect(k1).not.toBe(k2);
    c.set(k1, entry(10, 'a'));
    c.set(k2, entry(10, 'b'));
    expect((c.get(k1)!.payload as { tag: string }).tag).toBe('a');
    expect((c.get(k2)!.payload as { tag: string }).tag).toBe('b');
  });
});

describe('SliceCache — scan-resistant eviction + thrash stats', () => {
  // Regression (loop-aware eviction): cyclic playback over a working set
  // larger than the budget. Plain LRU evicts exactly the entry needed
  // soonest after the wrap → 0 loop-2 hits. Scan-hinted stores evict from
  // the MRU end instead, keeping the loop-head prefix resident.
  function playLoop(c: SliceCache, nKeys: number, scan: boolean): number {
    let hits = 0;
    for (let i = 0; i < nKeys; i++) {
      const k = SliceCache.makeKey('/node', `t${i}`);
      if (c.get(k) !== undefined) hits++;
      else c.set(k, entry(100, `t${i}`), { scan });
    }
    return hits;
  }

  it('cyclic loop over an overflowing budget: 0 loop-2 hits under LRU, loop-head hits under scan (fail-pre-fix)', () => {
    // Budget holds 3 of the 10 per-loop entries.
    const lru = new SliceCache({ maxSize: 300 });
    playLoop(lru, 10, false); // loop 1 (cold)
    expect(playLoop(lru, 10, false)).toBe(0); // the LRU scan pathology

    const scan = new SliceCache({ maxSize: 300 });
    playLoop(scan, 10, true); // loop 1 (cold)
    const loop2 = playLoop(scan, 10, true);
    expect(loop2).toBeGreaterThanOrEqual(2); // loop-head prefix survives
    // And the hits are the loop HEAD (instant frames right after the wrap).
    expect(scan.get(SliceCache.makeKey('/node', 't0'))).toBeDefined();
  });

  it('counts eviction-induced misses as thrashMisses (vs. cold misses)', () => {
    const c = new SliceCache({ maxSize: 300 });
    playLoop(c, 10, false); // stores t0..t9, evicting most along the way

    expect(c.getStats().thrashMisses).toBe(0); // stores don't count misses

    c.get(SliceCache.makeKey('/node', 't0')); // was stored → evicted → thrash
    expect(c.getStats().thrashMisses).toBe(1);

    c.get(SliceCache.makeKey('/node', 'never-stored')); // cold miss
    expect(c.getStats().thrashMisses).toBe(1); // unchanged
    expect(c.getStats().misses).toBeGreaterThan(c.getStats().thrashMisses!);
  });

  it('re-storing an evicted key clears its tombstone (a later hit is not thrash)', () => {
    const c = new SliceCache({ maxSize: 300 });
    playLoop(c, 10, false);
    const t0 = SliceCache.makeKey('/node', 't0');
    c.get(t0); // thrash miss #1
    c.set(t0, entry(100, 't0')); // re-stored
    expect(c.get(t0)).toBeDefined(); // hit — not a miss of any kind
    expect(c.getStats().thrashMisses).toBe(1);
  });

  it('clear() resets thrashMisses and tombstones', () => {
    const c = new SliceCache({ maxSize: 300 });
    playLoop(c, 10, false);
    c.get(SliceCache.makeKey('/node', 't0'));
    expect(c.getStats().thrashMisses).toBe(1);
    c.clear();
    expect(c.getStats().thrashMisses).toBe(0);
    // Post-clear, an old key is a COLD miss (tombstones were reset).
    c.get(SliceCache.makeKey('/node', 't0'));
    expect(c.getStats().thrashMisses).toBe(0);
  });
});

describe('SliceCache — ladder depth stats', () => {
  it('reports prefix depths and full-ladder count without requiring metadata on every entry', () => {
    const cache = new SliceCache({ maxSize: 1000 });
    cache.set('plain', entry(100, 'plain'));
    cache.set('prefix', {
      ...entry(100, 'prefix'),
      ladderDepth: 2,
      totalLadderDepth: 5,
    });
    cache.set('full', {
      ...entry(100, 'full'),
      ladderDepth: 5,
      totalLadderDepth: 5,
    });

    expect(cache.getStats().fullLadderCount).toBe(1);
    expect(cache.getStats().ladderDepthHistogram).toEqual({ '2/5': 1, '5/5': 1 });
  });
});

describe('SliceCache — prefetch pin', () => {
  // The SlicePrefetcher stores a projected t+1 slice with `{ pin: true }`; it
  // lands as the MRU entry and, without protection, would be the first victim
  // of the next scan store. The pin keeps it resident until the foreground
  // tick restores (and thereby unpins) it.
  it('a pinned entry survives a subsequent overflowing scan store', () => {
    const c = new SliceCache({ maxSize: 250 });
    const a = SliceCache.makeKey('/n', 'a');
    const b = SliceCache.makeKey('/n', 'b');
    const d = SliceCache.makeKey('/n', 'd');
    c.set(a, entry(100, 'a'), { pin: true }); // prefetched t+1
    c.set(b, entry(100, 'b'));
    c.set(d, entry(100, 'd'), { scan: true }); // 300 > 250 → evict MRU-unpinned ('b')
    expect(c.has(a)).toBe(true);
    expect(c.has(b)).toBe(false);
    expect(c.has(d)).toBe(true);
  });

  it('a get hit releases the pin so the entry rejoins normal eviction', () => {
    const c = new SliceCache({ maxSize: 250 });
    const a = SliceCache.makeKey('/n', 'a');
    c.set(a, entry(100, 'a'), { pin: true });
    expect(c.get(a)).toBeDefined(); // foreground consumes → unpin
    c.set(SliceCache.makeKey('/n', 'b'), entry(100, 'b'));
    c.set(SliceCache.makeKey('/n', 'd'), entry(100, 'd')); // now 'a' (oldest, unpinned) evictable
    expect(c.has(a)).toBe(false);
  });
});

describe('SliceCache — markOversizedWarned (warn-once, cleared on clear)', () => {
  it('returns true once per key, then false — deduping the oversized warning', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'big');
    expect(c.markOversizedWarned(k)).toBe(true);
    expect(c.markOversizedWarned(k)).toBe(false);
    expect(c.markOversizedWarned(k)).toBe(false);
  });

  it('clear() resets the dedup so a later dataset can warn afresh (no module-global leak)', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'big');
    expect(c.markOversizedWarned(k)).toBe(true);
    c.clear();
    expect(c.markOversizedWarned(k)).toBe(true); // reset by clear()
  });
});

describe('SliceCache pinned perf gauges', () => {
  const pinned = () => [
    perfCounters.get('scache.pinnedEntries'),
    perfCounters.get('scache.pinnedBytes'),
  ];

  it('tracks pin on set, release on hit, and drop on clear', () => {
    perfCounters.reset();
    const c = new SliceCache({ maxSize: 1024 });
    const a = SliceCache.makeKey('/n', 'a');
    const b = SliceCache.makeKey('/n', 'b');
    c.set(a, entry(100), { pin: true });
    c.set(b, entry(50), { pin: true });
    c.set(SliceCache.makeKey('/n', 'u'), entry(10)); // unpinned: not counted
    expect(pinned()).toEqual([2, 150]);

    c.get(a); // the consuming hit releases the pin
    expect(pinned()).toEqual([1, 50]);
    c.get(a); // already unpinned: no change
    expect(pinned()).toEqual([1, 50]);

    c.clear();
    expect(pinned()).toEqual([0, 0]);
  });

  it('keeps a re-set pinned key pinned at its new size (as the LRU does)', () => {
    perfCounters.reset();
    const c = new SliceCache({ maxSize: 1024 });
    const a = SliceCache.makeKey('/n', 'a');
    c.set(a, entry(100), { pin: true });
    c.set(a, entry(300)); // replace without pin: LRU pin persists
    expect(pinned()).toEqual([1, 300]);
    c.delete(a);
    expect(pinned()).toEqual([0, 0]);
  });

  it('drops a pinned entry evicted as a last resort', () => {
    perfCounters.reset();
    const c = new SliceCache({ maxSize: 100 });
    c.set(SliceCache.makeKey('/n', 'a'), entry(60), { pin: true });
    c.set(SliceCache.makeKey('/n', 'b'), entry(60), { pin: true }); // evicts pinned a
    expect(pinned()).toEqual([1, 60]);
  });

  it('does not count a pin request for an oversized (rejected) entry', () => {
    perfCounters.reset();
    const c = new SliceCache({ maxSize: 100 });
    c.set(SliceCache.makeKey('/n', 'big'), entry(500), { pin: true });
    expect(pinned()).toEqual([0, 0]);
  });
});

describe('SliceCache stage outputs (#2944 B2)', () => {
  const stage = (sig: string, bytes: number) => ({ sig, value: { sig }, bytes });
  const pinned = () => [
    perfCounters.get('scache.pinnedEntries'),
    perfCounters.get('scache.pinnedBytes'),
  ];

  it('answers a stage lookup only for the same payload object and signature', () => {
    const c = new SliceCache({ maxSize: 1024 });
    const k = SliceCache.makeKey('/n', 'a');
    const e = entry(100);
    c.set(k, e);
    expect(c.setStage(k, e.payload, stage('s1', 50))).toBe(true);
    expect(c.getStage(k, e.payload, 's1')).toEqual({ sig: 's1' });
    expect(c.getStage(k, e.payload, 's2')).toBeUndefined(); // other params
    expect(c.getStage(k, { ...(e.payload as object) }, 's1')).toBeUndefined(); // other payload
    const s = c.getStats();
    expect([s.stageHits, s.stageMisses]).toEqual([1, 1]);
    // Bookkeeping reads never touch the slice hit/miss statistic.
    expect([s.hits, s.misses]).toEqual([0, 0]);
  });

  it('counts the stage inside the budget and keeps the entry recency and pin', () => {
    perfCounters.reset();
    const c = new SliceCache({ maxSize: 1000 });
    const a = SliceCache.makeKey('/n', 'a');
    const b = SliceCache.makeKey('/n', 'b');
    const ea = entry(100);
    c.set(a, ea, { pin: true });
    c.set(b, entry(100));
    expect(c.setStage(a, ea.payload, stage('s', 200))).toBe(true);
    expect(c.getStats().size).toBe(400);
    expect(c.getStats().stageBytes).toBe(200);
    expect(pinned()).toEqual([1, 300]);
    // Attaching did not promote `a`: after its pin is released it is still
    // the LRU entry, so an overflow that shedding alone cannot absorb evicts
    // `a`, not `b`.
    c.unpin(a);
    c.set(SliceCache.makeKey('/n', 'c'), entry(850));
    expect(c.has(a)).toBe(false);
    expect(c.has(b)).toBe(true);
    expect(c.getStats().stageBytes).toBe(0);
  });

  it('never evicts a slice to admit a stage output', () => {
    const c = new SliceCache({ maxSize: 300 });
    const a = SliceCache.makeKey('/n', 'a');
    const ea = entry(100);
    c.set(a, ea);
    c.set(SliceCache.makeKey('/n', 'b'), entry(150));
    expect(c.canAdmitStage(a, ea.payload, 80)).toBe(false);
    expect(c.setStage(a, ea.payload, stage('s', 80))).toBe(false);
    expect(c.getStats().count).toBe(2);
    expect(c.getStats().stageRejected).toBe(2);
    expect(c.getStage(a, ea.payload, 's')).toBeUndefined();
  });

  it('sheds stage outputs (oldest first) before evicting any slice', () => {
    const c = new SliceCache({ maxSize: 1000 });
    const keys = ['a', 'b', 'c'].map((v) => SliceCache.makeKey('/n', v));
    const entries = keys.map(() => entry(100));
    keys.forEach((k, i) => c.set(k, entries[i]));
    keys.forEach((k, i) => expect(c.setStage(k, entries[i].payload, stage('s', 200))).toBe(true));
    expect(c.getStats().size).toBe(900);
    // A 250-byte slice needs 150 bytes: shedding `a`'s stage (oldest) suffices.
    c.set(SliceCache.makeKey('/n', 'd'), entry(250));
    expect(c.getStats().count).toBe(4);
    expect(c.getStage(keys[0], entries[0].payload, 's')).toBeUndefined();
    expect(c.getStage(keys[1], entries[1].payload, 's')).toEqual({ sig: 's' });
    expect(c.getStats().stageShed).toBe(1);
    expect(c.getStats().size).toBe(950);
  });

  it('inside a playback scan: admits into free budget only and sheds most-recent first', () => {
    // Under a cyclic scan the OLDEST outputs are the ones the loop reaches
    // next, so displacing them for the newest would only churn.
    const c = new SliceCache({ maxSize: 1000 });
    const keys = ['a', 'b', 'c', 'd'].map((v) => SliceCache.makeKey('/n', v));
    const entries = keys.map(() => entry(100));
    keys.forEach((k, i) => c.set(k, entries[i], { scan: true }));
    // 400 slice bytes + two 200-byte outputs fit; the third does not.
    expect(c.setStage(keys[0], entries[0].payload, stage('s', 200), { scan: true })).toBe(true);
    expect(c.setStage(keys[1], entries[1].payload, stage('s', 200), { scan: true })).toBe(true);
    expect(c.canAdmitStage(keys[2], entries[2].payload, 250, { scan: true })).toBe(false);
    expect(c.setStage(keys[2], entries[2].payload, stage('s', 250), { scan: true })).toBe(false);
    expect(c.getStats().stageShed).toBe(0); // nothing displaced to try
    // Outside a scan the same request would shed the oldest output instead.
    expect(c.canAdmitStage(keys[2], entries[2].payload, 250)).toBe(true);
    // A scan slice store needing room sheds the MOST recent output (b's).
    c.set(SliceCache.makeKey('/n', 'e'), entry(250), { scan: true });
    expect(c.getStage(keys[0], entries[0].payload, 's')).toEqual({ sig: 's' });
    expect(c.getStage(keys[1], entries[1].payload, 's')).toBeUndefined();
    expect(c.getStats().count).toBe(5);
  });

  it('releases a stage output with its slice (eviction, replacement, clear)', () => {
    const c = new SliceCache({ maxSize: 1000 });
    const a = SliceCache.makeKey('/n', 'a');
    const ea = entry(100);
    c.set(a, ea);
    c.setStage(a, ea.payload, stage('s', 100));
    const upgraded = entry(150);
    c.set(a, upgraded); // storeLadder's upgrade-if-longer replacement
    expect(c.getStats().stageBytes).toBe(0);
    expect(c.getStats().size).toBe(150);
    expect(c.getStage(a, upgraded.payload, 's')).toBeUndefined();
    c.setStage(a, upgraded.payload, stage('s', 100));
    c.delete(a);
    expect(c.getStats().stageBytes).toBe(0);
    c.set(a, ea);
    c.setStage(a, ea.payload, stage('s', 100));
    c.clear();
    expect(c.getStats().stageBytes).toBe(0);
    expect(c.getStats().size).toBe(0);
  });
});
