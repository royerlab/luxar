/**
 * Unit tests for the shared progressive-loader streaming policy
 * (data/loaders/progressive/streaming-policy.ts).
 *
 * The three geometry loaders (GSplats/Points/Lines) all drive their LOD
 * streaming loop through these pure decisions, so pinning them here keeps the
 * three loops identical by construction and documents the contract:
 *
 *  - playback: responsive — stream CACHE-RESIDENT levels within the pass
 *    budget, stop at the first cold/slow one, never block on a cold level.
 *  - prefetch: background — deepen toward the full ladder, never stop on a
 *    cache miss (bounded by budget + abort in the loop, not here).
 *  - refine:   foreground, unbudgeted — stream resident levels, stop at the
 *    first cold/slow one.
 */

import { describe, it, expect } from 'vitest';
import {
  classifyStreamingPass,
  shouldStopAfterLevel,
} from '../../../../../data/loaders/progressive/streaming-policy';
import { CACHE_HIT_THRESHOLD_MS } from '../../../../../data/loaders/progressive/constants';

describe('classifyStreamingPass', () => {
  it('is prefetch whenever the pass is a background shadow, budget or not', () => {
    expect(classifyStreamingPass(true, true)).toBe('prefetch');
    expect(classifyStreamingPass(false, true)).toBe('prefetch');
  });

  it('is playback for a budgeted foreground pass', () => {
    expect(classifyStreamingPass(true, false)).toBe('playback');
  });

  it('is refine for an unbudgeted foreground pass (static view / refine-on-pause)', () => {
    expect(classifyStreamingPass(false, false)).toBe('refine');
  });
});

describe('shouldStopAfterLevel', () => {
  const fast = CACHE_HIT_THRESHOLD_MS - 1;
  const slow = CACHE_HIT_THRESHOLD_MS + 1;

  it('refine stops at the first cold (cache-miss) level past the floor', () => {
    // allResident=false ⇒ this level required a fresh fetch.
    expect(shouldStopAfterLevel('refine', 2, 0, false, fast)).toBe(true);
    // ...but always keeps the ≥1-level floor (level === startLevel never stops).
    expect(shouldStopAfterLevel('refine', 0, 0, false, slow)).toBe(false);
  });

  it('refine stops at the first slow level even when resident (GC/probe-gap guard)', () => {
    expect(shouldStopAfterLevel('refine', 2, 0, true, slow)).toBe(true);
    expect(shouldStopAfterLevel('refine', 2, 0, true, fast)).toBe(false);
  });

  it('prefetch never stops here — it deepens regardless of residency', () => {
    expect(shouldStopAfterLevel('prefetch', 5, 0, false, slow)).toBe(false);
  });

  // Playback shares refine's rule: residency, not level index, is what
  // separates "affordable inside a tick" from "stalls the tick" (#2374/#2376).
  it('playback stops at the first cold level past the floor', () => {
    expect(shouldStopAfterLevel('playback', 1, 0, false, fast)).toBe(true);
    expect(shouldStopAfterLevel('playback', 7, 3, false, fast)).toBe(true);
  });

  it('playback stops at the first slow level past the floor, even when resident', () => {
    expect(shouldStopAfterLevel('playback', 1, 0, true, slow)).toBe(true);
  });

  it('playback KEEPS STREAMING while levels are resident and fast', () => {
    // The regression this replaces: playback used to load level 0 and nothing
    // else, so a sliced node whose LOD 0 is a few dozen elements rendered an
    // empty frame while levels 1..k sat in cache. Resident + fast must not stop.
    for (const level of [1, 2, 5, 13]) {
      expect(shouldStopAfterLevel('playback', level, 0, true, fast)).toBe(false);
    }
  });

  it('playback never abandons the >=1-level first-paint floor', () => {
    // level === startLevel is the floor: it must load even when cold AND slow,
    // otherwise a cold slice commits nothing at all.
    expect(shouldStopAfterLevel('playback', 0, 0, false, slow)).toBe(false);
    expect(shouldStopAfterLevel('playback', 4, 4, false, slow)).toBe(false);
  });
});
