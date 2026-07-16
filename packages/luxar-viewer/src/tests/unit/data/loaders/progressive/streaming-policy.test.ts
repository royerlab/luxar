/**
 * Unit tests for the shared progressive-loader streaming policy
 * (data/loaders/progressive/streaming-policy.ts).
 *
 * The three geometry loaders (GSplats/Points/Lines) all drive their LOD
 * streaming loop through these pure decisions, so pinning them here keeps the
 * three loops identical by construction and documents the contract:
 *
 *  - playback: responsive — commit the cached prefix + a LOD-0 first-paint
 *    floor, never block on fine levels.
 *  - prefetch: background — deepen toward the full ladder, never stop on a
 *    cache miss (bounded by budget + abort in the loop, not here).
 *  - refine:   foreground, unbudgeted — stream resident levels, stop at the
 *    first cold/slow one.
 */

import { describe, it, expect } from 'vitest';
import {
  classifyStreamingPass,
  shouldLoadLevel,
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

describe('shouldLoadLevel', () => {
  it('playback loads ONLY the LOD-0 first-paint floor when nothing is cached', () => {
    expect(shouldLoadLevel('playback', 0, 0)).toBe(true); // floor, cold slice
    expect(shouldLoadLevel('playback', 1, 0)).toBe(false); // never a 2nd fine level
  });

  it('playback loads NOTHING new when a prefix is already cached (commits it as-is)', () => {
    // startLevel > 0 ⇒ a restored prefix; the loop must not decode further.
    expect(shouldLoadLevel('playback', 3, 3)).toBe(false);
    expect(shouldLoadLevel('playback', 4, 3)).toBe(false);
  });

  it('prefetch and refine load every level (bounded elsewhere)', () => {
    for (const kind of ['prefetch', 'refine'] as const) {
      expect(shouldLoadLevel(kind, 0, 0)).toBe(true);
      expect(shouldLoadLevel(kind, 5, 2)).toBe(true);
    }
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

  it('playback never stops here (it is gated by shouldLoadLevel instead)', () => {
    expect(shouldStopAfterLevel('playback', 2, 0, false, slow)).toBe(false);
  });
});
