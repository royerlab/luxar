/**
 * The level-dissolve state machine on its own (`scene/lod-dissolve.ts`). The
 * registry-level behaviour (what is drawn, at which opacity) is pinned in
 * `lod-group-registry.test.ts`; this file pins the bookkeeping contract the
 * registry builds on, including its liveness (`tickUntilMs`).
 */

import { describe, expect, it } from 'vitest';
import { LodDissolves, retargetedFade, type DissolveEntry } from '../../../scene/lod-dissolve';
import { NO_TICK, UNTIL_RESOLVED } from '../../../scene/tick-demand';

const FADE_MS = 100;
const always = { canDissolve: () => true };
const never = { canDissolve: () => false };

function frame(nowMs: number) {
  return { nowMs, fadeMs: FADE_MS, version: null };
}

describe('LodDissolves', () => {
  it('starts when the displayed level changes and ends once the dissolve lands', () => {
    const d = new LodDissolves<DissolveEntry>();
    const entry: DissolveEntry = { path: '/g', displayedChildIndex: 0 };
    expect(d.tickUntilMs()).toBe(NO_TICK);
    const fade = d.advance(entry, 1, frame(1000), always);
    expect(fade).toMatchObject({ fromIdx: 0, toIdx: 1, progress: 0 });
    expect(d.isAnimating()).toBe(true);
    expect(d.tickUntilMs()).toBe(UNTIL_RESOLVED);
    expect(d.fadingFrom('/g')).toBe(0);

    entry.displayedChildIndex = 1;
    expect(d.advance(entry, 1, frame(1050), always)?.progress).toBeCloseTo(0.5, 9);
    expect(d.advance(entry, 1, frame(1100), always)).toBeNull(); // landed
    expect(d.isAnimating()).toBe(false);
    expect(d.tickUntilMs()).toBe(NO_TICK);
  });

  it('ends at once when the pair cannot dissolve', () => {
    const d = new LodDissolves<DissolveEntry>();
    expect(d.advance({ path: '/g', displayedChildIndex: 0 }, 1, frame(0), never)).toBeNull();
    expect(d.isAnimating()).toBe(false);
  });

  it('a drop records the outgoing level until the visibility pass settles it', () => {
    const d = new LodDissolves<DissolveEntry>();
    d.advance({ path: '/g', displayedChildIndex: 2 }, 1, frame(0), always);
    d.drop('/g');
    expect(d.isAnimating()).toBe(false);
    expect(d.droppedFromIdx('/g')).toBe(2);
    d.settled('/g');
    expect(d.droppedFromIdx('/g')).toBe(-1);

    d.advance({ path: '/a', displayedChildIndex: 0 }, 1, frame(0), always);
    d.advance({ path: '/b', displayedChildIndex: 3 }, 2, frame(0), always);
    d.dropAll();
    expect(d.droppedFromIdx('/a')).toBe(0);
    expect(d.droppedFromIdx('/b')).toBe(3);
    expect(d.tickUntilMs()).toBe(NO_TICK);
  });
});

describe('retargetedFade', () => {
  it('reversing an in-flight dissolve keeps each level at its current opacity', () => {
    const inFlight = { fromIdx: 0, toIdx: 1, startMs: 0, startProgress: 0, progress: 0.3 };
    expect(retargetedFade(inFlight, 1, 0, 500)).toMatchObject({
      fromIdx: 1,
      toIdx: 0,
      startProgress: 0.7,
    });
  });

  it('retargeting to a third level keeps the more opaque one as the outgoing level', () => {
    const early = { fromIdx: 0, toIdx: 1, startMs: 0, startProgress: 0, progress: 0.2 };
    expect(retargetedFade(early, 1, 2, 500)).toMatchObject({ fromIdx: 0, startProgress: 0.2 });
    const late = { fromIdx: 0, toIdx: 1, startMs: 0, startProgress: 0, progress: 0.8 };
    expect(retargetedFade(late, 1, 2, 500).fromIdx).toBe(1);
  });
});
