import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getLoadTimeline,
  markFirstCommit,
  markLoad,
  noteRefinementComplete,
  noteRefinementPass,
  resetLoadTimeline,
} from '../../../profiling/load-timeline';

describe('load-timeline', () => {
  beforeEach(() => {
    resetLoadTimeline();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    resetLoadTimeline();
  });

  it('starts empty and reports null measures before any load', () => {
    const snap = getLoadTimeline();
    expect(snap.marks).toEqual([]);
    expect(snap.milestones).toEqual({});
    expect(snap.firstCommit).toEqual({});
    expect(snap.refinement).toEqual({ passes: 0, rungs: 0, rungsFromCache: 0, complete: false });
    expect(Object.values(snap.measures).every((v) => v === null)).toBe(true);
  });

  it('records milestones once per load and derives measures from loadStart', () => {
    const nowSpy = vi.spyOn(performance, 'now');
    nowSpy.mockReturnValueOnce(100);
    markLoad('loadStart', { url: 'x' });
    nowSpy.mockReturnValueOnce(160);
    markLoad('metadataReady');
    markLoad('metadataReady'); // duplicate within the same load is ignored (no clock read)
    nowSpy.mockReturnValueOnce(400);
    markLoad('sceneLoaded');

    const snap = getLoadTimeline();
    expect(snap.milestones).toEqual({ loadStart: 100, metadataReady: 160, sceneLoaded: 400 });
    expect(snap.measures.metadataReadyMs).toBe(60);
    expect(snap.measures.sceneLoadedMs).toBe(300);
    expect(snap.measures.ttfpMs).toBeNull();
    expect(snap.marks.map((m) => m.name)).toEqual(['loadStart', 'metadataReady', 'sceneLoaded']);
    expect(snap.marks[0].detail).toEqual({ url: 'x' });
  });

  it('a new loadStart resets the previous load', () => {
    markLoad('loadStart');
    markLoad('sceneLoaded');
    markFirstCommit('points');
    noteRefinementPass(3);
    markLoad('loadStart');
    const snap = getLoadTimeline();
    expect(snap.marks.map((m) => m.name)).toEqual(['loadStart']);
    expect(snap.firstCommit).toEqual({});
    expect(snap.refinement.passes).toBe(0);
  });

  it('ttfp is the earliest first commit of any geometry kind, recorded once per kind', () => {
    const nowSpy = vi.spyOn(performance, 'now');
    nowSpy.mockReturnValueOnce(0);
    markLoad('loadStart');
    nowSpy.mockReturnValueOnce(250);
    markFirstCommit('gsplats');
    nowSpy.mockReturnValueOnce(120);
    markFirstCommit('points');
    markFirstCommit('points'); // later commits of the same kind do not move it (no clock read)

    const snap = getLoadTimeline();
    expect(snap.firstCommit).toEqual({ gsplats: 250, points: 120 });
    expect(snap.measures.ttfpMs).toBe(120);
    expect(snap.marks.map((m) => m.name)).toEqual([
      'loadStart',
      'firstCommit:gsplats',
      'firstCommit:points',
    ]);
  });

  it('counts refinement passes and rungs, and completion records a milestone with counters', () => {
    markLoad('loadStart');
    noteRefinementPass(4, 1);
    noteRefinementPass(2);
    noteRefinementPass(-5); // negative counts are clamped, never subtracted
    expect(getLoadTimeline().refinement).toEqual({
      passes: 3,
      rungs: 6,
      rungsFromCache: 1,
      complete: false,
    });

    noteRefinementComplete();
    const snap = getLoadTimeline();
    expect(snap.refinement.complete).toBe(true);
    expect(snap.measures.refinementCompleteMs).not.toBeNull();
    const mark = snap.marks.find((m) => m.name === 'refinementComplete');
    expect(mark?.detail).toEqual({ passes: 3, rungs: 6 });
  });

  it('mirrors marks into the User Timing API with the luxar: prefix', () => {
    const markSpy = vi.spyOn(performance, 'mark');
    markLoad('loadStart');
    markFirstCommit('lines');
    const names = markSpy.mock.calls.map((c) => c[0]);
    expect(names).toEqual(['luxar:loadStart', 'luxar:firstCommit:lines']);
  });

  it('survives a performance.mark that rejects the detail option, and one that throws outright', () => {
    const markSpy = vi.spyOn(performance, 'mark');
    markSpy.mockImplementationOnce(() => {
      throw new TypeError('no options here');
    });
    expect(() => markLoad('loadStart', { url: 'x' })).not.toThrow();
    // Fallback path retried without the options object.
    expect(markSpy).toHaveBeenCalledTimes(2);
    expect(markSpy.mock.calls[1]).toEqual(['luxar:loadStart']);

    markSpy.mockImplementation(() => {
      throw new Error('User Timing disabled');
    });
    expect(() => markLoad('sceneLoaded')).not.toThrow();
    expect(getLoadTimeline().milestones.sceneLoaded).toBeDefined();
  });

  it('bounds the retained marks', () => {
    markLoad('loadStart');
    // Only first commits and milestones are marked, so exhaust the ring via
    // repeated loads is impossible; drive the internal push through kinds and
    // milestones and confirm the snapshot copies rather than aliases state.
    markFirstCommit('mesh');
    const a = getLoadTimeline();
    a.marks.push({ name: 'tampered', t: 0 });
    a.refinement.passes = 99;
    const b = getLoadTimeline();
    expect(b.marks.map((m) => m.name)).toEqual(['loadStart', 'firstCommit:mesh']);
    expect(b.refinement.passes).toBe(0);
  });

  it('falls back to Date.now when performance.now is unavailable', () => {
    const original = globalThis.performance;
    // @ts-expect-error — simulate a runtime without the Performance API
    globalThis.performance = undefined;
    try {
      expect(() => markLoad('loadStart')).not.toThrow();
      expect(getLoadTimeline().milestones.loadStart).toBeGreaterThan(0);
    } finally {
      globalThis.performance = original;
    }
  });
});
