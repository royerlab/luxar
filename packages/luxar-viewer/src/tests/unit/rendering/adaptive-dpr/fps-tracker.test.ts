import { describe, it, expect } from 'vitest';
import { FPSTracker } from '../../../../rendering/adaptive-dpr/fps-tracker';

/** Push frames at a fixed cadence; returns the last timestamp pushed. */
function pushAt(tracker: FPSTracker, start: number, count: number, intervalMs: number): number {
  let t = start;
  for (let i = 0; i < count; i++) {
    t = start + i * intervalMs;
    tracker.push(t);
  }
  return t;
}

describe('FPSTracker', () => {
  it('returns 0 with fewer than two samples', () => {
    const tracker = new FPSTracker(1000);
    expect(tracker.getFPS()).toBe(0);
    tracker.push(100);
    expect(tracker.getFPS()).toBe(0);
    expect(tracker.span()).toBe(0);
  });

  it('computes interval-normalized FPS at a steady cadence', () => {
    const tracker = new FPSTracker(1000);
    pushAt(tracker, 0, 61, 1000 / 60);
    expect(tracker.getFPS()).toBeCloseTo(60, 1);
  });

  it('trims samples older than the window', () => {
    const tracker = new FPSTracker(1000);
    // 10 fps for 2s: only the last ~1s of samples should remain.
    pushAt(tracker, 0, 21, 100);
    expect(tracker.sampleCount()).toBeLessThanOrEqual(11);
    expect(tracker.span()).toBeLessThanOrEqual(1000);
    expect(tracker.getFPS()).toBeCloseTo(10, 1);
  });

  it('returns 0 for a degenerate non-positive span', () => {
    const tracker = new FPSTracker(1000);
    tracker.push(500);
    tracker.push(500); // duplicate timestamp — span 0
    expect(tracker.getFPS()).toBe(0);
  });

  it('compacts internally without changing results (long run)', () => {
    const tracker = new FPSTracker(1000);
    // 10s of 120fps — far past the compaction threshold of 120 trimmed
    // samples; the estimate must stay exact throughout.
    pushAt(tracker, 0, 1201, 1000 / 120);
    expect(tracker.getFPS()).toBeCloseTo(120, 0);
    expect(tracker.sampleCount()).toBeLessThanOrEqual(121);
  });

  it('stays defined below one frame per window (min retention)', () => {
    const tracker = new FPSTracker(1000);
    // 0.5fps: the previous frame is 2000ms old — twice the window — so
    // pure age trimming would leave a single sample and report 0 ("not
    // enough data") exactly where shedding pixels matters most.
    pushAt(tracker, 0, 4, 2000);
    expect(tracker.sampleCount()).toBe(2);
    expect(tracker.span()).toBe(2000);
    expect(tracker.getFPS()).toBeCloseTo(0.5, 3);
  });

  it('does NOT widen the window at healthy frame rates (min-retention control)', () => {
    // Negative control for the retention floor: at 60fps far more than
    // the two retained samples fit inside the window, so the floor never
    // binds and the trimming behaviour is unchanged.
    const tracker = new FPSTracker(1000);
    pushAt(tracker, 0, 300, 1000 / 60); // 5s of 60fps
    expect(tracker.sampleCount()).toBeLessThanOrEqual(61);
    expect(tracker.span()).toBeLessThanOrEqual(1000);
    expect(tracker.getFPS()).toBeCloseTo(60, 1);
  });

  it('honours an explicit minRetainedSamples above the default', () => {
    const tracker = new FPSTracker(1000, 4);
    pushAt(tracker, 0, 6, 2000); // every sample is older than the window
    expect(tracker.sampleCount()).toBe(4);
    expect(tracker.span()).toBe(6000);
    expect(tracker.getFPS()).toBeCloseTo(0.5, 3);
  });

  it('clamps minRetainedSamples below 2 (the floor is an invariant, not a preference)', () => {
    // A caller passing 0/1 would silently restore the undefined-estimate
    // failure the retention floor exists to prevent, so the ctor refuses.
    for (const bad of [0, 1, -5, 1.9, Number.NaN]) {
      const tracker = new FPSTracker(1000, bad);
      pushAt(tracker, 0, 4, 2000); // every sample older than the window
      expect(tracker.sampleCount()).toBe(2);
      expect(tracker.getFPS()).toBeCloseTo(0.5, 3);
    }
  });

  it('exposes the newest timestamp and clears fully', () => {
    const tracker = new FPSTracker(1000);
    expect(tracker.lastTimestamp).toBeNull();
    pushAt(tracker, 0, 5, 10);
    expect(tracker.lastTimestamp).toBe(40);

    tracker.clear();
    expect(tracker.lastTimestamp).toBeNull();
    expect(tracker.sampleCount()).toBe(0);
    expect(tracker.getFPS()).toBe(0);
  });
});
