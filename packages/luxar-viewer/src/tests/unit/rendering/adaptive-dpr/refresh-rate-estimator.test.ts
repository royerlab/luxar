import { describe, it, expect } from 'vitest';
import { RefreshRateEstimator } from '../../../../rendering/adaptive-dpr/refresh-rate-estimator';

/** Feed `count` samples of `fps` at 500ms cadence starting at `start`. */
function feed(
  est: RefreshRateEstimator,
  fps: number,
  count: number,
  start: number,
  jitter = 0
): number {
  let t = start;
  for (let i = 0; i < count; i++) {
    t = start + i * 500;
    est.addSample(fps + (i % 2 === 0 ? jitter : -jitter), t);
  }
  return t;
}

describe('RefreshRateEstimator', () => {
  it('returns the fallback before any samples', () => {
    const est = new RefreshRateEstimator(60);
    expect(est.getCap()).toBe(60);
  });

  it('learns a higher cap immediately from a single fast sample (120Hz display)', () => {
    const est = new RefreshRateEstimator(60);
    est.addSample(119.5, 0);
    expect(est.getCap()).toBeCloseTo(119.5, 5);
  });

  it('never lets a heavy scene drag the cap below the fallback', () => {
    const est = new RefreshRateEstimator(60);
    // A session that STARTS heavy: minutes of stable 20fps. The cap must
    // hold at the fallback so scale-down stays armed.
    feed(est, 20, 8, 0, 3); // 17..23 — spread 26% > 15%, NOT uniform
    expect(est.getCap()).toBe(60);
  });

  it('a steady heavy scene is NEVER misclassified as a throttled display (proven-rate guard)', () => {
    const est = new RefreshRateEstimator(60);
    // GPU-bound scene parked at the DPR floor: perfectly uniform 20fps
    // for far longer than the 10s downshift delay. The display never
    // proved it can beat 80% of the fallback, so the throttle detector
    // must stay cold — otherwise the cap collapses to 20, the relative
    // thresholds read 20fps as healthy (down 15 / up 18), scale-down is
    // disarmed and scale-up fires on the scene that most needs help.
    feed(est, 20, 60, 0); // 30s of uniform 20fps
    expect(est.getCap()).toBe(60);
  });

  it('keeps the high-water mark under subsequent load (60fps session, then heavy)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 61, 4, 0);
    feed(est, 20, 8, 2000, 3); // heavy but jittery
    expect(est.getCap()).toBeCloseTo(61, 5);
  });

  it('downshifts after a sustained uniform-low signature (120Hz → 30Hz throttle)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    expect(est.getCap()).toBe(120);

    // rAF throttled to 30: perfectly uniform 30fps samples. Uniform-low
    // must persist 10s beyond the first detection before the downshift.
    const t = feed(est, 30, 25, 2000); // 12s of uniform 30
    void t;
    expect(est.getCap()).toBe(30);
  });

  it('recovers instantly when the throttle lifts', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    feed(est, 30, 25, 2000); // downshifted to 30
    expect(est.getCap()).toBe(30);

    est.addSample(118, 20_000); // throttle lifted
    expect(est.getCap()).toBe(118);
  });

  it('a session throttled from the very first frame keeps the fallback cap (never proved a rate)', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 30, 25, 0); // 30Hz-throttled from the start — mark never beat 48
    // Without a proven higher rate the uniform-low signature is
    // indistinguishable from a heavy scene, so no downshift: the cap
    // holds at the fallback and the (futile) scale-down probes are
    // contained by the rejection-backoff machinery instead.
    expect(est.getCap()).toBe(60);
  });

  it('a downshifted sub-fallback cap survives slow samples until a fast one appears', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0); // prove the display can do 120
    feed(est, 30, 25, 2000); // genuine 120→30 throttle → downshift
    expect(est.getCap()).toBe(30);

    // Still throttled: a slightly higher-but-slow sample doesn't restore
    // the fallback bound (48 = 0.8 × 60 is the un-throttle line).
    est.addSample(35, 20_000);
    expect(est.getCap()).toBeLessThan(60);
  });

  it('clear() forgets the mark and the throttle state', () => {
    const est = new RefreshRateEstimator(60);
    feed(est, 120, 4, 0);
    est.clear();
    expect(est.getCap()).toBe(60);
  });
});
