import { describe, it, expect } from 'vitest';
import { StallDetector } from '../../../../rendering/adaptive-dpr/stall-detector';

/** Feed `count` identical intervals; returns the verdicts in order. */
function feed(detector: StallDetector, intervalMs: number, count: number): boolean[] {
  const verdicts: boolean[] = [];
  for (let i = 0; i < count; i++) verdicts.push(detector.isStall(intervalMs));
  return verdicts;
}

const FRAME_60 = 1000 / 60;

describe('StallDetector', () => {
  it('flags an isolated stall in a healthy stream and clears immediately after', () => {
    const d = new StallDetector(350);
    expect(feed(d, FRAME_60, 10)).not.toContain(true);

    expect(d.isStall(5000)).toBe(true); // 300× the 16.7ms median
    // The very next healthy frame must NOT read as a stall: the 5s entry
    // is one of five in the ring and cannot move the median.
    expect(d.isStall(FRAME_60)).toBe(false);
    expect(feed(d, FRAME_60, 10)).not.toContain(true);
  });

  it('flags TWO consecutive hitches in a healthy stream (both are outliers)', () => {
    const d = new StallDetector(350);
    feed(d, FRAME_60, 10);
    expect(d.isStall(400)).toBe(true);
    expect(d.isStall(400)).toBe(true);
  });

  it('respects the absolute floor: a large ratio under minGapMs is not dead time', () => {
    const d = new StallDetector(350);
    feed(d, 8, 10); // ~120fps
    // 40× the median, but 320ms of dead time is not worth discarding a
    // window over — the floor vetoes it.
    expect(d.isStall(320)).toBe(false);
    expect(d.isStall(360)).toBe(true); // just over the floor, still 45×
  });

  it('converges on a sustained slow cadence within a few intervals', () => {
    const d = new StallDetector(350);
    feed(d, FRAME_60, 10);
    // 0.5fps: every interval clears the floor, but once the median
    // follows the new cadence the outlier test stops firing.
    const verdicts = feed(d, 2000, 8);
    expect(verdicts.slice(0, 2)).toEqual([true, true]); // median still fast
    expect(verdicts.slice(2)).not.toContain(true); // 3 of 5 → median 2000
  });

  it('converges on an alternating slow cadence (400/300ms)', () => {
    const d = new StallDetector(350);
    feed(d, FRAME_60, 10);
    const verdicts: boolean[] = [];
    for (let i = 0; i < 20; i++) verdicts.push(d.isStall(i % 2 === 0 ? 400 : 300));
    // Only the first 400ms interval is an outlier against the 60fps
    // median; from then on this cadence IS the frame rate.
    expect(verdicts[0]).toBe(true);
    expect(verdicts.slice(1)).not.toContain(true);
  });

  it('converges on a bursty alternating cadence (500/100ms)', () => {
    const d = new StallDetector(350);
    feed(d, FRAME_60, 10);
    const verdicts: boolean[] = [];
    for (let i = 0; i < 20; i++) verdicts.push(d.isStall(i % 2 === 0 ? 500 : 100));
    // The 100ms halves keep the median low for one more cycle, so the
    // 3rd interval is still read as dead time — then it converges.
    expect(verdicts.filter(Boolean).length).toBe(2);
    expect(verdicts.slice(4)).not.toContain(true);
  });

  it('catches a stall again once frames get fast again', () => {
    const d = new StallDetector(350);
    feed(d, FRAME_60, 10);
    feed(d, 2000, 8); // slow episode, converged
    feed(d, FRAME_60, 10); // scene lightens
    expect(d.isStall(5000)).toBe(true);
  });

  it('falls back to the absolute floor while the cadence memory is cold', () => {
    // A stall in the FIRST measured interval of a session (startup, the
    // frame after a resume) has nothing to compare against — it would be
    // its own median — so the floor decides alone.
    const d = new StallDetector(350);
    expect(d.isStall(5000)).toBe(true);
    // ...and a second identical interval is the frame rate, not a stall.
    expect(d.isStall(5000)).toBe(false);
  });

  it('clear() forgets the cadence (session boundary)', () => {
    const d = new StallDetector(350);
    feed(d, 2000, 8); // converged on a slow cadence: 2000ms is normal
    expect(d.isStall(2000)).toBe(false);

    d.clear();
    // Cold again → floor-only, so the same interval now reads as a stall.
    expect(d.isStall(2000)).toBe(true);
  });

  it('honours explicit outlierFactor and windowSize', () => {
    // A tighter factor over a shorter memory: 2× the median of the last
    // 3 intervals.
    const d = new StallDetector(350, 2, 3);
    feed(d, 400, 3); // median 400
    expect(d.isStall(700)).toBe(false); // 1.75× — not an outlier
    expect(d.isStall(2000)).toBe(true); // ~2.9× the (now 700) median

    // Only 3 intervals are remembered, so a sustained cadence takes over
    // faster than with the default window.
    const e = new StallDetector(350, 4, 3);
    feed(e, FRAME_60, 5);
    expect(feed(e, 2000, 4)).toEqual([true, false, false, false]);
  });

  it('clamps degenerate outlierFactor / windowSize instead of going absolute-only', () => {
    // windowSize 0 would leave the memory permanently empty and every
    // over-floor interval "cold" — i.e. the pre-fix absolute rule, back
    // in silently. outlierFactor 0 would make every interval an outlier.
    const d = new StallDetector(350, 0, 0);
    expect(d.isStall(2000)).toBe(true); // cold: floor only
    expect(feed(d, 2000, 4)).not.toContain(true); // memory works, 1× median
  });

  it.each([
    ['NaN', NaN],
    ['Infinity', Infinity],
    ['-Infinity', -Infinity],
  ])('treats a %s knob as absent rather than propagating it', (_label, bad) => {
    // `Math.max(2, NaN)` is NaN and `Math.max(2, Infinity)` is Infinity,
    // so a plain clamp lets both through. A NaN/Infinity windowSize
    // means the ring NEVER trims (unbounded growth plus an O(n log n)
    // median every frame) and a NaN outlierFactor makes every comparison
    // false, so the detector never fires at all. Both must behave
    // exactly like the defaults.
    const outlier = new StallDetector(350, bad);
    feed(outlier, FRAME_60, 10);
    expect(outlier.isStall(5000)).toBe(true); // 300× — a stall, as with factor 4
    expect(outlier.isStall(FRAME_60)).toBe(false);

    const window = new StallDetector(350, 4, bad);
    feed(window, FRAME_60, 10);
    // Default window of 5: a sustained 2s cadence converges after
    // exactly two misreads (3 of 5 → median 2000). A shorter ring
    // converges sooner, an untrimmed one never converges at all.
    const verdicts = feed(window, 2000, 8);
    expect(verdicts.slice(0, 2)).toEqual([true, true]);
    expect(verdicts.slice(2)).not.toContain(true);
  });
});
