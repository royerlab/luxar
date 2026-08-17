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
    // The very next healthy frame must NOT read as a stall: it is under
    // the absolute floor whatever the 5s entry did to the median.
    expect(d.isStall(FRAME_60)).toBe(false);
    expect(feed(d, FRAME_60, 10)).not.toContain(true);
  });

  it('flags TWO consecutive hitches in a healthy stream (both are outliers)', () => {
    const d = new StallDetector(350);
    feed(d, FRAME_60, 10);
    expect(d.isStall(400)).toBe(true);
    expect(d.isStall(400)).toBe(true);
  });

  it.each([
    ['an EMPTY', 0],
    ['a ONE-interval', 1],
    ['a TWO-interval', 2],
    ['a THREE-interval', 3],
  ])('catches the first stall against %s cadence memory', (_label, warm) => {
    // The interval under test must never be part of the median it is
    // judged against. It used to be pushed FIRST, so with a two-sample
    // memory the candidate was half of the mean: after a single warm
    // frame a 5s shader-compile stall measured as "the frame rate"
    // (verdicts false/false/false), and the manager reduced DPR and
    // armed a probe on a window holding nothing but dead time — the
    // ordinary startup and idle-resume paths.
    const d = new StallDetector(350);
    feed(d, FRAME_60, warm);
    expect(d.isStall(5000)).toBe(true);
  });

  it.each([2, 3, 10])(
    'discards up to TWO consecutive stalls and absorbs the third (%i warm intervals)',
    (warm) => {
      // The documented bound the manager's cadence-trust gate is sized
      // against, and it must not depend on how long the session has been
      // running: by the third stall half the memory is dead time, the
      // median has risen to ~2.5s, and 5s is no longer a 4× outlier.
      const d = new StallDetector(350);
      feed(d, FRAME_60, warm);
      expect(feed(d, 5000, 4)).toEqual([true, true, false, false]);
    }
  );

  it('respects the absolute floor: a large ratio under minGapMs is not dead time', () => {
    const d = new StallDetector(350);
    feed(d, 8, 10); // ~120fps
    // 40× the median, but 320ms of dead time is not worth discarding a
    // window over — the floor vetoes it.
    expect(d.isStall(320)).toBe(false);
    expect(d.isStall(360)).toBe(true); // just over the floor, still 45×
  });

  it('needs STRICTLY more than minGapMs: exactly the floor is not dead time', () => {
    const d = new StallDetector(350);
    feed(d, 8, 4); // ~120fps median
    expect(d.isStall(350)).toBe(false); // 44× the median, but exactly the floor
    expect(d.isStall(350.001)).toBe(true);
  });

  it('keeps an over-floor interval that is only a MODEST multiple of the median', () => {
    // The "this is just the frame rate" branch, exercised well away from
    // ratio 1: a 700ms frame in a 200ms cadence clears the 350ms floor
    // and is 3.5× the median, which is not enough. Pins the outlier
    // factor from BELOW — a factor of 3 would discard this window.
    const d = new StallDetector(350);
    feed(d, 200, 5);
    expect(d.isStall(700)).toBe(false);
    expect(d.isStall(900)).toBe(true); // 4.5× — an outlier after all
  });

  it('averages the two middle neighbours when the memory is even', () => {
    // The even-count median branch, which is what a short (post-boundary)
    // memory and the default 4-neighbour memory both take. A lower- or
    // upper-middle convention would move the threshold to 100 or 500.
    const d = new StallDetector(350, 4, 2);
    d.isStall(100);
    d.isStall(500);
    // Memory [100, 500] → median 300 → the threshold is 1200.
    expect(d.isStall(1000)).toBe(false);

    const e = new StallDetector(350, 4, 2);
    e.isStall(100);
    e.isStall(500);
    expect(e.isStall(1300)).toBe(true);
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
    // A tighter factor over a shorter memory: 2× the median of the 3
    // PRECEDING intervals.
    const d = new StallDetector(350, 2, 3);
    feed(d, 400, 3); // memory [400, 400, 400]
    expect(d.isStall(700)).toBe(false); // 1.75× — not an outlier
    expect(d.isStall(2000)).toBe(true); // 5× the 400ms median

    // An ODD memory of 3 takes the median of the 3 preceding intervals,
    // so a sustained cadence needs a strict majority of slow neighbours:
    // two misreads, then absorbed. (The default even memory averages the
    // two middle values, so it absorbs one interval sooner.)
    const e = new StallDetector(350, 4, 3);
    feed(e, FRAME_60, 5);
    expect(feed(e, 2000, 4)).toEqual([true, true, false, false]);
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
  ])('treats a %s minGapMs as absent rather than propagating it', (_label, bad) => {
    // The floor was the one knob assigned raw. `!(intervalMs > NaN)` is
    // true for EVERY interval, so a NaN floor makes isStall answer false
    // forever: dead time is never discarded, a tab-resume gap folds into
    // the FPS window of a healthy 60fps session, reads as ~0.2fps and
    // ratchets a probe-ratified scale-down with a learned floor. An
    // Infinite floor is the same failure by a different route.
    const d = new StallDetector(bad);
    feed(d, FRAME_60, 6);
    expect(d.isStall(5000)).toBe(true); // as with the 350ms default
    expect(d.isStall(FRAME_60)).toBe(false);
    // -Infinity fails the OTHER way and needs its own probe: unguarded
    // it survives `Math.max(0, ...)` as a floor of 0, which the two
    // assertions above cannot see (a 5s gap is dead time and a 16.7ms
    // interval is not, either way). A 100ms hiccup is what separates
    // them — a 6× outlier against the 16.7ms median, but far under the
    // 350ms floor, so it must NOT cost the window.
    expect(d.isStall(100)).toBe(false);
  });

  it('falls back to the absolute floor when the median is not positive', () => {
    // Duplicate frame timestamps produce 0ms intervals; `factor * 0` is
    // 0, so without a guard every over-floor interval is an "outlier"
    // and the absolute-threshold-only rule is silently back.
    const zeros = new StallDetector(350);
    feed(zeros, 0, 4);
    expect(zeros.isStall(5000)).toBe(true);
    expect(zeros.isStall(300)).toBe(false); // the floor still vetoes

    // A NaN median is the opposite failure: every comparison against it
    // is false, so the detector would go permanently blind after one
    // NaN interval was remembered.
    const poisoned = new StallDetector(350);
    feed(poisoned, FRAME_60, 4);
    expect(feed(poisoned, NaN, 3)).not.toContain(true); // never dead time itself
    expect(poisoned.isStall(5000)).toBe(true);
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
    // Default memory of 4 neighbours: a sustained 2s cadence converges
    // after exactly two misreads (by the third, half the memory is the
    // new cadence). A shorter memory converges sooner, an untrimmed one
    // never converges at all.
    const verdicts = feed(window, 2000, 8);
    expect(verdicts.slice(0, 2)).toEqual([true, true]);
    expect(verdicts.slice(2)).not.toContain(true);
  });
});
