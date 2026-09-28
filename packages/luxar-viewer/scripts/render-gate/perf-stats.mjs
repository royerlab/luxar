/**
 * Statistics for the render gate's performance verdict.
 *
 * Each metric is measured in R interleaved rounds per build. The verdict
 * compares the candidate/baseline ratio of medians against the noise the SAME
 * measurement shows between two copies of the baseline (the A/A control), so a
 * scene that is inherently noisy on this machine needs a larger effect before
 * it fails, and a quiet scene fails on a small one.
 *
 * @module scripts/render-gate/perf-stats
 */

/**
 * @param {number[]} values Samples.
 * @returns {number} The median (mean of the middle two for even length); NaN when empty.
 */
export function median(values) {
  if (values.length === 0) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/**
 * Deterministic PRNG (mulberry32) so a report is reproducible from its samples.
 *
 * @param {number} seed Integer seed.
 * @returns {() => number} Uniform [0, 1) generator.
 */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Bootstrap 95% confidence interval of median(cand)/median(base).
 *
 * @param {number[]} base Baseline samples (one per round).
 * @param {number[]} cand Candidate samples (one per round).
 * @param {{ iterations?: number, seed?: number }} [opts]
 * @returns {{ ratio: number, lo: number, hi: number }} Point estimate and CI bounds.
 */
export function ratioCI(base, cand, { iterations = 2000, seed = 1 } = {}) {
  const rand = mulberry32(seed);
  const resample = (xs) => xs.map(() => xs[Math.floor(rand() * xs.length)]);
  const ratios = [];
  for (let i = 0; i < iterations; i++) {
    const r = median(resample(cand)) / median(resample(base));
    if (Number.isFinite(r)) ratios.push(r);
  }
  ratios.sort((a, b) => a - b);
  const at = (q) => ratios[Math.min(ratios.length - 1, Math.floor(q * ratios.length))];
  return { ratio: median(cand) / median(base), lo: at(0.025), hi: at(0.975) };
}

/**
 * Noise floor of one metric from an A/A run: how far above 1 the CI of a
 * baseline-vs-baseline ratio reaches. Never below `minFloor`, so a lucky
 * quiet A/A run does not turn the gate into a coin toss.
 *
 * @param {number[]} baseA First copy of the baseline.
 * @param {number[]} baseB Second copy of the baseline.
 * @param {number} [minFloor=0.01] Smallest floor admitted (1%).
 * @returns {number} The floor as a fraction (0.02 = 2%).
 */
export function noiseFloor(baseA, baseB, minFloor = 0.01) {
  const { lo, hi } = ratioCI(baseA, baseB);
  return Math.max(minFloor, hi - 1, 1 - lo);
}

/**
 * Perf verdict for one metric.
 *
 * The test is on the POINT ratio of medians: `floor` is the band a
 * no-change comparison (the A/A control, same session, same rounds) spans,
 * so a candidate ratio outside it is a change the noise does not explain.
 * Testing the candidate CI's upper bound against a floor that is itself a CI
 * half-width double-counts the sampling noise and fails unchanged builds at
 * small round counts; the CI is still reported, for reading.
 *
 * @param {number[]} base Baseline samples.
 * @param {number[]} cand Candidate samples.
 * @param {number} floor Noise floor for this scene/metric from the A/A control.
 * @param {number} [absTolerance=0] A median difference within this is never judged.
 * @param {{ better?: 'lower'|'higher' }} [opts] Direction of improvement.
 * @returns {{ ratio: number, lo: number, hi: number, floor: number,
 *   verdict: 'pass'|'fail'|'win' }} For `better: 'lower'` (the default), `fail`
 *   above 1 + floor and `win` below 1 − floor; mirrored for `'higher'`.
 */
export function judgePerf(base, cand, floor, absTolerance = 0, { better = 'lower' } = {}) {
  const ci = ratioCI(base, cand);
  // A metric near the timer's resolution (a wake that blocks ~0 ms) has a
  // ratio of 0/0 or x/0: a difference within `absTolerance` of the baseline
  // is too small to judge either way.
  const delta = median(cand) - median(base);
  const higher = better === 'higher';
  // The band is symmetric in log space, [1/(1+floor), 1+floor]. A linear
  // 1 - floor would be unreachable once a noisy counter's floor passes 1.
  const up = ci.ratio > 1 + floor;
  const down = ci.ratio < 1 / (1 + floor);
  const worse = higher ? down : up;
  const improved = higher ? up : down;
  let verdict = 'pass';
  if (Math.abs(delta) <= absTolerance) verdict = 'pass';
  else if (worse) verdict = 'fail';
  else if (improved) verdict = 'win';
  return { ...ci, floor, verdict };
}

/**
 * True when any arm is empty or holds a missing sample (undefined, null or
 * NaN). A metric a build does not report (a counter an older baseline lacks)
 * is not comparable and must read `n/a`, never `pass`.
 *
 * @param {...Array<number|undefined|null>} arms Sample arrays.
 * @returns {boolean}
 */
export function anyMissing(...arms) {
  return arms.some(
    (a) =>
      !Array.isArray(a) || a.length === 0 || a.some((v) => typeof v !== 'number' || Number.isNaN(v))
  );
}

/**
 * Verdict for a COUNTER metric (bytes uploaded, renders, decodes, ...).
 *
 * Counters are usually deterministic: when both baseline arms agree on one
 * single value there is no noise to estimate a floor from, and any change is
 * real, so the medians are compared exactly (`|delta| <= tol` passes, a move
 * in the `better` direction wins, the other way fails). A counter that does
 * vary between baseline arms (a render count that depends on frame timing) is
 * judged like a timing, against its A/A noise floor.
 *
 * @param {number[]} base Baseline samples.
 * @param {number[]} cand Candidate samples.
 * @param {number[]} base2 Second baseline arm (the A/A control).
 * @param {{ better?: 'lower'|'higher', tol?: number }} [opts]
 * @returns {{ verdict: 'pass'|'fail'|'win'|'n/a', baseMedian?: number,
 *   candMedian?: number, ratio?: number, exact?: boolean, floor?: number,
 *   lo?: number, hi?: number }}
 */
export function judgeCounter(base, cand, base2, { better = 'lower', tol = 0, relTol = 0 } = {}) {
  if (anyMissing(base, cand, base2)) return { verdict: 'n/a' };
  const baseMedian = median(base);
  const candMedian = median(cand);
  // A timing-driven count (renders or decodes over a timed workload) jitters
  // by a frame or two between arms even when the build is identical, so a
  // relative tolerance scales with the count instead of a fixed number.
  tol = Math.max(tol, relTol * Math.abs(baseMedian));
  const first = base[0];
  if ([...base, ...base2].every((v) => v === first)) {
    const delta = candMedian - baseMedian;
    let verdict = 'pass';
    if (Math.abs(delta) > tol) {
      const improved = better === 'higher' ? delta > 0 : delta < 0;
      verdict = improved ? 'win' : 'fail';
    }
    return { verdict, baseMedian, candMedian, ratio: candMedian / baseMedian, exact: true };
  }
  return {
    baseMedian,
    candMedian,
    exact: false,
    ...judgePerf(base, cand, noiseFloor(base, base2), tol, { better }),
  };
}

/**
 * The A/B/A2 arm order per round, rotating so no arm always runs first (a
 * warm browser, a thermally throttled GPU) or last.
 */
export const ROTATIONS = [
  ['base', 'cand', 'base2'],
  ['cand', 'base2', 'base'],
  ['base2', 'base', 'cand'],
];
