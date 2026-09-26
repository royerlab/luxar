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
 * Perf verdict for one metric (lower is better).
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
 * @returns {{ ratio: number, lo: number, hi: number, floor: number,
 *   verdict: 'pass'|'fail'|'win' }} `fail` above 1 + floor, `win` below 1 − floor.
 */
export function judgePerf(base, cand, floor) {
  const ci = ratioCI(base, cand);
  let verdict = 'pass';
  if (ci.ratio > 1 + floor) verdict = 'fail';
  else if (ci.ratio < 1 - floor) verdict = 'win';
  return { ...ci, floor, verdict };
}
