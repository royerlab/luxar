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
 * Standard normal quantile (inverse CDF), Acklam's rational approximation
 * (relative error below 1.2e-9 over (0, 1)).
 *
 * @param {number} p Probability in (0, 1).
 * @returns {number} z with Phi(z) = p.
 */
export function normalQuantile(p) {
  const a = [
    -39.69683028665376, 220.9460984245205, -275.928510446969, 138.357751867269, -30.66479806614716,
    2.506628277459239,
  ];
  const b = [
    -54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972,
    -13.28068155288572,
  ];
  const c = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
  if (p > 1 - lo) return -normalQuantile(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

/**
 * Widen one comparison's A/A floor for a run that makes `tests` of them.
 *
 * `noiseFloor` is a 95% band for ONE comparison, so a no-change run that
 * judges 150 metrics fails about one of them by chance, and any single fail
 * fails the run: measured on obsidian, identical builds failed 0.7% of
 * timing comparisons, which is most runs. The family-wise fix is Bonferroni:
 * keep the run's chance of a false fail at `alpha` by testing each metric at
 * `alpha / tests`. The floor is a log-symmetric band whose half-width is
 * 1.96 standard errors, so it is rescaled in log space by
 * `z(1 - alpha / (2 * tests)) / z(0.975)`: about x1.5 for 15 judgements and
 * x1.8 for 150. A single judgement (`tests <= 1`) keeps the floor as is.
 *
 * @param {number} floor One comparison's floor (a fraction, see `noiseFloor`).
 * @param {number} tests Judgements the run makes on the same footing.
 * @param {number} [alpha=0.05] Family-wise false-fail rate.
 * @returns {number} The widened floor (a fraction).
 */
export function familyFloor(floor, tests, alpha = 0.05) {
  if (!(tests > 1)) return floor;
  const k = normalQuantile(1 - alpha / (2 * tests)) / normalQuantile(1 - 0.05 / 2);
  return Math.exp(k * Math.log1p(floor)) - 1;
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

/**
 * Host load (1-minute load average per logical CPU) above which a report
 * warns that its timings shared the machine. The gate's own Chrome keeps two
 * or three threads runnable; half the cores busy is other work.
 */
export const HIGH_LOAD_PER_CPU = 0.5;

/**
 * Summarize the host load each arm ran under, from the `load1` every measured
 * sample records (`row.samples[arm][i].load1`). Rows without samples (exact
 * rows, error rows) are skipped.
 *
 * @param {Array<{ samples?: Record<string, Array<{ load1?: number }>> }>} rows
 * @param {number} cpus Logical CPUs of the host.
 * @returns {{ cpus: number, arms: Record<string, { median: number, max: number }>,
 *   medianPerCpu: number, maxPerCpu: number, high: boolean } | null} Null when
 *   no arm was sampled. `high` is judged on the median, so one busy minute
 *   does not flag a long run.
 */
export function loadSummary(rows, cpus) {
  const arms = {};
  let all = [];
  for (const arm of ['base', 'base2', 'cand']) {
    const xs = rows
      .flatMap((r) => (r.samples?.[arm] ?? []).map((x) => x?.load1))
      .filter((v) => Number.isFinite(v));
    if (xs.length === 0) continue;
    arms[arm] = { median: median(xs), max: Math.max(...xs) };
    all = all.concat(xs);
  }
  if (all.length === 0) return null;
  const n = Math.max(1, cpus);
  const medianPerCpu = median(all) / n;
  return {
    cpus,
    arms,
    medianPerCpu,
    maxPerCpu: Math.max(...all) / n,
    high: medianPerCpu > HIGH_LOAD_PER_CPU,
  };
}
