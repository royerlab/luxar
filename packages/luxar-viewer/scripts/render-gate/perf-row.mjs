import { pacingBound } from './pacing.mjs';
import { anyMissing, familyFloor, judgePerf, median, noiseFloor } from './perf-stats.mjs';

/**
 * Judge one performance metric while preserving unavailable and paced states.
 *
 * @param {number} [tests] Gating timing judgements across the run, for the
 *   Bonferroni-widened A/A floor (see `familyFloor`); defaults to a single
 *   comparison when the caller judges just this one metric.
 */
export function scorePerfMetric(name, backend, base, cand, base2, absTolerance = 0, tests = 1) {
  const medians = {
    baseMedian: anyMissing(base) ? undefined : median(base),
    candMedian: anyMissing(cand) ? undefined : median(cand),
  };
  const pacing =
    name === 'throughputMs'
      ? {
          pacingBound: {
            base: pacingBound(backend, medians.baseMedian),
            cand: pacingBound(backend, medians.candMedian),
          },
        }
      : {};
  if (anyMissing(base, cand, base2)) return { ...medians, ...pacing, verdict: 'n/a' };
  if (pacing.pacingBound?.base || pacing.pacingBound?.cand) {
    return { ...medians, ...pacing, verdict: 'pacing-bound' };
  }
  const aaFloor = noiseFloor(base, base2);
  return {
    ...medians,
    ...pacing,
    aaFloor,
    ...judgePerf(base, cand, familyFloor(aaFloor, tests), absTolerance),
  };
}

export function perfRowStatus(metrics) {
  return Object.values(metrics).some((metric) => metric.verdict === 'fail') ? 'fail' : 'pass';
}
