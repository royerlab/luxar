import { pacingBound } from './pacing.mjs';
import { anyMissing, judgePerf, median, noiseFloor } from './perf-stats.mjs';

/** Judge one performance metric while preserving unavailable and paced states. */
export function scorePerfMetric(name, backend, base, cand, base2, absTolerance = 0) {
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
  return { ...medians, ...pacing, ...judgePerf(base, cand, noiseFloor(base, base2), absTolerance) };
}
