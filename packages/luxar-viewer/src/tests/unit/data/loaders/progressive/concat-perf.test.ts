/**
 * Record-only micro-benchmark: progressive-ladder concat cost —
 * reference full rebuild (O(k·N) across the ladder) vs the arena
 * (amortized O(N)). The browser-level ladder bench is pacing-noisy
 * (streaming scheduling swings the load window several-fold between
 * runs), so the accept/reject signal for the arena lever comes from
 * here: same inputs, both implementations, deterministic.
 *
 * Runs under `pnpm test:perf` (vitest.perf.config.ts). Like the other
 * perf benches it only enforces a generous sanity floor — its job is
 * before/after comparison, not gating.
 */
import { describe, it, expect } from 'vitest';
import {
  concatenateGSplatsData,
  GSplatsLadderArena,
} from '../../../../../data/gsplats/gsplats-progressive-loader';
import type { LoadedGSplatsData } from '../../../../../types/gsplats';

/**
 * Ladder shapes: 'geometric' (doubling chunks — the stream-recipe shape;
 * prefix rebuilds sum to ~2N, the reference's BEST case) and 'equal'
 * (k equal levels — prefix rebuilds sum to N·(k+1)/2, the O(k·N) case).
 */
function makeLadder(
  totalSplats: number,
  levels: number,
  shape: 'geometric' | 'equal'
): LoadedGSplatsData[] {
  const parts: LoadedGSplatsData[] = [];
  let remaining = totalSplats;
  let size =
    shape === 'equal'
      ? Math.ceil(totalSplats / levels)
      : Math.max(1, Math.floor(totalSplats / (1 << levels)));
  for (let l = 0; l < levels && remaining > 0; l++) {
    const n = l === levels - 1 ? remaining : Math.min(size, remaining);
    remaining -= n;
    const positions = new Float32Array(n * 3);
    const amplitudes = new Float32Array(n);
    const cholesky = new Float32Array(n * 6);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = Math.sin(i * 0.37 + l);
      positions[i * 3 + 1] = Math.cos(i * 0.53 + l);
      positions[i * 3 + 2] = Math.sin(i * 0.71 - l);
      amplitudes[i] = 0.5 + 0.5 * Math.sin(i * 0.11);
      for (let c = 0; c < 6; c++) cholesky[i * 6 + c] = 0.1 + 0.01 * ((i + c) % 7);
      for (let c = 0; c < 3; c++) colors[i * 3 + c] = ((i + c) % 255) / 255;
    }
    parts.push({
      positions,
      amplitudes,
      choleskyFactors: cholesky,
      colors,
      splatCount: n,
      ndim: 3,
    } as unknown as LoadedGSplatsData);
    if (shape === 'geometric') size *= 2;
  }
  return parts;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

const TOTAL = 2_000_000;
const LEVELS = 8;
const RUNS = 5;

describe('progressive concat perf — reference rebuild vs arena (record-only)', () => {
  it.each(['geometric', 'equal'] as const)(
    `%s ladder of ${LEVELS} levels, ${TOTAL.toLocaleString()} splats total`,
    (shape) => {
      const ladder = makeLadder(TOTAL, LEVELS, shape);

      const rebuildTotals: number[] = [];
      const rebuildWorst: number[] = [];
      for (let r = 0; r < RUNS; r++) {
        let total = 0;
        let worst = 0;
        for (let k = 1; k <= ladder.length; k++) {
          const t0 = performance.now();
          concatenateGSplatsData(ladder.slice(0, k));
          const dt = performance.now() - t0;
          total += dt;
          worst = Math.max(worst, dt);
        }
        rebuildTotals.push(total);
        rebuildWorst.push(worst);
      }

      const arenaTotals: number[] = [];
      const arenaWorst: number[] = [];
      for (let r = 0; r < RUNS; r++) {
        const arena = new GSplatsLadderArena();
        let total = 0;
        let worst = 0;
        for (let k = 1; k <= ladder.length; k++) {
          const t0 = performance.now();
          arena.appendThrough(ladder.slice(0, k), k === ladder.length);
          arena.snapshot();
          const dt = performance.now() - t0;
          total += dt;
          worst = Math.max(worst, dt);
        }
        arenaTotals.push(total);
        arenaWorst.push(worst);
      }

      const refT = median(rebuildTotals);
      const areT = median(arenaTotals);
      const refW = median(rebuildWorst);
      const areW = median(arenaWorst);
      console.log(
        `concat ${shape} ladder (${LEVELS} levels, ${TOTAL.toLocaleString()} splats): ` +
          `reference total ${refT.toFixed(1)} ms (worst level ${refW.toFixed(1)} ms) | ` +
          `arena total ${areT.toFixed(1)} ms (worst level ${areW.toFixed(1)} ms) | ` +
          `speedup ${(refT / areT).toFixed(2)}x total, ${(refW / areW).toFixed(2)}x worst-level`
      );

      // Sanity floor only — the arena must never be dramatically slower.
      expect(areT).toBeLessThan(refT * 2);
    }
  );
});
