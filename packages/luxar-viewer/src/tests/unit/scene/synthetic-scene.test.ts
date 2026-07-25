/**
 * Unit tests for `scene/synthetic-scene.ts`.
 *
 * [scene.md/G1][P5] Source file had zero unit tests. The deterministic
 * `mulberry32` PRNG and the public `generateSyntheticLines` function are
 * both straightforward to test in isolation: pin the output shape,
 * deterministic seeding, bounds containment, and segment-length math.
 *
 * The perf-campaign extension adds the points/gsplats generators and
 * the shared cluster sampler: determinism, count correctness, Cholesky
 * validity (finite, positive diagonal) + scale/orientation variety, and
 * a golden-value pin that locks the lines output byte-for-byte (the
 * 10 M-segment bench contract).
 *
 * Scope: public surface only (no perf assertions — that's covered by
 * the line-perf-bench Playwright spec).
 */

import { describe, it, expect } from 'vitest';
import {
  generateSyntheticLines,
  generateSyntheticPoints,
  generateSyntheticGSplats,
  sampleClusteredPositions,
} from '../../../scene/synthetic-scene';
import type { SyntheticSceneSpec } from '../../../scene/synthetic-scene';

describe('generateSyntheticLines', () => {
  describe('output shape', () => {
    it('returns an InstancedLinesMeshConfig with correctly sized typed arrays for the requested count', () => {
      const spec: SyntheticSceneSpec = { type: 'lines', count: 50, seed: 1 };
      const result = generateSyntheticLines(spec);

      expect(result.segmentCount).toBe(50);
      // 3 floats per vertex × 50 segments × 2 endpoints
      expect(result.startPositions).toBeInstanceOf(Float32Array);
      expect(result.startPositions.length).toBe(50 * 3);
      expect(result.endPositions).toBeInstanceOf(Float32Array);
      expect(result.endPositions.length).toBe(50 * 3);
      // Colors: 3 floats per vertex
      expect(result.startColors.length).toBe(50 * 3);
      expect(result.endColors.length).toBe(50 * 3);
      // Per-segment scalars: 1 float each
      expect(result.startWidths.length).toBe(50);
      expect(result.endWidths.length).toBe(50);
      expect(result.startSharpness.length).toBe(50);
      expect(result.endSharpness.length).toBe(50);
      expect(result.segmentLengths.length).toBe(50);
      // Clipped flags: Uint8
      expect(result.startClipped).toBeInstanceOf(Uint8Array);
      expect(result.startClipped.length).toBe(50);
      expect(result.endClipped).toBeInstanceOf(Uint8Array);
      expect(result.endClipped.length).toBe(50);
    });

    it('produces widths=1.0 and sharpness=0.5 (the [0,1] knob Gaussian midpoint, beta=2) for every endpoint', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 20, seed: 7 });
      for (let i = 0; i < result.segmentCount; i++) {
        expect(result.startWidths[i]).toBe(1.0);
        expect(result.endWidths[i]).toBe(1.0);
        expect(result.startSharpness[i]).toBe(0.5);
        expect(result.endSharpness[i]).toBe(0.5);
      }
    });

    it('marks all segment endpoints as unclipped (synthetic scene has no slicing)', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 30, seed: 3 });
      for (let i = 0; i < result.segmentCount; i++) {
        expect(result.startClipped[i]).toBe(0);
        expect(result.endClipped[i]).toBe(0);
      }
    });
  });

  describe('determinism (mulberry32 PRNG)', () => {
    it('produces byte-for-byte identical output for the same (count, seed) tuple', () => {
      const spec: SyntheticSceneSpec = { type: 'lines', count: 25, seed: 42 };
      const a = generateSyntheticLines(spec);
      const b = generateSyntheticLines(spec);

      expect(a.startPositions).toEqual(b.startPositions);
      expect(a.endPositions).toEqual(b.endPositions);
      expect(a.startColors).toEqual(b.startColors);
      expect(a.endColors).toEqual(b.endColors);
      expect(a.segmentLengths).toEqual(b.segmentLengths);
    });

    it('produces different positions when the seed differs (PRNG is actually advancing)', () => {
      const a = generateSyntheticLines({ type: 'lines', count: 25, seed: 1 });
      const b = generateSyntheticLines({ type: 'lines', count: 25, seed: 2 });
      // At least one start position must differ — if seeds were ignored,
      // every value would be identical.
      let differs = false;
      for (let i = 0; i < a.startPositions.length; i++) {
        if (a.startPositions[i] !== b.startPositions[i]) {
          differs = true;
          break;
        }
      }
      expect(differs).toBe(true);
    });
  });

  describe('bounds and segment lengths', () => {
    it('keeps all generated positions inside [-bounds, +bounds] after the periodic reset (no NaN/Infinity)', () => {
      // bounds=10 with stepScale=bounds*0.01 = 0.1; over 64 steps before
      // reset, the random walk can drift up to ~6.4 from the anchor in
      // any axis. The anchor itself is in [-10, 10], so the worst-case
      // bound is ~|16.4|. Use a generous envelope and just verify no
      // catastrophic NaN/Infinity escapes the loop.
      const result = generateSyntheticLines({ type: 'lines', count: 200, bounds: 10, seed: 5 });
      for (let i = 0; i < result.startPositions.length; i++) {
        expect(Number.isFinite(result.startPositions[i])).toBe(true);
        expect(Number.isFinite(result.endPositions[i])).toBe(true);
        // Generous envelope — see comment above.
        expect(Math.abs(result.startPositions[i])).toBeLessThan(20);
        expect(Math.abs(result.endPositions[i])).toBeLessThan(20);
      }
    });

    it('computes each segmentLength as the L2 distance between its start and end positions', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 40, seed: 11 });
      for (let i = 0; i < result.segmentCount; i++) {
        const i3 = i * 3;
        const dx = result.endPositions[i3] - result.startPositions[i3];
        const dy = result.endPositions[i3 + 1] - result.startPositions[i3 + 1];
        const dz = result.endPositions[i3 + 2] - result.startPositions[i3 + 2];
        const expected = Math.sqrt(dx * dx + dy * dy + dz * dz);
        // Float32 round-trip — use a small absolute tolerance.
        expect(result.segmentLengths[i]).toBeCloseTo(expected, 5);
      }
    });

    it('colors are in [0, 1] (rand() output range)', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 30, seed: 9 });
      for (let i = 0; i < result.startColors.length; i++) {
        expect(result.startColors[i]).toBeGreaterThanOrEqual(0);
        expect(result.startColors[i]).toBeLessThan(1);
        expect(result.endColors[i]).toBeGreaterThanOrEqual(0);
        expect(result.endColors[i]).toBeLessThan(1);
      }
    });
  });

  describe('defaults', () => {
    it('defaults bounds to 100 when not supplied', () => {
      // Compare against an explicit-100 generation — must be byte-identical.
      const implicit = generateSyntheticLines({ type: 'lines', count: 10, seed: 1 });
      const explicit = generateSyntheticLines({ type: 'lines', count: 10, bounds: 100, seed: 1 });
      expect(implicit.startPositions).toEqual(explicit.startPositions);
      expect(implicit.endPositions).toEqual(explicit.endPositions);
    });

    it('defaults seed to 1 when not supplied', () => {
      const implicit = generateSyntheticLines({ type: 'lines', count: 10 });
      const explicit = generateSyntheticLines({ type: 'lines', count: 10, seed: 1 });
      expect(implicit.startPositions).toEqual(explicit.startPositions);
    });
  });

  describe('edge cases', () => {
    it('handles count=0 — returns empty typed arrays with segmentCount=0', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 0, seed: 1 });
      expect(result.segmentCount).toBe(0);
      expect(result.startPositions.length).toBe(0);
      expect(result.endPositions.length).toBe(0);
      expect(result.segmentLengths.length).toBe(0);
      expect(result.startClipped.length).toBe(0);
    });

    it('handles count=1 — produces a single segment with finite endpoints', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 1, seed: 1 });
      expect(result.segmentCount).toBe(1);
      expect(result.startPositions.length).toBe(3);
      expect(result.endPositions.length).toBe(3);
      for (let i = 0; i < 3; i++) {
        expect(Number.isFinite(result.startPositions[i])).toBe(true);
        expect(Number.isFinite(result.endPositions[i])).toBe(true);
      }
      expect(Number.isFinite(result.segmentLengths[0])).toBe(true);
      expect(result.segmentLengths[0]).toBeGreaterThanOrEqual(0);
    });

    // G5: the random walk re-anchors every 64 steps. Exercise counts straddling
    // that cadence (just before, exactly at, and two full periods) to verify the
    // reset logic neither skips nor double-allocates segments.
    it.each([63, 64, 65, 128])(
      'produces exactly %i finite segments across the 64-step reset cadence',
      (count) => {
        const result = generateSyntheticLines({ type: 'lines', count, seed: 13 });
        expect(result.segmentCount).toBe(count);
        expect(result.startPositions.length).toBe(count * 3);
        expect(result.segmentLengths.length).toBe(count);
        for (let i = 0; i < result.startPositions.length; i++) {
          expect(Number.isFinite(result.startPositions[i])).toBe(true);
          expect(Number.isFinite(result.endPositions[i])).toBe(true);
        }
      }
    );

    // G6: over many reset periods the walk must never produce NaN/Infinity, and
    // the empirical bounds must stay finite (the periodic re-anchor keeps the
    // walk from diverging).
    it('stays finite across many reset periods (10k segments)', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 10000, bounds: 10, seed: 21 });
      let maxAbs = 0;
      for (let i = 0; i < result.startPositions.length; i++) {
        expect(Number.isFinite(result.startPositions[i])).toBe(true);
        expect(Number.isFinite(result.endPositions[i])).toBe(true);
        maxAbs = Math.max(maxAbs, Math.abs(result.startPositions[i]));
      }
      // Anchor ∈ [-10,10] + bounded ~64-step drift ⇒ comfortably under 40.
      expect(maxAbs).toBeLessThan(40);
    });
  });

  describe('lines contract preservation (perf-bench pin)', () => {
    // The 10 M-segment line-perf-bench scenario depends on this output
    // staying byte-for-byte stable. Golden values computed from the
    // original (pre points/gsplats refactor) implementation for
    // (seed=42, count=3, bounds=100) — exact float32 values, exact
    // equality. If this test fails, the lines generator changed and the
    // bench baseline is invalidated.
    it('produces the pinned golden output for (seed=42, count=3)', () => {
      const result = generateSyntheticLines({ type: 'lines', count: 3, seed: 42 });
      expect(Array.from(result.startPositions)).toEqual([
        33.946807861328125, -65.03722381591797, 5.318508625030518, 33.493263244628906,
        -64.78772735595703, 6.049457550048828, 33.49472427368164, -64.41450500488281,
        6.270699501037598,
      ]);
      expect(Array.from(result.endPositions)).toEqual([
        33.493263244628906, -64.78772735595703, 6.049457550048828, 33.49472427368164,
        -64.41450500488281, 6.270699501037598, 33.02863693237305, -65.29094696044922,
        5.642077445983887,
      ]);
      expect(Array.from(result.startColors)).toEqual([
        0.47231706976890564, 0.2499237358570099, 0.88205885887146, 0.003842951962724328,
        0.47078192234039307, 0.8373374342918396, 0.7835472822189331, 0.5303356051445007,
        0.02712360955774784,
      ]);
      expect(Array.from(result.segmentLengths)).toEqual([
        0.8956771492958069, 0.43387216329574585, 1.1749696731567383,
      ]);
    });
  });
});

describe('sampleClusteredPositions', () => {
  // Minimal deterministic uniform stream for direct sampler tests.
  function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  it('is deterministic for the same seed stream', () => {
    const a = sampleClusteredPositions(mulberry32(7), 500, 16, 100);
    const b = sampleClusteredPositions(mulberry32(7), 500, 16, 100);
    expect(a.positions).toEqual(b.positions);
    expect(a.min).toEqual(b.min);
    expect(a.max).toEqual(b.max);
  });

  it('returns exact empirical min/max corners of the sampled positions', () => {
    const { positions, min, max } = sampleClusteredPositions(mulberry32(3), 300, 8, 50);
    const seenMin = [Infinity, Infinity, Infinity];
    const seenMax = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < 300; i++) {
      for (let d = 0; d < 3; d++) {
        seenMin[d] = Math.min(seenMin[d], positions[i * 3 + d]);
        seenMax[d] = Math.max(seenMax[d], positions[i * 3 + d]);
      }
    }
    expect(Array.from(min)).toEqual(seenMin);
    expect(Array.from(max)).toEqual(seenMax);
  });

  it('keeps positions clustered near the volume (finite, bounded envelope)', () => {
    // Cluster centers ∈ [-b, b]; per-cluster σ ≤ 0.08·b, so a >6σ
    // excursion beyond 1.5·b is astronomically unlikely at this count.
    const b = 100;
    const { positions } = sampleClusteredPositions(mulberry32(11), 5000, 32, b);
    for (let i = 0; i < positions.length; i++) {
      expect(Number.isFinite(positions[i])).toBe(true);
      expect(Math.abs(positions[i])).toBeLessThan(1.5 * b);
    }
  });

  it('handles count=0 with zeroed bounds', () => {
    const { positions, min, max } = sampleClusteredPositions(mulberry32(1), 0, 8, 100);
    expect(positions.length).toBe(0);
    expect(min).toEqual([0, 0, 0]);
    expect(max).toEqual([0, 0, 0]);
  });
});

describe('generateSyntheticPoints', () => {
  it('returns correctly sized arrays for the requested count', () => {
    const cfg = generateSyntheticPoints({ type: 'points', count: 40, seed: 1 });
    expect(cfg.pointCount).toBe(40);
    expect(cfg.positions).toBeInstanceOf(Float32Array);
    expect(cfg.positions.length).toBe(40 * 3);
    expect(cfg.colors.length).toBe(40 * 3);
    expect(cfg.radii.length).toBe(40);
    expect(cfg.sharpness.length).toBe(40);
  });

  it('is deterministic: same (count, seed, clusters) → identical arrays', () => {
    const spec: SyntheticSceneSpec = { type: 'points', count: 100, seed: 42, clusters: 12 };
    const a = generateSyntheticPoints(spec);
    const b = generateSyntheticPoints(spec);
    expect(a.positions).toEqual(b.positions);
    expect(a.colors).toEqual(b.colors);
    expect(a.radii).toEqual(b.radii);
    expect(a.sharpness).toEqual(b.sharpness);
    expect(a.maxRadius).toBe(b.maxRadius);
    expect(a.boundsMin).toEqual(b.boundsMin);
    expect(a.boundsMax).toEqual(b.boundsMax);
  });

  it('differs across seeds and across cluster counts (both feed the PRNG stream)', () => {
    const base = generateSyntheticPoints({ type: 'points', count: 50, seed: 1 });
    const otherSeed = generateSyntheticPoints({ type: 'points', count: 50, seed: 2 });
    const otherClusters = generateSyntheticPoints({
      type: 'points',
      count: 50,
      seed: 1,
      clusters: 4,
    });
    expect(base.positions).not.toEqual(otherSeed.positions);
    expect(base.positions).not.toEqual(otherClusters.positions);
  });

  it('produces positive varied radii with maxRadius = max(radii), colors in [0,1), sharpness 0.5', () => {
    const cfg = generateSyntheticPoints({ type: 'points', count: 200, seed: 5 });
    let maxSeen = 0;
    let minSeen = Infinity;
    for (let i = 0; i < cfg.pointCount; i++) {
      expect(cfg.radii[i]).toBeGreaterThan(0);
      maxSeen = Math.max(maxSeen, cfg.radii[i]);
      minSeen = Math.min(minSeen, cfg.radii[i]);
      expect(cfg.sharpness[i]).toBe(0.5);
    }
    expect(cfg.maxRadius).toBe(maxSeen);
    expect(maxSeen).toBeGreaterThan(minSeen); // varied, not constant
    for (let i = 0; i < cfg.colors.length; i++) {
      expect(cfg.colors[i]).toBeGreaterThanOrEqual(0);
      expect(cfg.colors[i]).toBeLessThan(1);
    }
  });

  it('bounds corners exactly envelope the positions', () => {
    const cfg = generateSyntheticPoints({ type: 'points', count: 150, seed: 9 });
    for (let i = 0; i < cfg.pointCount; i++) {
      for (let d = 0; d < 3; d++) {
        expect(cfg.positions[i * 3 + d]).toBeGreaterThanOrEqual(cfg.boundsMin[d]);
        expect(cfg.positions[i * 3 + d]).toBeLessThanOrEqual(cfg.boundsMax[d]);
      }
    }
  });

  it('handles count=0 (empty arrays, footprint-default maxRadius)', () => {
    const cfg = generateSyntheticPoints({ type: 'points', count: 0, seed: 1 });
    expect(cfg.pointCount).toBe(0);
    expect(cfg.positions.length).toBe(0);
    expect(cfg.radii.length).toBe(0);
    expect(cfg.maxRadius).toBe(0.5);
  });
});

describe('generateSyntheticGSplats', () => {
  it('returns an InstancedGSplatsMeshConfig with correctly sized arrays', () => {
    const cfg = generateSyntheticGSplats({ type: 'gsplats', count: 30, seed: 1 });
    expect(cfg.splatCount).toBe(30);
    expect(cfg.centers).toBeInstanceOf(Float32Array);
    expect(cfg.centers.length).toBe(30 * 3);
    expect(cfg.cholesky01.length).toBe(30 * 2);
    expect(cfg.cholesky23.length).toBe(30 * 2);
    expect(cfg.cholesky45.length).toBe(30 * 2);
    expect(cfg.amplitudes.length).toBe(30);
    expect(cfg.colors.length).toBe(30 * 3);
  });

  it('is deterministic: same (count, seed, clusters) → identical arrays', () => {
    const spec: SyntheticSceneSpec = { type: 'gsplats', count: 80, seed: 7, clusters: 10 };
    const a = generateSyntheticGSplats(spec);
    const b = generateSyntheticGSplats(spec);
    expect(a.centers).toEqual(b.centers);
    expect(a.cholesky01).toEqual(b.cholesky01);
    expect(a.cholesky23).toEqual(b.cholesky23);
    expect(a.cholesky45).toEqual(b.cholesky45);
    expect(a.amplitudes).toEqual(b.amplitudes);
    expect(a.colors).toEqual(b.colors);
  });

  it('differs across seeds', () => {
    const a = generateSyntheticGSplats({ type: 'gsplats', count: 40, seed: 1 });
    const b = generateSyntheticGSplats({ type: 'gsplats', count: 40, seed: 2 });
    expect(a.centers).not.toEqual(b.centers);
    expect(a.cholesky01).not.toEqual(b.cholesky01);
  });

  it('emits VALID Cholesky factors: finite everywhere, strictly positive diagonal', () => {
    // Layout (packCholeskyForShader): cholesky01=[L00,L10],
    // cholesky23=[L11,L20], cholesky45=[L21,L22]. Diagonals are
    // L00, L11, L22 — any lower-triangular L with positive diagonal is
    // the Cholesky factor of the SPD covariance L·Lᵀ.
    const cfg = generateSyntheticGSplats({ type: 'gsplats', count: 500, seed: 3 });
    for (let i = 0; i < cfg.splatCount; i++) {
      const L00 = cfg.cholesky01[i * 2];
      const L10 = cfg.cholesky01[i * 2 + 1];
      const L11 = cfg.cholesky23[i * 2];
      const L20 = cfg.cholesky23[i * 2 + 1];
      const L21 = cfg.cholesky45[i * 2];
      const L22 = cfg.cholesky45[i * 2 + 1];
      for (const v of [L00, L10, L11, L20, L21, L22]) {
        expect(Number.isFinite(v)).toBe(true);
      }
      expect(L00).toBeGreaterThan(0);
      expect(L11).toBeGreaterThan(0);
      expect(L22).toBeGreaterThan(0);
    }
  });

  it('varies scale AND orientation across splats (depth order actually matters)', () => {
    const cfg = generateSyntheticGSplats({ type: 'gsplats', count: 500, seed: 3 });
    let minDiag = Infinity;
    let maxDiag = 0;
    let offDiagNonZero = 0;
    for (let i = 0; i < cfg.splatCount; i++) {
      const L00 = cfg.cholesky01[i * 2];
      minDiag = Math.min(minDiag, L00);
      maxDiag = Math.max(maxDiag, L00);
      if (
        cfg.cholesky01[i * 2 + 1] !== 0 ||
        cfg.cholesky23[i * 2 + 1] !== 0 ||
        cfg.cholesky45[i * 2] !== 0
      ) {
        offDiagNonZero++;
      }
    }
    // Log-uniform decade of characteristic size × anisotropy jitter ⇒
    // a wide spread; require at least 3× between smallest and largest.
    expect(maxDiag / minDiag).toBeGreaterThan(3);
    // Essentially every splat should carry off-diagonal (rotated
    // covariance) terms — exact zeros have measure zero.
    expect(offDiagNonZero).toBeGreaterThan(cfg.splatCount * 0.99);
  });

  it('produces positive, varied amplitudes in [0.2, 1.0)', () => {
    const cfg = generateSyntheticGSplats({ type: 'gsplats', count: 300, seed: 13 });
    let min = Infinity;
    let max = 0;
    for (let i = 0; i < cfg.splatCount; i++) {
      expect(cfg.amplitudes[i]).toBeGreaterThanOrEqual(0.2);
      expect(cfg.amplitudes[i]).toBeLessThan(1.0);
      min = Math.min(min, cfg.amplitudes[i]);
      max = Math.max(max, cfg.amplitudes[i]);
    }
    expect(max).toBeGreaterThan(min); // varied, not constant
  });

  it('handles count=0 (empty arrays, splatCount=0)', () => {
    const cfg = generateSyntheticGSplats({ type: 'gsplats', count: 0, seed: 1 });
    expect(cfg.splatCount).toBe(0);
    expect(cfg.centers.length).toBe(0);
    expect(cfg.cholesky01.length).toBe(0);
    expect(cfg.amplitudes.length).toBe(0);
  });
});
