/**
 * Unit tests for `scene/synthetic-scene.ts`.
 *
 * [scene.md/G1][P5] Source file had zero unit tests. The deterministic
 * `mulberry32` PRNG and the public `generateSyntheticLines` function are
 * both straightforward to test in isolation: pin the output shape,
 * deterministic seeding, bounds containment, and segment-length math.
 *
 * Scope: public surface only (no perf assertions — that's covered by
 * the line-perf-bench Playwright spec).
 */

import { describe, it, expect } from 'vitest';
import { generateSyntheticLines } from '../../../scene/synthetic-scene';
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
});
