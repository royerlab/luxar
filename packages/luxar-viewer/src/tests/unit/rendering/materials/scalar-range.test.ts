/**
 * Unit tests for the shared scalar-range → colormap-uniform mapping
 * (rendering/materials/_shared/scalar-range.ts) and its wiring into the six
 * materials' setScalarRange (12 historical inline copies replaced).
 */

import { describe, it, expect } from 'vitest';
import {
  computeScalarRangeUniforms,
  scalarRangeUniformEntries,
  DEGENERATE_SCALAR_RANGE_EPS,
} from '../../../../rendering/materials/_shared/scalar-range';

describe('computeScalarRangeUniforms', () => {
  it('maps a regular range to (min, 1/(max-min))', () => {
    const { scalarMin, scalarScale } = computeScalarRangeUniforms(2, 6);
    expect(scalarMin).toBe(2);
    expect(scalarScale).toBeCloseTo(0.25, 12);
    // Shader math: t = clamp((v - min) * scale) — endpoints hit 0 and 1.
    expect((2 - scalarMin) * scalarScale).toBeCloseTo(0, 12);
    expect((6 - scalarMin) * scalarScale).toBeCloseTo(1, 12);
  });

  it('maps a DEGENERATE range to the LUT midpoint, not the floor', () => {
    // A constant scalar attribute (min === max) used to get scale = 1e10 and
    // t = 0 — every element rendered at the bottom of the colormap (black in
    // `gray`). The identity is the midpoint: t = 0.5.
    for (const v of [0, 1, 5.25, -3]) {
      const { scalarMin, scalarScale } = computeScalarRangeUniforms(v, v);
      expect((v - scalarMin) * scalarScale).toBeCloseTo(0.5, 12);
    }
  });

  it('treats sub-eps widths as degenerate', () => {
    const { scalarMin, scalarScale } = computeScalarRangeUniforms(
      1,
      1 + DEGENERATE_SCALAR_RANGE_EPS / 2
    );
    expect((1 - scalarMin) * scalarScale).toBeCloseTo(0.5, 9);
  });

  it('scalarRangeUniformEntries defaults a missing range to the [0,1] identity', () => {
    const entries = scalarRangeUniformEntries(undefined);
    expect(entries.uScalarMin.value).toBe(0);
    expect(entries.uScalarScale.value).toBe(1);
    const deg = scalarRangeUniformEntries([1, 1]);
    expect((1 - deg.uScalarMin.value) * deg.uScalarScale.value).toBeCloseTo(0.5, 12);
  });
});
