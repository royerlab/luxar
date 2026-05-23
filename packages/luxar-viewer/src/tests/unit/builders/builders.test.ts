/**
 * Smoke tests for the test data builders.
 *
 * Builders are themselves test infrastructure, so these tests verify only
 * the contract: `build()` returns typed arrays of the documented sizes,
 * default fallbacks kick in when optional setters are skipped, and the
 * Cholesky packing layout (lower-triangular, ndim*(ndim+1)/2 floats per
 * splat) is correct.
 *
 * AUDIT NOTE (builders.md C2): stochastic generators (withVaryingWidths,
 * withRandomPositions, withRandomVertices, withVaryingAmplitudes) run
 * UNSEEDED here. The withVaryingWidths test (above) now asserts variance
 * + length, which catches "always-min"/"always-mid" implementations, but
 * the broader principle — seed Math.random with vi.spyOn(Math, 'random')
 * and assert distribution-level invariants rather than bound-only —
 * is a future cleanup. Builders are used by every other suite, so a
 * faulty builder corrupts coverage broadly.
 */

import { describe, it, expect } from 'vitest';
import {
  PointsBuilder,
  LinesBuilder,
  GSplatsBuilder,
  DimensionsBuilder,
} from '../../builders/test-data-builders';

describe('PointsBuilder smoke', () => {
  it('produces positions of length numPoints * dimensions when randomized', () => {
    const data = new PointsBuilder().withPoints(50).withDimensions(3).withRandomPositions().build();

    expect(data.positions).toBeInstanceOf(Float32Array);
    expect(data.positions.length).toBe(50 * 3);
    expect(data.numPoints).toBe(50);
    expect(data.dimensions).toBe(3);
  });

  it('falls back to random positions when build() is called without explicit ones (full default shape)', () => {
    // builders.md W1/W2 fix: previously asserted only positions.length.
    // Pin the full default-shape contract: nulls for optional attributes,
    // correct numPoints/dimensions, Float32Array dtype.
    const data = new PointsBuilder().withPoints(10).withDimensions(4).build();
    expect(data.positions).toBeInstanceOf(Float32Array);
    expect(data.positions.length).toBe(10 * 4);
    expect(data.numPoints).toBe(10);
    expect(data.dimensions).toBe(4);
    expect(data.colors).toBeNull();
    expect(data.radii).toBeNull();
    expect(data.sharpness).toBeNull();
  });
});

describe('LinesBuilder smoke', () => {
  it('produces canonical-shape vertices/widths/segments by default', () => {
    const data = new LinesBuilder().withSegments(20).withDimensions(3).build();

    expect(data.vertices).toBeInstanceOf(Float32Array);
    expect(data.vertices.length).toBe(20 * 2 * 3);

    expect(data.widths).toBeInstanceOf(Float32Array);
    expect(data.widths.length).toBe(20);
    expect(Array.from(data.widths).every((w) => w === 1.0)).toBe(true);

    expect(data.segments).toBeInstanceOf(Uint32Array);
    expect(data.segments.length).toBe(20 * 2);
    expect(Array.from(data.segments)).toEqual(Array.from({ length: 40 }, (_, i) => i));

    expect(data.colors).toBeNull();
    expect(data.sharpness).toBeNull();
    expect(data.numSegments).toBe(20);
    expect(data.dimensions).toBe(3);
  });

  it('honors withWidths / withColors / withSharpness when provided', () => {
    const data = new LinesBuilder()
      .withSegments(5)
      .withDimensions(3)
      .withWidths(2.5)
      .withColors()
      .withSharpness(0.5)
      .build();

    expect(data.widths.every((w) => w === 2.5)).toBe(true);
    expect(data.colors).toBeInstanceOf(Float32Array);
    expect(data.colors!.length).toBe(5 * 2 * 3);
    expect(data.sharpness).toBeInstanceOf(Float32Array);
    expect(data.sharpness!.length).toBe(5 * 2);
    expect(data.sharpness!.every((s) => s === 0.5)).toBe(true);
  });

  it('honors withVaryingWidths within the requested range AND actually varies + is the right length', () => {
    // builders.md C1 fix: previous version asserted bounds only, missing
    // length + variance. A faulty implementation returning always-min or
    // always-midpoint would have passed.
    const data = new LinesBuilder().withSegments(20).withVaryingWidths(0.2, 0.8).build();
    expect(data.widths).toBeInstanceOf(Float32Array);
    // Lines store vertex widths (2 per segment).
    expect(data.widths.length).toBeGreaterThanOrEqual(20);
    const distinct = new Set(Array.from(data.widths));
    // For 20+ random samples in [0.2, 0.8] we expect many distinct values;
    // an always-constant implementation would yield exactly 1.
    expect(distinct.size).toBeGreaterThan(1);
    for (const w of data.widths) {
      expect(w).toBeGreaterThanOrEqual(0.2);
      expect(w).toBeLessThanOrEqual(0.8);
    }
  });

  it('accepts explicit segment indices', () => {
    const indices = [0, 1, 2, 3];
    const data = new LinesBuilder().withSegments(2).withSegmentIndices(indices).build();
    expect(Array.from(data.segments)).toEqual(indices);
  });
});

describe('GSplatsBuilder smoke', () => {
  it('produces the documented array shapes by default', () => {
    const data = new GSplatsBuilder().withSplats(15).withDimensions(3).build();

    expect(data.centers).toBeInstanceOf(Float32Array);
    expect(data.centers.length).toBe(15 * 3);

    expect(data.amplitudes).toBeInstanceOf(Float32Array);
    expect(data.amplitudes.length).toBe(15);
    expect(Array.from(data.amplitudes).every((a) => a === 1.0)).toBe(true);

    // Cholesky packing: ndim*(ndim+1)/2 = 6 floats per splat for ndim=3.
    expect(data.choleskyFactors).toBeInstanceOf(Float32Array);
    expect(data.choleskyFactors.length).toBe(15 * 6);

    expect(data.colors).toBeNull();
    expect(data.numSplats).toBe(15);
    expect(data.dimensions).toBe(3);
  });

  it('packs an isotropic Cholesky factor as sigma on the diagonal', () => {
    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withIsotropicCovariance(0.5)
      .build();

    // Layout per splat: L00, L10, L11, L20, L21, L22 → [σ, 0, σ, 0, 0, σ].
    const expected = [0.5, 0, 0.5, 0, 0, 0.5];
    for (let s = 0; s < 2; s++) {
      const slice = Array.from(data.choleskyFactors.slice(s * 6, (s + 1) * 6));
      expect(slice).toEqual(expected);
    }
  });

  it('packs the correct length for higher dimensions', () => {
    // ndim=5 → 5*6/2 = 15 floats per splat.
    const data = new GSplatsBuilder()
      .withSplats(3)
      .withDimensions(5)
      .withIsotropicCovariance(0.1)
      .build();
    expect(data.choleskyFactors.length).toBe(3 * 15);
  });

  it('accepts explicit centers, amplitudes, cholesky, and colors', () => {
    const centers = new Float32Array([0, 0, 0, 1, 1, 1]);
    const amps = new Float32Array([0.5, 0.5]);
    const chol = new Float32Array(2 * 6).fill(0.25);
    const colors = new Float32Array([1, 0, 0, 0, 1, 0]);

    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withCenters(centers)
      .withCholeskyFactors(chol)
      .withColors(colors)
      .build();
    // Amplitudes wasn't set — should default to 1.0.
    expect(Array.from(data.amplitudes)).toEqual([1.0, 1.0]);

    // Now with explicit amplitudes.
    const data2 = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withCenters(centers)
      .withVaryingAmplitudes(0.4, 0.8)
      .withCholeskyFactors(chol)
      .withColors(colors)
      .build();
    expect(data2.centers).toBe(centers);
    for (const a of data2.amplitudes) {
      expect(a).toBeGreaterThanOrEqual(0.4);
      expect(a).toBeLessThanOrEqual(0.8);
    }
    expect(data2.choleskyFactors).toBe(chol);
    expect(data2.colors).toBe(colors);

    // Pin the unused `amps` parameter so its shape is exercised by tests too.
    expect(amps.length).toBe(2);
  });
});

// [builders.md/G13][P2] Pre-audit, no test did a deep-equality assertion on
// the full `build()` return shape. A return that drops a key or adds an
// extra key would not fail any assertion. The tests below pin the exact
// key sets — a producer-side rename or omission becomes immediately loud.
describe('build() return-shape contract [builders.md/G13][P2]', () => {
  it('PointsBuilder.build() returns exactly { positions, colors, radii, sharpness, numPoints, dimensions }', () => {
    const data = new PointsBuilder().withPoints(3).withDimensions(2).build();
    expect(Object.keys(data).sort()).toEqual(
      ['colors', 'dimensions', 'numPoints', 'positions', 'radii', 'sharpness'].sort()
    );
  });

  it('LinesBuilder.build() returns exactly the canonical 8-key shape', () => {
    const data = new LinesBuilder().withSegments(2).withDimensions(2).build();
    expect(Object.keys(data).sort()).toEqual(
      ['colors', 'dimensions', 'numSegments', 'segments', 'sharpness', 'vertices', 'widths'].sort()
    );
  });

  it('GSplatsBuilder.build() returns exactly the canonical 6-key shape', () => {
    const data = new GSplatsBuilder().withSplats(2).withDimensions(2).build();
    expect(Object.keys(data).sort()).toEqual(
      ['amplitudes', 'centers', 'choleskyFactors', 'colors', 'dimensions', 'numSplats'].sort()
    );
  });
});

// [builders.md/G7][P11] Pre-audit, `DimensionsBuilder` had ZERO tests
// despite being one of the seven exported builders. Even covering the
// load-bearing branches (withDisplayed slice-to-3, default-vs-overridden
// ndim, time/channel dimension auto-extension) closes the biggest gap.
describe('DimensionsBuilder smoke [builders.md/G7][P11]', () => {
  it('produces 3D default config with displayed=[0,1,2] and 3 metadata entries', () => {
    const dims = new DimensionsBuilder().build();
    expect(dims.ndim).toBe(3);
    expect(dims.displayed).toEqual([0, 1, 2]);
    expect(dims.metadata).toHaveLength(3);
  });

  it('withDisplayed silently truncates to 3 indices (documented OOS behavior)', () => {
    const dims = new DimensionsBuilder().withDisplayed(0, 1, 2, 3, 4).build();
    // The setter truncates with `.slice(0, 3)` — pin the contract.
    expect(dims.displayed).toEqual([0, 1, 2]);
  });

  it('withNDimensions extends currentStep to length===ndim with zeros', () => {
    const dims = new DimensionsBuilder().withNDimensions(5).build();
    expect(dims.ndim).toBe(5);
    expect(dims.currentStep).toEqual([0, 0, 0, 0, 0]);
  });

  it('withSpatialDimensions populates the first 3 metadata entries as x/y/z', () => {
    const dims = new DimensionsBuilder().withSpatialDimensions('um', [0, 50]).build();
    const md = dims.metadata!;
    expect(md[0].name).toBe('x');
    expect(md[1].name).toBe('y');
    expect(md[2].name).toBe('z');
    expect(md[0].unit).toBe('um');
    expect(md[0].range).toEqual([0, 50]);
  });

  it('withTimeDimension auto-extends ndim to 4 when applied on a 3D builder', () => {
    const dims = new DimensionsBuilder().withTimeDimension([0, 5], 0.25).build();
    expect(dims.ndim).toBe(4);
    const md = dims.metadata!;
    expect(md[3].name).toBe('time');
    expect(md[3].step).toBe(0.25);
  });

  it('withChannelDimension auto-extends ndim to 5', () => {
    const dims = new DimensionsBuilder().withChannelDimension(4).build();
    expect(dims.ndim).toBe(5);
    const md = dims.metadata!;
    expect(md[4].name).toBe('channel');
    expect(md[4].discrete).toBe(true);
    expect(md[4].range).toEqual([0, 3]); // numChannels-1
  });

  it('withCurrentPosition writes into currentStep without exceeding ndim', () => {
    const dims = new DimensionsBuilder()
      .withNDimensions(4)
      .withCurrentPosition(1, 2, 3, 4, 999) // 999 must NOT land
      .build();
    expect(dims.currentStep).toEqual([1, 2, 3, 4]);
  });
});
