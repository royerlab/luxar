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

import { describe, it, expect, vi } from 'vitest';
import {
  PointsBuilder,
  LinesBuilder,
  GSplatsBuilder,
  DimensionsBuilder,
  ChunkBuilder,
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

  it('[builders.md C1] withVaryingWidths uses the EXACT formula min + r*(max-min) [Math.random spy]', () => {
    // builders.md C1[P6][P2]: prior test allowed mutations of the
    // multiplier (e.g. `min + r * min` instead of `min + r * (max-min)`)
    // to survive because the bounded-range check passes for any value
    // in [min, max]. By spying on Math.random and injecting deterministic
    // [0, 0.5, 1] samples we can assert exact outputs [min, mid, max].
    const samples = [0, 0.5, 1];
    let i = 0;
    const spy = vi.spyOn(Math, 'random').mockImplementation(() => samples[i++ % samples.length]);
    try {
      const data = new LinesBuilder().withSegments(3).withVaryingWidths(0.2, 0.8).build();
      // 3 segments → 3 width slots; with sample sequence [0, 0.5, 1] and
      // formula `min + r*(max-min)` we expect:
      //   r=0   → 0.2
      //   r=0.5 → 0.5
      //   r=1   → 0.8
      // A mutation `min + r*min` (instead of `min + r*(max-min)`) would
      // produce [0.2, 0.3, 0.4] — caught here.
      const expected = [0.2, 0.5, 0.8];
      for (let j = 0; j < expected.length; j++) {
        expect(data.widths[j]).toBeCloseTo(expected[j], 5);
      }
    } finally {
      spy.mockRestore();
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

  it('packs the correct length AND the diagonal positions for ndim=5 [builders.md C3]', () => {
    // builders.md C3[P5][P2]: prior test only verified the LENGTH of the
    // packed array at ndim=5 (15 floats). A mutation that swapped the
    // row/col packing order (e.g. column-major instead of row-major, or
    // upper-triangular instead of lower-triangular) would survive at
    // ndim=3 but slip through at ndim≥4 unless the diagonal positions
    // are explicitly verified.
    //
    // Lower-triangular row-major packing for ndim=5:
    //   indices  → 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
    //   (r,c)    → 00 10 11 20 21 22 30 31 32 33 40 41 42 43 44
    // Diagonals fall at indices 0, 2, 5, 9, 14 → these MUST equal σ
    // (0.1 here); every off-diagonal slot MUST be 0.
    const data = new GSplatsBuilder()
      .withSplats(3)
      .withDimensions(5)
      .withIsotropicCovariance(0.1)
      .build();
    expect(data.choleskyFactors.length).toBe(3 * 15);

    const DIAGONAL_INDICES = [0, 2, 5, 9, 14];
    for (let s = 0; s < 3; s++) {
      const splatSlice = data.choleskyFactors.slice(s * 15, (s + 1) * 15);
      for (let i = 0; i < 15; i++) {
        if (DIAGONAL_INDICES.includes(i)) {
          expect(splatSlice[i]).toBeCloseTo(0.1, 6);
        } else {
          expect(splatSlice[i]).toBe(0);
        }
      }
    }
  });

  // [builders.md/O2][P4] Split a single it() that bundled four independent
  // contracts (default amplitudes, explicit-centers identity, varying-amplitude
  // bounds, explicit-cholesky identity, explicit-colors identity) into focused
  // tests. Mutations now surface as a single failing test rather than an
  // aggregate.

  it('defaults amplitudes to 1.0 when only centers/cholesky/colors are set', () => {
    const centers = new Float32Array([0, 0, 0, 1, 1, 1]);
    const chol = new Float32Array(2 * 6).fill(0.25);
    const colors = new Float32Array([1, 0, 0, 0, 1, 0]);

    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withCenters(centers)
      .withCholeskyFactors(chol)
      .withColors(colors)
      .build();
    expect(Array.from(data.amplitudes)).toEqual([1.0, 1.0]);
  });

  // build() now returns CLONES of every typed-array field — see
  // tests/builders/test-data-builders.ts. The previous "preserves the
  // explicit X array reference" tests were locking in the aliasing bug
  // (one mutation to the result of build() would leak into every prior
  // build() output). Re-assert the safer contract: returned content
  // equals the input but is NOT the same instance.
  it('returns a fresh centers Float32Array equal to the input', () => {
    const centers = new Float32Array([0, 0, 0, 1, 1, 1]);
    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withCenters(centers)
      .build();
    expect(data.centers).not.toBe(centers);
    expect(data.centers).toStrictEqual(centers);
  });

  it('clamps every withVaryingAmplitudes value into [min, max]', () => {
    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withVaryingAmplitudes(0.4, 0.8)
      .build();
    for (const a of data.amplitudes) {
      expect(a).toBeGreaterThanOrEqual(0.4);
      expect(a).toBeLessThanOrEqual(0.8);
    }
  });

  it('returns a fresh choleskyFactors Float32Array equal to the input', () => {
    const chol = new Float32Array(2 * 6).fill(0.25);
    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withCholeskyFactors(chol)
      .build();
    expect(data.choleskyFactors).not.toBe(chol);
    expect(data.choleskyFactors).toStrictEqual(chol);
  });

  it('returns a fresh colors Float32Array equal to the input', () => {
    const colors = new Float32Array([1, 0, 0, 0, 1, 0]);
    const data = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withColors(colors)
      .build();
    expect(data.colors).not.toBe(colors);
    expect(data.colors).toStrictEqual(colors);
  });

  // Now pin the actual safety contract — mutating the build() return
  // does NOT propagate back into the builder's internal state, and a
  // second build() call returns yet-another fresh buffer.
  it('isolates build() outputs from builder state and from each other', () => {
    const builder = new GSplatsBuilder()
      .withSplats(2)
      .withDimensions(3)
      .withCenters(new Float32Array([0, 0, 0, 1, 1, 1]));
    const a = builder.build();
    const b = builder.build();
    expect(a.centers).not.toBe(b.centers);
    a.centers[0] = 999;
    expect(b.centers[0]).toBe(0);
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

  it('withDisplayed throws on > 3 indices instead of silent truncation', () => {
    expect(() => new DimensionsBuilder().withDisplayed(0, 1, 2, 3, 4)).toThrow(
      /at most 3 dimensions may be displayed/
    );
  });

  it('withDisplayed accepts 0 / 1 / 2 / 3 indices unchanged', () => {
    expect(new DimensionsBuilder().withDisplayed().build().displayed).toEqual([]);
    expect(new DimensionsBuilder().withDisplayed(2).build().displayed).toEqual([2]);
    expect(new DimensionsBuilder().withDisplayed(0, 1).build().displayed).toEqual([0, 1]);
    expect(new DimensionsBuilder().withDisplayed(0, 1, 2).build().displayed).toEqual([0, 1, 2]);
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

// [builders OOS3] ChunkBuilder.withRandomData previously used bare
// substring matching to pick the typed-array dtype:
//   dtype.includes('u1') || dtype.includes('uint8')
// which misclassified `'<u16'` and any future dtype string that
// happened to contain `u1` or `uint8` as a substring. Now uses an
// exact match against the static UINT8_DTYPES set.
describe('ChunkBuilder.withRandomData dtype detection', () => {
  it('routes the documented uint8 spellings to Uint8Array', () => {
    for (const dtype of ['u1', '|u1', '<u1', '>u1', 'uint8']) {
      const chunk = new ChunkBuilder()
        .withShape(4, 4)
        .withChunkShape(4, 4)
        .withDtype(dtype)
        .withRandomData()
        .build();
      expect(chunk.data, `dtype "${dtype}" → expected Uint8Array`).toBeInstanceOf(Uint8Array);
      expect(chunk.data.length).toBe(16);
    }
  });

  it('routes uint16 dtype `<u16` to Float32Array (NOT silently to Uint8Array)', () => {
    // Pre-fix: `'<u16'`.includes('u1') === true → wrongly routed to Uint8Array.
    // Post-fix: not in UINT8_DTYPES → falls through to Float32Array default.
    const chunk = new ChunkBuilder()
      .withShape(3)
      .withChunkShape(3)
      .withDtype('<u16')
      .withRandomData()
      .build();
    expect(chunk.data).toBeInstanceOf(Float32Array);
    expect(chunk.data).not.toBeInstanceOf(Uint8Array);
  });

  it('routes a hypothetical dtype whose name contains `uint8` to Float32Array', () => {
    // Pre-fix: `'complex_uint8_v2'`.includes('uint8') === true → Uint8Array.
    // Post-fix: exact-match set rejects it → Float32Array default.
    const chunk = new ChunkBuilder()
      .withShape(2)
      .withChunkShape(2)
      .withDtype('complex_uint8_v2')
      .withRandomData()
      .build();
    expect(chunk.data).toBeInstanceOf(Float32Array);
    expect(chunk.data).not.toBeInstanceOf(Uint8Array);
  });

  it('routes float dtypes to Float32Array (regression: must not regress with the new set check)', () => {
    for (const dtype of ['<f4', '<f8', 'float32', 'float64']) {
      const chunk = new ChunkBuilder()
        .withShape(5)
        .withChunkShape(5)
        .withDtype(dtype)
        .withRandomData()
        .build();
      expect(chunk.data, `dtype "${dtype}" → expected Float32Array`).toBeInstanceOf(Float32Array);
    }
  });

  it('UINT8_DTYPES is a frozen contract referenced by tests', () => {
    // Pin the exact set so a future caller can't silently grow the
    // accept list (e.g. adding 'uint16' would be a behavior change
    // that this test forces a deliberate update of).
    expect([...ChunkBuilder.UINT8_DTYPES].sort()).toEqual(
      ['<u1', '>u1', '|u1', 'u1', 'uint8'].sort()
    );
  });
});
