import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  computeTolerance,
  executeSpatialQuery,
  formatTolerance,
  gsplatsContinuousDimTolerance,
  type DimensionInfo,
} from '../../../../../data/loaders';
import {
  GSPLAT_CHOLESKY_EPSILON,
  GSPLAT_DEFAULT_TRUNCATION_RADIUS,
} from '../../../../../config/constants';
import { project_gsplats_nd_to_3d } from '../../../../../wasm/typescript/gsplats-processing';

const DISPLAYED = 1e10;

/** `src/` — the Rust kernel source this file pins the config constant against. */
const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../..');

/**
 * Width of the band a hidden dim with an ALL-ZERO covariance block still renders
 * in, in that dim's own units — the quantity the gsplats epsilon's absolute term
 * has to cover (issue #1183 review).
 *
 * Spelled as a literal here on purpose: `gsplatsContinuousDimTolerance` derives
 * the same number from
 * `GSPLAT_DEFAULT_TRUNCATION_RADIUS × sqrt(GSPLAT_CHOLESKY_EPSILON)`, so asserting
 * against that expression would be circular. The identity below is what keeps the
 * literal from going stale.
 */
const REGULARIZED_BAND = 2.75e-5;

it('the regularization band literal still equals the constants the projection kernel uses', () => {
  // `computeMarginalCholesky` floors an all-zero hidden block's pivot at
  // sqrt(CHOLESKY_EPSILON) = 1e-5, and the shifted-Gaussian attenuation in
  // `project_gsplats_nd_to_3d` only reaches exactly 0 at Mahalanobis distance
  // `truncation_radius`. Band = T × σ_floor. If either constant moves, this fails
  // and REGULARIZED_BAND (plus the docs quoting 2.75e-5) must be updated.
  expect(GSPLAT_DEFAULT_TRUNCATION_RADIUS * Math.sqrt(GSPLAT_CHOLESKY_EPSILON)).toBeCloseTo(
    REGULARIZED_BAND,
    12
  );
});

it('GSPLAT_CHOLESKY_EPSILON mirrors the Rust twin, the backend that actually runs', () => {
  // The band above is only the band the RENDERER shows if the config-side value
  // equals the constant the backend actually uses. The production backend is the
  // Rust kernel, which owns an independent copy (separate language), so it is
  // pinned by parsing its declaration — importing is not an option, and this is
  // the only place the Rust value is covered at all. The TypeScript reference
  // kernel (>16D / WASM-missing) needs no parse: it IMPORTS
  // `GSPLAT_CHOLESKY_EPSILON` from `config/constants.ts`, so there is no second
  // literal on that side to desync.
  //
  // This single assertion fails in BOTH directions of drift: a changed Rust
  // literal, or a changed `GSPLAT_CHOLESKY_EPSILON` (which would also move the
  // renderer's band, and therefore the fetch epsilon, away from the Rust kernel).
  const rust = readFileSync(resolve(SRC, 'wasm/rust/src/common.rs'), 'utf8');
  const rustDecl = /const\s+CHOLESKY_EPSILON\s*:\s*f32\s*=\s*([0-9eE.+-]+)\s*;/.exec(rust);
  expect(
    rustDecl,
    'CHOLESKY_EPSILON declaration not found in wasm/rust/src/common.rs'
  ).not.toBeNull();
  expect(Number(rustDecl?.[1])).toBe(GSPLAT_CHOLESKY_EPSILON);
});

describe('computeTolerance — common rules', () => {
  it('sets infinite tolerance for displayed dimensions across all geometry types', () => {
    for (const type of ['points', 'lines', 'gsplats', 'mesh'] as const) {
      const tol = computeTolerance(type, [0, 1, 2], 4);
      expect(tol[0]).toBe(DISPLAYED);
      expect(tol[1]).toBe(DISPLAYED);
      expect(tol[2]).toBe(DISPLAYED);
      // Pin the actual sentinel constant (1e10), not just the local alias, so a
      // mutated DISPLAYED_TOLERANCE in the source is caught.
      expect(tol[0]).toBe(1e10);
      // Dimension 3 is hidden here: its tolerance must be far below the
      // displayed sentinel (kills a swap of displayed/hidden branches).
      expect(tol[3]).toBeLessThan(1e9);
    }
  });

  it('returns an array of the requested length', () => {
    expect(computeTolerance('points', [0, 1, 2], 5)).toHaveLength(5);
    expect(computeTolerance('lines', [], 7)).toHaveLength(7);
    expect(computeTolerance('gsplats', [0], 3)).toHaveLength(3);
  });
});

describe('computeTolerance — points', () => {
  it('uses maxRadius for hidden spatial dimensions', () => {
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, { maxRadius: 5.0 });
    expect(tol[3]).toBe(5.0);
  });

  it('falls back to default maxRadius (1.0) when not supplied', () => {
    const tol = computeTolerance('points', [0, 1, 2], 4);
    expect(tol[3]).toBe(1.0);
  });

  it('uses the shared quarter-cell tolerance for hidden dims flagged non-spatial via spatialExtendDims', () => {
    // Dimension 3 is non-spatial (discrete-like) per the per-dimension flag array.
    // No dimension metadata is supplied, so the step falls back to 1 → 0.25.
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true, false],
    });
    expect(tol[3]).toBe(0.25);
  });

  it('uses maxRadius for hidden dimensions flagged spatial via spatialExtendDims', () => {
    // Same shape, but dimension 3 is flagged spatial (true): it must use
    // maxRadius, not the discrete quarter-cell value. This pins the true branch of
    // the spatialExtendDims flag so a flipped flag-check is caught.
    const tol = computeTolerance('points', [0, 1, 2], 4, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true, true],
    });
    expect(tol[3]).toBe(5.0);
    expect(tol[3]).not.toBe(0.25);
  });

  it('treats dimensions beyond spatialExtendDims length as spatial', () => {
    const tol = computeTolerance('points', [0, 1, 2], 5, undefined, {
      maxRadius: 5.0,
      spatialExtendDims: [true, true, true],
    });
    expect(tol[3]).toBe(5.0);
    expect(tol[4]).toBe(5.0);
  });
});

describe('computeTolerance — lines', () => {
  const dims: DimensionInfo[] = [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete: true, step: 1.0 },
  ];

  it('returns 0 for hidden spatial dimensions (segment bounds already include line width)', () => {
    const tol = computeTolerance('lines', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.25); // discrete, 0.25 × step (step 1)
    const spatialOnly: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false },
    ];
    const tol2 = computeTolerance('lines', [0, 1, 2], 4, spatialOnly);
    expect(tol2[3]).toBe(0);
  });

  it('uses 0.25 × step for discrete hidden dimensions', () => {
    const tol = computeTolerance('lines', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.25);
  });

  it('scales the discrete tolerance with a larger step (step 2 → 0.5)', () => {
    const dimsStep2: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 2.0 },
    ];
    const tol = computeTolerance('lines', [0, 1, 2], 4, dimsStep2);
    expect(tol[3]).toBe(0.5); // 0.25 × 2
  });

  it('falls back to a quarter-cell (0.25) for discrete hidden dimensions without a step', () => {
    const dimsNoStep: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true },
    ];
    const tol = computeTolerance('lines', [0, 1, 2], 4, dimsNoStep);
    expect(tol[3]).toBe(0.25);
  });

  // Regression (deep-double-check): the lines PROJECTION/CLIPPING path is a
  // MEMBERSHIP gate, not a fetch reach. When the quarter-cell query rule
  // landed (ee0971f3) it silently halved the lines visibility slab from
  // step/2 to step/4 — cross-category whiskers shrank and off-grid vertices
  // in the (0.25, 0.5]×step band vanished while identical points/gsplats
  // stayed visible. `discreteRole: 'membership'` restores the half-cell gate.
  describe('membership role (projection clipping slab)', () => {
    it('uses the half-cell (0.5 × step) for discrete hidden dims under discreteRole=membership', () => {
      const tol = computeTolerance('lines', [0, 1, 2], 4, dims, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(0.5); // 0.5 × step (step 1) — NOT the 0.25 query reach
    });

    it('scales the membership slab with the step (step 2 → 1.0)', () => {
      const dimsStep2: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true, step: 2.0 },
      ];
      const tol = computeTolerance('lines', [0, 1, 2], 4, dimsStep2, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(1.0);
    });

    it('keeps spatial dims at 0 and displayed dims infinite under the membership role', () => {
      const tol = computeTolerance('lines', [0, 1, 2], 4, dims, {
        discreteRole: 'membership',
      });
      expect(tol[0]).toBe(1e10);
      expect(tol[3]).toBe(0.5);
      const spatialOnly: DimensionInfo[] = [{ discrete: false }, { discrete: false }];
      const tol2 = computeTolerance('lines', [0], 2, spatialOnly, {
        discreteRole: 'membership',
      });
      expect(tol2[1]).toBe(0);
    });

    it('falls back to a half-cell of a unit step (0.5) when the discrete dim has no step', () => {
      const dimsNoStep: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true },
      ];
      const tol = computeTolerance('lines', [0, 1, 2], 4, dimsNoStep, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(0.5);
    });

    it('three-geometry parity: lines membership slab equals the half-step gate points/gsplats use', () => {
      // Points membership: absolute 0.5 on the unit grid
      // (effective-radius-calculator.ts); gsplats projection: step × 0.5
      // (workers/data-worker/projection). Lines must match at step 1.
      const tol = computeTolerance('lines', [0, 1, 2], 4, dims, {
        discreteRole: 'membership',
      });
      expect(tol[3]).toBe(0.5);
    });
  });
});

describe('computeTolerance — gsplats', () => {
  // Issue #1183: the continuous arm used to be `step × 3.0`, documented as "3σ"
  // although it was a multiple of the navigation STEP and had nothing to do with
  // the splat covariance. Chunk bounds already carry the ellipsoidal
  // `truncation_radius · σ` expansion on hidden continuous dims
  // (io/_ordering/gsplats.py) and the shader discards past the same radius, so
  // the extra reach could only fetch chunks that render nothing. It is now a
  // float-safety epsilon.
  it('gives a hidden continuous dim a float-safety epsilon, NOT step × 3', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 0.5 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).not.toBe(0.5 * 3.0); // the old "3σ" reach
    // Far below a half-cell: can never bleed into the k±1 cell.
    expect(tol[3]).toBeLessThan(0.5 * 0.5);
    expect(tol[3]).toBeLessThan(0.01 * 0.5);
    // But strictly positive: a zero-variance continuous dim has float-EXACT
    // bounds, and a literal 0 would make membership an exact float comparison.
    expect(tol[3]).toBeGreaterThan(0);
    // The pinned rule at step 0.5: term 1 (1e-3 × step = 5e-4) dominates the
    // regularization band (2.75e-5), so a literal 5e-4 is the expected value.
    // Deliberately NOT `gsplatsContinuousDimTolerance(dims[3])` — calling the
    // implementation on both sides makes the assertion unfalsifiable.
    expect(tol[3]).toBe(5e-4);
    // The exported helper and the dispatcher must agree (a real claim: the
    // gsplats arm of computeHiddenDimTolerance could be wired to another rule).
    expect(gsplatsContinuousDimTolerance(dims[3])).toBe(5e-4);
  });

  const withStep = (step: number): number => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step },
    ];
    return computeTolerance('gsplats', [0, 1, 2], 4, dims)[3];
  };

  /** Every step this rule is pinned at, micro-steps included. */
  const STEPS = [1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1, 1e3, 1e6];

  it('covers the band a degenerate hidden dim still RENDERS in, at EVERY step', () => {
    // The read side does not use a raw zero variance: `computeMarginalCholesky`
    // regularizes an all-zero hidden block's pivot to sqrt(CHOLESKY_EPSILON),
    // so such a splat renders out to ~2.75e-5 in that dim's units — an ABSOLUTE
    // band, since that floor is an absolute variance backstop. A fetch window
    // narrower than it drops splats the kernel renders at up to ~60% of full
    // brightness (the attenuation 1 σ out). This ordering property is the whole
    // contract of the rule, so it is asserted at every scale rather than at two:
    // it also kills the quarter-cell cap tried in review, which gave 2.5e-7 at
    // step 1e-6.
    for (const step of STEPS) {
      expect(withStep(step)).toBeGreaterThanOrEqual(REGULARIZED_BAND);
    }
    expect(withStep(0.01)).toBeGreaterThan(1e-3 * 0.01); // the old rule's value
  });

  it('stays sub-half-cell where a cell is bigger than the render band, and multi-cell below', () => {
    // The trade-off, pinned rather than hidden. Above step ≈ 5.5e-5
    // (= band / 0.5) the epsilon is still well inside the half-cell, so nothing
    // like the discrete arm's neighbour-cell bleed is possible …
    for (const step of [1e6, 1e3, 1, 1e-2, 1e-3, 1e-4]) {
      expect(withStep(step)).toBeLessThan(0.5 * step);
      expect(withStep(step)).toBeGreaterThan(0);
    }
    // … and below it the epsilon is deliberately MANY cells wide, because the
    // rendered band does not shrink with the declared step. A continuous axis has
    // no categories to bleed into, so this is bandwidth, not correctness — and
    // capping it would hide renderable content (the review finding).
    expect(withStep(1e-6) / 1e-6).toBeCloseTo(27.5, 6);
    expect(withStep(1e-9) / 1e-9).toBeCloseTo(27500, 3);
  });

  it('the two regimes and their single crossover are intentional', () => {
    // Regime 1 (step >= 2.75e-2): the `_BARRIER_BOUND_EPS` mirror, 1e-3 × step.
    expect(withStep(1e6)).toBe(1e3);
    expect(withStep(1)).toBe(1e-3);
    // Regime 2 (step < 2.75e-2): the constant regularization band, at every
    // scale below the crossover — no third regime.
    for (const step of [1e-2, 1e-3, 1e-4, 1e-6, 1e-9]) {
      expect(withStep(step)).toBeCloseTo(REGULARIZED_BAND, 12);
    }
    // The crossover itself: 1e-3 × 2.75e-2 == the band exactly, and one step
    // either side selects the other term.
    expect(withStep(2.75e-2)).toBeCloseTo(REGULARIZED_BAND, 12);
    expect(withStep(2.8e-2)).toBeCloseTo(2.8e-5, 12);
    expect(withStep(2.7e-2)).toBeCloseTo(REGULARIZED_BAND, 12);
    // Monotone and non-decreasing across the crossover (no discontinuous dip
    // that would silently narrow the window at one particular scale).
    for (let i = 1; i < STEPS.length; i++) {
      expect(withStep(STEPS[i])).toBeGreaterThanOrEqual(withStep(STEPS[i - 1]));
    }
  });

  it('uses 0.25 × step for hidden discrete dimensions', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 1.0 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.25);
  });

  it('scales the discrete tolerance with a larger step (step 2 → 0.5)', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 2.0 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    expect(tol[3]).toBe(0.5); // 0.25 × 2
  });

  it('falls back to a unit-step epsilon (1e-3) when no step is available', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims);
    // 1e-3 × unit step — exactly the write side's `_BARRIER_BOUND_EPS`
    // (luxar/io/_ordering/bounds.py), whose reader-side mirror this is.
    expect(tol[3]).toBe(1e-3);
    expect(tol[3]).not.toBe(3.0); // the old fallback
  });
});

/**
 * The POINT of #1183, measured on the AABB scan the loaders actually run rather
 * than on the constant: the epsilon selects exactly the chunks whose
 * coverage-expanded bounds contain the slice position, a wider reach admits
 * chunks beyond it, and the splats in those chunks are measured — by calling the
 * projection kernel — to contribute literally nothing at that slice.
 *
 * The fixture is built so the scan result DISCRIMINATES: chunk 1's expanded bound
 * stops 0.2 short of the slice, which a `0.4 × step` reach would admit and the
 * epsilon must not. A test that only ever returned `[0]` for every candidate
 * tolerance (including a literal `0`) would be a revert-detector, not a measurement.
 */
describe('gsplats hidden-dim tolerance — AABB scan over coverage-expanded chunk bounds', () => {
  const NDIM = 4;
  const HIDDEN = 3;
  const STEP = 1.0;
  const TRUNCATION_RADIUS = 3.0; // node truncation_radius == coverage_sigma
  const SIGMA = 0.2; // per-splat σ along the hidden dim
  const EXPANSION = TRUNCATION_RADIUS * SIGMA; // 0.6 — what the write side adds
  const SLICE = 5.5;

  const dims: DimensionInfo[] = [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete: false, step: STEP },
  ];

  /**
   * Centres of four chunks along the hidden dim, each carrying the ±0.6
   * coverage expansion:
   *   0 → [4.4, 5.6]  contains the slice
   *   1 → [5.7, 6.9]  stops 0.2 SHORT — inside a 0.4×step reach, outside the epsilon
   *   2 → [6.4, 7.6]  outside a 0.4×step reach, inside the old step×3 reach
   *   3 → [9.4, 10.6] outside even the old reach
   */
  const centres = [5.0, 6.3, 7.0, 10.0];

  /**
   * Chunk bounds in the on-disk layout: `(numChunks, ndim, 2)` flattened, with
   * the hidden dim expanded by `truncation_radius · σ` exactly as
   * `compute_chunk_bounds_gsplats` does. Displayed dims are irrelevant here
   * (they get the 1e10 sentinel), so they share one wide box.
   */
  const chunkBounds = new Float32Array(centres.length * NDIM * 2);
  centres.forEach((centre, c) => {
    for (let d = 0; d < NDIM; d++) {
      const o = c * NDIM * 2 + d * 2;
      if (d === HIDDEN) {
        chunkBounds[o] = centre - EXPANSION;
        chunkBounds[o + 1] = centre + EXPANSION;
      } else {
        chunkBounds[o] = -100;
        chunkBounds[o + 1] = 100;
      }
    }
  });

  const scan = (tolerance: number[]): number[] =>
    executeSpatialQuery({
      chunkBounds,
      queryPosition: [0, 0, 0, SLICE],
      queryTolerance: tolerance,
      numChunks: centres.length,
      ndim: NDIM,
    });

  const hiddenTol = (t: number): number[] => [1e10, 1e10, 1e10, t];

  it('selects exactly the chunks whose coverage-expanded bounds contain the slice', () => {
    const tol = computeTolerance('gsplats', [0, 1, 2], NDIM, dims);
    const selected = scan(tol);
    expect(selected).toEqual([0]);
    // Cross-check against the bounds themselves — no hand-waving about which
    // chunks "should" match.
    const containing = centres
      .map((centre, c) => (Math.abs(SLICE - centre) <= EXPANSION ? c : -1))
      .filter((c) => c >= 0);
    expect(selected).toEqual(containing);
  });

  it('discriminates the epsilon from a 0.4 × step reach and from the old step × 3 reach', () => {
    // The whole claim of #1183 is that the tolerance must not ADD reach on top of
    // the coverage expansion. Chunk 1 exists to make that measurable: its bound
    // ends 0.2 above the slice, so any reach >= 0.2 pulls it in.
    const epsilon = computeTolerance('gsplats', [0, 1, 2], NDIM, dims);
    expect(scan(epsilon)).toEqual([0]);
    // A modest 0.4 × step reach over-fetches chunk 1 …
    expect(scan(hiddenTol(0.4 * STEP))).toEqual([0, 1]);
    // … and the pre-#1183 step × 3 reach over-fetches 1 AND 2.
    expect(scan(hiddenTol(3.0 * STEP))).toEqual([0, 1, 2]);
    // So the epsilon is strictly narrower than either (kills a revert and kills
    // "any small number would do").
    expect(epsilon[HIDDEN]).toBeLessThan(0.2);
  });

  it('a splat from a chunk the old step × 3 reach added contributes NOTHING at this slice', () => {
    // The measurement the change rests on, taken with the real kernel rather than
    // re-derived from the fixture's own inequality. `project_gsplats_nd_to_3d`
    // applies the shifted-Gaussian attenuation over the hidden dims and drops any
    // splat whose attenuated amplitude falls below `minAmplitude`.
    const PACKED = (NDIM * (NDIM + 1)) / 2;
    const diagIdx = (d: number): number => (d * (d + 1)) / 2 + d;

    /** Visible-splat count for one splat sitting at `hiddenPos` on the hidden dim. */
    const visibleCount = (hiddenPos: number): number => {
      const positions = new Float32Array(NDIM);
      positions[HIDDEN] = hiddenPos;
      const cholesky = new Float32Array(PACKED);
      for (let d = 0; d < NDIM; d++) cholesky[diagIdx(d)] = d === HIDDEN ? SIGMA : 1.0;
      const slicePosition = new Float32Array(NDIM);
      slicePosition[HIDDEN] = SLICE;
      return project_gsplats_nd_to_3d(
        positions,
        cholesky,
        new Float32Array([1.0]), // amplitude
        new Float32Array([1, 1, 1]), // white RGB
        new Uint8Array([1]), // discrete gate passes
        slicePosition,
        new Uint32Array([HIDDEN]), // continuous hidden dims
        new Uint32Array([0, 1, 2]), // display dims
        NDIM,
        1, // splatCount
        3, // colorComponents
        1e-6, // minAmplitude
        TRUNCATION_RADIUS,
        new Float32Array(3),
        new Float32Array(6),
        new Float32Array(1),
        new Float32Array(3),
        new Uint32Array(0)
      );
    };

    // Non-vacuity control: a splat AT the slice is emitted, so a zero below means
    // "attenuated away", not "the kernel call was mis-wired".
    expect(visibleCount(SLICE)).toBe(1);
    // A splat at the centre of chunk 1 or chunk 2 — the two the old step × 3
    // reach downloaded — renders nothing. |Δ|/σ is 4.0 and 7.5 sigmas, past the
    // truncation radius of 3. These two zeros are also what pins the SHIFT in the
    // shifted Gaussian: a plain `exp(-m²/2)` would still emit at m = 4 (3.4e-4,
    // above the 1e-6 minAmplitude gate). The CLAMP is not observable through this
    // API — a negative attenuation fails the same gate — so this measures the
    // shift only.
    expect(visibleCount(centres[1])).toBe(0);
    expect(visibleCount(centres[2])).toBe(0);
    // And the boundary is inside truncation, not at it: a splat 2 σ from the slice
    // (5.9 — one σ short of the truncation radius of 3) IS still visible, so the
    // support really does extend to T and the zeros above are attenuation rather
    // than an over-tight gate. Note this is NOT chunk 1's near expanded edge,
    // which sits at 5.7 (Mahalanobis 1.0 from the slice).
    expect(visibleCount(SLICE + 2.0 * SIGMA)).toBe(1);
  });

  it('zero-σ regression: a float-exact continuous bound still matches a start + k·step query', () => {
    // A stacked axis declared CONTINUOUS has zero variance along the hidden dim,
    // so it gets no σ expansion — and the write side epsilon-pads DISCRETE dims
    // only (`_BARRIER_BOUND_EPS`). Its bound is therefore the axis value itself,
    // stored as FLOAT32 while the query position is a float64: that storage
    // rounding is the dominant perturbation (≈1.9e-7 at 5.3). The `start + k ×
    // step` arithmetic drift in the query position is real but ~8 orders of
    // magnitude smaller (≈9e-16 here). Either way, this is why the epsilon is not
    // a literal 0.
    const zeroSigmaDims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 0.1 },
    ];
    const value = 5.3;
    const bounds = new Float32Array(NDIM * 2);
    for (let d = 0; d < NDIM; d++) {
      bounds[d * 2] = d === HIDDEN ? value : -100;
      bounds[d * 2 + 1] = d === HIDDEN ? value : 100;
    }
    const queryPos = 0.1 * 53;
    // The float32 storage gap dominates the float64 arithmetic drift.
    const storageGap = Math.abs(bounds[HIDDEN * 2] - value);
    const arithmeticDrift = Math.abs(queryPos - value);
    expect(storageGap).toBeGreaterThan(1e-8);
    expect(arithmeticDrift).toBeLessThan(1e-14);
    expect(storageGap).toBeGreaterThan(arithmeticDrift * 1e6);
    expect(queryPos).not.toBe(bounds[HIDDEN * 2]);

    const query = (queryTolerance: number[]): number[] =>
      executeSpatialQuery({
        chunkBounds: bounds,
        queryPosition: [0, 0, 0, queryPos],
        queryTolerance,
        numChunks: 1,
        ndim: NDIM,
      });

    // A literal zero tolerance drops the chunk entirely (the failure the epsilon
    // exists to prevent).
    expect(query(hiddenTol(0))).toEqual([]);
    // The epsilon selects it.
    expect(query(computeTolerance('gsplats', [0, 1, 2], NDIM, zeroSigmaDims))).toEqual([0]);
  });
});

describe('formatTolerance — the ?debug query log must not hide a sub-0.01 tolerance', () => {
  it('prints sub-0.01 values in exponential form, so an epsilon is not indistinguishable from 0', () => {
    // This log is the surface that would reveal an "empty node because nothing
    // matched" bug. At toFixed(2) the gsplats epsilon (1e-3 … 1e-5), Lines' true
    // 0, and a zeroed/NaN tolerance all printed `0.00`.
    expect(formatTolerance(1e-3)).not.toBe('0.00');
    expect(formatTolerance(2.75e-5)).not.toBe('0.00');
    expect(formatTolerance(1e-3)).not.toBe(formatTolerance(2.75e-5));
    expect(formatTolerance(1e-3)).toBe('1.0e-3');
    // A literal zero still reads as an exact zero, and NaN is not disguised.
    expect(formatTolerance(0)).toBe('0.00');
    expect(formatTolerance(NaN)).toBe('NaN');
    // The common case stays quiet: no exponents for ordinary tolerances.
    expect(formatTolerance(0.25)).toBe('0.25');
    expect(formatTolerance(1)).toBe('1.00');
    expect(formatTolerance(1e10)).toBe('∞');
  });
});

describe('computeTolerance — mesh', () => {
  const HIDDEN_DISCRETE: DimensionInfo[] = [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete: true, step: 1.0 },
  ];

  it('uses the HALF-cell membership rule for discrete dims, not the quarter-cell query reach', () => {
    // Mesh is whole-node resident, so this slab is a per-element visibility gate
    // applied after fetch — like the lines projection-clipping slab — not a
    // chunk-fetch reach. The quarter-cell default is deliberately < 0.5 x step and
    // would drop on-grid geometry.
    const tol = computeTolerance('mesh', [0, 1, 2], 4, HIDDEN_DISCRETE);
    expect(tol[3]).toBe(0.5);
    // Explicitly NOT the value the other three get by default.
    expect(tol[3]).not.toBe(computeTolerance('gsplats', [0, 1, 2], 4, HIDDEN_DISCRETE)[3]);
  });

  it('scales the half-cell with the step', () => {
    const dims = [...HIDDEN_DISCRETE.slice(0, 3), { discrete: true, step: 4.0 }];
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims)[3]).toBe(2.0);
  });

  it('ignores discreteRole — mesh has no query role to serve', () => {
    // The other three switch behaviour on this option. Mesh has no spatial index and
    // issues no range query, so membership is the only rule it has; honouring a
    // 'query' role here would quietly hand a fetch reach to the one caller that is
    // asking about visibility.
    for (const role of ['query', 'membership'] as const) {
      expect(
        computeTolerance('mesh', [0, 1, 2], 4, HIDDEN_DISCRETE, { discreteRole: role })[3]
      ).toBe(0.5);
    }
  });

  it('gives a hidden CONTINUOUS dim one cell by default — emphatically not Lines’ 0', () => {
    // THE trap this arm exists for. Lines can use 0 because segment clipping
    // interpolates through the slab; mesh culls whole triangles with no
    // interpolation, so 0 reduces membership to exact float equality with the slice
    // plane and the node renders NOTHING.
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 2.0 },
    ];
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims)[3]).toBe(2.0);
    expect(computeTolerance('lines', [0, 1, 2], 4, dims)[3]).toBe(0);
  });

  it('never returns 0 for a hidden spatial dim, whatever the dimension metadata', () => {
    // The invariant behind the trap above, asserted across the shapes a store can
    // actually present: absent metadata, no step, and a zero step.
    const shapes: (DimensionInfo[] | undefined)[] = [
      undefined,
      [{ discrete: false }, { discrete: false }, { discrete: false }, { discrete: false }],
      [{ discrete: false }, { discrete: false }, { discrete: false }, { discrete: false, step: 0 }],
    ];
    for (const dims of shapes) {
      expect(computeTolerance('mesh', [0, 1, 2], 4, dims)[3]).toBeGreaterThan(0);
    }
  });

  it('honours meshSlabTolerance for the continuous arm', () => {
    const dims: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 1.0 },
    ];
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims, { meshSlabTolerance: 3.0 })[3]).toBe(3.0);
  });
});
