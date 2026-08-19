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
 * Issue #1655 item 2: the writer PUBLISHES the barrier set it used
 * (`slice_dims`, stamped by `io/_compiler/gsplat_assembly.py` whenever
 * `ordering != 'none'`) and the reader used to ignore it, re-deriving barrier-ness
 * from `dimensions[d].discrete`. The two agree only while the scene declares its
 * dimensions; with `scene_dimensions` absent the writer value-detects instead, and
 * a dim the two sides classify DIFFERENTLY gets the wrong rule — barrier-tight
 * bounds queried with a ~1e-3 continuous epsilon, or σ-expanded bounds queried with
 * a quarter-cell.
 *
 * The two rules are ~250× apart at step 1 (0.25 vs 1e-3), so every assertion below
 * discriminates them by a wide margin rather than by a rounding.
 */
describe('computeTolerance — barrierDims overrides the discrete flag (issue #1655)', () => {
  const QUARTER_CELL = 0.25; // barrier rule at step 1
  const CONTINUOUS_EPS = 1e-3; // gsplats continuous rule at step 1

  /** dim 3 hidden; `discrete` is whatever the caller says, step always 1. */
  const dims = (discrete: boolean): DimensionInfo[] => [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete, step: 1.0 },
  ];

  it('promotes a discrete:false dim to the BARRIER rule when the writer listed it', () => {
    // The dangerous direction: the writer gave dim 3 a tight `_BARRIER_BOUND_EPS`
    // pad (no σ expansion), so querying it with the continuous epsilon can drop a
    // σ-extended splat near a chunk edge. Honouring `slice_dims` widens to the
    // quarter-cell instead.
    const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims(false), { barrierDims: [3] });
    expect(tol[3]).toBe(QUARTER_CELL);
    // Without the option it is still the old (narrow) answer — so this test is
    // measuring the override, not a constant.
    expect(computeTolerance('gsplats', [0, 1, 2], 4, dims(false))[3]).toBe(CONTINUOUS_EPS);
  });

  /**
   * The DEMOTE direction, and the window it must carry.
   *
   * A demoted dim (omitted from `barrierDims`, but `discrete: true` in the scene)
   * takes the continuous arm's BOUNDS reasoning — the writer σ-expanded it — but its
   * fetch window is the HALF-cell, because the read side's per-splat gate for that dim
   * is unchanged and still keyed on the SCENE flag: `data-processor-gsplats.ts` fills
   * `discreteDims` from `viewState.dimensions[d].discrete`, `classifyHiddenDims` then
   * puts the dim in `discreteHiddenDims` and OUT of `continuousHiddenDims`, and the
   * projection's only test for it is `|slicePos − center| ≤ step × 0.5`. The fetch
   * window therefore has to equal that gate: the Gaussian band never applies here, and
   * the quarter-cell — whose `< 0.5` rationale is about the ±0.5 pad LEGACY stores put
   * on barrier dims, which a demoted dim by definition never got — only half-covers it.
   */
  describe('the demote direction gets the half-cell membership window', () => {
    const HALF_CELL = 0.5; // membership rule at step 1

    it('a discrete:true dim omitted from barrierDims gets the half-cell, not the quarter', () => {
      const tol = computeTolerance('gsplats', [0, 1, 2], 4, dims(true), { barrierDims: [] });
      expect(tol[3]).toBe(HALF_CELL);
      expect(tol[3]).not.toBe(CONTINUOUS_EPS);
      expect(tol[3]).not.toBe(QUARTER_CELL); // the half-fix this replaced
      // Strictly wider than what the same dim got before this plumbing existed, so
      // nothing that used to be fetched stops being fetched.
      expect(tol[3]).toBeGreaterThan(computeTolerance('gsplats', [0, 1, 2], 4, dims(true))[3]);
    });

    it('the ACTUALLY reachable shape: a standalone open synthesizes discrete dims over slice_dims: []', () => {
      // Pins the branch to the path that really reaches it, because the obvious story
      // does not. GRAFTING a standalone store into a scene cannot demote anything:
      // `add_gsplats_from_file_impl` writes through the scene gsplats writer, which
      // re-sorts and RE-STAMPS `slice_dims` from `scene_barrier_dims`
      // (`discrete and not display`), so a scene-declared discrete axis comes back IN
      // the set and takes the barrier arm.
      //
      // What does reach it is serving the `.gsplats.zarr` DIRECTLY
      // (`?src=….gsplats.zarr`, which `gsplats/io/save_gsplats.py` enables by stamping
      // `layer` on the root). With no `scene_dimensions`,
      // `load-scene.ts::synthesizeSceneDimensionsFromNode` marks every axis >= 3
      // `discrete: true, step: 1` regardless of the stored values, while the store's own
      // `detect_barrier_dims` published `[]` for an off-grid stacked axis. That exact
      // pair — synthesized dims + an authoritative empty set — is the fixture here.
      const synthesized: DimensionInfo[] = [
        { discrete: false }, // X  (isSpatial: no discrete/step stamped)
        { discrete: false }, // Y
        { discrete: false }, // Z
        { discrete: true, step: 1 }, // dim3: synthesized as discrete, step 1
      ];
      const tol = computeTolerance('gsplats', [0, 1, 2], 4, synthesized, { barrierDims: [] });
      expect(tol[3]).toBe(HALF_CELL);
      for (const d of [0, 1, 2]) expect(tol[d]).toBe(1e10); // displayed
      // Not just "everything is 0.5": the synthesized SPATIAL axes carry no `discrete`
      // flag, so when one of them is the hidden axis it still takes the epsilon.
      const hiddenSpatial = computeTolerance('gsplats', [0, 3], 4, synthesized, {
        barrierDims: [],
      });
      expect(hiddenSpatial[1]).toBe(CONTINUOUS_EPS);
      expect(hiddenSpatial[2]).toBe(CONTINUOUS_EPS);
    });

    it('at |Δ| == 0.5 × step exactly, fetch and render agree — both admit it', () => {
      // The boundary the half-cell creates, pinned on the two real predicates rather
      // than argued. They use OPPOSITE comparison directions, so agreement at equality
      // is a fact about the pair, not a tautology:
      //   - `executeSpatialQuery` rejects only on STRICT inequality
      //     (`chunkMax < queryMin || chunkMin > queryMax`), so a bound exactly touching
      //     the window matches.
      //   - the projection's discrete gate discards on `> step * 0.5`
      //     (`workers/data-worker/projection/gsplats.ts`), so equality is VISIBLE.
      // A half-cell fetch window therefore covers the gate's own closed boundary; a
      // quarter-cell does not reach it at all.
      const STEP = 2.0; // non-unit, so a hardcoded 0.5 cannot pass by accident
      const SLICE = 4.0;
      const SPLAT = SLICE + 0.5 * STEP; // exactly on the gate's edge
      const stacked: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true, step: STEP },
      ];
      const tol = computeTolerance('gsplats', [0, 1, 2], 4, stacked, { barrierDims: [] });
      expect(tol[3]).toBe(0.5 * STEP);

      // Render side: the gate's own predicate, at equality.
      expect(Math.abs(SLICE - SPLAT) > STEP * 0.5).toBe(false);

      // Fetch side: the real AABB scan over the bare (unpadded, unexpanded) bound.
      const bounds = new Float32Array(4 * 2);
      for (let d = 0; d < 4; d++) {
        bounds[d * 2] = d === 3 ? SPLAT : -100;
        bounds[d * 2 + 1] = d === 3 ? SPLAT : 100;
      }
      const scan = (t: number[]): number[] =>
        executeSpatialQuery({
          chunkBounds: bounds,
          queryPosition: [0, 0, 0, SLICE],
          queryTolerance: t,
          numChunks: 1,
          ndim: 4,
        });
      expect(scan(tol)).toEqual([0]);
      // Negative control: the quarter-cell reach does not reach the gate's boundary.
      expect(scan([1e10, 1e10, 1e10, 0.25 * STEP])).toEqual([]);
      // And just OUTSIDE the gate the renderer drops the splat anyway, so the window
      // not reaching there is correct rather than a miss.
      const beyond = SLICE + 0.5 * STEP + 0.01;
      expect(Math.abs(SLICE - beyond) > STEP * 0.5).toBe(true);
    });

    it('regression: an off-grid stacked axis at step 1 must not fetch a 1e-3 window', () => {
      // `luxar gsplat merge --as-dimension --values 0.2,1.2,2.2` writes a STANDALONE
      // store: no `scene_dimensions`, so the writer falls back to the value-based
      // `io/_ordering/compound.py::detect_barrier_dims`, which rejects non-near-integer
      // values and omits the axis from `slice_dims`. Opened DIRECTLY
      // (`?src=….gsplats.zarr` — see the reachable-path case below), the viewer
      // synthesizes `discrete: true, step: 1` for that axis, so the slice snaps to 1.0
      // while the splats sit at 1.2. σ = 0 on a stacked axis, so the stored bound gets
      // neither a σ expansion nor a `_BARRIER_BOUND_EPS` pad — it is the bare
      // coordinate 1.2. The render gate passes (|1.0 − 1.2| = 0.2 ≤ 0.5) and a 1e-3
      // window matches no chunk, so every chunk that sits entirely inside one stacked
      // value drops out — most of the node, though not all of it (with
      // `slice_dims: []` the sort is a pure spatial curve over ALL columns, so a chunk
      // straddling two adjacent values still spans the query and matches).
      const SLICE = 1.0;
      const SPLAT = 1.2;
      const stacked: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true, step: 1.0 },
      ];
      const tol = computeTolerance('gsplats', [0, 1, 2], 4, stacked, { barrierDims: [] });
      // The render side shows it …
      expect(Math.abs(SLICE - SPLAT)).toBeLessThanOrEqual(0.5 * 1.0);
      // … so the fetch window has to reach it. A bare continuous epsilon does not.
      expect(tol[3]).toBeGreaterThanOrEqual(Math.abs(SLICE - SPLAT));
      expect(CONTINUOUS_EPS).toBeLessThan(Math.abs(SLICE - SPLAT));
      // Measured on the real AABB scan over the bare (unpadded, unexpanded) bound.
      const bounds = new Float32Array(4 * 2);
      for (let d = 0; d < 4; d++) {
        bounds[d * 2] = d === 3 ? SPLAT : -100;
        bounds[d * 2 + 1] = d === 3 ? SPLAT : 100;
      }
      const scan = (t: number[]): number[] =>
        executeSpatialQuery({
          chunkBounds: bounds,
          queryPosition: [0, 0, 0, SLICE],
          queryTolerance: t,
          numChunks: 1,
          ndim: 4,
        });
      expect(scan(tol)).toEqual([0]);
      expect(scan([1e10, 1e10, 1e10, CONTINUOUS_EPS])).toEqual([]);
    });

    it('regression: an offset-0.3 stacked axis needs the HALF cell — a quarter misses it', () => {
      // The same shape one notch further off-grid (`--values 0.3,1.3,2.3`). This is the
      // half of the bug the quarter-cell left unfixed — 0.3 is inside the renderer's
      // half-cell gate and outside a 0.25-cell fetch window — so the window must be the
      // membership half-cell, not a floor calibrated to the barrier arm's `< 0.5` pad
      // budget (a budget a dim the writer never barrier-padded does not have).
      const SLICE = 1.0;
      const SPLAT = 1.3;
      const stacked: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true, step: 1.0 },
      ];
      const tol = computeTolerance('gsplats', [0, 1, 2], 4, stacked, { barrierDims: [] });
      // The render side shows it: |1.0 − 1.3| = 0.3 ≤ 0.5 × step.
      expect(Math.abs(SLICE - SPLAT)).toBeLessThanOrEqual(0.5 * 1.0);
      // … and 0.3 is beyond BOTH prior answers, which is what makes this case new.
      expect(QUARTER_CELL).toBeLessThan(Math.abs(SLICE - SPLAT));
      expect(CONTINUOUS_EPS).toBeLessThan(Math.abs(SLICE - SPLAT));
      expect(tol[3]).toBeGreaterThanOrEqual(Math.abs(SLICE - SPLAT));

      // Measured on the real AABB scan over the bare (unpadded, unexpanded) bound.
      const bounds = new Float32Array(4 * 2);
      for (let d = 0; d < 4; d++) {
        bounds[d * 2] = d === 3 ? SPLAT : -100;
        bounds[d * 2 + 1] = d === 3 ? SPLAT : 100;
      }
      const scan = (t: number[]): number[] =>
        executeSpatialQuery({
          chunkBounds: bounds,
          queryPosition: [0, 0, 0, SLICE],
          queryTolerance: t,
          numChunks: 1,
          ndim: 4,
        });
      expect(scan(tol)).toEqual([0]);
      // Negative controls: the quarter-cell floor this replaced misses the chunk, and
      // so does the bare continuous epsilon.
      expect(scan([1e10, 1e10, 1e10, QUARTER_CELL])).toEqual([]);
      expect(scan([1e10, 1e10, 1e10, CONTINUOUS_EPS])).toEqual([]);
    });

    it('the half-cell REPLACES the continuous term, it is not a max with it', () => {
      // At a normal step the two candidate rules are 500× apart, so `toBe(0.5)` alone
      // cannot tell "half-cell" from "max(half-cell, epsilon)". The micro-step axis
      // does: there the continuous term (2.75e-5, an absolute degenerate BAND) is far
      // WIDER than the half cell (5e-7), and a `max` would return it. It must not —
      // that band is only ever drawn through the Gaussian attenuation, which this dim
      // is excluded from (`classifyHiddenDims` puts a scene-discrete dim in
      // `discreteHiddenDims`), so honouring it here would be over-fetch justified by a
      // rule the renderer does not apply.
      const micro: DimensionInfo[] = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: true, step: 1e-6 },
      ];
      expect(computeTolerance('gsplats', [0, 1, 2], 4, dims(true), { barrierDims: [] })[3]).toBe(
        0.5
      );
      expect(computeTolerance('gsplats', [0, 1, 2], 4, micro, { barrierDims: [] })[3]).toBe(0.5e-6);
      expect(computeTolerance('gsplats', [0, 1, 2], 4, micro, { barrierDims: [] })[3]).toBeLessThan(
        REGULARIZED_BAND
      );
      // Still observably NOT `undefined`: without the published set the same dim takes
      // the barrier arm's quarter-cell.
      expect(computeTolerance('gsplats', [0, 1, 2], 4, micro)[3]).toBe(0.25e-6);
    });
  });

  it('an EMPTY barrierDims is authoritative, not "absent"', () => {
    // `[]` is a real answer from the writer ("I ordered purely spatially"), and
    // conflating it with `undefined` would silently restore the `discrete` guess: the
    // demoted dims take the half-cell membership window, the barrier arm the
    // quarter-cell. Asserted across all three hidden dims of a 4D node so a length
    // check masquerading as an emptiness check cannot pass.
    const allDiscrete: DimensionInfo[] = [
      { discrete: true, step: 1e-6 },
      { discrete: true, step: 1e-6 },
      { discrete: true, step: 1e-6 },
      { discrete: true, step: 1e-6 },
    ];
    const tol = computeTolerance('gsplats', [0], 4, allDiscrete, { barrierDims: [] });
    for (const d of [1, 2, 3]) expect(tol[d]).toBe(0.5e-6);
    expect(tol[0]).toBe(1e10); // displayed dim still wins
    // Without the option, the same dims take the barrier rule instead.
    const legacy = computeTolerance('gsplats', [0], 4, allDiscrete);
    for (const d of [1, 2, 3]) expect(legacy[d]).toBe(0.25e-6);
  });

  it('INVARIANT (holds before and after #1655): with no barrierDims, `discrete` alone decides', () => {
    // Not coverage of this change — a guard on the legacy path this change must leave
    // alone. A store that publishes no `slice_dims` (and every non-gsplats caller,
    // which passes no options at all) must keep classifying from the scene flag.
    for (const options of [undefined, {}, { maxRadius: 5 }]) {
      expect(computeTolerance('gsplats', [0, 1, 2], 4, dims(true), options)[3]).toBe(QUARTER_CELL);
      expect(computeTolerance('gsplats', [0, 1, 2], 4, dims(false), options)[3]).toBe(
        CONTINUOUS_EPS
      );
    }
  });

  it('INVARIANT: a TRUTHY non-boolean `discrete` classifies the same way the renderer does', () => {
    // `discrete` reaches here uncoerced from the store's `scene_dimensions` JSON
    // (`view-state-manager.ts::extractMetadata` copies the raw value), and the
    // renderer's own gate is a truthiness test —
    // `data-processor-gsplats.ts::buildGSplatsParams` fills `discreteDims` from
    // `if (viewState.dimensions[d]?.discrete)`. So a hand-authored `"discrete": 1`
    // gets the binary half-cell gate at render time, and a strict `=== true` here
    // would hand it the ~1e-3 continuous epsilon instead: a fetch window 250×
    // narrower than the gate the renderer applies, which is exactly the class of
    // divergence this module exists to prevent.
    const truthy = [1, 'true', 'yes'] as unknown as boolean[];
    for (const flag of truthy) {
      const d = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: flag, step: 1.0 },
      ];
      // No published set: the barrier arm, exactly as a literal `true` gets.
      expect(computeTolerance('gsplats', [0, 1, 2], 4, d)[3]).toBe(QUARTER_CELL);
      // Published set that OMITS it: the demote arm, exactly as a literal `true` gets.
      expect(computeTolerance('gsplats', [0, 1, 2], 4, d, { barrierDims: [] })[3]).toBe(0.5);
      // And the sibling geometries, whose arms read the same flag.
      expect(computeTolerance('lines', [0, 1, 2], 4, d)[3]).toBe(QUARTER_CELL);
      expect(computeTolerance('mesh', [0, 1, 2], 4, d)[3]).toBe(0.5);
    }
    // Falsy non-booleans stay continuous, so this is a truthiness rule and not
    // "anything present counts".
    for (const flag of [0, '', null] as unknown as boolean[]) {
      const d = [
        { discrete: false },
        { discrete: false },
        { discrete: false },
        { discrete: flag, step: 1.0 },
      ];
      expect(computeTolerance('gsplats', [0, 1, 2], 4, d)[3]).toBe(CONTINUOUS_EPS);
      expect(computeTolerance('lines', [0, 1, 2], 4, d)[3]).toBe(0);
    }
  });

  it('classifies each dim independently, not "any barrier ⇒ all barriers"', () => {
    // A realistic 5D fitted timelapse: dims 0-2 displayed, dim 3 a σ-expanded
    // spatial axis the scene happens to declare discrete, dim 4 the stacked
    // timepoint barrier. Dim 3 is demoted, so it takes the half-cell membership window
    // (0.5 at step 1) while dim 4 takes the barrier quarter-cell — two DIFFERENT
    // answers in one call, which is the point: dim 4 is classified separately.
    const d5: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: true, step: 1.0 },
      { discrete: false, step: 1.0 },
    ];
    const tol = computeTolerance('gsplats', [0, 1, 2], 5, d5, { barrierDims: [4] });
    expect(tol[3]).toBe(0.5);
    expect(tol[4]).toBe(QUARTER_CELL);
    // A truly continuous dim 3 (the scene agreeing with the writer) DOES take the
    // epsilon — so the assertion above is the demote rule, not a stuck constant.
    const d5cont: DimensionInfo[] = [...d5.slice(0, 3), { discrete: false, step: 1.0 }, d5[4]];
    const tol2 = computeTolerance('gsplats', [0, 1, 2], 5, d5cont, { barrierDims: [4] });
    expect(tol2[3]).toBe(CONTINUOUS_EPS);
    expect(tol2[4]).toBe(QUARTER_CELL);
  });

  it('the barrier rule still scales with the step under an override', () => {
    // The override picks the RULE; it must not replace the rule's own step scaling
    // with a hardcoded 0.25.
    const dimsStep4: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 4.0 },
    ];
    expect(computeTolerance('gsplats', [0, 1, 2], 4, dimsStep4, { barrierDims: [3] })[3]).toBe(1.0);
  });

  it('the seam is GSPLATS-ONLY: lines and mesh ignore barrierDims entirely', () => {
    // `isBarrierDim` reads the published set for gsplats and nowhere else, so passing
    // the option to another geometry is a documented no-op rather than a silent
    // behaviour change. That is deliberate, not an oversight: honouring it would
    // NARROW both other arms — lines' continuous arm is a literal `0` (while
    // `data-processor-lines.ts` keeps clipping against a half-cell slab), and mesh's
    // two arms are both MEMBERSHIP gates, so a promoted dim would take a full cell
    // down to a half and change what the user SEES. Wiring either up must add that
    // arm's floor first, which this pins as an explicit edit rather than a default.
    const linesDims = dims(false);
    expect(computeTolerance('lines', [0, 1, 2], 4, linesDims, { barrierDims: [3] })[3]).toBe(
      computeTolerance('lines', [0, 1, 2], 4, linesDims)[3]
    );
    expect(computeTolerance('lines', [0, 1, 2], 4, linesDims, { barrierDims: [3] })[3]).toBe(0);
    // Demote would have been just as invisible on the lines membership path.
    expect(
      computeTolerance('lines', [0, 1, 2], 4, dims(true), {
        barrierDims: [],
        discreteRole: 'membership',
      })[3]
    ).toBe(0.5);

    // Mesh: promote must NOT move the slab off its full cell, demote must NOT move it
    // off the half cell it gets from `discrete`.
    const meshDims = dims(false);
    expect(computeTolerance('mesh', [0, 1, 2], 4, meshDims, { barrierDims: [3] })[3]).toBe(1.0);
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims(true), { barrierDims: [] })[3]).toBe(0.5);
    expect(computeTolerance('mesh', [0, 1, 2], 4, dims(true))[3]).toBe(0.5);
  });

  it('INVARIANT (holds before and after #1655): POINTS classify from spatialExtendDims', () => {
    // Not coverage of this change — the invariant it must not break. Points never
    // read `discrete` (their live path does not even come through
    // `computeTolerance`), so wiring them to a second, differently-sourced
    // classifier would be an unrequested behaviour change.
    const pointDims = dims(false);
    const opts = { maxRadius: 5.0, spatialExtendDims: [true, true, true, true] };
    expect(computeTolerance('points', [0, 1, 2], 4, pointDims, opts)[3]).toBe(5.0);
    expect(
      computeTolerance('points', [0, 1, 2], 4, pointDims, { ...opts, barrierDims: [3] })[3]
    ).toBe(5.0);
  });
});

/**
 * Issue #1655 item 3: the degenerate-band term is `T × sqrt(CHOLESKY_EPSILON)` and
 * used to be pinned to the DEFAULT `T` because this function only ever saw a
 * `DimensionInfo`. A node may stamp a larger `truncation_radius`
 * (`clampTruncationRadius` bounds it only to the float32-representable range), and
 * the band it renders scales with that — so the window `(2.75e-5, T × 1e-5]` was
 * uncovered. The node's own radius now reaches the computer.
 */
describe('computeTolerance — gsplats truncationRadius scales the degenerate band', () => {
  /** Micro-step axis, so the BAND term dominates and the radius is observable. */
  const microDims: DimensionInfo[] = [
    { discrete: false },
    { discrete: false },
    { discrete: false },
    { discrete: false, step: 1e-6 },
  ];
  const band = (truncationRadius?: number): number =>
    computeTolerance('gsplats', [0, 1, 2], 4, microDims, { truncationRadius })[3];

  it('T = 6 gives a 6e-5 band where the default T gives 2.75e-5', () => {
    // Spelled as literals rather than re-derived from the implementation's own
    // expression: 6 × sqrt(1e-10) = 6e-5, 2.75 × sqrt(1e-10) = 2.75e-5.
    expect(band(6)).toBeCloseTo(6e-5, 12);
    expect(band()).toBeCloseTo(REGULARIZED_BAND, 12);
    expect(band(6)).toBeGreaterThan(band());
    // The uncovered window the issue names: (2.75e-5, 6e-5].
    expect(band()).toBeLessThan(6e-5);
  });

  it('scales linearly, pinned to LITERAL bands across a sweep', () => {
    // Literals, not `T * Math.sqrt(GSPLAT_CHOLESKY_EPSILON)`: re-deriving the
    // expectation from the implementation's own expression makes the assertion
    // unfalsifiable (the trap this file's own header comment warns about — the
    // constants are pinned to the kernels separately, at the top of this file).
    const EXPECTED: [number, number][] = [
      [0.5, 5e-6],
      [1, 1e-5],
      [2.75, 2.75e-5],
      [6, 6e-5],
      [20, 2e-4],
    ];
    for (const [T, expected] of EXPECTED) {
      expect(band(T)).toBeCloseTo(expected, 12);
    }
  });

  it('the exported helper and the dispatcher select the same rule', () => {
    // A real claim about wiring (the gsplats arm of `computeHiddenDimTolerance`
    // could call something else), kept separate from the value assertions above so
    // it cannot stand in for them.
    for (const T of [0.5, 2.75, 20, undefined]) {
      expect(gsplatsContinuousDimTolerance(microDims[3], T)).toBe(band(T));
    }
  });

  it('a hostile or broken radius falls back to the default instead of exploding', () => {
    // The failure to avoid is "the epsilon becomes fetch-the-entire-node" (or
    // collapses to nothing). `clampTruncationRadius` is the authoritative rule and
    // the loader applies it, but this arm keeps its own positive-finite backstop.
    //
    // The `0` / negative rows below are UNREACHABLE from the real caller, and
    // deliberately answer differently from the authoritative clamp: the loader routes
    // every attr through `clampTruncationRadius`, which floors those at
    // `MIN_TRUNCATION_RADIUS` (≈2.44e-4 — pinned from the loader side in
    // `tests/unit/data/gsplats/spatial-index-loader.test.ts`), so this branch only
    // ever sees them from a caller that skipped the clamp. Backstop, not a second
    // rule; see `ToleranceOptions.truncationRadius`.
    for (const hostile of [
      NaN,
      Infinity,
      -Infinity,
      1e308, // finite in float64, Infinity in float32
      1e30, // square overflows float32
      0,
      -1,
      -2.75,
    ]) {
      expect(band(hostile)).toBeCloseTo(REGULARIZED_BAND, 12);
    }
    // A genuinely tiny-but-valid radius is NOT rejected — it is a real authored
    // value the material clamps, not a hostile one. (Literal: 1e-3 × 1e-5.)
    expect(band(1e-3)).toBeCloseTo(1e-8, 15);
  });

  it('does not touch the step-fraction regime, the barrier arm, or the other geometries', () => {
    // Above the crossover term 1 dominates and the radius is irrelevant …
    const coarse: DimensionInfo[] = [
      { discrete: false },
      { discrete: false },
      { discrete: false },
      { discrete: false, step: 1.0 },
    ];
    expect(computeTolerance('gsplats', [0, 1, 2], 4, coarse, { truncationRadius: 6 })[3]).toBe(
      1e-3
    );
    // … a barrier dim keeps the quarter-cell whatever the radius …
    const barrier: DimensionInfo[] = [...coarse.slice(0, 3), { discrete: true, step: 1.0 }];
    expect(computeTolerance('gsplats', [0, 1, 2], 4, barrier, { truncationRadius: 6 })[3]).toBe(
      0.25
    );
    // … and lines / mesh ignore it entirely.
    expect(computeTolerance('lines', [0, 1, 2], 4, microDims, { truncationRadius: 6 })[3]).toBe(0);
    expect(computeTolerance('mesh', [0, 1, 2], 4, microDims, { truncationRadius: 6 })[3]).toBe(
      1e-6
    );
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
