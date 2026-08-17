/**
 * Canonical tolerance computer used by all geometry-specific spatial-index loaders.
 *
 * Each geometry type has slightly different requirements for hidden (non-displayed)
 * dimensions. Discrete dims share ONE rule across the three *query-path* geometries
 * ({@link discreteDimTolerance}); only the spatial/continuous branch differs:
 * - **Points**: `maxRadius` for spatial dims (selected via the `spatialExtendDims`
 *   option, the per-dimension flag array carried by `EffectiveRadiusConfig`).
 * - **Lines**: 0 for spatial dims (segment bounding boxes already include line
 *   width extent).
 * - **GSplats**: a float-safety EPSILON for continuous dims
 *   ({@link gsplatsContinuousDimTolerance}) — deliberately NOT a reach. Chunk
 *   bounds already carry the ellipsoidal `truncation_radius · σ` expansion on
 *   every continuous dim that the write side treats as spatial
 *   (`io/_ordering/gsplats.py::compute_chunk_bounds_gsplats`, with
 *   `coverage_sigma=truncation_radius` from `io/_compiler/gsplat_tree.py`), and
 *   the hidden-dim cutoff the projection kernel applies is that same radius
 *   (`wasm/typescript/gsplats-processing.ts::project_gsplats_nd_to_3d` — the
 *   shifted-Gaussian attenuation `(e^{-m²/2} − c)/(1 − c)`, clamped at 0 and
 *   therefore exactly 0 at `m = truncation_radius`; the shader's
 *   `uTruncate`/`uTruncateSq` discard is the DISPLAYED-dim counterpart and plays
 *   no part in hidden-dim membership). So a chunk that misses a zero-tolerance
 *   query holds only splats the projection would attenuate to nothing: widening
 *   the query cannot make a visible splat appear, it only fetches chunks that
 *   render nothing. That statement has a PRECONDITION — see the "conditional
 *   premise" note on {@link gsplatsContinuousDimTolerance} — and the epsilon
 *   itself exists for a second reason documented there.
 * - **Mesh**: `step × meshSlabTolerance` (default one cell) for continuous dims,
 *   and the half-cell MEMBERSHIP rule for discrete ones — the only type that does
 *   not use the quarter-cell query reach. See the mesh note below; it is the one
 *   arm where copying a neighbour silently renders nothing.
 *
 * Discrete dims (the query-path geometries): a "quarter-cell" `0.25 × step` for the QUERY
 * role (chunk-fetch reach). This is deliberately `< 0.5 × step`: a query on
 * category `k` must not reach the `k±1` cell even though chunk bounds are
 * padded on the write side (see `io/ordering.py` barrier padding). The two
 * half-steps (pad + tolerance) would otherwise sum to a full step and bleed
 * the entire neighbouring category (e.g. loading timepoint `t−1` in full when
 * scrubbing to `t`). A quarter-cell still comfortably catches the target cell
 * and genuine straddle chunks, and sits inside the half-cell (`0.5 × step`)
 * MEMBERSHIP gates: the points/gsplats projection rules, and the lines
 * projection-clipping slab, which requests `discreteRole: 'membership'` here
 * to get {@link discreteDimMembershipTolerance} instead of the fetch reach.
 *
 * Displayed dimensions always get infinite tolerance (1e10) regardless of type.
 *
 * Called from `SpatialQueryBuilder` (geometry-aware QUERY path), from
 * `data-processor-lines.ts` (lines projection clipping, MEMBERSHIP role), and from
 * `data-processor-mesh.ts` (mesh slab membership — always the MEMBERSHIP role).
 *
 * ## Mesh cannot reuse any of the other three rules
 *
 * Every existing strategy is derived from that type's **per-element extent**, and
 * a mesh has none: a triangle's extent comes from its own three vertices
 * (`docs/specs/MESH_NODE_SPEC.md` §2.2). Both halves of the mesh arm are therefore
 * chosen rather than inherited, and each has a specific failure mode if copied:
 *
 * - **Not Lines' `0` for spatial dims.** Lines get away with zero because segment
 *   clipping *interpolates through* the slab — a segment crossing the slice yields
 *   an intersection even at zero thickness. Mesh culls whole triangles with no
 *   interpolation, so a tolerance of `0` reduces membership to **exact float
 *   equality with the slice plane** and the node renders **nothing**. This is the
 *   single most tempting wrong answer here, because Lines is the nearest structural
 *   sibling.
 * - **Not the quarter-cell query reach for discrete dims.** Mesh's slab test is a
 *   MEMBERSHIP gate applied after fetch (the node is whole-node resident, §7), not
 *   a chunk-fetch reach, so it wants the half-cell — exactly like the lines
 *   clipping slab. The `'query'` default is deliberately `< 0.5 × step` and would
 *   drop on-grid geometry.
 *
 * Be honest about what the continuous arm means: with per-vertex cull there is no
 * such thing as a true planar cut, so a continuous hidden spatial dimension renders
 * a **thick slab** ("the surface near this slice"), and the slab thickness is the
 * only control. A mesh whose hidden dims are continuous and spatial is a poor fit
 * for this node type until exact nD clipping exists (§9). The dominant real case is
 * discrete — a mesh's hidden dimensions are almost always time or channel.
 *
 * @module data/tolerance-computer
 */

import {
  GSPLAT_CHOLESKY_EPSILON,
  GSPLAT_DEFAULT_TRUNCATION_RADIUS,
} from '../../../config/constants';
import type { GeometryKind } from '../../data-loader-types';

/** Per-dimension metadata needed for tolerance computation. */
export interface DimensionInfo {
  discrete?: boolean;
  step?: number;
}

/**
 * Configuration options for tolerance computation.
 */
export interface ToleranceOptions {
  /**
   * Maximum point radius (only used for 'points' type).
   * Non-displayed spatial dimensions use this as tolerance so that
   * all points whose radius intersects the slice are included.
   * @default 1.0
   */
  maxRadius?: number;

  /**
   * Slab thickness multiplier for mesh hidden CONTINUOUS dimensions, in cells.
   * Tolerance = `step * meshSlabTolerance`.
   *
   * Mesh is the only type with a *tunable* continuous arm, and the number is an
   * invention rather than a measurement: a mesh has no per-element extent
   * (`docs/specs/MESH_NODE_SPEC.md` §2.2), so nothing in the data says how thick
   * the slab should be. (GSplats, by contrast, has no knob at all — its chunk
   * bounds already carry the real `truncation_radius · σ` extent, so its
   * continuous arm is a fixed float-safety epsilon.) One cell is the neutral
   * choice — it admits a triangle whose vertices straddle the slice by up to a
   * voxel. It must never be `0`: see the module docstring.
   *
   * @default 1.0
   */
  meshSlabTolerance?: number;

  /**
   * Flags per dimension indicating whether it is spatial (true) or discrete (false).
   * Only used for 'points' type. Dimensions beyond the array length default to spatial.
   */
  spatialExtendDims?: boolean[];

  /**
   * Which ROLE the discrete-dim tolerance plays (default `'query'`).
   *
   * - `'query'` — chunk-fetch reach: the quarter-cell `0.25 × step` (must stay
   *   below the write-side pad + half step so the neighbour category never
   *   bleeds in; see the module docstring).
   * - `'membership'` — per-element visibility gate applied AFTER fetch (the
   *   projection/clipping slab): the half-cell `0.5 × step`, matching the
   *   points gate (`effective-radius-calculator.ts`, absolute 0.5 on a unit
   *   grid) and the gsplats projection gate (`step × 0.5`). The lines
   *   projection-clipping path must use this role — with the query role its
   *   rendered cross-category whiskers halve and off-grid vertices in the
   *   `(0.25, 0.5] × step` band vanish while identical points/gsplats stay
   *   visible.
   */
  discreteRole?: 'query' | 'membership';
}

/** Infinite tolerance sentinel for displayed dimensions. */
const DISPLAYED_TOLERANCE = 1e10;

/**
 * Fraction of a step used as the hidden-dimension tolerance for DISCRETE dims,
 * shared by all three geometry types. A "quarter-cell": `< 0.5` so a query on
 * category `k` never reaches the `k±1` cell (chunk bounds are padded on the
 * write side; pad + tolerance must stay below one step or the whole neighbour
 * category bleeds in), yet `> 0` so the target cell + genuine straddle chunks
 * always match. See the module docstring.
 *
 * Why 0.25 and not 0.5 (which would exactly match the half-cell membership
 * gates — points/gsplats projection, and the lines clipping slab via
 * {@link discreteDimMembershipTolerance}): LEGACY datasets written before this
 * change pad barrier chunk bounds by ±0.5 step, so a `0.5×step` query would
 * sum to a full step and re-introduce the neighbour-category over-fetch this
 * exists to fix. `0.25` keeps the fix working for both legacy (±0.5-padded)
 * and new (ε-padded) data. The resulting `(0.25, 0.5]×step` band — where an
 * element would pass the membership gate but its chunk isn't fetched — is
 * unreachable in practice: discrete-dim navigation snaps the slice position to
 * exact category values (see `SceneDimsManager.setDimensionValue`), so queries
 * are always on-grid (offset 0) and the target cell always matches.
 *
 * That on-grid premise has exactly one other way to break, and it is guarded
 * elsewhere: a non-unit affine `nd_transform` inverts an on-grid WORLD target
 * into an off-grid LOCAL one (`scale: 2` at world 7 → local 3.5), at which
 * point the half-cell membership window below admits BOTH neighbouring
 * categories. `invertNdTransformForQuery` detects that case and reports
 * `noPreimage`, and the per-geometry range queries return nothing — see the
 * "no-preimage rule" in `data/transforms/README.md`. If you widen or narrow
 * either fraction here, keep that guard in mind: it is what lets these windows
 * assume an on-grid target.
 */
const DISCRETE_TOLERANCE_FRACTION = 0.25;

/**
 * Canonical discrete-dimension query tolerance, shared by points/lines/gsplats.
 * `0.25 × step` (fallback quarter-cell of a unit step when no step metadata).
 *
 * Exported because the points loader's live query path builds its tolerance in
 * `effective-radius-calculator.ts` (`calculateSpatialQueryTolerance`, which is
 * `EffectiveRadiusConfig`-aware) and passes it to `SpatialQueryBuilder`
 * explicitly, bypassing `computeTolerance`. That path MUST apply the same
 * quarter-cell rule for discrete dims, or points regress to the
 * neighbour-category over-fetch this module fixes.
 */
export function discreteDimTolerance(dimInfo: DimensionInfo | undefined): number {
  const step =
    dimInfo?.step !== undefined && dimInfo.step !== null && dimInfo.step > 0 ? dimInfo.step : 1;
  return DISCRETE_TOLERANCE_FRACTION * step;
}

/**
 * Half-cell fraction for the MEMBERSHIP role (per-element visibility after
 * fetch). Kept at the historical `0.5 × step` so the three geometries stay in
 * lockstep: points gate at 0.5 (unit grid), gsplats projection gate at
 * `step × 0.5`, lines clipping slab at `0.5 × step` via this constant.
 */
const DISCRETE_MEMBERSHIP_FRACTION = 0.5;

/**
 * Discrete-dimension MEMBERSHIP tolerance (`0.5 × step`): the per-element
 * visibility slab applied after chunks are fetched. Deliberately wider than
 * {@link discreteDimTolerance} (the fetch reach) — fetch must stay under the
 * write-side pad budget, while membership defines what the user sees.
 */
export function discreteDimMembershipTolerance(dimInfo: DimensionInfo | undefined): number {
  const step =
    dimInfo?.step !== undefined && dimInfo.step !== null && dimInfo.step > 0 ? dimInfo.step : 1;
  return DISCRETE_MEMBERSHIP_FRACTION * step;
}

/**
 * Step fraction of the GSplats continuous-dim float-safety epsilon — TERM 1 of
 * the two in {@link gsplatsContinuousDimTolerance}.
 *
 * `1e-3 × step`, which at a unit step is exactly the write side's
 * `_BARRIER_BOUND_EPS = 1e-3` (`luxar/io/_ordering/bounds.py`) — this is that
 * epsilon's reader-side mirror, applied to the dims the write side does NOT pad
 * (barrier/discrete dims are padded there; continuous dims are not, because they
 * normally get the far larger `truncation_radius · σ` expansion instead). The
 * mirror is pinned from the Python side by
 * `io/tests/test_ordering_gsplats.py::test_barrier_bound_eps_matches_viewer_gsplats_step_fraction`,
 * which parses this declaration out of this file.
 *
 * Why this magnitude:
 * - **500× below `0.5 × step`**, so it can never reach into a neighbouring cell
 *   — a stronger margin than the discrete quarter-cell reach has.
 * - **Comfortably above float32 round-off** on realistic coordinates: chunk
 *   bounds are stored as float32 (`chunk_bounds` is `dtype=np.float32`), whose
 *   ~1.2e-7 relative spacing costs `magnitude × 1.2e-7` of absolute slack, so
 *   `1e-3 × step` covers coordinates out to ~8000 steps from the origin.
 */
const GSPLATS_CONTINUOUS_EPS_STEP_FRACTION = 1e-3;

/**
 * Width of the band a degenerate hidden dim still RENDERS in, in that dim's own
 * units — TERM 2 of the two in {@link gsplatsContinuousDimTolerance}.
 *
 * The read side does not use a raw zero variance. `computeMarginalCholesky`
 * (`wasm/typescript/gsplats-processing.ts`, and its Rust twin
 * `wasm/rust/src/gsplats_processing.rs::compute_marginal_cholesky`) REGULARIZES a
 * degenerate pivot; when the hidden block is entirely zero there is no scale to
 * anchor a relative floor to, so it falls back to the absolute backstop and the
 * pivot becomes `sqrt(GSPLAT_CHOLESKY_EPSILON)` = 1e-5. The splat therefore keeps
 * an effective σ of 1e-5 along that axis, and the projection kernel's
 * shifted-Gaussian attenuation only reaches exactly 0 at Mahalanobis distance
 * `truncation_radius`. So the renderable band is
 * `truncation_radius × sqrt(GSPLAT_CHOLESKY_EPSILON)` ≈ 2.75e-5 — derived here
 * from the two named constants the kernel uses (both live in
 * `config/constants.ts`, which the TS reference kernel imports and where the
 * cross-language value is pinned against the Rust kernel), so it cannot drift
 * away from them.
 *
 * This band is ABSOLUTE, in the dimension's own units — the regularization floor
 * it comes from is an absolute variance backstop, not a step-relative one. That
 * is the whole reason the epsilon below cannot be capped at a fraction of a cell:
 * on a micro-step axis the band is many cells wide, and the renderer genuinely
 * shows that content.
 *
 * The factor is the DEFAULT radius, not the node's, because this function is
 * handed only a {@link DimensionInfo} — the per-node `truncation_radius` attr
 * lives on the geometry node and never reaches here. `clampTruncationRadius`
 * bounds an authored radius only to the float32-representable range, so a node
 * CAN stamp a larger one and scale its degenerate band with it; that is recorded
 * as a known limit on {@link gsplatsContinuousDimTolerance} rather than papered
 * over with a fudge factor.
 */
const GSPLATS_REGULARIZED_HIDDEN_BAND =
  GSPLAT_DEFAULT_TRUNCATION_RADIUS * Math.sqrt(GSPLAT_CHOLESKY_EPSILON);

/**
 * GSplats hidden CONTINUOUS-dimension tolerance: a float-safety epsilon
 * (`max(1e-3 × step, 2.75e-5)`), NOT a query reach.
 *
 * ## Why it is not a reach
 *
 * The reach is already baked into the data. `compute_chunk_bounds_gsplats`
 * expands each of a chunk's AABB dims by `sqrt(covariance[d,d]) × coverage_sigma`
 * with `coverage_sigma` set to the node's own `truncation_radius`, and the
 * hidden-dim cutoff on the read side is that same radius: the projection kernel
 * (`wasm/typescript/gsplats-processing.ts::project_gsplats_nd_to_3d` and its Rust
 * twin) attenuates by `(e^{-m²/2} − c)/(1 − c)` with `c = e^{-T²/2}`, clamped at
 * 0, which is exactly 0 at `m = T = truncation_radius`. (The shader's
 * `uTruncate`/`uTruncateSq` discard is the DISPLAYED-dim cutoff and never sees a
 * hidden dim.) So a chunk that fails a zero-tolerance test contains only splats
 * the projection would attenuate to nothing. This value is a chunk-FETCH reach
 * only: per-splat hidden-dim visibility comes from that projection math, never
 * from this array — see `data-processor-gsplats.ts`.
 *
 * ## The premise is CONDITIONAL — precondition
 *
 * `compute_chunk_bounds_gsplats` applies the σ expansion to every dim NOT in its
 * `slice_dims` argument, and gives the ones that ARE only a tight
 * `_BARRIER_BOUND_EPS` pad. So "bounds already carry `truncation_radius · σ` on
 * every continuous dim" holds only while **the write side's barrier set equals
 * the set this module treats as `discrete`**. It does when the scene declares its
 * dimensions — `io/_compiler/geometry_writers/gsplats.py::scene_barrier_dims`
 * uses exactly `discrete and not display`. When `scene_dimensions` is absent
 * (a standalone `.gsplats.zarr` grafted in) the writer falls back to the
 * value-based `io/_ordering/compound.py::detect_barrier_dims`, whose own docstring
 * calls a false positive a correctness bug: an integer-valued, low-cardinality
 * axis that is really spatial gets tight bounds while this side, reading it as
 * continuous, applies only the epsilon — and a σ-extended splat near a chunk edge
 * can be dropped. This disagreement is pre-existing, but be clear that the change
 * to this function REMOVES A MASK: the old `step × 3.0` rule (a bare `3.0` when
 * dimension metadata was absent) was wide enough to cover a mis-detected barrier
 * dim's tight bounds, so the mis-detection stayed invisible; a `1e-3 × step`
 * epsilon does not. The eventual robust fix is plumbing: the writer ALREADY
 * publishes the set it used as the ordering `slice_dims` attr
 * (`io/_compiler/gsplat_assembly.py`), and this module does not read it. Reading
 * it — and treating a `slice_dims` member as barrier-padded regardless of the
 * scene's `discrete` flag — would make the premise unconditional.
 *
 * ## Why the epsilon is not a literal `0`
 *
 * Two reasons, and the formula has exactly one term for each:
 *
 * 1. `GSPLATS_CONTINUOUS_EPS_STEP_FRACTION × step` — a continuous dim along which
 *    the splats have **zero variance** (a stacked axis declared continuous rather
 *    than discrete) gets no σ expansion, and the write side pads only *discrete*
 *    dims, so its stored bound is the axis value itself. The dominant
 *    perturbation is that the bound is stored as **float32** (`chunk_bounds` is
 *    `dtype=np.float32`) while the query position is a float64 — ≈1.9e-7 of
 *    disagreement at a coordinate of 5.3. The `start + k × step` arithmetic drift
 *    in the query position is real but secondary (≈9e-16 there). At tolerance `0`
 *    membership would be an exact float comparison and could silently select
 *    nothing.
 * 2. `GSPLATS_REGULARIZED_HIDDEN_BAND` — the read side regularizes a
 *    degenerate pivot, so such a splat still renders out to ≈2.75e-5 in that
 *    dim's units. Term 1 alone falls below that whenever `step < 2.75e-2` (a
 *    physical-unit axis — 10 ms frames at `step = 0.01` — gives 1e-5 against a
 *    2.75e-5 band), so splats the kernel renders at up to ~60% of full brightness
 *    (the attenuation at the 1e-5 window edge, i.e. 1 σ of the regularized pivot)
 *    would not be fetched.
 *
 * Two regimes, with a single crossover at `step = 2.75e-2`:
 * - `step ≥ 2.75e-2` → term 1, `1e-3 × step` (the `_BARRIER_BOUND_EPS` mirror).
 * - `step < 2.75e-2` → the regularization band, an ABSOLUTE 2.75e-5.
 *
 * Below the crossover the epsilon is therefore many CELLS wide — ≈27.5 cells at
 * `step = 1e-6`, ≈27500 at `step = 1e-9` — and that is deliberate, not an
 * oversight. It is what the renderer actually shows: the regularization floor
 * behind term 2 is an absolute variance backstop, so the rendered band does not
 * shrink when the declared step does, and a cap at some fraction of a cell would
 * hide renderable content from the query. (A cap was tried and removed in review:
 * at `step = 1e-6` a `0.25 × step` ceiling gives 2.5e-7 while the renderer shows
 * content out to 2.75e-5, so a splat ~10 nav steps away renders at ~60% brightness
 * with its chunk never fetched — a NARROWING even versus the old `step × 3` rule.
 * The quarter-cell rationale belongs to the DISCRETE arm, where over-reach bleeds
 * a neighbouring CATEGORY; a continuous axis has no categories, so a wide window
 * there is a bandwidth question only. This file already hands mesh a FULL cell for
 * a continuous dim, so a sub-cell ceiling is not a rule of the file.)
 *
 * The ordering property that holds at EVERY step: the epsilon is never smaller
 * than the band a degenerate hidden dim can render in, so the query can never miss
 * renderable content on such an axis. Below the crossover the absolute cost is
 * bounded by 2.75e-5 in the dimension's own units — negligible as a distance,
 * however many cells it spans. That bounds the DISTANCE, not the fraction of the
 * node fetched: when the dim's REAL σ is micro-scale too (`step = σ = 1e-9`, a
 * metre-declared axis carrying nanometre structure) the window the data needs is
 * `T · σ` = 2.75e-9, so the epsilon over-fetches by ~1e4 and can pull the whole
 * node. That is the one regime where this is worse than the old `step × 3`, and
 * the fix there is the σ-plumbing named below, not a different constant.
 *
 * The real long-term fix is not a tolerance formula: make the kernel's
 * regularization floor SCALE-AWARE for a degenerate hidden dim (anchored to the
 * dimension's own step or extent, so the rendered band shrinks with the axis), or
 * plumb the splats' actual σ per hidden dim into the query so the window is
 * measured rather than bounded. Until then the tolerance cannot be both narrow and
 * complete, and this errs on complete.
 *
 * KNOWN LIMITS (all three are "your axis is mis-declared", not "widen this
 * epsilon" — widening it reintroduces the over-fetch it replaces):
 * - **Coordinates far from the origin.** `mins = (chunk_centers - extents)` is
 *   stored float32, so an expansion smaller than half an ULP of the coordinate is
 *   rounded away entirely. A float32 ULP is `|coord| × 2^-23 … 2^-24`, so with
 *   `T = 2.75` **any** `σ_d ≲ (1.1–2.2)e-8 × |coord|` behaves exactly like zero
 *   variance, not just an exactly-zero one. Combined with term 1, a dim whose
 *   coordinates sit more than ~8000 steps from the origin can round outside this
 *   window. Re-origin the axis, or declare it discrete.
 * - **A second hidden dim with very large σ.** The regularization floor is
 *   RELATIVE — `sqrt(maxDiag × CHOLESKY_RELATIVE_EPSILON)`, i.e.
 *   `σ_max × 1e-6` — whenever the hidden block is not all-zero. A node with a
 *   degenerate dim *and* another hidden dim of large σ therefore renders a band of
 *   `T × σ_max × 1e-6`, which exceeds this epsilon once
 *   `σ_max > max(1e-3 × step / (T × 1e-6), 10)` in that dim's units: that first
 *   branch is `≈363.6 × step` where term 1 dominates, and
 *   `2.75e-5 / (2.75 × 1e-6)` = 10 in the band regime. (The two agree exactly at
 *   the crossover `step = 2.75e-2`, where the unrounded coefficient gives 10.) Not
 *   covered here on purpose: covering it would make the fetch window depend on an
 *   unrelated axis's extent.
 * - **A node with a much larger `truncation_radius` than the default.** The band
 *   term scales linearly with `T`, and this function only sees a `DimensionInfo`,
 *   so it uses `GSPLAT_DEFAULT_TRUNCATION_RADIUS` (2.75, the value a fitted dataset
 *   stamps). A node stamping, say, `T = 6` renders a degenerate band ~2.2× wider;
 *   that only matters below the crossover, where the band term dominates. Plumbing
 *   the node's `truncation_radius` into `ToleranceOptions` would close it — the
 *   same follow-up as the `slice_dims` plumbing above.
 *
 * @param dimInfo - Metadata for the dimension (only `step` is read).
 * @returns The epsilon in that dimension's units.
 */
export function gsplatsContinuousDimTolerance(dimInfo: DimensionInfo | undefined): number {
  const step =
    dimInfo?.step !== undefined && dimInfo.step !== null && dimInfo.step > 0 ? dimInfo.step : 1;
  return Math.max(GSPLATS_CONTINUOUS_EPS_STEP_FRACTION * step, GSPLATS_REGULARIZED_HIDDEN_BAND);
}

/**
 * Compute per-dimension query tolerances for a spatial-index query.
 *
 * @param geometryType - Geometry type determines the strategy for hidden dimensions.
 * @param displayDims - Indices of the currently displayed (rendered) dimensions.
 * @param ndim - Total number of dimensions.
 * @param dimensions - Optional per-dimension metadata (step, discrete flag).
 * @param options - Additional options (maxRadius for points, meshSlabTolerance for mesh, etc.).
 * @returns Tolerance array of length ndim.
 */
export function computeTolerance(
  geometryType: GeometryKind,
  displayDims: readonly number[],
  ndim: number,
  dimensions?: DimensionInfo[],
  options: ToleranceOptions = {}
): number[] {
  const tolerance = new Array<number>(ndim).fill(0);

  for (let d = 0; d < ndim; d++) {
    if (displayDims.includes(d)) {
      tolerance[d] = DISPLAYED_TOLERANCE;
    } else {
      tolerance[d] = computeHiddenDimTolerance(geometryType, d, dimensions?.[d], options);
    }
  }

  return tolerance;
}

/**
 * Compute tolerance for a single hidden (non-displayed) dimension.
 */
function computeHiddenDimTolerance(
  geometryType: GeometryKind,
  dimIndex: number,
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  switch (geometryType) {
    case 'points':
      return computePointsHiddenTolerance(dimIndex, dimInfo, options);
    case 'lines':
      return computeLinesHiddenTolerance(dimInfo, options);
    case 'gsplats':
      return computeGSplatsHiddenTolerance(dimInfo);
    case 'mesh':
      return computeMeshHiddenTolerance(dimInfo, options);
  }
}

/**
 * Points hidden dimension tolerance.
 *
 * Spatial dimensions use `maxRadius`; discrete dimensions use the shared
 * quarter-cell rule.
 *
 * Don't hunt for a live caller: in production only `'lines'` and `'gsplats'` reach
 * {@link computeTolerance} through `geometryType` — the points loader always passes
 * a precomputed `tolerance` — so this arm serves tests and any future
 * `geometryType: 'points'` caller, and the LIVE equivalents of the reasoning below
 * are `calculateSpatialQueryTolerance` / `fallbackQueryTolerance` in
 * `data/points/effective-radius-calculator.ts`, which apply the same rule.
 *
 * Why points keep a full-radius reach where gsplats do not — and be precise,
 * because the rationale covers only ONE of the two write-side branches. The points
 * bound builder expands a chunk's AABB by the point radius when radii are actually
 * PRESENT, per-point or scalar (`io/_ordering/points.py:99-106`); THERE the bounds
 * already carry the full radius and this arm's `maxRadius` is exactly the kind of
 * pure over-fetch the gsplats arm just removed. With radii ABSENT the builder uses
 * a `max(1% of the coordinate range, 0.01)` safety margin instead
 * (`io/_ordering/points.py:113-115`), which is unrelated to — and generally much
 * smaller than — the default radius the renderer then applies, so those bounds
 * UNDER-expand and the reader-side `maxRadius` is what keeps a point just outside
 * them from being dropped. That split is the honest reason points cannot simply
 * copy the gsplats fix: this function is handed no radii-presence flag, so
 * narrowing the arm would silently drop points on every radii-absent dataset. Its
 * cost is over-fetch on the (common) radii-present ones.
 *
 * GSplats need no counterpart because their bounds carry the full
 * `truncation_radius · σ` extent on every dim the write side treated as spatial,
 * unconditionally — see {@link gsplatsContinuousDimTolerance}, including the
 * precondition under which that set matches the one this module reads.
 *
 * The thing that should eventually change is the Python side: expand the
 * radii-absent bounds by the renderer's default radius (or publish the
 * radii-presence flag), after which this arm could shrink to a float-safety
 * epsilon like the gsplats one.
 */
function computePointsHiddenTolerance(
  dimIndex: number,
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  const { maxRadius = 1.0, spatialExtendDims } = options;

  // Determine if this dimension is spatial.
  // Points use spatialExtendDims (from EffectiveRadiusConfig) rather than
  // per-dimension discrete flags. Dimensions beyond the array default to spatial.
  const isSpatial =
    spatialExtendDims === undefined ||
    dimIndex >= spatialExtendDims.length ||
    spatialExtendDims[dimIndex];

  if (isSpatial) {
    return maxRadius;
  }
  // Discrete (non-spatial) dimension
  return discreteDimTolerance(dimInfo);
}

/**
 * Lines hidden dimension tolerance.
 *
 * Spatial dimensions get 0 because segment bounding boxes already include
 * the line width extent. Discrete dimensions use the shared quarter-cell
 * query rule, or the half-cell membership rule when the caller is the
 * projection/clipping path (`options.discreteRole === 'membership'`).
 */
function computeLinesHiddenTolerance(
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  if (dimInfo?.discrete) {
    return options.discreteRole === 'membership'
      ? discreteDimMembershipTolerance(dimInfo)
      : discreteDimTolerance(dimInfo);
  }
  // Spatial dimension: bounds already include width
  return 0;
}

/**
 * GSplats hidden dimension tolerance.
 *
 * Discrete dimensions use the shared quarter-cell rule. Continuous dimensions get
 * the float-safety epsilon {@link gsplatsContinuousDimTolerance} — not a reach:
 * the chunk bounds already carry the `truncation_radius · σ` extent the projection
 * kernel attenuates to zero at, so anything further out is pure over-fetch. That
 * holds under the precondition documented on
 * {@link gsplatsContinuousDimTolerance} (the write side's barrier set agrees with
 * this module's `discrete` set).
 */
function computeGSplatsHiddenTolerance(dimInfo: DimensionInfo | undefined): number {
  if (dimInfo?.discrete) {
    return discreteDimTolerance(dimInfo);
  }
  return gsplatsContinuousDimTolerance(dimInfo);
}

/**
 * Mesh hidden dimension tolerance.
 *
 * Discrete dims get the **half-cell membership** rule, not the quarter-cell query
 * reach the other three default to: a mesh is whole-node resident, so this slab is
 * a per-element visibility gate applied after fetch, exactly like the lines
 * projection-clipping slab. This is the dominant real case — a mesh's hidden
 * dimensions are almost always time or channel.
 *
 * Continuous dims get `step × meshSlabTolerance` (default one cell). Emphatically
 * **not** Lines' `0`, which would reduce whole-triangle membership to exact float
 * equality with the slice plane and render nothing. See the module docstring.
 *
 * Unlike its three siblings this function ignores `options.discreteRole`. There is
 * no query role to serve: mesh has no spatial index and issues no range query, so
 * the membership rule is the only rule it has. Honouring a `'query'` role here
 * would mean quietly returning a fetch reach to the one caller that is asking about
 * visibility.
 */
function computeMeshHiddenTolerance(
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  if (dimInfo?.discrete) {
    return discreteDimMembershipTolerance(dimInfo);
  }
  const slabCells = options.meshSlabTolerance ?? 1.0;
  return dimInfo?.step ? dimInfo.step * slabCells : slabCells;
}
