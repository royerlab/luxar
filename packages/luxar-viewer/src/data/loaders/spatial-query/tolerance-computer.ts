/**
 * Canonical tolerance computer used by all geometry-specific spatial-index loaders.
 *
 * THE RULE A MAINTAINER NEEDS, in one sentence: the writer's published barrier set
 * says which rule matches the chunk BOUNDS; the scene's `discrete` flag says which
 * gate the RENDERER applies; when they disagree the fetch window follows the
 * renderer. Everything below is that sentence applied per geometry type.
 *
 * Displayed dimensions always get infinite tolerance (1e10) regardless of type.
 * Hidden (non-displayed) dimensions split by geometry:
 * - **Points**: `maxRadius` for spatial dims (selected via the `spatialExtendDims`
 *   option, the per-dimension flag array carried by `EffectiveRadiusConfig`), the
 *   shared quarter-cell {@link discreteDimTolerance} otherwise. Points never ask
 *   about barrier-ness at all — see `computePointsHiddenTolerance`.
 * - **Lines**: 0 for spatial dims (segment bounding boxes already include line
 *   width extent); {@link discreteDimTolerance} for barrier dims, or
 *   {@link discreteDimMembershipTolerance} when the caller is the
 *   projection-clipping slab (`discreteRole: 'membership'`).
 * - **GSplats**: three cases, not two, derived in `computeGSplatsHiddenTolerance` —
 *   barrier dims take the quarter-cell reach, genuinely continuous ones the
 *   float-safety epsilon {@link gsplatsContinuousDimTolerance} (deliberately NOT a
 *   reach: the chunk bounds already carry the `truncation_radius · σ` expansion the
 *   projection kernel attenuates to zero at), and a dim the writer's published set
 *   OMITS while the scene declares it `discrete` takes the half-cell membership
 *   window.
 * - **Mesh**: `step × meshSlabTolerance` (default one cell) for continuous dims,
 *   and the half-cell MEMBERSHIP rule for barrier ones — the only type whose barrier
 *   arm is not the quarter-cell query reach, because a mesh is whole-node resident
 *   and BOTH of its arms are visibility gates. See the mesh note below; it is the
 *   one arm where copying a neighbour silently renders nothing.
 *
 * The quarter-cell barrier reach and the half-cell membership gate are two different
 * numbers on purpose; `DISCRETE_TOLERANCE_FRACTION` is where that is argued.
 * "Is this dim a barrier?" is answered once per dimension by `isBarrierDim`, which
 * consults the writer's published set for GSPLATS ONLY — see its docstring for why
 * the seam stops there.
 *
 * Called from `SpatialQueryBuilder` (geometry-aware QUERY path), from
 * `data-processor-lines.ts` (lines projection clipping, MEMBERSHIP role), and from
 * `data-processor-mesh.ts` (mesh slab membership — mesh has no query role at all).
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
   * the slab should be. (GSplats, by contrast, exposes no slab knob — its chunk
   * bounds already carry the real `truncation_radius · σ` extent, so its
   * continuous arm is a float-safety epsilon, `max(1e-3 × step, T × 1e-5)`. `T`
   * is per-node rather than a constant, but it is read off the store, not
   * authored here: nothing about that arm is a tuning choice.) One cell is the neutral
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
   * The AUTHORITATIVE set of barrier (categorical) dimension indices, as
   * PUBLISHED BY THE WRITER — the top-level `slice_dims` ordering attr, in
   * center-column / `chunk_bounds` column indices, the same space as this
   * module's `d` loop over `ndim`.
   *
   * GSPLATS-ONLY today, and deliberately: `isBarrierDim` ignores this option for
   * the other three types, so supplying it to them is a no-op rather than a
   * silent behaviour change. It picks the rule matching the chunk BOUNDS, which
   * by itself can NARROW a window the renderer still draws, so each arm owes its
   * own check before honouring it — and only the gsplats arm has one (its demote
   * case, `computeGSplatsHiddenTolerance`). Before wiring lines or mesh up, give
   * that arm a floor FIRST: lines' continuous arm is a literal `0`, so a demoted
   * dim would lose its reach entirely while `data-processor-lines.ts` keeps
   * clipping against a half-cell membership slab, and mesh's two arms are both
   * MEMBERSHIP gates, so a promoted dim narrows a full cell to a half — changing
   * what the user SEES rather than how much is downloaded.
   *
   * When SUPPLIED, a gsplats dimension is a barrier iff `barrierDims.includes(d)`,
   * whatever `DimensionInfo.discrete` says — the write side is the only party
   * that knows which dims it gave a tight `_BARRIER_BOUND_EPS` pad instead of
   * the `truncation_radius · σ` expansion, so reading its answer makes the
   * gsplats premise structural rather than conditional (see
   * {@link gsplatsContinuousDimTolerance}).
   *
   * When ABSENT (`undefined`) barrier-ness falls back to
   * `DimensionInfo.discrete`, which is the historical behaviour and all a
   * legacy store with no `slice_dims` attr can offer. An EMPTY array is
   * meaningful and is NOT "absent": it means the writer used pure spatial
   * ordering, so NOTHING is a barrier.
   *
   * INDEX SPACE, for whoever eventually wires another type up. Center-column
   * space holds for GSPLATS (`io/_compiler/gsplat_assembly.py` stamps
   * `slice_dims` whenever `ordering != 'none'`) and for POINTS, the two types
   * that write their ordering keys flat. It does NOT hold for LINES, which has
   * two orderings and therefore namespaces them: there is no top-level
   * `slice_dims` at all (`io/_compiler/geometry_writers/lines.py` writes
   * `vertex_ordering` / `segment_ordering` dicts), and of those two only
   * `vertex_ordering["slice_dims"]` is in D space — `segment_ordering`'s copy is
   * in DOUBLED (2×D) space, since `io/_ordering/lines.py` appends both `i` and
   * `n_dims_original + i` for each discrete dim. The SEGMENT chunk bounds a
   * lines query scans are nevertheless built from the D-space
   * `vertex_ordering["slice_dims"]`, so a future lines caller must pass that one
   * and never `segment_ordering`'s.
   *
   * The caller owns validation: this array comes off disk and is untrusted, so
   * the gsplats loader rejects a bad attr WHOLESALE rather than filtering
   * it (see `gsplats-spatial-index-loader.ts::readBarrierDims`).
   */
  barrierDims?: readonly number[];

  /**
   * The node's own Gaussian `truncation_radius` T, in sigmas. Scales the
   * degenerate-band term of the gsplats continuous arm
   * ({@link gsplatsContinuousDimTolerance}).
   *
   * Pass the value the RENDERER will actually use, i.e. already through
   * `rendering/materials/gsplat/math.ts::clampTruncationRadius` — the band this
   * covers is the band the material draws, so the two must agree. The gsplats
   * loader does exactly that.
   *
   * Absent ⇒ `GSPLAT_DEFAULT_TRUNCATION_RADIUS`, which is also what the material
   * path substitutes for an absent attr. A `0`, a negative or a non-finite value
   * also lands on that default, but that is only a BACKSTOP against an
   * unsanitized value, not a second rule: `clampTruncationRadius` floors at
   * `MIN_TRUNCATION_RADIUS` (≈2.44e-4) rather than the default, so the real
   * caller can never send one. A caller that skips the clamp and passes an
   * UNCLAMPED radius in `(0, MIN_TRUNCATION_RADIUS)` gets a band narrower than
   * the one the material will draw with (the material raises it to MIN, this
   * takes it literally) — named here rather than hidden, and avoided by doing
   * what the paragraph above says.
   */
  truncationRadius?: number;

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
 * Fraction of a step used as the hidden-dimension QUERY tolerance for BARRIER dims,
 * shared by points/lines/gsplats. A "quarter-cell": `< 0.5` so a query on
 * category `k` never reaches the `k±1` cell (chunk bounds are padded on the
 * write side; pad + tolerance must stay below one step or the whole neighbour
 * category bleeds in — e.g. loading timepoint `t−1` in full when scrubbing to
 * `t`), yet `> 0` so the target cell + genuine straddle chunks always match.
 *
 * THE AUTHORITATIVE ANSWER to "why 0.25 and not 0.5", which would exactly match the
 * half-cell membership gates ({@link discreteDimMembershipTolerance}): LEGACY
 * datasets pad barrier chunk bounds by ±0.5 step, so a `0.5 × step` query would sum
 * to a full step and re-introduce the neighbour-category over-fetch this exists to
 * fix. `0.25` keeps the fix working for both legacy (±0.5-padded) and new (ε-padded)
 * data.
 *
 * That is a statement about dims the WRITER barrier-padded, and it is therefore NOT
 * available to a dim the writer left OUT of its barrier set: nothing padded such a
 * dim's bounds by ±0.5 step, so a half-cell query there cannot double up with a
 * half-step pad. That is why `computeGSplatsHiddenTolerance`'s demote arm uses the
 * half-cell instead — this quarter is not the file's universal discrete answer.
 *
 * The resulting `(0.25, 0.5] × step` band — where an element passes the membership
 * gate but its chunk is not fetched — remains open on the barrier arm, and is
 * unreachable for a CORRECTLY authored axis: discrete-dim navigation snaps the slice
 * position to exact category values (see `SceneDimsManager.setDimensionValue`), so an
 * axis whose stored values ARE multiples of its declared step is queried on-grid
 * (offset 0) and its target cell always matches. An axis whose stored values are
 * off-grid is a write-time authoring fault, which the compiler already reports —
 * `io/_compiler/finalize/validation.py::validate_discrete_dimension_ranges` warns
 * when a scene-declared discrete axis's declared range sits more than a quarter step
 * outside the data it holds.
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
 * {@link discreteDimTolerance} (the fetch reach) — see
 * `DISCRETE_TOLERANCE_FRACTION` for why the two differ.
 *
 * Not only a post-fetch rule: `computeGSplatsHiddenTolerance` also uses it as a
 * chunk-FETCH window for the one case where the two roles must coincide (its demote
 * arm — see that function).
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
 * A function of the node's truncation radius `T`, not a constant: `T × 1e-5`.
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
 * `truncation_radius × sqrt(GSPLAT_CHOLESKY_EPSILON)` — ≈ 2.75e-5 at the default
 * radius — derived here from the two named constants the kernel uses (both live
 * in `config/constants.ts`, which the TS reference kernel imports and where the
 * cross-language value is pinned against the Rust kernel), so it cannot drift
 * away from them.
 *
 * This band is ABSOLUTE, in the dimension's own units — the regularization floor
 * it comes from is an absolute variance backstop, not a step-relative one. That
 * is the whole reason the epsilon below cannot be capped at a fraction of a cell:
 * on a micro-step axis the band is many cells wide, and the renderer genuinely
 * shows that content.
 *
 * The factor is the NODE's own radius when the caller supplies one, and
 * {@link GSPLAT_DEFAULT_TRUNCATION_RADIUS} (2.75) otherwise; a node can legally stamp
 * e.g. `T = 6` and render a band ~2.2× wider than the default's, so scaling this term
 * with it is what keeps the window over that band. Where the value comes from and why
 * it must be the CLAMPED one: {@link ToleranceOptions.truncationRadius}.
 */
function regularizedHiddenBand(truncationRadius: number | undefined): number {
  // Backstop only, for a caller that skipped the authoritative
  // `clampTruncationRadius` (see {@link ToleranceOptions.truncationRadius}). Be exact
  // about its reach: it covers just the inputs that would COLLAPSE the term or POISON
  // it — `undefined`, `0`, a negative, a NaN, and a magnitude whose float32 square is
  // not finite (`1e308`, `1e30`). It does NOT bound the band from above, and cannot:
  // see the extreme-radius entry in {@link gsplatsContinuousDimTolerance}'s KNOWN
  // LIMITS.
  const T =
    truncationRadius !== undefined &&
    Number.isFinite(truncationRadius) &&
    truncationRadius > 0 &&
    Number.isFinite(Math.fround(truncationRadius * truncationRadius))
      ? truncationRadius
      : GSPLAT_DEFAULT_TRUNCATION_RADIUS;
  return T * Math.sqrt(GSPLAT_CHOLESKY_EPSILON);
}

/**
 * GSplats hidden CONTINUOUS-dimension tolerance: a float-safety epsilon
 * (`max(1e-3 × step, T × 1e-5)`, where `T` is the node's truncation radius —
 * so `max(1e-3 × step, 2.75e-5)` at the default `T`), NOT a query reach.
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
 * ## Which dims the premise applies to
 *
 * `compute_chunk_bounds_gsplats` applies the σ expansion to every dim NOT in its
 * `slice_dims` argument, and gives the ones that ARE only a tight
 * `_BARRIER_BOUND_EPS` pad. So "bounds already carry `truncation_radius · σ`" is a
 * statement about the dims OUTSIDE the write side's barrier set, and this arm is only
 * the right rule for exactly those dims — a barrier dim has tight bounds and wants
 * the (far wider) quarter-cell reach instead. Which dims those are is settled by
 * `isBarrierDim`, from the set the writer publishes; see its docstring for the
 * resolution rule, the legacy `discrete` fallback and the write-side
 * misclassification it does NOT fix.
 *
 * One consequence worth naming here: the #1183 change to this function REMOVED A MASK
 * for that write-side risk on a store with no published set. The previous `step × 3.0`
 * rule (a bare `3.0` when dimension metadata was absent) was wide enough to cover a
 * mis-detected barrier dim's tight bounds, so the mis-detection stayed invisible; a
 * `1e-3 × step` epsilon does not.
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
 * 2. `regularizedHiddenBand` — the read side regularizes a
 *    degenerate pivot, so such a splat still renders out to `T × 1e-5` in that
 *    dim's units (≈2.75e-5 at the default `T`). Term 1 alone falls below that
 *    whenever `step < T × 1e-2` (a physical-unit axis — 10 ms frames at
 *    `step = 0.01` — gives 1e-5 against a 2.75e-5 band), so splats the kernel
 *    renders at up to ~60% of full brightness (the attenuation at the 1e-5 window
 *    edge, i.e. 1 σ of the regularized pivot) would not be fetched.
 *
 * Two regimes, with a single crossover at `step = T × 1e-2` (2.75e-2 at the
 * default `T`):
 * - `step ≥ T × 1e-2` → term 1, `1e-3 × step` (the `_BARRIER_BOUND_EPS` mirror).
 * - `step < T × 1e-2` → the regularization band, ABSOLUTE in the dim's units.
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
 * bounded by `T × 1e-5` in the dimension's own units — negligible as a distance,
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
 * KNOWN LIMITS. The first two are "your axis is mis-declared", not "widen this
 * epsilon" (widening it reintroduces the over-fetch it replaces); the third is
 * "do not author that node":
 * - **Coordinates far from the origin.** No longer the write side's doing: since
 *   #1655 `io/_ordering/gsplats.py::compute_chunk_bounds_gsplats` accumulates
 *   `chunk_centers ± extents` in float64 and narrows it to the float32 store with
 *   OUTWARD rounding (`_store_outward_f32_array`), exactly as the points builder
 *   does — so an expansion smaller than half a float32 ULP is widened to a full
 *   ULP instead of being discarded, and a small `σ_d` on a large coordinate no
 *   longer behaves like zero variance. (It used to, and a store written before
 *   that fix still carries the tight bounds: with `T = 2.75`, **any**
 *   `σ_d ≲ (1.1–2.2)e-8 × |coord|` rounded away entirely.) What remains is the
 *   stored COORDINATE, not the pad — a dim with genuinely zero variance still
 *   gets a zero-width bound (the outward step has nothing to widen), sitting on a
 *   float32 value while the query position is a float64 `start + k × step`, and
 *   the float32 spacing at `|coord|` is `|coord| × 1.2e-7`. So a dim whose
 *   coordinates sit more than ~8000 steps from the origin can still fall outside
 *   term 1's window. Re-origin the axis, or declare it discrete.
 * - **A second CONTINUOUS-hidden dim with very large σ.** The regularization floor is
 *   RELATIVE — `sqrt(maxDiag × CHOLESKY_RELATIVE_EPSILON)`, i.e.
 *   `σ_max × 1e-6` — whenever the hidden block is not all-zero. That block is the
 *   marginal over `continuousHiddenDims` ONLY (`computeMarginalCholesky` is called
 *   with exactly that array, and `classifyHiddenDims` keeps a scene-discrete dim out
 *   of it), so a scene-discrete hidden dim's σ can never raise this floor however
 *   large it is — the other dim has to be continuous-hidden too. A node with a
 *   degenerate dim *and* another such dim of large σ therefore renders a band of
 *   `T × σ_max × 1e-6`, which exceeds this epsilon once
 *   `σ_max > max(1e-3 × step / (T × 1e-6), 10)` in that dim's units. Both branches
 *   in closed form: where term 1 dominates it is `1000 × step / T` (≈363.6 × step
 *   at the default `T` = 2.75, ≈166.7 × step at `T` = 6 — it SHRINKS as the radius
 *   grows, because the epsilon it is compared against grows), and in the band
 *   regime it is `(T × 1e-5) / (T × 1e-6)` = 10, independent of `T` entirely. (The
 *   two agree exactly at the crossover `step = T × 1e-2` for any `T`: there
 *   `1000 × step / T` = 10.) Not
 *   covered here on purpose: covering it would make the fetch window depend on an
 *   unrelated axis's extent.
 * - **An extreme but VALID `truncation_radius` fetches the whole node.** The band
 *   term is `T × 1e-5` with no ceiling, so a node stamping `T = 1e18` gets a window
 *   of ~1e13 in that dim's units — the debug query log prints it as `∞`
 *   (`formatTolerance`) and effectively every chunk matches. This is not a hole in
 *   the guard above: `1e18` is finite, `clampTruncationRadius` passes it (its upper
 *   bound is `sqrt(float32.max)` ≈ 1.84e19, `rendering/materials/gsplat/math.ts`)
 *   and so does the WRITE side (`MAX_TRUNCATION_RADIUS_FLOAT32`, the same value, in
 *   `packages/luxar/src/luxar/validation/types.py::validate_truncation_radius`), so
 *   it is a legal authored value on both sides of the contract. Capping the band
 *   here would be the wrong fix: the MATERIAL draws that band too, so a cap
 *   re-creates exactly the fetch-narrower-than-render under-fetch #1655 item 3
 *   removed. The window is following the renderer faithfully; the remedy is not to
 *   author a radius three orders of magnitude past anything physical (the default is
 *   {@link GSPLAT_DEFAULT_TRUNCATION_RADIUS} = 2.75, and the lower bound
 *   `MIN_TRUNCATION_RADIUS` ≈ 2.44e-4 lives beside the upper one).
 *
 * A third limit was closed rather than documented: a node with a much larger
 * `truncation_radius` than the default used to be scored against the DEFAULT band
 * (this function saw only a `DimensionInfo`), leaving the window `(2.75e-5, T×1e-5]`
 * uncovered for a node stamping a bigger `T`. `truncationRadius` now carries the
 * node's own value, so the band term scales with it.
 *
 * @param dimInfo - Metadata for the dimension (only `step` is read).
 * @param truncationRadius - The node's `truncation_radius`, already through
 *   `clampTruncationRadius` (see {@link ToleranceOptions.truncationRadius});
 *   omitted ⇒ `GSPLAT_DEFAULT_TRUNCATION_RADIUS`.
 * @returns The epsilon in that dimension's units.
 */
export function gsplatsContinuousDimTolerance(
  dimInfo: DimensionInfo | undefined,
  truncationRadius?: number
): number {
  const step =
    dimInfo?.step !== undefined && dimInfo.step !== null && dimInfo.step > 0 ? dimInfo.step : 1;
  return Math.max(
    GSPLATS_CONTINUOUS_EPS_STEP_FRACTION * step,
    regularizedHiddenBand(truncationRadius)
  );
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
 * Is dimension `dimIndex` a BARRIER (categorical) dimension?
 *
 * One resolution site for the whole module, so no arm can end up with a different
 * answer than its siblings. Two sources, in priority order:
 *
 * 1. `options.barrierDims` — the writer's own published set
 *    ({@link ToleranceOptions.barrierDims}), **for `'gsplats'` only**. Authoritative
 *    when supplied, INCLUDING when it is empty: `[]` means the writer ordered purely
 *    spatially, so no dim is a barrier even if the scene declares one `discrete`. It
 *    overrides `discrete` in BOTH directions — a `discrete: false` dim the writer
 *    barrier-padded becomes a barrier, and a `discrete: true` dim the writer
 *    σ-expanded does not (the "demote" case).
 * 2. `DimensionInfo.discrete` — the historical rule. The answer for every non-gsplats
 *    geometry, and for a gsplats store that publishes no barrier set (a legacy store,
 *    or one with `ordering: 'none'`, which has no chunk index and issues no query).
 *
 * Why source 1 is gated on the geometry rather than on the caller merely passing it:
 * it selects which rule matches the chunk BOUNDS and says nothing about the band the
 * read side renders, so honouring it can NARROW a window that still has content in
 * it. Each arm therefore owes its own check, and only the gsplats arm has one (its
 * demote case, {@link computeGSplatsHiddenTolerance}). Making the seam gsplats-only
 * means wiring another type up is a deliberate edit that has to add that arm's floor
 * — see {@link ToleranceOptions.barrierDims} for what lines and mesh would each need.
 *
 * The write side is what the published set buys: it is the only party that knows
 * which dims it gave a tight `_BARRIER_BOUND_EPS` pad instead of the
 * `truncation_radius · σ` expansion, so reading it removes the reader's independent
 * guess. It does NOT make the writer's own choice right — with `scene_dimensions`
 * present the writer uses
 * `io/_compiler/geometry_writers/gsplats.py::scene_barrier_dims`
 * (`discrete and not display`), and absent it falls back to the value-based
 * `io/_ordering/compound.py::detect_barrier_dims`, whose own docstring calls a false
 * positive a correctness bug (a really-spatial integer, low-cardinality axis gets
 * tight bounds). That remains a WRITE-side bug producing bounds tighter than the
 * splats' true extent; what is gone is the reader adding a second, independent
 * disagreement on top of it.
 *
 * POINTS never ask this question at all: they classify dims from
 * `options.spatialExtendDims` (the `EffectiveRadiusConfig` flag array), never from
 * `discrete`, and their live query path does not even come through here.
 *
 * The `discrete` arm is a TRUTHINESS test, not `=== true`, and deliberately so: the
 * flag rides through from the store's `scene_dimensions` JSON uncoerced
 * (`view-state-manager.ts::extractMetadata` copies it as-is), and the renderer's own
 * gate is truthy too — `data-processor-gsplats.ts::buildGSplatsParams` fills
 * `discreteDims` from `if (viewState.dimensions[d]?.discrete)`. A hand-authored
 * `"discrete": 1` must not classify one way here and the other way there, or the
 * fetch window stops matching the gate the renderer applies, which is the whole
 * failure this module exists to avoid.
 */
function isBarrierDim(
  geometryType: GeometryKind,
  dimIndex: number,
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): boolean {
  const { barrierDims } = options;
  if (geometryType === 'gsplats' && barrierDims !== undefined) {
    return barrierDims.includes(dimIndex);
  }
  return !!dimInfo?.discrete;
}

/**
 * Compute tolerance for a single hidden (non-displayed) dimension.
 *
 * Barrier-ness is resolved ONCE here (`isBarrierDim`) and passed down, so
 * the three arms that branch on it cannot drift apart.
 */
function computeHiddenDimTolerance(
  geometryType: GeometryKind,
  dimIndex: number,
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  // Points classify from `spatialExtendDims`, not from barrier-ness — see
  // `isBarrierDim`.
  if (geometryType === 'points') {
    return computePointsHiddenTolerance(dimIndex, dimInfo, options);
  }

  const isBarrier = isBarrierDim(geometryType, dimIndex, dimInfo, options);
  switch (geometryType) {
    case 'lines':
      return computeLinesHiddenTolerance(dimInfo, isBarrier, options);
    case 'gsplats':
      return computeGSplatsHiddenTolerance(dimInfo, isBarrier, options);
    case 'mesh':
      return computeMeshHiddenTolerance(dimInfo, isBarrier, options);
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
 * Why points keep a full-radius reach where gsplats do not, and what would have to
 * change to drop it. The WRITE side is no longer the obstacle: since #1658
 * `io/_ordering/points.py::compute_chunk_bounds_points` pads a chunk's AABB by the
 * point radius on every path — a per-point or scalar radius when radii are present,
 * and `DEFAULT_POINT_RADIUS` (the radius the renderer will actually draw with) when
 * they are absent — computes the interval in float64 and narrows it to the float32
 * store with OUTWARD rounding (`_store_outward_f32`), so a small absolute pad on a
 * large coordinate is no longer rounded away. Those bounds ARE the footprint, so on a
 * freshly written store this arm's `maxRadius` is the kind of pure over-fetch the
 * gsplats arm removed.
 *
 * What still blocks the shrink is old data, not the writer: a store written before
 * that fix carries the tight bounds (radii-absent chunks padded by a
 * `max(1% of range, 0.01)` margin unrelated to the renderer's default radius), and
 * nothing in the format lets this function tell the two apart — there is no bounds
 * version stamp and no radii-presence flag on the query path. Narrowing the arm
 * unconditionally would therefore silently drop points on every legacy radii-absent
 * dataset, so the reach stays until such a stamp exists.
 *
 * GSplats need no counterpart because their bounds carry the full
 * `truncation_radius · σ` extent on every dim the write side treated as spatial. See
 * {@link gsplatsContinuousDimTolerance}.
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
 * the line width extent. That premise is now underwritten on the WRITE side, and
 * it was not always: `chunk_bounds` is float32 while the vertices themselves are
 * stored as per-axis uint16 fixed point, so a bound built from the authored
 * vertices could be escaped by the DECODED ones by up to half a quantum
 * (`extent/131070`, 7.6e-3 at an axis extent of 1000) — a chunk this arm would
 * then never ask for, dropping geometry silently. Since #1655 the writer pads
 * both `vertex_chunk_bounds` and `segment_chunk_bounds` outward by the encoder's
 * own per-axis round-trip slack (`ArrayEncoder.coordinate_round_trip_slack`,
 * asked with `allow_lut=false` to match how the lines writer stores `vertices`),
 * on top of the width footprint, and accumulates in float64 before an
 * outward-rounded float32 store. So a stored lines bound now contains the
 * decoded, full-width segment footprint, and a `0` reach here is sound rather
 * than merely conventional. (Stores written before 2026-08 keep their old,
 * occasionally-too-tight bounds; the quantised RADIUS/WIDTH itself is still an
 * open gap on the writer side — a decoded width can exceed the authored one the
 * pad was sized for by up to half its own quantum, ~9.6e-3 on typical data.)
 *
 * `isBarrier` always comes from `DimensionInfo.discrete` here: `isBarrierDim` honours
 * the writer's published set for gsplats only, and this arm's literal `0` is exactly
 * why (see {@link ToleranceOptions.barrierDims}).
 */
function computeLinesHiddenTolerance(
  dimInfo: DimensionInfo | undefined,
  isBarrier: boolean,
  options: ToleranceOptions
): number {
  if (isBarrier) {
    return options.discreteRole === 'membership'
      ? discreteDimMembershipTolerance(dimInfo)
      : discreteDimTolerance(dimInfo);
  }
  // Spatial dimension: bounds already include width — and, since #1655, the
  // writer's uint16 round-trip slack too, so they contain the DECODED vertices.
  return 0;
}

/**
 * GSplats hidden dimension tolerance.
 *
 * THE PRINCIPLE: **the fetch window must equal the only visibility gate the renderer
 * applies to that dim.** The three cases below all follow from it, because the two
 * sides classify a hidden dim from DIFFERENT sources:
 * - The writer's published set (`options.barrierDims`, resolved by `isBarrierDim`)
 *   governs the **chunk-bounds** question — which dims got a tight
 *   `_BARRIER_BOUND_EPS` pad rather than the `truncation_radius · σ` expansion. A
 *   fact about the data on disk, so it decides which rule COULD be right.
 * - The scene's `DimensionInfo.discrete` flag governs the **per-splat membership**
 *   gate, and governs it EXCLUSIVELY: `data-processor-gsplats.ts::buildGSplatsParams`
 *   builds `discreteDims` from `viewState.dimensions[d].discrete` (never from
 *   `slice_dims`), `workers/data-worker/projection/gsplats.ts` hands that set to
 *   `classifyHiddenDims`, and a dim in it lands in `discreteHiddenDims` and is
 *   thereby EXCLUDED from `continuousHiddenDims`. Its only visibility test is then
 *   the binary half-cell gate `|slicePos[d] − center[d]| > step × 0.5 ⇒ invisible`;
 *   the Gaussian attenuation never runs on it.
 *
 * Hence three cases, not two:
 * - **BARRIER** (in the published set): the quarter-cell
 *   {@link discreteDimTolerance}. Bounds are tight, and the quarter rather than the
 *   half is a write-side pad-budget constraint — `DISCRETE_TOLERANCE_FRACTION` argues
 *   it. Be clear about what that leaves open: against the renderer's own half-cell
 *   gate this arm keeps a `(0.25, 0.5] × step` gap, unreachable for an axis whose
 *   stored values are multiples of its declared step and reachable for one whose are
 *   not. This function does not close it — the pad budget is real, and off-grid
 *   values on a scene-declared discrete axis are a write-time authoring fault the
 *   compiler already reports (`validate_discrete_dimension_ranges`).
 * - **CONTINUOUS** (neither side calls it a barrier): the float-safety epsilon
 *   {@link gsplatsContinuousDimTolerance} — not a reach, since the chunk bounds
 *   already carry the `truncation_radius · σ` extent the projection kernel
 *   attenuates to zero at, so anything further out is pure over-fetch.
 * - **DEMOTED** (the writer omitted it, the scene declares it `discrete`): the
 *   half-cell {@link discreteDimMembershipTolerance}. Here the principle bites twice.
 *   The continuous arm's degenerate-BAND term is not a co-equal to compare against,
 *   it is IRRELEVANT — nothing about such a dim goes through the Gaussian, so
 *   `T × 1e-5` describes a band the renderer will not draw on it. And the
 *   quarter-cell's `< 0.5` pad-budget rationale provably does not apply either: a dim
 *   the writer never barrier-padded has no ±0.5 pad for a half-cell to double up
 *   with. The one window left to match is the binary gate's half-cell. This is the
 *   only case in the file where a MEMBERSHIP window doubles as a chunk-FETCH window,
 *   and it does so because for this dim the two ARE the same window.
 *
 * ## What the demote arm is worth, and how it is reached
 *
 * Without it the same dim fell to `max(1e-3 × step, T × 1e-5)` — ~500× narrower at
 * `step = 1` — while the renderer still drew a half cell, so MOST OF THE NODE
 * disappeared, though not all of it: with `slice_dims: []` the compound sort is a
 * pure spatial curve over ALL center columns
 * (`io/_ordering/compound.py::_compound_sort` takes its `argsort(spatial_codes)`
 * branch), so a chunk whose splats straddle two adjacent stacked values has a bound
 * spanning both and still matches; what is lost is every chunk sitting entirely
 * inside one value, which on a stacked timelapse is nearly all of them.
 *
 * The half-cell rather than the quarter is what covers an OFF-GRID demoted axis. The
 * query snaps to a multiple of the step (`snapDiscreteValue`, `Math.round(v/s)·s`)
 * while the splats sit where they were written, so a stored value 0.3 off a `step: 1`
 * grid (`luxar gsplat merge --as-dimension --values 0.3,1.3,2.3` writes exactly that)
 * passes the render gate at 0.3 ≤ 0.5 and matches no chunk under a 0.25-cell window
 * either — σ is 0 on a stacked axis, so its bound is the bare coordinate, with
 * neither a σ expansion nor a barrier pad. On an on-grid axis the offset is 0 and
 * even the bare epsilon matches, so the half-cell costs a half cell of reach there
 * and buys the off-grid case outright.
 *
 * THE REACHABLE PATH IS THE STANDALONE OPEN, NOT A GRAFT. Serving a `.gsplats.zarr`
 * directly (`?src=….gsplats.zarr`; the writer supports it — `save_gsplats.py` stamps
 * `layer` on the root) leaves the scene with no `scene_dimensions`, so
 * `data/scene-loader/lifecycle/load-scene.ts::synthesizeSceneDimensionsFromNode`
 * marks every axis ≥ 3 `discrete: true, step: 1` REGARDLESS of the stored values,
 * while the store publishes whatever its writer detected — `[]` for a stacked axis
 * whose values are not near-integers, which `detect_barrier_dims` rejects.
 * Scene-discrete, writer-omitted: the demote arm. GRAFTING such a store into a scene
 * does NOT reach it: `add_gsplats_from_file_impl` routes through
 * `add_gsplats_from_data_impl` / `graft_gsplat_node`, both of which write through the
 * scene gsplats writer, which RE-SORTS and RE-STAMPS `slice_dims` from
 * `geometry_writers/gsplats.py::scene_barrier_dims` (`discrete and not display`) — so
 * an axis the scene declares discrete and non-displayed comes back IN the set and
 * lands on the BARRIER arm, with the `(0.25, 0.5]` gap above intact.
 *
 * Note what this arm is and is not. It is strictly WIDER than the pre-#1655 behaviour
 * on this branch (`0.5 × step > 0.25 × step` at every step), so nothing that used to
 * be fetched stops being fetched; it is not a claim that the continuous term is being
 * respected here, and on a micro-step axis the number this returns is smaller than
 * that term (`0.5e-6` against a `2.75e-5` band at `step = 1e-6`) — deliberately, the
 * band being undrawable on this dim.
 *
 * ## The promote arm needs no floor, but NOT because it only widens
 *
 * A dim the writer listed while the scene calls it continuous goes from
 * `max(1e-3 × step, T × 1e-5)` to `0.25 × step`, and that is NARROWER whenever
 * `0.25 × step < T × 1e-5`, i.e. `step < 4e-5 × T` (≈1.1e-4 at the default `T`) — a
 * 110× narrowing at `step = 1e-6`, squarely in the micro-step regime this file
 * elsewhere refuses to cap. The conclusion still holds; the reason is the write-side
 * pad, not monotonicity. The scene flag says continuous, so the renderer DOES use the
 * Gaussian gate on this dim, and the band a degenerate one can show is `T × 1e-5`
 * (2.75e-5 by default) — while the writer, having listed the dim, padded its bounds
 * by `_BARRIER_BOUND_EPS = 1e-3` (`packages/luxar/src/luxar/io/_ordering/bounds.py`),
 * which is an ABSOLUTE pad, not a step fraction. `1e-3` dwarfs the band for any
 * `T < 100`, so a splat close enough to the slice to render already has a chunk bound
 * containing it and even a zero tolerance would match. Being absolute is exactly why
 * this argument must not be ported: it would not survive being reused for a geometry
 * whose bounds are not barrier-padded, nor (see the extreme-radius KNOWN LIMIT) for a
 * node authoring `T ≳ 100`.
 *
 * `options.truncationRadius` (the node's own `T`) scales the continuous arm's
 * degenerate-band term; absent ⇒ the default radius.
 */
function computeGSplatsHiddenTolerance(
  dimInfo: DimensionInfo | undefined,
  isBarrier: boolean,
  options: ToleranceOptions
): number {
  if (isBarrier) {
    return discreteDimTolerance(dimInfo);
  }
  // Demoted dim (writer omitted it, scene declares it discrete): the fetch window
  // must EQUAL the half-cell membership window, because that binary gate is the only
  // visibility test the projection applies to this dim — the continuous arm's
  // degenerate band is not drawn on it at all. See the docstring above. Truthiness,
  // not `=== true`, for the same reason as `isBarrierDim`: this branch must fire on
  // exactly the dims the renderer's own truthy test routes into `discreteDims`.
  if (dimInfo?.discrete) {
    return discreteDimMembershipTolerance(dimInfo);
  }
  return gsplatsContinuousDimTolerance(dimInfo, options.truncationRadius);
}

/**
 * Mesh hidden dimension tolerance.
 *
 * Barrier (discrete) dims get the **half-cell membership** rule, not the
 * quarter-cell query reach the other three default to: a mesh is whole-node
 * resident, so this slab is a per-element visibility gate applied after fetch,
 * exactly like the lines projection-clipping slab. This is the dominant real case —
 * a mesh's hidden dimensions are almost always time or channel.
 *
 * `isBarrier` always comes from `DimensionInfo.discrete` here — a mesh node publishes
 * no `slice_dims` (it has no spatial index at all), and `isBarrierDim` would ignore
 * one anyway, since both of a mesh's arms are membership gates
 * (see {@link ToleranceOptions.barrierDims}).
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
  isBarrier: boolean,
  options: ToleranceOptions
): number {
  if (isBarrier) {
    return discreteDimMembershipTolerance(dimInfo);
  }
  const slabCells = options.meshSlabTolerance ?? 1.0;
  return dimInfo?.step ? dimInfo.step * slabCells : slabCells;
}
