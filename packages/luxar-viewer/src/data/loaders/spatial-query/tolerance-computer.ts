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
 * - **GSplats**: `step × gsplatsDefaultTolerance` (default 3σ) for continuous dims.
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
   * Default tolerance multiplier for gsplats hidden spatial dimensions.
   * Tolerance = step * defaultTolerance (captures N sigma of the Gaussian).
   * @default 3.0
   */
  gsplatsDefaultTolerance?: number;

  /**
   * Slab thickness multiplier for mesh hidden CONTINUOUS dimensions, in cells.
   * Tolerance = `step * meshSlabTolerance`.
   *
   * The mesh sibling of {@link gsplatsDefaultTolerance}, but it means something
   * different: a gsplat's `3.0` captures 3σ of a real per-element extent, whereas
   * a mesh has no extent, so this number *invents* a slab thickness. One cell is
   * the neutral choice — it admits a triangle whose vertices straddle the slice by
   * up to a voxel. It must never be `0`: see the module docstring.
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
 * Compute per-dimension query tolerances for a spatial-index query.
 *
 * @param geometryType - Geometry type determines the strategy for hidden dimensions.
 * @param displayDims - Indices of the currently displayed (rendered) dimensions.
 * @param ndim - Total number of dimensions.
 * @param dimensions - Optional per-dimension metadata (step, discrete flag).
 * @param options - Additional options (maxRadius for points, gsplatsDefaultTolerance, etc.).
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
      return computeGSplatsHiddenTolerance(dimInfo, options);
    case 'mesh':
      return computeMeshHiddenTolerance(dimInfo, options);
  }
}

/**
 * Points hidden dimension tolerance.
 *
 * Spatial dimensions use `maxRadius` so points whose radius intersects the slice
 * are loaded. Discrete dimensions use the shared quarter-cell rule.
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
 * Discrete dimensions use the shared quarter-cell rule. Continuous dimensions use
 * `step × defaultTolerance` (default 3.0 = 3 σ of the Gaussian), falling back to
 * `defaultTolerance` alone.
 */
function computeGSplatsHiddenTolerance(
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  const defaultTol = options.gsplatsDefaultTolerance ?? 3.0;

  if (dimInfo?.discrete) {
    return discreteDimTolerance(dimInfo);
  }
  if (dimInfo?.step) {
    return dimInfo.step * defaultTol;
  }
  return defaultTol;
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
