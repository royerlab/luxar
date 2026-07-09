/**
 * Canonical tolerance computer used by all geometry-specific spatial-index loaders.
 *
 * Each geometry type has slightly different requirements for hidden (non-displayed)
 * dimensions. Discrete dims share ONE rule across all three geometries
 * ({@link discreteDimTolerance}); only the spatial/continuous branch differs:
 * - **Points**: `maxRadius` for spatial dims (selected via the `spatialExtendDims`
 *   option, the per-dimension flag array carried by `EffectiveRadiusConfig`).
 * - **Lines**: 0 for spatial dims (segment bounding boxes already include line
 *   width extent).
 * - **GSplats**: `step × gsplatsDefaultTolerance` (default 3σ) for continuous dims.
 *
 * Discrete dims (all geometries): a "quarter-cell" `0.25 × step`. This is
 * deliberately `< 0.5 × step`: a query on category `k` must not reach the `k±1`
 * cell even though chunk bounds are padded on the write side (see
 * `io/ordering.py` barrier padding). The two half-steps (pad + tolerance) would
 * otherwise sum to a full step and bleed the entire neighbouring category (e.g.
 * loading timepoint `t−1` in full when scrubbing to `t`). A quarter-cell still
 * comfortably catches the target cell and genuine straddle chunks, and sits
 * inside the projection stage's `0.5 × step` membership rule.
 *
 * Displayed dimensions always get infinite tolerance (1e10) regardless of type.
 *
 * Called from `SpatialQueryBuilder` (geometry-aware path) and directly from
 * `scene-loader.ts` (lines projection clipping path).
 *
 * @module data/tolerance-computer
 */

/** Geometry types supported by the tolerance computer. */
export type GeometryType = 'points' | 'lines' | 'gsplats';

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
   * Flags per dimension indicating whether it is spatial (true) or discrete (false).
   * Only used for 'points' type. Dimensions beyond the array length default to spatial.
   */
  spatialExtendDims?: boolean[];
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
 */
const DISCRETE_TOLERANCE_FRACTION = 0.25;

/**
 * Canonical discrete-dimension query tolerance, shared by points/lines/gsplats.
 * `0.25 × step` (fallback quarter-cell of a unit step when no step metadata).
 */
function discreteDimTolerance(dimInfo: DimensionInfo | undefined): number {
  const step =
    dimInfo?.step !== undefined && dimInfo.step !== null && dimInfo.step > 0 ? dimInfo.step : 1;
  return DISCRETE_TOLERANCE_FRACTION * step;
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
  geometryType: GeometryType,
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
  geometryType: GeometryType,
  dimIndex: number,
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  switch (geometryType) {
    case 'points':
      return computePointsHiddenTolerance(dimIndex, dimInfo, options);
    case 'lines':
      return computeLinesHiddenTolerance(dimInfo);
    case 'gsplats':
      return computeGSplatsHiddenTolerance(dimInfo, options);
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
 * the line width extent. Discrete dimensions use the shared quarter-cell rule.
 */
function computeLinesHiddenTolerance(dimInfo: DimensionInfo | undefined): number {
  if (dimInfo?.discrete) {
    return discreteDimTolerance(dimInfo);
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
