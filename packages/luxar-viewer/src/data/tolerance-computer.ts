/**
 * Unified tolerance computation for spatial queries across all geometry types.
 *
 * Extracted from SceneLoader and consolidates the tolerance logic scattered
 * across multiple modules (effective-radius-calculator, lines-chunk-spatial-index,
 * gsplats-chunk-spatial-index, spatial-query-builder).
 *
 * Each geometry type has slightly different requirements for non-displayed dimensions:
 * - **Points**: Uses maxRadius for spatial dims, 0.5 for discrete dims
 * - **Lines**: 0 for spatial dims (segment bounds already include width), step/2 for discrete
 * - **GSplats**: step * defaultTolerance (3 sigma) for spatial, 0.5 for discrete
 *
 * Displayed dimensions always get infinite tolerance (1e10) across all types.
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
 * Compute per-dimension query tolerances for a spatial index query.
 *
 * This is the single entry point that replaces:
 * - `calculateSpatialQueryTolerance` (effective-radius-calculator.ts)
 * - `computeLinesTolerance` (lines-chunk-spatial-index.ts)
 * - `computeGSplatsTolerance` (gsplats-chunk-spatial-index.ts)
 * - `computeQueryTolerance` (loaders/spatial-query-builder.ts)
 *
 * @param geometryType - Geometry type determines the strategy for hidden dimensions
 * @param displayDims - Indices of the currently displayed (rendered) dimensions
 * @param ndim - Total number of dimensions
 * @param dimensions - Optional per-dimension metadata (step, discrete flag)
 * @param options - Additional options (maxRadius for points, gsplatsDefaultTolerance)
 * @returns Tolerance array of length ndim
 */
export function computeTolerance(
  geometryType: GeometryType,
  displayDims: number[],
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
 * Spatial dimensions use maxRadius so points whose radius intersects the slice
 * are loaded. Discrete dimensions use 0.5 for floating-point safety.
 * Mirrors `calculateSpatialQueryTolerance` from effective-radius-calculator.ts.
 */
function computePointsHiddenTolerance(
  dimIndex: number,
  _dimInfo: DimensionInfo | undefined,
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
  return 0.5;
}

/**
 * Lines hidden dimension tolerance.
 *
 * Spatial dimensions get 0 because segment bounding boxes already include
 * the line width extent. Discrete dimensions use step/2 (or 0.5 fallback).
 * Mirrors `computeLinesTolerance` from lines-chunk-spatial-index.ts.
 */
function computeLinesHiddenTolerance(dimInfo: DimensionInfo | undefined): number {
  if (dimInfo?.discrete) {
    if (dimInfo.step !== undefined && dimInfo.step !== null) {
      return dimInfo.step / 2;
    }
    return 0.5;
  }
  // Spatial dimension: bounds already include width
  return 0;
}

/**
 * GSplats hidden dimension tolerance.
 *
 * Discrete dimensions use 0.5. Continuous dimensions use step * defaultTolerance
 * (default 3.0 = 3 sigma of the Gaussian), falling back to defaultTolerance alone.
 * Mirrors `computeGSplatsTolerance` from gsplats-chunk-spatial-index.ts.
 */
function computeGSplatsHiddenTolerance(
  dimInfo: DimensionInfo | undefined,
  options: ToleranceOptions
): number {
  const defaultTol = options.gsplatsDefaultTolerance ?? 3.0;

  if (dimInfo?.discrete) {
    return 0.5;
  }
  if (dimInfo?.step) {
    return dimInfo.step * defaultTol;
  }
  return defaultTol;
}
