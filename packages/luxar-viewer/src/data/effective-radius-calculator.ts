/**
 * Effective radius calculation for nD hypersphere slicing.
 *
 * This module implements the mathematical calculations for determining
 * the visible radius of nD hyperspheres when intersected by a hyperplane.
 * Points only extend through dimensions marked as spatial.
 */

import { ViewState } from './data-loader-types';

/**
 * Configuration for effective radius calculation.
 */
export interface EffectiveRadiusConfig {
  /** Which dimensions points extend through spatially */
  spatialExtendDims: boolean[];
  /** Maximum radius in the dataset for query optimization */
  maxRadius: number;
  /** Which dimensions are discrete (for exact matching) - DEPRECATED: non-spatial is always discrete */
  discreteDims?: boolean[];
}

/**
 * Calculate effective radii for points based on their distance from the slice hyperplane.
 *
 * When an nD hypersphere of radius R is intersected by a hyperplane at distance D
 * from its center, the resulting (n-1)D cross-section has radius:
 * R_effective = √(R² - D²)
 *
 * This calculation only considers dimensions marked as spatial, meaning:
 * - Displayed dimensions: Points always extend fully (they're in the viewing plane)
 * - Non-displayed spatial dimensions: Apply the Pythagorean theorem
 * - Non-spatial dimensions (time, categories): No extension, no contribution to distance
 *
 * @param positions - Original nD positions (any typed array, flattened)
 * @param radii - Original radii for each point (any typed array)
 * @param viewState - Current slice position and display configuration
 * @param config - Spatial extension configuration
 * @param ndim - Number of dimensions
 * @returns Array of effective radii for rendering (always Float32Array)
 */
export function calculateEffectiveRadii(
  positions: ArrayLike<number>,
  radii: ArrayLike<number>,
  viewState: ViewState,
  config: EffectiveRadiusConfig,
  ndim: number
): Float32Array {
  const { displayDims, slicePosition } = viewState;
  const { spatialExtendDims } = config;
  const numPoints = positions.length / ndim;

  const effectiveRadii = new Float32Array(numPoints);

  for (let i = 0; i < numPoints; i++) {
    const originalRadius = radii[i];

    // Calculate distance ONLY in non-displayed spatial dimensions
    let sumSquaredDistances = 0;

    for (let d = 0; d < ndim; d++) {
      // Skip if dimension is displayed (it's in the viewing plane)
      if (displayDims.includes(d)) {
        continue;
      }

      // Skip if dimension is not spatial (no extension through it)
      if (!spatialExtendDims[d]) {
        continue;
      }

      // This is a non-displayed spatial dimension - calculate distance
      const value = positions[i * ndim + d];
      const target = slicePosition[d] || 0;
      const distance = value - target;
      sumSquaredDistances += distance * distance;
    }

    // Apply Pythagorean theorem: R_eff = √(R² - D²)
    const radiusSquared = originalRadius * originalRadius;
    const effectiveRadiusSquared = radiusSquared - sumSquaredDistances;

    // Clamp to zero for numerical stability (points at hypersphere boundary)
    effectiveRadii[i] = effectiveRadiusSquared > 0 ? Math.sqrt(effectiveRadiusSquared) : 0;
  }

  return effectiveRadii;
}

/**
 * Calculate query tolerance for spatial index based on spatial extension.
 *
 * The tolerance determines how far from the current slice position we need
 * to query for potentially visible points:
 * - Spatial dimensions: Use maxRadius (points extend through these)
 * - Non-spatial dimensions: Use minimal tolerance (essentially exact match)
 * - Displayed dimensions: No tolerance needed (they're in the view plane)
 *
 * @param viewState - Current view configuration
 * @param config - Spatial extension configuration
 * @param ndim - Number of dimensions
 * @returns Array of tolerances for each dimension
 */
export function calculateSpatialQueryTolerance(
  viewState: ViewState,
  config: EffectiveRadiusConfig,
  ndim: number
): number[] {
  const { displayDims } = viewState;
  const { spatialExtendDims, maxRadius } = config;

  const queryTolerance = new Array(ndim).fill(0);

  for (let d = 0; d < ndim; d++) {
    if (displayDims.includes(d)) {
      // Displayed dimensions need INFINITE tolerance - we want to see ALL points
      // regardless of their position in these dimensions (they're all in the view)
      // Use a very large number instead of Infinity for numerical stability
      queryTolerance[d] = 1e10;
    } else if (spatialExtendDims[d]) {
      // Non-displayed spatial dimensions need maxRadius tolerance
      // to catch all points that might intersect the slice
      // ALWAYS use maxRadius for spatial dimensions, ignore tolerance array
      queryTolerance[d] = maxRadius;
    } else {
      // Non-spatial dimensions are always discrete (by design)
      // Use 0 tolerance for exact matching
      queryTolerance[d] = 0;
    }
  }

  return queryTolerance;
}

/**
 * Check if effective radius calculation should be applied.
 *
 * Returns true if:
 * - There are non-displayed spatial dimensions
 * - Radii data is available
 * - Configuration is valid
 *
 * @param config - Spatial extension configuration
 * @param displayDims - Currently displayed dimensions
 * @param hasRadii - Whether radii data is available
 * @returns Whether to apply effective radius calculation
 */
export function shouldApplyEffectiveRadius(
  config: EffectiveRadiusConfig | null,
  displayDims: number[],
  hasRadii: boolean
): boolean {
  if (!config || !hasRadii) {
    return false;
  }

  // Check if there are any non-displayed spatial dimensions
  const { spatialExtendDims } = config;

  for (let d = 0; d < spatialExtendDims.length; d++) {
    // Found a non-displayed spatial dimension
    if (spatialExtendDims[d] && !displayDims.includes(d)) {
      return true;
    }
  }

  // All spatial dimensions are displayed, no need for effective radius
  return false;
}
