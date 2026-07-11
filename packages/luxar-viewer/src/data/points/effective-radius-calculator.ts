/**
 * Effective radius calculation for nD hypersphere slicing.
 *
 * This module implements the mathematical calculations for determining
 * the visible radius of nD hyperspheres when intersected by a hyperplane.
 * Points only extend through dimensions marked as spatial.
 */

import { ViewState } from '../data-loader-types';
import type { EffectiveRadiusConfig } from '../../types/points';
import { discreteDimTolerance } from '../loaders/spatial-query/tolerance-computer';

// Re-export so existing importers (`import { EffectiveRadiusConfig } from
// '.../data/points/effective-radius-calculator'`) keep resolving.
export type { EffectiveRadiusConfig };

/**
 * Calculate effective radii for points based on their distance from the slice hyperplane.
 *
 * When an nD hypersphere of radius R is intersected by a hyperplane at distance D
 * from its center, the resulting (n-1)D cross-section has radius:
 * R_effective = √(R² - D²)
 *
 * This calculation considers:
 * - Displayed dimensions: Points always extend fully (they're in the viewing plane)
 * - Non-displayed spatial dimensions: Apply the Pythagorean theorem
 * - Non-displayed discrete dimensions: Exact match required (zero radius if mismatch)
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

  // MEMBERSHIP gate for discrete dimensions: a point belongs to the queried
  // category iff |value − target| ≤ 0.5. This is deliberately DIFFERENT from
  // the chunk-QUERY tolerance (quarter-cell, see calculateSpatialQueryTolerance
  // below): the query decides which chunks to fetch; this decides which of the
  // fetched points are visible. Keep at 0.5 — the viewer snaps discrete
  // navigation TARGETS to the absolute k·step grid
  // (SceneDimsManager.setDimensionValue), and the compiler warns when discrete
  // DATA sits more than a quarter-step off that grid
  // (validate_discrete_dimension_ranges' on-grid check), so for conforming
  // data a half-unit reach selects exactly the target category. Off-grid data
  // in the (quarter-step, half-step] band would pass this gate without its
  // chunks being fetched — that's the contract the compile-time warning
  // guards. Mirrors the fixed 0.5 in the WASM parity kernel
  // (wasm/typescript/effective-radii.ts) — keep the two in 1:1 sync.
  const discreteTolerance = 0.5;

  // Helper function to safely check if a dimension is spatial
  // If spatialExtendDims doesn't cover this dimension, default to true (spatial)
  // This is the safer default as it won't unexpectedly filter out points
  const isSpatialDim = (d: number): boolean => {
    if (d >= spatialExtendDims.length) {
      // Dimension not covered by config - default to spatial (more permissive)
      return true;
    }
    return spatialExtendDims[d];
  };

  // MED-13: Precompute per-dimension invariants OUTSIDE the per-point loop.
  // displayDims.includes(d) was O(displayDims.length) per check; tolerance,
  // slicePosition, and isSpatialDim are also dimension-only (independent
  // of i). Precomputing them once turns the inner work from
  // O(ndim * displayDims.length * numPoints) into O(ndim * numPoints).
  const isDisplayDim = new Array<boolean>(ndim);
  const isExtendToAll = new Array<boolean>(ndim);
  const isSpatial = new Array<boolean>(ndim);
  const targetPos = new Array<number>(ndim);
  for (let d = 0; d < ndim; d++) {
    isDisplayDim[d] = displayDims.includes(d);
    isExtendToAll[d] = viewState.tolerance[d] >= 1e9;
    isSpatial[d] = isSpatialDim(d);
    targetPos[d] = slicePosition[d] ?? 0;
  }

  for (let i = 0; i < numPoints; i++) {
    const originalRadius = radii[i];
    const base = i * ndim;

    // First check discrete dimensions for exact match
    let discreteMatch = true;

    for (let d = 0; d < ndim; d++) {
      // Skip displayed dimensions (they're in the viewing plane)
      if (isDisplayDim[d]) {
        continue;
      }

      // Skip extend_to_all dimensions (tolerance >= 1e9) — always visible
      if (isExtendToAll[d]) {
        continue;
      }

      // For non-spatial (discrete) dimensions, require exact match
      if (!isSpatial[d]) {
        const value = positions[base + d];
        const target = targetPos[d];
        // Use tolerance for floating point comparison
        if (Math.abs(value - target) > discreteTolerance) {
          discreteMatch = false;
          break;
        }
      }
    }

    // If discrete dimensions don't match, point is invisible
    if (!discreteMatch) {
      effectiveRadii[i] = 0;
      continue;
    }

    // Calculate distance ONLY in non-displayed spatial dimensions
    let sumSquaredDistances = 0;

    for (let d = 0; d < ndim; d++) {
      // Skip if dimension is displayed (it's in the viewing plane)
      if (isDisplayDim[d]) {
        continue;
      }

      // Skip extend_to_all dimensions — no distance contribution
      if (isExtendToAll[d]) {
        continue;
      }

      // Skip if dimension is not spatial (already handled above)
      if (!isSpatial[d]) {
        continue;
      }

      // This is a non-displayed spatial dimension - calculate distance
      const value = positions[base + d];
      const target = targetPos[d];
      const distance = value - target;
      sumSquaredDistances += distance * distance;
    }

    // Apply Pythagorean theorem: R_eff = √(R² - D²)
    const radiusSquared = originalRadius * originalRadius;
    const effectiveRadiusSquared = radiusSquared - sumSquaredDistances;

    // Clamp to zero for numerical stability (points at hypersphere boundary).
    // Use `>= 0` so the exact-boundary case (D == R, where R² - D² == 0)
    // returns 0 from the sqrt path rather than falling into the fallback;
    // this also produces the mathematically correct R_eff = 0 instead of
    // implicitly relying on the fallback for the boundary case.
    effectiveRadii[i] = effectiveRadiusSquared >= 0 ? Math.sqrt(effectiveRadiusSquared) : 0;
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

  // Helper function to safely check if a dimension is spatial
  // If spatialExtendDims doesn't cover this dimension, default to true (spatial)
  // This is the safer default as it uses maxRadius tolerance rather than 0
  const isSpatialDim = (d: number): boolean => {
    if (d >= spatialExtendDims.length) {
      // Dimension not covered by config - default to spatial (more permissive)
      return true;
    }
    return spatialExtendDims[d];
  };

  for (let d = 0; d < ndim; d++) {
    if (displayDims.includes(d)) {
      // Displayed dimensions need INFINITE tolerance - we want to see ALL points
      // regardless of their position in these dimensions (they're all in the view)
      // Use a very large number instead of Infinity for numerical stability
      queryTolerance[d] = 1e10;
    } else if (viewState.tolerance[d] >= 1e9) {
      // extend_to_all dimension — use infinite tolerance so all chunks are loaded
      queryTolerance[d] = 1e10;
    } else if (isSpatialDim(d)) {
      // Non-displayed spatial dimensions need maxRadius tolerance
      // to catch all points that might intersect the slice
      // ALWAYS use maxRadius for spatial dimensions, ignore tolerance array
      queryTolerance[d] = maxRadius;
    } else {
      // Non-spatial dimensions are discrete (by design): the shared
      // quarter-cell rule (0.25 × step, fallback 0.25) used by all three
      // geometries. Deliberately < 0.5 × step: chunk bounds are padded on the
      // write side (±0.5 in legacy datasets), and a half-step query tolerance
      // would sum with that padding to a full step — fetching the ENTIRE
      // neighbouring category (e.g. all of timepoint t−1 when scrubbing to t).
      // A quarter-cell still always catches the target cell and genuine
      // straddle chunks (discrete navigation is on-grid). The precise
      // per-point filtering happens later in calculateEffectiveRadii with the
      // half-step MEMBERSHIP gate. Using 0 here would miss chunks at float
      // boundaries.
      queryTolerance[d] = discreteDimTolerance(viewState.dimensions?.[d]);
    }
  }

  return queryTolerance;
}

/**
 * Check if effective radius calculation should be applied.
 *
 * Returns true if:
 * - There are non-displayed spatial dimensions (need distance-based filtering)
 * - There are non-displayed discrete dimensions (need exact-match filtering)
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
  displayDims: readonly number[],
  hasRadii: boolean
): boolean {
  if (!config || !hasRadii) {
    return false;
  }

  const { spatialExtendDims } = config;

  for (let d = 0; d < spatialExtendDims.length; d++) {
    // Skip displayed dimensions (they're in the viewing plane)
    if (displayDims.includes(d)) {
      continue;
    }

    // Found a non-displayed dimension (either spatial or discrete)
    // Spatial dims need distance-based filtering (Pythagorean)
    // Discrete dims need exact-match filtering (zero radius if mismatch)
    return true;
  }

  // All dimensions are displayed, no filtering needed
  return false;
}
