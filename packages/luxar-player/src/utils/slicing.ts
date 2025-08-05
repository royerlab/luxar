/**
 * Slicing utilities for nD point clouds
 */

import { SimpleDims } from '../types/dims';

/**
 * Slice nD points based on current dims state
 * Returns indices of points that match the current step in non-displayed dimensions
 *
 * @param positions - nD positions array
 * @param dims - Current dimension state
 * @param numPoints - Number of points
 * @param radii - Optional radii array for radius-based slicing
 * @param fallbackTolerance - Tolerance to use if radii not provided
 */
export function slicePoints(
  positions: Float32Array,
  dims: SimpleDims,
  numPoints: number,
  radii?: Float32Array,
  fallbackTolerance = 0.1
): Uint32Array {
  const { ndim, currentStep, displayed } = dims;
  const visibleIndices: number[] = [];

  // Create a set of displayed dimensions for quick lookup
  const displayedSet = new Set(displayed);

  // Check each point
  for (let i = 0; i < numPoints; i++) {
    let isVisible = true;

    // Get the radius for this point (or use fallback)
    const radius = radii ? radii[i] : fallbackTolerance;

    // Check non-displayed dimensions
    for (let d = 0; d < ndim; d++) {
      if (!displayedSet.has(d)) {
        const value = positions[i * ndim + d];
        const target = currentStep[d];
        const distance = Math.abs(value - target);

        // Check if point's hypersphere intersects the slice hyperplane
        // A point is visible if its distance from the slice is less than its radius
        if (distance > radius) {
          isVisible = false;
          break;
        }
      }
    }

    if (isVisible) {
      visibleIndices.push(i);
    }
  }

  return new Uint32Array(visibleIndices);
}

/**
 * Extract displayed dimensions from nD positions
 * Returns 3D positions for rendering
 */
export function extractDisplayDimensions(
  positions: Float32Array,
  indices: Uint32Array,
  dims: SimpleDims
): Float32Array {
  const { ndim, displayed } = dims;
  const numVisible = indices.length;

  // Ensure we have at most 3 displayed dimensions
  const numDisplay = Math.min(displayed.length, 3);
  const positions3D = new Float32Array(numVisible * 3);

  // Extract positions for visible points
  for (let i = 0; i < numVisible; i++) {
    const pointIndex = indices[i];

    // Copy displayed dimensions
    for (let j = 0; j < numDisplay; j++) {
      const dimIndex = displayed[j];
      positions3D[i * 3 + j] = positions[pointIndex * ndim + dimIndex];
    }

    // Fill remaining dimensions with 0
    for (let j = numDisplay; j < 3; j++) {
      positions3D[i * 3 + j] = 0;
    }
  }

  return positions3D;
}

/**
 * Extract colors for visible points
 */
export function sliceColors(colors: Uint8Array | null, indices: Uint32Array): Uint8Array | null {
  if (!colors) return null;

  const numVisible = indices.length;
  const colors3 = new Uint8Array(numVisible * 3);

  for (let i = 0; i < numVisible; i++) {
    const srcIndex = indices[i] * 3;
    const dstIndex = i * 3;
    colors3[dstIndex] = colors[srcIndex];
    colors3[dstIndex + 1] = colors[srcIndex + 1];
    colors3[dstIndex + 2] = colors[srcIndex + 2];
  }

  return colors3;
}

/**
 * Extract scalar attributes for visible points
 */
/**
 * Compute effective radius for points after slicing
 * When an nD ball is sliced by a hyperplane at distance d,
 * the resulting (n-1)D ball has radius sqrt(r² - d²)
 */
export function computeEffectiveRadii(
  positions: Float32Array,
  indices: Uint32Array,
  dims: SimpleDims,
  originalRadii: Float32Array
): Float32Array {
  const { ndim, currentStep, displayed } = dims;
  const numVisible = indices.length;
  const effectiveRadii = new Float32Array(numVisible);

  // Create a set of displayed dimensions for quick lookup
  const displayedSet = new Set(displayed);

  for (let i = 0; i < numVisible; i++) {
    const pointIndex = indices[i];
    const originalRadius = originalRadii[pointIndex];

    // Calculate sum of squared distances in non-displayed dimensions
    let sumSquaredDistances = 0;
    for (let d = 0; d < ndim; d++) {
      if (!displayedSet.has(d)) {
        const value = positions[pointIndex * ndim + d];
        const target = currentStep[d];
        const distance = value - target;
        sumSquaredDistances += distance * distance;
      }
    }

    // Compute effective radius using Pythagorean theorem
    // r_effective = sqrt(r² - d²) where d is distance from hyperplane
    const radiusSquared = originalRadius * originalRadius;
    const effectiveRadiusSquared = radiusSquared - sumSquaredDistances;

    // Ensure we don't get negative values due to numerical errors
    effectiveRadii[i] = effectiveRadiusSquared > 0 ? Math.sqrt(effectiveRadiusSquared) : 0;
  }

  return effectiveRadii;
}

export function sliceScalarAttribute(
  attribute: Float32Array | null,
  indices: Uint32Array
): Float32Array | null {
  if (!attribute) return null;

  const numVisible = indices.length;
  const sliced = new Float32Array(numVisible);

  for (let i = 0; i < numVisible; i++) {
    sliced[i] = attribute[indices[i]];
  }

  return sliced;
}

/**
 * Get a human-readable label for the current dims state
 */
export function getDimsLabel(dims: SimpleDims): string {
  const { currentStep, displayed, metadata } = dims;
  const labels: string[] = [];

  for (let d = 0; d < dims.ndim; d++) {
    if (!displayed.includes(d)) {
      const name = metadata?.[d]?.name || `D${d}`;
      const value = currentStep[d].toFixed(2);
      const unit = metadata?.[d]?.unit || '';
      labels.push(`${name}=${value}${unit ? unit : ''}`);
    }
  }

  return labels.join(', ');
}
