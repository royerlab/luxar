/**
 * Advanced nD slicing algorithms for high-dimensional point cloud visualization.
 *
 * This module implements sophisticated slicing techniques that allow users to
 * navigate through nD datasets by fixing positions in non-displayed dimensions
 * while visualizing the remaining 3D projection.
 *
 * Key algorithms:
 * - Radius-based hypersphere slicing for continuous dimensions
 * - Exact matching for discrete dimensions
 * - Effective radius calculation for sliced nD spheres
 * - Efficient point filtering and attribute extraction
 */

import { SimpleDims } from '../types/dims';

/**
 * Performs nD slicing to determine which points are visible at the current position.
 *
 * This is the core slicing algorithm that implements radius-based hypersphere intersection
 * for continuous dimensions and exact matching for discrete dimensions. The mathematical
 * foundation is:
 *
 * For a point with nD position P and radius R:
 * - The point is visible if its nD hypersphere intersects the current slice hyperplane
 * - Distance from hyperplane = √(Σ(Pi - Ci)²) where C is current position in non-displayed dims
 * - Point is visible if distance ≤ radius
 *
 * Design decisions:
 * - Discrete dimensions require exact matches (important for categorical data like time points)
 * - Continuous dimensions use radius-based inclusion (smooth navigation through space)
 * - Efficient early termination when point is clearly outside slice
 *
 * @param positions - Flattened nD positions array (size: numPoints * ndim)
 * @param dims - Current dimension state with slice positions
 * @param numPoints - Total number of points in the dataset
 * @param radii - Per-point radii for hypersphere slicing (optional)
 * @param fallbackTolerance - Default tolerance when no per-point radii provided
 * @returns Indices of points visible in the current slice
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

  // Pre-compute displayed dimensions set for O(1) lookup
  const displayedSet = new Set(displayed);

  // Iterate through all points to test visibility
  for (let i = 0; i < numPoints; i++) {
    let isVisible = true;

    // Get per-point radius or use fallback tolerance
    const radius = radii ? radii[i] : fallbackTolerance;

    // Test visibility in all non-displayed dimensions
    for (let d = 0; d < ndim; d++) {
      if (!displayedSet.has(d)) {
        const value = positions[i * ndim + d];
        const target = currentStep[d];
        const dimMeta = dims.metadata?.[d];
        const isDiscrete = dimMeta?.discrete || false;

        if (isDiscrete) {
          // Discrete dimensions: require exact match (e.g., time frame 5 vs 6)
          // Use small epsilon for floating-point comparison safety
          if (Math.abs(value - target) > 1e-6) {
            isVisible = false;
            break; // Early termination for efficiency
          }
        } else {
          // Continuous dimensions: radius-based hypersphere intersection
          const distance = Math.abs(value - target);
          // Mathematical insight: point's nD sphere intersects slice plane if distance ≤ radius
          // This creates smooth transitions when navigating through continuous space
          if (distance > radius) {
            isVisible = false;
            break; // Early termination for efficiency
          }
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
 * Projects nD point positions onto the currently displayed 3D coordinate system.
 *
 * This function extracts the subset of dimensions that are being visualized
 * and creates a 3D positions array suitable for GPU rendering. The projection
 * maintains the spatial relationships within the selected dimensions.
 *
 * Design decisions:
 * - Maximum of 3 displayed dimensions (hardware/perceptual limitation)
 * - Missing dimensions are padded with zeros for consistent GPU buffer layout
 * - Preserves point order from the slicing operation for attribute alignment
 *
 * @param positions - Original nD positions array
 * @param indices - Indices of visible points from slicing operation
 * @param dims - Dimension state specifying which dimensions to display
 * @returns 3D positions array ready for GPU rendering (size: numVisible * 3)
 */
export function extractDisplayDimensions(
  positions: Float32Array,
  indices: Uint32Array,
  dims: SimpleDims
): Float32Array {
  const { ndim, displayed } = dims;
  const numVisible = indices.length;

  // GPU rendering requires exactly 3 coordinates per point
  const numDisplay = Math.min(displayed.length, 3);
  const positions3D = new Float32Array(numVisible * 3);

  // Project each visible point onto the 3D display coordinate system
  for (let i = 0; i < numVisible; i++) {
    const pointIndex = indices[i];

    // Extract coordinates from displayed dimensions
    for (let j = 0; j < numDisplay; j++) {
      const dimIndex = displayed[j];
      positions3D[i * 3 + j] = positions[pointIndex * ndim + dimIndex];
    }

    // Pad with zeros for consistent 3D layout (e.g., 2D data gets z=0)
    for (let j = numDisplay; j < 3; j++) {
      positions3D[i * 3 + j] = 0;
    }
  }

  return positions3D;
}

/**
 * Extracts RGB color data for points that survived the slicing operation.
 *
 * This function maintains the color-to-point correspondence after slicing
 * by reordering the color array to match the filtered point indices.
 *
 * @param colors - Original RGB color array (3 bytes per point) or null
 * @param indices - Indices of visible points from slicing
 * @returns Reordered color array matching visible points, or null if no colors
 */
export function sliceColors(colors: Uint8Array | null, indices: Uint32Array): Uint8Array | null {
  if (!colors) return null;

  const numVisible = indices.length;
  const colors3 = new Uint8Array(numVisible * 3);

  // Copy RGB triplets for each visible point
  for (let i = 0; i < numVisible; i++) {
    const srcIndex = indices[i] * 3;
    const dstIndex = i * 3;
    colors3[dstIndex] = colors[srcIndex]; // R
    colors3[dstIndex + 1] = colors[srcIndex + 1]; // G
    colors3[dstIndex + 2] = colors[srcIndex + 2]; // B
  }

  return colors3;
}

/**
 * Slices HDR float32 color data to include only visible points.
 * 
 * @param colors - HDR float32 color array (values can exceed 1.0)
 * @param indices - Indices of visible points
 * @returns Sliced HDR color array
 */
export function sliceColorsFloat32(colors: Float32Array | null, indices: Uint32Array): Float32Array | null {
  if (!colors) return null;

  const numVisible = indices.length;
  const colors3 = new Float32Array(numVisible * 3);

  // Copy RGB triplets for each visible point - HDR values preserved
  for (let i = 0; i < numVisible; i++) {
    const srcIndex = indices[i] * 3;
    const dstIndex = i * 3;
    colors3[dstIndex] = colors[srcIndex]; // R (can be > 1.0 for HDR)
    colors3[dstIndex + 1] = colors[srcIndex + 1]; // G (can be > 1.0 for HDR)
    colors3[dstIndex + 2] = colors[srcIndex + 2]; // B (can be > 1.0 for HDR)
  }

  return colors3;
}

/**
 * Computes the effective radii of nD hyperspheres after slicing by hyperplanes.
 *
 * This implements a fundamental geometric principle: when an nD hypersphere of radius R
 * is intersected by a hyperplane at distance D from its center, the resulting
 * (n-1)D cross-section has radius √(R² - D²).
 *
 * Mathematical foundation:
 * - Original nD sphere: x₁² + x₂² + ... + xₙ² ≤ R²
 * - Slice at fixed positions in non-displayed dims: xᵢ = cᵢ for i ∉ displayed
 * - Effective radius in displayed dims: R_eff = √(R² - Σ(xᵢ - cᵢ)²) for i ∉ displayed
 *
 * This is crucial for:
 * - Accurate visual representation of point sizes after slicing
 * - Maintaining proper spatial relationships in the projected view
 * - Preventing visual artifacts when navigating through nD space
 *
 * @param positions - Original nD positions array
 * @param indices - Indices of visible points after slicing
 * @param dims - Current dimension state with slice positions
 * @param originalRadii - Original nD radii before slicing
 * @returns Effective radii for rendering in the displayed dimensions
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

  // Pre-compute displayed dimensions set for efficient lookup
  const displayedSet = new Set(displayed);

  for (let i = 0; i < numVisible; i++) {
    const pointIndex = indices[i];
    const originalRadius = originalRadii[pointIndex];

    // Calculate Euclidean distance from slice hyperplane in non-displayed dimensions
    let sumSquaredDistances = 0;
    for (let d = 0; d < ndim; d++) {
      if (!displayedSet.has(d)) {
        const value = positions[pointIndex * ndim + d];
        const target = currentStep[d];
        const distance = value - target;
        sumSquaredDistances += distance * distance;
      }
    }

    // Apply Pythagorean theorem: R_effective = √(R² - D²)
    const radiusSquared = originalRadius * originalRadius;
    const effectiveRadiusSquared = radiusSquared - sumSquaredDistances;

    // Clamp to zero to handle numerical precision issues
    // (points at the boundary of the hypersphere)
    effectiveRadii[i] = effectiveRadiusSquared > 0 ? Math.sqrt(effectiveRadiusSquared) : 0;
  }

  return effectiveRadii;
}

/**
 * Extracts scalar attribute values for points that survived slicing.
 *
 * This utility maintains attribute-to-point correspondence after filtering,
 * ensuring that additional per-point data (e.g., intensity, confidence, etc.)
 * remains aligned with the filtered point set.
 *
 * @param attribute - Original scalar attribute array (one value per point) or null
 * @param indices - Indices of visible points from slicing operation
 * @returns Filtered attribute array matching visible points, or null if no attributes
 */
export function sliceScalarAttribute(
  attribute: Float32Array | null,
  indices: Uint32Array
): Float32Array | null {
  if (!attribute) return null;

  const numVisible = indices.length;
  const sliced = new Float32Array(numVisible);

  // Copy scalar values for each visible point
  for (let i = 0; i < numVisible; i++) {
    sliced[i] = attribute[indices[i]];
  }

  return sliced;
}

/**
 * Generates a human-readable label describing the current nD slice position.
 *
 * This creates informative text for the UI that shows users exactly where they
 * are positioned in the non-displayed dimensions. Essential for understanding
 * context when navigating through complex nD datasets.
 *
 * Example output: "Time=5.2s, Channel=1, Depth=12.5μm"
 *
 * @param dims - Current dimension state
 * @returns Formatted string describing slice position in non-displayed dimensions
 */
export function getDimsLabel(dims: SimpleDims): string {
  const { currentStep, displayed, metadata } = dims;
  const labels: string[] = [];

  for (let d = 0; d < dims.ndim; d++) {
    if (!displayed.includes(d)) {
      // Use metadata names when available, otherwise fallback to generic labels
      const name = metadata?.[d]?.name || `D${d}`;
      const value = currentStep[d].toFixed(2);
      const unit = metadata?.[d]?.unit || '';
      labels.push(`${name}=${value}${unit ? unit : ''}`);
    }
  }

  return labels.join(', ');
}
