/**
 * Slicing utilities for nD point clouds
 */

import { SimpleDims } from '../types/dims';

/**
 * Slice nD points based on current dims state
 * Returns indices of points that match the current step in non-displayed dimensions
 */
export function slicePoints(
  positions: Float32Array,
  dims: SimpleDims,
  numPoints: number,
  tolerance = 0.5
): Uint32Array {
  const { ndim, currentStep, displayed } = dims;
  const visibleIndices: number[] = [];
  
  // Create a set of displayed dimensions for quick lookup
  const displayedSet = new Set(displayed);
  
  // Check each point
  for (let i = 0; i < numPoints; i++) {
    let isVisible = true;
    
    // Check non-displayed dimensions
    for (let d = 0; d < ndim; d++) {
      if (!displayedSet.has(d)) {
        const value = positions[i * ndim + d];
        const target = currentStep[d];
        
        // Check if point is within tolerance of current step
        if (Math.abs(value - target) > tolerance) {
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
export function sliceColors(
  colors: Uint8Array | null,
  indices: Uint32Array
): Uint8Array | null {
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