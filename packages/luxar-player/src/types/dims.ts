/**
 * Simple dimension types for nD data visualization
 */

/**
 * Metadata for a single dimension
 */
export interface DimensionMetadata {
  name: string;
  unit: string;
  scale: number;
  range?: [number, number];
}

/**
 * Simple dims state for slicing nD data
 */
export interface SimpleDims {
  /** Total number of dimensions */
  ndim: number;
  
  /** Current position in each dimension */
  currentStep: number[];
  
  /** Which dimensions are displayed (indices) */
  displayed: number[];
  
  /** Optional metadata for each dimension */
  metadata?: DimensionMetadata[];
}

/**
 * Helper to initialize dims from positions array
 */
export function initializeDims(
  numPoints: number,
  totalElements: number,
  metadata?: DimensionMetadata[]
): SimpleDims {
  // Calculate number of dimensions from array size
  const ndim = totalElements / numPoints;
  
  if (!Number.isInteger(ndim)) {
    throw new Error(`Invalid positions array: ${totalElements} elements for ${numPoints} points`);
  }
  
  // Initialize at origin
  const currentStep = new Array(ndim).fill(0);
  
  // Display last 3 dimensions (or fewer if ndim < 3)
  const displayed = ndim <= 3 
    ? Array.from({ length: ndim }, (_, i) => i)
    : [ndim - 3, ndim - 2, ndim - 1];
  
  return {
    ndim,
    currentStep,
    displayed,
    metadata
  };
}

/**
 * Get dimension ranges from positions data
 */
export function getDimensionRanges(
  positions: Float32Array,
  ndim: number,
  numPoints: number
): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  
  // Initialize ranges
  for (let d = 0; d < ndim; d++) {
    ranges.push([Infinity, -Infinity]);
  }
  
  // Find min/max for each dimension
  for (let i = 0; i < numPoints; i++) {
    for (let d = 0; d < ndim; d++) {
      const value = positions[i * ndim + d];
      ranges[d][0] = Math.min(ranges[d][0], value);
      ranges[d][1] = Math.max(ranges[d][1], value);
    }
  }
  
  return ranges;
}