/**
 * Pure utility functions for Zarr data loading and processing
 *
 * This module contains side-effect-free functions extracted from zarr-loader.ts
 * to improve testability and maintainability. These functions handle data
 * transformation, dimension processing, and attribute management without
 * external dependencies.
 *
 * ## Architecture Benefits:
 * - **Testability**: Pure functions with predictable inputs/outputs
 * - **Reusability**: Can be used by multiple loading strategies
 * - **Maintainability**: Business logic separated from I/O operations
 * - **Performance**: Functions can be optimized independently
 */

import { SimpleDims, DimensionMetadata } from '../types/dims';

/**
 * Normalizes a path to a valid Zarr store URL
 *
 * @param path - Local path or URL to normalize
 * @param baseUrl - Base URL for relative paths (defaults to window.location)
 * @returns Normalized URL with trailing slash
 */
export function normalizeZarrPath(path: string, baseUrl?: string): string {
  // Handle absolute URLs
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path.endsWith('/') ? path : path + '/';
  }

  // Handle relative paths
  const base =
    baseUrl || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost');
  const cleanPath = path.replace(/^\/?/, '/');
  const url = new URL(cleanPath, base).toString();
  return url.endsWith('/') ? url : url + '/';
}

/**
 * Extracts dimension metadata from Zarr attributes
 *
 * @param attrs - Zarr group attributes
 * @returns Parsed dimension metadata or null
 */
export function extractDimensionMetadata(attrs: any): Record<number, DimensionMetadata> | null {
  if (!attrs || !attrs.dimensions) {
    return null;
  }

  const metadata: Record<number, DimensionMetadata> = {};

  // Handle different metadata formats
  if (attrs.dimensions.metadata) {
    // New format with explicit metadata
    return attrs.dimensions.metadata;
  }

  // Legacy format conversion
  if (attrs.dimensions.names && Array.isArray(attrs.dimensions.names)) {
    attrs.dimensions.names.forEach((name: string, index: number) => {
      metadata[index] = {
        name,
        unit: attrs.dimensions.units?.[index] || '',
        range: attrs.dimensions.ranges?.[index] || [0, 1],
        step: attrs.dimensions.steps?.[index],
        display: attrs.dimensions.displayed?.includes(index),
        discrete: attrs.dimensions.discrete?.[index],
        scale: 1.0,
      };
    });
    return metadata;
  }

  return null;
}

/**
 * Merges rendering attributes with parent attributes for inheritance
 *
 * @param attrs - Current attributes
 * @param parentAttrs - Parent attributes to inherit from
 * @returns Merged attributes with inheritance applied
 */
export function inheritRenderingAttributes(
  attrs: RenderingAttributes,
  parentAttrs?: RenderingAttributes
): RenderingAttributes {
  if (!parentAttrs) {
    return attrs;
  }

  return {
    ...attrs,
    opacity: attrs.opacity ?? parentAttrs.opacity,
    gamma: attrs.gamma ?? parentAttrs.gamma,
    blending_mode: attrs.blending_mode ?? parentAttrs.blending_mode,
    point_size: attrs.point_size ?? parentAttrs.point_size,
    depth_test: attrs.depth_test ?? parentAttrs.depth_test,
    depth_write: attrs.depth_write ?? parentAttrs.depth_write,
  };
}

/**
 * Validates and normalizes point cloud data
 *
 * @param positions - Position array to validate
 * @param expectedPoints - Expected number of points
 * @param ndim - Number of dimensions
 * @returns Validated positions or throws error
 */
export function validatePointCloudData(
  positions: Float32Array,
  expectedPoints: number,
  ndim: number
): Float32Array {
  const actualPoints = positions.length / ndim;

  if (actualPoints !== expectedPoints) {
    throw new Error(
      `Position data mismatch: expected ${expectedPoints} points, got ${actualPoints}`
    );
  }

  // Check for NaN or Infinity values
  for (let i = 0; i < positions.length; i++) {
    if (!isFinite(positions[i])) {
      throw new Error(`Invalid position value at index ${i}: ${positions[i]}`);
    }
  }

  return positions;
}

/**
 * Calculates initial slice position for nD data
 *
 * @param dims - Dimension configuration
 * @returns Initial position array for non-displayed dimensions
 */
export function calculateInitialSlicePosition(dims: SimpleDims): number[] {
  const position = new Array(dims.ndim).fill(0);

  // Set non-displayed dimensions to their range minimum or center
  for (let i = 0; i < dims.ndim; i++) {
    if (!dims.displayed.includes(i)) {
      const meta = dims.metadata?.[i];
      if (meta?.range) {
        // Start at minimum for discrete, center for continuous
        position[i] = meta.discrete ? meta.range[0] : (meta.range[0] + meta.range[1]) / 2;
      }
    }
  }

  return position;
}

/**
 * Determines if a Zarr group should be loaded as points
 *
 * @param attrs - Zarr group attributes
 * @param groupName - Name of the group
 * @returns True if group contains point cloud data
 */
export function isPointCloudGroup(attrs: any, groupName: string): boolean {
  // Check explicit type attribute
  if (attrs?.type === 'points' || attrs?.type === 'pointcloud') {
    return true;
  }

  // Check for point cloud data arrays
  if (attrs?.arrays?.includes('positions')) {
    return true;
  }

  // Check naming conventions
  const pointNames = ['points', 'pointcloud', 'particles', 'vertices'];
  return pointNames.some((name) => groupName.toLowerCase().includes(name));
}

/**
 * Calculates bounding box for point cloud data
 *
 * @param positions - Flattened position array
 * @param ndim - Number of dimensions per point
 * @returns Bounding box with min and max vectors
 */
export function calculateBoundingBox(
  positions: Float32Array,
  ndim: number
): { min: number[]; max: number[] } {
  if (positions.length === 0) {
    return {
      min: new Array(ndim).fill(0),
      max: new Array(ndim).fill(0),
    };
  }

  const min = new Array(ndim).fill(Infinity);
  const max = new Array(ndim).fill(-Infinity);
  const numPoints = positions.length / ndim;

  for (let i = 0; i < numPoints; i++) {
    for (let d = 0; d < ndim; d++) {
      const value = positions[i * ndim + d];
      min[d] = Math.min(min[d], value);
      max[d] = Math.max(max[d], value);
    }
  }

  return { min, max };
}

/**
 * Processes transform attribute from Zarr metadata
 *
 * @param transformAttr - Transform attribute (matrix or list)
 * @returns 4x4 transform matrix as Float32Array
 */
export function processTransformAttribute(transformAttr: any): Float32Array | null {
  if (!transformAttr) {
    return null;
  }

  // Handle flat array (16 elements)
  if (Array.isArray(transformAttr) && transformAttr.length === 16) {
    return new Float32Array(transformAttr);
  }

  // Handle 4x4 matrix
  if (Array.isArray(transformAttr) && transformAttr.length === 4) {
    const flat = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        flat[i * 4 + j] = transformAttr[i][j];
      }
    }
    return flat;
  }

  // Handle object with matrix property
  if (transformAttr.matrix) {
    return processTransformAttribute(transformAttr.matrix);
  }

  return null;
}

/**
 * Estimates memory usage for point cloud data
 *
 * @param numPoints - Number of points
 * @param ndim - Dimensions per point
 * @param hasColors - Whether colors are included
 * @param hasRadii - Whether radii are included
 * @param hasSharpness - Whether sharpness is included
 * @returns Estimated memory in MB
 */
export function estimatePointCloudMemory(
  numPoints: number,
  ndim: number,
  hasColors: boolean = false,
  hasRadii: boolean = false,
  hasSharpness: boolean = false
): number {
  let bytesPerPoint = ndim * 4; // Float32 positions

  if (hasColors) {
    bytesPerPoint += 3 * 4; // RGB Float32
  }

  if (hasRadii) {
    bytesPerPoint += 4; // Float32
  }

  if (hasSharpness) {
    bytesPerPoint += 4; // Float32
  }

  const totalBytes = numPoints * bytesPerPoint;
  return totalBytes / (1024 * 1024); // Convert to MB
}

/**
 * Rendering attributes interface
 */
interface RenderingAttributes {
  opacity?: number;
  gamma?: number;
  blending_mode?: string;
  point_size?: number;
  depth_test?: boolean;
  depth_write?: boolean;
}

/**
 * Validates rendering attributes and applies defaults
 *
 * @param attrs - Attributes to validate
 * @returns Validated attributes with defaults
 */
export function validateRenderingAttributes(attrs: any): RenderingAttributes {
  return {
    opacity: Math.max(0, Math.min(1, attrs?.opacity ?? 1)),
    gamma: Math.max(0.1, Math.min(3, attrs?.gamma ?? 1)),
    blending_mode: attrs?.blending_mode || 'normal',
    point_size: Math.max(0.001, attrs?.point_size ?? 0.1),
    depth_test: attrs?.depth_test ?? true,
    depth_write: attrs?.depth_write ?? true,
  };
}

/**
 * Determines the appropriate data loading strategy
 *
 * @param numPoints - Number of points in dataset
 * @param availableMemoryMB - Available memory in MB
 * @returns Loading strategy: 'full', 'chunked', or 'lazy'
 */
export function determineLoadingStrategy(
  numPoints: number,
  availableMemoryMB: number
): 'full' | 'chunked' | 'lazy' {
  const estimatedMB = estimatePointCloudMemory(numPoints, 3, true, true, true);

  if (estimatedMB < availableMemoryMB * 0.3) {
    return 'full'; // Small dataset, load everything
  } else if (estimatedMB < availableMemoryMB * 0.7) {
    return 'chunked'; // Medium dataset, load in chunks
  } else {
    return 'lazy'; // Large dataset, use lazy loading
  }
}

/**
 * Converts Zarr dtype string to TypedArray constructor
 *
 * @param dtype - Zarr dtype string (e.g., '<f4', '|u1')
 * @returns Appropriate TypedArray constructor
 */
export function getArrayConstructor(
  dtype: string
): Float32ArrayConstructor | Uint8ArrayConstructor {
  if (dtype.includes('f4') || dtype.includes('float32')) {
    return Float32Array;
  } else if (dtype.includes('f8') || dtype.includes('float64')) {
    return Float32Array; // Downcast to float32 for GPU
  } else if (dtype.includes('u1') || dtype.includes('uint8')) {
    return Uint8Array;
  } else {
    return Float32Array; // Default to float32
  }
}
