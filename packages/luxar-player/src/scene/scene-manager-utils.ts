/**
 * Pure utility functions for scene management calculations
 *
 * This module contains side-effect-free functions extracted from scene-manager.ts
 * to improve testability and maintainability. These functions handle camera
 * calculations, bounding box operations, and scene analysis without external
 * dependencies.
 */

import { config } from '../config';

/**
 * 3D bounding box representation
 */
export interface BoundingBox {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}

/**
 * Camera configuration for scene fitting
 */
export interface CameraConfig {
  fov: number; // Field of view in degrees
  aspect: number; // Aspect ratio
  near: number; // Near clipping plane
  far: number; // Far clipping plane
}

/**
 * Calculates bounding box from position array
 *
 * @param positions - Flattened position array (x,y,z,x,y,z,...)
 * @returns Bounding box with min and max points
 */
export function calculateBoundingBoxFromPositions(positions: Float32Array | number[]): BoundingBox {
  if (positions.length === 0) {
    return {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
    };
  }

  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
  };
}

/**
 * Merges multiple bounding boxes into one
 *
 * @param boxes - Array of bounding boxes to merge
 * @returns Combined bounding box
 */
export function mergeBoundingBoxes(boxes: BoundingBox[]): BoundingBox {
  if (boxes.length === 0) {
    return {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
    };
  }

  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;

  for (const box of boxes) {
    minX = Math.min(minX, box.min.x);
    minY = Math.min(minY, box.min.y);
    minZ = Math.min(minZ, box.min.z);
    maxX = Math.max(maxX, box.max.x);
    maxY = Math.max(maxY, box.max.y);
    maxZ = Math.max(maxZ, box.max.z);
  }

  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
  };
}

/**
 * Calculates bounding box center point
 *
 * @param box - Bounding box
 * @returns Center point coordinates
 */
export function getBoundingBoxCenter(box: BoundingBox): { x: number; y: number; z: number } {
  return {
    x: (box.min.x + box.max.x) / 2,
    y: (box.min.y + box.max.y) / 2,
    z: (box.min.z + box.max.z) / 2,
  };
}

/**
 * Calculates bounding box size
 *
 * @param box - Bounding box
 * @returns Size in each dimension
 */
export function getBoundingBoxSize(box: BoundingBox): { x: number; y: number; z: number } {
  return {
    x: box.max.x - box.min.x,
    y: box.max.y - box.min.y,
    z: box.max.z - box.min.z,
  };
}

/**
 * Calculates maximum dimension of bounding box
 *
 * @param box - Bounding box
 * @returns Maximum dimension value
 */
export function getBoundingBoxMaxDimension(box: BoundingBox): number {
  const size = getBoundingBoxSize(box);
  return Math.max(size.x, size.y, size.z);
}

/**
 * Calculates optimal camera distance to fit bounding box in view
 *
 * @param box - Bounding box to fit
 * @param camera - Camera configuration
 * @param fitRatio - How much of the view to fill (0-1, default 0.75)
 * @returns Optimal camera distance from center
 */
export function calculateCameraDistance(
  box: BoundingBox,
  camera: CameraConfig,
  fitRatio: number = config.scene.defaultFitRatio
): number {
  const maxDim = getBoundingBoxMaxDimension(box);

  // Calculate distance based on FOV
  const fovRadians = (camera.fov * Math.PI) / 180;
  const halfFov = fovRadians / 2;

  // Consider aspect ratio to ensure object fits in both dimensions
  const verticalFit = maxDim / fitRatio / (2 * Math.tan(halfFov));
  const horizontalFit = maxDim / fitRatio / (2 * Math.tan(halfFov) * camera.aspect);

  // Use the larger distance to ensure complete fit
  return Math.max(verticalFit, horizontalFit) * 1.1; // Add 10% margin
}

/**
 * Validates and clamps field of view value
 *
 * @param fov - Field of view in degrees
 * @param min - Minimum FOV (default 10)
 * @param max - Maximum FOV (default 120)
 * @returns Clamped FOV value
 */
export function validateFOV(fov: number, min: number = 10, max: number = 120): number {
  return Math.max(min, Math.min(max, fov));
}

/**
 * Calculates camera clipping planes based on scene bounds
 *
 * @param box - Scene bounding box
 * @param cameraDistance - Distance from camera to center
 * @returns Near and far clipping plane distances
 */
export function calculateClippingPlanes(
  box: BoundingBox,
  cameraDistance: number
): { near: number; far: number } {
  const maxDim = getBoundingBoxMaxDimension(box);

  // Near plane: 1% of camera distance, but at least 0.001
  const near = Math.max(0.001, cameraDistance * 0.01);

  // Far plane: camera distance + scene size + margin
  const far = cameraDistance + maxDim * 2;

  return { near, far };
}

/**
 * Determines if a bounding box is valid (non-zero volume)
 *
 * @param box - Bounding box to check
 * @returns True if box has non-zero volume
 */
export function isValidBoundingBox(box: BoundingBox): boolean {
  const size = getBoundingBoxSize(box);
  return size.x > 0 || size.y > 0 || size.z > 0;
}

/**
 * Expands bounding box by a margin
 *
 * @param box - Original bounding box
 * @param margin - Margin to add (can be negative to shrink)
 * @returns Expanded bounding box
 */
export function expandBoundingBox(box: BoundingBox, margin: number): BoundingBox {
  return {
    min: {
      x: box.min.x - margin,
      y: box.min.y - margin,
      z: box.min.z - margin,
    },
    max: {
      x: box.max.x + margin,
      y: box.max.y + margin,
      z: box.max.z + margin,
    },
  };
}

/**
 * Checks if a point is inside a bounding box
 *
 * @param point - Point coordinates
 * @param box - Bounding box
 * @returns True if point is inside box
 */
export function isPointInBoundingBox(
  point: { x: number; y: number; z: number },
  box: BoundingBox
): boolean {
  return (
    point.x >= box.min.x &&
    point.x <= box.max.x &&
    point.y >= box.min.y &&
    point.y <= box.max.y &&
    point.z >= box.min.z &&
    point.z <= box.max.z
  );
}

/**
 * Calculates bounding box diagonal length
 *
 * @param box - Bounding box
 * @returns Diagonal length
 */
export function getBoundingBoxDiagonal(box: BoundingBox): number {
  const size = getBoundingBoxSize(box);
  return Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z);
}

/**
 * Transforms bounding box by a 4x4 matrix
 *
 * @param box - Original bounding box
 * @param matrix - 4x4 transformation matrix (column-major, flat array)
 * @returns Transformed bounding box
 */
export function transformBoundingBox(box: BoundingBox, matrix: number[]): BoundingBox {
  // Get 8 corners of the box
  const corners = [
    { x: box.min.x, y: box.min.y, z: box.min.z },
    { x: box.max.x, y: box.min.y, z: box.min.z },
    { x: box.min.x, y: box.max.y, z: box.min.z },
    { x: box.max.x, y: box.max.y, z: box.min.z },
    { x: box.min.x, y: box.min.y, z: box.max.z },
    { x: box.max.x, y: box.min.y, z: box.max.z },
    { x: box.min.x, y: box.max.y, z: box.max.z },
    { x: box.max.x, y: box.max.y, z: box.max.z },
  ];

  // Transform each corner
  const transformedCorners = corners.map((corner) => {
    const w = matrix[3] * corner.x + matrix[7] * corner.y + matrix[11] * corner.z + matrix[15];
    return {
      x: (matrix[0] * corner.x + matrix[4] * corner.y + matrix[8] * corner.z + matrix[12]) / w,
      y: (matrix[1] * corner.x + matrix[5] * corner.y + matrix[9] * corner.z + matrix[13]) / w,
      z: (matrix[2] * corner.x + matrix[6] * corner.y + matrix[10] * corner.z + matrix[14]) / w,
    };
  });

  // Find new min/max
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;

  for (const point of transformedCorners) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    minZ = Math.min(minZ, point.z);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
    maxZ = Math.max(maxZ, point.z);
  }

  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
  };
}
