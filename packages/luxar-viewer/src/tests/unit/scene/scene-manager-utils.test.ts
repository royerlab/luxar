/**
 * Tests for scene manager utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  calculateBoundingBoxFromPositions,
  mergeBoundingBoxes,
  getBoundingBoxCenter,
  getBoundingBoxSize,
  getBoundingBoxMaxDimension,
  calculateCameraDistance,
  validateFOV,
  calculateClippingPlanes,
  isValidBoundingBox,
  expandBoundingBox,
  isPointInBoundingBox,
  getBoundingBoxDiagonal,
  transformBoundingBox,
  BoundingBox,
  CameraConfig,
} from '../../../scene/scene-manager-utils';

describe('scene-manager-utils', () => {
  describe('calculateBoundingBoxFromPositions', () => {
    it('should calculate correct bounds from positions', () => {
      const positions = new Float32Array([
        0,
        0,
        0, // Point 1
        1,
        2,
        3, // Point 2
        -1,
        -2,
        -3, // Point 3
        5,
        5,
        5, // Point 4
      ]);

      const box = calculateBoundingBoxFromPositions(positions);

      expect(box.min).toEqual({ x: -1, y: -2, z: -3 });
      expect(box.max).toEqual({ x: 5, y: 5, z: 5 });
    });

    it('should handle empty positions', () => {
      const box = calculateBoundingBoxFromPositions([]);

      expect(box.min).toEqual({ x: 0, y: 0, z: 0 });
      expect(box.max).toEqual({ x: 0, y: 0, z: 0 });
    });

    it('should handle single point', () => {
      const positions = [3, 4, 5];
      const box = calculateBoundingBoxFromPositions(positions);

      expect(box.min).toEqual({ x: 3, y: 4, z: 5 });
      expect(box.max).toEqual({ x: 3, y: 4, z: 5 });
    });
  });

  describe('mergeBoundingBoxes', () => {
    it('should merge multiple boxes correctly', () => {
      const boxes: BoundingBox[] = [
        { min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: 1, z: 1 } },
        { min: { x: -1, y: -1, z: -1 }, max: { x: 0.5, y: 0.5, z: 0.5 } },
        { min: { x: 2, y: 2, z: 2 }, max: { x: 3, y: 3, z: 3 } },
      ];

      const merged = mergeBoundingBoxes(boxes);

      expect(merged.min).toEqual({ x: -1, y: -1, z: -1 });
      expect(merged.max).toEqual({ x: 3, y: 3, z: 3 });
    });

    it('should handle empty array', () => {
      const merged = mergeBoundingBoxes([]);

      expect(merged.min).toEqual({ x: 0, y: 0, z: 0 });
      expect(merged.max).toEqual({ x: 0, y: 0, z: 0 });
    });
  });

  describe('getBoundingBoxCenter', () => {
    it('should calculate center correctly', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -2, z: -3 },
        max: { x: 1, y: 2, z: 3 },
      };

      const center = getBoundingBoxCenter(box);

      expect(center).toEqual({ x: 0, y: 0, z: 0 });
    });

    it('should handle offset box', () => {
      const box: BoundingBox = {
        min: { x: 10, y: 20, z: 30 },
        max: { x: 20, y: 30, z: 40 },
      };

      const center = getBoundingBoxCenter(box);

      expect(center).toEqual({ x: 15, y: 25, z: 35 });
    });
  });

  describe('getBoundingBoxSize', () => {
    it('should calculate size correctly', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -2, z: -3 },
        max: { x: 1, y: 2, z: 3 },
      };

      const size = getBoundingBoxSize(box);

      expect(size).toEqual({ x: 2, y: 4, z: 6 });
    });

    it('should handle zero-size box', () => {
      const box: BoundingBox = {
        min: { x: 5, y: 5, z: 5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const size = getBoundingBoxSize(box);

      expect(size).toEqual({ x: 0, y: 0, z: 0 });
    });
  });

  describe('getBoundingBoxMaxDimension', () => {
    it('should return maximum dimension', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 10, y: 20, z: 15 },
      };

      expect(getBoundingBoxMaxDimension(box)).toBe(20);
    });
  });

  describe('calculateCameraDistance', () => {
    it('should calculate distance for perspective camera', () => {
      const box: BoundingBox = {
        min: { x: -5, y: -5, z: -5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const camera: CameraConfig = {
        fov: 60,
        aspect: 16 / 9,
        near: 0.1,
        far: 1000,
      };

      const distance = calculateCameraDistance(box, camera);

      // Should be approximately 10 / tan(30°) * 1.1 * fitRatio factor
      expect(distance).toBeGreaterThan(10);
      expect(distance).toBeLessThan(50);
    });

    it('should handle different aspect ratios', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      const wideCamera: CameraConfig = {
        fov: 60,
        aspect: 21 / 9,
        near: 0.1,
        far: 1000,
      };

      const tallCamera: CameraConfig = {
        fov: 60,
        aspect: 9 / 16,
        near: 0.1,
        far: 1000,
      };

      const wideDistance = calculateCameraDistance(box, wideCamera);
      const tallDistance = calculateCameraDistance(box, tallCamera);

      // Tall camera needs more distance to fit same object
      expect(tallDistance).toBeGreaterThan(wideDistance);
    });

    it('should scale distance proportionally with scene size', () => {
      const camera: CameraConfig = { fov: 47, aspect: 16 / 9, near: 0.1, far: 1000 };

      const smallBox: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 1, y: 1, z: 1 },
      };
      const largeBox: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 500, y: 500, z: 500 },
      };

      const smallDist = calculateCameraDistance(smallBox, camera);
      const largeDist = calculateCameraDistance(largeBox, camera);

      // Distance should scale linearly with scene size
      expect(largeDist / smallDist).toBeCloseTo(500, 0);
    });

    it('should return 0 for zero-size bounding box', () => {
      const camera: CameraConfig = { fov: 60, aspect: 1, near: 0.1, far: 1000 };
      const zeroBox: BoundingBox = {
        min: { x: 5, y: 5, z: 5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const distance = calculateCameraDistance(zeroBox, camera);
      expect(distance).toBe(0);
    });

    it('should handle very small scenes (nanometer scale)', () => {
      const camera: CameraConfig = { fov: 47, aspect: 16 / 9, near: 0.001, far: 100 };
      const nanoBox: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0.001, y: 0.001, z: 0.001 },
      };

      const distance = calculateCameraDistance(nanoBox, camera);
      expect(distance).toBeGreaterThan(0);
      expect(distance).toBeLessThan(0.01);
      expect(Number.isFinite(distance)).toBe(true);
    });

    it('should handle very large scenes', () => {
      const camera: CameraConfig = { fov: 47, aspect: 16 / 9, near: 1, far: 100000 };
      const hugeBox: BoundingBox = {
        min: { x: -5000, y: -5000, z: -5000 },
        max: { x: 5000, y: 5000, z: 5000 },
      };

      const distance = calculateCameraDistance(hugeBox, camera);
      expect(distance).toBeGreaterThan(5000);
      expect(Number.isFinite(distance)).toBe(true);
    });
  });

  describe('validateFOV', () => {
    it('should accept valid FOV values', () => {
      expect(validateFOV(60)).toBe(60);
      expect(validateFOV(45)).toBe(45);
      expect(validateFOV(90)).toBe(90);
    });

    it('should clamp to minimum', () => {
      expect(validateFOV(5)).toBe(10);
      expect(validateFOV(-10)).toBe(10);
    });

    it('should clamp to maximum', () => {
      expect(validateFOV(150)).toBe(120);
      expect(validateFOV(200)).toBe(120);
    });

    it('should use custom limits', () => {
      expect(validateFOV(25, 20, 80)).toBe(25);
      expect(validateFOV(15, 20, 80)).toBe(20);
      expect(validateFOV(90, 20, 80)).toBe(80);
    });
  });

  describe('calculateClippingPlanes', () => {
    it('should calculate appropriate clipping planes when outside bounding sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      // Camera at z=50, facing center
      // Bounding sphere: center=(0,0,0), half-diag=sqrt(1200)/2≈17.32, radius=17.32*1.2≈20.78
      // near = 50 - 20.78 ≈ 29.2
      // far = 50 + 20.78 ≈ 70.8
      const cameraPos = { x: 0, y: 0, z: 50 };
      const planes = calculateClippingPlanes(box, cameraPos);

      expect(planes.near).toBeCloseTo(29.2, 0);
      expect(planes.far).toBeCloseTo(70.8, 0);
    });

    it('should use minimum near plane when inside bounding sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      // Camera inside the sphere at (0,0,8): distToCenter=8, radius≈20.78
      // near = max(0.0001, 8 - 20.78) = 0.0001
      // far = 8 + 20.78 ≈ 28.78
      const cameraPos = { x: 0, y: 0, z: 8 };
      const planes = calculateClippingPlanes(box, cameraPos);

      expect(planes.near).toBe(0.0001);
      expect(planes.far).toBeCloseTo(28.8, 0);
    });

    it('should enforce minimum near plane', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0.01, y: 0.01, z: 0.01 },
      };

      // Camera very close to tiny box
      const cameraPos = { x: 0.005, y: 0.005, z: 0.02 };
      const planes = calculateClippingPlanes(box, cameraPos);

      // Minimum near is MIN_NEAR_PLANE = 0.0001
      expect(planes.near).toBeGreaterThanOrEqual(0.0001);
    });

    it('should provide smooth near-plane transition approaching the sphere', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      // Bounding sphere radius ≈ 20.78
      // Camera moving from z=50 toward center — near should decrease smoothly
      const far = calculateClippingPlanes(box, { x: 0, y: 0, z: 50 });
      const mid = calculateClippingPlanes(box, { x: 0, y: 0, z: 30 });
      const close = calculateClippingPlanes(box, { x: 0, y: 0, z: 22 });

      expect(far.near).toBeGreaterThan(mid.near);
      expect(mid.near).toBeGreaterThan(close.near);
      expect(close.near).toBeGreaterThan(0); // Still outside sphere
    });
  });

  describe('isValidBoundingBox', () => {
    it('should validate non-zero boxes', () => {
      const valid: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 1, y: 1, z: 1 },
      };

      expect(isValidBoundingBox(valid)).toBe(true);
    });

    it('should reject zero-volume boxes', () => {
      const invalid: BoundingBox = {
        min: { x: 5, y: 5, z: 5 },
        max: { x: 5, y: 5, z: 5 },
      };

      expect(isValidBoundingBox(invalid)).toBe(false);
    });

    it('should accept boxes with single non-zero dimension', () => {
      const line: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 10, y: 0, z: 0 },
      };

      expect(isValidBoundingBox(line)).toBe(true);
    });
  });

  describe('expandBoundingBox', () => {
    it('should expand box by margin', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };

      const expanded = expandBoundingBox(box, 2);

      expect(expanded.min).toEqual({ x: -3, y: -3, z: -3 });
      expect(expanded.max).toEqual({ x: 3, y: 3, z: 3 });
    });

    it('should shrink box with negative margin', () => {
      const box: BoundingBox = {
        min: { x: -5, y: -5, z: -5 },
        max: { x: 5, y: 5, z: 5 },
      };

      const shrunk = expandBoundingBox(box, -1);

      expect(shrunk.min).toEqual({ x: -4, y: -4, z: -4 });
      expect(shrunk.max).toEqual({ x: 4, y: 4, z: 4 });
    });
  });

  describe('isPointInBoundingBox', () => {
    const box: BoundingBox = {
      min: { x: -1, y: -1, z: -1 },
      max: { x: 1, y: 1, z: 1 },
    };

    it('should detect points inside box', () => {
      expect(isPointInBoundingBox({ x: 0, y: 0, z: 0 }, box)).toBe(true);
      expect(isPointInBoundingBox({ x: 0.5, y: 0.5, z: 0.5 }, box)).toBe(true);
      expect(isPointInBoundingBox({ x: -0.5, y: -0.5, z: -0.5 }, box)).toBe(true);
    });

    it('should detect points on boundaries', () => {
      expect(isPointInBoundingBox({ x: 1, y: 0, z: 0 }, box)).toBe(true);
      expect(isPointInBoundingBox({ x: -1, y: -1, z: -1 }, box)).toBe(true);
    });

    it('should detect points outside box', () => {
      expect(isPointInBoundingBox({ x: 2, y: 0, z: 0 }, box)).toBe(false);
      expect(isPointInBoundingBox({ x: 0, y: 2, z: 0 }, box)).toBe(false);
      expect(isPointInBoundingBox({ x: 0, y: 0, z: -2 }, box)).toBe(false);
    });
  });

  describe('getBoundingBoxDiagonal', () => {
    it('should calculate diagonal length', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 3, y: 4, z: 0 },
      };

      // 3-4-5 triangle
      expect(getBoundingBoxDiagonal(box)).toBe(5);
    });

    it('should handle 3D diagonal', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 2, y: 2, z: 2 },
      };

      // sqrt(2^2 + 2^2 + 2^2) = sqrt(12)
      expect(getBoundingBoxDiagonal(box)).toBeCloseTo(3.464, 3);
    });
  });

  describe('transformBoundingBox', () => {
    it('should handle identity transform', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };

      // Identity matrix (column-major)
      const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      const transformed = transformBoundingBox(box, identity);

      expect(transformed.min).toEqual(box.min);
      expect(transformed.max).toEqual(box.max);
    });

    it('should handle translation', () => {
      const box: BoundingBox = {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 1, y: 1, z: 1 },
      };

      // Translation matrix (translate by 5, 10, 15)
      const translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 10, 15, 1];

      const transformed = transformBoundingBox(box, translation);

      expect(transformed.min).toEqual({ x: 5, y: 10, z: 15 });
      expect(transformed.max).toEqual({ x: 6, y: 11, z: 16 });
    });

    it('should handle scaling', () => {
      const box: BoundingBox = {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };

      // Scale matrix (scale by 2, 3, 4)
      const scale = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1];

      const transformed = transformBoundingBox(box, scale);

      expect(transformed.min).toEqual({ x: -2, y: -3, z: -4 });
      expect(transformed.max).toEqual({ x: 2, y: 3, z: 4 });
    });
  });
});
