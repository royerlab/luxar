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
    it('should calculate appropriate clipping planes when outside bounding box', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      // Camera at z=50, facing center (nearest face center at z=10, distance ~40)
      const cameraPos = { x: 0, y: 0, z: 50 };
      const planes = calculateClippingPlanes(box, cameraPos);

      // With default 50% margin:
      // nearDist ≈ 40 (distance to nearest face center at z=10)
      // near = 40 * (1 - 0.5) = 20
      expect(planes.near).toBeCloseTo(20, 0);
      // farDist ≈ 60.8 (distance to farthest corner)
      // far = 60.8 * 1.5 ≈ 91.2
      expect(planes.far).toBeGreaterThan(80);
      expect(planes.far).toBeLessThan(100);
    });

    it('should use minimum near plane when inside bounding box', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      // Camera inside the box, near the +Z face
      const cameraPos = { x: 0, y: 0, z: 8 };
      const planes = calculateClippingPlanes(box, cameraPos);

      // When inside the bounding box, use MIN_NEAR_PLANE to see all geometry
      expect(planes.near).toBe(0.0001);
      // farDist = distance to farthest corner (approx sqrt(10² + 10² + 18²) ≈ 23.2)
      // far ≈ 23.2 * 1.5 ≈ 34.8
      expect(planes.far).toBeGreaterThan(30);
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

    it('should support custom margin parameter', () => {
      const box: BoundingBox = {
        min: { x: -10, y: -10, z: -10 },
        max: { x: 10, y: 10, z: 10 },
      };

      // Camera outside at z=50
      const cameraPos = { x: 0, y: 0, z: 50 };

      // With 0% margin (no margin)
      const noMargin = calculateClippingPlanes(box, cameraPos, 0);
      // nearDist ≈ 40, near = 40 * 1.0 = 40
      expect(noMargin.near).toBeCloseTo(40, 0);

      // With 20% margin
      const smallMargin = calculateClippingPlanes(box, cameraPos, 0.2);
      // near = 40 * 0.8 = 32
      expect(smallMargin.near).toBeCloseTo(32, 0);
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
