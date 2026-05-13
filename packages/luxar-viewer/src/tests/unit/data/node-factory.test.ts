/**
 * Unit tests for NodeFactory.
 *
 * Tests geometry creation, validation, and transform utilities.
 * Migrated from geometry-update-manager.test.ts after removing the dead
 * GeometryUpdateManager class — NodeFactory is the single source of truth
 * for these operations.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import type { LoadedPointsData } from '../../../data/data-loader-types';

// Helper to create mock LoadedPointsData
function createMockPointsData(
  options: {
    pointCount?: number;
    hasColors?: boolean;
    hasRadii?: boolean;
    hasSharpness?: boolean;
    colorType?: 'float32' | 'uint8';
    radiiType?: 'float32' | 'uint8';
  } = {}
): LoadedPointsData {
  const {
    pointCount = 100,
    hasColors = false,
    hasRadii = false,
    hasSharpness = false,
    colorType = 'float32',
    radiiType = 'float32',
  } = options;

  const positions = new Float32Array(pointCount * 3);
  for (let i = 0; i < pointCount * 3; i++) {
    positions[i] = Math.random() * 10;
  }

  const data: LoadedPointsData = {
    positions,
    pointCount,
    ndim: 3,
    metadata: {
      totalPoints: pointCount,
      loadedPoints: pointCount,
      bounds: new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(10, 10, 10)),
      usedSpatialIndex: false,
    },
  };

  if (hasColors) {
    if (colorType === 'uint8') {
      data.colors = new Uint8Array(pointCount * 3);
      for (let i = 0; i < pointCount * 3; i++) {
        (data.colors as Uint8Array)[i] = Math.floor(Math.random() * 255);
      }
    } else {
      data.colors = new Float32Array(pointCount * 3);
      for (let i = 0; i < pointCount * 3; i++) {
        (data.colors as Float32Array)[i] = Math.random();
      }
    }
  }

  if (hasRadii) {
    if (radiiType === 'uint8') {
      data.radii = new Uint8Array(pointCount);
      for (let i = 0; i < pointCount; i++) {
        (data.radii as Uint8Array)[i] = Math.floor(Math.random() * 255);
      }
    } else {
      data.radii = new Float32Array(pointCount);
      for (let i = 0; i < pointCount; i++) {
        (data.radii as Float32Array)[i] = Math.random() * 0.5;
      }
    }
  }

  if (hasSharpness) {
    data.sharpness = new Float32Array(pointCount);
    for (let i = 0; i < pointCount; i++) {
      (data.sharpness as Float32Array)[i] = Math.random() * 10;
    }
  }

  return data;
}

describe('NodeFactory', () => {
  let factory: NodeFactory;

  beforeEach(() => {
    factory = new NodeFactory();
  });

  describe('createPointsGeometry', () => {
    it('should create geometry with positions only', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      expect(geometry).toBeInstanceOf(THREE.BufferGeometry);
      expect(geometry.getAttribute('aCenter')).toBeDefined();
      expect(geometry.getAttribute('aCenter').count).toBe(50);
    });

    it('should create geometry with colors', () => {
      const data = createMockPointsData({ pointCount: 50, hasColors: true });
      const geometry = factory.createPointsGeometry(data);

      expect(geometry.getAttribute('aColor')).toBeDefined();
      expect(geometry.getAttribute('aColor').count).toBe(50);
    });

    it('should create geometry with radii', () => {
      const data = createMockPointsData({ pointCount: 50, hasRadii: true });
      const geometry = factory.createPointsGeometry(data);

      expect(geometry.getAttribute('aRadius')).toBeDefined();
      expect(geometry.getAttribute('aRadius').count).toBe(50);
    });

    it('should create geometry with sharpness', () => {
      const data = createMockPointsData({ pointCount: 50, hasSharpness: true });
      const geometry = factory.createPointsGeometry(data);

      expect(geometry.getAttribute('aSharpness')).toBeDefined();
      expect(geometry.getAttribute('aSharpness').count).toBe(50);
    });

    it('should set default radius when not provided', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      const radiusAttr = geometry.getAttribute('aRadius');
      expect(radiusAttr).toBeDefined();
      expect(radiusAttr.count).toBe(50);
      // Default radius is 0.5
      expect(radiusAttr.array[0]).toBe(0.5);
    });

    it('should set default sharpness when not provided', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      const sharpnessAttr = geometry.getAttribute('aSharpness');
      expect(sharpnessAttr).toBeDefined();
      expect(sharpnessAttr.count).toBe(50);
      // Default sharpness is 2.0
      expect(sharpnessAttr.array[0]).toBe(2.0);
    });

    it('should handle uint8 colors with normalization', () => {
      const data = createMockPointsData({
        pointCount: 50,
        hasColors: true,
        colorType: 'uint8',
      });
      const geometry = factory.createPointsGeometry(data);

      const colorAttr = geometry.getAttribute('aColor');
      expect(colorAttr).toBeDefined();
      expect(colorAttr.normalized).toBe(true);
    });

    it('should handle uint8 radii with proper scaling', () => {
      const data = createMockPointsData({
        pointCount: 50,
        hasRadii: true,
        radiiType: 'uint8',
      });
      const geometry = factory.createPointsGeometry(data, 2.0);

      const radiusAttr = geometry.getAttribute('aRadius');
      expect(radiusAttr).toBeDefined();
      expect(radiusAttr.normalized).toBe(true);
      expect(geometry.userData.radiusScale).toBe(2.0);
    });

    it('should store radius and sharpness scales in userData', () => {
      const data = createMockPointsData({
        pointCount: 50,
        hasRadii: true,
        hasSharpness: true,
        radiiType: 'uint8',
      });
      const geometry = factory.createPointsGeometry(data, 1.5, 20.0);

      expect(geometry.userData.radiusScale).toBe(1.5);
      expect(geometry.userData.sharpnessScale).toBe(1.0); // Float32 sharpness = 1.0 scale
    });

    it('should set bounding box from metadata', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      expect(geometry.boundingBox).not.toBeNull();
      expect(geometry.boundingBox?.min.x).toBe(0);
      expect(geometry.boundingBox?.max.x).toBe(10);
    });
  });

  describe('validateLoadedPointsData', () => {
    it('should not throw for valid data', () => {
      const data = createMockPointsData({ pointCount: 50 });
      expect(() => factory.validateLoadedPointsData(data)).not.toThrow();
    });

    it('should throw for malformed positions (not divisible by 3)', () => {
      const data = createMockPointsData({ pointCount: 50 });
      // Corrupt the positions array
      data.positions = new Float32Array(151); // Not divisible by 3

      expect(() => factory.validateLoadedPointsData(data)).toThrow('Malformed positions array');
    });

    it('should handle empty dataset without throwing', () => {
      const data = createMockPointsData({ pointCount: 0 });
      expect(() => factory.validateLoadedPointsData(data)).not.toThrow();
    });

    it('should warn about colors length mismatch but not throw', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const data = createMockPointsData({ pointCount: 50, hasColors: true });
      // Corrupt colors array length
      data.colors = new Float32Array(100); // Wrong length

      expect(() => factory.validateLoadedPointsData(data)).not.toThrow();
      consoleSpy.mockRestore();
    });
  });

  describe('validateTransformFormat', () => {
    it('should not throw for valid column-major transform', () => {
      // Identity matrix with translation at [12,13,14]
      const transform = [
        1,
        0,
        0,
        0, // Column 0
        0,
        1,
        0,
        0, // Column 1
        0,
        0,
        1,
        0, // Column 2
        5,
        10,
        15,
        1, // Column 3 (translation)
      ];

      expect(() => factory.validateTransformFormat(transform)).not.toThrow();
    });

    it('should throw for row-major transform', () => {
      // Row-major matrix with translation at [3,7,11]
      const transform = [
        1,
        0,
        0,
        5, // Row 0 (tx at index 3)
        0,
        1,
        0,
        10, // Row 1 (ty at index 7)
        0,
        0,
        1,
        15, // Row 2 (tz at index 11)
        0,
        0,
        0,
        1, // Row 3
      ];

      expect(() => factory.validateTransformFormat(transform)).toThrow(/row-major/);
    });

    it('should not throw for identity matrix', () => {
      const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      expect(() => factory.validateTransformFormat(identity)).not.toThrow();
    });
  });

  describe('applyTransform', () => {
    it('should apply identity transform without changing object', () => {
      const object = new THREE.Object3D();
      const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      factory.applyTransform(object, identity);

      expect(object.position.x).toBeCloseTo(0);
      expect(object.position.y).toBeCloseTo(0);
      expect(object.position.z).toBeCloseTo(0);
      expect(object.scale.x).toBeCloseTo(1);
      expect(object.scale.y).toBeCloseTo(1);
      expect(object.scale.z).toBeCloseTo(1);
    });

    it('should apply translation correctly', () => {
      const object = new THREE.Object3D();
      // Column-major translation matrix
      const translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 10, 15, 1];

      factory.applyTransform(object, translation);

      expect(object.position.x).toBeCloseTo(5);
      expect(object.position.y).toBeCloseTo(10);
      expect(object.position.z).toBeCloseTo(15);
    });

    it('should apply scale correctly', () => {
      const object = new THREE.Object3D();
      // Column-major scale matrix
      const scale = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1];

      factory.applyTransform(object, scale);

      expect(object.scale.x).toBeCloseTo(2);
      expect(object.scale.y).toBeCloseTo(3);
      expect(object.scale.z).toBeCloseTo(4);
    });

    it('should throw on invalid transform length', () => {
      const object = new THREE.Object3D();
      const invalidTransform = [1, 0, 0, 0, 0, 1, 0, 0]; // Only 8 elements

      expect(() => factory.applyTransform(object, invalidTransform)).toThrow(
        /Invalid transform length/
      );
    });

    it('should throw on row-major transform', () => {
      const object = new THREE.Object3D();
      const rowMajor = [1, 0, 0, 5, 0, 1, 0, 10, 0, 0, 1, 15, 0, 0, 0, 1];

      expect(() => factory.applyTransform(object, rowMajor)).toThrow(/row-major/);
    });
  });

  describe('validateColorMode', () => {
    it('should accept Float32Array for HDR colors', () => {
      const colors = new Float32Array([1.5, 0.5, 2.0]); // HDR values > 1.0
      const metadata = { color_mode: 'hdr' };

      expect(() => factory.validateColorMode(colors, metadata)).not.toThrow();
    });

    it('should accept Uint8Array for SDR colors', () => {
      const colors = new Uint8Array([255, 128, 64]);
      const metadata = { color_mode: 'sdr' };

      expect(() => factory.validateColorMode(colors, metadata)).not.toThrow();
    });

    it('should warn about SDR array with HDR metadata', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const colors = new Uint8Array([255, 128, 64]);
      const metadata = { color_mode: 'hdr' };

      factory.validateColorMode(colors, metadata);
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });
  });

  describe('createPointsMaterial', () => {
    it('should create material with default options', () => {
      const attrs = { opacity: 1.0, gamma: 1.0 };
      const material = factory.createPointsMaterial(attrs);

      expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    });

    it('should pass radius and sharpness scales', () => {
      const attrs = { opacity: 0.8, gamma: 2.2, blending_mode: 'additive' as const };
      const material = factory.createPointsMaterial(attrs, 2.0, 10.0);

      expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    });
  });
});
