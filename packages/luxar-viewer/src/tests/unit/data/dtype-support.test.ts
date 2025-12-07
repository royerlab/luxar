/**
 * Tests for data type support in points loading
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointsData } from '../data/data-loader-types';

describe('Data Type Support', () => {
  describe('TypedArray type detection', () => {
    it('should detect Float32Array', () => {
      const data = new Float32Array([1, 2, 3]);
      expect(data instanceof Float32Array).toBe(true);
      expect(data instanceof Uint8Array).toBe(false);
    });

    it('should detect Uint8Array', () => {
      const data = new Uint8Array([1, 2, 3]);
      expect(data instanceof Uint8Array).toBe(true);
      expect(data instanceof Float32Array).toBe(false);
    });

    it('should detect Uint16Array', () => {
      const data = new Uint16Array([1, 2, 3]);
      expect(data instanceof Uint16Array).toBe(true);
      expect(data instanceof Float32Array).toBe(false);
    });
  });

  describe('BufferAttribute normalization', () => {
    it('should create normalized BufferAttribute for Uint8Array colors', () => {
      const colors = new Uint8Array([255, 128, 0, 0, 255, 128]);
      const needsNormalization = colors instanceof Uint8Array;

      const attribute = new THREE.BufferAttribute(colors, 3, needsNormalization);

      expect(attribute.normalized).toBe(true);
      expect(attribute.array).toBe(colors);
      expect(attribute.itemSize).toBe(3);
    });

    it('should create non-normalized BufferAttribute for Float32Array', () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
      const needsNormalization = false;

      const attribute = new THREE.BufferAttribute(positions, 3, needsNormalization);

      expect(attribute.normalized).toBe(false);
      expect(attribute.array).toBe(positions);
    });

    it('should handle normalized Uint8Array for scalar attributes', () => {
      const radii = new Uint8Array([255, 128, 64, 32]);
      const needsNormalization = radii instanceof Uint8Array;

      const attribute = new THREE.BufferAttribute(radii, 1, needsNormalization);

      expect(attribute.normalized).toBe(true);
      expect(attribute.itemSize).toBe(1);
    });
  });

  describe('PointsData with different dtypes', () => {
    it('should handle Float32Array positions', () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
      const data: PointsData = {
        positions,
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
          dtypes: {
            positions: 'float32',
          },
        },
      };

      expect(data.positions).toBeInstanceOf(Float32Array);
      expect(data.metadata.dtypes?.positions).toBe('float32');
    });

    it('should handle Uint8Array colors', () => {
      const positions = new Float32Array([1, 2, 3]);
      const colors = new Uint8Array([255, 128, 0]);

      const data: PointsData = {
        positions,
        colors,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
          dtypes: {
            positions: 'float32',
            colors: 'uint8',
          },
        },
      };

      expect(data.colors).toBeInstanceOf(Uint8Array);
      expect(data.metadata.dtypes?.colors).toBe('uint8');
    });

    it('should handle mixed dtype attributes', () => {
      const positions = new Float32Array([1, 2, 3, 4, 5, 6]);
      const colors = new Uint8Array([255, 128, 0, 0, 255, 128]);
      const radii = new Uint8Array([255, 128]);
      const sharpness = new Float32Array([2.0, 3.0]);

      const data: PointsData = {
        positions,
        colors,
        radii,
        sharpness,
        metadata: {
          totalPoints: 2,
          loadedPoints: 2,
          bounds: new THREE.Box3(),
          ndim: 3,
          usedSpatialIndex: false,
          dtypes: {
            positions: 'float32',
            colors: 'uint8',
            radii: 'uint8',
            sharpness: 'float32',
          },
        },
      };

      expect(data.positions).toBeInstanceOf(Float32Array);
      expect(data.colors).toBeInstanceOf(Uint8Array);
      expect(data.radii).toBeInstanceOf(Uint8Array);
      expect(data.sharpness).toBeInstanceOf(Float32Array);
    });
  });

  describe('Geometry creation with different dtypes', () => {
    it('should create geometry with Uint8Array colors', () => {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array([0, 0, 0, 1, 1, 1]);
      const colors = new Uint8Array([255, 0, 0, 0, 255, 0]);

      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      // Colors need normalization for uint8
      const needsNormalization = colors instanceof Uint8Array;
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, needsNormalization));

      const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
      expect(colorAttr.normalized).toBe(true);
      expect(colorAttr.array).toBeInstanceOf(Uint8Array);
    });

    it('should create geometry with mixed dtypes', () => {
      const geometry = new THREE.BufferGeometry();

      // Float32 positions
      const positions = new Float32Array([0, 0, 0]);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

      // Uint16 colors (HDR-compatible)
      const colors = new Uint16Array([65535, 32768, 16384]);
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));

      // Uint8 radii (normalized)
      const radii = new Uint8Array([128]);
      geometry.setAttribute('radius', new THREE.BufferAttribute(radii, 1, true));

      // Float32 sharpness
      const sharpness = new Float32Array([2.0]);
      geometry.setAttribute('sharpness', new THREE.BufferAttribute(sharpness, 1));

      // Verify all attributes
      expect((geometry.getAttribute('position') as THREE.BufferAttribute).normalized).toBe(false);
      expect((geometry.getAttribute('color') as THREE.BufferAttribute).normalized).toBe(true);
      expect((geometry.getAttribute('radius') as THREE.BufferAttribute).normalized).toBe(true);
      expect((geometry.getAttribute('sharpness') as THREE.BufferAttribute).normalized).toBe(false);
    });
  });

  describe('Memory efficiency', () => {
    it('should calculate memory savings with different dtypes', () => {
      const numPoints = 10000;

      // All Float32 (baseline)
      const float32Memory =
        numPoints *
        (3 * 4 + // positions
          3 * 4 + // colors
          1 * 4 + // radius
          1 * 4); // sharpness

      // Mixed types (optimized)
      const mixedMemory =
        numPoints *
        (3 * 4 + // positions (float32)
          3 * 1 + // colors (uint8)
          1 * 1 + // radius (uint8)
          1 * 1); // sharpness (uint8)

      const savings = ((float32Memory - mixedMemory) / float32Memory) * 100;

      expect(savings).toBeGreaterThan(30); // Should save at least 30%
      expect(savings).toBeLessThan(70); // But not more than 70%
    });

    it('should handle empty arrays efficiently', () => {
      const emptyFloat32 = new Float32Array(0);
      const emptyUint8 = new Uint8Array(0);

      expect(emptyFloat32.byteLength).toBe(0);
      expect(emptyUint8.byteLength).toBe(0);
    });
  });
});
