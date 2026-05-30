/**
 * Unit tests for NodeFactory (located under `tests/unit/rendering/node-factory/`
 * to match the production source layout at `src/rendering/node-factory/`).
 *
 * Tests geometry creation, validation, and transform utilities.
 * NodeFactory is the single source of truth for these operations.
 *
 * rendering.md O2 fix: this file previously lived under
 * `tests/unit/data/node-factory.test.ts`, conflating rendering and
 * data-loader test scopes. Moved to mirror the source-tree layout
 * (P10: tests follow source themes).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../../rendering/node-factory';
import type { LoadedPointsData } from '../../../../data/data-loader-types';
import { mulberry32 } from '../../../helpers/random';

// Audit C3 fix: `Math.random()` replaced with a seedable PRNG so failures
// can be reproduced. The seed is fixed per call site below; bump it if
// you need to explore alternate data shapes.
function createMockPointsData(
  options: {
    pointCount?: number;
    hasColors?: boolean;
    hasRadii?: boolean;
    hasSharpness?: boolean;
    colorType?: 'float32' | 'uint8';
    radiiType?: 'float32' | 'uint8';
    seed?: number;
  } = {}
): LoadedPointsData {
  const {
    pointCount = 100,
    hasColors = false,
    hasRadii = false,
    hasSharpness = false,
    colorType = 'float32',
    radiiType = 'float32',
    seed = 0xc0ffee,
  } = options;

  const rng = mulberry32(seed);

  const positions = new Float32Array(pointCount * 3);
  for (let i = 0; i < pointCount * 3; i++) {
    positions[i] = rng() * 10;
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
        (data.colors as Uint8Array)[i] = Math.floor(rng() * 255);
      }
    } else {
      data.colors = new Float32Array(pointCount * 3);
      for (let i = 0; i < pointCount * 3; i++) {
        (data.colors as Float32Array)[i] = rng();
      }
    }
  }

  if (hasRadii) {
    if (radiiType === 'uint8') {
      data.radii = new Uint8Array(pointCount);
      for (let i = 0; i < pointCount; i++) {
        (data.radii as Uint8Array)[i] = Math.floor(rng() * 255);
      }
    } else {
      data.radii = new Float32Array(pointCount);
      for (let i = 0; i < pointCount; i++) {
        (data.radii as Float32Array)[i] = rng() * 0.5;
      }
    }
  }

  if (hasSharpness) {
    data.sharpness = new Float32Array(pointCount);
    for (let i = 0; i < pointCount; i++) {
      (data.sharpness as Float32Array)[i] = rng() * 10;
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

      expect(geometry).toBeInstanceOf(THREE.InstancedBufferGeometry);
      expect(geometry.getAttribute('aCenter')).toBeDefined();
      expect(geometry.getAttribute('aCenter').count).toBe(50);
      expect((geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(50);
      expect(geometry.drawRange.count).toBe(6);
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
      // Uint8 normalized radii map to [0, maxRadius], so the footprint
      // baked into boundingBox is maxRadius (2.0): centers [0,10] → [-2,12].
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-2, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(12, 5);
    });

    it('should bake a large float32 radius footprint into boundingBox (radiusScale stays 1.0)', () => {
      // W1 regression: Float32 radii are already in world units, so the
      // shader normalization factor radiusScale stays 1.0. The rendered
      // footprint must instead be baked into boundingBox from the metadata
      // max_radius, or large-radius points get culled before pick readback.
      const data = createMockPointsData({
        pointCount: 50,
        hasRadii: true,
        radiiType: 'float32',
      });
      const geometry = factory.createPointsGeometry(data, 50.0);

      expect(geometry.getAttribute('aRadius').normalized).toBe(false);
      expect(geometry.userData.radiusScale).toBe(1.0); // shader contract unchanged
      // Footprint 50 baked into boundingBox: centers [0,10] → [-50,60].
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-50, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(60, 5);
    });

    it('should bake the fill radius footprint into boundingBox when radii are absent', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      // No radii → aRadius filled with 0.5; boundingBox grows by 0.5.
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-0.5, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(10.5, 5);
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

    it('should set bounding box from metadata (plus footprint margin)', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      // Centers bounds are [0,10]; with no radii the 0.5 fill footprint is
      // baked in → [-0.5, 10.5].
      expect(geometry.boundingBox).not.toBeNull();
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-0.5, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(10.5, 5);
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

    // rendering.md G1 boundary cases for validateTransformFormat:
    //   1. Both col-major AND row-major slots non-zero → guard throws
    //      with an "ambiguous" message. A genuine column-major matrix
    //      always has its last row [3,7,11,15] equal to [0,0,0,1], so
    //      a non-zero value at any of those indices alongside a
    //      non-zero col-major translation is a producer bug we refuse
    //      to interpret silently (MED-24).
    //   2. Near-zero translation: |v| ≤ 0.001 must NOT trigger the
    //      row-major heuristic (the threshold is the floating-point
    //      tolerance used in the source).
    it('should throw "ambiguous" when both col-major and row-major translation slots are non-zero', () => {
      // [3]=5 (row-major slot), [12]=10 (col-major slot). Ambiguous.
      // Producer must fix the encoding so the last row is [0,0,0,1].
      const ambiguous = [
        1,
        0,
        0,
        5, // index 3: non-zero (row-major slot)
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        10, // index 12: non-zero (col-major slot)
        0,
        0,
        1,
      ];
      expect(() => factory.validateTransformFormat(ambiguous)).toThrow(/ambiguous/);
    });

    it('should NOT throw when row-major slot is just under the 0.001 threshold', () => {
      // |0.0009| < 0.001 → does not register as "non-zero translation".
      const subThreshold = [
        1,
        0,
        0,
        0.0009, // index 3: below threshold
        0,
        1,
        0,
        0.0009, // index 7: below threshold
        0,
        0,
        1,
        0.0009, // index 11: below threshold
        0,
        0,
        0,
        1,
      ];
      expect(() => factory.validateTransformFormat(subThreshold)).not.toThrow();
    });

    it('should throw when row-major slot is just above the 0.001 threshold', () => {
      // |0.002| > 0.001 → registers; col-major slots are zero so it's
      // unambiguously row-major. Pins the exact threshold direction.
      const overThreshold = [1, 0, 0, 0.002, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
      expect(() => factory.validateTransformFormat(overThreshold)).toThrow(/row-major/);
    });

    it('error message references the [3,7,11] vs [12,13,14] index contract', () => {
      // P9: name what the assertion actually pins — the public error
      // message guides producers toward `matrix.T.ravel().tolist()`.
      const rowMajor = [1, 0, 0, 5, 0, 1, 0, 10, 0, 0, 1, 15, 0, 0, 0, 1];
      expect(() => factory.validateTransformFormat(rowMajor)).toThrow(/3,7,11/);
      expect(() => factory.validateTransformFormat(rowMajor)).toThrow(/12,13,14/);
    });
  });

  describe('applyTransform', () => {
    it('should apply identity transform without changing object', () => {
      const object = new THREE.Object3D();
      const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      factory.applyTransform(object, identity);

      expect(object.position.x).toBe(0);
      expect(object.position.y).toBe(0);
      expect(object.position.z).toBe(0);
      expect(object.scale.x).toBe(1);
      expect(object.scale.y).toBe(1);
      expect(object.scale.z).toBe(1);
    });

    it('should apply translation correctly', () => {
      const object = new THREE.Object3D();
      // Column-major translation matrix
      const translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 10, 15, 1];

      factory.applyTransform(object, translation);

      expect(object.position.x).toBe(5);
      expect(object.position.y).toBe(10);
      expect(object.position.z).toBe(15);
    });

    it('should apply scale correctly', () => {
      const object = new THREE.Object3D();
      // Column-major scale matrix
      const scale = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1];

      factory.applyTransform(object, scale);

      expect(object.scale.x).toBe(2);
      expect(object.scale.y).toBe(3);
      expect(object.scale.z).toBe(4);
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
