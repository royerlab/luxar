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
import { attachSplatStorage, getSplatTexture } from '../../../../rendering/gsplat-geometry';
import { attachPointStorage, getPointTexture } from '../../../../rendering/point-geometry';
import { attachLineStorage, getLineTexture } from '../../../../rendering/line-geometry';
import { POINT_FLOATS_PER_POINT } from '../../../../rendering/element-texture-layout';
import { DEFAULT_POINT_RADIUS } from '../../../../config/constants';

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

  it('stamps an ancestor-composed layer order on points', () => {
    const loader = { dispose: vi.fn() } as never;
    const mesh = factory.createEmptyPointsNode(
      '/points',
      { type: 'points', n_points: 0, layer_order: 7 } as never,
      loader
    );

    expect(mesh.userData.layerOrder).toBe(7);
  });

  describe('createPointsGeometry', () => {
    // Read one float of point i's texel block (layout in point-geometry.ts:
    // [0..2] center, [3] radius, [4..6] color, [7] sharpness, [8] scalar,
    // [9] alpha).
    const texel = (geometry: THREE.BufferGeometry, i: number, offset: number): number =>
      (getPointTexture(geometry)!.image.data as Float32Array)[i * POINT_FLOATS_PER_POINT + offset];

    it('should create geometry with positions only (texture storage pair)', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      expect(geometry).toBeInstanceOf(THREE.InstancedBufferGeometry);
      // Exact-size point texture + identity aSortedIndex replace the old
      // per-instance attributes.
      const texture = getPointTexture(geometry);
      expect(texture).not.toBeNull();
      expect((texture!.image.data as Float32Array).length).toBeGreaterThanOrEqual(
        50 * POINT_FLOATS_PER_POINT
      );
      const sortedIndex = geometry.getAttribute('aSortedIndex');
      expect(sortedIndex.array).toBeInstanceOf(Uint32Array);
      expect((sortedIndex.array as Uint32Array)[49]).toBe(49); // identity ordering
      expect((geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(50);
      expect(geometry.drawRange.count).toBe(6);
      // Positions land in texel 0.
      expect(texel(geometry, 0, 0)).toBe(data.positions[0]);
      expect(texel(geometry, 49, 2)).toBe(data.positions[49 * 3 + 2]);
    });

    it('should write colors into texel 1', () => {
      const data = createMockPointsData({ pointCount: 50, hasColors: true });
      const geometry = factory.createPointsGeometry(data);

      expect(texel(geometry, 0, 4)).toBeCloseTo((data.colors as Float32Array)[0], 6);
      expect(texel(geometry, 49, 6)).toBeCloseTo((data.colors as Float32Array)[49 * 3 + 2], 6);
    });

    it('should write radii into texel 0 alpha', () => {
      const data = createMockPointsData({ pointCount: 50, hasRadii: true });
      const geometry = factory.createPointsGeometry(data);

      expect(texel(geometry, 0, 3)).toBeCloseTo((data.radii as Float32Array)[0], 6);
      expect(texel(geometry, 49, 3)).toBeCloseTo((data.radii as Float32Array)[49], 6);
    });

    it('should write sharpness into texel 1 alpha', () => {
      const data = createMockPointsData({ pointCount: 50, hasSharpness: true });
      const geometry = factory.createPointsGeometry(data);

      expect(texel(geometry, 0, 7)).toBeCloseTo((data.sharpness as Float32Array)[0], 6);
      expect(texel(geometry, 49, 7)).toBeCloseTo((data.sharpness as Float32Array)[49], 6);
    });

    it('should set default radius when not provided', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      // Default radius is 0.5
      expect(texel(geometry, 0, 3)).toBe(0.5);
      expect(texel(geometry, 49, 3)).toBe(0.5);
    });

    it('should set default sharpness when not provided', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      // Default sharpness is 0.5 (-> beta=2, a true Gaussian).
      expect(texel(geometry, 0, 7)).toBe(0.5);
      expect(texel(geometry, 49, 7)).toBe(0.5);
    });

    it('writes the scalar identity 0.0 and alpha identity 1.0 unconditionally', () => {
      const data = createMockPointsData({ pointCount: 4 });
      const geometry = factory.createPointsGeometry(data);
      for (let i = 0; i < 4; i++) {
        expect(texel(geometry, i, 8)).toBe(0.0); // no scalars in this dataset
        expect(texel(geometry, i, 9)).toBe(1.0); // opaque per-point alpha
      }
    });

    it('should handle uint8 colors with normalization (÷255 at upload)', () => {
      const data = createMockPointsData({
        pointCount: 50,
        hasColors: true,
        colorType: 'uint8',
      });
      const geometry = factory.createPointsGeometry(data);

      // The texel holds the widened [0, 1] value the old
      // `normalized: true` binding produced in the shader.
      expect(texel(geometry, 0, 4)).toBeCloseTo((data.colors as Uint8Array)[0] / 255, 6);
      expect(texel(geometry, 49, 5)).toBeCloseTo((data.colors as Uint8Array)[49 * 3 + 1] / 255, 6);
    });

    it('should handle uint8 radii with proper scaling', () => {
      const data = createMockPointsData({
        pointCount: 50,
        hasRadii: true,
        radiiType: 'uint8',
      });
      const geometry = factory.createPointsGeometry(data, 2.0);

      // Uint8 radii widen to [0, 1] in the texel; the shader multiplies
      // by radiusScale (= maxRadius) exactly as the normalized binding
      // did.
      expect(texel(geometry, 0, 3)).toBeCloseTo((data.radii as Uint8Array)[0] / 255, 6);
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

      expect(geometry.userData.radiusScale).toBe(1.0); // shader contract unchanged
      // Footprint 50 baked into boundingBox: centers [0,10] → [-50,60].
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-50, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(60, 5);
    });

    it('should bake the fill radius footprint into boundingBox when radii are absent', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      // No radii → radius texels filled with DEFAULT_POINT_RADIUS; boundingBox
      // grows by the same amount.
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-DEFAULT_POINT_RADIUS, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(10 + DEFAULT_POINT_RADIUS, 5);
    });

    it('should store the radius scale in userData', () => {
      const data = createMockPointsData({
        pointCount: 50,
        hasRadii: true,
        hasSharpness: true,
        radiiType: 'uint8',
      });
      const geometry = factory.createPointsGeometry(data, 1.5);

      expect(geometry.userData.radiusScale).toBe(1.5);
      // Sharpness has no scale — authored natively in [0, 1].
      expect(geometry.userData.sharpnessScale).toBeUndefined();
    });

    it('should set bounding box from metadata (plus footprint margin)', () => {
      const data = createMockPointsData({ pointCount: 50 });
      const geometry = factory.createPointsGeometry(data);

      // Centers bounds are [0,10]; with no radii the DEFAULT_POINT_RADIUS fill
      // footprint is baked in → [-0.5, 10.5].
      expect(geometry.boundingBox).not.toBeNull();
      expect(geometry.boundingBox?.min.x).toBeCloseTo(-DEFAULT_POINT_RADIUS, 5);
      expect(geometry.boundingBox?.max.x).toBeCloseTo(10 + DEFAULT_POINT_RADIUS, 5);
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
    // These assert on `object.matrix`, not on position/quaternion/scale. The
    // transform is installed as a full affine matrix precisely so that shear
    // survives (see the shear case below), and TRS is no longer populated.
    it('should apply identity transform without changing object', () => {
      const object = new THREE.Object3D();
      const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      factory.applyTransform(object, identity);

      expect(Array.from(object.matrix.elements)).toEqual(identity);
      expect(object.matrixAutoUpdate).toBe(false);
    });

    it('should apply translation correctly', () => {
      const object = new THREE.Object3D();
      // Column-major translation matrix
      const translation = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 10, 15, 1];

      factory.applyTransform(object, translation);

      expect(Array.from(object.matrix.elements)).toEqual(translation);
      // The world matrix must be refreshed, not left stale: with
      // matrixAutoUpdate off nothing else marks it dirty, and the pick nodes
      // snapshot matrixWorld by value.
      const origin = new THREE.Vector3(0, 0, 0).applyMatrix4(object.matrixWorld);
      expect([origin.x, origin.y, origin.z]).toEqual([5, 10, 15]);
    });

    it('should apply scale correctly', () => {
      const object = new THREE.Object3D();
      // Column-major scale matrix
      const scale = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1];

      factory.applyTransform(object, scale);

      expect(Array.from(object.matrix.elements)).toEqual(scale);
      const p = new THREE.Vector3(1, 1, 1).applyMatrix4(object.matrixWorld);
      expect([p.x, p.y, p.z]).toEqual([2, 3, 4]);
    });

    it('should preserve shear, which a TRS decomposition cannot represent', () => {
      const object = new THREE.Object3D();
      // Column-major. Linear block shears x by y: (x, y, z) -> (x + y, y, z).
      // Decomposing this into position/quaternion/scale loses the off-diagonal
      // term and silently returns different geometry -- measured before the
      // fix, (0,1,0) came back as (0.5, 1.319, 0) instead of (1,1,0).
      const shear = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

      factory.applyTransform(object, shear);

      expect(Array.from(object.matrix.elements)).toEqual(shear);

      const p = new THREE.Vector3(0, 1, 0).applyMatrix4(object.matrixWorld);
      expect(p.x).toBeCloseTo(1, 10);
      expect(p.y).toBeCloseTo(1, 10);
      expect(p.z).toBeCloseTo(0, 10);
    });

    it('should preserve a rotate-then-non-uniform-scale composition', () => {
      // The idiom `compose(rotate_y(30), scale(1.5, 0.8, 1.2))` is advertised by
      // luxar.transforms and authored by examples/transform_example.py. Its S·R
      // matrix is not expressible as Q·S', so it is a genuine shear.
      const object = new THREE.Object3D();
      const composed = new THREE.Matrix4()
        .makeScale(1.5, 0.8, 1.2)
        .multiply(new THREE.Matrix4().makeRotationY(Math.PI / 6));

      factory.applyTransform(object, Array.from(composed.elements));

      const probe = new THREE.Vector3(1, 1, 1);
      const viaViewer = probe.clone().applyMatrix4(object.matrixWorld);
      const viaAuthored = probe.clone().applyMatrix4(composed);

      expect(viaViewer.x).toBeCloseTo(viaAuthored.x, 10);
      expect(viaViewer.y).toBeCloseTo(viaAuthored.y, 10);
      expect(viaViewer.z).toBeCloseTo(viaAuthored.z, 10);
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

  describe('rebuildAfterContextRestore — GPU full-dirty (Phase 4 Stage 2)', () => {
    // After a WebGL context loss the GPU buffers are gone while the CPU
    // mirror survives, so every element texture + aSortedIndex (gsplats,
    // points, AND lines — all three geometry types are texture-backed)
    // must be marked full-dirty (empty ranges → full upload) and the
    // append-fast-path flag cleared so the next commit does a full
    // rewrite, not a suffix append.
    const makeGSplatMesh = (): THREE.Mesh => {
      const geom = new THREE.InstancedBufferGeometry();
      attachSplatStorage(geom, 8);
      const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
      mesh.userData = { nodeType: 'gsplats', gpuPrefixIntact: true };
      return mesh;
    };

    // Texture-backed points mesh: point texture + aSortedIndex, same
    // storage shape on the pool AND non-pool paths.
    const makePointsTextureMesh = (): THREE.Mesh => {
      const geom = new THREE.InstancedBufferGeometry();
      attachPointStorage(geom, 8);
      const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
      mesh.userData = { nodeType: 'points', gpuPrefixIntact: true };
      return mesh;
    };

    // Texture-backed lines mesh: line texture + aSortedIndex, same
    // storage shape on the pool AND non-pool paths (lines texture-storage
    // migration).
    const makeLinesTextureMesh = (): THREE.Mesh => {
      const geom = new THREE.InstancedBufferGeometry();
      attachLineStorage(geom, 8);
      const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
      mesh.userData = { nodeType: 'lines', gpuPrefixIntact: true };
      return mesh;
    };

    it('clears pending ranges + marks textures/aSortedIndex full-dirty and drops gpuPrefixIntact', () => {
      const mesh = makeGSplatMesh();
      const geom = mesh.geometry as THREE.InstancedBufferGeometry;
      const tex = getSplatTexture(geom)!;
      const idx = geom.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
      // Seed a partial (append-style) pending range as if a suffix commit had
      // registered one. `needsUpdate` is a write-only setter (reading returns
      // undefined) that bumps `version`, so snapshot versions to prove the
      // hook re-armed the upload.
      tex.clearUpdateRanges();
      tex.addUpdateRange(64, 32);
      idx.clearUpdateRanges();
      idx.addUpdateRange(4, 4);
      const texVersion = tex.version;
      const idxVersion = idx.version;

      const root = new THREE.Group();
      root.add(mesh);
      factory.rebuildAfterContextRestore(root);

      // Full-image upload path: ranges emptied, needsUpdate re-armed (version++).
      expect(tex.updateRanges.length).toBe(0);
      expect(tex.version).toBeGreaterThan(texVersion);
      expect(idx.updateRanges.length).toBe(0);
      expect(idx.version).toBeGreaterThan(idxVersion);
      // Next commit must full-rewrite, not append.
      expect((mesh.userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(false);
    });

    it('marks the points texture + aSortedIndex full-dirty and drops gpuPrefixIntact', () => {
      // Points get the same texture treatment as gsplats since the
      // texture-storage migration (Stage 2).
      const mesh = makePointsTextureMesh();
      const geom = mesh.geometry as THREE.InstancedBufferGeometry;
      const tex = getPointTexture(geom)!;
      const idx = geom.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
      // Seed a partial (append-style) pending range as if a suffix commit
      // had registered one; snapshot versions to prove the hook re-armed
      // the upload (needsUpdate is a write-only setter).
      tex.clearUpdateRanges();
      tex.addUpdateRange(48, 24);
      idx.clearUpdateRanges();
      idx.addUpdateRange(4, 4);
      const texVersion = tex.version;
      const idxVersion = idx.version;

      const root = new THREE.Group();
      root.add(mesh);
      factory.rebuildAfterContextRestore(root);

      expect(tex.updateRanges.length).toBe(0);
      expect(tex.version).toBeGreaterThan(texVersion);
      expect(idx.updateRanges.length).toBe(0);
      expect(idx.version).toBeGreaterThan(idxVersion);
      expect((mesh.userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(false);
    });

    it('marks the line texture + aSortedIndex full-dirty and drops gpuPrefixIntact', () => {
      // Lines take the same branch as gsplats/points since the lines
      // texture-storage migration.
      const mesh = makeLinesTextureMesh();
      const geom = mesh.geometry as THREE.InstancedBufferGeometry;
      const tex = getLineTexture(geom)!;
      const idx = geom.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
      // Seed a partial (append-style) pending range as if a suffix commit
      // had registered one; snapshot versions to prove the hook re-armed
      // the upload (needsUpdate is a write-only setter).
      tex.clearUpdateRanges();
      tex.addUpdateRange(96, 48);
      idx.clearUpdateRanges();
      idx.addUpdateRange(4, 4);
      const texVersion = tex.version;
      const idxVersion = idx.version;

      const root = new THREE.Group();
      root.add(mesh);
      factory.rebuildAfterContextRestore(root);

      // Full-image upload path (markElementTextureFullDirty): ranges
      // emptied + the pending-full state registered, needsUpdate re-armed
      // (version++).
      expect(tex.updateRanges.length).toBe(0);
      expect(tex.version).toBeGreaterThan(texVersion);
      expect(idx.updateRanges.length).toBe(0);
      expect(idx.version).toBeGreaterThan(idxVersion);
      // Next commit must full-rewrite, not append.
      expect((mesh.userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(false);
    });

    it('runs without a picking system and ignores unrelated nodes', () => {
      const gsplat = makeGSplatMesh();
      const points = makePointsTextureMesh();
      const other = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      other.userData = { nodeType: 'axes-helper', gpuPrefixIntact: true };
      const root = new THREE.Group();
      root.add(gsplat, points, other);
      // factory has no pickingSystem (constructed bare in beforeEach).
      expect(() => factory.rebuildAfterContextRestore(root)).not.toThrow();
      // Both geometry nodes flipped; the unrelated node's userData is untouched.
      expect((gsplat.userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(false);
      expect((points.userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(false);
      expect((other.userData as { gpuPrefixIntact: boolean }).gpuPrefixIntact).toBe(true);
    });
  });

  describe('createPointsMaterial', () => {
    it('should create material with default options', () => {
      const attrs = { opacity: 1.0, gamma: 1.0 };
      const material = factory.createPointsMaterial(attrs);

      expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    });

    it('should pass the radius scale', () => {
      const attrs = { opacity: 0.8, gamma: 2.2, blending_mode: 'additive' as const };
      const material = factory.createPointsMaterial(attrs, 2.0);

      expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    });
  });
});

describe('DEFAULT_POINT_RADIUS', () => {
  it('is 0.5', () => {
    // MIRROR: DEFAULT_POINT_RADIUS in
    // packages/luxar/src/luxar/typing_utils/constants.py must hold this value.
    // The two languages cannot share a symbol, so each side pins the literal
    // and names the other — same convention as the truncation-radius mirror in
    // rendering/materials/falloff.test.ts. A Python test
    // (typing_utils/tests/test_constants.py) pins that side, where the constant
    // is what the spatial index expands a no-radii chunk's bounds by; drift
    // would make the stored bound tighter than the disc drawn here, and the
    // reader would miss points at a chunk boundary.
    expect(DEFAULT_POINT_RADIUS).toBe(0.5);
  });
});
