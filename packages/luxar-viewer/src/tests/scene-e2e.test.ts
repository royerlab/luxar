/**
 * End-to-End Scene Tests
 *
 * Tests the complete data pipeline from Python zarr files to rendered scenes:
 * - Hierarchical transforms
 * - Scene graph composition
 * - Metadata propagation
 * - Full integration tests
 */

import { describe, it, expect } from 'vitest';
import * as zarr from 'zarrita';
import { FileSystemStore } from '@zarrita/storage';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Get the directory name for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to test fixtures
const FIXTURES_DIR = path.resolve(__dirname, '../../tests/fixtures');

describe('Scene E2E Tests', () => {
  describe('Hierarchical Transforms', () => {
    it('should correctly store and read hierarchical transforms', async () => {
      // CRITICAL: This test verifies the full transform pipeline:
      // 1. Python creates hierarchy with transforms
      // 2. Transforms are stored in correct format (column-major for THREE.js)
      // 3. TypeScript can read and parse transforms
      // 4. Transform composition is correct

      const storePath = path.join(FIXTURES_DIR, 'test_hierarchical_transforms.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      // Read parent group metadata
      const parentLoc = rootLoc.resolve('parent_group');
      const parentGroup = await zarr.open(parentLoc, { kind: 'group' });
      const parentAttrs = parentGroup.attrs as any;

      // Verify parent transform exists and is correct format
      expect(parentAttrs.transform).toBeDefined();
      expect(parentAttrs.transform).toHaveLength(16); // 4x4 matrix flattened

      // Parse transform as THREE.js expects (column-major)
      // Translation should be at indices [12, 13, 14]
      const parentTransform = parentAttrs.transform;
      const parentTranslation = {
        x: parentTransform[12],
        y: parentTransform[13],
        z: parentTransform[14],
      };

      // Parent should translate by [10, 0, 0]
      expect(parentTranslation.x).toBeCloseTo(10.0, 1);
      expect(parentTranslation.y).toBeCloseTo(0.0, 1);
      expect(parentTranslation.z).toBeCloseTo(0.0, 1);

      // Read child points metadata
      const childLoc = rootLoc.resolve('parent_group/child_points');
      const childGroup = await zarr.open(childLoc, { kind: 'group' });
      const childAttrs = childGroup.attrs as any;

      // Verify child transform
      expect(childAttrs.transform).toBeDefined();
      expect(childAttrs.transform).toHaveLength(16);

      const childTransform = childAttrs.transform;
      const childTranslation = {
        x: childTransform[12],
        y: childTransform[13],
        z: childTransform[14],
      };

      // Child should translate by [0, 5, 0] relative to parent
      expect(childTranslation.x).toBeCloseTo(0.0, 1);
      expect(childTranslation.y).toBeCloseTo(5.0, 1);
      expect(childTranslation.z).toBeCloseTo(0.0, 1);

      // Verify final composed position would be [10, 5, 0]
      // (This is what THREE.js will compute when applying both transforms)
      const expectedFinalX = parentTranslation.x + childTranslation.x;
      const expectedFinalY = parentTranslation.y + childTranslation.y;
      const expectedFinalZ = parentTranslation.z + childTranslation.z;

      expect(expectedFinalX).toBeCloseTo(10.0, 1);
      expect(expectedFinalY).toBeCloseTo(5.0, 1);
      expect(expectedFinalZ).toBeCloseTo(0.0, 1);
    });

    it('should detect transforms in correct format (column-major)', async () => {
      // Verify that transforms are NOT in row-major (NumPy) format

      const storePath = path.join(FIXTURES_DIR, 'test_hierarchical_transforms.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      const parentLoc = rootLoc.resolve('parent_group');
      const parentGroup = await zarr.open(parentLoc, { kind: 'group' });
      const parentAttrs = parentGroup.attrs as any;

      const transform = parentAttrs.transform;

      // Check column-major translation (correct for THREE.js)
      const colMajorTranslation = [transform[12], transform[13], transform[14]];

      // Check row-major translation (WRONG for THREE.js)
      const rowMajorTranslation = [transform[3], transform[7], transform[11]];

      // At least one of column-major translation components should be non-zero
      const colMajorHasTranslation = colMajorTranslation.some((v) => Math.abs(v) > 0.001);
      expect(colMajorHasTranslation).toBe(true);

      // If row-major positions are non-zero but column-major are zero,
      // transform is in WRONG format
      const rowMajorHasTranslation = rowMajorTranslation.some((v) => Math.abs(v) > 0.001);

      if (rowMajorHasTranslation) {
        // Both should not be non-zero for a pure translation
        // This would indicate the transform is in wrong format
        expect(colMajorHasTranslation).toBe(true);
      }
    });

    it('should have identity scale and rotation for pure translation', async () => {
      // For transforms that are pure translations, verify no unexpected scaling or rotation

      const storePath = path.join(FIXTURES_DIR, 'test_hierarchical_transforms.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      const parentLoc = rootLoc.resolve('parent_group');
      const parentGroup = await zarr.open(parentLoc, { kind: 'group' });
      const parentAttrs = parentGroup.attrs as any;

      const m = parentAttrs.transform;

      // For a pure translation matrix in column-major format:
      // [1, 0, 0, 0,    <- first column
      //  0, 1, 0, 0,    <- second column
      //  0, 0, 1, 0,    <- third column
      //  tx, ty, tz, 1] <- fourth column (translation)

      // Verify rotation/scale components are identity
      expect(m[0]).toBeCloseTo(1.0, 5); // m00
      expect(m[5]).toBeCloseTo(1.0, 5); // m11
      expect(m[10]).toBeCloseTo(1.0, 5); // m22
      expect(m[15]).toBeCloseTo(1.0, 5); // m33

      // Verify off-diagonal elements are zero (no rotation/shear)
      expect(m[1]).toBeCloseTo(0.0, 5); // m10
      expect(m[2]).toBeCloseTo(0.0, 5); // m20
      expect(m[4]).toBeCloseTo(0.0, 5); // m01
      expect(m[6]).toBeCloseTo(0.0, 5); // m21
      expect(m[8]).toBeCloseTo(0.0, 5); // m02
      expect(m[9]).toBeCloseTo(0.0, 5); // m12
    });
  });

  describe('nD Data and Slicing (E2E)', () => {
    it('should load 4D dataset with time dimension', async () => {
      // Test 4D data loading: positions should have 4 coordinates (t, x, y, z)

      const storePath = path.join(FIXTURES_DIR, 'test_4d.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      // Read scene metadata
      const sceneGroup = await zarr.open(rootLoc, { kind: 'group' });
      const sceneAttrs = sceneGroup.attrs as any;

      // Verify scene has dimensions metadata
      expect(sceneAttrs.scene_dimensions).toBeDefined();
      expect(sceneAttrs.scene_dimensions.dimensions).toBeDefined();
      expect(sceneAttrs.scene_dimensions.dimensions.length).toBe(4);

      // Verify dimension names: time, x, y, z
      const dimensions = sceneAttrs.scene_dimensions.dimensions;
      const dimNames = dimensions.map((d: any) => d.name);
      expect(dimNames).toContain('time');
      expect(dimNames).toContain('x');
      expect(dimNames).toContain('y');
      expect(dimNames).toContain('z');

      // Verify time dimension properties
      const timeDim = dimensions.find((d: any) => d.name === 'time');
      expect(timeDim.display).toBe(false); // Time is not displayed
      expect(timeDim.discrete).toBe(true); // Time is discrete
    });

    it('should have correct 4D positions array shape', async () => {
      const storePath = path.join(FIXTURES_DIR, 'test_4d.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      // Read positions array
      const positionsLoc = rootLoc.resolve('points/positions');
      const positionsArray = await zarr.open(positionsLoc, { kind: 'array' });

      // Verify shape is [N, 4] for 4D data
      expect(positionsArray.shape.length).toBe(2);
      expect(positionsArray.shape[1]).toBe(4); // 4 coordinates: t, x, y, z

      // Verify number of points (500 points × 10 time steps = 5000)
      expect(positionsArray.shape[0]).toBe(5000);
    });

    it('should have time-varying colors in 4D dataset', async () => {
      // Test fixture has colors that change with time dimension
      // Colors use LUT encoding (10 unique values for 10 time steps)

      const storePath = path.join(FIXTURES_DIR, 'test_4d.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      // Read colors array
      const colorsLoc = rootLoc.resolve('points/colors');
      const colorsArray = await zarr.open(colorsLoc, { kind: 'array' });
      const colorsAttrs = colorsArray.attrs as any;

      // Colors use LUT encoding, so shape is [N] (indices), not [N, 3]
      expect(colorsArray.shape.length).toBe(1);
      expect(colorsArray.shape[0]).toBe(5000);

      // Verify LUT encoding metadata
      expect(colorsAttrs.encoding?.name).toBe('lut_uint8');
      expect(colorsAttrs.encoding?.lut).toBeDefined();
      expect(colorsAttrs.encoding?.lut.length).toBe(10); // 10 time steps = 10 unique colors

      // Verify original shape indicates RGB
      expect(colorsAttrs.encoding?.original_shape).toEqual([5000, 3]);
    });

    it('should define dimension ranges for nD navigation', async () => {
      const storePath = path.join(FIXTURES_DIR, 'test_4d.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      const sceneGroup = await zarr.open(rootLoc, { kind: 'group' });
      const sceneAttrs = sceneGroup.attrs as any;

      const dimensions = sceneAttrs.scene_dimensions.dimensions;

      // Check that time dimension has a defined range
      const timeDim = dimensions.find((d: any) => d.name === 'time');
      expect(timeDim.range).toBeDefined();
      expect(timeDim.range.length).toBe(2);
      expect(timeDim.range[0]).toBe(0); // min time
      expect(timeDim.range[1]).toBe(9); // max time (10 steps: 0-9)
    });

    it('should have correct dimension metadata for slicing', async () => {
      const storePath = path.join(FIXTURES_DIR, 'test_4d.zarr');
      const rawStore = new FileSystemStore(storePath);
      const store = await zarr.tryWithConsolidated(rawStore);
      const rootLoc = zarr.root(store);

      const sceneGroup = await zarr.open(rootLoc, { kind: 'group' });
      const sceneAttrs = sceneGroup.attrs as any;

      const dimensions = sceneAttrs.scene_dimensions.dimensions;

      // Verify displayed vs non-displayed dimensions
      const displayedDims = dimensions.filter((d: any) => d.display);
      const nonDisplayedDims = dimensions.filter((d: any) => !d.display);

      // Should have 3 displayed dimensions (x, y, z)
      expect(displayedDims.length).toBe(3);

      // Should have 1 non-displayed dimension (time)
      expect(nonDisplayedDims.length).toBe(1);
      expect(nonDisplayedDims[0].name).toBe('time');
    });
  });
});
