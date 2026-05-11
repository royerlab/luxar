/**
 * tests that scalar attributes are bound on geometries when
 * `data.scalars` / `processed.startScalars`/`endScalars` are supplied.
 *
 * The C1 fail-closed guard checks `geometry.hasAttribute('scalar')` for
 * Points and `aStartScalar`/`aEndScalar` for Lines — without C4's
 * binding, that guard would always trip. These tests demonstrate the
 * unblocking path.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import { createInstancedLinesMesh } from '../../../rendering/line-geometry';
import { LineMaterial } from '../../../rendering/line-material';
import { supportsScalarColormap } from '../../../rendering/material-colormap-helpers';
import type { LoadedPointsData } from '../../../data/data-loader-types';

describe('scalar attribute binding', () => {
  describe('Points', () => {
    it('binds `scalar` attribute when data.scalars is supplied', () => {
      const factory = new NodeFactory();
      const data: LoadedPointsData = {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        scalars: new Float32Array([0.1, 0.5, 0.9]),
        pointCount: 3,
        ndim: 3,
        metadata: {
          totalPoints: 3,
          loadedPoints: 3,
          bounds: new THREE.Box3(),
          usedSpatialIndex: false,
        },
      };
      const geometry = factory.createPointsGeometry(data);
      expect(geometry.hasAttribute('scalar')).toBe(true);
      const scalarAttr = geometry.getAttribute('scalar') as THREE.BufferAttribute;
      expect(scalarAttr.itemSize).toBe(1);
      expect(scalarAttr.count).toBe(3);
      // Colormap guard passes when scalar data is bound.
      expect(supportsScalarColormap('points', geometry)).toBe(true);
    });

    it('does NOT bind `scalar` when data.scalars is absent', () => {
      const factory = new NodeFactory();
      const data: LoadedPointsData = {
        positions: new Float32Array([0, 0, 0]),
        pointCount: 1,
        ndim: 3,
        metadata: {
          totalPoints: 1,
          loadedPoints: 1,
          bounds: new THREE.Box3(),
          usedSpatialIndex: false,
        },
      };
      const geometry = factory.createPointsGeometry(data);
      expect(geometry.hasAttribute('scalar')).toBe(false);
      // Colormap guard fails closed when scalar data is absent.
      expect(supportsScalarColormap('points', geometry)).toBe(false);
    });
  });

  describe('Lines', () => {
    it('binds aStartScalar/aEndScalar when both are supplied in config', () => {
      const config = {
        startPositions: new Float32Array([0, 0, 0]),
        endPositions: new Float32Array([1, 0, 0]),
        startColors: new Float32Array([1, 1, 1]),
        endColors: new Float32Array([1, 1, 1]),
        startWidths: new Float32Array([0.1]),
        endWidths: new Float32Array([0.1]),
        startSharpness: new Float32Array([2.0]),
        endSharpness: new Float32Array([2.0]),
        segmentLengths: new Float32Array([1.0]),
        startClipped: new Uint8Array([0]),
        endClipped: new Uint8Array([0]),
        startScalars: new Float32Array([0.0]),
        endScalars: new Float32Array([1.0]),
        segmentCount: 1,
      };
      const material = new LineMaterial();
      const mesh = createInstancedLinesMesh(config, material);
      const geometry = mesh.geometry;
      expect(geometry.hasAttribute('aStartScalar')).toBe(true);
      expect(geometry.hasAttribute('aEndScalar')).toBe(true);
      expect(supportsScalarColormap('lines', geometry)).toBe(true);
    });

    it('does NOT bind scalar attributes when only one side is supplied (fail-closed)', () => {
      const config = {
        startPositions: new Float32Array([0, 0, 0]),
        endPositions: new Float32Array([1, 0, 0]),
        startColors: new Float32Array([1, 1, 1]),
        endColors: new Float32Array([1, 1, 1]),
        startWidths: new Float32Array([0.1]),
        endWidths: new Float32Array([0.1]),
        startSharpness: new Float32Array([2.0]),
        endSharpness: new Float32Array([2.0]),
        segmentLengths: new Float32Array([1.0]),
        startClipped: new Uint8Array([0]),
        endClipped: new Uint8Array([0]),
        startScalars: new Float32Array([0.0]),
        // endScalars missing — broken pair
        segmentCount: 1,
      };
      const material = new LineMaterial();
      const mesh = createInstancedLinesMesh(config, material);
      const geometry = mesh.geometry;
      expect(geometry.hasAttribute('aStartScalar')).toBe(false);
      expect(geometry.hasAttribute('aEndScalar')).toBe(false);
      expect(supportsScalarColormap('lines', geometry)).toBe(false);
    });
  });
});
