/**
 * tests that scalar attributes are bound on geometries when
 * `data.scalars` / `processed.startScalars`/`endScalars` are supplied.
 *
 * The C1 fail-closed guard checks `geometry.hasAttribute('aScalar')` for
 * Points and `aStartScalar`/`aEndScalar` for Lines — without C4's
 * binding, that guard would always trip. These tests demonstrate the
 * unblocking path.
 */
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import { createInstancedLinesMesh } from '../../../rendering/line-geometry';
import { LineMaterial } from '../../../rendering/materials/line/material-glsl';
import { supportsScalarColormap } from '../../../rendering/material-colormap-helpers';
import type { LoadedPointsData, DataLoader } from '../../../data/data-loader-types';
import type { PointsMetadata } from '../../../types/points';
import type { LinesMetadata, LinesDataLoader } from '../../../types/lines';

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
      expect(geometry.hasAttribute('aScalar')).toBe(true);
      const scalarAttr = geometry.getAttribute('aScalar') as THREE.BufferAttribute;
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
      expect(geometry.hasAttribute('aScalar')).toBe(false);
      // Colormap guard fails closed when scalar data is absent.
      expect(supportsScalarColormap('points', geometry)).toBe(false);
    });

    // Regression: the placeholder-first loading pattern created the
    // material before scalars streamed in, so the colormap guard tripped
    // on the empty placeholder geometry, USE_COLORMAP was suppressed, and
    // the colormap was never re-enabled once the real scalars arrived —
    // leaving scalar+colormap points rendering white. createEmptyPointsNode
    // now pre-binds an empty `aScalar` when the node declares
    // has_scalars + colormap, so the guard passes and the material is built
    // colormap-enabled up front.
    it('placeholder enables colormap when attrs declare has_scalars + colormap', () => {
      const factory = new NodeFactory();
      const attrs = {
        n_points: 3,
        has_scalars: true,
        colormap: 'viridis',
        scalar_data_range: [0, 1],
      } as unknown as PointsMetadata;
      const loader = { dispose: vi.fn() } as unknown as DataLoader;

      const placeholder = factory.createEmptyPointsNode('/spiral', attrs, loader);

      // Empty `aScalar` is bound so the fail-closed guard passes even
      // though the placeholder carries zero points.
      expect(placeholder.geometry.hasAttribute('aScalar')).toBe(true);
      expect(supportsScalarColormap('points', placeholder.geometry)).toBe(true);

      // The material is colormap-enabled from the start (no suppression).
      const material = placeholder.material as THREE.Material;
      expect(material.defines && 'USE_COLORMAP' in material.defines).toBe(true);
    });

    it('placeholder does NOT bind aScalar when no colormap is declared', () => {
      const factory = new NodeFactory();
      const attrs = {
        n_points: 3,
        has_scalars: true,
        // colormap absent — nothing to map scalars through
      } as unknown as PointsMetadata;
      const loader = { dispose: vi.fn() } as unknown as DataLoader;

      const placeholder = factory.createEmptyPointsNode('/spiral', attrs, loader);
      expect(placeholder.geometry.hasAttribute('aScalar')).toBe(false);
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

    // Regression (mirrors the Points placeholder case above): the lines
    // placeholder omitted scalar arrays, so the colormap guard tripped on
    // the empty placeholder, logged "Colormap suppressed", and the LUT was
    // never re-enabled once real scalars streamed in (commit writes into the
    // existing placeholder geometry) — leaving colormapped lines white.
    // createEmptyLinesNode now pre-binds empty start/end scalars when the
    // node declares colormap + has_scalars.
    it('placeholder enables colormap when attrs declare has_scalars + colormap', () => {
      const factory = new NodeFactory();
      const nodeAttrs = { has_scalars: true, colormap: 'viridis', scalar_data_range: [0, 1] };
      const attrs = {
        type: 'lines',
        n_vertices: 0,
        n_segments: 0,
        ndim: 3,
        max_width: 1.0,
        has_colors: false,
        has_sharpness: false,
        has_scalars: true,
        colormap: 'viridis',
        scalar_data_range: [0, 1],
      } as unknown as LinesMetadata;
      const loader = { dispose: vi.fn() } as unknown as LinesDataLoader;

      const placeholder = factory.createEmptyLinesNode('/streamlines', nodeAttrs, attrs, loader);

      // Empty start/end scalars are bound so the fail-closed guard passes
      // even though the placeholder carries zero segments.
      expect(placeholder.geometry.hasAttribute('aStartScalar')).toBe(true);
      expect(placeholder.geometry.hasAttribute('aEndScalar')).toBe(true);
      expect(supportsScalarColormap('lines', placeholder.geometry)).toBe(true);

      // The material is colormap-enabled from the start (no suppression).
      const material = placeholder.material as THREE.Material;
      expect(material.defines && 'USE_COLORMAP' in material.defines).toBe(true);
    });

    it('placeholder does NOT bind scalars when no colormap is declared', () => {
      const factory = new NodeFactory();
      const nodeAttrs = { has_scalars: true };
      const attrs = {
        type: 'lines',
        n_vertices: 0,
        n_segments: 0,
        ndim: 3,
        max_width: 1.0,
        has_colors: false,
        has_sharpness: false,
        has_scalars: true,
      } as unknown as LinesMetadata;
      const loader = { dispose: vi.fn() } as unknown as LinesDataLoader;

      const placeholder = factory.createEmptyLinesNode('/streamlines', nodeAttrs, attrs, loader);
      expect(placeholder.geometry.hasAttribute('aStartScalar')).toBe(false);
      expect(placeholder.geometry.hasAttribute('aEndScalar')).toBe(false);
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
