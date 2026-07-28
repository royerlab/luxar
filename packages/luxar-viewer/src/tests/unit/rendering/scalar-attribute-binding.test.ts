/**
 * tests that scalar data is bound on geometries when
 * `data.scalars` / `processed.startScalars`/`endScalars` are supplied.
 *
 * The C1 fail-closed guard checks the `userData.hasScalars` stamp for
 * BOTH Points and Lines: scalar data rides texel2.x of the fixed-layout
 * point texture and texel5.xy of the fixed-layout line texture, so
 * presence is no longer readable off a geometry attribute — without the
 * stamp, that guard would always trip. These tests demonstrate the
 * unblocking path.
 */
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import { getPointTexture } from '../../../rendering/point-geometry';
import {
  LINE_FLOATS_PER_SEGMENT,
  POINT_FLOATS_PER_POINT,
} from '../../../rendering/element-texture-layout';
import { createInstancedLinesMesh, getLineTexture } from '../../../rendering/line-geometry';
import { LineMaterial } from '../../../rendering/materials/line/material-glsl';
import { supportsScalarColormap } from '../../../rendering/material-colormap-helpers';
import type { LoadedPointsData, DataLoader } from '../../../data/data-loader-types';
import type { PointsMetadata } from '../../../types/points';
import type { LinesMetadata, LinesDataLoader } from '../../../types/lines';

describe('scalar attribute binding', () => {
  describe('Points', () => {
    it('stamps userData.hasScalars + writes texel2.x when data.scalars is supplied', () => {
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
      // Scalar presence is the userData stamp (the fixed texel layout
      // always has a scalar slot, so no attribute probe exists anymore).
      expect(geometry.userData.hasScalars).toBe(true);
      // The scalar VALUES land in texel2.x (offset 8 of the 12-float
      // per-point stride) of the geometry-attached point texture.
      const texData = getPointTexture(geometry)!.image.data as Float32Array;
      expect(texData[0 * POINT_FLOATS_PER_POINT + 8]).toBeCloseTo(0.1, 5);
      expect(texData[1 * POINT_FLOATS_PER_POINT + 8]).toBeCloseTo(0.5, 5);
      expect(texData[2 * POINT_FLOATS_PER_POINT + 8]).toBeCloseTo(0.9, 5);
      // Colormap guard passes when scalar data is bound.
      expect(supportsScalarColormap('points', geometry)).toBe(true);
    });

    it('does NOT stamp hasScalars when data.scalars is absent', () => {
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
      expect(geometry.userData.hasScalars).toBe(false);
      // Colormap guard fails closed when scalar data is absent.
      expect(supportsScalarColormap('points', geometry)).toBe(false);
    });

    // Regression: the placeholder-first loading pattern created the
    // material before scalars streamed in, so the colormap guard tripped
    // on the empty placeholder geometry, USE_COLORMAP was suppressed, and
    // the colormap was never re-enabled once the real scalars arrived —
    // leaving scalar+colormap points rendering white. createEmptyPointsNode
    // declares an empty `scalars` field when the node has
    // has_scalars + colormap, which createPointsGeometry turns into the
    // userData.hasScalars stamp, so the guard passes and the material is
    // built colormap-enabled up front.
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

      // hasScalars is stamped so the fail-closed guard passes even
      // though the placeholder carries zero points.
      expect(placeholder.geometry.userData.hasScalars).toBe(true);
      expect(supportsScalarColormap('points', placeholder.geometry)).toBe(true);

      // The material is colormap-enabled from the start (no suppression).
      const material = placeholder.material as THREE.Material;
      expect(material.defines && 'USE_COLORMAP' in material.defines).toBe(true);
    });

    it('placeholder does NOT stamp hasScalars when no colormap is declared', () => {
      const factory = new NodeFactory();
      const attrs = {
        n_points: 3,
        has_scalars: true,
        // colormap absent — nothing to map scalars through
      } as unknown as PointsMetadata;
      const loader = { dispose: vi.fn() } as unknown as DataLoader;

      const placeholder = factory.createEmptyPointsNode('/spiral', attrs, loader);
      expect(placeholder.geometry.userData.hasScalars).toBe(false);
      expect(supportsScalarColormap('points', placeholder.geometry)).toBe(false);
    });
  });

  describe('Lines', () => {
    it('stamps userData.hasScalars + writes texel5.xy when both scalars are supplied', () => {
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
        startCapSuppression: new Float32Array([0]),
        endCapSuppression: new Float32Array([0]),
        startScalars: new Float32Array([0.25]),
        endScalars: new Float32Array([0.75]),
        segmentCount: 1,
      };
      const material = new LineMaterial();
      const mesh = createInstancedLinesMesh(config, material);
      const geometry = mesh.geometry;
      // Scalar presence is the userData stamp (the fixed 6-texel layout
      // always has the scalar slots, so no attribute probe exists anymore).
      expect(geometry.userData.hasScalars).toBe(true);
      // The scalar VALUES land in texel5.xy (offsets 20/21 of the 24-float
      // per-segment stride) of the geometry-attached line texture.
      const texData = getLineTexture(geometry)!.image.data as Float32Array;
      expect(texData[0 * LINE_FLOATS_PER_SEGMENT + 20]).toBeCloseTo(0.25, 5);
      expect(texData[0 * LINE_FLOATS_PER_SEGMENT + 21]).toBeCloseTo(0.75, 5);
      // Colormap guard passes when scalar data is bound.
      expect(supportsScalarColormap('lines', geometry)).toBe(true);
    });

    // Regression (mirrors the Points placeholder case above): the lines
    // placeholder omitted scalar arrays, so the colormap guard tripped on
    // the empty placeholder, logged "Colormap suppressed", and the LUT was
    // never re-enabled once real scalars streamed in (commit writes into the
    // existing placeholder geometry) — leaving colormapped lines white.
    // createEmptyLinesNode declares empty start/end scalars when the node
    // declares colormap + has_scalars, which createInstancedLinesMesh turns
    // into the userData.hasScalars stamp, so the guard passes and the
    // per-node material is colormap-enabled up front.
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

      // hasScalars is stamped so the fail-closed guard passes even
      // though the placeholder carries zero segments.
      expect(placeholder.geometry.userData.hasScalars).toBe(true);
      expect(supportsScalarColormap('lines', placeholder.geometry)).toBe(true);

      // The material is colormap-enabled from the start (no suppression).
      const material = placeholder.material as THREE.Material;
      expect(material.defines && 'USE_COLORMAP' in material.defines).toBe(true);
    });

    it('placeholder does NOT stamp hasScalars when no colormap is declared', () => {
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
      expect(placeholder.geometry.userData.hasScalars).toBe(false);
      expect(supportsScalarColormap('lines', placeholder.geometry)).toBe(false);
    });

    it('does NOT stamp hasScalars when only one side is supplied (fail-closed)', () => {
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
        startCapSuppression: new Float32Array([0]),
        endCapSuppression: new Float32Array([0]),
        startScalars: new Float32Array([0.0]),
        // endScalars missing — broken pair
        segmentCount: 1,
      };
      const material = new LineMaterial();
      const mesh = createInstancedLinesMesh(config, material);
      const geometry = mesh.geometry;
      expect(geometry.userData.hasScalars).toBe(false);
      expect(supportsScalarColormap('lines', geometry)).toBe(false);
    });
  });
});
