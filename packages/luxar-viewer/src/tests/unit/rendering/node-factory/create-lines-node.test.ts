/**
 * Unit tests for `createLinesNode` material wiring
 * (`src/rendering/node-factory/create-lines-node.ts`).
 *
 * Two regression areas:
 *
 * 1. COMPOSED attrs drive the material. The loader passes
 *    `applyEffectiveAttrs(node)` as `nodeAttrs` (ancestor-composed
 *    opacity/gamma/intensity/offset/blending_mode) and the raw
 *    `node.attrs` as `attrs`. The material must read the composed
 *    values — reading the RAW attrs silently dropped ancestor
 *    contributions until the first panel interaction (the exact bug
 *    `createGSplatsNode` documents and points avoids by passing the
 *    composed attrs as its sole attrs param).
 *
 * 2. The colormap clone path must NOT orphan the pooled original: the
 *    original stays in the LRU cache serving future cache hits, so it
 *    must keep receiving `updateCameraParams` (stale-resolution line
 *    widths otherwise) and must still be reachable by manager
 *    `dispose()` (GPU program leak on embedder re-init otherwise).
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../../rendering/node-factory';
import {
  materialManager,
  __resetMaterialManagerForTests,
} from '../../../../rendering/material-manager';
import { applyEffectiveAttrs } from '../../../../data/scene-loader/view-state/effective-attrs';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import type { InstancedLinesMeshConfig } from '../../../../rendering/line-geometry';
import type { LinesMetadata, LinesDataLoader } from '../../../../types/lines';

/** One-segment processed config; optionally scalar-bearing (colormap path). */
function makeProcessed(withScalars = false): InstancedLinesMeshConfig {
  const config: InstancedLinesMeshConfig = {
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
    segmentCount: 1,
  };
  if (withScalars) {
    config.startScalars = new Float32Array([0.0]);
    config.endScalars = new Float32Array([1.0]);
  }
  return config;
}

function makeLoader(): LinesDataLoader {
  return { dispose: vi.fn() } as unknown as LinesDataLoader;
}

const rawAttrs = {
  type: 'lines',
  n_vertices: 2,
  n_segments: 1,
  ndim: 3,
  max_width: 1.0,
} as unknown as LinesMetadata;

describe('createLinesNode material wiring', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  describe('composed effective attrs drive the material', () => {
    it('ancestor opacity 0.5 × own opacity 0.5 → material opacity 0.25', () => {
      // Real composition path: group(0.5) → lines leaf(0.5).
      const leaf: SceneNode = {
        path: '/group/streamlines',
        type: 'lines',
        attrs: { ...rawAttrs, opacity: 0.5 } as unknown as SceneNode['attrs'],
        hasSpatialIndex: false,
      };
      const root = {
        path: '/',
        type: 'group',
        attrs: { opacity: 0.5 },
        children: [{ path: '/group', type: 'group', attrs: { opacity: 0.5 }, children: [leaf] }],
      } as unknown as SceneNode;
      // Root contributes identity here; the /group ancestor carries 0.5.
      root.attrs = {};

      const composed = applyEffectiveAttrs(root, leaf);
      expect(composed.opacity).toBeCloseTo(0.25, 6);

      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        leaf.path,
        composed,
        leaf.attrs as unknown as LinesMetadata,
        makeProcessed(),
        makeLoader()
      );

      const material = mesh.material as LineMaterial;
      // The RAW attrs say 0.5 — the composed 0.25 must win.
      expect(material.uniforms.uOpacity.value).toBeCloseTo(0.25, 6);
    });

    it('reads opacity/gamma/intensity/offset/blending_mode from nodeAttrs, not raw attrs', () => {
      const factory = new NodeFactory();
      const nodeAttrs = {
        opacity: 0.25,
        gamma: 2.2,
        intensity: 2.0,
        offset: 0.1,
        blending_mode: 'normal',
      };
      // Raw attrs carry DIFFERENT values for every composited field —
      // if any leaks into the material, the assertion below catches it.
      const divergentRaw = {
        ...rawAttrs,
        opacity: 0.9,
        gamma: 1.0,
        intensity: 1.0,
        offset: 0.0,
        blending_mode: 'additive',
      } as unknown as LinesMetadata;

      const mesh = factory.createLinesNode(
        '/streamlines',
        nodeAttrs,
        divergentRaw,
        makeProcessed(),
        makeLoader()
      );

      const material = mesh.material as LineMaterial;
      expect(material.uniforms.uOpacity.value).toBeCloseTo(0.25, 6);
      expect(material.userData.gamma).toBeCloseTo(2.2, 6);
      expect(material.uniforms.uIntensity.value).toBeCloseTo(2.0, 6);
      expect(material.uniforms.uOffset.value).toBeCloseTo(0.1, 6);
      expect(material.userData.blendingMode).toBe('normal');
    });

    it('per-leaf geometry attrs (max_width) still come from raw attrs', () => {
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/streamlines',
        { opacity: 0.5 }, // composed attrs carry no max_width
        { ...rawAttrs, max_width: 4.0 } as unknown as LinesMetadata,
        makeProcessed(),
        makeLoader()
      );
      expect(mesh.userData.maxWidth).toBe(4.0);
    });
  });

  describe('colormap clone path leaves the cached original live', () => {
    // Material props derived from empty nodeAttrs = the getLineMaterial
    // defaults, so a direct getLineMaterial call below shares the cache key.
    const defaultProps = {
      opacity: 1.0,
      gamma: 1.0,
      intensity: 1.0,
      offset: 0.0,
      blendingMode: 'additive' as const,
    };
    const colormapNodeAttrs = {
      colormap: 'viridis',
      has_scalars: true,
      scalar_data_range: [0, 1],
    };

    it('a cache hit after the clone still receives updateCameraParams', () => {
      // Prime the cache with the original, then trigger the clone path.
      const original = materialManager.getLineMaterial(defaultProps) as LineMaterial;
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/streamlines',
        colormapNodeAttrs,
        rawAttrs,
        makeProcessed(/* withScalars */ true),
        makeLoader()
      );

      // The mesh got a clone; the original still serves the cache.
      expect(mesh.material).not.toBe(original);
      expect(materialManager.getLineMaterial(defaultProps)).toBe(original);

      // The cached original must keep tracking global camera params —
      // a detached-but-cached entry would render stale line widths
      // after the next resize/FOV change.
      materialManager.updateCameraParams(Math.PI / 3, new THREE.Vector2(800, 600), false);
      expect(original.uniforms.uResolution.value.x).toBe(800);
      expect(original.uniforms.uResolution.value.y).toBe(600);
    });

    it('manager dispose() disposes both the cached original and the clone', () => {
      const original = materialManager.getLineMaterial(defaultProps) as LineMaterial;
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/streamlines',
        colormapNodeAttrs,
        rawAttrs,
        makeProcessed(/* withScalars */ true),
        makeLoader()
      );
      const clone = mesh.material as LineMaterial;
      expect(clone).not.toBe(original);

      const originalDispose = vi.spyOn(original, 'dispose');
      const cloneDispose = vi.spyOn(clone, 'dispose');

      materialManager.dispose();

      // The original was in the LRU cache when dispose() ran; it must be
      // reachable through the registries (a detached-while-cached original
      // sat in neither and leaked its GPU program on embedder re-init).
      expect(originalDispose).toHaveBeenCalled();
      expect(cloneDispose).toHaveBeenCalled();
    });
  });
});
