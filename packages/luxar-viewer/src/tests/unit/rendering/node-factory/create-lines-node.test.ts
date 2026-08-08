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
 * 2. Line materials are PER NODE (each carries the node's own
 *    `uLineTex`, since the texture-backed storage migration; the
 *    line-material LRU is gone): `createLinesNode` gets a fresh
 *    node-owned material from the manager on every call, applies the
 *    colormap DIRECTLY to it (the historical clone-on-divergence dance
 *    is gone — mirrors `createPointsMaterial`), stamps
 *    `_layerMaterialCloned: true`, and binds the geometry-owned line
 *    texture at creation via `syncLineMaterialWithGeometry`.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type * as THREE from 'three';
import { NodeFactory } from '../../../../rendering/node-factory';
import {
  materialManager,
  __resetMaterialManagerForTests,
} from '../../../../rendering/material-manager';
import { applyEffectiveAttrs } from '../../../../data/scene-loader/view-state/effective-attrs';
import type { SceneNode } from '../../../../data/data-loader-types';
import type { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import type { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import type { PickingSystem } from '../../../../rendering/picking/picking-system';
import { getLineTexture, type InstancedLinesMeshConfig } from '../../../../rendering/line-geometry';
import { LINE_JOIN_UNIFORM } from '../../../../types/line-join';
import type { LinesMetadata, LinesDataLoader } from '../../../../types/lines';
import { log } from '../../../../utils/log';

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
    startJointCode: new Float32Array([0]),
    endJointCode: new Float32Array([0]),
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

    it('reads the join style from nodeAttrs, onto BOTH the visual and pick materials', () => {
      // `join` is a COMPOSITING attr, so on a partitioned lines node it arrives
      // here composed from the wrapper and is absent from the raw leaf attrs.
      // The pick material must get the same style — the pick pass builds the same
      // screen-space quad, so a divergence makes a mitred corner unpickable.
      const factory = new NodeFactory();
      const pickIds: number[] = [];
      const registered: THREE.Mesh[] = [];
      factory.setPickingSystem({
        allocatePickId: () => {
          pickIds.push(pickIds.length + 1);
          return pickIds.length;
        },
        registerNode: (_main: THREE.Object3D, pick: THREE.Mesh) => registered.push(pick),
      } as unknown as PickingSystem);

      const mesh = factory.createLinesNode(
        '/tracks/part_0',
        { join: 'none' },
        rawAttrs,
        makeProcessed(),
        makeLoader()
      );

      const material = mesh.material as LineMaterial;
      expect(material.uniforms.uLineJoin.value).toBe(LINE_JOIN_UNIFORM.none);
      const pick = registered[0].material as LinePickingMaterial;
      expect(pick.uniforms.uLineJoin.value).toBe(LINE_JOIN_UNIFORM.none);
    });

    it('warns and falls back to the default on an unrecognised join style', () => {
      // A typo must never quietly mean "no joins": the file would render with the
      // default and the author would have nothing to go on.
      const warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/tracks',
        { join: 'mitre' },
        rawAttrs,
        makeProcessed(),
        makeLoader()
      );

      const material = mesh.material as LineMaterial;
      expect(material.uniforms.uLineJoin.value).toBe(LINE_JOIN_UNIFORM.miter);
      expect(warn).toHaveBeenCalledWith(expect.anything(), expect.stringContaining('"mitre"'));
      warn.mockRestore();
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

  describe('per-node material: colormap applies directly (no clone)', () => {
    const colormapNodeAttrs = {
      colormap: 'viridis',
      has_scalars: true,
      scalar_data_range: [0.5, 2.5],
    };

    it('the material is node-owned — distinct across two creations with identical attrs', () => {
      // Line materials are per node (each carries its own uLineTex), so
      // two nodes created from the SAME attrs must never share one.
      const factory = new NodeFactory();
      const meshA = factory.createLinesNode(
        '/streamlines-a',
        {},
        rawAttrs,
        makeProcessed(),
        makeLoader()
      );
      const meshB = factory.createLinesNode(
        '/streamlines-b',
        {},
        rawAttrs,
        makeProcessed(),
        makeLoader()
      );
      expect(meshA.material).not.toBe(meshB.material);

      // Per-node from creation: LayersPanel / LOD-cross-fade mutate the
      // material directly instead of clone-on-first-use.
      expect(meshA.userData._layerMaterialCloned).toBe(true);
      expect(meshB.userData._layerMaterialCloned).toBe(true);

      // The per-node materials live only in the camera-update registry.
      const stats = materialManager.getCacheStats();
      expect(stats.totalRegistered).toBe(2);
    });

    it('applies the colormap directly to THE mesh material (no clone, single registration)', () => {
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/streamlines',
        colormapNodeAttrs,
        rawAttrs,
        makeProcessed(/* withScalars */ true),
        makeLoader()
      );

      // The colormap landed on the material actually attached to the
      // mesh — the clone-era indirection (cached original + clone) is gone.
      const material = mesh.material as LineMaterial;
      expect('USE_COLORMAP' in material.defines).toBe(true);
      expect(material.uniforms.uColormapTex.value).not.toBeNull();
      expect(material.uniforms.uScalarMin.value).toBe(0.5);
      expect(material.uniforms.uScalarScale.value).toBeCloseTo(0.5, 5); // 1/(2.5-0.5)
      expect(material.userData.scalarRange).toEqual([0.5, 2.5]);

      // Exactly ONE material exists for this node (clone-era: 2 — the
      // cached original plus the clone).
      const stats = materialManager.getCacheStats();
      expect(stats.totalRegistered).toBe(1);
    });

    it('binds the geometry-owned line texture (uLineTex) on the material at creation', () => {
      // syncLineMaterialWithGeometry runs inside createLinesNode so a
      // mesh created WITH data renders before any commit.
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/streamlines',
        {},
        rawAttrs,
        makeProcessed(),
        makeLoader()
      );
      const geometryTexture = getLineTexture(mesh.geometry);
      expect(geometryTexture).not.toBeNull();
      const material = mesh.material as LineMaterial;
      expect(material.getLineTexture()).toBe(geometryTexture);
    });

    it('manager dispose() disposes the node-owned material via the registry', () => {
      // Per-node materials sit in NO cache, so the registry is the only
      // path manager teardown has to them — a registry miss would leak
      // the GPU program on embedder re-init.
      const factory = new NodeFactory();
      const mesh = factory.createLinesNode(
        '/streamlines',
        colormapNodeAttrs,
        rawAttrs,
        makeProcessed(/* withScalars */ true),
        makeLoader()
      );
      const material = mesh.material as LineMaterial;
      const materialDispose = vi.spyOn(material, 'dispose');

      materialManager.dispose();

      expect(materialDispose).toHaveBeenCalled();
    });
  });
});
