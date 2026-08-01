/**
 * Regression (#1082): an authored `intensity`/`offset` on a COLORMAPPED
 * points node must define the display WINDOW only — never a post-LUT color
 * gain on top of it.
 *
 * The point fragment shader always multiplies `vColor * uIntensity + uOffset`
 * post-LUT (the LUXAR_NO_GOG compile-out fires only for the identity
 * intensity==1 && offset==0), so leaving an authored gain stamped while ALSO
 * inverting it into the scalar window applies the value twice, with two
 * different meanings. The fix mirrors the gsplat fix (#1081):
 *
 *  - Node factory (`layer=false`): stamp the color GOG at IDENTITY when a
 *    colormap is applied, and derive the scalar window from the authored
 *    gain/offset exactly as the panel does — so `layer=false` matches
 *    `layer=true`. The identity-vs-window decision keys on the RAW LEAF gain,
 *    which is threaded through the points factory as `rawAttrs`.
 *  - `applyColorAdjustments` (`layer=true`): in colormap mode push the window
 *    via `updateScalarRange` AND actively reset the color GOG to identity.
 *
 * Direct-color nodes are unaffected — they still receive the authored gain.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import { __resetMaterialManagerForTests } from '../../../rendering/material-manager';
import { PointMaterial } from '../../../rendering/materials/point/material-glsl';
import { getColormapTexture } from '../../../rendering/colormap-textures';
import { applyColorAdjustments } from '../../../ui/layers/luxar-material';
import { computeDisplayRange, computeUniforms } from '../../../rendering/display-range';
import type { PointsMetadata } from '../../../types/points';
import type { DataLoader, LoadedPointsData } from '../../../data/data-loader-types';

/** createEmptyPointsNode only stashes the loader in userData — never calls it. */
function makeLoader(): DataLoader {
  return { dispose: vi.fn() } as unknown as DataLoader;
}

/** One point, optionally with a scalar field for the fail-closed guard. */
function makePointsData(withScalars: boolean = true): LoadedPointsData {
  const data: LoadedPointsData = {
    positions: new Float32Array([0, 0, 0]) as LoadedPointsData['positions'],
    pointCount: 1,
    ndim: 3,
    metadata: {
      totalPoints: 1,
      loadedPoints: 1,
      bounds: new THREE.Box3(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0)),
      usedSpatialIndex: false,
      dtypes: {},
    },
  };
  if (withScalars) data.scalars = new Float32Array([1]) as LoadedPointsData['scalars'];
  return data;
}

describe('#1082 colormapped points authored intensity is a window, not a double gain', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  describe('node-factory path (layer=false)', () => {
    it('colormapped node: color GOG is IDENTITY, scalar window = computeDisplayRange(intensity)', () => {
      // Leaf authors intensity=0.09 with no ancestor gain → composed == raw.
      const composed = {
        colormap: 'viridis',
        has_scalars: true,
        intensity: 0.09,
        offset: 0.0,
      } as unknown as PointsMetadata;
      const leafRaw = { intensity: 0.09, offset: 0.0 } as unknown as PointsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader(), leafRaw);

      const mat = mesh.material as PointMaterial;
      // Colormap active.
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      // Color GOG is identity — the authored 0.09 is NOT stamped as a gain.
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // Scalar window inverts the authored gain: 0.09 → [0, ~11.11].
      const { min, max } = computeDisplayRange(0.09, 0.0);
      expect(max).toBeCloseTo(11.111, 3);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(min, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (max - min), 5);
    });

    it('colormapped node with identity gain keeps the data-range window', () => {
      const composed = {
        colormap: 'viridis',
        has_scalars: true,
        intensity: 1.0,
        offset: 0.0,
        scalar_data_range: [0, 200],
      } as unknown as PointsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader());

      const mat = mesh.material as PointMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // No authored window → data range drives the LUT lookup.
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(0, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / 200, 5);
    });

    it('ancestor-only gain (composed != 1, raw leaf identity): gain folds onto the data-range window', () => {
      // Colormapped leaf with NO own gain under an ancestor group with
      // intensity=0.5: the loader passes the COMPOSED 0.5 as the rendering
      // attrs while the RAW leaf has no gain. The identity decision follows
      // the RAW leaf gain (identity), so the window STARTS from
      // scalar_data_range — but the ancestor gain still folds onto it:
      // [0,100] → window uniforms (0.01, 0) → × ancestor 0.5 → (0.005, 0) →
      // window [0, 200].
      const composed = {
        colormap: 'viridis',
        has_scalars: true,
        intensity: 0.5,
        offset: 0.0,
        scalar_data_range: [0, 100],
      } as unknown as PointsMetadata;
      const leafRaw = {} as unknown as PointsMetadata; // raw leaf: no gain → identity
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader(), leafRaw);

      const mat = mesh.material as PointMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // Data-range window with the ancestor 0.5 folded in: [0, 200].
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(0, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / 200, 5);
    });

    it('ancestor gain over a NON-zero-min data range: window matches the panel (additive-offset spec)', () => {
      // scalar_data_range [10, 110] → window uniforms w = (0.01, -0.1). An
      // ancestor gain (0.5, 0.2) composes per the attrs-composer spec —
      // intensity MULTIPLIES, offset ADDS (deliberately NOT nested-affine
      // `O_parent·I_child + O_child`; see attrs-composer.ts) — so the panel
      // pushes computeDisplayRange(0.5·0.01, 0.2 + (-0.1)) = [-20, 180].
      // Locks the fold to the panel's composition for windows with a
      // non-zero w.offset, where the two models diverge.
      const composed = {
        colormap: 'viridis',
        has_scalars: true,
        intensity: 0.5,
        offset: 0.2,
        scalar_data_range: [10, 110],
      } as unknown as PointsMetadata;
      const leafRaw = {} as unknown as PointsMetadata; // raw leaf: no gain → identity
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader(), leafRaw);

      const mat = mesh.material as PointMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      const w = computeUniforms(10, 110);
      const expected = computeDisplayRange(0.5 * w.intensity, 0.2 + w.offset);
      expect(expected.min).toBeCloseTo(-20, 5);
      expect(expected.max).toBeCloseTo(180, 5);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(expected.min, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (expected.max - expected.min), 5);
      // First panel interaction pushes the SAME composed gain through
      // applyColorAdjustments — the window must not flip.
      applyColorAdjustments(mat, 1.0, 0.5 * w.intensity, 0.2 + w.offset);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(expected.min, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (expected.max - expected.min), 5);
    });

    it('leaf-authored window under an ancestor gain uses the composed window', () => {
      const composed = {
        colormap: 'viridis',
        has_scalars: true,
        intensity: 0.25,
        offset: 0.0,
        scalar_data_range: [1.16, 45.9],
      } as unknown as PointsMetadata;
      const leafRaw = { intensity: 0.5, offset: 0.0 } as unknown as PointsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader(), leafRaw);

      const mat = mesh.material as PointMaterial;
      const { min, max } = computeDisplayRange(0.25, 0.0);
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(min, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (max - min), 5);
    });

    it('direct-color node (no colormap) still receives the authored gain', () => {
      const composed = { intensity: 0.09, offset: 0.02 } as unknown as PointsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader());

      const mat = mesh.material as PointMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(false);
      expect(mat.uniforms.uIntensity.value).toBeCloseTo(0.09, 6);
      expect(mat.uniforms.uOffset.value).toBeCloseTo(0.02, 6);
    });

    it('unresolvable colormap keeps the authored direct-color GOG', () => {
      const composed = {
        colormap: 'definitely_not_a_real_colormap',
        has_scalars: true,
        intensity: 0.4,
        offset: 0.02,
      } as unknown as PointsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createEmptyPointsNode('/points', composed, makeLoader(), composed);

      const mat = mesh.material as PointMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(false);
      expect(mat.uniforms.uIntensity.value).toBeCloseTo(0.4, 6);
      expect(mat.uniforms.uOffset.value).toBeCloseTo(0.02, 6);
    });

    it('declared colormap without bound scalar data keeps the authored direct-color GOG', () => {
      const composed = {
        colormap: 'viridis',
        has_scalars: true,
        intensity: 0.4,
        offset: 0.02,
      } as unknown as PointsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createPointsNode(
        '/points',
        composed,
        makePointsData(false),
        makeLoader(),
        false,
        composed
      );

      const mat = mesh.material as PointMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(false);
      expect(mat.uniforms.uIntensity.value).toBeCloseTo(0.4, 6);
      expect(mat.uniforms.uOffset.value).toBeCloseTo(0.02, 6);
    });
  });

  describe('layers-panel path (layer=true, applyColorAdjustments)', () => {
    it('resets a previously-stamped gain to identity and moves it into the scalar window', () => {
      const tex = getColormapTexture('viridis');
      const mat = new PointMaterial({
        colormapTexture: tex ?? undefined,
        intensity: 0.09,
        offset: 0.0,
      });
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      expect(mat.uniforms.uIntensity.value).toBeCloseTo(0.09, 6);

      // The panel composes the authored 0.09 and pushes it through here.
      applyColorAdjustments(mat, 1.0, 0.09, 0.0);

      // Gain reset to identity — no double application.
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // Same window as the node-factory path: [0, ~11.11].
      const { min, max } = computeDisplayRange(0.09, 0.0);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(min, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (max - min), 5);
    });
  });
});
