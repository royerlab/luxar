/**
 * Regression (#936): an authored `intensity`/`offset` on a COLORMAPPED
 * gsplat node must define the display WINDOW only — never a post-LUT color
 * gain on top of it.
 *
 * The gsplat fragment shader always multiplies `vColor * uIntensity + uOffset`
 * post-LUT (the LUXAR_NO_GOG compile-out fires only for the identity
 * intensity==1 && offset==0), so leaving an authored gain stamped while ALSO
 * inverting it into the scalar window applies the value twice, with two
 * different meanings. The fix:
 *
 *  - Node factory (`layer=false`): stamp the color GOG at IDENTITY when a
 *    colormap is applied, and derive the scalar window from the authored
 *    gain/offset exactly as the panel does — so `layer=false` matches
 *    `layer=true`.
 *  - `applyColorAdjustments` (`layer=true`): in colormap mode push the window
 *    via `updateScalarRange` AND actively reset the color GOG to identity.
 *
 * Direct-color nodes are unaffected — they still receive the authored gain.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { NodeFactory } from '../../../rendering/node-factory';
import { __resetMaterialManagerForTests } from '../../../rendering/material-manager';
import { GSplatMaterial } from '../../../rendering/materials/gsplat/material-glsl';
import { getColormapTexture } from '../../../rendering/colormap-textures';
import { applyColorAdjustments } from '../../../ui/layers/luxar-material';
import { computeDisplayRange } from '../../../ui/layers/layer-state';
import type { InstancedGSplatsMeshConfig } from '../../../rendering/gsplat-geometry';
import type { GSplatsMetadata, GSplatsDataLoader } from '../../../types/gsplats';

/** Minimal one-splat processed config. */
function makeConfig(): InstancedGSplatsMeshConfig {
  return {
    centers: new Float32Array([0, 0, 0]),
    choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
    amplitudes: new Float32Array([1.0]),
    colors: new Float32Array([1, 1, 1]),
    splatCount: 1,
  };
}

/** createGSplatsNode only stashes the loader in userData — never calls it. */
function makeLoader(): GSplatsDataLoader {
  return { dispose: vi.fn() } as unknown as GSplatsDataLoader;
}

/** Only `truncation_radius`/`transform` are read off `attrs` at creation. */
const rawAttrs = { type: 'gsplats' } as unknown as GSplatsMetadata;

describe('#936 colormapped gsplat authored intensity is a window, not a double gain', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  describe('node-factory path (layer=false)', () => {
    it('colormapped node: color GOG is IDENTITY, scalar window = computeDisplayRange(intensity)', () => {
      // Leaf authors intensity=0.09 with no ancestor gain → composed == raw.
      // The window decision keys on the RAW leaf gain (FIX 1), so the leaf
      // gain must be present on the raw `attrs` param, not just `nodeAttrs`.
      const leafRaw = {
        type: 'gsplats',
        intensity: 0.09,
        offset: 0.0,
      } as unknown as GSplatsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { colormap: 'viridis', intensity: 0.09, offset: 0.0 },
        leafRaw,
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
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
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { colormap: 'viridis', intensity: 1.0, offset: 0.0, amplitude_data_range: [0.5, 2.5] },
        rawAttrs,
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // No authored window → data range drives the LUT lookup.
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(0.5, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(0.5, 5); // 1/(2.5-0.5)
    });

    it('direct-color node (no colormap) still receives the authored gain', () => {
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { intensity: 0.09, offset: 0.02 },
        rawAttrs,
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(false);
      expect(mat.uniforms.uIntensity.value).toBeCloseTo(0.09, 6);
      expect(mat.uniforms.uOffset.value).toBeCloseTo(0.02, 6);
    });

    it('ancestor-only gain (composed != 1, raw leaf identity): gain folds onto the data-range window', () => {
      // Simulate a colormapped leaf with NO own intensity sitting under an
      // ancestor group with intensity=0.5: the loader passes the COMPOSED
      // 0.5 as `nodeAttrs.intensity`, but the RAW leaf `attrs` has no gain.
      // The identity decision follows the RAW leaf gain (identity), so the
      // window STARTS from `amplitude_data_range` — NOT computeDisplayRange(
      // 0.5)=[0,2], which would saturate the LUT. But the ancestor gain must
      // still act (the panel composes it onto the window): data range
      // [0,100] → window uniforms (0.01, 0) → × ancestor 0.5 → (0.005, 0) →
      // window [0, 200]. Fails on code that keys the decision on the
      // composed value ([0,2]) AND on code that drops the ancestor gain
      // entirely ([0,100]).
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { colormap: 'viridis', intensity: 0.5, offset: 0.0, amplitude_data_range: [0, 100] },
        rawAttrs, // raw leaf: no intensity/offset → identity
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      // Color GOG identity (colormap active).
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // Data-range window with the ancestor 0.5 folded in: [0, 200].
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(0, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / 200, 5);
    });

    it('colormapped node with offset != 0 (raw non-identity): window = computeDisplayRange(1, 0.5)', () => {
      // Raw leaf authors a non-identity window via offset alone.
      const leafRaw = {
        type: 'gsplats',
        intensity: 1.0,
        offset: 0.5,
      } as unknown as GSplatsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { colormap: 'viridis', intensity: 1.0, offset: 0.5 },
        leafRaw,
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      // Color GOG identity.
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // Window inverts the authored offset: computeDisplayRange(1, 0.5) = [-0.5, 0.5].
      const { min, max } = computeDisplayRange(1.0, 0.5);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(min, 5);
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (max - min), 5);
    });

    it('leaf-authored window under an ancestor gain: window folds the COMPOSED gain, not the raw leaf gain', () => {
      // Leaf authors a non-identity window (intensity=0.5) AND sits under an
      // ancestor group that contributes another 0.5, so the loader passes the
      // COMPOSED 0.25 as `nodeAttrs.intensity` (0.5 ancestor × 0.5 leaf) while
      // the RAW leaf `attrs` keeps its own 0.5. The identity-vs-window decision
      // keys on the RAW leaf gain (non-identity → windowed), but the window
      // VALUE must fold the COMPOSED gain so the ancestor gain lands on the
      // leaf window (FIX: line uses `composedIntensity`, not `leafIntensity`).
      // Guards against reverting that line to the raw leaf gain — with raw==0.5
      // and composed==0.25 the two produce different windows.
      const leafRaw = {
        type: 'gsplats',
        intensity: 0.5,
        offset: 0.0,
      } as unknown as GSplatsMetadata;
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { colormap: 'viridis', intensity: 0.25, offset: 0.0 }, // composed: ancestor 0.5 × leaf 0.5
        leafRaw, // raw leaf: intensity 0.5 → windowed
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
      expect('USE_COLORMAP' in mat.defines).toBe(true);
      // Color GOG identity (colormap active).
      expect(mat.uniforms.uIntensity.value).toBe(1.0);
      expect(mat.uniforms.uOffset.value).toBe(0.0);
      // Window uses the COMPOSED 0.25 → [0, 4], scale 0.25 — NOT the raw leaf
      // 0.5 → [0, 2], scale 0.5.
      const { min, max } = computeDisplayRange(0.25, 0.0);
      expect(max).toBeCloseTo(4, 5);
      expect(mat.uniforms.uScalarMin.value).toBeCloseTo(min, 5); // 0
      expect(mat.uniforms.uScalarScale.value).toBeCloseTo(1 / (max - min), 5); // 0.25
    });

    it('colormap attr present but texture unresolvable: behaves as a direct-color node', () => {
      // An unknown builtin colormap name → getColormapTexture returns
      // undefined (only 'custom' falls back to viridis). With no texture the
      // node is direct-color: the authored gain stays on the color GOG and no
      // colormap-derived window is applied.
      const factory = new NodeFactory();
      const mesh = factory.createGSplatsNode(
        '/splats',
        { colormap: 'definitely_not_a_real_colormap', intensity: 0.09, offset: 0.0 },
        rawAttrs,
        makeConfig(),
        makeLoader()
      );

      const mat = mesh.material as GSplatMaterial;
      // No colormap actually applied.
      expect('USE_COLORMAP' in mat.defines).toBe(false);
      // Authored gain kept as the color GOG (direct-color behavior).
      expect(mat.uniforms.uIntensity.value).toBeCloseTo(0.09, 6);
      expect(mat.uniforms.uOffset.value).toBeCloseTo(0.0, 6);
    });
  });

  describe('layers-panel path (layer=true, applyColorAdjustments)', () => {
    it('resets a previously-stamped gain to identity and moves it into the scalar window', () => {
      // Simulate a colormapped material that still carries a stale post-LUT
      // gain (0.09), as an older writer / prior stamp would have left it.
      const tex = getColormapTexture('viridis');
      const mat = new GSplatMaterial({
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
