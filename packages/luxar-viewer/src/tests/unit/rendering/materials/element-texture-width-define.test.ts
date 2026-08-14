/**
 * The baked element-texture width (per-layout `widthDefine`): every GLSL
 * material (visual + picking, all three element-texture geometry
 * types) must stamp the define with the SAME width the texture
 * writers allocate at (`getElementTextureWidth`) — the shaders index
 * `uXTex` with `base % W` / `base / W`, so a drifted value reads
 * garbage texels. The TSL twins bake the same value as a literal int
 * node; that side is pinned by the codegen snapshots (the generated
 * WGSL/GLSL contains the literal, no textureSize query).
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import {
  elementTextureWidthDefines,
  getElementTextureWidth,
  LINE_TEXTURE_LAYOUT,
  POINT_TEXTURE_LAYOUT,
  SPLAT_TEXTURE_LAYOUT,
  type ElementTextureLayout,
} from '../../../../rendering/element-texture-layout';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';

const CASES: Array<{
  label: string;
  layout: ElementTextureLayout;
  make: () => { defines?: Record<string, unknown> };
}> = [
  { label: 'LineMaterial', layout: LINE_TEXTURE_LAYOUT, make: () => new LineMaterial() },
  { label: 'PointMaterial', layout: POINT_TEXTURE_LAYOUT, make: () => new PointMaterial() },
  { label: 'GSplatMaterial', layout: SPLAT_TEXTURE_LAYOUT, make: () => new GSplatMaterial() },
  {
    label: 'LinePickingMaterial',
    layout: LINE_TEXTURE_LAYOUT,
    make: () => new LinePickingMaterial({ nodeId: 1 }),
  },
  {
    label: 'PointPickingMaterial',
    layout: POINT_TEXTURE_LAYOUT,
    make: () => new PointPickingMaterial({ nodeId: 1 }),
  },
  {
    label: 'GSplatPickingMaterial',
    layout: SPLAT_TEXTURE_LAYOUT,
    make: () => new GSplatPickingMaterial({ nodeId: 1 }),
  },
];

describe('element-texture width define', () => {
  it.each(CASES)('$label bakes the writer-side width', ({ layout, make }) => {
    const material = make();
    expect(material.defines?.[layout.widthDefine]).toBe(String(getElementTextureWidth(layout)));
  });

  it('the harness-injection helper carries every layout exactly once', () => {
    expect(elementTextureWidthDefines()).toEqual({
      LUXAR_LINE_TEX_W: String(getElementTextureWidth(LINE_TEXTURE_LAYOUT)),
      LUXAR_POINT_TEX_W: String(getElementTextureWidth(POINT_TEXTURE_LAYOUT)),
      LUXAR_SPLAT_TEX_W: String(getElementTextureWidth(SPLAT_TEXTURE_LAYOUT)),
    });
  });

  it('binding a different-width texture re-stamps the define and flags a rebuild', () => {
    const material = new LineMaterial();
    const before = material.defines?.[LINE_TEXTURE_LAYOUT.widthDefine];
    // 12 texels wide (2 segments/row) — a multiple of 6, so the
    // row-straddle invariant holds, but not the session width.
    const tiny = new THREE.DataTexture(new Float32Array(12 * 4), 12, 1, THREE.RGBAFormat);
    // three's needsUpdate is a setter that bumps `version` — assert on that.
    const versionBefore = material.version;
    material.updateLineTexture(tiny);
    expect(material.defines?.[LINE_TEXTURE_LAYOUT.widthDefine]).toBe('12');
    expect(material.defines?.[LINE_TEXTURE_LAYOUT.widthDefine]).not.toBe(before);
    expect(material.version).toBe(versionBefore + 1);
    tiny.dispose();
  });

  it('a clone carries the width of the texture it binds, not the session pre-stamp', () => {
    // `clone()` builds a fresh material through the constructor (which
    // pre-stamps the session width) and copies the texture binding over,
    // so it has to re-stamp too — otherwise a clone taken while a
    // non-session-width texture is bound (renderer swap straggler)
    // addresses with the wrong row stride.
    const material = new LineMaterial();
    const tiny = new THREE.DataTexture(new Float32Array(12 * 4), 12, 1, THREE.RGBAFormat);
    material.updateLineTexture(tiny);
    const cloned = material.clone();
    expect(cloned.getLineTexture()).toBe(tiny);
    expect(cloned.defines?.[LINE_TEXTURE_LAYOUT.widthDefine]).toBe('12');
    tiny.dispose();
  });

  it('re-binding the placeholder keeps the pre-stamped width (empty draws, no recompile)', () => {
    const material = new LineMaterial();
    const before = material.defines?.[LINE_TEXTURE_LAYOUT.widthDefine];
    const versionBefore = material.version;
    material.updateLineTexture(null); // falls back to the shared placeholder
    expect(material.defines?.[LINE_TEXTURE_LAYOUT.widthDefine]).toBe(before);
    expect(material.version).toBe(versionBefore);
  });

  it('widths are positive multiples of texels-per-element (row-straddle guard)', () => {
    for (const layout of [LINE_TEXTURE_LAYOUT, POINT_TEXTURE_LAYOUT, SPLAT_TEXTURE_LAYOUT]) {
      const w = getElementTextureWidth(layout);
      expect(w).toBeGreaterThan(0);
      expect(w % layout.texelsPerElement).toBe(0);
    }
  });

  it('no GLSL shader source still queries textureSize for the element texture', async () => {
    // The whole point of the define is deleting the per-vertex query —
    // pin the sources so a refactor cannot quietly reintroduce it.
    const sources = await Promise.all([
      import('../../../../rendering/materials/line/shader-glsl'),
      import('../../../../rendering/materials/line/shader-glsl-capsule'),
      import('../../../../rendering/materials/point/shader-glsl'),
      import('../../../../rendering/materials/gsplat/shader-glsl'),
      import('../../../../rendering/picking/line/shaders'),
      import('../../../../rendering/picking/line/shaders-capsule'),
      import('../../../../rendering/picking/point/shaders'),
      import('../../../../rendering/picking/gsplat/shaders'),
      import('../../../../rendering/materials/_shared/glsl-lib'),
    ]);
    for (const mod of sources) {
      for (const value of Object.values(mod)) {
        if (typeof value === 'string') {
          expect(value).not.toContain('textureSize(');
        } else if (value && typeof value === 'object') {
          for (const inner of Object.values(value)) {
            if (typeof inner === 'string') expect(inner).not.toContain('textureSize(');
          }
        }
      }
    }
  });
});
