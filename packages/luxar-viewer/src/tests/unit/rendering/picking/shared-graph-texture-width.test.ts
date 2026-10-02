/**
 * A pick graph shared under #2992 bakes the element-texture width of the
 * texture its materials actually bind, not the width of the shared graph's
 * stand-in texture.
 *
 * The shared graph is built ONCE per configuration over FORWARDING leaves.
 * Since the fleet commit "release shared TSL texture references on dispose",
 * a forwarding texture leaf is built over an independent 1x1 stand-in with
 * the same sample type, so the shared graph never pins a material's texture.
 * The visual factories take the baked width from their config
 * (`elementTextureWidth`), but the pick factories read it from
 * `nodes.uXTex.value` at build time — which is now the stand-in — so every
 * shared pick graph addressed its data texture as 1 texel wide: each element
 * read another element's texels. On WebGPU every pick buffer of a
 * points/lines/gsplat node was wrong (render gate: node ids differ on ~all
 * covered pixels of `mixed` and `partition_normal`); WebGL, which draws the
 * GLSL pick materials, was unaffected.
 *
 * Observable without a GPU (a TSL `Fn` body is only traced at build): the
 * width each shared-graph build hands its factory, i.e. the factory's own
 * rule `config.elementTextureWidth ?? resolveElementTextureWidth(layout,
 * nodes.<tex>.value)`, recorded through a spy around the real factory.
 */
import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import {
  LINE_TEXTURE_LAYOUT,
  POINT_TEXTURE_LAYOUT,
  resolveElementTextureWidth,
  SPLAT_TEXTURE_LAYOUT,
  type ElementTextureLayout,
} from '../../../../rendering/element-texture-layout';
import { floatTexture } from '../materials/graph-sharing-cases';

/** Widths the factories were asked to bake, by family. */
const baked: Record<string, number[]> = { gsplat: [], point: [], line: [], capsule: [] };

function effectiveWidth(
  layout: ElementTextureLayout,
  nodes: Record<string, { value?: unknown }>,
  texName: string,
  args: unknown[]
): number {
  const config = args.find(
    (a): a is { elementTextureWidth?: number } =>
      !!a && typeof a === 'object' && !(a as { isMaterial?: boolean }).isMaterial
  );
  return (
    config?.elementTextureWidth ??
    resolveElementTextureWidth(
      layout,
      nodes[texName]?.value as { image?: { width?: number } } | null
    )
  );
}

vi.mock('../../../../rendering/picking/gsplat/pick.tsl', async (orig) => {
  const mod = await orig<typeof import('../../../../rendering/picking/gsplat/pick.tsl')>();
  return {
    ...mod,
    gsplatPickWebGPUFactory: (nodes: never, ...rest: unknown[]) => {
      baked.gsplat.push(effectiveWidth(SPLAT_TEXTURE_LAYOUT, nodes, 'uSplatTex', rest));
      return (mod.gsplatPickWebGPUFactory as (...a: unknown[]) => unknown)(nodes, ...rest);
    },
  };
});
vi.mock('../../../../rendering/picking/point/pick.tsl', async (orig) => {
  const mod = await orig<typeof import('../../../../rendering/picking/point/pick.tsl')>();
  return {
    ...mod,
    pointPickWebGPUFactory: (nodes: never, ...rest: unknown[]) => {
      baked.point.push(effectiveWidth(POINT_TEXTURE_LAYOUT, nodes, 'uPointTex', rest));
      return (mod.pointPickWebGPUFactory as (...a: unknown[]) => unknown)(nodes, ...rest);
    },
  };
});
vi.mock('../../../../rendering/picking/line/pick.tsl', async (orig) => {
  const mod = await orig<typeof import('../../../../rendering/picking/line/pick.tsl')>();
  return {
    ...mod,
    linePickWebGPUFactory: (nodes: never, ...rest: unknown[]) => {
      baked.line.push(effectiveWidth(LINE_TEXTURE_LAYOUT, nodes, 'uLineTex', rest));
      return (mod.linePickWebGPUFactory as (...a: unknown[]) => unknown)(nodes, ...rest);
    },
  };
});
vi.mock('../../../../rendering/picking/line/pick-capsule.tsl', async (orig) => {
  const mod = await orig<typeof import('../../../../rendering/picking/line/pick-capsule.tsl')>();
  return {
    ...mod,
    capsuleLinePickWebGPUFactory: (nodes: never, ...rest: unknown[]) => {
      baked.capsule.push(effectiveWidth(LINE_TEXTURE_LAYOUT, nodes, 'uLineTex', rest));
      return (mod.capsuleLinePickWebGPUFactory as (...a: unknown[]) => unknown)(nodes, ...rest);
    },
  };
});

const { GSplatPickingTSLMaterial } =
  await import('../../../../rendering/picking/gsplat/material-tsl');
const { PointPickingTSLMaterial } =
  await import('../../../../rendering/picking/point/material-tsl');
const { LinePickingTSLMaterial } = await import('../../../../rendering/picking/line/material-tsl');

type Bindable = THREE.Material & { customProgramCacheKey(): string };

describe.each([
  // Distinct widths per family so a graph cached by an earlier case cannot mask a later one.
  [
    'gsplat',
    'gsplat',
    3072,
    (id: number) => new GSplatPickingTSLMaterial({ nodeId: id }),
    'updateSplatTexture',
  ],
  [
    'point',
    'point',
    3060,
    (id: number) => new PointPickingTSLMaterial({ nodeId: id }),
    'updatePointTexture',
  ],
  [
    'line (screen-space)',
    'line',
    3054,
    (id: number) => new LinePickingTSLMaterial({ nodeId: id, primitive: 'screen-space' }),
    'updateLineTexture',
  ],
  [
    'line (capsule)',
    'capsule',
    3048,
    (id: number) => new LinePickingTSLMaterial({ nodeId: id, primitive: 'capsule' }),
    'updateLineTexture',
  ],
] as const)(
  '%s pick: the shared graph bakes the bound texture width',
  (_n, family, width, make, bind) => {
    it('two materials binding one width share a graph built for that width', () => {
      const a = make(1) as unknown as Bindable & Record<string, (t: THREE.DataTexture) => void>;
      const b = make(2) as unknown as Bindable & Record<string, (t: THREE.DataTexture) => void>;
      baked[family].length = 0;
      a[bind](floatTexture(width));
      b[bind](floatTexture(width));
      expect(b.customProgramCacheKey()).toBe(a.customProgramCacheKey());
      // Built once for the pair (the second material reuses it) — at the real width.
      expect(baked[family]).toEqual([width]);
    });
  }
);
