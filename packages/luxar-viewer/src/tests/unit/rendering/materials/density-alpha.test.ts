/**
 * `uDensityAlphaExp` plumbing — the alpha compensation of a thinned
 * alpha-over (`normal`) node. The density guard draws a `keep` fraction of
 * such a node's elements and raises each survivor's alpha to
 * `1 − (1 − α)^(1/keep)`, so `keep·N` survivors transmit what `N` elements
 * did. Every visual leaf material (GLSL and TSL) must carry the uniform at
 * the identity exponent 1, and every fragment shader must route its
 * alpha-over output through the helper.
 */
import { describe, expect, it } from 'vitest';

import * as densityDrop from '../../../../rendering/materials/_shared/density-drop';
import * as glslLib from '../../../../rendering/materials/_shared/glsl-lib';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { POINT_FRAGMENT_SHADER } from '../../../../rendering/materials/point/shader-glsl';
import { LINE_FRAGMENT_SHADER } from '../../../../rendering/materials/line/shader-glsl';
import { CAPSULE_LINE_FRAGMENT_SHADER } from '../../../../rendering/materials/line/shader-glsl-capsule';
import { GSPLAT_FRAGMENT_SHADER } from '../../../../rendering/materials/gsplat/shader-glsl';

type Uniforms = { uniforms: Record<string, { value: unknown } | undefined> };
const helpers = densityDrop as unknown as Record<string, unknown>;
const setAlphaExp = helpers.setDensityAlphaExp as
  ((material: unknown, exponent: number) => boolean) | undefined;
const getAlphaExp = helpers.getDensityAlphaExp as ((material: unknown) => number) | undefined;

describe('uDensityAlphaExp', () => {
  it.fails('every visual leaf material declares it at the identity exponent, and the helpers write it', () => {
    const materials: [string, () => unknown][] = [
      ['PointMaterial', () => new PointMaterial({})],
      ['LineMaterial', () => new LineMaterial({})],
      ['GSplatMaterial', () => new GSplatMaterial({})],
      ['PointTSLMaterial', () => new PointTSLMaterial({ blendingMode: 'normal' })],
      ['LineTSLMaterial', () => new LineTSLMaterial({ blendingMode: 'normal' })],
      ['GSplatTSLMaterial', () => new GSplatTSLMaterial({ blendingMode: 'normal' })],
    ];
    for (const [name, make] of materials) {
      const mat = make() as Uniforms;
      expect(mat.uniforms.uDensityAlphaExp?.value, name).toBe(1);
      expect(setAlphaExp?.(mat, 4), name).toBe(true);
      expect(getAlphaExp?.(mat), name).toBe(4);
      expect(mat.uniforms.uDensityAlphaExp?.value, name).toBe(4);
    }
  });

  it.fails('the shared GLSL helper is the identity at an exponent ≤ 1 (an unset uniform reads 0)', () => {
    const chunk = (glslLib as unknown as Record<string, unknown>).GLSL_DENSITY_ALPHA;
    expect(typeof chunk).toBe('string');
    expect(chunk as string).toContain('uniform float uDensityAlphaExp;');
    expect(chunk as string).toContain('if (!(uDensityAlphaExp > 1.0)) return a;');
  });

  it.fails('every GLSL fragment shader routes its alpha-over output through luxarDensityAlpha', () => {
    const sources: [string, string][] = [
      ['point', POINT_FRAGMENT_SHADER],
      ['line', LINE_FRAGMENT_SHADER],
      ['capsule line', CAPSULE_LINE_FRAGMENT_SHADER],
      ['gsplat', GSPLAT_FRAGMENT_SHADER],
    ];
    for (const [name, source] of sources) {
      expect(source, name).toContain('luxarDensityAlpha(');
    }
  });
});
