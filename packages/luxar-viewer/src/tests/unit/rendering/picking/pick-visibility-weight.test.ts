// @vitest-environment jsdom
/**
 * The pick pass weighs every element by what the VISUAL pass scales it by
 * (review A19): per-element alpha (optical depth under volumetric), node
 * opacity, and the gain the visual discards honour through max(gain, 1) —
 * plus, for gsplats, the sum-projection ray-integral boost and dilation
 * compensation that turn the amplitude into what additive/luminous/volumetric
 * actually draw.
 *
 * Before: the pick shaders saw only the geometry's own falloff, so a fully
 * transparent floater won tooltips, a dim element lifted by a high gain could
 * not be picked, and a gsplat's pickability ignored the ray-integral boost
 * (world-unit sigma: invisible tiny-unit splats stayed pickable, visible
 * large-unit ones were culled by the 1e-4 floor).
 *
 * Pinned here: the shaders read the shared weight (GLSL `luxarPickWeight` /
 * `luxarPickAlphaFactor`, TSL `pickWeightTSL` / `pickAlphaFactorTSL`), the
 * pick materials declare its inputs at neutral defaults, and the pick render
 * syncs them from each node's visual material.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GSPLAT_DEFAULT_TRUNCATION_RADIUS } from '../../../../config/constants';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { GSPLAT_VERTEX_SHADER } from '../../../../rendering/materials/gsplat/shader-glsl';
import { computeRayIntegralFactor } from '../../../../rendering/materials/gsplat/math';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import {
  GSPLAT_PICK_FRAGMENT_SHADER,
  GSPLAT_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/gsplat/shaders';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import {
  LINE_PICK_FRAGMENT_SHADER,
  LINE_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/line/shaders';
import {
  CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  CAPSULE_LINE_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/line/shaders-capsule';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import {
  POINT_PICK_FRAGMENT_SHADER,
  POINT_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/point/shaders';
import { makePickHarness, setUniform, uniformValue } from './render-pick-helper';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const src = (rel: string): string =>
  readFileSync(path.resolve(HERE, '../../../../rendering/picking', rel), 'utf8');

/** The four inputs every non-mesh pick material weighs by, and their neutral defaults. */
const WEIGHT_DEFAULTS = { uIntensity: 1, uOpacity: 1, uHasElementAlpha: 0, uVolumetric: 0 };

describe('GLSL pick shaders weigh by the visual factors', () => {
  it('gsplat: the vertex stage sizes the quad from alpha x opacity and the gain', () => {
    expect(GSPLAT_PICK_VERTEX_SHADER).toContain('float luxarPickWeight(float alpha)');
    const calls = [
      ...GSPLAT_PICK_VERTEX_SHADER.matchAll(
        /gsplatVisibleMahalSq\(\s*gsplatFootprintPeakScale\(([^)]*)\)/g
      ),
    ];
    expect(calls).toHaveLength(1);
    expect(calls[0][1].split(',').map((a) => a.trim())).toEqual([
      'vAmplitude2D',
      'uInvOneMinusC',
      'pickAlphaFactor',
      'uIntensity',
    ]);
    // The fragment test multiplies by the same weight the quad was sized for.
    expect(GSPLAT_PICK_FRAGMENT_SHADER).toContain('intensity * vPickWeight');
  });

  it('gsplat: sum projection carries the visual ray-integral amplitude', () => {
    const sumLine =
      'vAmplitude2D = aAmplitude * rayIntegrationBoost * nearFade * dilationCompensation;';
    expect(GSPLAT_VERTEX_SHADER).toContain(sumLine);
    expect(GSPLAT_PICK_VERTEX_SHADER).toContain(sumLine);
    expect(GSPLAT_PICK_VERTEX_SHADER).toContain('if (uProjectionMode == 0) {');
    expect(GSPLAT_PICK_VERTEX_SHADER).toContain('uniform float uRayIntegralFactor;');
  });

  it.each([
    ['point', POINT_PICK_VERTEX_SHADER, POINT_PICK_FRAGMENT_SHADER],
    ['line', LINE_PICK_VERTEX_SHADER, LINE_PICK_FRAGMENT_SHADER],
    ['line capsule', CAPSULE_LINE_PICK_VERTEX_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER],
  ])('%s: the fragment brightness carries luxarPickWeight(vAlpha)', (_n, vs, fs) => {
    expect(vs).toMatch(/vAlpha = .*sanitizeAlpha\(/);
    expect(fs).toContain('float luxarPickWeight(float alpha)');
    expect(fs).toMatch(/brightness = .*\* luxarPickWeight\(vAlpha\);/);
  });
});

describe('TSL pick factories weigh by the visual factors', () => {
  it.each([
    ['gsplat/pick.tsl.ts', 'pickAlphaFactorTSL('],
    ['point/pick.tsl.ts', 'pickWeightTSL('],
    ['line/pick.tsl.ts', 'pickWeightTSL('],
    ['line/pick-capsule.tsl.ts', 'pickWeightTSL('],
  ])('%s uses %s', (file, helper) => {
    expect(src(file)).toContain(helper);
  });

  it('gsplat/pick.tsl.ts carries the sum-projection boost', () => {
    const tsl = src('gsplat/pick.tsl.ts');
    expect(tsl).toContain('uRayIntegralFactor');
    expect(tsl).toContain('dilationCompensation');
  });
});

describe.each([
  ['GSplatPickingMaterial', () => new GSplatPickingMaterial({ nodeId: 1 })],
  ['GSplatPickingTSLMaterial', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
  ['PointPickingMaterial', () => new PointPickingMaterial({ nodeId: 1 })],
  ['PointPickingTSLMaterial', () => new PointPickingTSLMaterial({ nodeId: 1 })],
  ['LinePickingMaterial', () => new LinePickingMaterial({ nodeId: 1 })],
  ['LinePickingTSLMaterial', () => new LinePickingTSLMaterial({ nodeId: 1 })],
])('%s declares the weight inputs', (_n, make) => {
  it('at neutral defaults, carried by clone()', () => {
    const mat = make() as unknown as THREE.Material;
    for (const [name, value] of Object.entries(WEIGHT_DEFAULTS)) {
      expect(uniformValue(mat, name), name).toBe(value);
    }
    setUniform(mat, 'uIntensity', 5);
    setUniform(mat, 'uVolumetric', 1);
    const cloned = mat.clone();
    expect(uniformValue(cloned, 'uIntensity')).toBe(5);
    expect(uniformValue(cloned, 'uVolumetric')).toBe(1);
  });
});

describe.each([
  ['GSplatPickingMaterial', () => new GSplatPickingMaterial({ nodeId: 1 })],
  ['GSplatPickingTSLMaterial', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
])('%s declares the sum-projection inputs', (_n, make) => {
  it('defaulting to the peak projection the pick pass always used', () => {
    const mat = make() as unknown as THREE.Material;
    expect(uniformValue(mat, 'uProjectionMode')).toBe(1);
    expect(uniformValue(mat, 'uRayIntegralFactor')).toBe(
      computeRayIntegralFactor(GSPLAT_DEFAULT_TRUNCATION_RADIUS)
    );
  });
});

describe('renderPickBuffer syncs the weight inputs from the visual material', () => {
  const visuals = {
    gsplat: [
      [
        'GLSL',
        (o: object) => new GSplatMaterial(o),
        () => new GSplatPickingMaterial({ nodeId: 1 }),
      ],
      [
        'TSL',
        (o: object) => new GSplatTSLMaterial(o),
        () => new GSplatPickingTSLMaterial({ nodeId: 1 }),
      ],
    ],
    point: [
      ['GLSL', (o: object) => new PointMaterial(o), () => new PointPickingMaterial({ nodeId: 1 })],
      [
        'TSL',
        (o: object) => new PointTSLMaterial(o),
        () => new PointPickingTSLMaterial({ nodeId: 1 }),
      ],
    ],
    line: [
      ['GLSL', (o: object) => new LineMaterial(o), () => new LinePickingMaterial({ nodeId: 1 })],
      [
        'TSL',
        (o: object) => new LineTSLMaterial(o),
        () => new LinePickingTSLMaterial({ nodeId: 1 }),
      ],
    ],
  } as const;

  for (const [type, pairs] of Object.entries(visuals)) {
    for (const [backend, makeMain, makePick] of pairs) {
      it(`${type} [${backend}]: gain, opacity, element alpha and the volumetric flag`, () => {
        const { register, renderPickBuffer } = makePickHarness();
        const main = makeMain({
          blendingMode: 'volumetric',
          hasElementAlpha: true,
        }) as THREE.Material;
        setUniform(main, 'uIntensity', 8);
        setUniform(main, 'uOpacity', 0.5);
        const pick = makePick() as THREE.Material;
        register(main, pick);

        renderPickBuffer();

        expect(uniformValue(pick, 'uIntensity')).toBe(8);
        expect(uniformValue(pick, 'uOpacity')).toBe(0.5);
        expect(uniformValue(pick, 'uHasElementAlpha')).toBe(1);
        expect(uniformValue(pick, 'uVolumetric')).toBe(1);
        if (type === 'gsplat') {
          // volumetric is a sum-projection mode.
          expect(uniformValue(pick, 'uProjectionMode')).toBe(0);
          expect(uniformValue(pick, 'uRayIntegralFactor')).toBe(
            uniformValue(main, 'uRayIntegralFactor')
          );
        }
      });

      it(`${type} [${backend}]: a non-volumetric visual clears the volumetric flag`, () => {
        const { register, renderPickBuffer } = makePickHarness();
        const pick = makePick() as THREE.Material;
        setUniform(pick, 'uVolumetric', 1);
        register(makeMain({ blendingMode: 'normal' }) as THREE.Material, pick);
        renderPickBuffer();
        expect(uniformValue(pick, 'uVolumetric')).toBe(0);
        if (type === 'gsplat') expect(uniformValue(pick, 'uProjectionMode')).toBe(1);
      });
    }
  }
});
