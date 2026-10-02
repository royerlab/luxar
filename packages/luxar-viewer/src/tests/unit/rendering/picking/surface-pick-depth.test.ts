// @vitest-environment jsdom
/**
 * Points and lines pick the FRONT-MOST element under the surface modes.
 *
 * Brightness-as-depth (brightest wins) is right for the commutative modes, but
 * under `opaque` (depth-written) and `normal` (depth-sorted alpha-over) the
 * user sees an occluding surface: an opaque atom behind another still won the
 * pick whenever it was brighter. The gsplat and mesh pick materials already
 * switched to real projected depth there; points and lines wrote
 * `gl_FragDepth = 1 - brightness` unconditionally. They now carry the same
 * `uSurfaceDepth` switch (`setSurfacePickDepth`), synced per pick render from
 * the visual material's blending mode.
 */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import type { BlendingMode } from '../../../../types/blending';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { LINE_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/line/shaders';
import { CAPSULE_LINE_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/line/shaders-capsule';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import { POINT_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/point/shaders';
import { GSPLAT_PICK_FRAGMENT_SHADER } from '../../../../rendering/picking/gsplat/shaders';
import { makePickHarness, uniformValue } from './render-pick-helper';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const src = (rel: string): string =>
  readFileSync(path.resolve(HERE, '../../../../rendering/picking', rel), 'utf8');

describe('GLSL point/line pick fragments switch the depth convention', () => {
  it.each([
    ['point', POINT_PICK_FRAGMENT_SHADER],
    ['line', LINE_PICK_FRAGMENT_SHADER],
    ['line capsule', CAPSULE_LINE_PICK_FRAGMENT_SHADER],
  ])('%s', (_n, fs) => {
    expect(fs).toContain('uniform int uSurfaceDepth;');
    expect(fs).toMatch(
      /gl_FragDepth = \(uSurfaceDepth == 1\) \? gl_FragCoord\.z : 1\.0 \/ \(1\.0 \+ brightness\);/
    );
  });
});

describe('TSL point/line pick factories switch the depth convention', () => {
  it.each(['point/pick.tsl.ts', 'line/pick.tsl.ts', 'line/pick-capsule.tsl.ts'])('%s', (file) => {
    const tsl = src(file);
    const start = tsl.indexOf('const depthNode');
    const depthBody = tsl.slice(start, tsl.indexOf('const material', start));
    // Branchless: a mix on the 0/1 flag (a branch would bury the shared
    // brightness assignment in one arm — see the factories' comments).
    expect(depthBody).toMatch(/mix\([\s\S]*float\(uSurfaceDepth\)/);
    expect(depthBody).toContain('float(1.0).div(float(1.0).add(brightness))');
  });
});

it('gsplat commutative depth uses unclamped salience in both shader backends', () => {
  expect(GSPLAT_PICK_FRAGMENT_SHADER).toContain('1.0 / (1.0 + salience)');
  expect(src('gsplat/pick.tsl.ts')).toContain('float(1.0).div(float(1.0).add(intensity))');
  const depth = (salience: number): number => 1 / (1 + salience);
  expect(depth(2.25)).toBeLessThan(depth(1.5));
});

const PAIRS = [
  [
    'point GLSL',
    (m: BlendingMode) => new PointMaterial({ blendingMode: m }),
    () => new PointPickingMaterial({ nodeId: 1 }),
  ],
  [
    'point TSL',
    (m: BlendingMode) => new PointTSLMaterial({ blendingMode: m }),
    () => new PointPickingTSLMaterial({ nodeId: 1 }),
  ],
  [
    'line GLSL',
    (m: BlendingMode) => new LineMaterial({ blendingMode: m }),
    () => new LinePickingMaterial({ nodeId: 1 }),
  ],
  [
    'line TSL',
    (m: BlendingMode) => new LineTSLMaterial({ blendingMode: m }),
    () => new LinePickingTSLMaterial({ nodeId: 1 }),
  ],
] as const;

describe.each(PAIRS)(
  '%s: renderPickBuffer syncs the depth convention',
  (_n, makeMain, makePick) => {
    it.each([
      ['opaque', 1],
      ['normal', 1],
      ['additive', 0],
      ['max', 0],
      ['volumetric', 0],
    ])('%s -> uSurfaceDepth %i', (mode, expected) => {
      const { register, renderPickBuffer } = makePickHarness();
      const pick = makePick() as unknown as THREE.Material;
      register(makeMain(mode as BlendingMode) as unknown as THREE.Material, pick);
      renderPickBuffer();
      expect(uniformValue(pick, 'uSurfaceDepth')).toBe(expected);
    });

    it('clone() carries the convention', () => {
      const pick = makePick() as unknown as THREE.Material & {
        setSurfacePickDepth?: (on: boolean) => void;
      };
      expect(typeof pick.setSurfacePickDepth).toBe('function');
      pick.setSurfacePickDepth?.(true);
      expect(uniformValue(pick.clone(), 'uSurfaceDepth')).toBe(1);
    });
  }
);
