/**
 * The visual → pick sync, per pick type: which uniform inputs each pick
 * material receives from its node's visual material (the set of uniforms it
 * declares), and that the GLSL and TSL wrappers of a type receive the same
 * set. One table (`PICK_SYNC_INPUTS`) drives all of them, so a new input a
 * shader starts reading only needs the uniform declared on both wrappers.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { MeshPickingMaterial } from '../../../../rendering/picking/mesh/material';
import { MeshPickingTSLMaterial } from '../../../../rendering/picking/mesh/material-tsl';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { PointPickingTSLMaterial } from '../../../../rendering/picking/point/material-tsl';
import {
  effectiveBlendingMode,
  pickSyncedInputs,
  syncPickFromVisual,
} from '../../../../rendering/picking/picking-system/visibility-sync';
import { uniformValue } from './render-pick-helper';

const WEIGHT = ['uOpacity', 'uIntensity', 'uHasElementAlpha', 'uVolumetric'];

const EXPECTED: [string, () => unknown, () => unknown, string[]][] = [
  [
    'gsplat',
    () => new GSplatPickingMaterial({ nodeId: 1 }),
    () => new GSplatPickingTSLMaterial({ nodeId: 1 }),
    [
      'uDensityDrop',
      'uCoverageTruncate',
      'uMaxExtentFactor',
      'uCov2DDilation',
      ...WEIGHT,
      'uProjectionMode',
      'uRayIntegralFactor',
    ],
  ],
  [
    'point',
    () => new PointPickingMaterial({ nodeId: 1 }),
    () => new PointPickingTSLMaterial({ nodeId: 1 }),
    ['uDensityDrop', ...WEIGHT],
  ],
  [
    'line',
    () => new LinePickingMaterial({ nodeId: 1 }),
    () => new LinePickingTSLMaterial({ nodeId: 1 }),
    ['uDensityDrop', ...WEIGHT],
  ],
  [
    'mesh',
    () => new MeshPickingMaterial({ nodeId: 1 }),
    () => new MeshPickingTSLMaterial({ nodeId: 1 }),
    ['uOpacity', 'uNearFade'],
  ],
];

describe.each(EXPECTED)('%s pick: synced inputs', (_type, glsl, tsl, expected) => {
  it('GLSL and TSL wrappers receive the same documented set', () => {
    expect(pickSyncedInputs(glsl()).sort()).toEqual([...expected].sort());
    expect(pickSyncedInputs(tsl()).sort()).toEqual([...expected].sort());
  });
});

describe('syncPickFromVisual', () => {
  it('copies the uniform inputs and sets the depth convention in one call', () => {
    const visual = new GSplatMaterial({ blendingMode: 'normal', truncationRadius: 2 });
    visual.uniforms.uDensityDrop.value = 0.5;
    const pick = new GSplatPickingMaterial({ nodeId: 1 });

    syncPickFromVisual(pick, visual);

    expect(uniformValue(pick, 'uDensityDrop')).toBe(0.5);
    expect(uniformValue(pick, 'uCoverageTruncate')).toBe(2);
    expect(uniformValue(pick, 'uSurfaceDepth')).toBe(1);
  });

  it('a mesh pick gets its mode-derived state and the visual face culling', () => {
    const visual = new MeshMaterial({ blendingMode: 'additive' });
    visual.side = THREE.BackSide;
    const pick = new MeshPickingMaterial({ nodeId: 1 });

    syncPickFromVisual(pick, [visual]);

    expect(uniformValue(pick, 'uSurfaceDepth')).toBe(0);
    expect(uniformValue(pick, 'uAlphaCutout')).toBe(0);
    expect(pick.side).toBe(THREE.BackSide);
  });

  it('no visual material: nothing dropped, defaults otherwise untouched', () => {
    const pick = new PointPickingMaterial({ nodeId: 1 });
    pick.uniforms.uDensityDrop.value = 0.25;
    syncPickFromVisual(pick, undefined);
    expect(uniformValue(pick, 'uDensityDrop')).toBe(0);
    expect(uniformValue(pick, 'uOpacity')).toBe(1);
  });

  it('effectiveBlendingMode: physical meshes follow their translucency', () => {
    const physical = new THREE.MeshBasicMaterial({ transparent: true });
    physical.userData.material = 'physical';
    expect(effectiveBlendingMode(physical)).toBe('normal');
    physical.transparent = false;
    expect(effectiveBlendingMode(physical)).toBe('opaque');
    expect(effectiveBlendingMode(undefined)).toBe('additive');
  });
});
