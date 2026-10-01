/**
 * A visual material's clone keeps the density guard's thinning state.
 *
 * The guard writes `uDensityDrop` (the dropped fraction) and, for an
 * alpha-over node, `uDensityAlphaExp` (the compensation exponent) on the
 * visual material. The Layers panel clones a node's material on its first
 * interaction, and a clone that reset both to the identity drew the node
 * un-thinned while its brightness still carried the thinning compensation
 * (an over-bright flash), until the guard's next visit re-asserted the
 * uniforms — on-screen nodes the next frame, an off-screen node only when it
 * returns to view. `uSortedIndexSlot` already rides along for the same reason.
 */
import { describe, expect, it } from 'vitest';

import {
  DENSITY_ALPHA_EXP_UNIFORM,
  DENSITY_DROP_UNIFORM,
} from '../../../../rendering/materials/_shared/density-drop';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../../../rendering/materials/point/material-tsl';

type Uniforms = { uniforms: Record<string, { value: unknown }>; clone(): Uniforms };

describe.each([
  ['PointMaterial', () => new PointMaterial({})],
  ['PointTSLMaterial', () => new PointTSLMaterial({})],
  ['LineMaterial', () => new LineMaterial({})],
  ['LineTSLMaterial', () => new LineTSLMaterial({})],
  ['GSplatMaterial', () => new GSplatMaterial({})],
  ['GSplatTSLMaterial', () => new GSplatTSLMaterial({})],
])('%s.clone() carries the density-guard state', (_name, make) => {
  it.fails('copies uDensityDrop and uDensityAlphaExp', () => {
    const mat = make() as unknown as Uniforms;
    mat.uniforms[DENSITY_DROP_UNIFORM].value = 0.75;
    mat.uniforms[DENSITY_ALPHA_EXP_UNIFORM].value = 4;

    const cloned = mat.clone();

    expect(cloned.uniforms[DENSITY_DROP_UNIFORM].value).toBe(0.75);
    expect(cloned.uniforms[DENSITY_ALPHA_EXP_UNIFORM].value).toBe(4);
  });
});
