/**
 * The gsplat TSL snapshot adapters fall back to PRODUCTION defaults.
 *
 * `buildGSplatTSLNodesFromUniforms` / `buildGSplatPickTSLNodesFromUniforms`
 * build a node set from a flat uniform record for callers that own no
 * persistent nodes (the `ShaderSource.webgpu` entry points, the parity
 * harness). Their fallbacks had drifted from what the production materials
 * construct with — a coverage limit of 1.0 instead of 0.33, a near cull of
 * 1e-4 instead of 0.1, no 2D dilation instead of 0.3, and shifted-Gaussian
 * constants (C = 0, 1/(1-C) = 1) that belong to no truncation radius at all,
 * so a record carrying only `uTruncate` produced an unshifted falloff that
 * never reached zero at the truncation edge.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GSPLAT_COV2D_DILATION_DEFAULT } from '../../../../rendering/materials/gsplat/math';
import { buildGSplatTSLNodesFromUniforms } from '../../../../rendering/materials/gsplat/shader-tsl';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { buildGSplatPickTSLNodesFromUniforms } from '../../../../rendering/picking/gsplat/pick.tsl';

const value = (node: unknown): unknown => (node as { value: unknown }).value;

describe('gsplat pick adapter fallbacks match GSplatPickingMaterial', () => {
  it('for every uniform the production constructor sets a non-trivial default', () => {
    const nodes = buildGSplatPickTSLNodesFromUniforms({}) as unknown as Record<string, unknown>;
    const prod = new GSplatPickingMaterial({ nodeId: 0 }).uniforms;
    for (const name of [
      'uTruncate',
      'uTruncateSq',
      'uShiftC',
      'uInvOneMinusC',
      'uNearCull',
      'uMaxExtentFactor',
      'uCov2DDilation',
    ]) {
      expect(value(nodes[name]), name).toBeCloseTo(prod[name].value as number, 12);
    }
  });
});

describe('gsplat visual adapter fallbacks', () => {
  it('use the production coverage limit, near cull and dilation', () => {
    const nodes = buildGSplatTSLNodesFromUniforms({}) as unknown as Record<string, unknown>;
    expect(value(nodes.uNearCull)).toBe(0.1);
    expect(value(nodes.uMaxExtentFactor)).toBe(0.33);
    expect(value(nodes.uCov2DDilation)).toBe(GSPLAT_COV2D_DILATION_DEFAULT);
  });

  it.each([
    ['visual', buildGSplatTSLNodesFromUniforms],
    ['pick', buildGSplatPickTSLNodesFromUniforms],
  ])('%s: the shifted-Gaussian constants follow the truncation radius supplied', (_n, build) => {
    const t = 2.5;
    const nodes = build({ uTruncate: { value: t } } as Record<
      string,
      THREE.IUniform
    >) as unknown as Record<string, unknown>;
    const c = Math.exp(-0.5 * t * t);
    expect(value(nodes.uShiftC)).toBeCloseTo(c, 12);
    expect(value(nodes.uInvOneMinusC)).toBeCloseTo(1 / (1 - c), 12);
  });
});
