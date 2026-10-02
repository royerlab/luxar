// @vitest-environment jsdom
/**
 * A physical mesh stays pickable inside the near-fade band.
 *
 * The house mesh shader fades a surface out across [nearCull, 2·nearCull] in
 * front of the camera, and the shared mesh pick material mirrors that fade
 * (discarding below 0.01) so pick coverage tracks visible coverage. A physical
 * mesh renders through three's PBR material, which has NO near fade
 * (mesh-physical/README.md) — so with the pick fade applied unconditionally a
 * physical surface fully visible right in front of the camera was unpickable.
 * The pick fade is now a runtime switch, turned off per pick render for a
 * physical visual material.
 */
import { describe, expect, it } from 'vitest';

import { MeshMaterial } from '../../../../../rendering/materials/mesh/material-glsl';
import { PhysicalMeshMaterial } from '../../../../../rendering/materials/mesh-physical/material-glsl';
import { PhysicalMeshTSLMaterial } from '../../../../../rendering/materials/mesh-physical/material-tsl';
import { MeshPickingMaterial } from '../../../../../rendering/picking/mesh/material';
import { MeshPickingTSLMaterial } from '../../../../../rendering/picking/mesh/material-tsl';
import { MESH_PICK_FRAGMENT_SHADER } from '../../../../../rendering/picking/mesh/shaders';
import { makePickHarness, uniformValue } from '../render-pick-helper';

describe('mesh pick near fade is switchable', () => {
  it('GLSL: the fade is gated on uNearFade (1 = apply, 0 = none)', () => {
    expect(MESH_PICK_FRAGMENT_SHADER).toContain('uniform int uNearFade;');
    expect(MESH_PICK_FRAGMENT_SHADER).toMatch(
      /float nearFade = \(uNearFade == 1\)\s*\?\s*perspectiveNearFade\(/
    );
  });

  it.each([
    ['GLSL', () => new MeshPickingMaterial({ nodeId: 1 })],
    ['TSL', () => new MeshPickingTSLMaterial({ nodeId: 1 })],
  ])('%s pick material applies the fade by default', (_n, make) => {
    expect(uniformValue(make(), 'uNearFade')).toBe(1);
  });
});

describe.each([
  ['GLSL', () => new MeshPickingMaterial({ nodeId: 1 })],
  ['TSL', () => new MeshPickingTSLMaterial({ nodeId: 1 })],
])('renderPickBuffer sets the pick near fade from the visual material [%s]', (_n, makePick) => {
  it.each([
    ['physical GLSL', () => new PhysicalMeshMaterial({}), 0],
    ['physical TSL', () => new PhysicalMeshTSLMaterial({}), 0],
    ['house mesh', () => new MeshMaterial({}), 1],
  ])('%s visual sets the matching uNearFade', (_v, makeMain, expected) => {
    const { register, renderPickBuffer } = makePickHarness();
    const pick = makePick();
    register(makeMain(), pick);
    renderPickBuffer();
    expect(uniformValue(pick, 'uNearFade')).toBe(expected);
  });
});
