// @vitest-environment jsdom
/**
 * The mesh pick material follows the visual material's LIVE opacity.
 *
 * A mesh's pick coverage is node opacity × per-vertex alpha (spec §6.5), so a
 * mesh dissolved on screen must stop being pickable. The pick opacity used to
 * be pushed only by the Layers panel (the authored value), but the live visual
 * `uOpacity` has other writers that never touched the pick material: the LOD
 * cross-fade (`scene/lod-fade.ts`) and the embedder exposure path
 * (`core/layer/luxar-layer.ts`). A faded-out mesh level then stayed pickable at
 * full coverage. The pick render now copies the visual `uOpacity` across for
 * every node, whichever writer set it.
 */
import { describe, expect, it } from 'vitest';

import { MeshMaterial } from '../../../../../rendering/materials/mesh/material-glsl';
import { MeshTSLMaterial } from '../../../../../rendering/materials/mesh/material-tsl';
import { PhysicalMeshMaterial } from '../../../../../rendering/materials/mesh-physical/material-glsl';
import { MeshPickingMaterial } from '../../../../../rendering/picking/mesh/material';
import { MeshPickingTSLMaterial } from '../../../../../rendering/picking/mesh/material-tsl';
import { makePickHarness, setUniform, uniformValue } from '../render-pick-helper';

describe.each([
  ['GLSL', () => new MeshMaterial({}), () => new MeshPickingMaterial({ nodeId: 1 })],
  ['TSL', () => new MeshTSLMaterial({}), () => new MeshPickingTSLMaterial({ nodeId: 1 })],
])('mesh pick opacity sync [%s]', (_name, makeMain, makePick) => {
  it('a visual-only opacity write (LOD fade, exposure) reaches the pick material', () => {
    const { register, renderPickBuffer } = makePickHarness();
    const main = makeMain();
    const pick = makePick();
    register(main, pick);
    // What lod-fade.ts / the embedder exposure path do: write the VISUAL uniform only.
    setUniform(main, 'uOpacity', 0.125);

    renderPickBuffer();

    expect(uniformValue(pick, 'uOpacity')).toBe(0.125);
  });
});

describe('mesh pick opacity sync [physical]', () => {
  it("a physical mesh's opacity (kept off the uniform record) reaches the pick material", () => {
    const { register, renderPickBuffer } = makePickHarness();
    const main = new PhysicalMeshMaterial({});
    const pick = new MeshPickingMaterial({ nodeId: 1 });
    register(main, pick);
    main.updateOpacity(0.25);

    renderPickBuffer();

    expect(uniformValue(pick, 'uOpacity')).toBe(0.25);
  });
});
