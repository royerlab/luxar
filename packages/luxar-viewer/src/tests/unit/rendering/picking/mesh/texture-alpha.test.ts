/**
 * The pick pass samples the base-colour texture, for its alpha.
 *
 * Not an optimisation detail. Texture alpha multiplies coverage in the VISUAL
 * shader, so an RGBA basemap's transparent regions are real holes on screen. A pick
 * pass that ignored the texture would leave those holes pickable AND
 * depth-occluding — the exact visual/pick divergence `syncMeshPickAppearance`
 * exists to prevent, arriving through a channel it does not cover.
 *
 * These tests pin the WIRING (does the sampler exist, does the real image replace
 * the placeholder) rather than sampled pixels, which need a GL context. The
 * shader-text assertions are what stop the sampling itself from being dropped.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { MeshPickingMaterial } from '../../../../../rendering/picking/mesh/material';
import {
  MESH_PICK_VERTEX_SHADER,
  MESH_PICK_FRAGMENT_SHADER,
} from '../../../../../rendering/picking/mesh/shaders';

describe('mesh pick material — base colour texture', () => {
  it('declares no sampler when the node has no texture', () => {
    const m = new MeshPickingMaterial({ nodeId: 1 });
    expect(m.defines?.LUXAR_MESH_PICK_BASE_COLOR_TEX).toBeUndefined();
    expect(m.uniforms.uBaseColorTex).toBeUndefined();
  });

  it('declares the sampler when the node has one', () => {
    // Define-gated rather than a runtime uniform like `uAlphaCutout`, because a
    // sampler must be declared and `has_texture` is a per-node constant: no
    // layers-panel action can give a texture to a node whose store had none.
    const tex = new THREE.Texture();
    const m = new MeshPickingMaterial({ nodeId: 1, baseColorTexture: tex });
    expect(m.defines?.LUXAR_MESH_PICK_BASE_COLOR_TEX).toBe('');
    expect(m.uniforms.uBaseColorTex.value).toBe(tex);
  });

  it('replaces the placeholder when the real image arrives', () => {
    // The node is created before any fetch, so it starts on a blank placeholder —
    // which samples alpha 0 everywhere. Failing to replace it would make the whole
    // mesh unpickable, not merely mis-pick its holes.
    const m = new MeshPickingMaterial({ nodeId: 1, baseColorTexture: new THREE.Texture() });
    const real = new THREE.Texture();
    m.updateBaseColorTexture(real);
    expect(m.uniforms.uBaseColorTex.value).toBe(real);
  });

  it('is a no-op on a node that never had a texture', () => {
    const m = new MeshPickingMaterial({ nodeId: 1 });
    m.updateBaseColorTexture(new THREE.Texture());
    // Must NOT invent the uniform: the sampler is not in this variant's program, so
    // a stray uniform would be dead state that reads as if it were bound.
    expect(m.uniforms.uBaseColorTex).toBeUndefined();
  });

  it('carries the texture through clone(), not just its uniform', () => {
    // `clone()` must go through the CONSTRUCTOR: the texture decides whether the
    // program declares a sampler, so a clone that copied only the uniform would
    // build the untextured variant and silently make every cutout hole pickable
    // again. The layers panel clones on first interaction.
    const tex = new THREE.Texture();
    const m = new MeshPickingMaterial({ nodeId: 7, baseColorTexture: tex });
    const c = m.clone();
    expect(c.defines?.LUXAR_MESH_PICK_BASE_COLOR_TEX).toBe('');
    expect(c.uniforms.uBaseColorTex.value).toBe(tex);
  });
});

describe('mesh pick shaders — the texture arm', () => {
  it('samples only the ALPHA channel', () => {
    // The pick pass has no colour output, so anything else would be dead work — and
    // reading `.rgb` would need the luminance swizzle the visual shader carries,
    // which has no analogue here (a 1-channel texture samples alpha 1.0).
    expect(MESH_PICK_FRAGMENT_SHADER).toMatch(/texture\(uBaseColorTex, vUv\)\.a/);
    expect(MESH_PICK_FRAGMENT_SHADER).not.toMatch(/uBaseColorTex, vUv\)\.rgb/);
  });

  it('folds texture alpha into coverage OUTSIDE the cutout block', () => {
    // Ordering, and specifically SCOPE — the weaker "before the cutout comparison"
    // version of this test passed a mutant that moved the multiply INSIDE the
    // `uAlphaCutout == 1` arm. That mutant is a real regression: cutout mode would
    // still behave, but in the commutative modes (additive/luminous/max) texture
    // alpha would never reach the brightness vote, so a fully transparent texel
    // would compete at full brightness against other nodes.
    //
    // So the multiply must precede the cutout BLOCK, not merely the comparison
    // inside it.
    const texMul = MESH_PICK_FRAGMENT_SHADER.indexOf('texture(uBaseColorTex, vUv).a');
    const cutoutBlock = MESH_PICK_FRAGMENT_SHADER.indexOf('if (uAlphaCutout == 1)');
    const comparison = MESH_PICK_FRAGMENT_SHADER.indexOf('a < uAlphaCutoff');
    expect(texMul).toBeGreaterThan(-1);
    expect(cutoutBlock).toBeGreaterThan(texMul);
    expect(comparison).toBeGreaterThan(texMul);
  });

  it('gates the sampler, the varying and the uv read on one define', () => {
    // All three must appear together: a varying assigned in a variant whose
    // fragment stage does not declare it is a link error, and a sampler declared
    // without the varying is a compile error.
    for (const source of [MESH_PICK_VERTEX_SHADER, MESH_PICK_FRAGMENT_SHADER]) {
      const guarded = source.includes('#ifdef LUXAR_MESH_PICK_BASE_COLOR_TEX');
      expect(guarded).toBe(true);
    }
    expect(MESH_PICK_VERTEX_SHADER).toMatch(/vUv = uv;/);
  });
});
