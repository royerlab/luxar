/**
 * syncLineMaterialWithGeometry tests — the lines twin of
 * `sync-point-material-with-geometry.test.ts` (three-geometry test
 * symmetry).
 *
 * Verifies that geometry commits propagate the geometry-owned line
 * texture binding onto the render AND pick materials by STRICT IDENTITY
 * (not mere non-nullness — every line material is constructed on a
 * non-null shared placeholder, so a non-null assertion survives deleting
 * the production rebind) AND push the `hasElementAlpha` presence stamp
 * into the material's `uHasElementAlpha` gate. Mutation-found (volumetric
 * phase 4 double-check): severing the stamp→uniform chain survived the
 * entire unit suite, silently disabling the volumetric w(a) optical-depth
 * map for every RGBA lines dataset.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { LineMaterial } from '../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../rendering/materials/line/material-tsl';
import { LinePickingMaterial } from '../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../rendering/picking/line/material-tsl';
import { syncLineMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { attachLineStorage, getLineTexture } from '../../../rendering/line-geometry';
import { getPlaceholderElementTexture } from '../../../rendering/element-texture-layout';

/**
 * Build an instanced lines mesh with geometry-owned line-texture
 * storage. Accepts any THREE.Material so the same helper backs the GLSL
 * and TSL render-material meshes.
 */
function makeLinesMesh(mat: THREE.Material): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  attachLineStorage(geometry, 2);
  return new THREE.Mesh(geometry, mat);
}

describe('syncLineMaterialWithGeometry', () => {
  it('rebinds the geometry-owned line texture on the render material (strict identity)', () => {
    const mat = new LineMaterial();
    const mesh = makeLinesMesh(mat);

    // The geometry owns a real DataTexture, distinct from the shared
    // placeholder the material was constructed with.
    const geomTex = getLineTexture(mesh.geometry);
    expect(geomTex).not.toBeNull();
    // Pre-sync: still on the placeholder — a DIFFERENT texture. Asserting
    // strict inequality here (not non-nullness) is what makes the
    // post-sync `toBe` a real proof that the rebind happened.
    expect(mat.getLineTexture()).not.toBe(geomTex);

    syncLineMaterialWithGeometry(mesh);

    // The material must now sample the geometry's own texture by identity.
    expect(mat.getLineTexture()).toBe(geomTex);
  });

  it('rebinds the geometry-owned line texture on BOTH the render and GLSL pick materials', () => {
    const mat = new LineMaterial();
    const mesh = makeLinesMesh(mat);
    const geomTex = getLineTexture(mesh.geometry);
    expect(geomTex).not.toBeNull();

    // Pick shadow node shares the SAME geometry (picking-system.ts wires
    // it into `userData.pickNode`). The GLSL pick material starts with a
    // null texture binding.
    const pickMat = new LinePickingMaterial({ nodeId: 1 });
    const pickNode = new THREE.Mesh(mesh.geometry, pickMat);
    mesh.userData.pickNode = pickNode;

    // Pre-sync: neither material samples the geometry texture.
    expect(mat.getLineTexture()).not.toBe(geomTex);
    expect(pickMat.uniforms.uLineTex.value).not.toBe(geomTex);
    expect(pickMat.uniforms.uLineTex.value).toBeNull();

    syncLineMaterialWithGeometry(mesh);

    // Both must land the geometry-owned texture by identity — so this
    // single test fails if EITHER the render-material or the pick-material
    // update is removed from the helper.
    expect(mat.getLineTexture()).toBe(geomTex);
    expect(pickMat.uniforms.uLineTex.value).toBe(geomTex);
  });

  it('rebinds the geometry-owned line texture on BOTH the render and pick TSL materials', () => {
    // The TSL wrappers take a different rebind code path from the GLSL
    // ones: `updateLineTexture` swaps the uniform-proxy node and rebuilds
    // the shader graph. Both start on the shared placeholder.
    const renderMat = new LineTSLMaterial();
    const mesh = makeLinesMesh(renderMat);
    const geomTex = getLineTexture(mesh.geometry);
    expect(geomTex).not.toBeNull();

    const pickMat = new LinePickingTSLMaterial({ nodeId: 1 });
    const pickNode = new THREE.Mesh(mesh.geometry, pickMat);
    mesh.userData.pickNode = pickNode;

    // Pre-sync: both sit on the shared placeholder, distinct from geomTex.
    const placeholder = getPlaceholderElementTexture();
    expect(renderMat.getLineTexture()).toBe(placeholder);
    expect(pickMat.uniforms.uLineTex?.value).toBe(placeholder);
    expect(renderMat.getLineTexture()).not.toBe(geomTex);
    expect(pickMat.uniforms.uLineTex?.value).not.toBe(geomTex);

    syncLineMaterialWithGeometry(mesh);

    // Both must land the geometry-owned texture by identity.
    expect(renderMat.getLineTexture()).toBe(geomTex);
    expect(pickMat.uniforms.uLineTex?.value).toBe(geomTex);
  });

  it('pushes uHasElementAlpha=1 when geometry.userData.hasElementAlpha is true (RGBA colors)', () => {
    // The texel-write paths stamp `hasElementAlpha` via
    // stampLinePresenceFlags when the dataset carries per-endpoint
    // alphas (texel5.zw); the sync must land it on the uniform that
    // gates the volumetric w(a) map.
    const mat = new LineMaterial();
    const mesh = makeLinesMesh(mat);
    mesh.geometry.userData.hasElementAlpha = true;
    syncLineMaterialWithGeometry(mesh);
    expect(mat.uniforms.uHasElementAlpha.value).toBe(1);
  });

  it('resets uHasElementAlpha=0 when the geometry has no alpha stamp (pool-swap leak guard)', () => {
    const mat = new LineMaterial();
    mat.updateHasElementAlpha(true); // previous tenant left it on
    const mesh = makeLinesMesh(mat);
    mesh.geometry.userData.hasElementAlpha = false;
    syncLineMaterialWithGeometry(mesh);
    expect(mat.uniforms.uHasElementAlpha.value).toBe(0);
  });

  it('is a no-op when the geometry has no line texture (placeholder mesh)', () => {
    const mat = new LineMaterial();
    mat.updateHasElementAlpha(true);
    const mesh = new THREE.Mesh(new THREE.InstancedBufferGeometry(), mat);
    syncLineMaterialWithGeometry(mesh);
    // No texture → early return; the flag must not be touched either
    // (the placeholder carries no presence information).
    expect(mat.uniforms.uHasElementAlpha.value).toBe(1);
  });
});
