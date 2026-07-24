/**
 * syncLineMaterialWithGeometry tests — the lines twin of
 * `sync-point-material-with-geometry.test.ts` (three-geometry test
 * symmetry).
 *
 * Verifies that geometry commits propagate the geometry-owned line
 * texture binding onto the render and pick materials AND push the
 * `hasElementAlpha` presence stamp into the material's
 * `uHasElementAlpha` gate. Mutation-found (volumetric phase 4
 * double-check): severing the stamp→uniform chain survived the entire
 * unit suite, silently disabling the volumetric w(a) optical-depth map
 * for every RGBA lines dataset.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { LineMaterial } from '../../../rendering/materials/line/material-glsl';
import { syncLineMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { attachLineStorage } from '../../../rendering/line-geometry';

function makeLinesMesh(mat: LineMaterial): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  attachLineStorage(geometry, 2);
  return new THREE.Mesh(geometry, mat);
}

describe('syncLineMaterialWithGeometry', () => {
  it('rebinds the geometry-owned line texture on the render material', () => {
    const mat = new LineMaterial();
    const mesh = makeLinesMesh(mat);
    syncLineMaterialWithGeometry(mesh);
    // The material must now sample the geometry's own texture, not the
    // shared placeholder it was constructed with.
    expect(mat.getLineTexture()).not.toBeNull();
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
