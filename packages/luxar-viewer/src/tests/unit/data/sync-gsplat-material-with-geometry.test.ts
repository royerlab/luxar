/**
 * syncGSplatMaterialWithGeometry tests — the gsplat twin of
 * `sync-line-material-with-geometry.test.ts` (three-geometry test
 * symmetry).
 *
 * Verifies that geometry commits propagate the geometry-owned splat
 * texture binding onto the render and pick materials AND push the
 * `hasElementAlpha` presence stamp into the material's
 * `uHasElementAlpha` gate. Mutation-found (volumetric phase 4
 * double-check, lines twin): severing the stamp→uniform chain survived
 * the entire unit suite, silently disabling the volumetric w(a)
 * optical-depth map for every RGBA dataset. Also pins the stamp source
 * itself: `stampGSplatPresenceFlags` derives the flag from
 * `colorComponents` (the fixed 4-texel layout always carries the
 * texel3.y alpha slot, so presence is not readable off the texture).
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { GSplatMaterial } from '../../../rendering/materials/gsplat/material-glsl';
import { syncGSplatMaterialWithGeometry } from '../../../rendering/material-sync-helpers';
import { attachSplatStorage, stampGSplatPresenceFlags } from '../../../rendering/gsplat-geometry';

function makeGSplatsMesh(mat: GSplatMaterial): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  attachSplatStorage(geometry, 2);
  return new THREE.Mesh(geometry, mat);
}

describe('stampGSplatPresenceFlags', () => {
  it('stamps hasElementAlpha=true for RGBA sources (colorComponents=4)', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    stampGSplatPresenceFlags(geometry, { colorComponents: 4 });
    expect(geometry.userData.hasElementAlpha).toBe(true);
  });

  it('stamps hasElementAlpha=false for RGB (3) and for an absent colorComponents', () => {
    // Refreshed on EVERY write (pool geometries are reused across
    // tenants) — an RGBA tenant's stamp must not leak onto the next
    // RGB dataset.
    const geometry = new THREE.InstancedBufferGeometry();
    stampGSplatPresenceFlags(geometry, { colorComponents: 4 });
    stampGSplatPresenceFlags(geometry, { colorComponents: 3 });
    expect(geometry.userData.hasElementAlpha).toBe(false);
    stampGSplatPresenceFlags(geometry, { colorComponents: 4 });
    stampGSplatPresenceFlags(geometry, {});
    expect(geometry.userData.hasElementAlpha).toBe(false);
  });
});

describe('syncGSplatMaterialWithGeometry', () => {
  it('rebinds the geometry-owned splat texture on the render material', () => {
    const mat = new GSplatMaterial();
    const mesh = makeGSplatsMesh(mat);
    syncGSplatMaterialWithGeometry(mesh);
    // The material must now sample the geometry's own texture, not the
    // shared placeholder it was constructed with.
    expect(mat.getSplatTexture()).not.toBeNull();
  });

  it('pushes uHasElementAlpha=1 when geometry.userData.hasElementAlpha is true (RGBA colors)', () => {
    // The texel-write paths stamp `hasElementAlpha` via
    // stampGSplatPresenceFlags when the dataset carries per-splat
    // alphas (texel3.y); the sync must land it on the uniform that
    // gates the volumetric w(a) map.
    const mat = new GSplatMaterial();
    const mesh = makeGSplatsMesh(mat);
    mesh.geometry.userData.hasElementAlpha = true;
    syncGSplatMaterialWithGeometry(mesh);
    expect(mat.uniforms.uHasElementAlpha.value).toBe(1);
  });

  it('resets uHasElementAlpha=0 when the geometry has no alpha stamp (pool-swap leak guard)', () => {
    const mat = new GSplatMaterial();
    mat.updateHasElementAlpha(true); // previous tenant left it on
    const mesh = makeGSplatsMesh(mat);
    mesh.geometry.userData.hasElementAlpha = false;
    syncGSplatMaterialWithGeometry(mesh);
    expect(mat.uniforms.uHasElementAlpha.value).toBe(0);
  });

  it('is a no-op when the geometry has no splat texture (placeholder mesh)', () => {
    const mat = new GSplatMaterial();
    mat.updateHasElementAlpha(true);
    const mesh = new THREE.Mesh(new THREE.InstancedBufferGeometry(), mat);
    syncGSplatMaterialWithGeometry(mesh);
    // No texture → early return; the flag must not be touched either
    // (the placeholder carries no presence information).
    expect(mat.uniforms.uHasElementAlpha.value).toBe(1);
  });

  it('stamp → sync end-to-end: colorComponents=4 lands uHasElementAlpha=1 on the material', () => {
    // The full chain the commit runs: stampGSplatPresenceFlags writes
    // the geometry stamp from the staged source's colorComponents, and
    // syncGSplatMaterialWithGeometry pushes it into the uniform gate.
    const mat = new GSplatMaterial();
    const mesh = makeGSplatsMesh(mat);
    stampGSplatPresenceFlags(mesh.geometry, { colorComponents: 4 });
    syncGSplatMaterialWithGeometry(mesh);
    expect(mat.uniforms.uHasElementAlpha.value).toBe(1);
  });
});
