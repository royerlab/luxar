/**
 * syncPointMaterialWithGeometry tests.
 *
 * Verifies that placeholder→real-data geometry commits propagate the
 * dtype-aware radius scale AND the geometry-owned point-texture binding
 * onto the render and pick material uniforms. Without this sync, a
 * normalized Uint8 radii geometry rendered with a placeholder-built
 * material would scale by 1.0 instead of `max_radius`, and a pool
 * acquire that handed the node a different geometry+texture pair would
 * leave the materials sampling the OLD texture.
 * (Sharpness has no scale — it is authored natively in [0, 1].)
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../rendering/materials/point/material-glsl';
import { PointPickingMaterial } from '../../../rendering/picking/point/material';
import { syncPointMaterialWithGeometry } from '../../../data/scene-loader/commit/commit-points-geometry';
import { attachPointStorage } from '../../../rendering/point-geometry';

describe('syncPointMaterialWithGeometry', () => {
  it('updates radiusScale uniform from geometry.userData', () => {
    const mat = new PointMaterial({ radiusScale: 1.0 });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    geometry.userData = { radiusScale: 4.0 };
    const points = new THREE.Mesh(geometry, mat);

    syncPointMaterialWithGeometry(points);

    expect(mat.uniforms.radiusScale.value).toBe(4.0);
  });

  it('falls back to 1.0 when geometry.userData is missing', () => {
    const mat = new PointMaterial({ radiusScale: 4.0 });
    // Reset uniforms to non-default to verify they are written
    mat.uniforms.radiusScale.value = 99;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    const points = new THREE.Mesh(geometry, mat);
    syncPointMaterialWithGeometry(points);
    expect(mat.uniforms.radiusScale.value).toBe(1.0);
  });

  it('also propagates to the linked picking material', () => {
    const mat = new PointMaterial();
    const pickMat = new PointPickingMaterial({ nodeId: 1, radiusScale: 1.0 });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    geometry.userData = { radiusScale: 8.0 };
    const points = new THREE.Mesh(geometry, mat);
    const pickNode = new THREE.Mesh(geometry, pickMat);
    points.userData.pickNode = pickNode;

    syncPointMaterialWithGeometry(points);

    expect(mat.uniforms.radiusScale.value).toBe(8.0);
    expect(pickMat.uniforms.radiusScale.value).toBe(8.0);
  });

  it('rebinds the geometry-owned point texture on render AND pick materials', () => {
    // Mirrors syncGSplatMaterialWithGeometry: the pool may hand the
    // node a DIFFERENT geometry+texture pair (growth / best-fit reuse),
    // so the commit sync must re-point uPointTex on both materials.
    const mat = new PointMaterial();
    const pickMat = new PointPickingMaterial({ nodeId: 1 });
    const geometry = new THREE.InstancedBufferGeometry();
    const texture = attachPointStorage(geometry, 4);
    const points = new THREE.Mesh(geometry, mat);
    const pickNode = new THREE.Mesh(geometry, pickMat);
    points.userData.pickNode = pickNode;

    syncPointMaterialWithGeometry(points);

    expect(mat.uniforms.uPointTex.value).toBe(texture);
    expect(pickMat.uniforms.uPointTex.value).toBe(texture);
  });

  it('leaves the material texture binding alone when the geometry has no point texture', () => {
    // Placeholder-era geometries (plain BufferGeometry without attached
    // storage) must not clobber an existing binding with null.
    const mat = new PointMaterial();
    const bound = new THREE.DataTexture(
      new Float32Array(12),
      3,
      1,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    mat.updatePointTexture(bound);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    const points = new THREE.Mesh(geometry, mat);

    syncPointMaterialWithGeometry(points);

    expect(mat.uniforms.uPointTex.value).toBe(bound);
  });

  it('is a no-op when material is not a PointMaterial', () => {
    // Defensive: external callers might attach a non-Luxar material.
    const otherMat = new THREE.MeshBasicMaterial();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    geometry.userData = { radiusScale: 2.0 };
    const points = new THREE.Mesh(geometry, otherMat as unknown as THREE.ShaderMaterial);
    expect(() => syncPointMaterialWithGeometry(points)).not.toThrow();
  });
});
