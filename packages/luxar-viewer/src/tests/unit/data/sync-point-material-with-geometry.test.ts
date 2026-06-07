/**
 * syncPointMaterialWithGeometry tests.
 *
 * Verifies that placeholder→real-data geometry commits propagate the
 * dtype-aware radius scale onto the render and pick material uniforms.
 * Without this sync, a normalized Uint8 radii geometry rendered with a
 * placeholder-built material would scale by 1.0 instead of `max_radius`.
 * (Sharpness has no scale — it is authored natively in [0, 1].)
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../rendering/materials/point/material-glsl';
import { PointPickingMaterial } from '../../../rendering/picking/point/material';
import { syncPointMaterialWithGeometry } from '../../../data/scene-loader/commit/commit-points-geometry';

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
