/**
 * syncPointMaterialWithGeometry tests.
 *
 * Verifies that placeholder→real-data geometry commits propagate
 * dtype-aware radius/sharpness scales onto the render and pick
 * material uniforms. Without this sync, a normalized Uint8 radii
 * geometry rendered with a placeholder-built material would scale by
 * 1.0 instead of `max_radius`.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../rendering/point-material';
import { PointPickingMaterial } from '../../../rendering/picking/point-picking-material';
import { syncPointMaterialWithGeometry } from '../../../data/scene-loader/commit/commit-points-geometry';

describe('syncPointMaterialWithGeometry', () => {
  it('updates radiusScale and sharpnessScale uniforms from geometry.userData', () => {
    const mat = new PointMaterial({ radiusScale: 1.0, sharpnessScale: 1.0 });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    geometry.userData = { radiusScale: 4.0, sharpnessScale: 31.0 };
    const points = new THREE.Mesh(geometry, mat);

    syncPointMaterialWithGeometry(points);

    expect(mat.uniforms.radiusScale.value).toBe(4.0);
    expect(mat.uniforms.sharpnessScale.value).toBe(31.0);
  });

  it('falls back to 1.0 when geometry.userData is missing', () => {
    const mat = new PointMaterial({ radiusScale: 4.0, sharpnessScale: 31.0 });
    // Reset uniforms to non-default to verify they are written
    mat.uniforms.radiusScale.value = 99;
    mat.uniforms.sharpnessScale.value = 99;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    const points = new THREE.Mesh(geometry, mat);
    syncPointMaterialWithGeometry(points);
    expect(mat.uniforms.radiusScale.value).toBe(1.0);
    expect(mat.uniforms.sharpnessScale.value).toBe(1.0);
  });

  it('also propagates to the linked picking material', () => {
    const mat = new PointMaterial();
    const pickMat = new PointPickingMaterial({ nodeId: 1, radiusScale: 1.0, sharpnessScale: 1.0 });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    geometry.userData = { radiusScale: 8.0, sharpnessScale: 16.0 };
    const points = new THREE.Mesh(geometry, mat);
    const pickNode = new THREE.Mesh(geometry, pickMat);
    points.userData.pickNode = pickNode;

    syncPointMaterialWithGeometry(points);

    expect(mat.uniforms.radiusScale.value).toBe(8.0);
    expect(pickMat.uniforms.radiusScale.value).toBe(8.0);
    expect(pickMat.uniforms.sharpnessScale.value).toBe(16.0);
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
