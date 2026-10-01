/**
 * Ray-parity inside test for closed meshes — what tells the refraction split that the
 * camera is inside a refracting shell. Pinned on the shapes a bounding-volume test gets
 * wrong (a torus's hole, a long box's corners) as well as the sphere the bubbles are.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { isPointInsideClosedMesh } from '../../../../../rendering/post-processing/post-processing-manager/inside-closed-mesh';

const v = (x: number, y: number, z: number): THREE.Vector3 => new THREE.Vector3(x, y, z);

function mesh(geometry: THREE.BufferGeometry, place?: (m: THREE.Mesh) => void): THREE.Mesh {
  const m = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  place?.(m);
  m.updateMatrixWorld(true);
  return m;
}

describe('isPointInsideClosedMesh', () => {
  it('a sphere: inside near the centre and the wall, outside past it', () => {
    const sphere = mesh(new THREE.IcosahedronGeometry(1, 3));
    expect(isPointInsideClosedMesh(sphere, v(0, 0, 0))).toBe(true);
    expect(isPointInsideClosedMesh(sphere, v(0.6, -0.5, 0.3))).toBe(true);
    expect(isPointInsideClosedMesh(sphere, v(0, 0, 1.2))).toBe(false);
    expect(isPointInsideClosedMesh(sphere, v(5, 5, 5))).toBe(false);
  });

  it('a torus: its hole is outside although it is inside the bounding sphere', () => {
    const torus = mesh(new THREE.TorusGeometry(1, 0.3, 24, 48));
    expect(isPointInsideClosedMesh(torus, v(0, 0, 0))).toBe(false);
    expect(isPointInsideClosedMesh(torus, v(1, 0, 0))).toBe(true);
    expect(isPointInsideClosedMesh(torus, v(0, 1, 0.1))).toBe(true);
  });

  it('a long box: its bounding-sphere corners are outside', () => {
    const box = mesh(new THREE.BoxGeometry(4, 0.5, 0.5));
    expect(isPointInsideClosedMesh(box, v(1.5, 0, 0))).toBe(true);
    expect(isPointInsideClosedMesh(box, v(0, 1, 0))).toBe(false);
  });

  it('follows the world transform, and works on non-indexed geometry', () => {
    const moved = mesh(new THREE.IcosahedronGeometry(1, 2).toNonIndexed(), (m) => {
      m.position.set(10, 0, 0);
      m.scale.set(2, 1, 1);
    });
    expect(isPointInsideClosedMesh(moved, v(11.5, 0, 0))).toBe(true);
    expect(isPointInsideClosedMesh(moved, v(0, 0, 0))).toBe(false);
    expect(isPointInsideClosedMesh(moved, v(10, 1.2, 0))).toBe(false);
  });

  it('a mesh with no triangles has no inside', () => {
    expect(isPointInsideClosedMesh(mesh(new THREE.BufferGeometry()), v(0, 0, 0))).toBe(false);
  });
});
