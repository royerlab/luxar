/**
 * Ray-parity inside test for closed meshes — what tells the refraction split that the
 * camera is inside a refracting shell. Pinned on the shapes a bounding-volume test gets
 * wrong (a torus's hole, a long box's corners) as well as the sphere the bubbles are.
 */

import { describe, it, expect, vi } from 'vitest';
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

  it('uses only the drawn index prefix, including after a draw-range change', () => {
    const inner = new THREE.IcosahedronGeometry(1, 1);
    const outer = new THREE.IcosahedronGeometry(2, 1);
    const positions = new Float32Array([
      ...(inner.getAttribute('position').array as Float32Array),
      ...(outer.getAttribute('position').array as Float32Array),
    ]);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(Array.from({ length: positions.length / 3 }, (_, i) => i));
    geometry.setDrawRange(0, inner.getAttribute('position').count);
    const shell = mesh(geometry);
    expect(isPointInsideClosedMesh(shell, v(0, 0, 0))).toBe(true);
    expect(isPointInsideClosedMesh(shell, v(1.5, 0, 0))).toBe(false);
    geometry.setDrawRange(
      inner.getAttribute('position').count,
      outer.getAttribute('position').count
    );
    expect(isPointInsideClosedMesh(shell, v(1.5, 0, 0))).toBe(true);
  });

  it('rechecks after a position update and reuses the parity result for a still camera', () => {
    const shell = mesh(new THREE.IcosahedronGeometry(1, 1));
    const eye = v(0.5, 0, 0);
    const intersect = vi.spyOn(THREE.Ray.prototype, 'intersectTriangle');
    try {
      expect(isPointInsideClosedMesh(shell, eye)).toBe(true);
      const firstCount = intersect.mock.calls.length;
      expect(firstCount).toBeGreaterThan(0);
      expect(isPointInsideClosedMesh(shell, eye)).toBe(true);
      expect(intersect).toHaveBeenCalledTimes(firstCount);
      const position = shell.geometry.getAttribute('position') as THREE.BufferAttribute;
      for (let i = 0; i < position.count; i++)
        position.setXYZ(i, 2 * position.getX(i), 2 * position.getY(i), 2 * position.getZ(i));
      position.needsUpdate = true;
      shell.geometry.computeBoundingBox();
      shell.geometry.computeBoundingSphere();
      expect(isPointInsideClosedMesh(shell, v(1.5, 0, 0))).toBe(true);
      expect(intersect.mock.calls.length).toBeGreaterThan(firstCount);
    } finally {
      intersect.mockRestore();
    }
  });

  it('refuses open surfaces even when a parity ray crosses one', () => {
    const plane = mesh(new THREE.PlaneGeometry(4, 4));
    const dome = mesh(new THREE.SphereGeometry(2, 16, 8, 0, 2 * Math.PI, 0, Math.PI / 2));
    expect(isPointInsideClosedMesh(plane, v(0, 0, -0.5))).toBe(false);
    expect(isPointInsideClosedMesh(dome, v(0, 0.5, 0))).toBe(false);
  });

  it('invalidates the closure check when an existing index buffer changes', () => {
    const shell = mesh(new THREE.BoxGeometry(2, 2, 2));
    expect(isPointInsideClosedMesh(shell, v(0, 0, 0))).toBe(true);
    const index = shell.geometry.getIndex()!;
    index.setX(0, index.getX(1));
    index.needsUpdate = true;
    expect(isPointInsideClosedMesh(shell, v(0, 0, 0))).toBe(false);
  });

  it('rechecks a cached eye when the mesh world matrix changes', () => {
    const shell = mesh(new THREE.BoxGeometry(4, 0.5, 0.5));
    const eye = v(1, 0.2, 0);
    expect(isPointInsideClosedMesh(shell, eye)).toBe(true);
    shell.rotation.z = Math.PI / 4;
    shell.updateMatrixWorld(true);
    expect(isPointInsideClosedMesh(shell, eye)).toBe(false);
  });
});
