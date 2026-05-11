/**
 * Unit tests for the scene-graph disposal helpers.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  disposeObjectTree,
  clearLoadedSceneContent,
  disposeSceneGraphResources,
} from '../../../../scene/scene-setup/scene-disposal';

function makeMesh(): {
  mesh: THREE.Mesh;
  geometryDispose: ReturnType<typeof vi.fn>;
  materialDispose: ReturnType<typeof vi.fn>;
} {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.MeshBasicMaterial();
  const geometryDispose = vi.spyOn(geometry, 'dispose');
  const materialDispose = vi.spyOn(material, 'dispose');
  const mesh = new THREE.Mesh(geometry, material);
  return { mesh, geometryDispose, materialDispose };
}

function makePoints(): {
  points: THREE.Points;
  geometryDispose: ReturnType<typeof vi.fn>;
  materialDispose: ReturnType<typeof vi.fn>;
} {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial();
  const geometryDispose = vi.spyOn(geometry, 'dispose');
  const materialDispose = vi.spyOn(material, 'dispose');
  const points = new THREE.Points(geometry, material);
  return { points, geometryDispose, materialDispose };
}

describe('disposeObjectTree', () => {
  it('disposes geometry and single material on a Mesh', () => {
    const { mesh, geometryDispose, materialDispose } = makeMesh();
    disposeObjectTree(mesh);
    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
  });

  it('disposes each material on a Mesh with a material array', () => {
    const geometry = new THREE.BufferGeometry();
    const m1 = new THREE.MeshBasicMaterial();
    const m2 = new THREE.MeshBasicMaterial();
    const d1 = vi.spyOn(m1, 'dispose');
    const d2 = vi.spyOn(m2, 'dispose');
    const mesh = new THREE.Mesh(geometry, [m1, m2]);
    disposeObjectTree(mesh);
    expect(d1).toHaveBeenCalledTimes(1);
    expect(d2).toHaveBeenCalledTimes(1);
  });

  it('disposes Points objects (geometry + material)', () => {
    const { points, geometryDispose, materialDispose } = makePoints();
    disposeObjectTree(points);
    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
  });

  it('does not throw on plain Group / Object3D (no geometry to dispose)', () => {
    const group = new THREE.Group();
    expect(() => disposeObjectTree(group)).not.toThrow();
  });

  it('walks descendants depth-first and removes them from their parent', () => {
    // Group → Mesh → child Mesh
    const root = new THREE.Group();
    const { mesh: parent, geometryDispose: parentG } = makeMesh();
    const { mesh: child, geometryDispose: childG } = makeMesh();
    parent.add(child);
    root.add(parent);

    disposeObjectTree(root);

    expect(parentG).toHaveBeenCalledTimes(1);
    expect(childG).toHaveBeenCalledTimes(1);
    // Children removed from their parents during the walk.
    expect(root.children.length).toBe(0);
    expect(parent.children.length).toBe(0);
  });
});

describe('clearLoadedSceneContent', () => {
  it('removes plain meshes and reports the count', () => {
    const scene = new THREE.Scene();
    const { mesh: a } = makeMesh();
    const { mesh: b } = makeMesh();
    scene.add(a);
    scene.add(b);

    const removed = clearLoadedSceneContent(scene);
    expect(removed).toBe(2);
    expect(scene.children.length).toBe(0);
  });

  it('preserves Lights', () => {
    const scene = new THREE.Scene();
    const light = new THREE.AmbientLight(0xffffff);
    const { mesh } = makeMesh();
    scene.add(light);
    scene.add(mesh);

    const removed = clearLoadedSceneContent(scene);
    expect(removed).toBe(1);
    expect(scene.children).toContain(light);
    expect(scene.children).not.toContain(mesh);
  });

  it('preserves objects flagged userData.isBackground', () => {
    const scene = new THREE.Scene();
    const bg = new THREE.Group();
    bg.userData.isBackground = true;
    const { mesh } = makeMesh();
    scene.add(bg);
    scene.add(mesh);

    const removed = clearLoadedSceneContent(scene);
    expect(removed).toBe(1);
    expect(scene.children).toContain(bg);
  });

  it('disposes geometry/material of removed meshes', () => {
    const scene = new THREE.Scene();
    const { mesh, geometryDispose, materialDispose } = makeMesh();
    scene.add(mesh);

    clearLoadedSceneContent(scene);

    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
  });

  it('returns 0 for an empty scene', () => {
    const scene = new THREE.Scene();
    expect(clearLoadedSceneContent(scene)).toBe(0);
  });
});

describe('disposeSceneGraphResources', () => {
  it('disposes geometry and material of every renderable in the scene', () => {
    const scene = new THREE.Scene();
    const { mesh: a, geometryDispose: aG, materialDispose: aM } = makeMesh();
    const { points: p, geometryDispose: pG, materialDispose: pM } = makePoints();
    scene.add(a);
    scene.add(p);

    disposeSceneGraphResources(scene);

    expect(aG).toHaveBeenCalledTimes(1);
    expect(aM).toHaveBeenCalledTimes(1);
    expect(pG).toHaveBeenCalledTimes(1);
    expect(pM).toHaveBeenCalledTimes(1);
  });

  it('does NOT mutate the scene graph (children remain attached)', () => {
    const scene = new THREE.Scene();
    const { mesh } = makeMesh();
    scene.add(mesh);

    disposeSceneGraphResources(scene);
    // Unlike clearLoadedSceneContent, the final-shutdown pass leaves the
    // graph alone — the renderer/scene/camera are about to be dropped anyway.
    expect(scene.children).toContain(mesh);
  });

  it('handles material-array meshes', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.BufferGeometry();
    const m1 = new THREE.MeshBasicMaterial();
    const m2 = new THREE.MeshBasicMaterial();
    const d1 = vi.spyOn(m1, 'dispose');
    const d2 = vi.spyOn(m2, 'dispose');
    const mesh = new THREE.Mesh(geometry, [m1, m2]);
    scene.add(mesh);

    disposeSceneGraphResources(scene);
    expect(d1).toHaveBeenCalledTimes(1);
    expect(d2).toHaveBeenCalledTimes(1);
  });
});
