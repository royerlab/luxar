/**
 * The cube capture's pure pieces: which objects count as physical (and are hidden for
 * the six draws), and the clip planes a probe inside or outside the bounds gets.
 * The orchestration itself is exercised in `scene-environment.test.ts`.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  captureClipPlanes,
  captureSceneCube,
  isPhysicalMeshObject,
} from '../../../../rendering/environment/cube-capture';
import { PhysicalMeshMaterial } from '../../../../rendering/materials/mesh-physical/material-glsl';
import { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';

describe('isPhysicalMeshObject', () => {
  it('is true only for a single physical wrapper material', () => {
    expect(
      isPhysicalMeshObject(new THREE.Mesh(new THREE.BufferGeometry(), new PhysicalMeshMaterial({})))
    ).toBe(true);
    expect(
      isPhysicalMeshObject(new THREE.Mesh(new THREE.BufferGeometry(), new MeshMaterial({})))
    ).toBe(false);
    expect(isPhysicalMeshObject(new THREE.Group())).toBe(false);
    expect(
      isPhysicalMeshObject(
        new THREE.Mesh(new THREE.BufferGeometry(), [
          new PhysicalMeshMaterial({}),
          new PhysicalMeshMaterial({}),
        ])
      )
    ).toBe(false);
  });
});

describe('captureClipPlanes', () => {
  it('reaches past the far side of the bounds from the probe, with a near plane well below it', () => {
    const bounds = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 10);
    const inside = captureClipPlanes(new THREE.Vector3(0, 0, 0), bounds);
    expect(inside.far).toBe(20);
    expect(inside.near).toBeLessThan(inside.far * 1e-3);
    expect(inside.near).toBeGreaterThan(0);
    const outside = captureClipPlanes(new THREE.Vector3(30, 0, 0), bounds);
    expect(outside.far).toBe(80);
  });

  it('has a sane default without bounds', () => {
    expect(captureClipPlanes(new THREE.Vector3(), null)).toEqual({ near: 0.01, far: 1000 });
  });
});

describe('captureSceneCube', () => {
  it('draws with the previous environment cleared and restores it, even when a draw throws', () => {
    const scene = new THREE.Scene();
    const previous = new THREE.Texture();
    scene.environment = previous;
    const physical = new THREE.Mesh(new THREE.BufferGeometry(), new PhysicalMeshMaterial({}));
    scene.add(physical);
    const seenEnvironment: Array<THREE.Texture | null> = [];
    const renderer = {
      isWebGLRenderer: true,
      coordinateSystem: THREE.WebGLCoordinateSystem,
      xr: { enabled: false },
      state: { buffers: { depth: { getReversed: () => false } } },
      getRenderTarget: () => null,
      getActiveCubeFace: () => 0,
      getActiveMipmapLevel: () => 0,
      setRenderTarget: () => {},
      render: () => {
        seenEnvironment.push(scene.environment);
        if (seenEnvironment.length === 3) throw new Error('lost context');
      },
    };
    const target = { texture: new THREE.CubeTexture(), width: 8, height: 8, dispose: () => {} };
    expect(() =>
      captureSceneCube({
        renderer,
        scene,
        target,
        probe: new THREE.Vector3(),
        near: 0.1,
        far: 10,
        root: scene,
      })
    ).toThrow('lost context');
    expect(seenEnvironment).toEqual([null, null, null]);
    // Restored in `finally`: the environment and the hidden mesh.
    expect(scene.environment).toBe(previous);
    expect(physical.visible).toBe(true);
  });
});
