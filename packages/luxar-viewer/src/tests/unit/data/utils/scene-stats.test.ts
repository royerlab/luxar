/**
 * Unit tests for `computeSceneStats`.
 *
 * Pure traversal extracted from `zarr-loader.ts::logSceneStats`. Uses real
 * THREE objects (Group, Mesh, InstancedBufferGeometry, InstancedBufferAttribute) —
 * none of these need a WebGL context, so tests run without any mocks.
 *
 * After the container migration, points render as `THREE.Mesh` with
 * `userData.nodeType === 'points'` and `InstancedBufferGeometry.instanceCount`,
 * mirroring lines and gsplats.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { computeSceneStats } from '../../../../data/stats/scene-stats';

function makePoints(
  count: number,
  opts: { spatialIndex?: boolean; instanceCount?: number } = {}
): THREE.Mesh {
  const geom = new THREE.InstancedBufferGeometry();
  geom.instanceCount = opts.instanceCount ?? count;
  geom.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3));
  const mesh = new THREE.Mesh(geom);
  mesh.userData = {
    nodeType: 'points',
    ...(opts.spatialIndex ? { attrs: { has_spatial_index: true } } : {}),
  };
  return mesh;
}

function makeGSplatsMesh(splatCount: number, opts: { spatialIndex?: boolean } = {}): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.userData = {
    nodeType: 'gsplats',
    visibleSplatCount: splatCount,
    ...(opts.spatialIndex ? { attrs: { has_spatial_index: true } } : {}),
  };
  return mesh;
}

describe('computeSceneStats', () => {
  it('returns null for a scene without a traverse method', () => {
    expect(computeSceneStats(null)).toBeNull();
    expect(computeSceneStats(undefined)).toBeNull();
    // An object that looks scene-like but has no traverse:
    expect(computeSceneStats({} as THREE.Object3D)).toBeNull();
  });

  it('returns zeroed counts for an empty scene', () => {
    const scene = new THREE.Group();
    expect(computeSceneStats(scene)).toEqual({
      pointsObjects: 0,
      totalPoints: 0,
      gsplatsObjects: 0,
      totalGSplats: 0,
      spatialIndexed: 0,
    });
  });

  it('counts points meshes and sums visible instance counts', () => {
    const scene = new THREE.Group();
    scene.add(makePoints(100));
    // Simulate pooled geometry: aCenter has capacity for 250, but only 75
    // instances are visible/drawn.
    scene.add(makePoints(250, { instanceCount: 75 }));

    const stats = computeSceneStats(scene)!;
    expect(stats.pointsObjects).toBe(2);
    expect(stats.totalPoints).toBe(175);
  });

  it('counts a points mesh without an aCenter attribute (count contributes 0)', () => {
    const scene = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry());
    mesh.userData = { nodeType: 'points' };
    scene.add(mesh);

    const stats = computeSceneStats(scene)!;
    expect(stats.pointsObjects).toBe(1);
    expect(stats.totalPoints).toBe(0);
  });

  it('counts gsplats meshes (Mesh + nodeType=gsplats) and sums visibleSplatCount', () => {
    const scene = new THREE.Group();
    scene.add(makeGSplatsMesh(1000));
    scene.add(makeGSplatsMesh(500));

    const stats = computeSceneStats(scene)!;
    expect(stats.gsplatsObjects).toBe(2);
    expect(stats.totalGSplats).toBe(1500);
  });

  it('treats a Mesh without nodeType=gsplats as ignorable (not a gsplat)', () => {
    const scene = new THREE.Group();
    const meshWithoutTag = new THREE.Mesh();
    meshWithoutTag.userData = { visibleSplatCount: 999 };
    scene.add(meshWithoutTag);

    const stats = computeSceneStats(scene)!;
    expect(stats.gsplatsObjects).toBe(0);
    expect(stats.totalGSplats).toBe(0);
  });

  it('defaults visibleSplatCount to 0 when not present on userData', () => {
    const scene = new THREE.Group();
    const mesh = new THREE.Mesh();
    mesh.userData = { nodeType: 'gsplats' }; // no visibleSplatCount
    scene.add(mesh);

    const stats = computeSceneStats(scene)!;
    expect(stats.gsplatsObjects).toBe(1);
    expect(stats.totalGSplats).toBe(0);
  });

  it('counts spatialIndexed across both points and gsplats', () => {
    const scene = new THREE.Group();
    scene.add(makePoints(100, { spatialIndex: true }));
    scene.add(makePoints(50)); // no spatial index
    scene.add(makeGSplatsMesh(1000, { spatialIndex: true }));
    scene.add(makeGSplatsMesh(500)); // no spatial index

    const stats = computeSceneStats(scene)!;
    expect(stats.spatialIndexed).toBe(2);
    expect(stats.pointsObjects).toBe(2);
    expect(stats.gsplatsObjects).toBe(2);
  });

  it('descends into nested Groups', () => {
    const scene = new THREE.Group();
    const subgroup = new THREE.Group();
    subgroup.add(makePoints(100));
    subgroup.add(makeGSplatsMesh(50));
    scene.add(subgroup);

    const stats = computeSceneStats(scene)!;
    expect(stats.pointsObjects).toBe(1);
    expect(stats.totalPoints).toBe(100);
    expect(stats.gsplatsObjects).toBe(1);
    expect(stats.totalGSplats).toBe(50);
  });
});
