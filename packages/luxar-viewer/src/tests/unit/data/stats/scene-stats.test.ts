/**
 * Unit tests for `computeSceneStats`.
 *
 * Uses real THREE objects (Group, Mesh, InstancedBufferGeometry,
 * InstancedBufferAttribute); none of these need a WebGL context, so tests
 * run without any mocks. Points, Lines, and GSplats all report visible
 * counts through mesh userData or `InstancedBufferGeometry.instanceCount`.
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

function makeLinesMesh(
  segmentCount: number,
  opts: { spatialIndex?: boolean; instanceCount?: number } = {}
): THREE.Mesh {
  const geom = new THREE.InstancedBufferGeometry();
  geom.instanceCount = opts.instanceCount ?? segmentCount;
  const mesh = new THREE.Mesh(geom);
  mesh.userData = {
    nodeType: 'lines',
    visibleSegmentCount: segmentCount,
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
      linesObjects: 0,
      totalSegments: 0,
      gsplatsObjects: 0,
      totalGSplats: 0,
      meshObjects: 0,
      totalTriangles: 0,
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

  // data.md OOS: three-geometry symmetry — Lines were entirely missing
  // from SceneStats. These tests pin the new linesObjects + totalSegments
  // contract parallel to Points / GSplats.
  it('counts lines meshes and sums visible instance counts', () => {
    const scene = new THREE.Group();
    scene.add(makeLinesMesh(80)); // 80 segments
    // Simulate over-allocated pooled geometry: instanceCount drawn (60)
    // is less than the userData fallback (200).
    scene.add(makeLinesMesh(200, { instanceCount: 60 }));

    const stats = computeSceneStats(scene)!;
    expect(stats.linesObjects).toBe(2);
    // Both meshes have isInstancedBufferGeometry, so totalSegments uses
    // instanceCount (80 + 60 = 140), NOT the userData fallback.
    expect(stats.totalSegments).toBe(140);
  });

  it('falls back to visibleSegmentCount when geometry is not instanced (test synth path)', () => {
    const scene = new THREE.Group();
    // Synthesize a Lines mesh with a plain BufferGeometry to exercise the
    // fallback branch — tests sometimes do this to avoid full GPU setup.
    const mesh = new THREE.Mesh(new THREE.BufferGeometry());
    mesh.userData = { nodeType: 'lines', visibleSegmentCount: 42 };
    scene.add(mesh);

    const stats = computeSceneStats(scene)!;
    expect(stats.linesObjects).toBe(1);
    expect(stats.totalSegments).toBe(42);
  });

  it('counts a lines mesh with neither instanceCount nor visibleSegmentCount as contributing 0', () => {
    const scene = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry());
    mesh.userData = { nodeType: 'lines' }; // intentionally bare
    scene.add(mesh);

    const stats = computeSceneStats(scene)!;
    expect(stats.linesObjects).toBe(1);
    expect(stats.totalSegments).toBe(0);
  });

  it('counts spatialIndexed across points, lines, AND gsplats', () => {
    const scene = new THREE.Group();
    scene.add(makePoints(100, { spatialIndex: true }));
    scene.add(makePoints(50)); // no spatial index
    scene.add(makeLinesMesh(80, { spatialIndex: true }));
    scene.add(makeLinesMesh(40)); // no spatial index
    scene.add(makeGSplatsMesh(1000, { spatialIndex: true }));
    scene.add(makeGSplatsMesh(500)); // no spatial index

    const stats = computeSceneStats(scene)!;
    // 3 spatial-indexed objects across all three geometry types.
    expect(stats.spatialIndexed).toBe(3);
    expect(stats.pointsObjects).toBe(2);
    expect(stats.linesObjects).toBe(2);
    expect(stats.gsplatsObjects).toBe(2);
  });

  it('descends into nested Groups (including Lines)', () => {
    const scene = new THREE.Group();
    const subgroup = new THREE.Group();
    subgroup.add(makePoints(100));
    subgroup.add(makeLinesMesh(75));
    subgroup.add(makeGSplatsMesh(50));
    scene.add(subgroup);

    const stats = computeSceneStats(scene)!;
    expect(stats.pointsObjects).toBe(1);
    expect(stats.totalPoints).toBe(100);
    expect(stats.linesObjects).toBe(1);
    expect(stats.totalSegments).toBe(75);
    expect(stats.gsplatsObjects).toBe(1);
    expect(stats.totalGSplats).toBe(50);
  });

  it('counts mesh objects and sums their committed triangle counts', () => {
    const scene = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    mesh.userData = { nodeType: 'mesh', visibleTriangleCount: 1200 };
    scene.add(mesh);
    const stats = computeSceneStats(scene)!;
    expect(stats.meshObjects).toBe(1);
    expect(stats.totalTriangles).toBe(1200);
  });

  it('does NOT count a mesh toward spatialIndexed, even if the attr is present', () => {
    // `spatialIndexed` means "nodes that can skip chunks on a slice change". Mesh has
    // no spatial index by design, so counting it would inflate that metric — and a
    // corrupt store setting the attr must not change the answer.
    const scene = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    mesh.userData = {
      nodeType: 'mesh',
      visibleTriangleCount: 5,
      attrs: { has_spatial_index: true },
    };
    scene.add(mesh);
    expect(computeSceneStats(scene)!.spatialIndexed).toBe(0);
  });

  it('treats a missing visibleTriangleCount as 0 rather than NaN', () => {
    const scene = new THREE.Group();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    mesh.userData = { nodeType: 'mesh' };
    scene.add(mesh);
    expect(computeSceneStats(scene)!.totalTriangles).toBe(0);
  });
});
