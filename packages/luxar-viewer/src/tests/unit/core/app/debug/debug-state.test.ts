/**
 * Unit tests for core/debug-state.ts.
 *
 * Pure scene-walking helper — uses real THREE.js objects (Points,
 * Mesh, InstancedBufferGeometry) under jsdom. Only WebGL renderer
 * needs a real GL context; the geometry types we walk over here
 * work fine.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  computeDebugState,
  type DebugStateContext,
} from '../../../../../core/app/debug/debug-state';
import type { SimpleDims } from '../../../../../types/dims';

function makePointCloud(
  count: number,
  options: {
    name?: string;
    visible?: boolean;
    drawRange?: number;
    instanceCount?: number;
    hasColors?: boolean;
    hasRadii?: boolean;
    hasSharpness?: boolean;
  } = {}
): THREE.Mesh {
  // Point clouds are THREE.Mesh with instanced quad geometry and
  // per-instance attributes prefixed `a*`. `computeDebugState` selects
  // on `userData.nodeType === 'points'`.
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = options.instanceCount ?? count;
  geometry.setAttribute(
    'aCenter',
    new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
  );
  if (options.hasColors) {
    geometry.setAttribute(
      'aColor',
      new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3)
    );
  }
  if (options.hasRadii) {
    geometry.setAttribute(
      'aRadius',
      new THREE.InstancedBufferAttribute(new Float32Array(count), 1)
    );
  }
  if (options.hasSharpness) {
    geometry.setAttribute(
      'aSharpness',
      new THREE.InstancedBufferAttribute(new Float32Array(count), 1)
    );
  }
  if (options.drawRange !== undefined) {
    geometry.setDrawRange(0, options.drawRange);
  }
  const points = new THREE.Mesh(geometry);
  points.userData = { nodeType: 'points' };
  if (options.name !== undefined) points.name = options.name;
  if (options.visible !== undefined) points.visible = options.visible;
  return points;
}

function makeGSplatMesh(
  splatCount: number,
  options: { name?: string; visible?: boolean } = {}
): THREE.Mesh {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = splatCount;
  // Set a position attribute so the picture is realistic (not strictly required
  // for the count, which reads instanceCount).
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
  const mesh = new THREE.Mesh(geometry);
  mesh.userData = { nodeType: 'gsplats' };
  if (options.name !== undefined) mesh.name = options.name;
  if (options.visible !== undefined) mesh.visible = options.visible;
  return mesh;
}

function makeContext(
  scene: THREE.Object3D,
  overrides: Partial<DebugStateContext> = {}
): DebugStateContext {
  const camera = new THREE.PerspectiveCamera(50);
  camera.position.set(1, 2, 3);
  return {
    scene,
    camera,
    currentFov: 47,
    isAnimating: false,
    initialized: true,
    dims: null,
    ...overrides,
  };
}

describe('computeDebugState', () => {
  it('returns zero counts on an empty scene', () => {
    const scene = new THREE.Scene();
    const state = computeDebugState(makeContext(scene));

    expect(state.totalPoints).toBe(0);
    expect(state.totalGSplats).toBe(0);
    expect(state.totalElements).toBe(0);
    expect(state.pointClouds).toEqual([]);
    expect(state.gsplatMeshes).toEqual([]);
  });

  describe('point cloud counting', () => {
    it('counts a single point cloud by instance count', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(100, { name: 'a' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(100);
      expect(state.pointClouds).toHaveLength(1);
      expect(state.pointClouds[0].name).toBe('a');
      expect(state.pointClouds[0].pointCount).toBe(100);
    });

    it('honours geometry.instanceCount when pooled buffers are over-allocated', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(1000, { instanceCount: 250 }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(250);
    });

    it('uses instanceCount even when drawRange is Infinity', () => {
      const scene = new THREE.Scene();
      const points = makePointCloud(500);
      // Three.js default drawRange is { start: 0, count: Infinity } — keep that.
      points.geometry.setDrawRange(0, Infinity);
      scene.add(points);

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(500);
    });

    it('does not mistake base-quad drawRange.count for point count', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(100, { drawRange: 6 }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(100);
    });

    it('reports has* flags from geometry attributes', () => {
      const scene = new THREE.Scene();
      scene.add(
        makePointCloud(10, {
          hasColors: true,
          hasRadii: true,
          hasSharpness: false,
        })
      );

      const state = computeDebugState(makeContext(scene));
      expect(state.pointClouds[0].hasColors).toBe(true);
      expect(state.pointClouds[0].hasRadii).toBe(true);
      expect(state.pointClouds[0].hasSharpness).toBe(false);
    });

    it('reports visible flag', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(10, { name: 'hidden', visible: false }));
      scene.add(makePointCloud(10, { name: 'shown', visible: true }));

      const state = computeDebugState(makeContext(scene));
      const hidden = state.pointClouds.find((p) => p.name === 'hidden');
      const shown = state.pointClouds.find((p) => p.name === 'shown');
      expect(hidden?.visible).toBe(false);
      expect(shown?.visible).toBe(true);
    });

    it('uses "unnamed" fallback when name is empty', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(10));

      const state = computeDebugState(makeContext(scene));
      expect(state.pointClouds[0].name).toBe('unnamed');
    });

    it('aggregates counts across multiple point clouds', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(100));
      scene.add(makePointCloud(50));
      scene.add(makePointCloud(25));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(175);
      expect(state.pointClouds).toHaveLength(3);
    });
  });

  describe('gsplat mesh counting', () => {
    it('counts gsplat instances by InstancedBufferGeometry.instanceCount', () => {
      const scene = new THREE.Scene();
      scene.add(makeGSplatMesh(500, { name: 'splats' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalGSplats).toBe(500);
      expect(state.gsplatMeshes).toHaveLength(1);
    });

    it('skips Mesh without nodeType=gsplats userData', () => {
      const scene = new THREE.Scene();
      const geometry = new THREE.InstancedBufferGeometry();
      geometry.instanceCount = 100;
      scene.add(new THREE.Mesh(geometry)); // no userData → not counted

      const state = computeDebugState(makeContext(scene));
      expect(state.totalGSplats).toBe(0);
      expect(state.gsplatMeshes).toHaveLength(0);
    });

    it('skips Mesh with nodeType=gsplats but a non-instanced geometry', () => {
      const scene = new THREE.Scene();
      const mesh = new THREE.Mesh(new THREE.BufferGeometry());
      mesh.userData = { nodeType: 'gsplats' };
      scene.add(mesh);

      const state = computeDebugState(makeContext(scene));
      expect(state.totalGSplats).toBe(0);
    });
  });

  describe('totalElements', () => {
    it('sums points + gsplats', () => {
      const scene = new THREE.Scene();
      scene.add(makePointCloud(100));
      scene.add(makeGSplatMesh(50));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(100);
      expect(state.totalGSplats).toBe(50);
      expect(state.totalElements).toBe(150);
    });
  });

  describe('camera + animation reporting', () => {
    it('reports camera.position.{x,y,z} and currentFov', () => {
      const scene = new THREE.Scene();
      const ctx = makeContext(scene, { currentFov: 35 });
      ctx.camera.position.set(10, 20, 30);

      const state = computeDebugState(ctx);
      expect(state.camera.position).toEqual({ x: 10, y: 20, z: 30 });
      expect(state.camera.fov).toBe(35);
      // Top-level mirror for compatibility.
      expect(state.cameraPosition).toEqual({ x: 10, y: 20, z: 30 });
      expect(state.cameraFov).toBe(35);
    });

    it('reports isAnimating + initialized flags', () => {
      const scene = new THREE.Scene();
      const state = computeDebugState(makeContext(scene, { isAnimating: true, initialized: true }));
      expect(state.isAnimating).toBe(true);
      expect(state.initialized).toBe(true);
    });
  });

  describe('dimensions reporting', () => {
    it('returns null when dims is null', () => {
      const state = computeDebugState(makeContext(new THREE.Scene(), { dims: null }));
      expect(state.dimensions).toBeNull();
    });

    it('extracts ndim / displayed / currentStep from SimpleDims', () => {
      const dims: SimpleDims = {
        ndim: 4,
        currentStep: [0, 0, 5, 10],
        displayed: [0, 1, 2],
      };
      const state = computeDebugState(makeContext(new THREE.Scene(), { dims }));
      expect(state.dimensions).toEqual({
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 5, 10],
      });
    });
  });

  describe('nested scene graph', () => {
    it('traverses through groups (scene.traverse() walks the entire subtree)', () => {
      const scene = new THREE.Scene();
      const group = new THREE.Group();
      group.add(makePointCloud(75));
      const subgroup = new THREE.Group();
      subgroup.add(makeGSplatMesh(25));
      group.add(subgroup);
      scene.add(group);

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(75);
      expect(state.totalGSplats).toBe(25);
    });
  });
});
