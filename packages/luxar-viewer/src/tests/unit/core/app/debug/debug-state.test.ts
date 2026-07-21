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
  // Point clouds are THREE.Mesh with instanced quad geometry whose
  // per-point data lives in the point texture (fixed 3-texel layout;
  // only `aSortedIndex` remains a per-instance attribute).
  // `computeDebugState` selects on `userData.nodeType === 'points'`,
  // counts via `instanceCount`, and reports field presence from the
  // texel writers' `geometry.userData.has*` stamps — attribute probing
  // is impossible under the fixed texel layout, and the zarr node attrs
  // carry no has_colors/has_radii/has_sharpness.
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = options.instanceCount ?? count;
  geometry.setAttribute(
    'aSortedIndex',
    new THREE.InstancedBufferAttribute(new Uint32Array(count), 1)
  );
  geometry.userData = {
    hasColors: !!options.hasColors,
    hasRadii: !!options.hasRadii,
    hasSharpness: !!options.hasSharpness,
  };
  if (options.drawRange !== undefined) {
    geometry.setDrawRange(0, options.drawRange);
  }
  const points = new THREE.Mesh(geometry);
  points.userData = {
    nodeType: 'points',
  };
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

function makeLineMesh(
  segmentCount: number,
  options: { name?: string; visible?: boolean; hasColormap?: boolean } = {}
): THREE.Mesh {
  // Lines render as THREE.Mesh + InstancedBufferGeometry (one instance
  // per segment), matching the Points/GSplats symmetry contract.
  // `computeDebugState` selects on `userData.nodeType === 'lines'`.
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = segmentCount;
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
  const material = new THREE.ShaderMaterial({
    vertexShader: 'void main() {}',
    fragmentShader: 'void main() {}',
    defines: options.hasColormap ? { USE_COLORMAP: 1 } : {},
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData = { nodeType: 'lines' };
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

    it('reports has* flags from the node metadata (userData.attrs)', () => {
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

  describe('line mesh counting (core.md G17 three-geometry symmetry)', () => {
    // Lines must be counted the same way Points and GSplats are
    // (per-instance from InstancedBufferGeometry.instanceCount). A
    // regression that dropped the lines branch would leave totalLines
    // at 0 even when Lines exist in the scene.
    it('counts a single line mesh by instance count (segment count)', () => {
      const scene = new THREE.Scene();
      scene.add(makeLineMesh(50, { name: 'edges' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalLines).toBe(50);
      expect(state.lineMeshes).toHaveLength(1);
      expect(state.lineMeshes[0].name).toBe('edges');
      expect(state.lineMeshes[0].segmentCount).toBe(50);
    });

    it('reports hasColormap=true when material.defines.USE_COLORMAP is set', () => {
      const scene = new THREE.Scene();
      scene.add(makeLineMesh(10, { name: 'cm', hasColormap: true }));

      const state = computeDebugState(makeContext(scene));
      expect(state.lineMeshes[0].hasColormap).toBe(true);
    });

    it('reports hasColormap=false when material.defines.USE_COLORMAP is absent', () => {
      const scene = new THREE.Scene();
      scene.add(makeLineMesh(10, { name: 'plain', hasColormap: false }));

      const state = computeDebugState(makeContext(scene));
      expect(state.lineMeshes[0].hasColormap).toBe(false);
    });

    it('sums multiple line meshes into totalLines', () => {
      const scene = new THREE.Scene();
      scene.add(makeLineMesh(30, { name: 'a' }));
      scene.add(makeLineMesh(70, { name: 'b' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalLines).toBe(100);
      expect(state.lineMeshes).toHaveLength(2);
    });

    it('reports visible=false for hidden line meshes', () => {
      const scene = new THREE.Scene();
      scene.add(makeLineMesh(10, { name: 'hidden', visible: false }));

      const state = computeDebugState(makeContext(scene));
      // Visibility is reported but still counted (consistent with
      // points/gsplats — totalLines is the buffer count, not the
      // rendered count).
      expect(state.lineMeshes[0].visible).toBe(false);
      expect(state.totalLines).toBe(10);
    });

    it('totalElements is points + gsplats + lines', () => {
      // Symmetry check: the aggregate must be the sum of all three
      // geometry types. A regression that dropped lines from the sum
      // would surface here even if the per-geometry counts stayed
      // correct.
      const scene = new THREE.Scene();
      scene.add(makePointCloud(100, { name: 'p' }));
      scene.add(makeGSplatMesh(200, { name: 'g' }));
      scene.add(makeLineMesh(300, { name: 'l' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalPoints).toBe(100);
      expect(state.totalGSplats).toBe(200);
      expect(state.totalLines).toBe(300);
      expect(state.totalElements).toBe(600);
    });

    it('skips Mesh without nodeType=lines userData', () => {
      const scene = new THREE.Scene();
      const geometry = new THREE.InstancedBufferGeometry();
      geometry.instanceCount = 99;
      const mesh = new THREE.Mesh(geometry);
      // userData.nodeType not set → not counted.
      scene.add(mesh);

      const state = computeDebugState(makeContext(scene));
      expect(state.totalLines).toBe(0);
      expect(state.lineMeshes).toEqual([]);
    });
  });

  describe('LOD / partition group reporting', () => {
    it('reports a kind=lod group with its active level', () => {
      const scene = new THREE.Scene();
      const lod = new THREE.Group();
      lod.name = '/lod';
      lod.userData.kind = 'lod';
      const l0 = new THREE.Group();
      const l1 = new THREE.Group();
      const l2 = new THREE.Group();
      l0.visible = false;
      l1.visible = true; // active level = index 1
      l2.visible = false;
      lod.add(l0, l1, l2);
      scene.add(lod);

      const state = computeDebugState(makeContext(scene));
      expect(state.lodGroups).toEqual([{ name: '/lod', levelCount: 3, activeLevel: 1 }]);
      expect(state.partitions).toEqual([]);
    });

    it('reports activeLevel -1 when no level is visible', () => {
      const scene = new THREE.Scene();
      const lod = new THREE.Group();
      lod.userData.kind = 'lod';
      const a = new THREE.Group();
      a.visible = false;
      lod.add(a);
      scene.add(lod);

      expect(computeDebugState(makeContext(scene)).lodGroups[0].activeLevel).toBe(-1);
    });

    it('reports a kind=partition group with part / visible-part counts', () => {
      const scene = new THREE.Scene();
      const part = new THREE.Group();
      part.name = '/parted';
      part.userData.kind = 'partition';
      const p0 = new THREE.Group();
      const p1 = new THREE.Group();
      const p2 = new THREE.Group();
      p2.visible = false; // culled
      part.add(p0, p1, p2);
      scene.add(part);

      const state = computeDebugState(makeContext(scene));
      expect(state.partitions).toEqual([{ name: '/parted', partCount: 3, visibleParts: 2 }]);
      expect(state.lodGroups).toEqual([]);
    });

    it('reports empty arrays when no specialized groups exist', () => {
      const state = computeDebugState(makeContext(new THREE.Scene()));
      expect(state.lodGroups).toEqual([]);
      expect(state.partitions).toEqual([]);
    });
  });
});
