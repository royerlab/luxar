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
  computeDrawOrder,
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
  // only the `aSortedIndex`/`aSortedIndexB` ordering pair remains
  // per-instance).
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

/**
 * A mesh node: a plain INDEXED BufferGeometry, not an instanced one.
 *
 * That asymmetry is the point — every other helper here builds an
 * `InstancedBufferGeometry` and the count comes from `instanceCount`. A mesh has no
 * instances, so the drawn count comes from the DRAW RANGE, which is what the nD slice
 * compaction narrows.
 */
function makeMeshNode(
  triangles: number,
  vertices: number,
  options: {
    name?: string;
    visible?: boolean;
    drawTriangles?: number;
    defines?: Record<string, number>;
  } = {}
): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(triangles * 3), 1));
  if (options.drawTriangles !== undefined) {
    geometry.setDrawRange(0, options.drawTriangles * 3);
  }
  const material = new THREE.ShaderMaterial({
    vertexShader: 'void main() {}',
    fragmentShader: 'void main() {}',
    defines: options.defines ?? {},
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData = { nodeType: 'mesh' };
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

  describe('mesh node counting (four-geometry symmetry)', () => {
    it('reports a mesh node with its drawn triangles and its vertex count', () => {
      const scene = new THREE.Scene();
      scene.add(makeMeshNode(40, 25, { name: 'surface' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.meshNodes).toHaveLength(1);
      expect(state.meshNodes[0].name).toBe('surface');
      expect(state.meshNodes[0].triangleCount).toBe(40);
      expect(state.meshNodes[0].vertexCount).toBe(25);
      expect(state.totalTriangles).toBe(40);
    });

    it('reports the DRAW RANGE, not the whole index buffer', () => {
      // The assertion that matters for an nD mesh. The slice compaction rewrites the
      // index buffer in place and narrows `drawRange` (§5.4) — the vertex arrays and
      // the index LENGTH stay put — so reading `index.count` would report the full
      // surface no matter where the slice sits, which is precisely the number a debug
      // driver must not be lied to about.
      const scene = new THREE.Scene();
      scene.add(makeMeshNode(100, 60, { name: 'sliced', drawTriangles: 12 }));

      const state = computeDebugState(makeContext(scene));
      expect(state.meshNodes[0].triangleCount).toBe(12);
      expect(state.totalTriangles).toBe(12);
      // The vertex count is deliberately NOT narrowed: it is the pick-id domain and is
      // invariant across slices (there is no vertex compaction).
      expect(state.meshNodes[0].vertexCount).toBe(60);
    });

    it('falls back to the index length when drawRange is the default Infinity', () => {
      // `BufferGeometry.drawRange.count` starts at Infinity meaning "draw everything".
      // Dividing that by 3 would report Infinity triangles.
      const scene = new THREE.Scene();
      scene.add(makeMeshNode(7, 9, { name: 'whole' }));
      const state = computeDebugState(makeContext(scene));
      expect(Number.isFinite(state.meshNodes[0].triangleCount)).toBe(true);
      expect(state.meshNodes[0].triangleCount).toBe(7);
    });

    it('surfaces the two shader variants and the colormap flag', () => {
      // Neither variant is readable from the geometry or the node attrs: the
      // flat/smooth choice folds in the live displayDims and the cutout follows the
      // composed blending mode. This is what an E2E assertion on shading state reads.
      const scene = new THREE.Scene();
      scene.add(
        makeMeshNode(4, 6, {
          name: 'flat',
          defines: { LUXAR_MESH_FLAT_NORMAL: 1, LUXAR_MESH_ALPHA_CUTOUT: 1, USE_COLORMAP: 1 },
        })
      );
      const state = computeDebugState(makeContext(scene));
      expect(state.meshNodes[0].flatNormal).toBe(true);
      expect(state.meshNodes[0].alphaCutout).toBe(true);
      expect(state.meshNodes[0].hasColormap).toBe(true);
    });

    it('reports the smooth/non-cutout build as false rather than absent', () => {
      const scene = new THREE.Scene();
      scene.add(makeMeshNode(4, 6, { name: 'smooth' }));
      const state = computeDebugState(makeContext(scene));
      expect(state.meshNodes[0].flatNormal).toBe(false);
      expect(state.meshNodes[0].alphaCutout).toBe(false);
      expect(state.meshNodes[0].hasColormap).toBe(false);
    });

    it('carries the visible flag', () => {
      const scene = new THREE.Scene();
      scene.add(makeMeshNode(4, 6, { name: 'hidden', visible: false }));
      expect(computeDebugState(makeContext(scene)).meshNodes[0].visible).toBe(false);
    });

    it('totalElements is points + gsplats + lines + TRIANGLES', () => {
      // The four-way symmetry check, extending the three-way one above. A regression
      // that dropped triangles from the sum surfaces here even with every per-type
      // count correct.
      const scene = new THREE.Scene();
      scene.add(makePointCloud(100, { name: 'p' }));
      scene.add(makeGSplatMesh(200, { name: 'g' }));
      scene.add(makeLineMesh(300, { name: 'l' }));
      scene.add(makeMeshNode(400, 250, { name: 'm' }));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalTriangles).toBe(400);
      expect(state.totalElements).toBe(1000);
    });

    it('does NOT count an instanced geometry stamped nodeType=mesh', () => {
      // The guard that keeps the four arms disjoint. Every other type is selected by
      // `InstancedBufferGeometry`; mesh is selected by its ABSENCE, so a scene where
      // both matched would double-count. Not reachable from the loader — the assertion
      // exists so the two selectors stay mutually exclusive if either is edited.
      const scene = new THREE.Scene();
      const geometry = new THREE.InstancedBufferGeometry();
      geometry.instanceCount = 99;
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(30), 1));
      const impostor = new THREE.Mesh(geometry);
      impostor.userData = { nodeType: 'mesh' };
      scene.add(impostor);

      const state = computeDebugState(makeContext(scene));
      expect(state.meshNodes).toEqual([]);
      expect(state.totalTriangles).toBe(0);
    });

    it('skips a Mesh without nodeType=mesh userData', () => {
      const scene = new THREE.Scene();
      const geometry = new THREE.BufferGeometry();
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(30), 1));
      scene.add(new THREE.Mesh(geometry));

      const state = computeDebugState(makeContext(scene));
      expect(state.totalTriangles).toBe(0);
      expect(state.meshNodes).toEqual([]);
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

describe('computeDrawOrder', () => {
  /** A data mesh with a material carrying the queried draw-order state. */
  function makeDataMesh(
    nodeType: 'points' | 'gsplats' | 'lines',
    opts: {
      name: string;
      elements: number;
      transparent: boolean;
      depthWrite: boolean;
      renderOrder: number;
    }
  ): THREE.Mesh {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.instanceCount = opts.elements;
    const material = new THREE.MeshBasicMaterial();
    material.transparent = opts.transparent;
    material.depthWrite = opts.depthWrite;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData = { nodeType };
    mesh.name = opts.name;
    mesh.renderOrder = opts.renderOrder;
    return mesh;
  }

  it('returns an empty array for a scene with no data meshes', () => {
    expect(computeDrawOrder(new THREE.Scene())).toEqual([]);
  });

  it('reports a MESH node with its committed triangle count, not 0', () => {
    // Regression. Mesh reaches this walk (`DATA_NODE_TYPES` is `LOADER_TYPES`, which
    // includes it), but the element count fell through to a local
    // `visiblePointCount ?? visibleSplatCount ?? visibleSegmentCount ?? 0` chain — a
    // partial copy of the shared per-type reader with `visibleTriangleCount` missing.
    // So every mesh reported `elements: 0` while appearing in the report, which is the
    // worst shape for a diagnostic: present, plausible, and wrong.
    //
    // Mesh is also the only type here whose geometry is NOT instanced, so it is the
    // only one that takes the fallback path at all.
    const scene = new THREE.Scene();
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(90), 1));
    const material = new THREE.MeshBasicMaterial();
    material.transparent = false;
    material.depthWrite = true;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData = { nodeType: 'mesh', loader: {}, attrs: {}, visibleTriangleCount: 30 };
    mesh.name = '/surface';
    scene.add(mesh);

    const order = computeDrawOrder(scene);
    expect(order).toHaveLength(1);
    expect(order[0].path).toBe('/surface');
    expect(order[0].elements).toBe(30);
    // `opaque` is the mesh default, so it belongs in the depth-first bucket.
    expect(order[0].bucket).toBe('opaque');
    expect(order[0].depthWrite).toBe(true);
  });

  it('reports bucket / depthWrite / renderOrder / elements per data mesh', () => {
    const scene = new THREE.Scene();
    scene.add(
      makeDataMesh('gsplats', {
        name: '/cloud',
        elements: 15_000_000,
        transparent: true,
        depthWrite: false,
        renderOrder: 1,
      })
    );
    scene.add(
      makeDataMesh('points', {
        name: '/earth',
        elements: 4200,
        transparent: false,
        depthWrite: true,
        renderOrder: 0,
      })
    );

    const order = computeDrawOrder(scene);
    // Opaque bucket first, then renderOrder → the opaque backdrop precedes
    // the transparent cloud.
    expect(order).toEqual([
      { path: '/earth', bucket: 'opaque', depthWrite: true, renderOrder: 0, elements: 4200 },
      {
        path: '/cloud',
        bucket: 'transparent',
        depthWrite: false,
        renderOrder: 1,
        elements: 15_000_000,
      },
    ]);
  });

  it('reports an opaque mesh before transparent ones even when its renderOrder is higher', () => {
    // THREE renders its whole opaque list before the transparent list;
    // renderOrder only orders meshes WITHIN a list. An opaque mesh can carry
    // a stale positive renderOrder (assigned while it was in a sorted mode,
    // never reset on a live blending-mode switch) — the report must still
    // place it first, or the tool misdiagnoses the very compositing bug it
    // exists to surface.
    const scene = new THREE.Scene();
    scene.add(
      makeDataMesh('gsplats', {
        name: '/cloud',
        elements: 100,
        transparent: true,
        depthWrite: false,
        renderOrder: 0,
      })
    );
    scene.add(
      makeDataMesh('points', {
        name: '/earth',
        elements: 10,
        transparent: false,
        depthWrite: true,
        renderOrder: 5, // stale, from before a switch to opaque
      })
    );

    expect(computeDrawOrder(scene).map((e) => e.path)).toEqual(['/earth', '/cloud']);
  });

  it('omits hidden meshes (their renderOrder is stale, never reset)', () => {
    const scene = new THREE.Scene();
    scene.add(
      makeDataMesh('points', {
        name: '/shown',
        elements: 10,
        transparent: true,
        depthWrite: false,
        renderOrder: 0,
      })
    );
    const hidden = makeDataMesh('gsplats', {
      name: '/hidden',
      elements: 20,
      transparent: true,
      depthWrite: false,
      renderOrder: 9,
    });
    hidden.visible = false;
    scene.add(hidden);

    expect(computeDrawOrder(scene).map((e) => e.path)).toEqual(['/shown']);
  });

  it('prunes a visible mesh nested under a hidden group (subtree, not just self)', () => {
    // A hidden substitutive-LOD level is a hidden THREE.Group whose inner
    // meshes stay visible=true; the walk must prune the whole subtree.
    const scene = new THREE.Scene();
    const hiddenGroup = new THREE.Group();
    hiddenGroup.visible = false;
    hiddenGroup.add(
      makeDataMesh('gsplats', {
        name: '/hiddenLevel/mesh',
        elements: 20,
        transparent: true,
        depthWrite: false,
        renderOrder: 9,
      })
    );
    scene.add(hiddenGroup);

    expect(computeDrawOrder(scene)).toEqual([]);
  });

  it('ignores non-data objects and breaks renderOrder ties in scene-graph order', () => {
    const scene = new THREE.Scene();
    scene.add(new THREE.Group()); // no nodeType → skipped
    const a = makeDataMesh('lines', {
      name: '/a',
      elements: 10,
      transparent: true,
      depthWrite: false,
      renderOrder: 0,
    });
    const b = makeDataMesh('lines', {
      name: '/b',
      elements: 20,
      transparent: true,
      depthWrite: false,
      renderOrder: 0,
    });
    scene.add(a, b);

    const order = computeDrawOrder(scene);
    expect(order.map((e) => e.path)).toEqual(['/a', '/b']);
  });
});
