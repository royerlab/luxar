/**
 * Unit tests for the scene-graph disposal helpers.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  disposeObjectTree,
  clearLoadedSceneContent,
  disposeSceneGraphResources,
} from '../../../../../scene/scene-manager/render-pipeline/scene-disposal';
import {
  releaseDepthSortNode,
  releaseAllDepthSortNodes,
} from '../../../../../rendering/depth-sort-coordinator';

// The disposal helpers call into the depth-sort coordinator (per-mesh
// release + the dataset-switch wholesale sweep); mock it so these tests
// assert the WIRING without spinning up worker machinery.
vi.mock('../../../../../rendering/depth-sort-coordinator', () => ({
  releaseDepthSortNode: vi.fn(),
  releaseAllDepthSortNodes: vi.fn(),
}));

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
  points: THREE.Mesh;
  geometryDispose: ReturnType<typeof vi.fn>;
  materialDispose: ReturnType<typeof vi.fn>;
} {
  // The disposal helper treats Mesh and InstancedMesh uniformly.
  const geometry = new THREE.InstancedBufferGeometry();
  const material = new THREE.MeshBasicMaterial();
  const geometryDispose = vi.spyOn(geometry, 'dispose');
  const materialDispose = vi.spyOn(material, 'dispose');
  const points = new THREE.Mesh(geometry, material);
  points.userData.nodeType = 'points';
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

  it('disposes points-mesh objects (geometry + material)', () => {
    const { points, geometryDispose, materialDispose } = makePoints();
    disposeObjectTree(points);
    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
  });

  it('does not throw on an empty plain Group (no geometry to dispose)', () => {
    const group = new THREE.Group();
    expect(() => disposeObjectTree(group)).not.toThrow();
  });

  // W1: a Group has no geometry of its own, but its CHILDREN must still be
  // disposed and detached. "doesn't throw" alone would survive a mutant that
  // skipped the children walk for non-renderable roots.
  it('disposes and detaches a renderable child held by a plain Group', () => {
    const group = new THREE.Group();
    const { mesh, geometryDispose, materialDispose } = makeMesh();
    group.add(mesh);

    disposeObjectTree(group);

    expect(geometryDispose).toHaveBeenCalledTimes(1);
    expect(materialDispose).toHaveBeenCalledTimes(1);
    expect(group.children.length).toBe(0);
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

  // G5: pin the traversal ORDER. The source disposes an object's OWN
  // geometry/material first, then recurses into its children — i.e. pre-order
  // depth-first (ancestor before descendant). A BFS or post-order mutant would
  // reorder these.
  it('disposes an ancestor before its descendant (pre-order depth-first)', () => {
    const order: string[] = [];
    const tag = (m: THREE.Mesh, name: string) => {
      vi.spyOn(m.geometry, 'dispose').mockImplementation(() => {
        order.push(name);
      });
      return m;
    };
    // root(Group) → a(Mesh) → b(Mesh grandchild)
    const a = tag(makeMesh().mesh, 'a');
    const b = tag(makeMesh().mesh, 'b');
    a.add(b);
    const root = new THREE.Group();
    root.add(a);

    disposeObjectTree(root);

    // 'a' (the ancestor mesh) disposes before its child 'b'.
    expect(order).toEqual(['a', 'b']);
  });
});

describe('clearLoadedSceneContent', () => {
  it('drops ALL depth-sort registrations wholesale (dataset-switch teardown)', () => {
    vi.mocked(releaseAllDepthSortNodes).mockClear();
    const scene = new THREE.Scene();
    scene.add(makeMesh().mesh);
    clearLoadedSceneContent(scene);
    expect(releaseAllDepthSortNodes).toHaveBeenCalledTimes(1);
  });

  it('releases the per-mesh depth-sort registration of every mesh (gsplats, points, plain)', () => {
    // The release is deliberately UNCONDITIONAL: only sortable nodes
    // (gsplats, points, and lines) ever register, and the call
    // is a cheap map-delete no-op for everything else — a nodeType gate
    // here would silently strand registrations when a new geometry type
    // joins the sorted set.
    vi.mocked(releaseDepthSortNode).mockClear();
    const scene = new THREE.Scene();
    const { mesh: gsplats } = makeMesh();
    gsplats.userData.nodeType = 'gsplats';
    const { mesh: points } = makeMesh();
    points.userData.nodeType = 'points';
    const { mesh: plain } = makeMesh();
    scene.add(gsplats);
    scene.add(points);
    scene.add(plain);
    clearLoadedSceneContent(scene);
    expect(releaseDepthSortNode).toHaveBeenCalledWith(gsplats);
    expect(releaseDepthSortNode).toHaveBeenCalledWith(points);
    expect(releaseDepthSortNode).toHaveBeenCalledWith(plain);
  });

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
  it('drops ALL depth-sort registrations (self-sufficient final shutdown)', () => {
    // Defense-in-depth: the dispose pipeline calls disposeDepthSort()
    // separately, but an embedder driving only this shutdown path must
    // not leave the module-scoped coordinator map pinning old meshes.
    vi.mocked(releaseAllDepthSortNodes).mockClear();
    const scene = new THREE.Scene();
    scene.add(makeMesh().mesh);
    disposeSceneGraphResources(scene);
    expect(releaseAllDepthSortNodes).toHaveBeenCalledTimes(1);
  });

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
