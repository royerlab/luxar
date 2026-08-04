/**
 * Unit tests for the draw-order provider.
 *
 * Uses real THREE objects under jsdom (only the WebGL renderer needs a GL
 * context). Verifies the provider reads the live blending bucket / depthWrite
 * / renderOrder off each data mesh, keyed by scene-graph path (mesh `name`).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { createDrawOrderProvider } from '../../../../../data/scene-loader/monitor/draw-order-provider';

/** A data mesh with a material carrying the queried draw-order state. */
function makeDataMesh(
  nodeType: 'points' | 'gsplats' | 'lines',
  opts: { name: string; transparent: boolean; depthWrite: boolean; renderOrder: number }
): THREE.Mesh {
  const material = new THREE.MeshBasicMaterial();
  material.transparent = opts.transparent;
  material.depthWrite = opts.depthWrite;
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
  mesh.userData = { nodeType };
  mesh.name = opts.name;
  mesh.renderOrder = opts.renderOrder;
  return mesh;
}

describe('createDrawOrderProvider', () => {
  it('returns an empty map for a null root group', () => {
    expect(createDrawOrderProvider(null).getDrawOrderStates().size).toBe(0);
  });

  it('reads bucket / depthWrite / renderOrder per data mesh, keyed by path', () => {
    const root = new THREE.Group();
    root.add(
      makeDataMesh('gsplats', {
        name: '/cloud',
        transparent: true,
        depthWrite: false,
        renderOrder: 1,
      })
    );
    root.add(
      makeDataMesh('points', {
        name: '/earth',
        transparent: false,
        depthWrite: true,
        renderOrder: 0,
      })
    );
    // A non-data object contributes nothing.
    root.add(new THREE.Group());

    const states = createDrawOrderProvider(root).getDrawOrderStates();
    expect(states.size).toBe(2);
    expect(states.get('/earth')).toEqual({
      bucket: 'opaque',
      depthWrite: true,
      renderOrder: 0,
    });
    expect(states.get('/cloud')).toEqual({
      bucket: 'transparent',
      depthWrite: false,
      renderOrder: 1,
    });
  });

  it('omits hidden meshes (renderOrder is stale on non-visible subtrees)', () => {
    const root = new THREE.Group();
    root.add(
      makeDataMesh('points', {
        name: '/shown',
        transparent: true,
        depthWrite: false,
        renderOrder: 0,
      })
    );
    // A toggled-off layer keeps its last renderOrder but is not drawn — it
    // must not be reported.
    const hidden = makeDataMesh('gsplats', {
      name: '/hidden',
      transparent: true,
      depthWrite: false,
      renderOrder: 9,
    });
    hidden.visible = false;
    root.add(hidden);

    const states = createDrawOrderProvider(root).getDrawOrderStates();
    expect(states.has('/shown')).toBe(true);
    expect(states.has('/hidden')).toBe(false);
  });

  it('prunes a visible mesh nested under a hidden group (subtree, not just self)', () => {
    // The real scenario: the LOD-group registry hides a THREE.Group while its
    // inner meshes keep visible=true. Pruning must stop at the hidden group —
    // a per-object check that still descended would leak the inner mesh.
    const root = new THREE.Group();
    const hiddenGroup = new THREE.Group();
    hiddenGroup.visible = false;
    hiddenGroup.add(
      makeDataMesh('gsplats', {
        name: '/hiddenLevel/mesh',
        transparent: true,
        depthWrite: false,
        renderOrder: 9,
      })
    );
    root.add(hiddenGroup);

    expect(createDrawOrderProvider(root).getDrawOrderStates().has('/hiddenLevel/mesh')).toBe(false);
  });

  it('re-reads renderOrder live on each call (camera-dependent)', () => {
    const root = new THREE.Group();
    const mesh = makeDataMesh('lines', {
      name: '/line',
      transparent: true,
      depthWrite: false,
      renderOrder: 3,
    });
    root.add(mesh);
    const provider = createDrawOrderProvider(root);
    expect(provider.getDrawOrderStates().get('/line')?.renderOrder).toBe(3);

    // The depth-sort coordinator reassigns renderOrder per frame; the next
    // poll must reflect it.
    mesh.renderOrder = 7;
    expect(provider.getDrawOrderStates().get('/line')?.renderOrder).toBe(7);
  });
});
