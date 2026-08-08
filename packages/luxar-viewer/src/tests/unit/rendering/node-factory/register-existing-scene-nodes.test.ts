/**
 * `NodeFactory.registerExistingSceneNodes` — the retro-registration pass.
 *
 * This is the path production actually takes. `initPicking` traverses the FINISHED
 * scene to decide whether any node declares labels, and only then constructs the
 * `PickingSystem` and calls `setPickingSystem` — so on a first load
 * `this.pickingSystem` is still null while nodes are being created, and every pick
 * registration comes from here instead.
 *
 * Which makes an omission here uniquely nasty: a geometry type wired into its
 * `createEmptyXNode` but missing from this pass is unpickable in every real scene and
 * appears to work only on a SECOND dataset load. It had no direct test at all before
 * this file — the only existing coverage asserted that a mocked
 * `registerExistingSceneNodes` was called, which cannot see what it does.
 *
 * @module tests/unit/rendering/node-factory/register-existing-scene-nodes
 */
import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../../rendering/node-factory';
import {
  materialManager,
  __resetMaterialManagerForTests,
} from '../../../../rendering/material-manager';
import { PointPickingMaterial } from '../../../../rendering/picking/point/material';
import { LinePickingMaterial } from '../../../../rendering/picking/line/material';
import { GSplatPickingMaterial } from '../../../../rendering/picking/gsplat/material';
import { MeshPickingMaterial } from '../../../../rendering/picking/mesh/material';
import { GEOMETRY_TYPES } from '../../../../types/format-contract';
import { LINE_JOIN_UNIFORM, type LineJoinStyle } from '../../../../types/line-join';
import type { PickingSystem } from '../../../../rendering/picking/picking-system';

/** Minimal PickingSystem stand-in: the three members the pass touches. */
function stubPickingSystem() {
  let next = 1;
  const registered: Array<{ main: THREE.Object3D; pick: THREE.Mesh; pickId: number }> = [];
  const stub = {
    allocatePickId: () => next++,
    registerNode(main: THREE.Object3D, pick: THREE.Object3D, pickId: number) {
      // The real implementation stamps this, and `syncMeshPickMaterialToVisual`
      // depends on it — so the stub must too, or the sync would silently no-op and
      // this test would pass against a broken ordering.
      main.userData.pickNode = pick;
      registered.push({ main, pick: pick as THREE.Mesh, pickId });
    },
    get registeredNodeCount() {
      return registered.length;
    },
  };
  return { stub: stub as unknown as PickingSystem, registered };
}

/**
 * A node shaped the way each `createEmptyXNode` leaves it.
 *
 * The return type narrows `material` to a single `Material` — `THREE.Mesh`'s own
 * declaration allows an array, which the production code has to handle but which this
 * constructor never produces.
 */
function makeNode(nodeType: string, name: string): THREE.Mesh & { material: THREE.Material } {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
  mesh.name = name;
  mesh.userData = { nodeType, attrs: {} };
  return mesh as THREE.Mesh & { material: THREE.Material };
}

/** A real line VISUAL material at neutral appearance, varying only the join style. */
function makeLineMaterial(join?: LineJoinStyle) {
  return materialManager.getLineMaterial({
    blendingMode: 'additive',
    opacity: 1.0,
    gamma: 1.0,
    intensity: 1.0,
    offset: 0.0,
    join,
  });
}

describe('registerExistingSceneNodes', () => {
  let factory: NodeFactory;

  beforeEach(() => {
    __resetMaterialManagerForTests();
    factory = new NodeFactory();
  });

  it('registers a node of EVERY geometry type in the contract', () => {
    // Driven off `GEOMETRY_TYPES` rather than a hand-written list of four, so a fifth
    // geometry type joins this assertion automatically instead of quietly shipping
    // unpickable. This is the assertion that mesh's absence would have failed.
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);

    const root = new THREE.Group();
    for (const type of GEOMETRY_TYPES) root.add(makeNode(type, `/${type}`));

    factory.registerExistingSceneNodes(root);

    expect(registered).toHaveLength(GEOMETRY_TYPES.length);
    expect(registered.map((r) => r.main.name).sort()).toEqual(
      GEOMETRY_TYPES.map((t) => `/${t}`).sort()
    );
  });

  it('gives each type its OWN pick material class', () => {
    // Not just "something was registered": a table keyed by type could still map two
    // types to the same builder, which would silently give one of them the other's
    // element-id semantics.
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);

    const root = new THREE.Group();
    for (const type of GEOMETRY_TYPES) root.add(makeNode(type, `/${type}`));
    factory.registerExistingSceneNodes(root);

    const byName = new Map(registered.map((r) => [r.main.name, r.pick.material]));
    expect(byName.get('/points')).toBeInstanceOf(PointPickingMaterial);
    expect(byName.get('/lines')).toBeInstanceOf(LinePickingMaterial);
    expect(byName.get('/gsplats')).toBeInstanceOf(GSplatPickingMaterial);
    expect(byName.get('/mesh')).toBeInstanceOf(MeshPickingMaterial);
  });

  it('shares the visual geometry with the pick node and copies its world matrix', () => {
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('mesh', '/surface');
    node.position.set(3, 0, 0);
    node.updateMatrixWorld(true);
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const entry = registered[0];
    expect(entry.pick.geometry).toBe(node.geometry);
    expect(entry.pick.matrixWorld.elements).toEqual(node.matrixWorld.elements);
  });

  it('seeds the mesh pick material from the visual material, before the first pick render', () => {
    // The picking system re-pushes `side` and the mode on every pick render, so this
    // governs only the window before the first one — which is exactly the window
    // containing the first hover. Without it, a mesh whose epoch forced DoubleSide
    // would pick against a FrontSide pass and drop half its surface.
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('mesh', '/surface');
    node.material.side = THREE.DoubleSide;
    node.userData.attrs = { blending_mode: 'additive' };
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const pick = registered[0].pick.material as MeshPickingMaterial;
    expect(pick.side).toBe(THREE.DoubleSide);
    // 'additive' is commutative: no cutout, brightness-as-depth.
    expect(pick.uniforms.uAlphaCutout.value).toBe(0);
    expect(pick.uniforms.uSurfaceDepth.value).toBe(0);
  });

  it('prefers the visual material LIVE state over the node attrs', () => {
    // The context-restore case. `rebuildAfterContextRestore` re-runs this pass with
    // FRESH pick materials (the old ones were compiled against the dead context) while
    // the visual material survives carrying whatever the user dragged in the layers
    // panel. Seeding from `userData.attrs` there would silently revert the pick
    // coverage to the authored values — a mesh you had faded would go back to being
    // pickable at its original threshold, self-healing only on the next panel edit.
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('mesh', '/surface');
    // Authored values...
    node.userData.attrs = { opacity: 1.0, alpha_cutoff: 0.5, blending_mode: 'opaque' };
    // ...and a live material that disagrees with all three, as after a panel drag plus
    // a volumetric-by-inheritance resolution.
    (node.material as unknown as { uniforms: Record<string, { value: number }> }).uniforms = {
      uOpacity: { value: 0.42 },
      uAlphaCutoff: { value: 0.9 },
    };
    node.material.userData.blendingMode = 'additive';
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const pick = registered[0].pick.material as MeshPickingMaterial;
    expect(pick.uniforms.uOpacity.value).toBeCloseTo(0.42);
    expect(pick.uniforms.uAlphaCutoff.value).toBeCloseTo(0.9);
    // The RESOLVED mode, not the authored one.
    expect(pick.uniforms.uAlphaCutout.value).toBe(0);
  });

  it('gives the lines pick material the join its VISUAL material resolved to', () => {
    // `join` is a COMPOSITING attr, so on a partitioned lines node it is authored on
    // the wrapper and never appears in a part's own `userData.attrs`. The visual
    // material is where the composed value survives — seeding the pick material from
    // the attrs instead would leave the outer wedge of every mitred corner pickable
    // on a `join="none"` scene, and only on a FIRST load (`createLinesNode` handles
    // the second, and passes the style through correctly).
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('lines', '/tracks/part_0');
    node.material = makeLineMaterial('none');
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const pick = registered[0].pick.material as LinePickingMaterial;
    expect(pick.uniforms.uLineJoin.value).toBe(LINE_JOIN_UNIFORM.none);
  });

  it('falls back to the default join when the visual material declares none', () => {
    // The other half: without it, "reads the visual material" could be satisfied by
    // hard-coding `none`. An unauthored node must still get the default.
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('lines', '/tracks');
    node.material = makeLineMaterial();
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const pick = registered[0].pick.material as LinePickingMaterial;
    expect(pick.uniforms.uLineJoin.value).toBe(LINE_JOIN_UNIFORM.miter);
  });

  it('reads the TSL backend unresolved userData.lineJoin marker too', () => {
    // The backends store the style differently on purpose: GLSL keeps a `uLineJoin`
    // uniform, TSL bakes the graph variant and stamps the UNRESOLVED style on
    // `userData.lineJoin`. Only handling the uniform would silently give every
    // WebGPU session the default join in its pick pass.
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('lines', '/tracks');
    node.material.userData.lineJoin = 'none';
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const pick = registered[0].pick.material as LinePickingMaterial;
    expect(pick.uniforms.uLineJoin.value).toBe(LINE_JOIN_UNIFORM.none);
  });

  it('carries the mesh coverage inputs from the node attrs', () => {
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);
    const root = new THREE.Group();
    const node = makeNode('mesh', '/surface');
    node.userData.attrs = { opacity: 0.75, alpha_cutoff: 0.2 };
    root.add(node);

    factory.registerExistingSceneNodes(root);

    const pick = registered[0].pick.material as MeshPickingMaterial;
    expect(pick.uniforms.uOpacity.value).toBeCloseTo(0.75);
    expect(pick.uniforms.uAlphaCutoff.value).toBeCloseTo(0.2);
  });

  it('skips already-registered nodes, non-geometry nodes, and non-Mesh objects', () => {
    const { stub, registered } = stubPickingSystem();
    factory.setPickingSystem(stub);

    const root = new THREE.Group();
    const already = makeNode('mesh', '/already');
    already.userData.pickId = 99;
    root.add(already);
    // A group: has no nodeType, so it is not a data node.
    const group = new THREE.Group();
    group.userData = { nodeType: 'group' };
    root.add(group);
    // A geometry type that draws as something other than a Mesh would over-read the
    // `obj.geometry` the pick node shares, so the instanceof gate is load-bearing.
    const notAMesh = new THREE.Object3D();
    notAMesh.userData = { nodeType: 'points', attrs: {} };
    root.add(notAMesh);

    factory.registerExistingSceneNodes(root);

    expect(registered).toHaveLength(0);
    expect(already.userData.pickId).toBe(99);
  });

  it('is a no-op with no picking system (picking disabled)', () => {
    const root = new THREE.Group();
    const node = makeNode('mesh', '/surface');
    root.add(node);
    factory.registerExistingSceneNodes(root);
    expect(node.userData.pickId).toBeUndefined();
  });
});
