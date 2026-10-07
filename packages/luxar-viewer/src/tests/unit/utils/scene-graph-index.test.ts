/**
 * `SceneGraphIndex` / `findObjectByName` (B9a): the index must return EXACTLY
 * what `Object3D.getObjectByName` returns, for every name and lookup root, after
 * any sequence of the graph edits the viewer performs — three-API add / remove /
 * move / clear, renames (including into and out of a duplicate), the eager-child
 * loader's event-less slot flattening, disposal, and a dataset switch onto a
 * fresh root. Property-style: seeded random operation streams, with the walk as
 * the oracle after every step.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  attachSceneGraphIndex,
  findObjectByName,
  sceneGraphIndexOf,
} from '../../../utils/scene-graph-index';
import { disposeObjectTree } from '../../../scene/scene-manager/render-pipeline/scene-disposal';

/** Deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const NAMES = ['', '/a', '/b', '/c', '/a/x', '/a/y', '/b/z', '/dup', '/p0', '/p1', '/p2'];

function nodes(root: THREE.Object3D): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  root.traverse((n) => out.push(n));
  return out;
}

function makeNode(rand: () => number, counter: { n: number }): THREE.Object3D {
  const node = rand() < 0.5 ? new THREE.Group() : new THREE.Mesh();
  // Mostly unique paths (the loader's reality), sometimes a pooled name so
  // duplicates and renames-into-a-duplicate occur.
  node.name = rand() < 0.6 ? `/u${counter.n++}` : NAMES[Math.floor(rand() * NAMES.length)];
  return node;
}

function pick<T>(rand: () => number, items: readonly T[]): T {
  return items[Math.floor(rand() * items.length)];
}

/** The eager-child loader's `replaceSlot`: splice children up, no events. */
function flattenSilently(slot: THREE.Object3D): void {
  const parent = slot.parent;
  if (!parent) return;
  const at = parent.children.indexOf(slot);
  const moved = slot.children.splice(0);
  for (const child of moved) child.parent = parent;
  slot.parent = null;
  parent.children.splice(at, 1, ...moved);
}

type Op = (root: THREE.Object3D, rand: () => number, counter: { n: number }) => void;

const OPS: Op[] = [
  // add a leaf
  (root, rand, c) => pick(rand, nodes(root)).add(makeNode(rand, c)),
  // add a pre-built subtree
  (root, rand, c) => {
    const top = makeNode(rand, c);
    for (let i = 0; i < 3; i++) top.add(makeNode(rand, c));
    top.children[0].add(makeNode(rand, c));
    pick(rand, nodes(root)).add(top);
  },
  // remove
  (root, rand) => {
    const n = pick(rand, nodes(root));
    if (n !== root) n.removeFromParent();
  },
  // move (reparent, not into its own subtree)
  (root, rand) => {
    const all = nodes(root);
    const n = pick(rand, all);
    const target = pick(rand, all);
    if (n === root) return;
    let t: THREE.Object3D | null = target;
    while (t) {
      if (t === n) return;
      t = t.parent;
    }
    target.add(n);
  },
  // rename (to a pooled name, possibly a duplicate, or to '')
  (root, rand) => {
    pick(rand, nodes(root)).name = pick(rand, NAMES);
  },
  // rename to a fresh unique name
  (root, rand, c) => {
    pick(rand, nodes(root)).name = `/r${c.n++}`;
  },
  // clear a node's children
  (root, rand) => {
    if (rand() < 0.3) pick(rand, nodes(root)).clear();
  },
  // event-less slot flattening
  (root, rand) => {
    const n = pick(rand, nodes(root));
    if (n !== root) flattenSilently(n);
  },
  // dispose a subtree the way the teardown does
  (root, rand) => {
    const n = pick(rand, nodes(root));
    if (n === root) return;
    disposeObjectTree(n);
    n.removeFromParent();
  },
];

function expectAgrees(root: THREE.Object3D, rand: () => number, extra: string[]): void {
  const all = nodes(root);
  const names = new Set([...NAMES, ...extra, ...all.map((n) => n.name)]);
  for (const name of names) {
    expect(findObjectByName(root, name)).toBe(root.getObjectByName(name));
  }
  // A lookup rooted at an inner node of the indexed tree.
  const scope = pick(rand, all);
  for (const name of names) {
    expect(findObjectByName(scope, name)).toBe(scope.getObjectByName(name));
  }
}

describe('SceneGraphIndex — agrees with getObjectByName', () => {
  it.each([1, 2, 3, 4, 5, 6, 7, 8])(
    'after a random edit stream (seed %i), including dataset switches',
    (seed) => {
      const rand = rng(seed);
      const counter = { n: 0 };
      let root: THREE.Object3D = new THREE.Group();
      root.name = 'LuxarScene';
      attachSceneGraphIndex(root);
      const seen: string[] = [];
      for (let step = 0; step < 400; step++) {
        if (step % 150 === 149) {
          // Dataset switch: tear the old tree down, index a fresh root.
          disposeObjectTree(root);
          sceneGraphIndexOf(root)?.detach();
          root = new THREE.Group();
          root.name = 'LuxarScene';
          attachSceneGraphIndex(root);
        }
        pick(rand, OPS)(root, rand, counter);
        for (const n of nodes(root)) if (seen.length < 500) seen.push(n.name);
        expectAgrees(root, rand, seen.slice(-20));
      }
    }
  );

  it('answers unique names without walking, and walks for duplicates (first match)', () => {
    const root = new THREE.Group();
    const a = new THREE.Group();
    a.name = '/a';
    const dupFirst = new THREE.Mesh();
    dupFirst.name = '/dup';
    const dupSecond = new THREE.Mesh();
    dupSecond.name = '/dup';
    a.add(dupFirst);
    root.add(a, dupSecond);
    attachSceneGraphIndex(root);

    const walk = THREE.Object3D.prototype.getObjectByName;
    let walks = 0;
    THREE.Object3D.prototype.getObjectByName = function (this: THREE.Object3D, name: string) {
      walks++;
      return walk.call(this, name);
    };
    try {
      expect(findObjectByName(root, '/a')).toBe(a);
      expect(walks).toBe(0);
      // Depth-first pre-order: the one under `/a` comes first.
      expect(findObjectByName(root, '/dup')).toBe(dupFirst);
      expect(walks).toBe(1);
      // A rename that resolves the ambiguity makes the name O(1) again.
      dupFirst.name = '/renamed';
      expect(findObjectByName(root, '/dup')).toBe(dupSecond);
      expect(findObjectByName(root, '/renamed')).toBe(dupFirst);
      expect(walks).toBe(1);
    } finally {
      THREE.Object3D.prototype.getObjectByName = walk;
    }
  });

  it('walks once for a repeated miss, until the graph could make it a hit', () => {
    // A path not built yet (a lazy LOD level, a deferred partition part) is
    // looked up on every pass; each miss used to be a full subtree walk.
    const root = new THREE.Group();
    const a = new THREE.Group();
    a.name = '/a';
    const outside = new THREE.Mesh();
    outside.name = '/later';
    root.add(a);
    attachSceneGraphIndex(root);

    const walk = THREE.Object3D.prototype.getObjectByName;
    let walks = 0;
    THREE.Object3D.prototype.getObjectByName = function (this: THREE.Object3D, name: string) {
      walks++;
      return walk.call(this, name);
    };
    try {
      expect(findObjectByName(root, '/later')).toBeUndefined();
      expect(findObjectByName(root, '/later')).toBeUndefined();
      expect(findObjectByName(root, '/later')).toBeUndefined();
      expect(walks).toBe(1);

      // An add can turn the miss into a hit: answered at once, no stale miss.
      root.add(outside);
      expect(findObjectByName(root, '/later')).toBe(outside);
      outside.removeFromParent();
      expect(findObjectByName(root, '/later')).toBeUndefined();

      // A rename can too.
      const before = walks;
      expect(findObjectByName(a, '/renamed')).toBeUndefined();
      expect(findObjectByName(a, '/renamed')).toBeUndefined();
      expect(walks).toBe(before + 1);
      a.name = '/renamed';
      expect(findObjectByName(a, '/renamed')).toBe(a);

      // And so can moving an existing member INTO the looked-up scope.
      const inner = new THREE.Group();
      inner.name = '/inner';
      const elsewhere = new THREE.Group();
      elsewhere.name = '/elsewhere';
      root.add(inner, elsewhere);
      const moved = new THREE.Mesh();
      moved.name = '/moved';
      elsewhere.add(moved);
      expect(findObjectByName(inner, '/moved')).toBeUndefined();
      inner.add(moved);
      expect(findObjectByName(inner, '/moved')).toBe(moved);
    } finally {
      THREE.Object3D.prototype.getObjectByName = walk;
    }
  });

  it('forgets a disposed subtree and indexes a node added after attach', () => {
    const root = new THREE.Group();
    attachSceneGraphIndex(root);
    const layer = new THREE.Group();
    layer.name = '/layer';
    const leaf = new THREE.Mesh();
    leaf.name = '/layer/leaf';
    layer.add(leaf);
    root.add(layer);
    expect(findObjectByName(root, '/layer/leaf')).toBe(leaf);
    disposeObjectTree(layer);
    root.remove(layer);
    expect(findObjectByName(root, '/layer/leaf')).toBeUndefined();
    expect(findObjectByName(root, '/layer')).toBeUndefined();
    expect(sceneGraphIndexOf(leaf)).toBeUndefined();
  });

  it('falls back to the walk for an unindexed root, and yields undefined for none', () => {
    const loose = new THREE.Group();
    const child = new THREE.Mesh();
    child.name = '/c';
    loose.add(child);
    expect(findObjectByName(loose, '/c')).toBe(child);
    expect(findObjectByName(null, '/c')).toBeUndefined();
    expect(findObjectByName(undefined, '/c')).toBeUndefined();
  });
});
