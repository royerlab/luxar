/**
 * Path → `Object3D` index for a scene subtree (B9a, tracking #2944).
 *
 * The viewer resolves a node by its PATH (its `name`) on every per-node step of
 * an update pass: the sweep's eligibility probe, each type's process and
 * commit, the lazy-LOD release hooks, the layers panel. `Object3D.getObjectByName`
 * answers that with a depth-first walk of the whole subtree, so one pass over N
 * nodes costs O(N²) visits — millions per slice move on a 2000-part partition
 * tree. {@link findObjectByName} answers it from a map instead.
 *
 * **Maintained by the scene graph itself**, not by the call sites that build it,
 * so no add or remove path can forget it: every member node carries
 * `childadded` / `childremoved` listeners (dispatched by three's
 * `add` / `remove` / `attach` / `clear` / `removeFromParent`), and a member's
 * `name` becomes an accessor that re-keys the entry on a rename. A new subtree
 * is indexed as it is added; a removed one (a node disposal, a dataset switch's
 * teardown) is dropped as it is removed.
 *
 * **Identical answers to `getObjectByName`, by construction:**
 * - A UNIQUE name returns the one indexed object after an O(depth) check that it
 *   is still attached below the lookup root. That check is what keeps the index
 *   safe against graph edits that bypass three's events (e.g. the eager-child
 *   loader's slot flattening splices `children` arrays directly): a stale hit is
 *   dropped and the lookup falls back to the walk.
 * - A name held by TWO OR MORE live objects is ambiguous, and three's answer is
 *   the FIRST in depth-first pre-order — an ordering the map does not keep. Such
 *   lookups fall back to `getObjectByName` itself, so the first-match semantic is
 *   preserved exactly rather than approximated. (Loader paths are unique, so this
 *   is the exception, not the hot path.)
 * - A name the index does not hold falls back to the walk too (a node inserted
 *   without an event, or a genuinely absent path), and a hit found that way is
 *   indexed so the next lookup is O(1).
 * - A walk that MISSES is remembered per lookup root, so a path that is not
 *   built yet (a lazy LOD level, a deferred partition part — looked up on every
 *   pass) costs one walk, not one per lookup. The memory is invalidated by an
 *   EPOCH bumped on every event that could turn a miss into a hit: a
 *   `childadded` anywhere in the tree (a new node, or a member moved into the
 *   looked-up root) and a member rename. A removal, and the eager-child
 *   loader's event-less slot flattening, can only SHRINK the set of nodes below
 *   a root, so they never stale a miss. (A node inserted with no event at all
 *   would — nothing in the viewer does that, and the property test in
 *   `tests/unit/utils/scene-graph-index.test.ts` checks every lookup against
 *   the walk after each edit, flattening included.)
 * - The empty name is never indexed (every unnamed group would collide).
 *
 * A lookup rooted at a node of an indexed tree (not only at the indexed root) is
 * served by the same index, with the ancestor check bounded at that node.
 *
 * @module utils/scene-graph-index
 */

import type * as THREE from 'three';

/** Scene object tracked by the name index. */
export type Node = THREE.Object3D;

/** Three.js child event delivered when the graph changes. */
export interface ChildEvent {
  child: Node;
}

/** Node → the index it currently belongs to. */
const MEMBERSHIP = new WeakMap<Node, SceneGraphIndex>();
/** Nodes whose `name` is already an index-aware accessor. */
const NAME_TRACKED = new WeakSet<Node>();

/**
 * Replace `node.name` (a plain own data property on every `Object3D`) with an
 * accessor holding the same value, whose setter re-keys the node in whichever
 * index it belongs to at the time. Installed once per node, on first indexing;
 * a node that has left every index keeps it as a plain getter/setter.
 */
function trackName(node: Node): void {
  if (NAME_TRACKED.has(node)) return;
  NAME_TRACKED.add(node);
  let value = node.name;
  Object.defineProperty(node, 'name', {
    configurable: true,
    enumerable: true,
    get(): string {
      return value;
    },
    set(next: string): void {
      if (next === value) return;
      const previous = value;
      value = next;
      MEMBERSHIP.get(node)?.renamed(node, previous, next);
    },
  });
}

/** Whether `node` is `scope` or a descendant of it (O(depth) parent walk). */
function isWithin(node: Node, scope: Node): boolean {
  for (let n: Node | null = node; n !== null; n = n.parent) {
    if (n === scope) return true;
  }
  return false;
}

/** Bound on remembered misses per lookup root (then the memory starts over). */
const MAX_MISSES_PER_SCOPE = 4096;

/** Misses remembered for one lookup root, valid for one {@link SceneGraphIndex} epoch. */
/** Cached miss for one scene-graph search epoch. Do not un-export: TypeDoc needs this name. */
export interface MissMemo {
  epoch: number;
  names: Set<string>;
}

/** The path → node map for one attached root. */
export class SceneGraphIndex {
  /** Name → live members holding it (length > 1 ⇒ ambiguous). */
  private readonly byName = new Map<string, Node[]>();
  /** Bumped by every edit that could turn a remembered miss into a hit. */
  private epoch = 0;
  /** Lookup root → the names a walk from it missed during {@link epoch}. */
  private misses = new WeakMap<Node, MissMemo>();
  private readonly onChildAdded = (event: ChildEvent): void => {
    this.addSubtree(event.child);
  };
  private readonly onChildRemoved = (event: ChildEvent): void => {
    this.removeSubtree(event.child);
  };

  constructor(readonly root: Node) {
    this.addSubtree(root);
  }

  /** Stop maintaining the index and release every member. */
  detach(): void {
    this.removeSubtree(this.root);
    this.byName.clear();
    this.misses = new WeakMap();
    this.epoch++;
  }

  /** Number of distinct indexed names (diagnostics / tests). */
  get size(): number {
    return this.byName.size;
  }

  /** `scope.getObjectByName(name)`, answered from the map where it is exact. */
  find(scope: Node, name: string): Node | undefined {
    const holders = name === '' ? undefined : this.byName.get(name);
    if (holders === undefined) {
      if (this.isKnownMiss(scope, name)) return undefined;
    } else {
      const hit = this.uniqueHolderWithin(holders, scope);
      if (hit !== undefined) return hit;
    }
    return this.walk(scope, name);
  }

  /** The one live holder of a name, when it lies within `scope`. */
  private uniqueHolderWithin(holders: Node[], scope: Node): Node | undefined {
    // Forget holders detached without an event (stale) first, so one stale
    // twin cannot leave a name ambiguous forever.
    if (holders.length > 1) this.pruneDetached(holders);
    if (holders.length !== 1) return undefined;
    const hit = holders[0];
    if (isWithin(hit, scope)) return hit;
    // Attached but outside `scope`: the walk is the only exact answer for an
    // unindexed twin. Detached: forget it.
    if (!isWithin(hit, this.root)) this.removeNode(hit);
    return undefined;
  }

  /** The exact answer by walking, remembering a miss and repairing a hit. */
  private walk(scope: Node, name: string): Node | undefined {
    const found = scope.getObjectByName(name);
    if (found === undefined) this.rememberMiss(scope, name);
    // Repair: a node that reached the tree without an event becomes a member,
    // so the next lookup of it is O(1).
    else if (MEMBERSHIP.get(found) !== this) this.addSubtree(found);
    return found;
  }

  /** @internal — called by a member's `name` accessor. */
  renamed(node: Node, previous: string, next: string): void {
    this.epoch++;
    this.unkey(node, previous);
    this.key(node, next);
  }

  private isKnownMiss(scope: Node, name: string): boolean {
    const memo = this.misses.get(scope);
    return memo !== undefined && memo.epoch === this.epoch && memo.names.has(name);
  }

  private rememberMiss(scope: Node, name: string): void {
    let memo = this.misses.get(scope);
    if (
      memo === undefined ||
      memo.epoch !== this.epoch ||
      memo.names.size >= MAX_MISSES_PER_SCOPE
    ) {
      memo = { epoch: this.epoch, names: new Set() };
      this.misses.set(scope, memo);
    }
    memo.names.add(name);
  }

  private pruneDetached(holders: readonly Node[]): void {
    for (const node of [...holders]) {
      if (!isWithin(node, this.root)) this.removeNode(node);
    }
  }

  private addSubtree(top: Node): void {
    // Even a subtree of existing members (a move) can bring a name below a
    // root a miss was remembered for.
    this.epoch++;
    top.traverse((node) => this.addNode(node));
  }

  private removeSubtree(top: Node): void {
    top.traverse((node) => {
      if (MEMBERSHIP.get(node) === this) this.removeNode(node);
    });
  }

  private addNode(node: Node): void {
    const current = MEMBERSHIP.get(node);
    if (current === this) return;
    // Moved in from another tree without that tree seeing a removal.
    current?.removeNode(node);
    MEMBERSHIP.set(node, this);
    trackName(node);
    this.key(node, node.name);
    node.addEventListener('childadded', this.onChildAdded);
    node.addEventListener('childremoved', this.onChildRemoved);
  }

  private removeNode(node: Node): void {
    if (MEMBERSHIP.get(node) !== this) return;
    MEMBERSHIP.delete(node);
    this.unkey(node, node.name);
    node.removeEventListener('childadded', this.onChildAdded);
    node.removeEventListener('childremoved', this.onChildRemoved);
  }

  private key(node: Node, name: string): void {
    if (name === '') return;
    const holders = this.byName.get(name);
    if (holders === undefined) this.byName.set(name, [node]);
    else if (!holders.includes(node)) holders.push(node);
  }

  private unkey(node: Node, name: string): void {
    const holders = this.byName.get(name);
    if (holders === undefined) return;
    const i = holders.indexOf(node);
    if (i < 0) return;
    holders.splice(i, 1);
    if (holders.length === 0) this.byName.delete(name);
  }
}

/**
 * Index `root`'s subtree and keep it maintained from here on. Idempotent: an
 * already-indexed root returns its existing index.
 */
export function attachSceneGraphIndex(root: Node): SceneGraphIndex {
  const existing = MEMBERSHIP.get(root);
  if (existing?.root === root) return existing;
  return new SceneGraphIndex(root);
}

/**
 * Stop maintaining the index rooted at `root`, if `root` is an indexed root
 * (a node that merely belongs to an indexed tree is left alone).
 */
export function detachSceneGraphIndex(root: Node): void {
  const index = MEMBERSHIP.get(root);
  if (index?.root === root) index.detach();
}

/** The index serving lookups rooted at `node` (its tree's), if any. */
export function sceneGraphIndexOf(node: Node): SceneGraphIndex | undefined {
  return MEMBERSHIP.get(node);
}

/**
 * Drop-in for `root.getObjectByName(name)`: the same object, from the index when
 * `root` belongs to an indexed tree, and from the walk otherwise (a nullish
 * `root` yields `undefined`, as `root?.getObjectByName(name)` does).
 */
export function findObjectByName(root: Node | null | undefined, name: string): Node | undefined {
  if (!root) return undefined;
  const index = MEMBERSHIP.get(root);
  return index ? index.find(root, name) : root.getObjectByName(name);
}
