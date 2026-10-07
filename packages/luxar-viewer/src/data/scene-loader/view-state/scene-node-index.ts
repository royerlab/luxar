/**
 * Path index over a loaded scene graph: `path → { node, worldNdTransform }`.
 *
 * Every view derivation needs the node at a path and the world nD transform
 * composed from the root down to it. Walking the graph for each
 * (`computeWorldNdTransform`, a depth-first search from the root) made one
 * pass O(N²) on a many-part partition: every part's derivation walked every
 * other part. The SceneNode graph is built once by `buildSceneGraph` and never
 * mutated afterwards (deferred activations build THREE objects and loaders,
 * not SceneNodes), so the index is computed once, right after the build, and
 * every lookup is O(1). The same pass records each node's ancestor chain, so
 * attribute composition (`effective-attrs.ts`) costs O(depth) per node rather
 * than a linear `children.find` descent per level.
 *
 * Results are identical to the walk: a path that occurs twice resolves to its
 * first occurrence in depth-first pre-order (the walk's first match), an
 * unknown path has the identity transform (`{}`), and a node reached twice —
 * a cycle or a shared subtree — is refused with the walk's error.
 *
 * @module data/scene-loader/view-state/scene-node-index
 */

import { collectAncestorNodes } from '../../attrs-composer';
import type { SceneNode } from '../../data-loader-types';
import type { NdTransformMap } from '../../../types/zarr';
import { composeNdTransforms } from '../../transforms/nd-transform';

/** One indexed scene node. */
export interface IndexedSceneNode {
  readonly node: SceneNode;
  /** Composed root → node nD transform (empty map = identity). */
  readonly worldNdTransform: NdTransformMap;
  /**
   * Root → node chain, the node included and a `type: 'scene'` root left out —
   * exactly what `attrs-composer.collectAncestorNodes` returns for this path.
   */
  readonly ancestors: readonly SceneNode[];
}

const IDENTITY: NdTransformMap = Object.freeze({}) as NdTransformMap;

/** Immutable `path → node + world nD transform` index of one scene graph. */
export class SceneNodeIndex {
  private readonly byPath = new Map<string, IndexedSceneNode>();

  constructor(readonly root: SceneNode) {
    this.visit(root, [], [], new Set<SceneNode>());
  }

  private visit(
    node: SceneNode,
    chain: NdTransformMap[],
    above: readonly SceneNode[],
    visited: Set<SceneNode>
  ): void {
    if (visited.has(node)) {
      throw new Error(
        `SceneNodeIndex: malformed scene graph — node "${node.path}" ` +
          'encountered twice (cycle or shared reference). Aborting traversal.'
      );
    }
    visited.add(node);
    const own = node.attrs.nd_transform as NdTransformMap | undefined;
    const nodeChain = own ? [...chain, own] : chain;
    // The composer's chain omits a scene root (it carries no composable attrs).
    const ancestors = node === this.root && node.type === 'scene' ? above : [...above, node];
    if (!this.byPath.has(node.path)) {
      this.byPath.set(node.path, {
        node,
        worldNdTransform: nodeChain.length === 0 ? IDENTITY : composeNdTransforms(...nodeChain),
        ancestors,
      });
    }
    for (const child of node.children ?? []) this.visit(child, nodeChain, ancestors, visited);
  }

  /** The indexed node at `path`, or `undefined`. */
  get(path: string): IndexedSceneNode | undefined {
    return this.byPath.get(path);
  }

  /** The node at `path`, or `null`. */
  node(path: string): SceneNode | null {
    return this.byPath.get(path)?.node ?? null;
  }

  /** World nD transform at `path` (identity for an unknown path). */
  worldNdTransform(path: string): NdTransformMap {
    return this.byPath.get(path)?.worldNdTransform ?? IDENTITY;
  }

  /**
   * Root → node chain for `path` (see {@link IndexedSceneNode.ancestors}), or
   * `undefined` for a path the graph does not hold.
   */
  ancestors(path: string): readonly SceneNode[] | undefined {
    return this.byPath.get(path)?.ancestors;
  }

  /**
   * `collectAncestorNodes(root, path)`, answered from the index: O(1) for an
   * indexed path. A path the graph does not hold, and the composer's empty
   * "nothing requested" path, fall back to the descent so the answer is
   * identical in every case.
   */
  ancestorChain(path: string): readonly SceneNode[] {
    return (path ? this.ancestors(path) : undefined) ?? collectAncestorNodes(this.root, path);
  }

  /** Whether the world nD transform at `path` is not the identity. */
  hasNdTransform(path: string): boolean {
    return Object.keys(this.worldNdTransform(path)).length > 0;
  }

  /** Number of distinct indexed paths. */
  get size(): number {
    return this.byPath.size;
  }
}
