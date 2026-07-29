/**
 * Ancestor-aware ("effective") visibility for scene-graph nodes.
 *
 * `object.visible` is a LOCAL flag: THREE hides a subtree by clearing the flag
 * on one ancestor and leaves every descendant's own flag untouched. Anything
 * that asks "does this node actually render / is it worth spending work on?"
 * must therefore walk the parent chain. Two independent places got this wrong
 * in different ways, so the walk lives here once:
 *
 *   - a layer authored `visible=false` (or toggled off in the layers panel)
 *     hides the LAYER object, while an LOD level underneath keeps
 *     `visible === true`;
 *   - the LOD registry hides a LEVEL, which can itself be a GROUP (partition
 *     tiles), while the member meshes underneath keep `visible === true`.
 *
 * Consumers: the LOD-group registry's load gate (never fetch a hidden layer's
 * lazy levels), the LOD eviction policy (a hidden level is evictable), the
 * pick pass, and the depth-sort scheduler.
 *
 * @module utils/object-visibility
 */

/**
 * Minimal structural shape of a scene-graph node for the visibility walk.
 * `THREE.Object3D` satisfies it structurally (`visible: boolean`,
 * `parent: Object3D | null`), and so do the deliberately-minimal mock shapes
 * used by the LOD modules (which never import THREE for this).
 *
 * Both members are optional so a partial shape stays usable: a missing
 * `visible` counts as visible (only an explicit `false` hides), and a missing
 * `parent` ends the walk.
 */
export interface VisibilityNode {
  visible?: boolean;
  parent?: VisibilityNode | null;
}

/**
 * True when `node` and every one of its ancestors is visible — i.e. the node
 * is reachable on screen, not buried under a hidden subtree.
 *
 * Only an explicit `visible === false` hides (see {@link VisibilityNode}), so
 * `undefined`/`null` nodes and partial shapes are treated as visible. That
 * makes "ancestors only" expressible as `isEffectivelyVisible(node.parent)`,
 * which is what callers that own the node's own flag (the LOD registry sets
 * the displayed level's `visible`) need.
 *
 * O(depth) with no allocation — safe on a per-frame path.
 */
export function isEffectivelyVisible(node: VisibilityNode | null | undefined): boolean {
  for (let o: VisibilityNode | null | undefined = node; o; o = o.parent) {
    if (o.visible === false) return false;
  }
  return true;
}
