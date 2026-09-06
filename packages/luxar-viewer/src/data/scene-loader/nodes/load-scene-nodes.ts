/**
 * Recursive scene-graph walk: for each leaf node dispatch to the
 * per-type initial-load helper (`loadPointsNode` / `loadLinesNode` /
 * `loadGSplatsNode` / `loadMeshNode`, plus `loadSoundNode` for the heard-not-drawn
 * `sound` type); for each group node create a `THREE.Group`,
 * apply its transform if present, and recurse into its children.
 *
 * Each leaf call is wrapped in `loadLeafNode` so a single failing node doesn't
 * sink the whole scene — its siblings still render. Archive container faults
 * remain fatal because no sibling backed by the same store can recover. The
 * placeholder pattern (each leaf attaches an empty placeholder to parentThree
 * *before* fetching data) makes leaf-local failures recoverable through
 * `retryFailedLoader`.
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import type { SceneNode } from '../../data-loader-types';
import { loadLeafNode } from './load-leaf-error-dispatch';
import { geometryDescriptorFor } from '../geometry-descriptors';
import { loadLodGroupNode } from './load-lod-group-node';
import { loadPartitionGroupNode } from './load-partition-group-node';
import { loadSoundNode } from './load-sound-node';
import type { NodeBuildCtx } from './build-ctx';
import { loadChildrenConcurrently } from './load-children-concurrently';

/**
 * Walk the scene-graph rooted at `node` and load every leaf via the
 * per-geometry helpers. Each group node becomes a THREE.Group; each
 * leaf node attaches its placeholder mesh through its type-specific
 * helper. Recurses into child groups.
 */
export async function loadSceneNodes(
  node: SceneNode,
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<void> {
  const descriptor = geometryDescriptorFor(node.type);
  if (descriptor) {
    // Each loadXNode attaches its own placeholder to parentThree before
    // fetching data; no caller-side `if (node) add(node)` is needed. The
    // placeholder stays in the scene even on failure so retry can populate it.
    await loadLeafNode(() => descriptor.loadNode(node, parentThree, parentLoc, ctx), node.path);
  } else if (node.type === 'sound') {
    // A sound node is heard, not drawn: it is in the contract's `node_types`
    // but not in `geometry_types`, so it has no descriptor. Its placeholder
    // carries the descriptor the audio engine consumes after the scene attaches.
    await loadLeafNode(() => loadSoundNode(node, parentThree, parentLoc, ctx), node.path);
  } else if (node.type === 'group' && node.attrs.kind === 'lod') {
    // A kind=lod Group is a specialized container that recurses into
    // children itself (it needs to capture each child's THREE node +
    // coverage_fraction to register the LOD entry). No outer recursion
    // afterwards. Wrap with ``loadLeafNode`` for the same error-capture
    // semantics as the leaf branches above — a failing LOD group
    // shouldn't sink the rest of the scene.
    await loadLeafNode(
      () => loadLodGroupNode(node, parentThree, parentLoc, ctx, loadSceneNodes),
      node.path
    );
  } else if (node.type === 'group' && node.attrs.kind === 'partition') {
    // A kind=partition Group is a specialized container that recurses into
    // children itself. Its registry entry frustum-gates spatial parts without
    // selecting among them: every in-frustum child renders simultaneously.
    // Same error-capture wrapping as above.
    await loadLeafNode(
      () => loadPartitionGroupNode(node, parentThree, parentLoc, ctx, loadSceneNodes),
      node.path
    );
  } else if (node.children) {
    // Create group and recurse
    const group = new THREE.Group();
    group.name = node.path;

    // Apply transform if present
    if (node.attrs.transform) {
      ctx.nodeFactory.applyTransform(group, node.attrs.transform);
    }

    parentThree.add(group);

    await loadChildrenConcurrently(node.children, group, parentLoc, ctx, loadSceneNodes);
  }
}
