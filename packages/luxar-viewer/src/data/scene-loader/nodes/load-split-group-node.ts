/**
 * Initial-load path for a single kind=`split` `Group` scene-graph node.
 *
 * Mirrors the generic-group branch in `load-scene-nodes.ts` — creates
 * a `THREE.Group`, applies the transform, recurses into children — and
 * additionally:
 *
 *   1. Reads `display_type` and `max_elements` from the parent's attrs
 *      for layers-panel use (currently the layer-state walker pulls
 *      these directly; this loader doesn't need to expose anything).
 *   2. Recurses each child via `loadSceneNodes` so the children's own
 *      type-specific loaders run. **All children stay visible** — no
 *      LOD-style selector — relying on THREE's per-mesh frustum culling
 *      for the per-part culling benefit.
 *
 * The wrapper's `position_bounds` (union over children) is on the
 * on-disk attrs already; loaders that need it read `node.attrs
 * .position_bounds` rather than re-computing it.
 *
 * Sibling of `load-lod-group-node.ts` (which has a per-frame selector).
 *
 * @module data/scene-loader/nodes/load-split-group-node
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { SceneNode } from '../../data-loader-types';
import type { SplitGroupMetadata } from '../../../types/split-group';
import type { NodeBuildCtx } from './build-ctx';
import { loadSceneNodes } from './load-scene-nodes';

/**
 * Load a kind=`split` `Group` on initial scene construction.
 *
 * Pattern:
 *   1. Create a `THREE.Group` for the wrapper; apply transform.
 *   2. Recurse each child through `loadSceneNodes` so its
 *      geometry-specific loader runs and a placeholder mesh attaches.
 *
 * No registry needed — Split has no per-frame decision to make. THREE's
 * per-mesh frustum culling handles per-part culling automatically once
 * the children attach.
 */
export async function loadSplitGroupNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
): Promise<THREE.Group> {
  const attrs = node.attrs as unknown as SplitGroupMetadata;
  log.custom(
    '✂️',
    Modules.SCENE_LOADER,
    `Loading split-kind group: ${node.path} (display_type=${attrs.display_type}, max_elements=${attrs.max_elements})`
  );

  const splitGroup = new THREE.Group();
  splitGroup.name = node.path;
  if (attrs.transform) {
    ctx.nodeFactory.applyTransform(splitGroup, attrs.transform);
  }
  parentThree.add(splitGroup);

  const sceneChildren = node.children ?? [];
  if (sceneChildren.length === 0) {
    log.warning(
      Modules.SCENE_LOADER,
      `split-kind group ${node.path} has no children`
    );
    return splitGroup;
  }

  for (const child of sceneChildren) {
    const childLoc = parentLoc.resolve(child.path.slice(1));
    await loadSceneNodes(child, splitGroup, childLoc, ctx);
  }

  log.info(
    Modules.SCENE_LOADER,
    `  Loaded ${sceneChildren.length} part(s) for split group ${node.path}`
  );

  return splitGroup;
}
