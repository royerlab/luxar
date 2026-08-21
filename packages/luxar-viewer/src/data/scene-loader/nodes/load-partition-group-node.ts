/**
 * Initial-load path for a single kind=`partition` `Group` scene-graph node.
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
 * @module data/scene-loader/nodes/load-partition-group-node
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { SceneNode } from '../../data-loader-types';
import type { BspTreeNode, PartitionGroupMetadata } from '../../../types/partition-group';
import type { NodeBuildCtx } from './build-ctx';
import type { LoadSceneChildren } from './load-lod-group-node';

interface PositionBounds {
  min: readonly number[];
  max: readonly number[];
}

function partIndexForChild(child: SceneNode, loadIndex: number): number {
  return (child.attrs?.child_index as number | undefined) ?? loadIndex;
}

function readPositionBounds(child: SceneNode): PositionBounds | null {
  const attrs = child.attrs as Record<string, unknown>;
  const raw = (attrs.position_bounds ?? attrs.center_bounds) as
    { min?: unknown; max?: unknown } | undefined;
  if (!raw || !Array.isArray(raw.min) || !Array.isArray(raw.max)) return null;
  if (raw.min.length === 0 || raw.min.length !== raw.max.length) return null;
  for (let axis = 0; axis < raw.min.length; axis++) {
    const low = raw.min[axis];
    const high = raw.max[axis];
    if (typeof low !== 'number' || typeof high !== 'number') return null;
    if (!Number.isFinite(low) || !Number.isFinite(high) || low > high) return null;
  }
  return { min: raw.min as number[], max: raw.max as number[] };
}

function indexedPartBounds(children: SceneNode[]): PositionBounds[] | null {
  const bounds: Array<PositionBounds | undefined> = new Array(children.length);
  let dimensions: number | undefined;
  for (let loadIndex = 0; loadIndex < children.length; loadIndex++) {
    const partIndex = partIndexForChild(children[loadIndex], loadIndex);
    const partBounds = readPositionBounds(children[loadIndex]);
    if (
      !Number.isInteger(partIndex) ||
      partIndex < 0 ||
      partIndex >= children.length ||
      bounds[partIndex] !== undefined ||
      partBounds === null
    ) {
      return null;
    }
    dimensions ??= partBounds.min.length;
    if (partBounds.min.length !== dimensions) return null;
    bounds[partIndex] = partBounds;
  }
  return bounds.every((partBounds) => partBounds !== undefined)
    ? (bounds as PositionBounds[])
    : null;
}

function straddlingLeafLabels(node: unknown, bounds: PositionBounds[]): number[] | null {
  if (!node || typeof node !== 'object') return null;
  const record = node as Record<string, unknown>;
  if (Object.prototype.hasOwnProperty.call(record, 'part')) {
    return Number.isInteger(record.part) && (record.part as number) >= 0
      ? [record.part as number]
      : null;
  }

  const axis = record.axis;
  const split = record.split;
  if (
    !Number.isInteger(axis) ||
    (axis as number) < 0 ||
    (axis as number) > 2 ||
    (axis as number) >= bounds[0].min.length ||
    typeof split !== 'number' ||
    !Number.isFinite(split)
  ) {
    return null;
  }

  const leftLabels = straddlingLeafLabels(record.left, bounds);
  const rightLabels = straddlingLeafLabels(record.right, bounds);
  if (!leftLabels || !rightLabels) return null;
  const labels = [...leftLabels, ...rightLabels];
  if (labels.some((part) => part >= bounds.length)) return null;

  const splitAxis = axis as number;
  let leftCenter = -Infinity;
  let rightCenter = Infinity;
  let leftHigh = -Infinity;
  let rightLow = Infinity;
  for (const part of leftLabels) {
    leftCenter = Math.max(
      leftCenter,
      0.5 * (bounds[part].min[splitAxis] + bounds[part].max[splitAxis])
    );
    leftHigh = Math.max(leftHigh, bounds[part].max[splitAxis]);
  }
  for (const part of rightLabels) {
    rightCenter = Math.min(
      rightCenter,
      0.5 * (bounds[part].min[splitAxis] + bounds[part].max[splitAxis])
    );
    rightLow = Math.min(rightLow, bounds[part].min[splitAxis]);
  }
  const overlap = Math.max(0, leftHigh - rightLow);
  if (split < leftCenter - overlap || split > rightCenter + overlap) return null;
  return labels;
}

function validatedBspTree(tree: unknown, children: SceneNode[]): BspTreeNode | undefined {
  const bounds = indexedPartBounds(children);
  if (!bounds) return undefined;
  try {
    const labels = straddlingLeafLabels(tree, bounds);
    if (!labels || labels.length !== bounds.length) return undefined;
    labels.sort((a, b) => a - b);
    if (labels.some((label, index) => label !== index)) return undefined;
    return tree as BspTreeNode;
  } catch {
    return undefined;
  }
}

/**
 * Load a kind=`partition` `Group` on initial scene construction.
 *
 * Pattern:
 *   1. Create a `THREE.Group` for the wrapper; apply transform.
 *   2. Recurse each child through the caller-supplied
 *      ``loadChildren`` (a thin handle to ``loadSceneNodes``) so its
 *      geometry-specific loader runs and a placeholder mesh attaches.
 *
 * The recursion handle is passed in (not imported) to break what would
 * otherwise be a cyclic dependency with ``load-scene-nodes.ts`` — the
 * dep-cruiser check rejects static back-references. Same pattern as
 * ``load-lod-group-node.ts``.
 *
 * No registry needed — Partition has no per-frame decision to make. THREE's
 * per-mesh frustum culling handles per-part culling automatically once
 * the children attach.
 */
export async function loadPartitionGroupNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx,
  loadChildren: LoadSceneChildren
): Promise<THREE.Group> {
  const attrs = node.attrs as unknown as PartitionGroupMetadata;
  log.custom(
    '✂️',
    Modules.SCENE_LOADER,
    `Loading partition-kind group: ${node.path} (display_type=${attrs.display_type}, max_elements=${attrs.max_elements})`
  );

  const partitionGroup = new THREE.Group();
  partitionGroup.name = node.path;
  // Mark the THREE node with its specialized-group kind so the picking
  // system can resolve a hit on an inner ``part_<i>`` child back to the
  // wrapper's path. Mirrors the convention in load-lod-group-node.ts.
  partitionGroup.userData.kind = 'partition';
  if (attrs.transform) {
    ctx.nodeFactory.applyTransform(partitionGroup, attrs.transform);
  }
  parentThree.add(partitionGroup);

  const sceneChildren = node.children ?? [];
  if (sceneChildren.length === 0) {
    log.warning(Modules.SCENE_LOADER, `partition-kind group ${node.path} has no children`);
    return partitionGroup;
  }

  for (let i = 0; i < sceneChildren.length; i++) {
    const child = sceneChildren[i];
    const childLoc = parentLoc.resolve(child.path.slice(1));
    const before = partitionGroup.children.length;
    await loadChildren(child, partitionGroup, childLoc, ctx);
    // Tag the part's THREE object(s) with their part index (the on-disk
    // `child_index`, falling back to load order) so the depth-sort coordinator
    // can map a part's render mesh back to a `bsp_tree` leaf for exact
    // back-to-front ordering. A part subtree may add >1 object (e.g. a per-part
    // lod group) — tag them all.
    const partIndex = partIndexForChild(child, i);
    for (let j = before; j < partitionGroup.children.length; j++) {
      partitionGroup.children[j].userData.partIndex = partIndex;
    }
  }

  // Stash the BSP split-plane tree (when present) for the coordinator's exact
  // back-to-front part ordering; absent for streamed grid/content merges, where
  // the coordinator falls back to a per-part centroid heuristic.
  if (attrs.bsp_tree) {
    const bspTree = validatedBspTree(attrs.bsp_tree, sceneChildren);
    if (bspTree) {
      partitionGroup.userData.bspTree = bspTree;
    } else {
      log.warning(
        Modules.SCENE_LOADER,
        `partition-kind group ${node.path} has an invalid bsp_tree; falling back to centroid ordering`
      );
    }
  }

  log.info(
    Modules.SCENE_LOADER,
    `  Loaded ${sceneChildren.length} part(s) for partition group ${node.path}`
  );

  return partitionGroup;
}
