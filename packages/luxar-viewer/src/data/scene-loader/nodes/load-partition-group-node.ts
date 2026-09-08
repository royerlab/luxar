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
 *      type-specific loaders run, then registers validated per-part bounds
 *      for frustum-only visibility and fetch gating.
 *   3. Validates a stored `bsp_tree` before exposing it to depth sorting;
 *      malformed or geometrically unsound trees fall back to centroid order.
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
import type { PartitionGroupChild } from '../../../scene/lod-group-registry';
import { loadChildrenConcurrently, type LoadSceneChildren } from './load-children-concurrently';

interface PositionBounds {
  min: readonly number[];
  max: readonly number[];
}

interface BspBoundsSummary {
  minCenter: number[];
  maxCenter: number[];
  minLow: number[];
  maxHigh: number[];
}

interface BspTreeValidation {
  tree: BspTreeNode;
  verified: boolean;
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

function summarizeLeaf(bounds: PositionBounds): BspBoundsSummary {
  const center = bounds.min.map((low, axis) => 0.5 * (low + bounds.max[axis]));
  return {
    minCenter: center.slice(),
    maxCenter: center.slice(),
    minLow: [...bounds.min],
    maxHigh: [...bounds.max],
  };
}

function mergeSummaries(left: BspBoundsSummary, right: BspBoundsSummary): BspBoundsSummary {
  return {
    minCenter: left.minCenter.map((value, axis) => Math.min(value, right.minCenter[axis])),
    maxCenter: left.maxCenter.map((value, axis) => Math.max(value, right.maxCenter[axis])),
    minLow: left.minLow.map((value, axis) => Math.min(value, right.minLow[axis])),
    maxHigh: left.maxHigh.map((value, axis) => Math.max(value, right.maxHigh[axis])),
  };
}

// Deliberately the bounds-free subset of summarizeStraddlingTree's structural checks.
function bspTreeStructureIsValid(node: unknown, partCount: number, seenParts: boolean[]): boolean {
  if (!node || typeof node !== 'object') return false;
  const record = node as Record<string, unknown>;
  if (Object.prototype.hasOwnProperty.call(record, 'part')) {
    const part = record.part;
    if (
      typeof part !== 'number' ||
      !Number.isInteger(part) ||
      part < 0 ||
      part >= partCount ||
      seenParts[part]
    ) {
      return false;
    }
    seenParts[part] = true;
    return true;
  }

  const axis = record.axis;
  const split = record.split;
  if (
    typeof axis !== 'number' ||
    !Number.isInteger(axis) ||
    axis < 0 ||
    typeof split !== 'number' ||
    !Number.isFinite(split)
  ) {
    return false;
  }
  return (
    bspTreeStructureIsValid(record.left, partCount, seenParts) &&
    bspTreeStructureIsValid(record.right, partCount, seenParts)
  );
}

function summarizeStraddlingTree(
  node: unknown,
  bounds: PositionBounds[],
  seenParts: boolean[],
  overlapFloors: number[],
  validateSplits: boolean
): BspBoundsSummary | null {
  if (!node || typeof node !== 'object') return null;
  const record = node as Record<string, unknown>;
  if (Object.prototype.hasOwnProperty.call(record, 'part')) {
    const part = record.part;
    if (
      typeof part !== 'number' ||
      !Number.isInteger(part) ||
      part < 0 ||
      part >= bounds.length ||
      seenParts[part]
    ) {
      return null;
    }
    seenParts[part] = true;
    return summarizeLeaf(bounds[part]);
  }

  const axis = record.axis;
  const split = record.split;
  if (
    typeof axis !== 'number' ||
    !Number.isInteger(axis) ||
    axis < 0 ||
    axis >= bounds[0].min.length ||
    typeof split !== 'number' ||
    !Number.isFinite(split)
  ) {
    return null;
  }

  const left = summarizeStraddlingTree(
    record.left,
    bounds,
    seenParts,
    overlapFloors,
    validateSplits
  );
  const right = summarizeStraddlingTree(
    record.right,
    bounds,
    seenParts,
    overlapFloors,
    validateSplits
  );
  if (!left || !right) return null;

  const measuredOverlap = Math.max(0, left.maxHigh[axis] - right.minLow[axis]);
  if (!validateSplits) {
    overlapFloors[axis] = Math.max(overlapFloors[axis], measuredOverlap);
  } else {
    // Keep this in parity with Python's serialized_bsp_tree_straddles_centers /
    // serialized_bsp_tree_axis_overlap_floors in core/group/partition.py. The
    // axis floor is a deliberately coarse worst-case halo tolerance, not a
    // tight bound for this particular cut.
    const overlap = Math.max(overlapFloors[axis], measuredOverlap);
    if (split < left.maxCenter[axis] - overlap || split > right.minCenter[axis] + overlap) {
      return null;
    }
  }
  return mergeSummaries(left, right);
}

/** Return a well-formed tree; `verified` says whether part bounds also graded its splits. */
function validatedBspTree(tree: unknown, children: SceneNode[]): BspTreeValidation | undefined {
  try {
    const structuredParts = new Array<boolean>(children.length).fill(false);
    if (
      !bspTreeStructureIsValid(tree, children.length, structuredParts) ||
      !structuredParts.every(Boolean)
    ) {
      return undefined;
    }
    const bounds = indexedPartBounds(children);
    if (!bounds) {
      return { tree: tree as BspTreeNode, verified: false };
    }
    const overlapFloors = new Array<number>(Math.max(3, bounds[0].min.length)).fill(0);
    const collectedParts = new Array<boolean>(bounds.length).fill(false);
    if (
      !summarizeStraddlingTree(tree, bounds, collectedParts, overlapFloors, false) ||
      !collectedParts.every(Boolean)
    ) {
      return undefined;
    }
    const validatedParts = new Array<boolean>(bounds.length).fill(false);
    if (
      !summarizeStraddlingTree(tree, bounds, validatedParts, overlapFloors, true) ||
      !validatedParts.every(Boolean)
    ) {
      return undefined;
    }
    return { tree: tree as BspTreeNode, verified: true };
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
 * ``load-children-concurrently.ts``.
 *
 * Partition selection is frustum-only: every intersecting part remains
 * visible, while off-frustum parts are hidden so their registered loaders can
 * skip slice updates. No LOD coverage, hysteresis, or level substitution is
 * involved because hiding an in-frustum spatial part would delete geometry.
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

  await loadChildrenConcurrently(sceneChildren, partitionGroup, parentLoc, ctx, loadChildren, {
    configureSlot: (slot, child, index) => {
      slot.userData.partIndex = partIndexForChild(child, index);
    },
    configureLoadedChild: (object, child, index) => {
      object.userData.partIndex = partIndexForChild(child, index);
    },
  });

  const partBounds = indexedPartBounds(sceneChildren);
  if (ctx.lodGroupRegistry) {
    if (!partBounds) {
      log.warning(
        Modules.SCENE_LOADER,
        `Partition frustum selection disabled for ${node.path}: part bounds are missing, invalid, or inconsistent`
      );
    } else {
      const registryChildren: Array<PartitionGroupChild | undefined> = new Array(
        sceneChildren.length
      ).fill(undefined);
      for (let loadIndex = 0; loadIndex < sceneChildren.length; loadIndex++) {
        const partIndex = partIndexForChild(sceneChildren[loadIndex], loadIndex);
        const objects = partitionGroup.children.filter(
          (candidate) => candidate.userData.partIndex === partIndex
        );
        if (objects.length > 0) {
          registryChildren[partIndex] = {
            path: sceneChildren[loadIndex].path,
            objects,
            positionBounds: partBounds[partIndex],
          };
        }
      }
      if (registryChildren.every((child) => child !== undefined)) {
        ctx.lodGroupRegistry.registerPartition({
          path: node.path,
          groupObject: partitionGroup,
          children: registryChildren as PartitionGroupChild[],
        });
      } else {
        log.warning(
          Modules.SCENE_LOADER,
          `Partition frustum selection disabled for ${node.path}: one or more parts produced no scene object`
        );
      }
    }
  }

  // Stash the BSP split-plane tree (when present) for the coordinator's exact
  // back-to-front part ordering; absent for streamed grid/content merges, where
  // the coordinator falls back to a per-part centroid heuristic.
  if (attrs.bsp_tree) {
    const validation = validatedBspTree(attrs.bsp_tree, sceneChildren);
    if (validation) {
      partitionGroup.userData.bspTree = validation.tree;
      if (!validation.verified) {
        log.info(
          Modules.SCENE_LOADER,
          `partition-kind group ${node.path} has a bsp_tree without verifiable part bounds; keeping the stored tree`
        );
      }
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
