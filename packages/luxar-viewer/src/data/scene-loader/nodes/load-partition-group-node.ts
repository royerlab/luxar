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
import type { GeometryKind, SceneNode, ViewState } from '../../data-loader-types';
import type { BspTreeNode, PartitionGroupMetadata } from '../../../types/partition-group';
import type { NodeBuildCtx } from './build-ctx';
import type { PartitionGroupChild } from '../../../scene/lod-group-registry';
import {
  acquireEagerWorkingSet,
  loadChildrenConcurrently,
  type LoadSceneChildren,
} from './load-children-concurrently';
import { perfCounters } from '../../../profiling/perf-counters';
import { normalizeExtendDims } from '../view-state/extend-tolerance';
import { partBoundsIntersectSlice } from '../view-state/partition-slice-gate';
import { isUnderAny } from '../loaders/run-loader-updates';

/** Perf counter: partition parts whose subtree load (loader init) resolved. */
const S_PARTS_INITIALISED = perfCounters.slot('partition.partsInitialised');

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

/** Per-part slice-gate inputs (B4), by load index. */
interface PartGate {
  /** An `nd_transform` on the part's path or in its subtree: never slice-gate it. */
  sliceExempt: boolean;
  /** Every `extend_to_all` name in the part's subtree (never gated on). */
  extendDims: string[];
}

function hasNdTransform(attrs: Record<string, unknown>): boolean {
  const transform = attrs.nd_transform;
  return typeof transform === 'object' && transform !== null && Object.keys(transform).length > 0;
}

function collectSubtreeGate(node: SceneNode, gate: PartGate): void {
  const attrs = node.attrs as Record<string, unknown>;
  if (hasNdTransform(attrs)) gate.sliceExempt = true;
  for (const name of normalizeExtendDims(attrs.extend_to_all)) {
    if (!gate.extendDims.includes(name)) gate.extendDims.push(name);
  }
  for (const child of node.children ?? []) collectSubtreeGate(child, gate);
}

function partGates(children: SceneNode[], ctx: NodeBuildCtx): PartGate[] {
  return children.map((child) => {
    const gate: PartGate = {
      sliceExempt: ctx.pathHasNdTransform?.(child.path) === true,
      extendDims: [],
    };
    collectSubtreeGate(child, gate);
    return gate;
  });
}

/** Everything the gated part load needs, bundled (indices are load indices). */
interface PartitionLoadJob {
  node: SceneNode;
  group: THREE.Group;
  children: SceneNode[];
  parentLoc: zarr.Location<zarr.Readable>;
  ctx: NodeBuildCtx;
  loadPart: LoadSceneChildren;
  partBounds: PositionBounds[] | null;
  gates: PartGate[];
}

/** Which parts to load now and in what order (indices are load indices). */
interface PartitionLoadPlan {
  deferred: Set<number>;
  order?: number[];
}

/** Whether a part can draw anything for `view` (see `partition-slice-gate.ts`). */
function partInSlice(bounds: PositionBounds, gate: PartGate, view: ViewState): boolean {
  return gate.sliceExempt || partBoundsIntersectSlice(bounds, view, gate.extendDims);
}

/**
 * Gated loading (B4): load now only the ACTIVE parts — in the padded frustum
 * of the current camera (when the registry can rank them) AND able to draw
 * something for the current hidden-dim slice — nearest first. Every other part
 * is deferred: it keeps an empty slot and is activated by the LOD registry the
 * first time a loader pass needs it. No registry, or unusable part bounds,
 * means nothing can activate a deferred part, so everything loads (pre-B4).
 */
function planPartitionLoad(job: PartitionLoadJob): PartitionLoadPlan {
  const { children, ctx, partBounds } = job;
  const registry = ctx.lodGroupRegistry;
  if (!registry || !partBounds) return { deferred: new Set() };
  const bounds = children.map((child, index) => partBounds[partIndexForChild(child, index)]);
  const ranking = registry.rankPartitionPartsForLoad?.(job.group, bounds) ?? null;
  const view = ctx.getSliceView?.() ?? ctx.viewState;
  const deferred = new Set<number>();
  for (let index = 0; index < children.length; index++) {
    const inFrustum = ranking === null || ranking.inFrustum[index];
    if (!inFrustum || !partInSlice(bounds[index], job.gates[index], view)) deferred.add(index);
  }
  return { deferred, order: ranking?.order };
}

/**
 * One idempotent activation thunk per deferred part: load its subtree into its
 * slot. `registerOnly` (the registry's activators) attaches and registers the
 * part's loaders without loading their data — the loader pass that activates
 * the part sweeps them (see `NodeBuildCtx.registerOnly`). A failed activation
 * is forgotten, so the next call (a Retry re-arming the part) runs it again.
 */
function deferredActivators(
  job: PartitionLoadJob,
  slots: THREE.Group[],
  deferred: ReadonlySet<number>,
  registerOnly: boolean
): Map<number, () => Promise<void>> {
  const ctx = registerOnly ? { ...job.ctx, registerOnly: true } : job.ctx;
  const activators = new Map<number, () => Promise<void>>();
  for (const index of deferred) {
    let started: Promise<void> | null = null;
    activators.set(index, () => {
      started ??= activateDeferredPart(job, job.children[index], slots[index], ctx).catch(
        (error: unknown) => {
          started = null;
          throw error;
        }
      );
      return started;
    });
  }
  return activators;
}

/**
 * Load one deferred part into its slot, unless its dataset is no longer live
 * (a fire-and-forget activation can outlive a dataset switch). A failure — the
 * part's loader construction, which is not a leaf `LoaderError` — is recorded
 * on the part's path so Retry lists it, after discarding whatever the part had
 * attached or registered, so a re-run starts clean.
 */
async function activateDeferredPart(
  job: PartitionLoadJob,
  child: SceneNode,
  slot: THREE.Group,
  ctx: NodeBuildCtx
): Promise<void> {
  const releaseWorkingSet = await acquireEagerWorkingSet(child, ctx);
  try {
    if (!ctx.isDatasetLive()) return;
    await job.loadPart(child, slot, job.parentLoc.resolve(child.path.slice(1)), ctx);
    if (ctx.registerOnly) ctx.registry.clearFailure(child.path);
  } catch (error) {
    if (!ctx.isDatasetLive()) return;
    discardPartialPart(slot, ctx, child.path);
    ctx.registry.recordFailure(
      child.path,
      error instanceof Error ? error : new Error(String(error))
    );
    throw error;
  } finally {
    releaseWorkingSet();
  }
}

const LOADER_KINDS: readonly GeometryKind[] = ['points', 'lines', 'gsplats', 'mesh'];

/** Drop the placeholders and loaders a failed part activation left behind. */
function discardPartialPart(slot: THREE.Group, ctx: NodeBuildCtx, partPath: string): void {
  const under = new Set([partPath]);
  for (const kind of LOADER_KINDS) {
    const loaders = ctx.registry.loadersOf(kind);
    for (const [path, loader] of loaders) {
      if (!isUnderAny(path, under)) continue;
      loaders.delete(path);
      loader.dispose();
    }
  }
  slot.clear();
}

/** The registry entry of each part, by part index (`undefined` = no scene object). */
function registryChildrenFor(
  job: PartitionLoadJob,
  partBounds: PositionBounds[],
  activators: Map<number, () => Promise<void>> | null
): Array<PartitionGroupChild | undefined> {
  const { children, group, gates } = job;
  const registryChildren: Array<PartitionGroupChild | undefined> = new Array(children.length).fill(
    undefined
  );
  for (let loadIndex = 0; loadIndex < children.length; loadIndex++) {
    const partIndex = partIndexForChild(children[loadIndex], loadIndex);
    const objects = group.children.filter(
      (candidate) => candidate.userData.partIndex === partIndex
    );
    if (objects.length === 0) continue;
    const activate = activators?.get(loadIndex);
    registryChildren[partIndex] = {
      path: children[loadIndex].path,
      objects,
      positionBounds: partBounds[partIndex],
      sliceExempt: gates[loadIndex].sliceExempt,
      extendDims: gates[loadIndex].extendDims,
      ...(activate ? { activate } : {}),
    };
  }
  return registryChildren;
}

/**
 * Register the parts for frustum + slice gating. Returns whether the
 * partition was registered (a deferred part can only be activated if so).
 */
function registerPartitionParts(
  job: PartitionLoadJob,
  activators: Map<number, () => Promise<void>> | null
): boolean {
  const { node, partBounds, ctx } = job;
  const registry = ctx.lodGroupRegistry;
  if (!registry) return false;
  if (!partBounds) {
    log.warning(
      Modules.SCENE_LOADER,
      `Partition frustum selection disabled for ${node.path}: part bounds are missing, invalid, or inconsistent`
    );
    return false;
  }
  const registryChildren = registryChildrenFor(job, partBounds, activators);
  if (!registryChildren.every((child) => child !== undefined)) {
    log.warning(
      Modules.SCENE_LOADER,
      `Partition frustum selection disabled for ${node.path}: one or more parts produced no scene object`
    );
    return false;
  }
  registry.registerPartition({
    path: node.path,
    groupObject: job.group,
    children: registryChildren as PartitionGroupChild[],
  });
  return true;
}

/** Load the active parts, then register every part (deferred ones as lazy). */
async function loadAndRegisterParts(job: PartitionLoadJob): Promise<void> {
  const plan = planPartitionLoad(job);
  const slots = await loadChildrenConcurrently(
    job.children,
    job.group,
    job.parentLoc,
    job.ctx,
    job.loadPart,
    {
      configureSlot: (slot, child, index) => {
        slot.userData.partIndex = partIndexForChild(child, index);
      },
      configureLoadedChild: (object, child, index) => {
        object.userData.partIndex = partIndexForChild(child, index);
      },
      deferred: plan.deferred,
      order: plan.order,
    }
  );
  const activators =
    plan.deferred.size > 0 ? deferredActivators(job, slots, plan.deferred, true) : null;
  // Registration is what activates a deferred part; without it, load them now
  // (fully: no pass will sweep them for their first data).
  if (!registerPartitionParts(job, activators) && activators) {
    const eager = deferredActivators(job, slots, plan.deferred, false);
    await Promise.all([...eager.values()].map((activate) => activate()));
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

  const loadPart: LoadSceneChildren = async (child, parent, loc, childCtx) => {
    await loadChildren(child, parent, loc, childCtx);
    perfCounters.add(S_PARTS_INITIALISED);
  };
  await loadAndRegisterParts({
    node,
    group: partitionGroup,
    children: sceneChildren,
    parentLoc,
    ctx,
    loadPart,
    partBounds: indexedPartBounds(sceneChildren),
    gates: partGates(sceneChildren, ctx),
  });

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
