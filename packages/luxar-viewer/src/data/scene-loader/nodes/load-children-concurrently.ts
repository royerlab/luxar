/**
 * Bounded concurrent recursion for eager scene children.
 *
 * @module data/scene-loader/nodes/load-children-concurrently
 */

import * as THREE from 'three';
import { computeWorkingSetBudgetBytes } from '../../../cache/heap-budget';
import * as zarr from '../../zarr';
import type { SceneNode } from '../../data-loader-types';
import type { NodeBuildCtx } from './build-ctx';

// This is a per-parent bound: nested groups may multiply the total fan-out.
// Eight siblings expose roughly 40 rung-array requests at one level, which
// stays below the global 64-request fetch gate while collapsing waterfalls.
export const EAGER_CHILD_LOAD_CONCURRENCY = 8;

const EAGER_CHILD_LOAD_MEMORY_FALLBACK_BYTES = 256 * 1024 * 1024;
const ESTIMATED_LINE_SEGMENT_WORKING_SET_BYTES = 220;

interface WorkingSetWaiter {
  bytes: number;
  resolve: (release: () => void) => void;
}

class WorkingSetGate {
  private activeBytes = 0;
  private readonly waiters: WorkingSetWaiter[] = [];

  constructor(private readonly budgetBytes: number) {}

  acquire(estimatedBytes: number): Promise<() => void> {
    const bytes = Math.min(this.budgetBytes, Math.max(0, estimatedBytes));
    if (bytes === 0) return Promise.resolve(() => undefined);

    return new Promise((resolve) => {
      this.waiters.push({ bytes, resolve });
      this.drain();
    });
  }

  private drain(): void {
    while (this.waiters.length > 0) {
      const waiterIndex =
        this.activeBytes === 0
          ? 0
          : this.waiters.findIndex((waiter) => this.activeBytes + waiter.bytes <= this.budgetBytes);
      if (waiterIndex < 0) return;

      const [waiter] = this.waiters.splice(waiterIndex, 1);
      this.activeBytes += waiter.bytes;
      let released = false;
      waiter.resolve(() => {
        if (released) return;
        released = true;
        this.activeBytes -= waiter.bytes;
        this.drain();
      });
    }
  }
}

const workingSetGates = new WeakMap<NodeBuildCtx, WorkingSetGate>();

function workingSetGateFor(ctx: NodeBuildCtx): WorkingSetGate {
  let gate = workingSetGates.get(ctx);
  if (!gate) {
    gate = new WorkingSetGate(
      computeWorkingSetBudgetBytes() ?? EAGER_CHILD_LOAD_MEMORY_FALLBACK_BYTES
    );
    workingSetGates.set(ctx, gate);
  }
  return gate;
}

function estimateWorkingSetBytes(node: SceneNode): number {
  if (node.type !== 'lines') return 0;
  const readCount = (value: unknown): number =>
    typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.ceil(value) : 0;
  const vertexCount = readCount(node.attrs.n_vertices);
  const segmentCount = readCount(node.attrs.n_segments) || vertexCount;
  const rawNdim = readCount(node.attrs.ndim);
  const ndim = rawNdim || 3;
  const vertexBytes = 10 * ndim + 70;

  // Accumulator growth can briefly retain old + new vertex/segment buffers;
  // projection then emits endpoint attributes, and the committed geometry owns
  // a 6-texel RGBA32F texture plus ordering storage per segment. Vertex pressure
  // is `2.5 × (4 × ndim + 28)` = `10 × ndim + 70` bytes: the 1.5x
  // accumulator grow can retain both old and new capacities. Segment pressure
  // rounds the remaining overlap to 220 B/segment. Legacy nodes without
  // n_segments use the accumulator's conservative 1:1 fallback. Admission only
  // changes overlap, never stored or rendered data.
  return vertexCount * vertexBytes + segmentCount * ESTIMATED_LINE_SEGMENT_WORKING_SET_BYTES;
}

/**
 * Signature of the recursive scene-graph walker. Injected at the call site to
 * break the otherwise-cyclic import with `load-scene-nodes.ts`; a static
 * back-reference would fail the dep-cruiser cycle check.
 */
export type LoadSceneChildren = (
  node: SceneNode,
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
) => Promise<void>;

/** Hooks for decorating temporary slots and their flattened children. */
export interface LoadChildrenOptions {
  /** Configure the attached slot before its child starts loading. */
  configureSlot?: (slot: THREE.Group, child: SceneNode, index: number) => void;
  /** Configure each loaded object immediately before it replaces the slot. */
  configureLoadedChild?: (object: THREE.Object3D, child: SceneNode, index: number) => void;
}

function replaceSlot(
  parent: THREE.Object3D,
  slot: THREE.Group,
  configureLoadedChild: ((object: THREE.Object3D) => void) | undefined
): void {
  const slotIndex = parent.children.indexOf(slot);
  if (slotIndex < 0) return;

  const loadedChildren = slot.children.splice(0);
  for (const child of loadedChildren) {
    configureLoadedChild?.(child);
    child.parent = parent;
  }
  slot.parent = null;
  parent.children.splice(slotIndex, 1, ...loadedChildren);
}

export async function loadChildrenConcurrently(
  sceneChildren: SceneNode[],
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx,
  loadChild: LoadSceneChildren,
  options: LoadChildrenOptions = {}
): Promise<void> {
  // Slots reserve authored sibling order before any async work starts. They
  // stay attached while loading so commit paths that search from the scene
  // root can still find placeholders created under them. Each slot is
  // flattened in place when its child finishes, leaving the final hierarchy
  // identical to the serial loader's hierarchy.
  const slots = sceneChildren.map((child, index) => {
    const slot = new THREE.Group();
    options.configureSlot?.(slot, child, index);
    parentThree.add(slot);
    return slot;
  });

  let nextIndex = 0;
  let failed = false;
  let firstError: unknown;
  const workingSetGate = workingSetGateFor(ctx);

  const worker = async (): Promise<void> => {
    while (!failed) {
      const index = nextIndex++;
      if (index >= sceneChildren.length) return;
      const child = sceneChildren[index];
      const slot = slots[index];
      const releaseWorkingSet = await workingSetGate.acquire(estimateWorkingSetBytes(child));
      if (failed) {
        releaseWorkingSet();
        return;
      }
      try {
        const childLoc = parentLoc.resolve(child.path.slice(1));
        await loadChild(child, slot, childLoc, ctx);
      } catch (error) {
        if (!failed) firstError = error;
        failed = true;
      } finally {
        releaseWorkingSet();
        replaceSlot(parentThree, slot, (object) =>
          options.configureLoadedChild?.(object, child, index)
        );
      }
    }
  };

  const workerCount = Math.min(EAGER_CHILD_LOAD_CONCURRENCY, sceneChildren.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  // An unexpected error stops new work but lets already-started siblings
  // settle. Remove slots for children that were never started before
  // rethrowing, so a failed load cannot leave synthetic groups behind.
  for (const slot of slots) {
    if (slot.parent === parentThree) parentThree.remove(slot);
  }
  if (failed) throw firstError;
}
