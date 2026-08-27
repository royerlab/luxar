/**
 * Bounded concurrent recursion for eager scene children.
 *
 * @module data/scene-loader/nodes/load-children-concurrently
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import type { SceneNode } from '../../data-loader-types';
import type { NodeBuildCtx } from './build-ctx';

// This is a per-parent bound: nested groups may multiply the total fan-out.
// Eight siblings expose roughly 40 rung-array requests at one level, which
// stays below the global 64-request fetch gate while collapsing waterfalls.
export const EAGER_CHILD_LOAD_CONCURRENCY = 8;

/** Signature of the injected recursive scene-graph walker. */
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

  const worker = async (): Promise<void> => {
    while (!failed) {
      const index = nextIndex++;
      if (index >= sceneChildren.length) return;
      const child = sceneChildren[index];
      const slot = slots[index];
      try {
        const childLoc = parentLoc.resolve(child.path.slice(1));
        await loadChild(child, slot, childLoc, ctx);
      } catch (error) {
        if (!failed) firstError = error;
        failed = true;
      } finally {
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
