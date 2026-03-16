/**
 * nD Transform utilities for non-displayed dimension transforms.
 *
 * These transforms operate per-dimension on non-displayed dimensions:
 * - Continuous/discrete: affine (scale * x + offset)
 * - Categorical: permutation (index remapping)
 *
 * **Design: Inverse-Query Approach**
 *
 * Instead of transforming millions of point coordinates per frame (O(N)),
 * we inverse-transform the query (slicePosition + tolerance) once (O(1)).
 * The spatial index stores raw (untransformed) coordinates, so we convert
 * the viewer's world-space query back to local space before querying.
 *
 * This means NO changes are needed in any loader internals (projectTo3D,
 * clipSegmentToSlice, buildInstanceBuffers, etc.).
 *
 * @module data/nd-transform
 */

import { NdTransformMap, NdTransformEntry, isPermutation } from '../types/zarr';
import type { SceneNode } from './data-loader-types';

/**
 * Inverse-transform slicePosition and tolerance from world space to local space.
 *
 * For affine (effective = scale * raw + offset):
 *   local_pos = (world_pos - offset) / scale
 *   local_tol = world_tol / |scale|
 *
 * For permutation: compute inverse permutation and remap the slice index.
 *   Tolerance is unchanged (categorical dims use integer matching).
 *
 * This allows querying the spatial index in local (stored) coordinates
 * without transforming any point data — O(1) per dimension.
 *
 * @param slicePosition - Current slice position in world (transformed) space
 * @param tolerance - Per-dimension tolerance in world space
 * @param ndTransform - Composed world nD transform for this node
 * @param dimensionNames - Dimension name for each index (length = ndim)
 * @param displayDims - Indices of displayed dimensions (to skip)
 * @returns New slicePosition and tolerance in local (raw) space
 */
export function invertNdTransformForQuery(
  slicePosition: number[],
  tolerance: number[],
  ndTransform: NdTransformMap,
  dimensionNames: string[],
  displayDims: number[]
): { slicePosition: number[]; tolerance: number[] } {
  const localSlice = [...slicePosition];
  const localTolerance = [...tolerance];

  for (let d = 0; d < localSlice.length; d++) {
    if (displayDims.includes(d)) continue;
    if (d >= dimensionNames.length) continue;

    const entry = ndTransform[dimensionNames[d]];
    if (!entry) continue;

    if (isPermutation(entry)) {
      // Compute inverse permutation: if perm[i] = j, then inversePerm[j] = i
      const perm = entry.permutation;
      const inversePerm = new Array<number>(perm.length);
      for (let i = 0; i < perm.length; i++) {
        inversePerm[perm[i]] = i;
      }
      // Map world index back to local index
      const worldIndex = Math.round(localSlice[d]);
      if (worldIndex >= 0 && worldIndex < inversePerm.length) {
        localSlice[d] = inversePerm[worldIndex];
      }
      // Tolerance unchanged for categorical (integer matching)
    } else {
      // Affine inverse: raw = (effective - offset) / scale
      const scale = entry.scale ?? 1.0;
      const offset = entry.offset ?? 0.0;

      if (scale === 0) {
        // Scale=0 collapses all values — can't invert, skip
        continue;
      }

      localSlice[d] = (localSlice[d] - offset) / scale;
      localTolerance[d] = localTolerance[d] / Math.abs(scale);
    }
  }

  return { slicePosition: localSlice, tolerance: localTolerance };
}

/**
 * Compose multiple nD transforms (applied root-first, leaf-last).
 *
 * For affine: parent(child(x)) = s_p * (s_c * x + o_c) + o_p
 *   composed_scale = s_p * s_c
 *   composed_offset = s_p * o_c + o_p
 *
 * For permutation: composed[i] = parent_perm[child_perm[i]]
 *
 * @param transforms - Array of transforms, ordered root-first
 * @returns Composed transform (empty map = identity)
 */
export function composeNdTransforms(...transforms: NdTransformMap[]): NdTransformMap {
  if (transforms.length === 0) return {};

  const allDims = new Set<string>();
  for (const t of transforms) {
    for (const key of Object.keys(t)) {
      allDims.add(key);
    }
  }

  if (allDims.size === 0) return {};

  const result: NdTransformMap = {};

  for (const dimName of allDims) {
    const entries: NdTransformEntry[] = [];
    for (const t of transforms) {
      if (dimName in t) {
        entries.push(t[dimName]);
      }
    }

    if (entries.length === 0) continue;
    if (entries.length === 1) {
      result[dimName] = entries[0];
      continue;
    }

    // Check type consistency across entries
    const allPerm = entries.every((e) => isPermutation(e));
    const allAffine = entries.every((e) => !isPermutation(e));
    if (!allPerm && !allAffine) {
      // Mixed types — skip this dimension (cannot compose)
      continue;
    }

    if (allPerm) {
      // Compose permutations: start from leaf (last), apply parent outward
      let perm = [...(entries[entries.length - 1] as { permutation: number[] }).permutation];
      for (let j = entries.length - 2; j >= 0; j--) {
        const parentPerm = (entries[j] as { permutation: number[] }).permutation;
        perm = perm.map((p) => parentPerm[p]);
      }
      result[dimName] = { permutation: perm };
    } else {
      // Compose affines: start from leaf, apply parent outward
      let scale =
        (entries[entries.length - 1] as NdTransformEntry & { scale?: number }).scale ?? 1.0;
      let offset =
        (entries[entries.length - 1] as NdTransformEntry & { offset?: number }).offset ?? 0.0;
      for (let j = entries.length - 2; j >= 0; j--) {
        const sP = (entries[j] as NdTransformEntry & { scale?: number }).scale ?? 1.0;
        const oP = (entries[j] as NdTransformEntry & { offset?: number }).offset ?? 0.0;
        offset = sP * offset + oP;
        scale = sP * scale;
      }
      const entry: Record<string, number> = {};
      if (scale !== 1.0) entry.scale = scale;
      if (offset !== 0.0) entry.offset = offset;
      if (Object.keys(entry).length > 0) {
        result[dimName] = entry;
      }
    }
  }

  return result;
}

/**
 * Compute the world nD transform for a node by walking the scene graph.
 *
 * Collects nd_transforms from root to target and composes them.
 *
 * @param sceneGraph - Root scene node
 * @param targetPath - Path to the target node
 * @returns Composed world nD transform (empty map = identity)
 */
export function computeWorldNdTransform(sceneGraph: SceneNode, targetPath: string): NdTransformMap {
  const chain: NdTransformMap[] = [];

  function findPath(node: SceneNode, path: string): boolean {
    if (node.attrs.nd_transform) {
      chain.push(node.attrs.nd_transform as NdTransformMap);
    }
    if (node.path === path) return true;
    if (node.children) {
      for (const child of node.children) {
        if (findPath(child, path)) return true;
      }
    }
    // Backtrack: remove this node's transform if target not found in subtree
    if (node.attrs.nd_transform) {
      chain.pop();
    }
    return false;
  }

  findPath(sceneGraph, targetPath);

  if (chain.length === 0) return {};
  return composeNdTransforms(...chain);
}
