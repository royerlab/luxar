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
 * This means no changes are needed in the per-element projection kernels
 * (projectTo3D, clipSegmentToSlice, projectLinesTo3D, the WASM/TS
 * effective-radii kernels): they keep comparing raw coordinates against a
 * query they never know was transformed.
 *
 * The one exception is the no-preimage rule below — a world slice that no local
 * value can occupy is not expressible as a query position, so each geometry's
 * range query carries a single early-out for it. See
 * {@link invertNdTransformForQuery}.
 *
 * @module data/nd-transform
 */

import { NdTransformMap, NdTransformEntry, isPermutation } from '../../types/zarr';
import type { SceneNode } from '../data-loader-types';

/**
 * Per-dimension metadata the inverse-query needs: the `name` to look the
 * transform entry up by, plus enough to know where a DISCRETE dimension's
 * values may sit (see the no-preimage rule in
 * {@link invertNdTransformForQuery}).
 *
 * Structurally a subset of `DimensionMetadata`, so callers pass
 * `viewState.dimensions` straight through.
 */
export interface QueryDimensionInfo {
  name?: string;
  discrete?: boolean;
  step?: number;
}

/**
 * Tolerance at or above this is the `extend_to_all` "infinite" sentinel
 * (`EXTEND_TO_ALL_TOLERANCE` is 1e10, and the inverse divides it by |scale|,
 * which keeps it far above this floor for any sane scale). A dimension flagged
 * that way is not being sliced, so the no-preimage rule must not fire on it.
 */
const EXTENDED_TOLERANCE_FLOOR = 1e9;

/**
 * Relative slack when testing whether a local position lands on a discrete
 * dimension's `k · step` grid: generous enough to absorb the float error of one
 * divide, far tighter than any real fraction-of-a-step gap.
 */
const ON_GRID_EPSILON = 1e-6;

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
 * ## The no-preimage rule
 *
 * A DISCRETE dimension's values live on the `k · step` grid — that is the
 * format's discrete contract, and the whole slicing stack leans on it: the
 * navigation UI snaps slice targets to that grid, and the per-element
 * MEMBERSHIP gates therefore use a half-step window (`|value − target| ≤ 0.5 ×
 * step`; see `tolerance-computer.ts` and `effective-radius-calculator.ts`),
 * which selects exactly one category *for an on-grid target*.
 *
 * A non-unit affine `nd_transform` breaks that premise: an on-grid WORLD target
 * inverts to an OFF-grid LOCAL target. `scale: 2` at world T = 7 gives local
 * 3.5, and the half-step window then admits local 3 AND local 4 — two
 * neighbouring categories drawn at once, neither of which belongs to the
 * requested world slice. At `scale: 3`, world T = 7 (local 2.333) silently
 * admits local 2.
 *
 * The spec's forward rule for discrete ordinals is
 * `effective = round(scale · original + offset)`
 * (`docs/guides/specs/ND_TRANSFORMS_SPEC.md` §4.1), so a world value that is not
 * the image of any local grid point has **no preimage** and must display
 * nothing. "Nothing" cannot be encoded as a slice position, so it is reported
 * out of band as `noPreimage`; the per-geometry range queries turn that into an
 * empty range list, which every loader already renders as "cleared".
 *
 * `extend_to_all` dimensions are exempt — their tolerance is the infinite
 * sentinel, meaning the dimension is not being sliced at all.
 *
 * @param slicePosition - Current slice position in world (transformed) space
 * @param tolerance - Per-dimension tolerance in world space
 * @param ndTransform - Composed world nD transform for this node
 * @param dimensions - Per-dimension metadata (length = ndim). `name` is matched
 *   against `ndTransform` keys; `discrete`/`step` drive the no-preimage rule.
 * @param displayDims - Indices of displayed dimensions (to skip)
 * @returns New slicePosition and tolerance in local (raw) space, plus
 *   `noPreimage` when the world slice has no local counterpart at all
 */
export function invertNdTransformForQuery(
  slicePosition: readonly number[],
  tolerance: readonly number[],
  ndTransform: NdTransformMap,
  dimensions: readonly QueryDimensionInfo[],
  displayDims: readonly number[]
): { slicePosition: number[]; tolerance: number[]; noPreimage: boolean } {
  const localSlice = [...slicePosition];
  const localTolerance = [...tolerance];
  let noPreimage = false;

  for (let d = 0; d < localSlice.length; d++) {
    if (displayDims.includes(d)) continue;
    if (d >= dimensions.length) continue;

    const dim = dimensions[d];
    if (dim?.name === undefined) continue;

    const entry = ndTransform[dim.name];
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
      // Tolerance unchanged for categorical (integer matching). A permutation
      // is a bijection, so every world index has exactly one preimage — the
      // no-preimage rule cannot apply to a categorical dimension.
    } else {
      // Affine inverse: raw = (effective - offset) / scale
      const scale = entry.scale ?? 1.0;
      const offset = entry.offset ?? 0.0;

      if (scale === 0) {
        // Scale=0 collapses all values — can't invert, skip
        continue;
      }

      // Read the extend_to_all sentinel BEFORE it is rescaled below.
      const isExtended = localTolerance[d] >= EXTENDED_TOLERANCE_FLOOR;
      const local = (localSlice[d] - offset) / scale;

      localSlice[d] = local;
      localTolerance[d] = localTolerance[d] / Math.abs(scale);

      if (dim.discrete && !isExtended && !isOnDiscreteGrid(local, dim.step)) {
        noPreimage = true;
      }
    }
  }

  return { slicePosition: localSlice, tolerance: localTolerance, noPreimage };
}

/**
 * True when `local` sits on a discrete dimension's `k · step` grid (within
 * {@link ON_GRID_EPSILON} of an exact multiple). A missing or non-positive step
 * falls back to 1, matching the rest of the slicing stack's default.
 */
function isOnDiscreteGrid(local: number, step: number | undefined): boolean {
  const gridStep = step !== undefined && step !== null && step > 0 ? step : 1;
  const ratio = local / gridStep;
  return Math.abs(ratio - Math.round(ratio)) <= ON_GRID_EPSILON;
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
  // Guard against malformed scene graphs that contain the same node
  // reference twice (cycles or shared subtrees). Without this, a cycle
  // would push the same nd_transform onto `chain` repeatedly, then the
  // backtrack `pop()` would only undo one push per `findPath` frame —
  // resulting in either an infinite recursion or a chain with duplicate
  // entries that gets double-composed downstream.
  const visited = new Set<SceneNode>();

  function findPath(node: SceneNode, path: string): boolean {
    if (visited.has(node)) {
      throw new Error(
        `computeWorldNdTransform: malformed scene graph — node "${node.path}" ` +
          'encountered twice (cycle or shared reference). Aborting traversal.'
      );
    }
    visited.add(node);

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
