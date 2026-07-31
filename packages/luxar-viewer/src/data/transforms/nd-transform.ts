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
 * which keeps it above this floor for any scale ≤ 10). A dimension flagged that
 * way is not being sliced, so the no-preimage rule must not fire on it.
 *
 * This is a BACKSTOP only — the authoritative signal is the node's own
 * `extend_to_all` name list, passed in as `extendDims`. The sentinel never
 * reaches here for a Lines node (every lines call site derives with
 * `applyPartialExtendTolerance: false`), so inferring extension from the
 * tolerance alone would blank extended lines nodes.
 */
const EXTENDED_TOLERANCE_FLOOR = 1e9;

/**
 * Float-comparison slack (in units of the dimension's step) when deciding
 * whether a candidate's forward image hits the queried world value. Generous
 * enough to absorb one multiply + one divide, far tighter than any real gap.
 */
const PREIMAGE_EPSILON = 1e-6;

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
 * The question is therefore not "is the inverse on the grid?" but the spec's
 * own forward rule for discrete ordinals,
 * `effective = round(scale · original + offset)`
 * (`docs/guides/specs/ND_TRANSFORMS_SPEC.md` §4.1): **does any local grid point
 * map to the queried world value?** Those two predicates coincide only for
 * integer `scale`/`offset`. Testing inverse-on-grid instead would blank every
 * slice of a node with, say, `offset: 0.4` — for which `round(k + 0.4) = k`
 * gives every world value a preimage — and §11.3 explicitly blesses fractional
 * scale on discrete dims ("valid but lossy"), as does the Python validator.
 *
 * So {@link resolveDiscretePreimage} walks the local grid candidates bracketing
 * the exact inverse and keeps the first whose forward image rounds to the
 * queried world value. On success the local slice position is **snapped to that
 * candidate**, which additionally makes the query exactly on-grid — so the
 * downstream half-step window selects that one category and can no longer admit
 * a neighbour at an exact midpoint. On failure the world value is the image of
 * no local value and must display nothing; "nothing" cannot be encoded as a
 * slice position, so it is reported out of band as `noPreimage` and the
 * per-geometry range queries turn it into an empty range list, which every
 * loader already renders as "cleared".
 *
 * Exempt: `extend_to_all` dimensions (named in `extendDims` — the dimension is
 * not being sliced at all) and categorical permutations (a bijection always has
 * exactly one preimage).
 *
 * KNOWN LIMITATION: the local grid is taken to be the dimension's declared
 * `step`, which is a WORLD-space quantity. For a unit-converting transform the
 * local data may sit on a different grid, and there is no metadata describing
 * it. This matches what the downstream membership window already assumes (it
 * applies the same `step` as a LOCAL half-width), so the two stay consistent.
 *
 * @param slicePosition - Current slice position in world (transformed) space
 * @param tolerance - Per-dimension tolerance in world space
 * @param ndTransform - Composed world nD transform for this node
 * @param dimensions - Per-dimension metadata (length = ndim). `name` is matched
 *   against `ndTransform` keys; `discrete`/`step` drive the no-preimage rule.
 * @param displayDims - Indices of displayed dimensions (to skip)
 * @param extendDims - The node's `extend_to_all` dimension NAMES. Authoritative
 *   exemption list: the tolerance sentinel is absent on every Lines path, so it
 *   cannot be inferred from `tolerance` alone.
 * @returns New slicePosition (snapped to the resolved local grid point where a
 *   preimage exists) and tolerance in local (raw) space, plus `noPreimage` when
 *   the world slice has no local counterpart at all
 */
export function invertNdTransformForQuery(
  slicePosition: readonly number[],
  tolerance: readonly number[],
  ndTransform: NdTransformMap,
  dimensions: readonly QueryDimensionInfo[],
  displayDims: readonly number[],
  extendDims: readonly string[] = []
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

      // A dimension the node extends is not being sliced. The NAME list is
      // authoritative; the rescaled-sentinel check is only a backstop (it never
      // fires on a Lines path). Read it BEFORE the tolerance is rescaled below.
      const isExtended =
        extendDims.includes(dim.name) || localTolerance[d] >= EXTENDED_TOLERANCE_FLOOR;
      const world = localSlice[d];
      const local = (world - offset) / scale;

      localSlice[d] = local;
      localTolerance[d] = localTolerance[d] / Math.abs(scale);

      if (dim.discrete && !isExtended) {
        const resolved = resolveDiscretePreimage(world, scale, offset, dim.step);
        if (resolved === null) {
          noPreimage = true;
        } else {
          // Snap to the resolved grid point: the query is now exactly on-grid,
          // so the downstream half-step window selects that single category.
          localSlice[d] = resolved;
        }
      }
    }
  }

  return { slicePosition: localSlice, tolerance: localTolerance, noPreimage };
}

/**
 * Resolve which local grid point (if any) the queried world value is the image
 * of, under the spec's forward rule for discrete ordinals
 * (`effective = round(scale · local + offset)`, §4.1).
 *
 * Returns the winning local value — snap the query to it — or `null` when the
 * world value is the image of no local grid point.
 *
 * Only the two grid points bracketing the exact inverse need testing: the
 * forward map is affine hence monotonic, so the local values sharing a world
 * image form one contiguous run around the inverse. `|scale| > 1` admits at most
 * one; `|scale| < 1` (a lossy downsample, §11.3) admits several and the nearest
 * is the right representative. A missing or non-positive step falls back to 1,
 * matching the rest of the slicing stack's default.
 */
function resolveDiscretePreimage(
  world: number,
  scale: number,
  offset: number,
  step: number | undefined
): number | null {
  const gridStep = step !== undefined && step !== null && step > 0 ? step : 1;
  const exact = (world - offset) / scale;
  const ratio = exact / gridStep;
  const tol = gridStep * PREIMAGE_EPSILON;

  let best: number | null = null;
  let bestDist = Infinity;
  for (const k of [Math.floor(ratio), Math.ceil(ratio)]) {
    const candidate = k * gridStep;
    // The forward rule rounds to the nearest WORLD grid point.
    const image = Math.round((scale * candidate + offset) / gridStep) * gridStep;
    if (Math.abs(image - world) > tol) continue;
    // Both candidates can qualify at |scale| < 1 — keep the one nearest the
    // exact inverse.
    const dist = Math.abs(candidate - exact);
    if (dist < bestDist) {
      best = candidate;
      bestDist = dist;
    }
  }
  return best;
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
