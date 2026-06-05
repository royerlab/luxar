# nD transform helpers

Per-dimension transforms for **non-displayed** dimensions, used during
slice-based nD navigation. These are not 4x4 matrices — they are
scalar-per-dimension affines (continuous/discrete) or integer
permutations (categorical), composed along the scene graph and
inverted at query time so that the spatial index can stay in local
(untransformed) coordinates.

Single file: `nd-transform.ts`.

## Why inverse-query

A scene may contain tens of millions of elements. Each frame, the
viewer needs to know which are visible for the current
`slicePosition` + `tolerance` in **world** space. The forward path
(transform every element per frame) is O(N); the inverse path
(transform the query once, query the untransformed index) is O(1) per
dimension. This folder implements the second path — loader internals
(`projectTo3D`, `clipSegmentToSlice`, `projectLinesTo3D`, …) never
see the transform.

## Not 4x4 matrices

The NumPy-row-major / THREE.js-column-major gotcha called out in
`CLAUDE.md` applies to the 4x4 `transform` attribute on each
`SceneNode`, not to anything here. `NdTransformMap` is a
`Record<string, NdTransformEntry>` keyed by **dimension name**, where
each entry is either `{ scale?, offset? }` or
`{ permutation: number[] }`. There is no matrix layout to transpose.
The 4x4 row-vs-column-major validation lives in
`rendering/node-factory/validation.ts::validateTransformFormat`.

## Public API

```typescript
// Inverse-transform (slicePosition, tolerance) from world to local
// space. Returns fresh arrays; inputs are read-only. Displayed dims
// are skipped. Permutations use the inverse permutation (tolerance
// unchanged). Affine entries with scale === 0 are skipped.
export function invertNdTransformForQuery(
  slicePosition: readonly number[],
  tolerance: readonly number[],
  ndTransform: NdTransformMap,
  dimensionNames: string[],
  displayDims: readonly number[]
): { slicePosition: number[]; tolerance: number[] };

// Compose a root-first chain. Affines: s = s_p·s_c, o = s_p·o_c + o_p.
// Permutations: composed[i] = parent_perm[child_perm[i]]. Mixed types
// on the same dimension are dropped. Identity entries are omitted.
export function composeNdTransforms(...transforms: NdTransformMap[]): NdTransformMap;

// Walk sceneGraph to targetPath collecting every ancestor's
// attrs.nd_transform, then compose root-first. Returns {} if the
// path is not found or no ancestor sets one.
export function computeWorldNdTransform(sceneGraph: SceneNode, targetPath: string): NdTransformMap;
```

`NdTransformMap`, `NdTransformEntry`, and `isPermutation` come from
`../../types/zarr.ts`.

## Invariants

- **Pure / read-only.** All three functions return fresh objects and
  never mutate inputs.
- **Identity is `{}`.** `composeNdTransforms()` with no args returns
  `{}`; an entry that collapses to `scale=1, offset=0` is omitted.
- **scale === 0 is non-invertible** and left untouched by
  `invertNdTransformForQuery` rather than dividing by zero.
- **Mixed types drop.** Affine ∘ permutation on the same dimension
  has no meaning and that dimension is omitted from the composition.
- **Backtracking walk.** `computeWorldNdTransform` does a DFS with
  push/pop on subtree exit so sibling transforms never leak.
- **Cycle guard.** `computeWorldNdTransform` tracks visited
  `SceneNode`s and **throws** if the same node is encountered twice
  (cycle or shared subtree), rather than double-composing a transform
  or recursing forever.

## Caller

`../scene-loader/view-state/derive-node-view-state.ts` is the sole consumer: it
calls `computeWorldNdTransform(sceneGraph, path)` then
`invertNdTransformForQuery(...)` to convert the orchestrator's
world-space `ViewState` into the per-node local-space query that goes
into the spatial index.

## See also

- `../../types/zarr.ts` — `NdTransformMap`, `NdTransformEntry`,
  `isPermutation`.
- `../scene-loader/view-state/derive-node-view-state.ts` — the only caller.
- `../README.md` — parent overview, "nD Transforms on Non-Displayed
  Dimensions" section.
- `docs/guides/specs/ND_TRANSFORMS_SPEC.md` — full spec, Python-side
  encoder format, and orthogonality with the 4x4 `transform`.
