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
dimension. This folder implements the second path — the per-element
projection kernels (`projectTo3D`, `clipSegmentToSlice`,
`projectLinesTo3D`, …) never see the transform. The one loader-side
concession is the `noPreimage` early-out in each geometry's range
query; see "The no-preimage rule" below.

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
// On a DISCRETE dimension the local position is snapped to the grid
// point whose forward image is the queried world value, or `noPreimage`
// is set when there is none — see "The no-preimage rule" below.
// `extendDims` = the node's extend_to_all NAMES (exemption list).
export function invertNdTransformForQuery(
  slicePosition: readonly number[],
  tolerance: readonly number[],
  ndTransform: NdTransformMap,
  dimensions: readonly QueryDimensionInfo[],
  displayDims: readonly number[],
  extendDims?: readonly string[]
): { slicePosition: number[]; tolerance: number[]; noPreimage: boolean };

// Per-dimension metadata the inverse-query needs. A structural subset
// of DimensionMetadata, so callers pass `viewState.dimensions` through.
export interface QueryDimensionInfo {
  name?: string;
  discrete?: boolean;
  step?: number;
}

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

## The no-preimage rule

A DISCRETE dimension's values live on the `k · step` grid. The slicing stack
relies on that: navigation snaps slice targets to the grid, so the per-element
MEMBERSHIP gates can use a half-step window (`|value − target| ≤ 0.5 × step` —
`../loaders/spatial-query/tolerance-computer.ts`,
`../points/effective-radius-calculator.ts`) and still select exactly one
category.

A non-unit affine `nd_transform` breaks that premise, because it makes the
**query** off-grid even though the slider is on-grid:

| transform  | world T | local query | half-step window admits |
| ---------- | ------- | ----------- | ----------------------- |
| `scale: 2` | 8       | 4           | local 4 — correct       |
| `scale: 2` | 7       | 3.5         | local 3 **and** local 4 |
| `scale: 3` | 7       | 2.333       | local 2                 |

The right question is not "is the inverse on the grid?" but the spec's own
forward rule for discrete ordinals,
`effective = round(scale · local + offset)`: **does any local grid point map to
the queried world value?** Those two predicates coincide only for integer
`scale`/`offset`, and §11.3 explicitly blesses fractional scale ("valid but
lossy"). Testing inverse-on-grid instead would blank valid data — with
`offset: 0.4`, `round(k + 0.4) = k` gives every world value a preimage, yet the
inverse `w − 0.4` is never on the grid, so the node would go permanently dark.

So `resolveDiscretePreimage` walks the local grid candidates bracketing the
exact inverse and keeps the one whose forward image rounds to the queried world
value:

| transform     | world | resolves to | why                            |
| ------------- | ----- | ----------- | ------------------------------ |
| `scale: 2`    | 8     | local 4     | `round(2·4) = 8`               |
| `scale: 2`    | 7     | **none**    | `round(2·3)=6`, `round(2·4)=8` |
| `scale: 1.2`  | 1     | local 1     | `round(1.2·1) = 1`             |
| `offset: 0.4` | _w_   | local _w_   | `round(w + 0.4) = w`           |
| `scale: 0.5`  | 3     | local 6     | lossy downsample, nearest rep  |

On success the local slice position is **snapped to that candidate**, so the
query is exactly on-grid and the downstream half-step window can no longer admit
a neighbour at an exact midpoint. On failure `invertNdTransformForQuery` reports
`noPreimage: true`; it cannot be encoded as a slice position, so it rides the
derived `ViewState.noPreimage` and each geometry's range query
(`queryVisiblePointRanges` / `queryVisibleSegmentRanges` /
`queryVisibleSplatRanges`) returns an empty range list — whose existing "no
visible elements" path already clears the geometry. That placement is
deliberate: it sits ahead of both the no-spatial-index load-all fallback and the
`extend_to_all` short-circuit.

Exemptions. Categorical permutations are exempt because a permutation is a
bijection. `extend_to_all` dimensions are exempt via the node's `extend_to_all`
**name list**, passed in as `extendDims` — _not_ inferred from the 1e10 tolerance
sentinel, because every Lines call site derives with
`applyPartialExtendTolerance: false`, so a lines node's extended dims never
carry it. The sentinel check remains only as a backstop.

Known limitation: the local grid is taken to be the dimension's declared `step`,
a WORLD-space quantity, and no metadata describes the local grid. This matches
what the downstream membership window already assumes, so the two stay
consistent.

The `demos/demo_nd_transforms.py` bench is the visual regression harness for
this — it draws one row per transform with markers that print their own local
index, so a wrong row lights up visibly.

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
