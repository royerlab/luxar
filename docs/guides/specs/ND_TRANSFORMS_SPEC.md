# nD Transforms Specification

**Status:** Implemented (Python + TypeScript)
**Author:** Transform review session
**Date:** 2026-03-10

## 1. Motivation

Luxar scenes can have N dimensions, but transforms today are strictly 4x4 matrices operating on 3 displayed spatial dimensions. Non-displayed dimensions (Time, Channel, Depth, etc.) have no transform support — their raw coordinates are used directly for slicing/visibility filtering.

This creates limitations:
- Cannot align two datasets captured at different times (time offset)
- Cannot compose datasets with different physical units on non-spatial axes
- Cannot remap categorical channels between datasets in the same scene
- Hierarchical scene composition is limited to 3D spatial transforms

## 2. Design Principles

1. **Separate from 4x4 transform** — nD transforms are a distinct attribute, not embedded in the spatial matrix. Different algebraic structures (affine vs permutation) require different handling.
2. **Domain-aware** — Non-displayed dimensions form distinct algebraic domains with different valid operations.
3. **Hierarchical** — nD transforms compose through the scene graph parent chain, just like spatial transforms.
4. **Bounds-aware** — Scene-level dimension ranges auto-expand to reflect transformed coordinates.
5. **Backward compatible** — Scenes without `nd_transform` behave exactly as today.

## 3. Dimension Domains

Each dimension belongs to one algebraic domain based on its properties:

| Domain | Detection | Valid Transforms | Composition Rule |
|--------|-----------|-----------------|-----------------|
| **Displayed (spatial)** | `display=True` | 4x4 affine matrix (existing) | Matrix multiplication |
| **Continuous** | `display=False, categories=None` | Scale + offset (affine) | Affine composition |
| **Discrete ordinal** | `display=False, discrete=True, categories=None` | Scale + offset (with rounding) | Affine composition + round |
| **Categorical** | `categories is not None` | Permutation map | Permutation composition |

Detection uses existing `Dimension` class fields:
```python
if dim.display:
    domain = "displayed"
elif dim.is_categorical:
    domain = "categorical"
elif dim.discrete:
    domain = "discrete_ordinal"
else:
    domain = "continuous"
```

## 4. Transform Definitions

### 4.1 Continuous / Discrete Ordinal Dimensions

Per-dimension affine transform:

```
effective_value = scale * original_value + offset
```

For discrete ordinal dimensions, the result is rounded to the nearest integer:
```
effective_value = round(scale * original_value + offset)
```

**Parameters:**
- `scale: float` (default 1.0) — multiplicative factor
- `offset: float` (default 0.0) — additive shift

**Use cases:**
- Unit conversion: `{"scale": 0.001}` (ms to seconds)
- Time alignment: `{"offset": 100.0}` (shift dataset by 100 time units)
- Combined: `{"scale": 0.001, "offset": 50.0}` (convert ms to seconds, then shift by 50s)

### 4.2 Categorical Dimensions

Permutation map relabeling category indices:

```
effective_index = permutation[original_index]
```

**Parameters:**
- `permutation: list[int]` — maps old index to new index. Length must equal number of categories.

**Example:** For categories `["DAPI", "GFP", "mCherry"]`:
- `[2, 1, 0]` swaps DAPI and mCherry
- `[0, 2, 1]` swaps GFP and mCherry
- `[0, 1, 2]` is the identity (no change)

**Constraints:**
- Must be a valid permutation (each index appears exactly once)
- Length must match the number of categories in the dimension definition
- Unmapped indices are identity-mapped

## 5. Storage Format

### 5.1 Zarr Attributes

Stored alongside the existing `transform` attribute on any node:

```json
{
  "type": "group",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 5,0,0,1],
  "nd_transform": {
    "Time": {"scale": 0.001, "offset": 50.0},
    "Channel": {"permutation": [2, 1, 0]}
  }
}
```

**Rules:**
- `nd_transform` is optional. Absence means identity on all non-displayed dimensions.
- Keys are dimension **names** (matching `Dimension.name` in scene dimensions).
- Only non-displayed dimensions may appear as keys. Displayed dimensions are handled by the 4x4 `transform`.
- Omitted dimensions are identity-transformed.

### 5.2 Affine Entry Schema

```json
{
  "scale": 1.0,
  "offset": 0.0
}
```

Both fields are optional (default to 1.0 and 0.0 respectively). At least one must differ from the default for the entry to be meaningful.

### 5.3 Permutation Entry Schema

```json
{
  "permutation": [2, 1, 0]
}
```

Array of integers. Length must equal the number of categories for that dimension.

## 6. Python API

### 6.1 Setting nD Transforms

```python
import luxar
from luxar import transforms, Dimension, Dimensions

dims = Dimensions([
    Dimension("X", display=True),
    Dimension("Y", display=True),
    Dimension("Z", display=True),
    Dimension("Time", display=False, range=(0, 100)),
    Dimension("Channel", display=False, categories=["DAPI", "GFP", "mCherry"]),
])

with luxar.LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Group with spatial transform + nD transform
    group = scene.add_group(
        "DatasetB",
        transform=transforms.translate(10, 0, 0),       # 3D spatial
        nd_transform={
            "Time": {"scale": 0.001, "offset": 50.0},   # ms → s, shifted
            "Channel": {"permutation": [2, 1, 0]},       # swap DAPI/mCherry
        },
    )

    # Points inherit parent's nd_transform
    positions_5d = np.random.rand(1000, 5).astype(np.float32)
    group.add_points("cells", positions_5d)
```

### 6.2 Node Properties

```python
# Read nD transform (returns dict or None)
nd_xform = group.nd_transform
# → {"Time": {"scale": 0.001, "offset": 50.0}, "Channel": {"permutation": [2, 1, 0]}}

# Set nD transform
group.nd_transform = {"Time": {"offset": 200.0}}

# Remove nD transform
group.nd_transform = None

# Get composed world nD transform (walks parent chain)
world_nd = group.world_nd_transform
```

### 6.3 Validation

On write, the system validates:
- Dimension names exist in the scene's `Dimensions`
- Dimension names are NOT displayed dimensions
- Continuous/discrete dims get affine params (`scale`, `offset`)
- Categorical dims get `permutation` param
- Permutation is valid (correct length, each index once)
- Scale is non-zero
- No mixing of affine and permutation params on a single dimension

## 7. Hierarchical Composition

nD transforms compose through the parent chain, per-dimension:

### 7.1 Affine Composition (Continuous / Discrete)

```
parent: y = s_p * x + o_p
child:  y = s_c * x + o_c

composed: y = s_p * (s_c * x + o_c) + o_p
        = (s_p * s_c) * x + (s_p * o_c + o_p)
```

So: `composed_scale = parent_scale * child_scale`, `composed_offset = parent_scale * child_offset + parent_offset`

### 7.2 Permutation Composition (Categorical)

```
parent: p_p
child:  p_c

composed[i] = p_p[p_c[i]]
```

Child permutation applied first, then parent.

### 7.3 Missing Transforms

If a node in the chain has no `nd_transform` (or no entry for a specific dimension), it contributes the identity transform for that dimension:
- Affine identity: `{scale: 1.0, offset: 0.0}`
- Permutation identity: `[0, 1, 2, ..., n-1]`

## 8. Bounds Expansion

When `nd_transform` is present on a node, the scene-level position bounds for non-displayed dimensions are expanded to include the transformed range.

**For affine transforms:**
```python
original_min, original_max = node_bounds[dim]
transformed_min = scale * original_min + offset
transformed_max = scale * original_max + offset
# Handle negative scale (flips min/max)
if scale < 0:
    transformed_min, transformed_max = transformed_max, transformed_min
scene_bounds[dim] = union(scene_bounds[dim], (transformed_min, transformed_max))
```

**For categorical permutations:**
Bounds don't change — the range is still `[0, n_categories - 1]`.

**Status:** Implemented. During `finalize()`, the compiler walks the zarr tree, composes world nd_transforms for each leaf node, applies `apply_nd_transform_to_bounds()`, and stores the union of all world-space bounds as `position_bounds` in root attrs. Per-node bounds remain in local space. The viewer's `SceneDimsManager` uses these scene-level bounds as automatic slider ranges when `Dimension.range` is not explicitly set.

## 9. Viewer Implementation (TypeScript) — Inverse-Query Approach

### 9.1 Data Flow (Unchanged!)

```
load nD positions → slice/clip (raw coords) → project to 3D → apply 4x4 transform
```

The data flow is **unchanged**. Instead of transforming point data, we inverse-transform the **query** before it enters the pipeline.

### 9.2 Inverse-Query Design

The spatial index stores raw (untransformed) coordinates. The viewer's slice position is in "world" (transformed) space. To query correctly, we convert the query back to "local" space:

```typescript
// O(1) per dimension — transform the query, not the data
function invertNdTransformForQuery(
  slicePosition: number[],   // world space
  tolerance: number[],        // world space
  ndTransform: NdTransformMap,
  // Per-dimension metadata: `name` matches the ndTransform keys,
  // `discrete`/`step`/`range` drive the no-preimage rule (§9.2.1).
  dimensions: readonly {
    name?: string;
    discrete?: boolean;
    step?: number;
    range?: readonly [number, number];
  }[],
  displayDims: number[]
): { slicePosition: number[]; tolerance: number[]; noPreimage: boolean }
```

For each non-displayed dimension with a transform:
- **Affine** (`effective = scale * raw + offset`):
  - `local_pos = (world_pos - offset) / scale`
  - `local_tol = world_tol / |scale|`
- **Permutation** (index remapping):
  - Compute inverse permutation, remap slice index
  - Tolerance unchanged (categorical matching)

#### 9.2.1 The no-preimage rule (discrete dimensions)

The forward rule for discrete ordinals rounds (§4.1), so not every world value
is the image of a local one. Inverting `scale: 2` at world `T = 7` gives local
`3.5`, which is **no category at all** — `round(2k) = 7` has no integer
solution.

The inverse query alone cannot express that. Left unguarded, the per-element
membership window (a half-step, `|value − target| ≤ 0.5 × step`) admits both
local 3 and local 4, drawing two frames that belong to world 6 and world 8
while the slider reads 7; with `scale: 3` at `T = 7` (local `2.333`) it admits
local 2.

The test is the forward rule itself, not exact inverse-grid alignment — the two
agree only for integer `scale`/`offset`, and §11.3 blesses fractional scale.
`resolveDiscretePreimage` walks the `range[0] + k · step` local grid candidates
bracketing the exact inverse and keeps the one whose forward image rounds to
the queried world value on the same anchored grid:

| transform     | world | resolves to | why                              |
| ------------- | ----- | ----------- | -------------------------------- |
| `scale: 2`    | 8     | local 4     | `round(2·4) = 8`                 |
| `scale: 2`    | 7     | **none**    | `round(2·3)=6`, `round(2·4)=8`   |
| `scale: 1.2`  | 1     | local 1     | `round(1.2·1) = 1`               |
| `offset: 0.4` | *w*   | local *w*   | `round(w + 0.4) = w` — never dark |

On success the local slice position is **snapped** to that candidate, which also
removes the midpoint tie that caused the original double-draw. On failure
`invertNdTransformForQuery` reports **`noPreimage`**, which rides the derived
per-node `ViewState.noPreimage`, and each geometry's range query
(`queryVisiblePointRanges` / `queryVisibleSegmentRanges` /
`queryVisibleSplatRanges`) returns an empty range list, which every loader
already renders as "cleared". The guard sits ahead of both the no-spatial-index
load-all fallback and the `extend_to_all` short-circuit, since either would
otherwise pass every element to the membership gate.

Exemptions: categorical permutations (a bijection always has exactly one
preimage), and `extend_to_all` dimensions — keyed off the node's `extend_to_all`
**name list**, not the tolerance sentinel, because every Lines call site derives
with `applyPartialExtendTolerance: false` and so never carries it.

Known limitation: the local grid is taken to use the dimension's declared
`range[0]` anchor and `step` (both world-space quantities); no metadata describes
the local grid, and the downstream membership window makes the same assumption.

The `nd_transforms` demo (`demos/demo_nd_transforms.py`) is the visual
regression harness: one row per transform, markers that print their own local
index against a world ruler and cursor.

### 9.3 Where It's Applied

In `data/scene-loader/view-state/derive-node-view-state.ts`, centralized alongside the existing `extend_to_all` tolerance modification. Applied ONCE per node update, BEFORE passing viewState to the loader:

```typescript
// After the extend_to_all tolerance override:
const worldNdT = computeWorldNdTransform(sceneGraph, path);
if (hasOwnProperties(worldNdT) && derived.dimensions) {
  const inverted = invertNdTransformForQuery(
    derived.slicePosition, derived.tolerance,
    worldNdT, derived.dimensions, derived.displayDims
  );
  // `noPreimage` only rides the derived state when set (§9.2.1); otherwise
  // just the inverted position + tolerance are folded in.
  derived = inverted.noPreimage
    ? { ...derived, ...inverted }
    : { ...derived, slicePosition: inverted.slicePosition, tolerance: inverted.tolerance };
}
// Then pass to loader — the only loader-side change is the noPreimage guard
// at the top of each geometry's range query.
```

### 9.4 Advantages Over Per-Point Transform

| Aspect | Per-Point (rejected) | Inverse-Query (implemented) |
|--------|---------------------|---------------------------|
| Complexity | O(N * D_nd) per frame | O(D_nd) per frame |
| Data mutation | Modifies position arrays | No data mutation |
| Code changes | Every loader's internals | Scene-loader, plus a one-line no-preimage early-out per geometry range query (§9.2.1) |
| Cached data | Must copy before transform | Untouched |
| WASM | Would need changes | No changes needed |

### 9.5 `extend_to_all` Interaction

If a dimension has `extend_to_all`, its tolerance is already set to 1e10 (infinite). The inverse transform passes that sentinel through **unscaled**: `1e10 / |scale|` is *not* still effectively infinite once `|scale| > 10`, because every downstream extend check tests `tolerance >= 1e9` (`effective-radius-calculator`'s `isExtendToAll`, `calculateSpatialQueryTolerance`, `fallbackQueryTolerance`). Rescaling it would drop an extended dimension back into being sliced under ordinary unit-conversion scales — e.g. `{"scale": 1000}` (s → ms), the reverse direction of §4.1's flagship ms → s example. Only finite tolerances carry a meaningful world→local conversion.

The no-preimage rule (§9.2.1) exempts extended dimensions from the node's `extend_to_all` **name list** rather than from the sentinel, because a Lines node never carries the sentinel at derive time — every lines call site uses `applyPartialExtendTolerance: false`.

### 9.6 Hierarchical Composition

**Python side**: Full hierarchical composition is implemented via `world_nd_transform` property, which walks the parent chain and composes all nd_transforms.

**TypeScript viewer side**: Parent nd_transforms ARE automatically composed via `computeWorldNdTransform()` in `scene-loader.ts`. This function traverses the scene graph and composes nd_transforms from parent to child. This means:
- `nd_transform` on a group propagates correctly to all child data nodes in the viewer
- `nd_transform` on a points/lines/gsplats/mesh node applies correctly in the viewer
- Composition follows the same rules as the Python side (affine composition for continuous/discrete, permutation composition for categorical)

## 10. Performance Analysis

| Operation | Cost | When |
|-----------|------|------|
| nD transform (per-point) | O(N * D_nd) multiply+add | Every slice position change |
| Affine composition | O(D_nd) per node in chain | Once on scene load |
| Permutation composition | O(D_cat * n_categories) | Once on scene load |
| Bounds recomputation | O(D_nd) per node | Once on write |

Where:
- N = number of points
- D_nd = number of non-displayed dimensions with transforms
- D_cat = number of categorical dimensions with permutations

The per-point cost is dominated by the existing slicing pass. The nD transform adds ~1-2 FMAs per non-displayed dimension per point, which is negligible.

## 11. Edge Cases

### 11.1 Scale = 0
Rejected at validation time. Zero scale is not a supported transform.

### 11.2 Negative Scale
Flips the dimension. Valid — reverses the ordering. Bounds computation handles min/max swap.

### 11.3 Fractional Scale on Discrete Dimensions
`scale=0.5` on a discrete dimension means indices 0,1,2,3 become 0,1,1,2 (after rounding). Valid but lossy — emit a warning about information loss.

### 11.4 Permutation on Non-Categorical Dimension
Rejected at validation time. Permutations only apply to categorical dimensions.

### 11.5 Affine on Categorical Dimension
Rejected at validation time. Affine transforms only apply to continuous/discrete ordinal dimensions.

### 11.6 Cyclic Dimensions
For cyclic dimensions (e.g., angle wrapping at 360), the transformed value should be wrapped: `effective_value = (scale * value + offset) % cycle_length`. This requires the dimension's range to define the cycle period.

## 12. Backward Compatibility

- Scenes without `nd_transform` attribute behave identically to today
- The `nd_transform` attribute is optional on all node types
- Older viewers that don't understand `nd_transform` will ignore it (unknown attrs are silently skipped in the viewer)
- No changes to the 4x4 `transform` attribute or its processing

## 13. Future Extensions

- **Cross-dimension mixing within the same domain:** A 2x2 rotation between two continuous non-displayed dims (e.g., rotate in "time × depth" space). Would require a sub-matrix for groups of continuous dims.
- **Animated nD transforms:** Time-varying nd_transform for progressive dataset alignment.
- **Inverse nD transform:** For the reader API, expose the inverse to convert world coordinates back to local.
