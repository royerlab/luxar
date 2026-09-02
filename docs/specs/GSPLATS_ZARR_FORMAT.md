# luxar.gsplats.io - Technical Specification

**Version**: 3.4.0
**Last Updated**: 2026-08-13

> For a version-policy and migration summary that contrasts this format with the scene container, see [Formats & Migration](../guides/user/FORMAT_AND_MIGRATION.md).

> **Cross-language contract:** the format *vocabulary* shared by the Python
> writer and the TypeScript viewer — format versions, encoding-scheme names,
> node types/kinds, and canonical attr/array keys — is single-sourced in
> [`format-contract/contract.yaml`](../../format-contract/contract.yaml). Edit
> that file and run `make gen-contract` to regenerate the Python
> (`typing_utils/_format_contract.py`) and TypeScript
> (`types/format-contract.ts`) projections; `hatch run check-contract` gates
> against drift. This document remains the prose reference; the YAML is the
> machine-checked source of truth for the values.

## Purpose

The `gsplats.io` package provides I/O operations for persisting and loading Gaussian splat data in a dedicated zarr format (`.gsplats.zarr`). This enables efficient storage, compression, and retrieval of fitted Gaussian splat results.

**Related Specifications**:
- [Luxar Zarr Format](../guides/user/LUXAR_ZARR_FORMAT.md) — Scene-level format including GSplats nodes
- [GSplats Dimension Mapping](GSPLATS_DIMENSION_MAPPING.md) — Dimension mapping for nD scenes
- [nD Transforms](../guides/specs/ND_TRANSFORMS_SPEC.md) — nD navigation and transforms

> **Scope note (v3.0).** A `.gsplats.zarr` is a **detached scene-node subtree** —
> structurally identical to the gsplats node a Luxar scene already contains, just
> with the file root standing in for the scene root. The file root **is** the
> node. There is no separate matrix format: the same three nestable primitives
> the scene uses (a gsplats **leaf** with an optional `additive_<i>/` ladder; a
> `kind=lod` Group; a `kind=partition` Group) express every LOD / partition
> combination.
>
> Consequences of the unification:
> - **The viewer consumes a `.gsplats.zarr` directly** — `?src=<file>.gsplats.zarr`
>   loads it as a scene root and frames on its `position_bounds`.
> - **Embedding into a scene is a graft, not a lowering.** A multi-substitutive
>   pyramid is *already* a `kind=lod` Group on disk; the scene path no longer
>   "auto-lowers" a matrix — it writes the same node tree, through the same
>   serializer (`io/_compiler/gsplat_tree.write_gsplat_node`), so a standalone
>   leaf / ladder / kind=lod / kind=partition subtree is byte-identical to the
>   scene one. Pass `lod_group=False` to `Group.add_gsplats_from_data` to collapse
>   to the finest substitutive level at embed time.

---

## Use Cases

1. **Save/Load fitted results** - Persist expensive fitting results for later use
2. **Lightweight rendering** - Load only splat data for visualization
3. **Provenance tracking** - Record what image/parameters produced the splats
4. **LOD / partition storage** - Single self-describing node tree carrying any of:
   - one splat set (a bare leaf)
   - an additive ladder (one leaf with `additive_<i>/` prefix-sum sub-LODs)
   - a substitutive hierarchy (a `kind=lod` group of leaves, coarsest→finest)
   - a full pyramid (a `kind=lod` group whose children are additive-ladder leaves)
   - a spatial partition (a `kind=partition` group of `part_<i>/` leaves)
   - any nesting of the above (e.g. a partition of LOD groups)
5. **Direct viewing** - The viewer opens the file as a scene root (`?src=…`)
6. **Future: Checkpoint/Resume** - Pause and resume fitting (deferred)

## Core Data Structure

Each Gaussian splat is parameterized by:

| Field | Shape | Dtype | Semantic Type | Description |
|-------|-------|-------|---------------|-------------|
| `centers` | (N, d) | uint16 / uint8 (LUT) / float32 | COORDINATE | Splat center positions (not broadcastable). AUTO/MEMORY: uint16 per-axis fixed-point (`linear_perchannel_u16`), decoded to float32, with a *gridded* axis snapped so it round-trips exactly — or `lut_uint8` (exact, values stored verbatim) when few enough distinct values make a LUT eligible; PRECISION / large-extent / able to displace splats past their own σ on an axis that is neither gridded nor LUT-eligible (see the sigma rail below): float32 |
| `amplitudes` | (N,) or (1,) | uint8/uint16/float32 | POSITIVE_SCALAR | Non-negative intensity |
| `cholesky_factors_diag` | (N, d) or (1, d) | uint8/uint16/float32 | CHOLESKY_DIAG | Diagonal of L (positive, scale-like) |
| `cholesky_factors_offdiag` | (N, d*(d-1)/2) or (1, …) | uint8/uint16/float32 | CHOLESKY_OFFDIAG | Strictly-lower elements of L (signed); absent when d=1 |
| `colors` | (N, 3\|4) or (1, 3\|4) | uint8/uint16/float32 | COLOR | RGB or RGBA colors (optional); SDR → `rgb_uint8`; HDR → `geolog_perchannel_u16` (AUTO; u8 under MEMORY, float32 under PRECISION); absent if not present. The optional 4th channel is per-splat opacity α ∈ [0, 1] (per-element opacity: every blending mode scales a splat's contribution by α; volumetric maps it into optical depth w = −ln(1−α) — see VOLUMETRIC_BLENDING_SPEC.md §5.4.1). α is never HDR. No format-version bump: readers key off the array shape, and codecs are channel-agnostic. |
| `label_ids` | (N,) or (1,) | uint8/uint16/uint32/uint64 | INDEX | Optional categorical class id per splat; a constant channel may use broadcast encoding. Stored as the smallest exact unsigned integer with LUT and lossy quantization disabled. `label_vocabulary` maps every stored id to its name as a JSON object keyed by the id's decimal string form. |

`label_ids` is an optional leaf channel and does not bump the format version.
`label_ids` / `label_vocabulary` is the exact categorical channel, distinct
from the per-element string `labels` / `has_labels` tooltip channel; both may
coexist on one node.
Its vocabulary is explicit rather than inferred: every observed id must have a
name, while a filtered subset may retain unused vocabulary entries so ids keep
the same meaning across related leaves. Any row permutation or subset operation
must apply the identical operation to `label_ids`. Concatenation, partition
flattening, and other operations that combine rows across leaves carry the channel
only when all contributing leaves have identical presence and vocabularies;
mixed presence or different vocabularies is an error. Per-leaf rewrites such as
re-encoding, migration, restriding, optimisation, and batch merge carry each
leaf's vocabulary unchanged without comparing leaves, so vocabularies may differ
between leaves in one store. Merge-based coarsening (`lod levels`, `overview`,
`adaptive`, and merge decimation) is refused because there is no defined class id
for a splat synthesized from differently labeled inputs. Prefix/additive LOD is
safe because it only reorders or subsets existing splats. Exporters without a
vocabulary-bearing categorical field must refuse the channel rather than drop it.
The vocabulary is duplicated on every leaf and additive rung and therefore lands
in consolidated metadata; keep it to the small id set actually in use. If that
cost becomes material, the format should add a shared subtree-level vocabulary.
This refusal is a writer-side rule, not an on-disk capability stamp, and does not
bump the format version. Older Luxar versions therefore cannot distinguish a
labeled store before rewriting it and may silently drop the channel in operations
such as flattening or LOD construction; use a label-aware version for all edits.

**Note**: Since **v3.1** the packed lower-triangular factor L (where Σ = LLᵀ) is
stored as **two arrays** — the diagonal (`cholesky_factors_diag`) and the
strictly-lower off-diagonal (`cholesky_factors_offdiag`) — so each can be encoded
independently. Each is **differentially quantized** with a generic per-channel
scheme (one scale per column): the diagonal (positive, wide range) with
**per-channel log** (`log_perchannel_u8`/`u16`), the off-diagonal (signed,
zero-centred) with **per-channel signed-log** (`signed_log_perchannel_u8`/`u16`).
Encoding mode picks the bit depth: PRECISION→float32; **AUTO→uint8 with an
encode-time certificate** — the writer measures the actual Σ = L·Lᵀ reconstruction
error (p95 relative Frobenius) and escalates to uint16 only when it exceeds 0.05
(e.g. merged stores whose σ columns span many decades) — with a final float32 rung
should even uint16 fail (practically unreachable on real fits; degenerate/tiny
groups, e.g. single-splat sub-LODs, may decline certification and store float32
directly) — recording the measurement in
each array's `encoding.certificate`; MEMORY→uint8 unconditionally. uint8 is visually
lossless on real fits (94.5 dB vs the float32 render, ~46 dB below the fit-error
floor; 2.48 B/splat compressed vs 8.25 at uint16). Per-array `encoding` metadata
carries the per-column scales (`col_lo`/`col_hi`). Since 2026-07 the writer emits
`zero_level: true`: scales are anchored at each column's **nonzero** min/max and
code 0 is **reserved for exact zeros** (same layout as `geolog_scalar`), so
exactly-zero entries — e.g. the off-diagonal of an axis-aligned splat — decode to
exactly 0 (codes `1..2^bits-1` span `[col_lo, col_hi]`, denominator `2^bits-2`);
arrays without the flag keep the legacy all-levels, zero-anchored decode.
Readers decode to float32 and
recombine into the packed row-major form `[L00, L10, L11, L20, L21, L22, …]`
(for d=3) immediately on load; everything above the storage layer sees the single
packed (N, k) float32 array, k = d*(d+1)/2. **v3.0** files store a single packed
`cholesky_factors` array and are still read (the loaders fall back when no split
is present).

**Semantic Types**: Each field maps to an encoding semantic type (see `luxar.encoding.SemanticType`). This determines valid encodings and quantization options for each array.

### Broadcasting Convention

Broadcasting uses the standard `luxar.encoding` format. When all elements share the same value, the array is stored with shape `(1,)` or `(1, d)` with encoding metadata:

```json
// amplitudes/.zattrs - all splats have amplitude=1.0
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

The `n_splats` attribute on a leaf's `.zattrs` always reflects the true count (N), regardless of broadcasting.

---

## Format Versions

The current format is **v3.4**, a node tree (§ "On-disk grammar"). It differs
from **v3.3** only in the `kind=lod` selector: the group `selector` attr gains
the value `"screen-area"` (what every DERIVED ladder now stamps), under which
the per-child `coverage_fraction` is a literal **screen-area fraction**
(projected bbox rect area / viewport area; occupancy halving — whole-object
finest `0.5`, partition tile `1.0` — see the `kind=lod` section). Stores with
`selector: "coverage"` keep the legacy diagonal-metric units and are read and
round-tripped unchanged, so every v3.3 store is also a valid v3.4 store.
**v3.3** differs
from **v3.2** only in allowing quantized code arrays (coordinates, Cholesky
halves, amplitudes) to carry the optional `luxar_delta_v1` zarr v2 **filter**
(columnar per-chunk delta+zigzag — § "The `luxar_delta_v1` delta filter");
a v3.3 store without the filter is byte-identical to v3.2. **v3.2** differs
from **v3.1** only in the `kind=lod` selector attrs: the group `selector` value
`pixel_size` and the per-child `min_pixel_size` (absolute pixels) are renamed
to `coverage` / `coverage_fraction` (viewport-relative `sqrt(N_i/N_finest)` in
`[0, 1]`, strictly ascending coarsest→finest, finest `1.0`; the bound later
widened to `MAX_COVERAGE_FRACTION` = 4.0 when the diagonal metric was rescaled
×4, and is superseded for derived ladders by v3.4's `screen-area` anchors above
— see the `kind=lod` section).
**v3.1** differs
from **v3.0** only in storing the Cholesky factors as two arrays
(`cholesky_factors_diag` + `cholesky_factors_offdiag`) instead of a single packed
`cholesky_factors`. v3.0 / v3.1 files are still read transparently (the web
viewer auto-adapts the legacy lod selector attrs, with a warning); upgrade them
with ``luxar gsplat migrate-format`` — which also converts the earlier versions
(v1.0 single-LOD; v1.1 additive multi-LOD; the pre-v2.0 substitutive directory +
manifest.json; and the interim v2.0 `substitutive_<s>/additive_<a>/` matrix)
that are no longer read by the runtime.

A v3.x file is one of (each freely nestable):

| Shape            | On-disk form                                                       |
|------------------|--------------------------------------------------------------------|
| single leaf      | `centers`/`amplitudes`/`cholesky_factors_diag`(+`_offdiag`)(/`colors`) at the root |
| additive ladder  | `additive_<i>/` sub-LOD subgroups + `n_additive_sublods`           |
| substitutive lod | `type=group, kind=lod`; `child_<i>/` coarsest→finest + `coverage_fraction` |
| full pyramid     | a `kind=lod` group whose `child_<i>/` are additive-ladder leaves   |
| partition        | `type=group, kind=partition`; `part_<i>/` + `max_elements`         |

Every node carries `position_bounds`; the root additionally carries
`format_version:"3.4"`, `format_type:"gsplats_zarr"`, `timestamp`,
`luxar_gsplats_version`, and `content_hash` (a metadata-only xxhash64 over the
tree's group attrs plus, per array, its name, shape, chunk shape, shard shape,
dtype, codec ids and own attrs, plus each child group's name alongside its own
digest). Storage layout counts as identity because the viewer caches encoded
chunks keyed by chunk index, so a re-chunked store — or one renamed, or one
compressed with a different codec — must not share its input's hash; the
per-array attrs count because that is where the `encoding` document lives, which
is what turns the stored ints back into scientific values. The hash is distinct
per save because the per-save `timestamp` folds in — the web viewer's persistent
cache compares it to invalidate when a file is regenerated in place. The
historical `[N, M_i]` matrix is just the "full pyramid" shape expressed as a node
tree.

---

## Zarr Structure (v3.x — node tree)

The file root IS the node. The same three primitives nest arbitrarily:

### Shape 1 — bare leaf (single splat set)

```
fitted.gsplats.zarr/
├── .zattrs           # type: "gsplats", n_splats, ndim, has_colors, has_label_ids,
│                     # ordering, ordering_min/max/bits, slice_dims, ordering_dims,
│                     # chunk_size, amplitude_range, amplitude_data_range,
│                     # amplitude_mass, amplitude_mass_weighted_mean,
│                     # label_vocabulary? (decimal-string id keys), center_bounds,
│                     # position_bounds, truncation_radius,
│                     # opacity, absorption, gamma, intensity, offset, blending_mode?,
│                     # format_version: "3.4", format_type: "gsplats_zarr",
│                     # timestamp, luxar_gsplats_version, description?
├── .zmetadata        # Consolidated metadata for fast loading
├── centers                   # (N, d) uint16 (AUTO; lut_uint8 when a LUT is eligible; float32 if an axis extent ≥ 2¹⁶, or if a neither-gridded-nor-LUT axis's grid is too coarse for the splats' σ) / float32 (PRECISION), spatially ordered
├── amplitudes                # (N,) uint8/uint16 (AUTO) / float32 (PRECISION)
├── cholesky_factors_diag     # (N, d) uint8 (AUTO, certified — escalates to uint16 if the covariance certificate fails) / float32 (PRECISION)  (diagonal of L)
├── cholesky_factors_offdiag  # (N, d*(d-1)/2) uint8 (AUTO, certified as above) / float32 (PRECISION) (off-diagonal; absent if d=1)
├── colors            # (N, 3) uint8/uint16 (AUTO) / float32 (PRECISION)  (optional)
├── label_ids         # (N,) or broadcast (1,) smallest exact uint (optional; never LUT/quantized)
├── chunk_bounds      # (num_chunks, d, 2) float32  (when ordering ≠ "none")
├── fitting/          # Optimization info (optional)
│   ├── .zattrs       # time_seconds, iterations, converged, psnr_db, …
│   └── config/.zattrs  # Fitter hyperparameters
├── pipeline/         # Reduction/topology stats (optional)
│   └── .zattrs       # lod_kind, method, compression_factor, coverage_inflation, refine, …
└── provenance/       # Image lineage (optional)
    └── .zattrs       # source_file, shape, dtype, normalization
```

### Shape 2 — additive ladder (prefix-sum LODs over the same N splats)

```
fitted.gsplats.zarr/
├── .zattrs           # type: "gsplats", n_splats (total), ndim, n_additive_sublods,
│                     # position_bounds, format_version: "3.4", …
├── additive_0/       # Coarsest additive sub-LOD (index 0 = coarsest)
│   ├── centers, amplitudes, cholesky_factors_diag, cholesky_factors_offdiag, colors?, chunk_bounds?
│   └── .zattrs       # type: "gsplats", n_splats, ndim, ordering, lod_stats?, …
├── additive_1/       # Only present when n_additive_sublods > 1
│   └── …
└── additive_{M-1}/   # Finest sub-LOD
    └── …
```

`n_additive_sublods` on the parent leaf group declares the ladder depth.
Sub-LOD groups carry lightweight attrs (no rendering defaults).

### Shape 3 — substitutive LOD (`kind=lod` group)

```
fitted.gsplats.zarr/
├── .zattrs           # type: "group", kind: "lod", selector: "screen-area",
│                     # default_level: <int>, display_type: "gsplats",
│                     # position_bounds, format_version: "3.4", …
├── child_0/          # Coarsest child (child_0 = coarsest on disk)
│   ├── .zattrs       # coverage_fraction: 0.0, compression_factor, level_index, …
│   ├── centers, amplitudes, cholesky_factors_diag, cholesky_factors_offdiag, colors?, chunk_bounds?
│   └── …
├── child_1/
│   ├── .zattrs       # coverage_fraction: <0..1>, …
│   └── …
└── child_{N-1}/      # Finest child (coverage_fraction: 0.5, or 1.0 when the
    │                 #   ladder is bound to a spatial partition — see below)
    └── …
```

Children are written **coarsest→finest** on disk (child_0 = coarsest,
child_{N-1} = finest) — the SAME order the in-memory tree
(`GSplatLodGroup.children`) uses, so the serializer writes them straight through
with no reversal. The on-disk `default_level` is `0` (the coarsest child) — the
viewer's progressive-load hint (render cheap first, then refine). This is a
distinct concept from the data-model default (the finest level the `.centers`
accessor returns); they are deliberately decoupled, so the writer stamps
`default_level: 0` independently. Each child carries `coverage_fraction`,
strictly ascending coarsest→finest with the coarsest child always `0.0`; the
group's `selector` attr names the UNITS. Under `selector: "screen-area"` (what
every derived ladder stamps since v3.4) a threshold is a literal screen-area
fraction — the node's projected bbox rect area over the viewport area — and
writers derive the ladder by SCREEN-OCCUPANCY HALVING: authored detail is
meant to be viewed full screen, so a whole-object ladder anchors its FINEST
level at `0.5` (full detail while the node occupies at least half the screen)
and each coarser level halves the threshold (`…, 1/8, 1/4, 1/2`) —
deliberately independent of per-level element counts (a count ratio is blind
to element size, overlap, and intent; the retired derivation
`sqrt(N_i/N_finest)` held the finest level until the object was far away on
dense sub-pixel data). The metric is built from NDC fractions, so selection is
independent of viewport resolution and size — though occupancy still moves with
viewport ASPECT, since the camera framing is fitted to one screen axis.

Note the semantics this deliberately REVISES: under the retired diagonal
metric, a fitted high-aspect object read HIGH (a rod's diagonal ≈ its
length) and dense elongated content could render its most expensive level
across the whole usable zoom range. Under occupancy the same object reads
its literal screen share — a fitted full-width, quarter-height object opens
at 25% occupancy, one level below finest on the standard 4-level ladder,
with full detail one modest zoom away. Authors who want a high-aspect
object finest-at-opening use an explicit `coverage_fractions=[...]` list
(legacy units).

Two refinements are NORMATIVE parts of the `screen-area` metric (they decide
which end of a ladder renders, so consumers must agree on them):

* **Visible occupancy.** The projected rect is intersected with the viewport
  before the area is taken; a rect with no viewport overlap on either axis
  reads exactly `0` (coarsest), and full coverage tops out at exactly `1.0`
  (thresholds are satisfied inclusively, `threshold <= metric`). Under a
  PERSPECTIVE camera, a node whose bounds reach the camera's near plane has
  no meaningful projection (the homogeneous divide degenerates), so the
  metric saturates to the finest level — the same near-plane guard the
  legacy diagonal metric applies. An ORTHOGRAPHIC projection never
  degenerates (`w` stays 1), so no saturation applies and the plain clipped
  metric is used directly: a camera inside a large node still reads full
  coverage naturally (its rect spans the viewport), and both selectors
  behave identically here by design.
* **Degenerate (lower-dimensional) content.** A node whose projected bounds
  are (near-)zero-thickness — an axis-aligned straight polyline, a planar
  dataset viewed edge-on — has area ~0 no matter how much screen it spans.
  When the RAW (pre-clip) thin half-extent is at/below a sub-pixel floor
  `ε` (implementations should use `ε` ≈ one pixel of the viewport axis; the
  reference viewer uses 1e-3 of the axis), the metric is
  `max(area, clippedSpan × (1 − rawThin/ε))` — a continuous ramp from the
  clipped LINEAR span at zero thickness down to the plain area product at the
  floor, so a full-width line reads `1.0` (its faithful occupancy) instead of
  being pinned to the coarsest level, an edge-on rotation crosses no
  discontinuity, and a wide node panned to a thin visible sliver still reads
  its true (tiny) visible area because the gate is on the content's raw
  thinness, not the clipped one.

Both selectors size a child from its optional nD `lod_bounds` attribute when
present:

```json
{"lod_bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]}}
```

The arrays MUST be finite, ordered (`min[i] <= max[i]`), have the same length
and axis order as `position_bounds`, and be contained within that complete
bound. A bound that is not contained is rejected and the child falls back to
`position_bounds`. This is a producer-chosen robust extent — for example
percentile bounds that exclude a sparse tail — and affects only the selector
metric. A robust bound MUST NOT make a node select a finer level than its
complete `position_bounds`; the screen-area selector clamps the robust metric
to the complete-bound metric to preserve that invariant. Frustum gating,
eviction, root framing, clipping, and scene ranges
continue to use the complete `position_bounds`, so excluded outliers remain
part of the drawable geometry. A missing or malformed `lod_bounds` falls back
to that child's `position_bounds`; producers SHOULD stamp every child in a
ladder when they intend one consistent robust extent. Any operation that
decimates, culls, or filters a child MUST recompute or remove its `lod_bounds`.
Producer-side authoring policy is tracked in #1655. This optional metadata is
backward-compatible and does not change the v3.4 format version.

Under the legacy `selector: "coverage"`
(older stores; never written for derived ladders since v3.4) the thresholds
are diagonal-metric units in `[0, 4]`: the viewer compares them against the
projected bbox diagonal over `FILL_FACTOR=0.5 ×` the fitted screen axis
(`min(viewport.width, viewport.height)` — the extent the camera framing
actually fits; see `scene/lod-group-registry.ts`). (Same
contract as `docs/guides/user/LUXAR_ZARR_FORMAT.md`.)

**Which anchor the finest child gets.** A **whole-object** ladder (the `levels`
recipe, and any `kind=lod` group whose levels are alternative renderings of the
whole node) anchors its finest at `0.5` (half the screen area). A ladder
bound to a **spatial partition** anchors its finest at `1.0` — the tile alone
occupying the whole screen — because a tile's projected rect is intrinsically
a fraction of the whole object's. That covers the `adaptive`
recipe (one `kind=lod` group per tile) and the `overview` recipe (whose fine
child *is* a `kind=partition`, reached by zooming in). See
`partitioned_coverage_fractions` in `luxar/core/group/lod/group.py`. Any node shape (bare leaf, additive
ladder) is valid as a child.

### Shape 4 — spatial partition (`kind=partition` group)

```
fitted.gsplats.zarr/
├── .zattrs           # type: "group", kind: "partition", display_type: "gsplats",
│                     # max_elements: <int>, position_bounds, bsp_tree?, format_version: "3.4", …
├── part_0/           # BSP part 0 (any node shape valid per part)
│   ├── .zattrs       # position_bounds (per-part bounds for frustum culling), child_index
│   └── centers, amplitudes, cholesky_factors_diag, cholesky_factors_offdiag, colors?, chunk_bounds?
├── part_1/
│   └── …
└── part_{P-1}/
    └── …
```

The viewer renders every part that intersects the camera frustum simultaneously.
Its frustum-only partition selector also gates fetch and eviction for off-screen
parts; it does not perform LOD substitution. The partition writer uses recursive
BSP (`median`, `midpoint`, or `sah` rule) to build spatially balanced parts.

**`bsp_tree` (optional).** When the parts came from a recursive axis-aligned
decomposition, the root additionally carries that decomposition's split-plane
record as a nested dict. Written by every partition producer that has one:
the `to_spatial_partition` path (the `tiles`/`adaptive` recipes and the
`gsplat partition` command), a content-tiled fit (the planner's own box
recursion), a uniform-tiled fit and the batch-fit streaming merge:

```json
{ "axis": 0, "split": 12.5,
  "left":  { "axis": 2, "split": -3.0, "left": {"part": 0}, "right": {"part": 1} },
  "right": { "part": 2 } }
```

An internal node holds the split `axis` (a position-column index; gsplat
producers currently split only the first three center dims) and `split`
coordinate (in the centers' own space), with `left` = the side where
`coord < split` and `right` = `coord >= split`. A leaf holds `{"part": i}`,
referencing `part_<i>` (the same index as its `child_index`), numbered in
left-first DFS order.

Because these are BSP cells, a viewer can order the parts **exactly**
back-to-front (painter's algorithm, Fuchs–Kedem–Naylor): recurse the far side
of each split first — correct for any camera pose, including inside the volume.
This matters for order-dependent compositing (`normal`/alpha-over and
`volumetric`); it is inert for additive/commutative rendering.

Exactness holds wherever the parts are genuinely **disjoint** — every producer
above except one. A **uniform**-tiled fit is the exception: its tiles are
apodized and each part keeps its overlap band, so neighbouring parts really do
share space and no exact part order exists. Its cuts are the midplanes of those
bands, which confines misordering to the band rather than letting whole tiles
swap; treat such a tree as a good approximation, not a guarantee.

A partition may still omit `bsp_tree` — a pre-2026.7 store, a decomposition that
is not axis-aligned, or a transform that could not carry the planes (see below) —
and a viewer may reject a stored tree that fails structural or split validation.
The viewer then falls back to a per-part centroid-distance heuristic, which is
*not* a valid painter's order: it flips discretely as the camera moves and shows
as popping at the seams between parts.

**Keeping it valid.** `split` is a coordinate in the centers' own space, so any
tool that moves centers must map the tree through the same affine or drop it —
a stale tree is worse than none, since it still yields a plausible permutation
and so degrades the ordering silently. Translation, per-axis scale and
quarter-turn rotations map (a reflection also swaps each node's `left`/`right`);
an arbitrary rotation shears the cells out of axis-alignment and cannot be
represented, so `gsplat transform` drops the tree and says so. Likewise, a tool
that DROPS parts must renumber the surviving leaves to the new `child_index`
values — the leaf labels and the written part indices are the same numbering.

### Shape 5 — full pyramid (substitutive × additive, nested)

A `kind=lod` group whose children are additive-ladder leaves combines both axes:

```
fitted.gsplats.zarr/
├── .zattrs           # type: "group", kind: "lod", …
├── child_0/          # Coarsest substitutive level — additive ladder
│   ├── .zattrs       # type: "gsplats", n_additive_sublods, …
│   ├── additive_0/
│   └── additive_{M-1}/
└── child_{N-1}/      # Finest substitutive level — additive ladder
    ├── additive_0/
    └── additive_{M-1}/
```

Partitions of LOD groups (`kind=partition` whose parts are `kind=lod` nodes)
are also valid and nest in the same way.

### Root Attributes (.zattrs)

The root carries both the node-type attrs (stamped by the shared walker) and
the self-identifying file header (stamped by `write_gsplats_tree`):

```json
{
  "format_version": "3.4",
  "format_type": "gsplats_zarr",
  "timestamp": "2026-06-09T10:00:00Z",
  "luxar_gsplats_version": "X.Y.Z",
  "description": "Optional user description"
}
```

For a bare-leaf root, the node attrs (`type`, `n_splats`, `ndim`, `ordering`,
`position_bounds`, rendering defaults) live alongside these header keys on the
same `.zattrs`. For a group root (`kind=lod` or `kind=partition`), the node
attrs are `type`, `kind`, `selector`, `default_level`, `display_type`,
`position_bounds`, and any group-level meta.

### Per-Leaf Splat Attributes (`.zattrs` on a leaf group)

```json
{
  "type": "gsplats",
  "n_splats": 10000,
  "ndim": 3,
  "has_colors": true,
  "truncation_radius": 2.75,
  "ordering": "hilbert",
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [256.0, 256.0, 128.0],
  "ordering_bits_per_dim": 21,
  "slice_dims": [],
  "ordering_dims": [0, 1, 2],
  "chunk_size": 2048,
  "amplitude_range": {"min": 0.01, "max": 1.5},
  "amplitude_data_range": [0.01, 1.5],
  "amplitude_mass": 8421.7,
  "amplitude_mass_weighted_mean": 0.32,
  "center_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  },
  "position_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  },
  "opacity": 1.0,
  "absorption": 1.0,
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0
}
```

`blending_mode` (`normal` / `additive` / `max` / `opaque` / `luminous` /
`volumetric`) is written only when explicitly set — unset nodes inherit from the nearest
ancestor that sets it (viewer default: `additive`). No default is stamped:
blending has no identity value, so a stamped default would shadow
ancestor-set modes.

`layer_order` — the authored cross-layer draw order, an integer where **higher =
nearer the camera = drawn later** — follows the identical rule for an identical
reason: written only when explicitly set, inherited nearest-setter-wins, and with
**no default stamped**. Here the absence carries meaning rather than merely
avoiding a shadow: an unset level means "use the viewer's inferred containment
ordering", so a stamped 0 would be indistinguishable from an authored one and
would silently disable that inference everywhere. It is refused outright inside a
`kind=partition` / `kind=lod` group, where it would respectively destroy the
exact BSP part order or be inert — set it on the wrapper, which is the layer.
See `docs/guides/specs/LAYER_ORDER_SPEC.md`.

**`slice_dims` is read, not just recorded**: besides describing the compound
ordering, the barrier (categorical) column indices it lists are the set
`compute_chunk_bounds_gsplats` gave the fixed epsilon pad plus any encoder
coordinate round-trip slack instead of the `truncation_radius · σ` expansion —
so the viewer's gsplats loader now CONSUMES it,
to classify each hidden dimension's chunk-fetch tolerance instead of inferring
barrier-ness from the scene's `discrete` flags
(`data/loaders/spatial-query/tolerance-computer.ts`). The on-disk format is
unchanged; a store that stamps no `slice_dims` falls back to the old inference.

**Bounds clarification**: `center_bounds` records the tight center AABB;
`position_bounds` is the same value (centers only — chunk bounds widen per-chunk
by the ellipsoidal extent). Encoding metadata on each array carries tighter
per-array quantization bounds.

Those encoding bounds are leaf-local. Writing the same source splats as one
AUTO leaf and as an AUTO partition therefore quantizes each part against
different ranges and does not promise identical decoded float32 values. The
window harmonization below changes only the display range, not those encoding
bounds; use PRECISION when a structural rewrite must preserve decoded values.

**Amplitude ranges**: `amplitude_range` (`{"min", "max"}` dict) is the
metadata bounds record and holds the true min/max of the original
(pre-quantization) amplitudes. `amplitude_data_range` (`[min, max]` list,
written alongside it whenever amplitudes are given as a non-empty array — a
scalar amplitude skips it) mirrors the Points/Lines `color_data_range`
convention and is the viewer's colormap **window**, seeding the layer
display-range controls. Each writer first derives it per node as
`[min(a), p99.9(a)]` (gsplat amplitudes are heavily right-skewed, so a `[0, max]`
window would map ~99% of splats to near-black), and then **finalize HARMONIZES
it across each gsplat structure**.

> **`amplitude_data_range` is a colormap window, NOT a render divisor.** The
> viewer feeds it to `uScalarMin`/`uScalarScale`, which the shader uses only as
> `t = clamp((A - min) * scale, 0, 1)` — a LUT index, clamped to `[0, 1]`. It
> selects a *colour*. It cannot scale brightness, and nothing downstream divides
> the amplitude by it. Emitted radiance and, under `volumetric`, optical depth
> (`tau = absorption * opacity * intensity`, `intensity ∝ A`) are both **linear
> in the raw stored amplitude** and are not windowed by anything.
>
> **Consequence — amplitudes must be normalised at scene insertion.** Fitted
> archives store amplitudes in raw source units: the fitter multiplies its
> `[0,1]` working copy back out by the volume's intensity range, so a fit from a
> uint16 detector stack carries detector counts. An amplitude of 800 emits 1000x
> the radiance of 0.8 and saturates `1 - exp(-tau)` into an opaque shell, and no
> viewer control can compensate — the only remaining lever is `opacity`, which
> would have to carry a ~1/500 factor on a `[0,1]` control. `add_gsplats_from_data`,
> `add_gsplats_from_file`, `add_gsplats_from_volume` and the graft path therefore
> normalise by default (robust p99.9 -> 1.0, **one factor for the whole
> structure**), recording it as `amplitude_normalization_factor`; pass
> `normalize_amplitudes=False` to opt out. A child inserted directly into a
> `kind=lod` or `kind=partition` group defaults to no normalisation because its
> exposure must stay shared with its siblings.
> The factor must be shared across every substitutive level and additive rung —
> a per-level factor scales the levels against each other and the brightness pops
> at every LOD switch. See `core/group/gsplats_pipeline/amplitude_norm.py`.

**Window harmonization (#1691)**. A per-node window is right for one flat leaf
and wrong for a multi-node structure: on a `kind=lod` ladder a coarse level's
merged representatives carry the same mass in far fewer splats, so its p99.9
lands ~4.5x above the finest level's and the object re-tones and pops in
brightness at every LOD switch; adjacent `kind=partition` tiles of one object
were measured windowed ~100x apart. So for every maximal `kind in {lod,
partition}` group holding gsplats, one reference window is taken and handed
down:

- **LOD levels** take the reference window scaled by
  `child.amplitude_mass_weighted_mean / reference.amplitude_mass_weighted_mean` —
  the mass-weighted amplitude ratio measures exactly the representation change a
  substitutive reduction makes (measured luminance ratio vs the finest level:
  1.04 / 1.01 / 1.00, against 0.49 / 0.51 / 0.71 for per-level windows). The
  ratio is **clamped into `[1/10, 10]`** (mirroring
  `gsplats/lod/substitutive.py`'s `_MASS_SCALE_BOUND`): real ratios are ~1.1-1.2,
  `mwma` is a second-moment ratio and so not robust on heavy-tailed amplitudes,
  and a 10x rescale would be a bigger switch pop than the defect. Out of bounds
  the ratio is clamped, not discarded — discarding applies the full uncorrected
  error, clamping caps it (at a genuine ratio of 0.02, scale 1.0 leaves the level
  50x too wide and it renders black; clamping to 0.1 caps that at 5x). Scale 1.0
  is kept only when there is no usable ratio at all: missing statistics, a
  non-positive `mwma`, a non-finite quotient.
- **Partition parts** share the window **verbatim**, unscaled: parts are
  disjoint pieces of ONE object with no representation change between them, and
  a dim tile really is dim (measured 1.00 shared vs 0.62 per-part). A partition's
  own reference is pooled from its parts: `lo` is the exact union minimum, `hi`
  the **count-weighted mean of the usable part tops** (weights = each part's
  `n_splats`). Both candidate rules are biased, in opposite directions, and
  neither recovers the union's true p99.9: `max` over part tops drifts upward
  without bound in the part count (1.05x at 2 parts, 1.58x at 256, 3.40x at 5000,
  ~1500x for 60 dim tiles plus one small bright one, at which point the whole
  object renders black), while the count-weighted mean is biased slightly low
  (a part's own p99.9 already under-estimates the union's). The mean is chosen
  because its bias does not grow with the part count —
  `write_partition_streaming` routinely emits thousands of parts — and because
  its failure mode, clipping outlier-bright content, is what a p99.9 window does
  by design. It is neither unbiased nor a consistent estimator of what a flat
  store would derive.
- **Reference child**: the finest child that actually carries a **usable**
  `amplitude_data_range` (`hi > lo`), walking finest→coarsest — not necessarily
  the literal finest. `add_points(substitutive_lod=…)` puts a Points leaf there,
  a scalar-amplitude level writes no window at all, and a constant-amplitude
  level writes the degenerate `[x, x]`; taking the finest child unconditionally
  left the whole structure un-harmonized in all three cases. That donor supplies
  both the window and the reference `mwma`. A degenerate part top is likewise
  excluded from the partition pool (its `lo` still bounds the union minimum).
- **Child enumeration** is name-agnostic: `Node.add_lod_group()` /
  `add_partition_group()` are public, so LOD levels and partition parts are
  every child group whose `type` is one of
  `group`/`gsplats`/`points`/`lines`/`mesh`, whatever it is named — minus the
  reserved root buckets `fitting` / `provenance` / `pipeline`, which are excluded
  by name too (`pipeline_info` is an open passthrough of caller keys, so a stray
  `type` in it must not rank `pipeline` as the finest child). LOD children are
  ordered coarsest→finest by `child_index` when every candidate has one, else by
  a `child_<i>` numeric suffix, else by sorted name — that last rule assuming
  alphabetically-last is finest, and unreachable from any Python producer, which
  always stamps `child_index`. (`additive_<i>` sub-LODs keep the prefix+digit
  rule — those names are writer-owned.)
- **Fallback**: a level whose statistics are missing (a legacy store) shares the
  reference window verbatim, i.e. scale 1.0. That is not necessarily better than
  the self-consistent window that level already had — it can be clipped by the
  shared one — but it is the honest answer with no ratio to scale by, and it puts
  the structure on ONE window, as the partition arm does regardless.

The pass only ever OVERWRITES an existing `amplitude_data_range`; it never
creates one on a node that lacked it, so the attr set on disk is unchanged. It
also refuses to write anything that is not a finite `lo < hi` (a degenerate
`[x, x]` reads as identity in the viewer, an inverted `lo > hi` inverts the
colormap), leaving the writer's value in place. One consequence of pooling is
cross-recipe: a `levels` structure's reference top is a real p99.9 while an
`overview` / `adaptive` one is a pooled estimate, so the same splats can tone
slightly differently depending on the topology they were written in (measured
222.34 vs 160.50 on one dataset).

**Mass statistics** — two per-leaf `float` attrs the writer stamps on each splat
set, including each `additive_<i>` sub-LOD (whose parent carries the ladder
aggregate `Σ massᵢ` and `Σ(massᵢ·mwmaᵢ) / Σ massᵢ`). They are independent of
`amplitude_data_range` — a scalar amplitude skips the window but still gets
these — and, unlike it, **unconditional**: an empty, mass-less or otherwise
degenerate set is stamped `0.0` / `0.0` rather than skipped (there is no
non-finite case to skip; `compute_amplitude_mass_stats` normalizes them all to
zero, so a bare `NaN`/`Infinity` token, which is not JSON and would cost a
strict reader the whole store, can never be produced). A ladder parent whose
summed mass is not positive likewise stamps `0.0` / `0.0`: "present and zero"
and "absent" must stay distinguishable, because the harmonization reads absence
as a legacy store. Both use the RAW amplitudes, the same units
`amplitude_data_range` windows, and drop the shared `(2π)^{D/2}` constant:

- **`amplitude_mass`** — total integral mass `Σᵢ aᵢ·|Σᵢ|^½`, with
  `|Σ|^½ = Π diag(L)`.
- **`amplitude_mass_weighted_mean`** — `Σᵢ aᵢ²·|Σᵢ|^½ / Σᵢ aᵢ·|Σᵢ|^½`, i.e.
  total self-energy over total mass: the amplitude a unit of mass typically
  carries. This is the quantity the harmonization scales LOD windows by.

**Note**: Broadcasting information is stored per-array via encoding metadata
(see Broadcasting Convention above), not in the group attributes.

### Quality Stamps (`lod_stats` / `level_stats`, format-additive)

Builds stamp measured approximation quality alongside the LOD structure so the
viewer can make principled display decisions (raw element counts compare
apples to oranges across substitutive levels). All keys live inside the
existing `lod_stats` / `level_stats` attr dicts — additive, no version bump;
readers treat absence as "unstamped" and fall back to counts.

- **`lod_stats.energy_fraction_cum`** (per additive sub-LOD, `additive_<i>`
  group or single-set leaf): cumulative self-energy fraction *e(k)* ∈ (0, 1]
  of the ladder prefix up to and including this sub-LOD
  (`Σ aᵢ²·|Σᵢ|^½` over the prefix ÷ the leaf total; last entry = 1.0).
  Here *aᵢ* is the **alpha-effective** amplitude `A·α` (raw amplitude times the
  per-splat color-alpha opacity for RGBA splats; equal to the raw amplitude
  when there is no RGBA alpha) — the rendered mass the additive orderer ranks
  by, so energy-ordered ladders front-load it and a small prefix carries most
  of the energy.
- **`level_stats.reference_energy`** (per leaf): the leaf's absolute total
  self-energy *w* = `Σ aᵢ²·π^{D/2}·|Σᵢ|^½` (again with the **alpha-effective**
  amplitude `A·α`). Disjoint partition parts sum, so
  *w* is the weight for aggregating per-leaf qualities across a partition.
  Inside a `kind=lod` group every level carries the **finest** content's
  total (group-consistent — self-energy is quadratic in amplitude, so
  per-level totals differ and would skew weighted aggregation).
- **`level_stats.quality`** (per `kind=lod` child): measured mixture-L²
  quality *Q* = `1 − ‖level − finest‖²/‖finest‖²` ∈ [0, 1] of the COMPLETE
  level vs its group's finest content (constant-cost sampled estimator, see
  `luxar.gsplats.lod.quality`). The finest side is 1.0 by definition
  (including each part leaf of an `overview` fine partition).

The viewer's recursive quality algebra: a leaf currently shows quality
`q = Q·e(k)`; a partition shows `Σ wₚ qₚ / Σ wₚ`; a lod group shows its
visible child's `q`. The LOD display gate releases an upgrade swap once the
candidate's committed energy reaches a threshold (0.6) instead of waiting for
the count crossover; `Q` feeds the layers-panel / data-monitor readouts.

Stamps are written by every recipe build (`RecipeParams.quality_stamps`,
default on; `Q` measurement can be disabled with `--no-quality-stamps`) and
can be retrofitted onto existing stores in place — no refit, no re-ladder —
with `luxar gsplat annotate-quality <store> [--with-quality]` (re-stamps the
root `content_hash`, so viewer caches invalidate automatically).

Related build-side geometry: `stream:<C>` ladders inside a lod group are
**sibling-aware** — every level with a coarser sibling starts its ladder at
`max(C, ceil(n/(2·K)))` so an upgrade's committed prefix passes the sibling
within 1-2 chunks (the group's coarsest keeps the small user base as the
fast-first-paint level).

### Fitting Group Attributes (Fitter-Agnostic)

The `fitting/` group is **optional** and designed to be **fitter-agnostic**. Different fitting implementations can store their own parameters while sharing common fields.

**Common fields** (fitting/.zattrs):
```json
{
  "fitter_name": "luxar.gsplats",
  "fitter_version": "0.1.0",
  "time_seconds": 45.3,
  "iterations": 850,
  "converged": true,
  "timestamp": "2025-01-15T14:30:00Z"
}
```

**Fitter-specific config** (fitting/config/.zattrs):
```json
{
  "n_iters": 1000,
  "lr": 0.01,
  "loss_type": "l1",
  "asymmetric_penalty": 1.0,
  "init_sigma_vox": 0.5,
  "seed_method": "auto",
  "enable_dynamic_ops": true
}
```

**Design principle**: Other programs that produce gsplats can write their own
`fitter_name` and custom config. Readers should:
1. Always read common fields from `fitting/.zattrs`
2. Only interpret `fitting/config/.zattrs` if they recognize the `fitter_name`

**Source grid** (fitting/.zattrs, optional) — what the splats are a
representation *of*, so that "how much did this compress?" is answerable from
the artifact alone:

```json
{
  "source_shape": [24, 32, 32],
  "source_declared": true,
  "source_dtype": "uint16",
  "source_voxels": 24576,
  "source_bytes": 49152,
  "source_stored_bytes": 6144,
  "fitted_shape": [12, 16, 16],
  "fitted_voxels": 3072,
  "occupancy": 0.02197,
  "voxels_per_splat": 205.0
}
```

`source_*` describes the volume as handed to the fitter, in the element type it
was **stored** in — `source_bytes` is the honest denominator of a compression
ratio, and a 16-bit acquisition must not be recorded as the float32 the fitter
works in. `source_declared` is present (and `true`) only when the producer
*stated* the source grid instead of it being measured from the array the fitter
saw — a producer that pulls one channel out of a 5D store and downscales it
before fitting must declare the acquisition, or the ratio would be quoted
against its own working copy. Absent means measured; readers that quote the
ratio should carry the distinction, because a stated denominator is a claim and
a measured one is an observation. `source_stored_bytes` is what that
acquisition **occupies** as opposed to what it decodes to — the compressed file
you download — and it is optional and never inferred, because the source
codec's own factor is precisely what is unknown: `source_bytes / <store size>`
credits the splats with it, while `source_stored_bytes / <store size>` does not,
and on real microscopy the two differ by more than an order of magnitude.
Absent means unknown, and a reader should then quote one ratio rather than
guess the other. `fitted_*` is the grid actually optimised
against, which differs when
the fit downscaled first; collapsing the two would overstate compression by the
downscale factor cubed. `occupancy` is the fraction of fitted voxels carrying
signal — above 1% of the normalized intensity range — a ratio means something
quite different at 0.03% occupancy than at 50%. (A bare "above the subtracted
background floor" test would instead measure camera noise: the floor sits at the
background's own level, so roughly half of a noisy background is above it.) The
threshold is relative to the fit's own normalization, so a fit that did not
suppress its background counts that background as occupied — which is what it
spent splats on.
`voxels_per_splat` is `fitted_voxels` over the splat count **at the end of the
fit**, so it predates any post-fit culling.

Absent on datasets written before these keys existed; readers should report
nothing rather than infer a source grid from the bounding box.

**Quality metrics** (fitting/.zattrs, optional) — the round-trip score of the
fit, measured by re-rendering the splats against the fit-basis reference
`clip(V - image_min, 0, None)`, not the raw acquisition. The PSNR `data_range`
and relative-L2 denominator are derived from that shifted reference too:

```json
{
  "psnr_db": 37.0,
  "foreground_psnr_db": 20.0,
  "foreground_threshold": 0.1274,
  "foreground_fraction": 0.00098,
  "ssim": 0.9912,
  "mse": 0.000199
}
```

`psnr_db` / `ssim` / `mse` are taken over **every** voxel, which on sparse data
makes them largely a score for reproducing the background: a fit that discards
nine tenths of the signal in a 99.9%-empty volume still reports 37 dB.
`foreground_psnr_db` is the same PSNR averaged over `target > foreground_threshold`
only (Otsu's threshold on the **target**, so a fit that hallucinates structure is
still scored where the signal actually is), while `data_range` stays the whole
volume's — the convention `calibration.metrics.held_out_psnr_foreground` uses, so
the two are comparable. Quote `foreground_fraction` with it: a dB figure taken
over 0.01% of a volume means something quite different from one taken over 40%,
and the bare number cannot say which.

The score is always taken on the fitter's own **voxel grid**, whatever frame the
centers were handed back in. A dataset fitted with `output_space: real` and a
`voxel_size` stores physical centers, so re-rendering the stored splats directly
onto the source grid will not reproduce these numbers — divide the spacing back
out first. The metrics describe the fit, not the coordinate frame it was
delivered in.

**Stacked component provenance** (`fitting/part_provenance`, optional) keeps the
component fits addressable without pretending their scores aggregate into one
number. It is a JSON list in stacked-coordinate order:

```json
[
  {
    "coordinate": 0.0,
    "fit_reference": {
      "kind": "preprocessed",
      "note": "connected-component filter, minimum 4 voxels"
    },
    "fitting": {
      "psnr_db": 47.01,
      "foreground_psnr_db": 22.81,
      "source_shape": [96, 640, 640],
      "source_dtype": "uint8",
      "source_bytes": 39321600
    }
  }
]
```

`fit_reference.kind` is one of `acquisition`, `preprocessed`, or `synthetic`;
`note` is optional free text. Missing reference metadata means unknown. A reader
may quote a bare component dB range only when **every** entry says
`acquisition`; `preprocessed`, `synthetic`, and unknown records remain useful
provenance but are not scores against the published acquisition. Non-finite
fields are omitted from that entry rather than shortening the list. When every
entry records the same source grid and dtype, a composition that creates a
stacked axis also carries
`source_shape = [parts, *part_shape]` in source-array order (stack first, unlike
the appended center column), the shared `source_dtype`, unanimous
`source_declared`, and sums of `source_voxels`, `source_bytes`, and
`source_stored_bytes` at the root. Source fields recoverable from that unanimous
root aggregate may be omitted from component records; differing per-component
values remain explicit. Values inside each `fitting` record describe
the component as fitted; a caller may subsequently filter or normalize the
splats before stacking, so per-part counts need not sum to the archived root
count. The current record does not identify which center column holds its
coordinate, so a later rewrite can scrub stale fitting/source fields but cannot
generically remove records for coordinates eliminated wholesale. After such a
rewrite, the list length is provenance cardinality from stack time, not a
surviving-frame count; re-stack the rewritten components to refresh it.
Flattening a partition discards its slot-keyed record because those spatial part
coordinates no longer exist in the resulting leaf, even when only one part
survives.

Composition may nest the same record recursively in an entry's `fitting` block.
`batch-fit merge` uses this for its multi-level fan-in: root entries are spatial
partition parts; a channel level appears only when multiple channels are selected,
and a timepoint level appears only when multiple timepoints are selected. Records
are therefore one to three levels deep, with timepoints directly under a part when
only one channel is selected. Coordinates at each level are the real slot, channel,
or timepoint indices. Because the batch manifest does not state what preprocessing
preceded the selected input array, these generated records omit `fit_reference`
(unknown) rather than claiming acquisition scores.
An unstamped tile remains present without quality or source keys in its `fitting`
dictionary, so a mixed store re-merged after a partial re-fit is explicit rather
than silently partial. The root `fitting/part_provenance` list is the canonical
home for this nested batch record; streamed `part_<i>` `lod_stats` do not repeat
it. A one-slot merge still records coordinate `0.0` there and does not promote
the tile's component quality to an unqualified root scalar.

On a node tree, a root `fitting/` score is a whole-tree claim: the additive sum
of the finest surviving parts, after each part's post-fit cull, measured on that
same fitter voxel grid. It does not describe the coarser content a viewer may
select initially from a `levels` or `stream` per-part recipe.

A metric that is mathematically undefined is **omitted, not written**: a volume
with no foreground (a constant tile, a signal-free crop) has no
`foreground_psnr_db`, and an exact reconstruction has no `psnr_db`. Writing them
would put a bare `NaN` / `Infinity` token in the metadata document — not JSON,
and fatal to a strict reader for the whole store rather than for that one key.
`foreground_fraction` is still present in that case, so the artifact says why.

**What survives a rewrite.** Inherited `fitting/` and `pipeline/` stamps fall
into three categories, invalidated along three independent axes, and a tool that
rewrites a store must apply all three rules:

* **Content-scoped** — every number MEASURED against the source volume: the
  scores above, `final_loss` / `final_rel_l2` / `final_max_abs_error`, the
  error-budget cull's own `error_budget` / `max_joint_error` (and its
  `phase1_candidates` / `phase2_iterations` search counters, which mean nothing
  without it), and the per-sub-LOD `lod_stats.cumulative_psnr_db` /
  `delta_psnr_db` a progressive fit stamps one level down. Beside them, the
  **record of the reduction that produced the artifact** — `culled`,
  `culling_method`, `n_original`, `n_culled`, `amplitude_retention` — which is
  true of the operation that stamped it and false of anything downstream. They
  describe one specific splat set, so they are **dropped by any operation that
  changes which splats the artifact holds**: `cull`, `filter`, `slice`, `decimate`
  (a merge-family reduction lands on the requested count while replacing every
  splat with a representative), a reduced LOD **view** (a strict additive prefix,
  or a coarser substitutive level — which is why `lod --recipe overview` does not
  put the input fit's `psnr_db` on its merged coarse cap), and any intensity edit
  — PSNR and MSE are absolute-error metrics, so a global `x0.5` changes them
  outright. An operation that stamps its own record does so *after* the scrub, so
  a rewrite publishes the reduction it actually performed and no other.

  **Artifact-local measured stamps:** the LOD Q·e ladder stamps —
  `lod_stats.energy_fraction_cum` (a rung's prefix energy e(k)),
  `level_stats.reference_energy` (its weight w) and `level_stats.quality` (a
  level's measured Q against its group's finest). These are measured on the
  artifact's **own content** rather than against a source volume, so a coarse
  level's stamps are statements about that coarse level and a plain accessor keeps
  them exactly as authored; the scene-authoring path builds every coarse child of
  a `kind=lod` group through `at_substitutive` and copies those numbers onto it. A
  content-changing rewrite, however, **recomputes** the counts, e(k), and the
  group-consistent finest-content w from the rewritten artifact. It removes
  stale Q rather than hiding its expensive Torch kNN measurement inside ordinary
  filtering; `gsplat annotate-quality --with-quality` restores it explicitly.
  A `--refine l2|volume` level's `level_stats.refine_stats` (`mse_seed` /
  `mse_refit`) is source-volume measured and cannot be remeasured by a rewriter;
  a reduction removes that nested block while keeping the descriptive `refine`
  method.

  A geometry-only transform (scale / rotate / translate / center) **keeps** them:
  the splat set is identical and only the frame moved. Dimensional embedding is
  a widening rather than a geometry-only transform: it preserves dataset-level
  source-volume metrics, but recomputes counts, e(k), and w in the promoted
  dimensionality while removing stale Q and `refine_stats`. Note this is a weaker
  claim than the reproducibility paragraph above — that argument holds for a
  `voxel_size` fit because the spacing is *recorded*, whereas `gsplat transform
  --scale` records no factor and does not update `fitted_shape` / `source_shape`,
  so the score is not reproducible from the artifact afterwards. It is kept
  because it is still a true statement about these splats, not because you could
  re-derive it. A rewrite that changes nothing at all keeps them too — a
  threshold that removed no splat, `flatten`, `additive`, `annotate-quality`,
  `partition` (the same splats, regrouped), and `reencode -e precision` (exact).
  `reencode -e auto` / `-e memory` and `migrate-format` re-quantize the Cholesky
  factors *and* (for `auto`/`memory`) the centers to fixed point, so the decoded
  values are not bit-identical to the ones that were scored; the loss is
  deliberate and bounded (~93 dB at `memory`, far below the reconstruction error
  any of these scores report), so the scores are kept as still-valid to well
  within their own precision rather than thrown away.

  One exemption, on the producing side: the fitters end with a high-retention
  cumulative trim (`cull_retention`, 0.95 by default) *after* scoring, and carry
  their measurement across it — so a stored fit's score is taken on the pre-trim
  splats, which hold 100% of the amplitude minus the retention. Re-scoring would
  cost a second full render of the volume, and the alternative is a fit that
  publishes no score at all. A content-planned box fit gets the opposite
  treatment: its score was measured on the halo-padded crop, with neighbour
  splats present and against a larger target region, so a part whose crop was
  padded (or whose core mask dropped splats) publishes no score at all.
* **Region-scoped** — the source grid (`source_shape`, `source_voxels`,
  `source_bytes`, `source_stored_bytes`, `fitted_shape`, `fitted_voxels`,
  `occupancy`, `voxels_per_splat`, and the `source_declared` marker that
  qualifies the grid). Dropped only by a **spatial** restriction that actually
  excluded splats, because that is what makes the compression ratio quote a
  volume the artifact no longer represents. `source_dtype` is exempt: a crop
  cannot change the element type. A non-spatial cull keeps this whole block.
* **Structure-scoped** — the artifact's own **topology** record in `pipeline/`:
  `lod_kind`, `recipe`, `compression_factor`, `method`, `n_substitutive_levels`,
  `coverage_inflation`, `conserve_mass`, `refine`, `refine_iters`, the additive
  ladder summary (`lod_method`, `lod_n_lods`, `lod_breakpoints_kind`,
  `lod_cutpoints`, `lod_substitutive_level`) and the `batch-fit merge` per-part
  knobs (`per_part`, `n_lods`, `breakpoints`, `levels`, `additive_ladders`).
  Dropped only by a rewrite that changes the **structure kind**. The test for a
  new command is one question — *can this command's output have a different
  structure kind than its input?* — and if the answer is yes it must scrub, even
  when a particular run happens to preserve the kind. Four commands qualify
  today: `flatten` (one flat leaf), `partition` (a `kind=partition` of bare
  leaves), `decimate` (one flat leaf whenever it actually reduces; its
  `target >= n_splats` early return hands the input straight back, which
  correctly republishes the record because nothing changed) and `lod`, whose
  every `--recipe` starts from `data.flattened()` — so no `lod` output preserves
  its input's shape, and a laddered or substitutive store is a legal input (the
  gate is matrix-shaped-ness, so a partition and a lod group with non-leaf
  children are both refused). `lod` is also the one of the four that publishes
  a topology record of its own, and it publishes **only** what its own builder
  stamped: `recipe` alone for `flat` / `tiles` / `overview` / `adaptive`, plus the
  additive ladder summary for `stream`, plus the substitutive block as well for
  `levels`. Nothing positive is invented to fill the gap — an absent `lod_kind`
  is the format's "this artifact does not know", and the alternative (stamping
  `lod_kind: additive` on a `stream` output) would be a new claim rather than a
  scrub. Content-changing but structure-**preserving** ops keep the block and
  *re-stamp* the counts that moved instead: a `cull` of a substitutive pyramid is
  still that pyramid, with refreshed `lod_n_lods` / `lod_cutpoints` — and, when
  the store carries it, the un-prefixed `n_lods` that says the same thing (see
  the second-spelling paragraph below). Its two siblings stay: `method` is the
  additive *ordering*, which pruning an emptied rung does not change, and
  `breakpoints` is the build **spec** that was requested — this rewrite built no
  new ladder from another one, it pruned the ladder that spec produced.

  A structure-preserving rewrite that REPLACES the thing a stamp summarises owes
  the same refresh. `additive` re-ladders every leaf, so the root ladder summary
  is rebuilt from the tree it wrote — `lod_n_lods` / `lod_cutpoints` from the
  ladder, `lod_method` / `lod_breakpoints_kind` read back off the rebuilt leaf so
  an `auto` request publishes the method it resolved to (and each is *deleted*
  when the rebuilt leaf does not publish it — absence is the format's "this artifact does
  not know", while the inherited value would describe the ladder that is gone).
  The summary describes
  ONE ladder: the leaf itself for a flat store, and for a `kind=lod` group the
  level `lod_substitutive_level` names (the rule `cull` already follows).
  `lod_substitutive_level` itself is untouched — a re-ladder moves no level. On a
  shape where no single leaf can be the summary (a `kind=partition`, whose parts
  hold different counts and therefore different rung counts) the block is
  *dropped* rather than filled from an arbitrary part; that is also what the
  `tiles` / `overview` / `adaptive` builders publish at the root. Present keys
  only, in both directions: a store that never published a summary does not
  acquire one.

  The **same ladder has a second spelling** in the same group, and it gets the
  same treatment: the un-prefixed `n_lods` / `method` / `breakpoints` that
  `batch-fit merge --recipe stream` stamps for its per-part recipe, which
  `additive` over a batch-fit partition would otherwise leave asserting the rung
  count the drop above had just refused to assert. `n_lods` is refreshed from the
  ladder and `method` read back off the rebuilt leaf, exactly as their prefixed
  twins are; `breakpoints` is *dropped* rather than refreshed, because it holds
  the build **spec** (`"stream:14000"`, `"counts:5,15,40"`) in a different
  vocabulary from the leaf's resolved `lod_breakpoints_kind` and cannot be read
  back off a ladder. `per_part` is left alone — a per-leaf re-ladder leaves a
  per-part ladder per-part. The `levels` branch of that same producer stamps a
  `method` too, but it is the *substitutive* merge method, which a re-ladder does
  not touch; `lod_kind` is what tells the two apart. The prune-family rewrites
  above (`cull`, `filter`, `slice`, `transform`) refresh the un-prefixed count as
  well, for the same reason and from the same helper: a `batch-fit merge --recipe
  stream` output that is culled until a rung empties has one fewer rung, whichever
  spelling states it.

  Two things in `pipeline/` are **exempt**, which is why this is a deny-list of
  key names rather than "drop the group". The normalization block (`floor`,
  `image_min`, `image_max`, `intensity_range`) describes the *input volume's*
  intensity scale, which regrouping splats cannot change. And `coarsen_dims` is
  read back by the writer — `write_gsplats_tree` derives the chunk-ordering
  barrier axes from its complement — so dropping it would silently change the
  output's chunk layout, not just its metadata.

  That complement is taken **only from a list**. A written `null` and an absent
  key are indistinguishable to the reader (`_barrier_from_coarsen_dims`), so both
  mean *no provenance* and fall through to `detect_barrier_dims` auto-detection —
  a *guess* about the stored coordinates, not "no barrier". It re-imposes a
  barrier on an axis a reduction just blended exactly when the reduction leaves
  that axis' grid **intact** (timepoints far enough apart that no cluster spans
  two of them, so the coordinates stay integral); where the reduction averages
  the grid away, auto-detection finds nothing on the levels it merged but still
  finds the axis on a level it left alone, so a ladder comes out with a per-level
  *mixture* of layouts. Which of the two you get is a property of the data, not
  of the metadata, so the explicit spelling is the honest one either way.
  "Coarsen everything" therefore has an explicit spelling and a
  non-spelling: `[0, …, d-1]`, whose complement is the empty list (a real,
  authoritative *no barrier*), versus `null`, which asserts nothing and lands on
  the heuristic. **Three producers write the explicit list**, resolved through
  one shared function (`resolved_merge_coarsen_dims`): `decimate`'s `merge`
  family, `make_substitutive_lod` (and so every `lod --recipe levels` build),
  and the `batch-fit merge` per-part record. `batch-fit merge` still writes
  `null` when its manifest never recorded a `spatial_shape`: the merge cannot
  establish the part width there and declines to invent one, since a wrong
  explicit list is worse than an absent one — the writer acts on it.

  **Three substitutive producers publish nothing at all**, and an absent key is
  read exactly as a `null`: `lod --recipe adaptive`, `lod --recipe overview`,
  and `fit --recipe levels`. All three write their `pipeline/` group from their
  *input's* stats rather than from the recipe they just ran (`cli/lod.py`'s
  composed-recipe branch and `fit_utils.save_fit_output`), so the reduction's
  own choice never reaches the store — including a `--coarsen-dims` the user
  passed explicitly. Measured on a 200-splat 4D fixture over three timepoints
  spaced 1000 apart, `--parts 2 -K 4 -L 2`: `adaptive` writes no `coarsen_dims`
  and comes out with the per-level *mixture* described above (16 groups at `[]`,
  8 at `[3]`) whether or not `--coarsen-dims 0,1,2` was given; `overview` writes
  none and gets `[3]` on all 12 groups, the barrier its merged coarse level
  blended away. Fixing that means plumbing the composed recipes' own parameters
  into the record, which is tracked on #1600 and is not what the shared resolver
  above changed. (`flat`, `stream` and `tiles` publish nothing either, but
  correctly: they run no substitutive reduction, so they blend no axis and the
  inherited value — or the fallback — stays true of the result.)

  Otherwise `null` appears only in stores written before this was settled
  (#1600); it still reads correctly — as no provenance — and nothing rewrites
  those stores in place.

  Newly built `levels` pipelines that coarsen every dimension consequently get a
  **different chunk layout** from the ones built before: the stacked axis loses
  the barrier auto-detection used to re-impose on it, so chunks are ordered
  purely spatially and no longer group by timepoint. That is the layout the
  reduction earned — it blended that axis — but a store rebuilt at a *new*
  `content_hash` cannot be served under the old URL to a warm viewer cache.

  The move does **not stop at the `levels` store**. `coarsen_dims` is exempt
  from the structure scrub (above), so the explicit list is *inherited* by every
  downstream rewrite of that store and takes the new layout with it. Measured on
  the same fixture: `flatten`, `additive`, `partition`, `cull`, `filter`,
  `slice`, `transform`, `reencode`, and a `lod --recipe tiles|stream` rebuilt off
  the `levels` store all carry `[0, 1, 2, 3]` and write `slice_dims: []` where
  before they wrote `[3]`. Two things make that acceptable rather than merely
  tolerated. The direction is safe: a *missing* barrier only costs over-fetch,
  while a false one gives a spatial axis tight chunk bounds and can drop splats
  from a query — the asymmetry `detect_barrier_dims` is written around, so no
  splats are lost either way. And the inherited value is *true of these outputs
  except the finest level*: none of those commands coarsens anything, so the axes
  the `levels` build blended stay blended in their results — but a substitutive
  ladder's finest level is the input **unreduced**, and in it that axis was never
  blended at all.

  That exception is the honest edge of the move. `flatten` of a 4D coarsen-all
  `levels` store hands back exactly those 200 original, never-blended splats
  while carrying `coarsen_dims: [0, 1, 2, 3]` — a claim about splats it is not
  true of, and with it the loss of a barrier that would have been legitimate.
  That is the price of one layout per ladder instead of a heuristic answering
  each level separately, and it is why a `>3D` build that means to keep a
  time/channel axis should say so with `--coarsen-dims` rather than rely on the
  fallback to notice.

  Exempt from the scrub is not exempt from being TRUE. A rewrite that coarsens
  over its own choice of axes owes the output a fresh `coarsen_dims`, because the
  inherited one would put the barrier on an axis this reduction just blended.
  `decimate`'s `merge` family therefore re-stamps the set it resolved, always as
  an explicit sorted list: the requested dims for a proper subset, and the full
  `[0, …, d-1]` for coarsen-everything (the default, and what a request naming
  every dim normalises to). Its `prefix` family stamps nothing: a prefix merges
  no axis and every survivor is one of the input's splats at its own coordinates,
  so the inherited value — and the layout derived from it — stays true.
  `--coarsen-dims` is a merge-only knob, never published by a prefix, and
  `decimate` warns (a `UserWarning`, displayed as an arbol line under the CLI)
  when a run resolved to `prefix` and dropped it
  (with `method="auto"` the family flips at the 50 %-kept crossover, taking the
  output's chunk layout with it). The request is range-validated for both
  families before the family is chosen, so the same argument cannot be a hard
  error on one path and silently accepted on the other.

No category subsumes another, which is why one predicate cannot serve them: an
amplitude-threshold cull loses the scores and keeps the grid and the topology, a
whole-volume bbox that excluded nothing keeps all three, a real crop loses the
scores and the grid but keeps the topology, and `flatten` loses only the
topology. Descriptive counters are never dropped by any rule — `iterations`,
`best_iteration`, `converged`, `time_seconds`, `fitter_name` and `filtered` /
`filter_criteria` describe the run or the edit, both of which happened.

**What survives a rewrite: the authored appearance.** The rules above govern
what a rewriting tool must *drop*; the mirror-image obligation is what it must
*keep*. A rewriting command owns the **structure**, not the **look**: the
builders construct fresh nodes that know nothing about the input, so unless the
source root's authored compositing attrs are handed back to the writer, its own
defaults take over — `blending_mode` disappears entirely and
`opacity` / `absorption` / `gamma` / `intensity` / `offset` snap back to their
identity, silently resetting whatever was tuned in the Layers panel. The key set
is `AUTHORED_APPEARANCE_ATTRS` (`core/group/compositing.py`): the compositing
attrs minus `transform` (a stored matrix is column-major and would be transposed
a second time on the way back in), plus `colormap`. It is read with
`gsplats/io/load_gsplats.read_authored_appearance` — directories and `.zip` /
`.tar.gz` archives alike — and passed as `root_attrs=` to
`write_gsplats_tree` / `GSplatData.save`, which seeds it at **lowest
precedence** so the command's own structural attrs still win. The one attr
refused on the way through is a `colormap` of `"custom"`: it names a sibling
`colormap_lut` array the attrs-only read cannot carry, so the bare sentinel
would dangle.

With **several** inputs (`gsplat merge`) there is no single source root, so the
carried value must be **agreed**: a key rides along only when every input that
*has* an opinion on it agrees, and an input with no opinion casts no vote (at
least one input must have one for the key to appear). On any disagreement the
key is dropped and the command **says so**, naming the key, the differing values
(each with the input it came from) and what lands on disk instead — the same
unanimity rule as
`agreed_normalization_stats`, but loud rather than silent, because appearance is
hand-authored and a user who tuned two datasets has to be told which choice did
not survive.

"Having an opinion" is narrower than "carrying the key", and that is the
load-bearing detail. The writer STAMPS identity values on every save —
`opacity: 1.0` / `absorption: 1.0` / `gamma: 1.0` / `intensity: 1.0` /
`offset: 0.0` / `layer: true`, plus `colormap: "gray"` on a colorless store —
so a value **equal to the writer's manufactured default** counts as silence,
exactly like an absent key (the values live in one place,
`WRITER_STAMPED_APPEARANCE_DEFAULTS` in `core/group/compositing.py`, which is
what both the writers and the vote read). The cost is stated plainly: nothing on
disk distinguishes a deliberately authored `opacity: 1.0` from an untouched
store, so a deliberate identity loses to a sibling's `0.75`. The alternative is
worse — it is what the code did first: a tuned dataset merged with a freshly
fitted one disagreed on **seven** keys, dropped all seven, and the writer then
stamped its defaults back, which *is* the untouched input's value. Same result,
plus seven warnings.

`visible` is the one key where ABSENCE is itself a vote. The viewer treats a
missing `visible` as visible, so an input without the key is positively saying
"shown": `visible: false` is carried only when **every** input hides, and one
hidden input plus one silent one is a disagreement rather than a unanimous hide.
Without that exception a single hidden input opened the whole merged dataset
hidden. `blending_mode` / `nd_transform` / `join` keep the plain no-vote rule,
where absence genuinely means "no opinion".

One key is additionally dropped because the merge itself invalidates it:
`colormap`, whenever the merged output carries per-splat RGB that the inputs'
palettes do not describe. Three predicates cover that: `--channel-colors`
always bakes RGB, `GSplatData.concatenate` white-fills a colorless input to
match a colored sibling, and a colored input with no authored palette is
positively asking the viewer to use its per-splat RGB. The white-fill case
happens on a plain merge and under `--as-dimension` too. The viewer makes an
ancestor palette override per-splat RGB unconditionally, so a carried palette
would render those splats through a scalar ramp. `colormap` is also refused
outright when any input root declares the `"custom"` sentinel: that palette
cannot be carried, and treating the input as having no opinion would hand the
merged root a *sibling's* palette. `--as-dimension` does **not** invalidate
`nd_transform`: the new axis is appended LAST, so every existing dimension keeps
its name and its index and the new one simply has no entry — the identity
default.

### Pipeline Group Attributes (Optional)

The `pipeline/` group records the reduction/topology provenance of a dataset —
every top-level `stats` key that is neither a fitting metric, the fitter
config, provenance, nor a header key stamped by the loader. Written by the
same single-sourced splitter (`split_fitting_info`) all writers use, so a
substitutive/pyramid/recipe build round-trips its parameters:

```json
{
  "lod_kind": "substitutive",
  "method": "kmeans_lloyd",
  "compression_factor": 4,
  "coverage_inflation": 3.0,
  "refine": "l2",
  "refine_iters": 120,
  "coarsen_dims": [0, 1, 2],
  "n_substitutive_levels": 4,
  "image_min": 110.0,
  "image_max": 4095.0,
  "intensity_range": 3985.0,
  "floor": 110.0
}
```

A fit also persists its **normalization metadata** here (see the per-writer
table below for exactly which keys, and where the record comes from):
`image_min` / `image_max` / `intensity_range` record how the source volume
was normalized, and `floor` is the background level subtracted before
fitting (`null` when floor suppression was disabled or refused). **Semantic
contract:** the floor is NOT added back — stored amplitudes are
background-relative (intensity above the subtracted pedestal), so renders
reconstruct the floor-suppressed volume, not the raw one.

Wherever it is recorded it is under one key name (`floor`) in one location
(`pipeline/`). The full key set is `NORMALIZATION_STATS_KEYS` in
`gsplats/io/save_gsplats.py`; which of those four a given writer can honestly
fill differs, and the last column says so.

| Writer | How it reaches `pipeline/` | Keys |
|---|---|---|
| flat fit (`--tiling none`) | `stats` → `split_fitting_info` (`gsplats/fitting/results.py`) | all four |
| progressive fit | `stats`, stamped by `lift_normalization_stats` — the pedestal is removed once up front, so no individual pass records it | all four |
| sequential tiled merge (`fit_tiled`), flat leaf or `kind=partition` | `_stamp_merge_normalization` on the merged `stats` / the ROOT node's `meta`, from the level the merge applied plus the bounds its tiles agree on | all four |
| parallel tiled merge (`fit -j N`) | same stamp, but the merge applied no level itself: the tiles are reloaded WITH stats and the block is recovered from what they unanimously recorded | all four |
| `--tiling content` | the one level `resolve_shared_floor` gave every box, stamped on the merged leaf's `stats` or the root node's `meta` — unless the boxes themselves recorded a level, which wins (content boxes now share one `norm_range`, so their recorded bounds agree across boxes; a recorded floor can still exceed the planned level when the shared low endpoint does) | `floor`, plus any bound the in-process boxes of a `--flat` fit agreed on |
| `batch-fit merge` (default `kind=partition`, and its K=1 bare leaf) | `manifest.floor_level`, the ONE level the plan pinned for every `(t, c)` task, folded into `pipeline_info` | `floor` only |

Two paths deliberately write nothing rather than guess. `batch-fit merge` is
silent when the manifest pinned no level — a negative resolved level is
forwarded as a SPEC for each task to re-resolve, so there is no single answer —
and the legacy `batch-fit merge --flat` fan-in (a multi-stage reload through
`combine_as_new_dimension` / `merge_with_channel_colors`) carries no block at
all. An **absent** key means "this artifact does not know"; `floor: null`
asserts that no pedestal was removed, so the two are never interchangeable.

Where a writer records `image_min`, it records it in the **input volume's own
units** and, when a floor was applied, equal to `floor`: `_normalize_data` sets
`image_min = max(resolved_floor, image_min)` and records that applied value as
`floor`, so `image_min == floor` whenever suppression ran. #1616 makes the
pre-clamp `image_min` shared across content and batch children. A tiled or progressive path subtracts the pedestal OUTSIDE the fitter and
then fits with `floor="none"`, so it shifts its inner `image_min` / `image_max`
back by the applied level before recording them — otherwise `image_min` would
mean a post-subtraction minimum on one path and the applied level on another.

The two differ in what is left for the inner fit to remove, and therefore in
what `floor` means. A **tiled** fit hands every tile a shared `norm_range`
pinned at `image_min = 0`, precisely so no second constant comes out (a
per-tile one would be subtracted twice across an overlap band and reintroduce
the seam apodization exists to hide), so its `floor` is the level it subtracted
up front, verbatim. A **progressive** fit has no such shared range: pass 0's own
normalization removes whatever pedestal is LEFT on top, so the level actually
taken out is the sum of the two, and its `floor` is the shifted `image_min`
rather than the requested level — which would understate the removal whenever
the request sits below the volume's minimum.

Merging is unanimous-or-silent: `GSplatData.concatenate` carries a key only when
every input that records it agrees, because two independently fitted volumes
have two different pedestals and promoting the first would mislabel the rest.
`gsplat info` lists the block among its headline metadata, on both its flat
report and its node-tree (`kind=partition`) one.

Values are JSON-attr-safe (numpy scalars coerced; non-serializable values
dropped at write). Readers merge these into `stats` on
`load(include_stats=True)`; on key collision the `fitting/` and header keys
take precedence. Per-node `level_stats` attrs (e.g. a substitutive level's
`refine_stats` block) may likewise carry JSON-safe **nested dicts**.

### Provenance Group Attributes (Optional)

The `provenance/` group records information about the source image:

```json
{
  "source_file": "/path/to/image.tif",
  "source_hash": "sha256:abc123...",
  "shape": [128, 256, 256],
  "dtype": "uint16",
  "normalization": {
    "method": "percentile",
    "low": 0.1,
    "high": 99.9
  }
}
```

---

## Spatial Ordering and Indexing

GSplats arrays are spatially ordered for compression and efficient spatial queries using the same algorithms as Points.

**Algorithm Reference**: See `packages/luxar/src/luxar/io/README.md` and the implementation under `luxar.io` for complete details on:
- Morton/Hilbert ordering algorithms
- Compound ordering (discrete dimensions first, then Morton)
- Chunk bounds calculation

**Implementation Location**: Spatial ordering is implemented in `luxar.io` and reused by `gsplats.io` (single code path).

### GSplats-Specific Details

**Extent Calculation for Chunk Bounds**:

GSplats have ellipsoidal extent (unlike point radii). Chunk bounds include this extent:

```python
# For each dimension d, compute extent from Cholesky factors
# Covariance diagonal: covariance[d,d] = sum(L[start_idx + i]^2 for i in 0..d)
# For packed Cholesky (row-major):
#   2D: [L00, L10, L11] → cov[0,0]=L00², cov[1,1]=L10²+L11²
#   3D: [L00, L10, L11, L20, L21, L22] → cov[2,2]=L20²+L21²+L22²

extent[d] = sqrt(covariance[d, d]) * truncation_radius  # default 2.75 (per-axis support radius, in sigmas)

# Chunk bounds include extent -- EXCEPT on a barrier/categorical axis
# (slice_dims: time, channel), which gets no sigma expansion at all, only a
# tiny float-boundary epsilon, so a category never bleeds into its neighbour:
#   chunk_bounds[i, d, 0] = min(centers[chunk_i, d]) - 1e-3
#   chunk_bounds[i, d, 1] = max(centers[chunk_i, d]) + 1e-3
chunk_bounds[i, d, 0] = min(centers[chunk_i, d] - extent[chunk_i, d])
chunk_bounds[i, d, 1] = max(centers[chunk_i, d] + extent[chunk_i, d])

# Both arms are accumulated in float64 and narrowed to the float32 store
# OUTWARD (lo down, hi up, by one ULP -- but only when the cast moved the bound
# the wrong way). Without that step a small absolute pad past |x| ~ 2**23 falls
# under half a float32 ULP and rounds away, storing an interval TIGHTER than
# the footprint. A stored interval therefore always contains the chunk's
# geometric footprint, at any coordinate magnitude.
```

**Ordering Metadata** (stored on each leaf group's `.zattrs`):
- `ordering`: "morton", "hilbert", or "none"
- `ordering_min`, `ordering_max`: Coordinate bounds for normalization
- `ordering_bits_per_dim`: Bits allocated per dimension (typically 21 for 3D)

  (Legacy files may carry `morton_*` keys; only the Python inspector
  (`gsplats/io/inspect_gsplats.py`) still accepts those as a fallback — the
  web viewer reads only the `ordering_*` keys.)

**Spatial Index Array**:
- `chunk_bounds`: (num_chunks, d, 2) float32 array
- Enables efficient spatial queries without loading splat data
- Same query algorithm as Points (AABB intersection test)

---

## Splat Ordering for Compression

### Why Order Matters

Splats are inherently unordered, but storage order significantly affects compression:
- **Nearby splats** have similar centers → excellent delta compression
- **Nearby splats** often have similar covariances (local structure)
- **Chunk access** becomes coherent for spatial queries

### Space-Filling Curves

Two options for spatial ordering:

#### Morton (Z-order)

**Pros**:
- Simple bit-interleaving implementation
- Trivial nD extension
- Fast to compute

**Cons**:
- Occasional "jumps" in locality (at quadrant boundaries)

**Algorithm**:
```python
def morton_encode_nd(coords: np.ndarray, bits_per_dim: int = 16) -> np.ndarray:
    """Encode nD integer coordinates to Morton codes via bit interleaving."""
    n_points, n_dims = coords.shape
    morton = np.zeros(n_points, dtype=np.uint64)

    for bit in range(bits_per_dim):
        for dim in range(n_dims):
            coord_bit = (coords[:, dim] >> bit) & 1
            morton |= coord_bit.astype(np.uint64) << (bit * n_dims + dim)

    return morton
```

#### Hilbert Curve

**Pros**:
- Better locality preservation (never jumps far)
- ~10% better compression than Morton in practice

**Cons**:
- More complex algorithm
- Harder nD extension (but libraries exist)

**Libraries**:
- [`numpy-hilbert-curve`](https://github.com/PrincetonLIPS/numpy-hilbert-curve) - Princeton LIPS, numpy-native
- [`hilbertcurve`](https://pypi.org/project/hilbertcurve/) - Supports nD, based on Skilling 2004

### Ordering Specification

The system supports both Morton and Hilbert ordering methods:

```python
def sort_splats_spatial(
    centers: np.ndarray,
    method: Literal["morton", "hilbert"] = "hilbert",
    resolution: Optional[int] = None,  # Ignored (kept for signature stability)
    slice_dims: Optional[Sequence[int]] = None,  # Barrier axes (time/channel)
) -> tuple[np.ndarray, dict]:
    """Return sort indices + ordering metadata (barrier-aware compound sort)."""
    ...
```

The grid resolution is **derived per-axis from the bit budget** (capped at 21
bits per dimension — `min(21, budget // n_ordering_dims)`, exact for the
typical ≤3 ordering dims, matching the `ordering_bits_per_dim: 21` leaf attr) — the
`resolution` parameter is ignored. When `slice_dims` names categorical/barrier
axes (time, channel), splats are grouped by those axes first and the space-
filling curve orders spatially within each barrier value, so a chunk never
straddles two timepoints.

---

## Additional Compression Techniques

Based on [3DGS compression survey](https://arxiv.org/html/2502.19457v1) and related research:

### 1. Quantization

Quantization is handled by `luxar.encoding` based on semantic types:

| Field | Semantic Type | MEMORY Mode Encoding |
|-------|---------------|---------------------|
| `centers` | COORDINATE | `linear_perchannel_u16` (uint16 per-axis fixed-point) |
| `amplitudes` | POSITIVE_SCALAR | `bounded_scalar_uint8/16` (narrow range) or `geolog_scalar_uint8` (wide range; AUTO uses `geolog_scalar_uint16`) |
| `cholesky_factors_diag` | CHOLESKY_DIAG | `log_perchannel_u8` (per-column log) |
| `cholesky_factors_offdiag` | CHOLESKY_OFFDIAG | `signed_log_perchannel_u8` (per-column signed-log; absent if d==1) |

**Centers** are uint16 per-axis fixed-point (`linear_perchannel_u16`) in AUTO and
MEMORY — each axis quantized over its own [min, max] to 65536 levels, decoded back to
float32 (visually lossless, sub-unit, ~2× smaller). float16 is NOT used (relative
precision is a footgun for absolute positions); a per-axis extent ≥ 2¹⁶ falls back to
float32.

A **gridded axis** — one whose distinct values all sit on a single regular grid, which a
stacked/categorical axis built with `sigma=0` (e.g. `combine_as_new_dimension`) normally
is, though `values=` is arbitrary and a stack with more distinct values than uint16 has
levels is not gridded either — keeps its uint16 encoding but has its grid **snapped onto
the data's own spacing**:
the stored `col_hi` is widened to `col_lo + step·65535`, so every value round-trips
bit-exactly. That is what lands a stacked axis exactly on its integer frame coordinates,
and it changes no dtype and no bytes on disk (`col_lo`/`col_hi` are stored per axis
regardless).

A third, geometry-aware **sigma rail** backstops what no grid can cover, falling back to
float32 (with a `UserWarning`) when **half** an axis's grid step `(hi - lo) / 65535` —
the worst-case round-trip displacement — exceeds `MAX_CENTER_DISPLACEMENT_SIGMAS` (1.0)
× the marginal σ of **more than `MAX_UNREPRESENTABLE_SPLAT_FRACTION` (0.1%) of the
splats** on that axis, i.e. when quantization can move those centers clear of the cores
they were fitted to describe and out of a slice query that used to match them. That
population gate is necessary but **not sufficient**: the rail stands down wherever the
encoder is already exact. An axis the snap will store exactly is skipped, and so is a
LUT-eligible centers array (stored verbatim at ~1 B/value, which float32 would only make
4× larger) — so the case this catches is a degenerate
sub-population on an axis that is neither gridded nor LUT-eligible — a 2,000-splat
`sigma=0` track stack merged
into a 300,000-splat fit whose time axis is continuous is 0.662% of the store and
displaced by up to 1,373 σ. Sub-σ displacement is deliberately left alone: an
8192-voxel axis has a 0.125-voxel step, so its worst displacement is 0.0625 voxel, and
the handful of needle splats every real fit contains (measured ≤ 0.03% under this
criterion) keeps the uint16 size win. Only `centers` escalates; the Cholesky tier is
unaffected. An escalated `centers` array is never stored as an `array_ref` — the
encoder's content dedup is keyed on the centers bytes, which do not determine the rail's
verdict.
**Cholesky factors** are stored split (diagonal + off-diagonal); bit depth
follows the mode: PRECISION→float32; AUTO→uint8, escalating to uint16 only when the
encode-time covariance certificate measures excessive Σ error (float32 as the
practically-unreachable last rung); MEMORY→uint8.

**Log-scale amplitudes**: For high dynamic range (HDR) amplitudes, use log encoding:
```python
result.save(
    "hdr_splats.gsplats.zarr",
    encoding_mode=EncodingMode.MEMORY,
    positive_scalar_encoding="log",  # log1p/expm1 for numerical stability
)
```

See `packages/luxar/src/luxar/encoding/README.md` for complete quantization details and precision guarantees.

**Research note**: [OMG](https://arxiv.org/html/2503.16924) achieves 100-300× compression with quantization while maintaining quality.

### 2. Entropy Coding with Spatial Coherence

From research: "Entropy encoding linearizes 3D Gaussians along a space-filling curve to exploit the spatial coherence of scene parameters."

This validates our space-filling curve approach! After Morton/Hilbert ordering:
- Delta encoding of centers (store differences)
- Run-length encoding for similar values
- Standard compressors (zstd, blosc) work better on ordered data

### 3. Attribute Factorization

**F-3DGS approach**: Instead of storing full attributes per splat, factorize:
```
attributes = basis_vectors @ coefficients
```

For our case, could factorize:
- Cholesky factors → shared basis shapes + per-splat coefficients
- Would require fitting a basis during save (more complex)

**Status**: Deferred to future version - initial implementation uses ordering + standard compression.

### 4. Culling Before Storage

Not a storage format concern, but worth noting:
- Remove low-amplitude splats before saving
- User can cull before saving: `result.cull(method="cumulative", retention=0.95).save(...)`

---

## Compression Configuration

### Blosc Settings

The default is a width-aware per-dtype policy (`luxar.encoding.compression`),
resolved from the stored dtype at write time:

```python
# multi-byte integer codes (uint16 quantized/fixed-point)
Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
# uint8 codes and float arrays
Blosc(cname="zstd", clevel=9, shuffle=Blosc.NOSHUFFLE)
```

Decode speed is level-independent (natively and in wasm), so the high level
is purely a write-time budget. Pass an explicit `Blosc(...)` to override, or
`None` to store uncompressed.

### The `luxar_delta_v1` delta filter (v3.3, optional, probe-gated)

Quantized code arrays (COORDINATE `linear_perchannel_u16`, the Cholesky
`log_perchannel` / `signed_log_perchannel` halves, the scalar
`bounded_scalar` / `geolog_scalar` amplitudes, and COLOR `rgb_uint8` /
`geolog_perchannel` / integer-passthrough codes) may carry a zarr v2
**filter** in `.zarray`:

```json
"filters": [{"id": "luxar_delta_v1", "cols": 3, "bits": 16}]
```

Hilbert ordering makes consecutive codes a smooth ramp; the filter stores
per-axis **modular delta + zigzag** residuals, laid out **column-major within
each chunk** (all column-0 residuals, then column-1, …), which the Blosc
policy above then compresses 12-16% smaller whole-store (lossless — a pure
storage transform below the `encoding` layer; `encoding` attrs, decode
kernels, and the range-loader are untouched). Per chunk of `rows × cols`
codes, per column, all arithmetic mod `2^bits` with an implicit `0` anchor at
each chunk start:

```
encode:  d  = (code - prev) mod 2^bits          # prev = 0 at chunk start
         s  = d >= 2^(bits-1) ? d - 2^bits : d  # signed interpretation
         zz = (s << 1) ^ (s >> (bits-1))        # zigzag -> uint8/uint16
decode:  s    = (zz >> 1) ^ -(zz & 1)
         code = (prev + s) mod 2^bits
```

The filter is **probe-gated at encode time**: one representative chunk is
compressed both ways and the filter is applied only where it wins
(deterministic; never worse). Arrays where it cannot apply are excluded
structurally: float32 fallbacks, LUT/broadcast/array_ref priority paths, and
INDEX arrays never carry it.

Corruption blast radius: Blosc/zstd carries no payload checksum (a
pre-existing property of every Luxar array, with or without this filter), so
a silently corrupted byte decodes to wrong values. Without delta the damage
is one element; with delta a corrupted residual propagates through the rest
of that chunk's column — still bounded to a single chunk (each chunk has its
own implicit 0 anchor).

Reader requirements: chunks are whole-chunk reconstructed inside the zarr
codec pipeline, so sub-chunk range reads keep working unchanged. In Python
the codec registers via the numcodecs `numcodecs.codecs` **entry point**
(declared in luxar's `pyproject.toml`), so any vanilla `zarr.open(...)` on a
machine with luxar *installed* resolves it with no import; `import
luxar.encoding` also registers it eagerly. The web viewer registers the
TypeScript twin (`data/codecs/luxar-delta.ts`) as `numcodecs.luxar_delta_v1`
in its zarr facade. Readers without luxar installed fail loudly (unknown
codec), never silently corrupt.

### Chunk Sizing

**Strategy**: Byte-based target converted to element counts (Zarr chunks by elements, not bytes).

**Constants** (from `luxar.typing_utils.constants` - single source of truth):
- `TARGET_CHUNK_BYTES = 65536` (64KB) - target chunk size

**IMPORTANT**: Zarr's chunking system operates on **element counts**, not byte counts. Therefore:
1. The byte target is defined once in `typing_utils`
2. At write time, we convert bytes → elements based on each array's dtype
3. Metadata stores the resulting **element count** (what Zarr needs)

**Bytes-to-elements conversion** (per array):
```python
from luxar.typing_utils import TARGET_CHUNK_BYTES  # 65536 (64KB)

# For each array, compute elements per chunk based on its specific layout:
# - centers (N, d) float32:           bytes_per_row = d * 4
# - amplitudes (N,) float32:          bytes_per_row = 4
# - cholesky_factors_diag (N, d):     bytes_per_row = d * 4
# - cholesky_factors_offdiag (N, k-d): bytes_per_row = (k-d) * 4  (k = d*(d+1)/2; absent if d==1)
# - colors (N, 3) uint8/float32:      bytes_per_row = 3 or 12

chunk_elements = TARGET_CHUNK_BYTES // bytes_per_row
```

**Example: 3D splats with all arrays**:
| Array | Shape per row | Bytes/row | Elements/chunk (64KB) |
|-------|---------------|-----------|----------------------|
| centers | (3,) float32 | 12 | 5,461 |
| amplitudes | () float32 | 4 | 16,384 |
| cholesky_factors_diag | (3,) float32 | 12 | 5,461 |
| cholesky_factors_offdiag | (3,) float32 | 12 | 5,461 |
| colors | (3,) float32 | 12 | 5,461 |

**Note**: Each array has its own optimal chunk size. The `chunk_size` in group metadata is a **reference value** for the primary arrays (centers), not a universal constant.

**Spatial-index alignment**: when the leaf carries a spatial index, each array's
count above is rounded DOWN to a whole multiple of the group's `chunk_size` atom
(never below one atom, never above the row count) — so 5,461 becomes 4,680 =
4 × 1,170 for a 1,170-row atom. That keeps every `chunk_bounds` partition's row
range inside a single zarr chunk, which is what makes a spatial-query read one
request per array per partition.

**Chunk shape specification**:
```python
# 1D arrays (amplitudes)
chunks = (chunk_elements,)

# 2D arrays (centers, cholesky_factors_diag, cholesky_factors_offdiag, colors)
chunks = (chunk_elements, n_cols)  # Keep all columns together
```

**Stored in metadata**: The leaf group's `.zattrs["chunk_size"]` records the **element count** for reference (typically computed from centers array).

---

## Encoding Metadata Preservation

When arrays are written to `.gsplats.zarr`, encoding transformations are applied and metadata is preserved for automatic decoding on load.

### Encoding Metadata Storage

**Format**: Each array stores encoding metadata in its `.zattrs` file:
```json
// Example: amplitudes/.zattrs
{
  "encoding": {
    "name": "bounded_scalar_uint8",
    "min": 0.0,
    "max": 10.0,
    "bits": 8,
    "original_dtype": "float32"
  }
}
```

**Color Storage**:
There is no separate `color_mode` attribute — the SDR/HDR decision is implied
by the encoding name chosen at write time:
```json
// colors/.zattrs - SDR float colors (all values ≤ 1.0) under AUTO
{
  "encoding": {
    "name": "rgb_uint8",
    "original_dtype": "float32"
  }
}

// colors/.zattrs - HDR float colors (any value > 1.0) under AUTO
{
  "encoding": {
    "name": "geolog_perchannel_u16",
    "col_lo": [0.001, 0.001, 0.001],
    "col_hi": [42.0, 38.5, 40.1],
    "bits": 16,
    "zero_level": true,
    "original_dtype": "float32"
  }
}
```

**Broadcasting Metadata** (when all splats share same value):
```json
// amplitudes/.zattrs - all splats have amplitude=1.5
{
  "encoding": {
    "name": "broadcasted",
    "n_elements": 10000
  }
}
```

### Automatic Decoding on Load

When loading `.gsplats.zarr`:
1. Read array from zarr
2. Check for `encoding` metadata in `.zattrs`
3. Apply appropriate decoder based on `encoding.name`
4. Return decoded float32 array

**Transparency**: Encoding is a storage detail - users always work with float32 arrays. Quantization and broadcasting are transparent.

**Implementation**: Uses `luxar.encoding.ArrayDecoder` (see `packages/luxar/src/luxar/encoding/README.md`).

---

## API Design

### Saving

```python
from luxar.gsplats import GSplatData
from luxar.encoding import EncodingMode

result = fit_gaussian_splats(image, n_iters=1000)

# Simple save with defaults (hilbert ordering, zstd compression, auto encoding)
result.save("fitted.gsplats.zarr")

# With options
result.save(
    "fitted.gsplats.zarr",
    ordering="morton",           # or "hilbert", "none"
    encoding_mode=EncodingMode.AUTO,  # AUTO, PRECISION, or MEMORY
    include_fitting_info=True,   # Store stats and config
    include_provenance=True,     # Store image metadata
    description="DAPI nuclei fitting",
)
# Colors: SDR (uint8) vs HDR (values > 1 -> geolog_perchannel_u16) is auto-detected from the
# color values and recorded in the color encoding metadata — no color_mode param.

# Memory-optimized save (quantization enabled)
result.save(
    "compressed.gsplats.zarr",
    encoding_mode=EncodingMode.MEMORY,  # Enable quantization
)
```

**Encoding modes** (see `packages/luxar/src/luxar/encoding/README.md`):
- `AUTO`: Analyze data and select encoding (may be lossy for some types, e.g., SDR colors → uint8)
- `PRECISION`: Full float32, lossless only (broadcasting still allowed)
- `MEMORY`: Aggressive quantization for minimum storage (8-bit where AUTO uses
  16-bit for the geolog family; centers stay uint16; Cholesky uint8)
- `CUSTOM`: Explicit encoder selection per array (advanced use)

**Float16 compatibility** (`float16_allowed` parameter on `ArrayEncoder` /
`LuxarZarrCompiler` — `save_gsplats()` itself has no such parameter):
- Default: `False` for TypeScript/WebGL compatibility (no native float16 support)
- Today this flag only affects UNIT_VECTOR arrays, the legacy packed-CHOLESKY
  dtype encoder, and the wide-range bounded-scalar float fallback — centers,
  colors, and split Cholesky factors never use float16 in any mode
- Set to `True` only if the decoder supports float16 natively

### Loading

```python
# Load for rendering (splats only)
result = GSplatData.load("fitted.gsplats.zarr")

# Load with all metadata
result = GSplatData.load(
    "fitted.gsplats.zarr",
    include_stats=True,
)
print(result.stats['time_seconds'])
```

**Note**: Arrays are automatically decoded based on encoding metadata. Files saved with `EncodingMode.MEMORY` (quantized) are transparently decoded to float32 on load.

### Inspection

```python
from luxar.gsplats.io import inspect_gsplats_zarr

info = inspect_gsplats_zarr("fitted.gsplats.zarr")  # returns a plain dict
print(info["n_splats"], info["ndim"])               # 10000 3
print(info["ordering"], info["ordering_bits_per_dim"])  # hilbert 21
print(info["storage_mb"], info["compression_ratio"])  # ratio is None if unmeasurable
print(info["fitting"]["time_seconds"], info["fitting"]["iterations"])
```

---

## Standalone vs Embedded Formats

In v3.0 the distinction between "standalone" and "embedded" is structural only,
not semantic: a `.gsplats.zarr` IS a detached scene-node subtree, and embedding
one into a scene is a graft of that subtree.

### Standalone Format (`.gsplats.zarr`)

**Purpose**: Persist fitted results as independent, directly-loadable files.

**Structure**: A node-tree root (leaf / kind=lod / kind=partition) plus the
self-identifying header (`format_version:"3.4"`, `format_type:"gsplats_zarr"`,
`timestamp`, `luxar_gsplats_version`, `content_hash`) and optional
`fitting/` / `provenance/`.

**Direct viewer load**: `?src=<file>.gsplats.zarr` loads the file as a scene
root. The viewer frames on `position_bounds` at the root group. No intermediate
scene conversion is required.

**Use cases**:
- Save/load fitted results between sessions
- Share fitted splats with others
- Serve directly to the viewer without wrapping in a scene
- Archive expensive computation results

### Embedded Format (Luxar Scene)

**Purpose**: Multi-object visualization with scene graph, transforms, and
side-by-side geometry layers (Points, Lines, GSplats, Mesh).

**Structure**: See `packages/luxar/src/luxar/core/README.md`. The gsplat node
subtree is BYTE-IDENTICAL to its standalone counterpart — both go through
`io/_compiler/gsplat_tree.write_gsplat_node`. Scene dimensions, 4x4
transforms, and rendering attributes are attached at the scene level.

**Use cases**:
- Visualize splats alongside other data (points, lines)
- Apply hierarchical transforms
- Multi-layer scenes with groups

### Relationship

**Unified authoring path**: Both write through the same
`io/_compiler/gsplat_tree.write_gsplat_node` walker, backed by the shared
`gsplat_assembly.py` leaf functions. A standalone leaf is byte-identical to
a scene leaf (same arrays, chunking, ordering, attrs) — enforced by parity
tests in `io/_compiler/tests/test_scene_leaf_parity.py`.

**Common foundation**:
- Same spatial ordering (Morton/Hilbert) via `luxar.io`
- Same encoding system (`luxar.encoding`)
- Same `chunk_bounds` for spatial queries
- Identical array structure and semantics

**Scene integration**: `scene.add_gsplats_from_file()` grafts a `.gsplats.zarr`
node tree into the scene. `scene.add_gsplats_from_data()` converts a `GSplatData`
through the same tree serializer.

---

## Luxar Scene Integration

The `.gsplats.zarr` format integrates with the Luxar scene system.
Single-LOD and multi-LOD data are both supported:

```python
from luxar import LuxarZarrCompiler, Dimensions
from luxar.gsplats import fit_gaussian_splats, fit_progressive_gaussian_splats

dims = Dimensions.default_3d()

# Single-pass fitting → flat gsplats node
result = fit_gaussian_splats(image, n_iters=1000)
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_data("nuclei", result)

# Progressive fitting → multi-LOD gsplats node (per-LOD subgroups)
result = fit_progressive_gaussian_splats(image, max_splats=50000)
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_data("nuclei", result)  # auto-detects multi-LOD

# From saved .gsplats.zarr file (preserves LOD structure)
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_file("nuclei", "fitted.gsplats.zarr")

# Fit-and-add in one step (supports progressive=True)
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_gsplats_from_volume("nuclei", image, progressive=True)
```

Multi-additive-LOD gsplats nodes use per-sub-LOD subgroups
(``additive_0/``, ``additive_1/``, …) under the leaf node —
``n_additive_sublods`` on the parent declares the depth. The viewer streams
these progressively (prefix-sum LODs). When the input ``GSplatData`` carries a
substitutive pyramid, `add_gsplats_from_data` writes a ``kind=lod`` Group with
one child per substitutive level (pass ``lod_group=False`` to collapse to the
finest level instead). Both paths go through the shared
``gsplat_tree.write_gsplat_node`` walker.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| File extension | `.gsplats.zarr` | Clear, descriptive |
| Encoding system | `luxar.encoding` package | Shared infrastructure, semantic types |
| Ordering methods | Both Morton and Hilbert | User choice, Hilbert default |
| Resolution | Auto with cap at 2^16 | Safe default, configurable |
| Chunk size | 64KB (TARGET_CHUNK_BYTES = 65536) | Optimal for blosc |
| Broadcasting | Standard encoding metadata | Consistent with other Luxar formats |
| Covariance storage | Cholesky (packed) | Already have it, compresses well |
| Fitting info | Fitter-agnostic design | Allows other programs to use format |
| Checkpoint/Resume | Deferred | Focus on basic I/O first |
| Node-tree LOD | v3.0 nestable primitives (leaf / kind=lod / kind=partition) | Substitutive, additive, and partition axes compose freely as a tree rather than a fixed matrix |
| Image embedding | No | Keep format focused on splats |
| Compression | Blosc zstd-9, width-aware shuffle | Byte shuffle for multi-byte int codes; no shuffle for uint8/floats (see §Blosc Settings) |
| Delta encoding | Yes (v3.3, probe-gated) | `luxar_delta_v1` zarr filter on quantized codes; 12-16% smaller whole-store (see §The `luxar_delta_v1` delta filter) |
| Streaming write | No | Not needed |
| `numpy-hilbert-curve` | Required dependency | Needed for Hilbert ordering |

---

## References

- [3DGS Compression Survey](https://arxiv.org/html/2502.19457v1) - Comprehensive overview of compression techniques
- [3DGS.zip Survey](https://arxiv.org/abs/2407.09510) - Another survey on compression methods
- [OMG: Optimized  MinimalGaussians](https://arxiv.org/html/2503.16924) - 100-300× compression
- [numpy-hilbert-curve](https://github.com/PrincetonLIPS/numpy-hilbert-curve) - nD Hilbert implementation
- [Skilling 2004](https://doi.org/10.1063/1.1751381) - "Programming the Hilbert Curve" algorithm

---

## Changelog

- **amplitude mass statistics** (2026-08-19, format-additive, no version bump):
  each written splat set may now carry `amplitude_mass` and
  `amplitude_mass_weighted_mean` (see **Mass statistics** above) alongside its
  `amplitude_data_range`. Every splat set a current writer produces carries both
  (a degenerate or mass-less one as `0.0` / `0.0`), but they remain OPTIONAL in
  the format: a reader that does not know them is unaffected, and a store
  written before they existed stays valid — the harmonization reads their
  absence as a legacy store and falls back to sharing the reference window
  verbatim. They exist so finalize can put every node of a `kind=lod` /
  `kind=partition` structure on ONE colormap window (**Window harmonization
  (#1691)** above) instead of a per-node `[min(a), p99.9(a)]`. No existing attr
  changed meaning; only `amplitude_data_range` **values** are corrected, and
  only where a writer had already stamped one.

- **v3.4.0** (2026-08-12): `kind=lod` gains the `selector: "screen-area"` mode
  - Per-child `coverage_fraction` under this selector is a literal screen-area
    fraction (projected bbox rect area / viewport area). Derived ladders use
    SCREEN-OCCUPANCY HALVING: whole-object finest `0.5` (full detail while the
    node occupies at least half the screen; one level coarser per halving of
    occupied area), partition-tile finest `1.0` (fills-screen). Every DERIVED
    ladder now stamps this selector; explicit `coverage_fractions=[...]` lists
    and existing stores keep `selector: "coverage"` (the legacy diagonal
    metric, in `[0, 4]`) and round-trip unchanged.
  - Motivation: the diagonal metric could not serve both the opening-framing
    contract (#1361) and predictable zoom-out coarsening — a zoomed-out pose
    measured 2.37/4 on the diagonal (still finest) while a legitimate opening
    framing measured 1.24; the two separate cleanly in area (≈37% vs ≈80%).
    Area is also the semantics users mean by "portion of the screen occupied".

- **v3.3.0** (2026-07-18): optional `luxar_delta_v1` delta filter on quantized codes
  - Quantized code arrays (coordinates, Cholesky halves, amplitudes, colors)
    may carry the zarr v2 filter `{"id": "luxar_delta_v1", "cols", "bits"}`: per-axis
    modular delta + zigzag residuals, column-major within each chunk, under
    the unchanged Blosc policy. Lossless and probe-gated at encode time (one
    representative chunk compressed both ways; applied only where it wins) —
    measured **12-16% smaller whole-store** on real Hilbert-ordered fits
    (14.9-15.7% end-to-end on real h2afva light-sheet leaves)
    (centers 1.20–1.31×, cholesky_offdiag ~1.15×, diag ~1.07×).
  - A pure storage transform below the `encoding` layer: `encoding` attrs,
    decode kernels (WASM/TS), and the sub-chunk range-loader are untouched —
    chunks are whole-chunk reconstructed inside the zarr codec pipeline.
    Origin: the PlayCanvas SOG comparison — SOG's size edge was WebP's spatial
    prediction; this replicates it while keeping chunked random access.
  - Readers must have the codec registered: Python via `import luxar.encoding`
    (numcodecs), viewer via its zarr facade (`numcodecs.luxar_delta_v1`).
    Stores not carrying the filter are byte-identical to v3.2.

- **encoding policy** (2026-07-05, no format change): HDR COLOR arrays are now
  quantized with the new **`geolog_perchannel_u8/u16`** encoding (AUTO → u16,
  MEMORY → u8; PRECISION keeps float32): per-column min/max-anchored TRUE-log
  grid (attrs `col_lo`/`col_hi` in ln-domain), uniform relative precision
  across each channel's dynamic range, code 0 reserved for exact zeros (the
  reserved level is the name's contract — no legacy variant). Chosen by the
  2026-07 HDR-color spike (6 datasets, 2–12.6 realized decades): true-log
  dominated linear fixed-point and log1p per-channel everywhere; ~4× smaller
  than float32 on realistic data with faint-exposure renders ≥147 dB.
  Completes the scalar↔per-channel family matrix
  (`bounded`↔`linear_perchannel`, `geolog`↔`geolog_perchannel`).

- **encoding + compression policy** (2026-07-05, no format change):
  - Wide-dynamic-range POSITIVE_SCALAR arrays (gsplat amplitudes, and any
    radii/widths spanning > 65536:1) are now stored as
    **`geolog_scalar_uint16`** (AUTO; `uint8` under MEMORY) instead of
    float32: a min/max-anchored geometric-log grid — quantisation happens
    AFTER rescaling to the array's own nonzero `[min, max]`
    (`min_log`/`max_log` attrs), giving uniform relative precision
    (~0.013% over 7 decades at u16). **Level 0 is reserved for exact
    zeros**, so no nonzero amplitude can quantise to zero by construction
    (the legacy 0-anchored `log_scalar_*` encodings zeroed 3k+ splats on
    real data; they remain decodable but are no longer produced).
  - Per-dtype **compressor policy** (measured in the manuscript
    `codec_selection` supplementary): multi-byte integer codes →
    `blosc-zstd` level 9 with byte shuffle; uint8 codes and floats →
    `blosc-zstd` level 9 unshuffled. Replaces the uniform
    `zstd l3 + bitshuffle` default (Blosc silently neutralises bit shuffle
    above level 1 at 64 KiB chunks). Zarr arrays self-describe their
    compressor, so readers need no changes.
  - The rescale-first principle generalised to the sibling encodings:
    `bounded_scalar_u8/u16` now anchor at the array's own `[min, max]`
    (new `min` attr, default 0 on decode — old arrays unaffected), and the
    per-channel `log_perchannel_*` / `signed_log_perchannel_*` pair
    (Cholesky diag/offdiag) gains **`zero_level: true`**: per-column scales
    from the nonzero min/max and code 0 reserved for exact zeros, so
    axis-aligned splats keep exactly-zero correlations (legacy arrays
    without the flag keep the old all-levels decode).

- **encoding policy** (2026-07-04, no format change): AUTO Cholesky quantization
  uint16 → **uint8 with an encode-time covariance certificate**
  - `ArrayEncoder.encode_cholesky_split` (the joint diag+offdiag entry point) now
    measures the actual Σ = L·Lᵀ reconstruction error of a u8 round-trip (p95
    per-splat relative Frobenius) and escalates to u16 — or, as a practically
    unreachable last rung, float32 — only when it exceeds
    `COV_CERT_RELF_P95_MAX = 0.05`. Both halves always share one tier.
  - The measured certificate is stored as provenance in each array's own
    `encoding` attrs: `{"metric": "cov_relf_p95", "value", "threshold", "tier"}`.
    Decode does not need it (readers were already u8/u16-agnostic — layout and
    format version unchanged).
  - Motivation: the 2026-07 covariance spike measured u8 at 94.5 dB vs the
    float32 render (~46 dB below the fit-error floor) and 2.48 B/splat
    compressed vs 8.25 at u16 (~3.3×); VQ/codebook alternatives were refuted
    (index streams defeat zstd+bitshuffle). Typical escalation trigger: merged
    heterogeneous stores whose σ columns span many decades.

- **v3.2.0** (2026-07-03): `kind=lod` selector attrs renamed to coverage semantics
  - Group `selector: "pixel_size"` → `"coverage"`; per-child `min_pixel_size`
    (absolute pixel threshold) → `coverage_fraction` (viewport-relative
    `sqrt(N_i/N_finest)` in `[0, 1]`, strictly ascending coarsest→finest,
    finest `1.0`) — device-independent LOD switching. (The bound later widened
    to `MAX_COVERAGE_FRACTION` = 4.0 so a hand-tuned `coverage_fractions=[...]`
    list stayed expressible after the viewer's fill anchor was loosened so the
    finest level engaged at a normal full-frame view instead of only once the
    object overfilled the screen (a ×4 rescale of the diagonal metric) —
    approximately fills-screen in these legacy diagonal units. That is not
    today's anchor: since v3.4 a derived threshold is a screen-AREA fraction,
    whole-object finest `0.5` and partition tile `1.0`; see the `kind=lod`
    section.)
  - v3.0 / v3.1 stores that still carry the legacy attrs remain loadable: the
    Python re-save derives fresh `coverage_fraction` thresholds, and the web
    viewer auto-adapts the legacy ladder (normalizing `min_pixel_size` by its
    finest value) with a warning. `luxar gsplat migrate-format` upgrades such
    stores in one step (detected as `v3.x-lod-pixel-size`).

- **v3.1.0** (2026-06-29): Differential Cholesky quantization (on top of the split)
  - The split diagonal / off-diagonal arrays are now **differentially quantized**
    with a generic per-channel scheme: diagonal → `log_perchannel_u8`/`u16`
    (per-column log), off-diagonal → `signed_log_perchannel_u8`/`u16` (per-column
    signed-log). Encoding mode sets the bit depth: PRECISION→float32,
    **AUTO (default)→uint8 with an encode-time covariance certificate** (escalates
    to uint16 when the measured Σ relF p95 exceeds 0.05; certificate recorded in
    `encoding.certificate` — see the 2026-07-04 changelog entry),
    **MEMORY→uint8** (visually lossless 94.5 dB vs the float32 render, ~4× raw /
    ~10× on disk after zstd). Per-array `encoding` carries per-column `col_lo/col_hi`.
  - The encodings are **geometry-agnostic** (semantic types `CHOLESKY_DIAG` /
    `CHOLESKY_OFFDIAG` select them as policy; the `*_perchannel_*` encodings are
    reusable for any positive/signed per-channel field). Decoders return float32,
    so everything above the storage layer is unchanged.

- **v3.1.0** (2026-06-28): Cholesky factors split into two arrays
  - Leaves now store `cholesky_factors_diag` (N, d) and
    `cholesky_factors_offdiag` (N, d*(d-1)/2) instead of a single packed
    `cholesky_factors`, so the diagonal (positive, scale-like) and off-diagonal
    (signed, zero-centred) can be encoded/quantised independently. The
    off-diagonal array is omitted for d=1.
  - Recombined into the packed (N, k) form on read; everything above the storage
    layer is unchanged. **v3.0** files are still read transparently (the loaders
    fall back to the single packed array when no split is present).
  - This is an enabling step for future differential quantization; on its own it
    leaves stored bytes ≈ unchanged.

- **v3.0.0** (2026-06-09): `.gsplats.zarr` on-disk format v3.0 — node tree
  - On-disk format bumped to **v3.0**; the v2.0 `substitutive_<s>/additive_<a>/`
    matrix and all earlier layouts (v1.0 flat, v1.1 lod_<i>, pre-v2.0
    substitutive directory + manifest.json) are no longer read at runtime.
    Convert with `luxar gsplat migrate-format <input> <output.gsplats.zarr>`.
  - A `.gsplats.zarr` is now a **detached scene-node subtree**: the file root
    IS the node. No `splats/` wrapper group. Three nestable primitives cover
    every LOD / partition combination: a gsplats **leaf** (single set or
    additive ladder), a `kind=lod` group (`child_<i>/` coarsest→finest), and
    a `kind=partition` group (`part_<i>/`).
  - **Viewer loads `.gsplats.zarr` directly**: `?src=<file>.gsplats.zarr`
    opens the file as a scene root, framing on `position_bounds`.
  - **Single authoring path**: standalone and scene writes go through the same
    `io/_compiler/gsplat_tree.write_gsplat_node` walker; parity is enforced
    by `io/_compiler/tests/test_scene_leaf_parity.py`.
  - `LuxarZarrCompiler.write_gsplats_multi_lod` deleted (was a duplicate of
    the additive-ladder writer); scene additive writes now go through
    `write_gsplat_leaf_subtree` → the shared walker.
  - Python: new `GSplatLeaf` / `GSplatLodGroup` / `GSplatPartition` node types
    in `luxar.gsplats.tree`; `GSplatData` gains `.tree` / `GSplatData.from_tree`;
    new `GSplatData.to_spatial_partition(max_elements, rule)`.
  - `luxar gsplat partition` now writes a single `kind=partition` file via
    spatial BSP (`--parts` / `--max-elements` / `--rule`); the old index-based
    `--indices` flag is removed.
  - Root attrs: `format_version:"3.0"`, `format_type:"gsplats_zarr"`,
    `timestamp`, `luxar_gsplats_version`; the v2.0 root-level `n_substitutive`
    / `default_substitutive` keys are gone.

- **v2.0.0** (2026-03-27): Multi-LOD format (v1.1) and scene integration
  - Added format v1.1 with per-LOD subgroups (`splats/lod_0/`, `splats/lod_1/`, ...)
  - Produced by `fit_progressive_gaussian_splats()` (iterative residual decomposition)
  - LODs are additive: render LODs 0..L to get cumulative approximation at level L
  - Each LOD has independent spatial ordering, chunk bounds, and per-LOD stats
  - Loader auto-detects v1.0 vs v1.1 from `format_version` in root attrs
  - Scene API now writes multi-LOD nodes with per-LOD subgroups (matching standalone format)
  - Updated Luxar Integration section — `add_gsplats_from_data()`, `add_gsplats_from_volume(progressive=True)`, `add_gsplats_from_file()` all preserve LOD structure
  - Replaced "Future: Multiscale storage (deferred)" with implemented multi-LOD support

- **v1.2.0** (2026-03-18): Removed sharpness from GSplats
  - Removed `sharpnesses` array from core data structure, zarr schema, and all examples
  - GSplats now use fixed standard Gaussian falloff (equivalent to sharpness=2.0)
  - Removed `has_sharpness` and `sharpness_bounds` from splats group attributes

- **v1.1.1** (2025-11-28): Chunk sizing source of truth
  - Removed local TARGET_CHUNK_BYTES redefinition
  - Now explicitly imports from `typing_utils` (single source of truth)
  - Code example shows `from luxar.typing_utils import TARGET_CHUNK_BYTES`

- **v1.1.0** (2025-11-28): Chunk sizing clarification
  - Clarified that Zarr chunks by elements, not bytes
  - Documented the bytes→elements conversion formula per array
  - Added example table showing different chunk sizes per array type
  - Clarified that `chunk_size` in metadata is an element count (Zarr's requirement)

- **v1.0.0** (2025-11-28): Initial versioned specification
  - Relocated from the old `gsplats/GSPLATS_ZARR_FORMAT.md` path
  - Converted from design document to technical specification format
  - Updated sharpness bounds from [0, 32] to [0, 31] for consistency
  - **Spatial ordering**: Added Morton/Hilbert ordering with chunk_bounds (aligned with embedded format)
  - **Byte-based chunking**: Changed from element-based (8192) to byte-based (64KB target) for Luxar consistency
  - **Colors support**: Added optional colors array to core data structure and zarr schema
  - **Color mode**: Specified color_mode storage in encoding metadata (sdr/hdr)
  - **Encoding metadata**: Documented how encoding metadata is preserved for automatic decoding
  - **Code reuse**: Specified that spatial ordering implementation lives in `luxar.io` (single code path)
  - **Format relationship**: Added section clarifying standalone vs embedded format relationship
  - **Cross-references**: Updated to use proper relative paths (../../)
  - Fitter-agnostic metadata design
  - Integration with `luxar.encoding` for semantic types
