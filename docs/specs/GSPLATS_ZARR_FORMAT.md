# luxar.gsplats.io - Technical Specification

**Version**: 3.2.0
**Last Updated**: 2026-07-13

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
| `centers` | (N, d) | uint16 / float32 | COORDINATE | Splat center positions (not broadcastable). AUTO/MEMORY: uint16 per-axis fixed-point (`linear_perchannel_u16`), decoded to float32; PRECISION / large-extent: float32 |
| `amplitudes` | (N,) or (1,) | uint8/uint16/float32 | POSITIVE_SCALAR | Non-negative intensity |
| `cholesky_factors_diag` | (N, d) or (1, d) | uint8/uint16/float32 | CHOLESKY_DIAG | Diagonal of L (positive, scale-like) |
| `cholesky_factors_offdiag` | (N, d*(d-1)/2) or (1, …) | uint8/uint16/float32 | CHOLESKY_OFFDIAG | Strictly-lower elements of L (signed); absent when d=1 |
| `colors` | (N, 3) or (1, 3) | uint8/uint16/float32 | COLOR | RGB colors (optional); SDR → `rgb_uint8`; HDR → `geolog_perchannel_u16` (AUTO; u8 under MEMORY, float32 under PRECISION); absent if not present |

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
should even uint16 fail (practically unreachable) — recording the measurement in
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

The current format is **v3.2**, a node tree (§ "On-disk grammar"). It differs
from **v3.1** only in the `kind=lod` selector attrs: the group `selector` value
`pixel_size` and the per-child `min_pixel_size` (absolute pixels) are renamed
to `coverage` / `coverage_fraction` (viewport-relative `sqrt(N_i/N_finest)` in
`[0, 1]`, strictly ascending coarsest→finest, finest `1.0`). **v3.1** differs
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
`format_version:"3.2"`, `format_type:"gsplats_zarr"`, `timestamp`,
`luxar_gsplats_version`, and `content_hash` (a metadata-only xxhash64 over
the tree's attrs + array names/shapes/dtypes, distinct per save because the
per-save `timestamp` folds in — the web viewer's persistent cache compares it
to invalidate when a file is regenerated in place). The historical `[N, M_i]` matrix is just the "full
pyramid" shape expressed as a node tree.

---

## Zarr Structure (v3.x — node tree)

The file root IS the node. The same three primitives nest arbitrarily:

### Shape 1 — bare leaf (single splat set)

```
fitted.gsplats.zarr/
├── .zattrs           # type: "gsplats", n_splats, ndim, has_colors, ordering,
│                     # ordering_min/max/bits, slice_dims, ordering_dims,
│                     # chunk_size, amplitude_range, amplitude_data_range,
│                     # center_bounds, position_bounds, truncation_radius,
│                     # opacity, gamma, intensity, offset, blending_mode,
│                     # format_version: "3.2", format_type: "gsplats_zarr",
│                     # timestamp, luxar_gsplats_version, description?
├── .zmetadata        # Consolidated metadata for fast loading
├── centers                   # (N, d) uint16 (AUTO; float32 if an axis extent ≥ 2¹⁶) / float32 (PRECISION), spatially ordered
├── amplitudes                # (N,) uint8/uint16 (AUTO) / float32 (PRECISION)
├── cholesky_factors_diag     # (N, d) uint8 (AUTO, certified — escalates to uint16 if the covariance certificate fails) / float32 (PRECISION)  (diagonal of L)
├── cholesky_factors_offdiag  # (N, d*(d-1)/2) uint8 (AUTO, certified as above) / float32 (PRECISION) (off-diagonal; absent if d=1)
├── colors            # (N, 3) uint8/uint16 (AUTO) / float32 (PRECISION)  (optional)
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
│                     # position_bounds, format_version: "3.2", …
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
├── .zattrs           # type: "group", kind: "lod", selector: "coverage",
│                     # default_level: <int>, display_type: "gsplats",
│                     # position_bounds, format_version: "3.2", …
├── child_0/          # Coarsest child (child_0 = coarsest on disk)
│   ├── .zattrs       # coverage_fraction: 0.0, compression_factor, level_index, …
│   ├── centers, amplitudes, cholesky_factors_diag, cholesky_factors_offdiag, colors?, chunk_bounds?
│   └── …
├── child_1/
│   ├── .zattrs       # coverage_fraction: <0..1>, …
│   └── …
└── child_{N-1}/      # Finest child (coverage_fraction: 1.0)
    └── …
```

Children are written **coarsest→finest** on disk (child_0 = coarsest,
child_{N-1} = finest) — the SAME order the in-memory tree
(`GSplatLodGroup.children`) uses, so the serializer writes them straight through
with no reversal. The on-disk `default_level` is `0` (the coarsest child) — the
viewer's progressive-load hint (render cheap first, then refine). This is a
distinct concept from the data-model default (the finest level the `.centers`
accessor returns); they are deliberately decoupled, so the writer stamps
`default_level: 0` independently. Each child carries `coverage_fraction`, a
dimensionless value in `[0, 1]` computed as `sqrt(N_i / N_finest)` (`N_i` =
level i's total splat count) and strictly ascending coarsest→finest; the
coarsest child is always `0.0` and the finest is always `1.0`. Being a count
ratio, it is immune to non-displayed-dimension multiplicity (e.g. a stacked
time axis inflates every level's count equally and cancels out). At render
time the viewer multiplies `coverage_fraction` by the viewport diagonal (times
a fill-factor constant) to get a pixel threshold, so the finest level activates
when the object fills the screen and coarser levels step in as it shrinks —
identically on any monitor/viewport. Any node shape (bare leaf, additive
ladder) is valid as a child.

### Shape 4 — spatial partition (`kind=partition` group)

```
fitted.gsplats.zarr/
├── .zattrs           # type: "group", kind: "partition", display_type: "gsplats",
│                     # max_elements: <int>, position_bounds, format_version: "3.2", …
├── part_0/           # BSP part 0 (any node shape valid per part)
│   ├── .zattrs       # position_bounds (per-part bounds for frustum culling)
│   └── centers, amplitudes, cholesky_factors_diag, cholesky_factors_offdiag, colors?, chunk_bounds?
├── part_1/
│   └── …
└── part_{P-1}/
    └── …
```

The viewer renders ALL parts simultaneously; THREE.js per-mesh frustum culling
selects visible parts. The partition writer uses recursive BSP (`median`,
`midpoint`, or `sah` rule) to build spatially balanced parts.

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
  "format_version": "3.2",
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
  "truncation_radius": 3.0,
  "ordering": "hilbert",
  "ordering_min": [0.0, 0.0, 0.0],
  "ordering_max": [256.0, 256.0, 128.0],
  "ordering_bits_per_dim": 21,
  "slice_dims": [],
  "ordering_dims": [0, 1, 2],
  "chunk_size": 2048,
  "amplitude_range": {"min": 0.01, "max": 1.5},
  "amplitude_data_range": [0.01, 1.5],
  "center_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  },
  "position_bounds": {
    "min": [0.0, 0.0, 0.0],
    "max": [256.0, 256.0, 128.0]
  },
  "opacity": 1.0,
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0,
  "blending_mode": "additive"
}
```

**Bounds clarification**: `center_bounds` records the tight center AABB;
`position_bounds` is the same value (centers only — chunk bounds widen per-chunk
by the ellipsoidal extent). Encoding metadata on each array carries tighter
per-array quantization bounds.

**Amplitude ranges**: `amplitude_range` (`{"min", "max"}` dict) is the
metadata bounds record; `amplitude_data_range` (`[min, max]` list, written
alongside it whenever amplitudes are given as a non-empty array — a scalar amplitude
skips it) mirrors the Points/Lines
`color_data_range` convention and seeds the viewer's layer display-range
controls. Both hold the min/max of the original (pre-quantization) amplitudes.

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
  The additive orderer's own ranking criterion — energy-ordered ladders
  front-load it, so a small prefix carries most of the energy.
- **`level_stats.reference_energy`** (per leaf): the leaf's absolute total
  self-energy *w* = `Σ aᵢ²·π^{D/2}·|Σᵢ|^½`. Disjoint partition parts sum, so
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
  "coarsen_dims": null,
  "n_substitutive_levels": 4,
  "image_min": 98.0,
  "image_max": 4095.0,
  "intensity_range": 3997.0,
  "floor": 110.0
}
```

Every fit also persists its **normalization metadata** here (routed through
the same splitter from the fit `stats` — see `gsplats/fitting/results.py`):
`image_min` / `image_max` / `intensity_range` record how the source volume
was normalized, and `floor` is the background level subtracted before
fitting (`null` when floor suppression was disabled). **Semantic contract:**
the floor is NOT added back — stored amplitudes are background-relative
(intensity above the subtracted pedestal), so renders reconstruct the
floor-suppressed volume, not the raw one.

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

extent[d] = sqrt(covariance[d, d]) * truncation_radius  # default 3.0 (3σ = 99.7%)

# Chunk bounds include extent
chunk_bounds[i, d, 0] = min(centers[chunk_i, d] - extent[chunk_i, d])
chunk_bounds[i, d, 1] = max(centers[chunk_i, d] + extent[chunk_i, d])
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

The grid resolution is **derived per-axis from the bit budget** (21 bits per
dimension, matching the `ordering_bits_per_dim: 21` leaf attr) — the
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
float32. **Cholesky factors** are stored split (diagonal + off-diagonal); bit depth
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

**Float16 compatibility** (`float16_allowed` parameter in save_gsplats()):
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

info = inspect_gsplats_zarr("fitted.gsplats.zarr")
print(info)
# GSplats: 10,000 splats, 3D
# Ordering: hilbert (bits_per_dim=21)
# Size: 1.2 MB (compression ratio: 3.2x)
# Fitting time: 45.3s, 850 iterations
```

---

## Standalone vs Embedded Formats

In v3.0 the distinction between "standalone" and "embedded" is structural only,
not semantic: a `.gsplats.zarr` IS a detached scene-node subtree, and embedding
one into a scene is a graft of that subtree.

### Standalone Format (`.gsplats.zarr`)

**Purpose**: Persist fitted results as independent, directly-loadable files.

**Structure**: A node-tree root (leaf / kind=lod / kind=partition) plus the
self-identifying header (`format_version:"3.2"`, `format_type:"gsplats_zarr"`,
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
side-by-side geometry layers (Points, Lines, GSplats).

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
| Delta encoding | No | Blosc shuffle sufficient |
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
    finest `1.0`) — device-independent LOD switching.
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
