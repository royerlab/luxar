# Luxar Zarr Format Specification

## Version: 0.1

**Features in v0.1:**
- Chunk-based spatial index for efficient nD point queries
- Points are reordered using Morton/Hilbert space-filling curves for spatial locality
- Compound ordering: discrete dimensions (time/channel) + spatial dimensions
- HDR color support (float32 values in/out; stored quantized per-channel true-log)
- Transform system with matrix transposition for THREE.js compatibility

This document specifies the Zarr-based storage format used by Luxar for high-performance 3D and nD scientific visualization.

## Overview

The Luxar Zarr format is a hierarchical data structure designed for efficient storage and streaming of large-scale scientific data with support for arbitrary dimensionality, transformations, and rendering attributes.

## End-to-End Data Flow

This diagram shows how data flows from Python creation through storage to WebGL rendering:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ PYTHON LAYER (luxar packages)                                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  luxar.core                                                                 │
│  ┌──────────────────┐                                                      │
│  │ Scene, Points    │  Define scene graph with transforms                  │
│  │ Dimensions       │  positions: float32[N, D]                            │
│  └────────┬─────────┘  colors: float32[N, 3]                               │
│           │            radii: float32[N]                                    │
│           ↓                                                                 │
│  luxar.validation                                                           │
│  ┌──────────────────┐                                                      │
│  │ Type checking    │  Validate shapes, ranges, semantic types             │
│  │ Shape validation │  Ensure data consistency                             │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ↓                                                                 │
│  luxar.encoding                                                             │
│  ┌──────────────────┐                                                      │
│  │ Semantic typing  │  COORDINATE → uint16 fixed-point (AUTO/MEM), f32 (PREC)    │
│  │ Quantization     │  COLOR → uint8 (SDR), geolog u16 (HDR)               │
│  │ Broadcasting     │  Uniform values → single scalar                      │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ↓                                                                 │
│  luxar.io                                                                   │
│  ┌──────────────────┐                                                      │
│  │ Spatial ordering │  Morton/Hilbert space-filling curves                 │
│  │ Compound sort    │  Discrete dims (time) → spatial (x,y,z)              │
│  │ Chunking         │  Split into 16KB-256KB chunks (target 64KB)          │
│  │ AABB calculation │  Per-chunk bounding boxes                            │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            ↓  Write
┌───────────────────────────────────────────────────────────────────────────┐
│ STORAGE LAYER (Zarr)                                                       │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  scene.luxar.zarr/                                                          │
│  ├── .zattrs              Scene metadata (dimensions, units, transforms)   │
│  ├── .zmetadata           Consolidated metadata                            │
│  └── node_name/                                                            │
│      ├── positions/       Blosc(zstd-9) compressed uint16 chunks           │
│      ├── colors/          Blosc compressed uint8/float32                   │
│      ├── radii/           Compressed or broadcast scalar                   │
│      └── chunk_bounds/    AABB per chunk for spatial queries               │
│                           [xmin,xmax, ymin,ymax, zmin,zmax, ...]           │
│                                                                             │
│  Compression: 2-10× (blosc/zstd) + quantization: 2-4× = 4-40× total       │
│                                                                             │
└───────────┬───────────────────────────────────────────────────────────────┘
            │
            ↓  HTTP/zarrita
┌───────────────────────────────────────────────────────────────────────────┐
│ TYPESCRIPT LAYER (luxar-viewer)                                            │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  cache/                                                                     │
│  ┌──────────────────┐                                                      │
│  │ S-cache + L0     │  Decoded slices + decompressed chunks (RAM)          │
│  │ L1: Memory LRU   │  ~100MB, ~1μs access                                 │
│  │ L2: OPFS         │  ~2GB, ~1ms access                                   │
│  │ HTTP fetch       │  Unlimited, ~100ms access                            │
│  │ Prefetcher       │  Adjacent chunks (±1 in each dimension)              │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ↓                                                                 │
│  data/                                                                      │
│  ┌──────────────────┐                                                      │
│  │ Scene loader     │  Parse .zattrs, build THREE.js scene graph           │
│  │ Spatial index    │  Query chunk_bounds for AABB intersection            │
│  │ Array decoder    │  Dequantize uint16 → float32                         │
│  │ nD slicer        │  Hypersphere visibility (effective radius)           │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ↓                                                                 │
│  rendering/                                                                 │
│  ┌──────────────────┐                                                      │
│  │ Material manager │  Create shaders with world-space sizing              │
│  │ Post-processing  │  Bloom, tone mapping, detector noise                 │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            ↓  BufferGeometry
┌───────────────────────────────────────────────────────────────────────────┐
│ WEBGL LAYER                                                                │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Vertex Shader                                                              │
│  ┌──────────────────┐                                                      │
│  │ Transform points │  Apply 4×4 matrices (scene, view, projection)        │
│  │ Size calculation │  Angular diameter → pixel size                       │
│  │ Color pass       │  Pass attributes to fragment shader                  │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ↓                                                                 │
│  Fragment Shader                                                            │
│  ┌──────────────────┐                                                      │
│  │ Gaussian kernel  │  Smooth point splatting                              │
│  │ HDR rendering    │  Float16 framebuffer for >1.0 colors                 │
│  │ Alpha blending   │  Additive/premultiplied modes                        │
│  └────────┬─────────┘                                                      │
│           │                                                                 │
│           ↓                                                                 │
│  Display: 60 FPS interactive visualization of 100K-10M elements            │
│                                                                             │
└───────────────────────────────────────────────────────────────────────────┘

**Key Performance Characteristics:**
- Python write: ~1-5M points/sec (spatial ordering overhead)
- Compression ratio: 4-40× (quantization + blosc)
- Network bandwidth: 50-500KB/sec for smooth navigation
- Cache hit rate: 80-95% with prefetching
- GPU rendering: 100K-10M elements at 60 FPS
```

## Format Structure

```
scene.luxar.zarr/
├── .zattrs                  # Scene-level metadata
├── .zgroup                  # Zarr group marker
├── .zmetadata              # Consolidated metadata (optional, created by finalize())
├── <node_name>/            # Scene nodes (groups or points)
│   ├── .zattrs             # Node-level metadata (includes spatial index metadata)
│   ├── .zgroup             # Zarr group marker
│   ├── positions/          # Point positions (required for points, spatially sorted)
│   ├── colors/             # Point colors (optional, same order as positions)
│   ├── radii/              # Point radii (optional, same order as positions)
│   ├── sharpnesses/        # Point sharpness (optional, same order as positions)
│   ├── chunk_bounds/       # Chunk bounding boxes for spatial queries (optional)
│   ├── label_offsets/      # Per-element label byte offsets, CSR-style (optional)
│   ├── label_bytes/        # Concatenated UTF-8 label strings (optional)
│   ├── image_label_offsets/ # Per-element image byte offsets, CSR-style (optional)
│   ├── image_label_bytes/  # Concatenated encoded image blobs (optional)
│   └── <child_nodes>/      # Nested child nodes (recursive structure)
└── overlays/               # Screen-space overlays (optional)
    └── <overlay_name>/     # Individual overlay
        ├── .zattrs         # Overlay metadata (type, position, style, visible_range, hover)
        ├── .zgroup
        └── image.png       # Raw image file (image overlays only)
```

### Compression & the `luxar_delta_v1` filter

All arrays use Blosc zstd level 9 with a width-aware shuffle policy (byte
shuffle for multi-byte integer codes, no shuffle for uint8/floats — see
`luxar.encoding.compression`). Additionally, any quantized uint8/uint16 code
array (positions/vertices/centers, Cholesky halves, radii/widths/amplitudes,
sharpness, colors) MAY carry the Luxar-owned zarr v2 filter
`{"id": "luxar_delta_v1", "cols": C, "bits": 8|16}` in its `.zarray`:
columnar per-chunk delta+zigzag residuals, applied probe-gated at encode time
(only where it measurably shrinks the store — 12-16% whole-store, lossless).
It is a pure storage transform below the `encoding` attrs; zarr/zarrita undo
it during whole-chunk reconstruction, so decode and random access are
unchanged. Readers need the codec registered: Python registers it on
`import luxar.encoding`; the viewer registers `numcodecs.luxar_delta_v1` in
its zarr facade. Vanilla-zarr readers without the codec fail loudly (unknown
codec), never silently. Full wire-format spec:
`docs/specs/GSPLATS_ZARR_FORMAT.md` § "The `luxar_delta_v1` delta filter".

## Scene-Level Metadata (.zattrs)

The root `.zattrs` file contains scene-wide configuration:

```javascript
{
  "luxar_version": "0.1",
  "type": "scene",
  "units": "um",  // Physical units (nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au)
  "scene_dimensions": {  // Scene-level dimension specification (REQUIRED for nD data)
    "dimensions": [
      {
        "name": "x",
        "unit": "um",
        "range": [-100.0, 100.0],  // Optional bounds
        "step": 1.0,                // Optional navigation step size
        "display": true,             // Whether dimension is displayed (max 3)
        "discrete": false,           // Whether dimension has discrete values
        "cyclic": false,            // Whether dimension wraps around
        "scale": 1.0,               // Physical scale factor
        "spatial": true,            // Whether points extend through this dimension
        "description": "X axis"     // Optional description
      },
      // ... more dimensions
    ]
  }
}
```

## Node Types

### 1. Group Nodes

Group nodes organize the scene hierarchy and can contain child nodes.

**Attributes (.zattrs):**
```javascript
{
  "type": "group",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],  // 4x4 matrix as 16-element array
  "nd_transform": {                                        // Optional, per-dimension transforms
    "Time": {"scale": 0.001, "offset": 50.0},             //   for non-displayed dimensions
    "Channel": {"permutation": [2, 1, 0]}
  },
  "opacity": 1.0,           // 0.0-1.0, inherited by children
  "gamma": 1.0,            // 0.1-10.0, per-node gamma correction
  "intensity": 1.0,        // 0.0-100.0, per-node linear color multiplier (gain)
  "offset": 0.0,           // -10.0-10.0, per-node additive brightness shift (black level)
  "blending_mode": "additive",  // normal, additive, max, opaque, luminous — written
                           //   only when explicitly set; unset ⇒ inherited from the
                           //   nearest ancestor that sets it (viewer default: additive)
  "layer": false,          // Optional: if true, node appears in the viewer's Layers panel
  "visible": true,         // Optional: initial visibility when the scene loads (default true)
  "child_index": 0         // Insertion order among siblings (stamped on add). The viewer
                           //   sorts siblings by this so the scene graph / layers panel
                           //   follow napari-style add order, not zarr's alphabetical
                           //   consolidated-metadata enumeration. Absent → enumeration order.
}
```

### 2. Group Kinds — Specialized `group` Nodes

A `Group` may carry an optional `kind` attribute that turns it into a
specialized container with viewer-aware semantics. Specialized groups
also carry a `display_type` attribute (one of `"points"`, `"lines"`,
`"gsplats"`) — the layers panel uses this for the user-facing type
label, so a layer reads as one logical entity of `display_type` rather
than as a "group".

#### `kind: "lod"` — Level-of-Detail group

Picks **one of N alternative children** at runtime based on the current
view. Each child carries a `coverage_fraction` threshold — a dimensionless,
viewport-relative value in `[0, 1]`. The viewer projects the LOD group's bbox
to screen, takes the diagonal in pixels, multiplies each child's
`coverage_fraction` by the viewport diagonal (times a small fill-factor
constant) to get a pixel threshold, and renders the **finest** child whose
threshold is satisfied (with 10% asymmetric hysteresis on the downgrade
direction to suppress flicker). Because the threshold is viewport-relative,
the finest child activates when the object roughly fills the screen —
identically on any monitor/viewport size. When the camera is inside or
straddling a group's bounding box, the group is treated as filling the screen
and its **finest** child is selected.

`kind="lod"` is **geometry-agnostic**: children can be points, lines,
gsplats, or themselves specialized groups (e.g. a Partition group inside an
LOD group). The finest child's resolved `display_type` becomes the LOD
group's `display_type`.

**Standalone `.gsplats.zarr` root**: a `kind=lod` group is also a valid
root of a standalone `.gsplats.zarr` file — the file root IS the node
(current standalone format version: **v3.2**, see
`docs/specs/GSPLATS_ZARR_FORMAT.md`). The viewer opens such a file directly
(`?src=<file>.gsplats.zarr`) and frames on its `position_bounds`. On-disk,
children are `child_<i>/` in **coarsest→finest** order; the writer always
stamps `default_level: 0` (the coarsest child) — a progressive-load hint
(render cheap first, then refine), deliberately decoupled from the data-model
default (the finest level the `.centers` accessor returns).

**Attributes (.zattrs):**
```javascript
{
  "type": "group",
  "kind": "lod",
  "display_type": "gsplats",  // Resolved at write time from the finest
                              //   child; the layers panel uses this as
                              //   the user-facing layer type.
  "selector": "coverage",     // Reserved; only "coverage" supported today.
  "default_level": 0,         // 0-based initial active level (coarsest→finest).
                              //   Seeds the "Active level" dropdown in the
                              //   Layers panel; does not lock the runtime
                              //   selector by itself.
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "nd_transform": { ... },    // Optional, same shape as on plain Group nodes
  "opacity": 1.0,             // Compositing — inherited by children
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0,
  "blending_mode": "additive",  // Only when explicitly set (unset ⇒ inherited)
  "layer": false,             // Optional: expose in the Layers panel with
                              //   an "Active level" dropdown + "N LODs" badge
  "visible": true
}
```

**Children**:
- Subgroup naming is **not** enforced; Python's convenience API writes
  `child_0`, `child_1`, … in **coarsest→finest** order, and the loader
  treats insertion order as authoritative.
- Each child's `.zattrs` MUST carry `"coverage_fraction": <float in [0, 1]>`.
  Values must be strictly monotonic increasing in coarsest→finest order;
  the coarsest is always `coverage_fraction: 0.0` (always applicable) and the
  finest is always `coverage_fraction: 1.0` (fills the screen).
- Children themselves are standard nodes — they retain their own
  `type` (`gsplats` / `points` / `lines` / `group`, possibly with their
  own `kind` attr) and full attr set.

**Builder API (Python):**
```python
# Manual:
lod = scene.add_lod_group("multires")
lod.add_gsplats_from_data("child_0", coarse_data, coverage_fraction=0.0)
lod.add_gsplats_from_data("child_1", medium_data, coverage_fraction=0.5)
lod.add_gsplats_from_data("child_2", fine_data, coverage_fraction=1.0)

# Convenience (auto-derives coverage_fractions via sqrt(N_i / N_finest), the
# per-level splat-count ratio — coarsest 0.0, finest 1.0):
scene.add_gsplats_from_data(
    "multires", flat_data,
    lod_group=dict(compression_factor=4, levels=2),
    additive_lod=dict(n_lods=4),
)
```

#### `kind: "partition"` — Spatial-decomposition group

Decomposes a single large leaf node (10M+ elements) into N smaller
children for per-child frustum culling and per-child LOD. The user
adds one node; the writer partitions it via recursive BSP at
compile time (balanced **median** split by default; `midpoint` and
`sah` rules are also available). The viewer renders all children
simultaneously (no per-frame selector — THREE's per-mesh frustum
culling does the per-part culling).

Children are **homogeneous**: every child's resolved `display_type`
must match the wrapper's (you cannot decompose a single logical
layer into mixed-type parts).

**Standalone `.gsplats.zarr` root**: a `kind=partition` group is also a
valid root of a standalone `.gsplats.zarr` file. The viewer opens it
directly (`?src=<file>.gsplats.zarr`) and frames on `position_bounds`. The
`luxar gsplat partition` CLI writes this shape via spatial BSP
(`--parts` / `--max-elements` / `--rule median|midpoint|sah`).

**Attributes (.zattrs):**
```javascript
{
  "type": "group",
  "kind": "partition",
  "display_type": "points",     // All children resolve to this type.
  "max_elements": 1000000,      // Per-part cap that drove the BSP recursion.
  "position_bounds": {           // Union of children's bboxes — lets
    "min": [-10, -10, -10],     //   picking / framing / scene-bounds-cache
    "max": [10, 10, 10]          //   treat the layer as one logical entity.
  },
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "nd_transform": { ... },      // Optional, same shape as on plain Group nodes
  "opacity": 1.0,
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0,
  "blending_mode": "additive",  // Only when explicitly set (unset ⇒ inherited)
  "layer": false,               // Optional: expose in the Layers panel with
                                //   a "N parts" badge
  "visible": true
}
```

**Children**:
- Subgroup naming is **not** enforced; the convenience kwarg writes
  `part_0`, `part_1`, … in BSP recursion order.
- Each child is a standard `points` / `lines` / `gsplats` node (or
  itself a kind=lod / kind=partition group). All must resolve to the
  same `display_type`.

**Builder API (Python):**
```python
# Convenience kwarg on the leaf adders — partition applies at compile time
# and the user never sees the wrapper unless they inspect the zarr:
scene.add_points("pts", positions, partition=True)                 # default cap
scene.add_points("pts", positions, partition=dict(max_elements=500_000))
scene.add_gsplats("splats", centers, amplitudes, cholesky,
                  partition=dict(max_elements=2_000_000, rule="median"))

# Manual (explicit tree construction):
part = scene.add_partition_group("manual",
                                 display_type="points",
                                 max_elements=500_000,
                                 layer=True)
part.add_points("part_0", subset0)
part.add_points("part_1", subset1)
```

#### Multi-additive LOD (progressive loading) — Points / Lines / GSplats

All three leaf types (Points, Lines, GSplats) support a uniform
**multi-additive-LOD** layout for progressive loading. A leaf node
with `n_additive_sublods > 1` carries `additive_<i>/` subgroups under
its path; each subgroup is a fully-formed leaf node of the same type,
carrying a subset of the parent's data plus its own spatial index.
The viewer's progressive loader concatenates loaded levels for render
and refines toward the full data over `requestAnimationFrame()`
frames once initial paint commits.

**Parent attributes (.zattrs):**
```javascript
{
  "type": "points",                 // or "lines" / "gsplats"
  "n_points": 10000,                // (or n_segments / n_splats — total across levels)
  "n_additive_sublods": 4,
  "position_bounds": { "min": [...], "max": [...] },
  "additive_lod_method": "random",  // "random" / "salience" / "spatial-uniform"
  /* compositing attrs ride here */
}
```

**Subgroup naming:** `additive_<i>/` for `i = 0..n-1`, in **coarsest →
finest** order. The convenience writer (`additive_lod=` kwarg on
`add_points` / `add_lines` / `add_gsplats_from_data`) emits the
subgroups in this convention.

**Per-type unit:**

- **Points** — per-element. Each subgroup contains a subset of
  `positions` + per-element attrs (`colors` / `radii` / `sharpnesses` /
  `scalars`).
- **GSplats** — per-element. Each subgroup contains a subset of
  `centers` / `amplitudes` / `cholesky_factors_diag` (+ `cholesky_factors_offdiag`) / `colors`.
  (Since format v3.1 the in-memory packed `cholesky_factors` is stored on disk split
  into `_diag` + `_offdiag` so each can be encoded independently; `_offdiag` is absent
  for 1D splats. Legacy v3.0 files store a single packed `cholesky_factors`, read via a
  presence-detect fallback. The current gsplats format is **v3.2**, which only
  renames the `kind=lod` selector attrs to the coverage semantics described
  above — `selector: "coverage"` + per-child `coverage_fraction`; see
  `docs/specs/GSPLATS_ZARR_FORMAT.md`, the authoritative gsplats format spec.)
- **Lines** — per-polyline. Each subgroup contains WHOLE polylines
  (vertices + their segments). Segment indices are local to the
  subgroup so topology stays valid during partial loads; the viewer
  offset-adjusts on concatenation. Supports all four `line_type`
  variants (`segments` / `polyline` / `loop` / `indexed`).

**Builder API (Python):**
```python
# Same kwarg surface across all three leaf types.
scene.add_points("pts", positions, additive_lod=True)               # 4-level default
scene.add_lines("ln", verts, widths, line_type="segments",          # explicit dict
                additive_lod=dict(n_lods=3, method="salience"))
scene.add_gsplats_from_data("splats", data,                         # gsplats kwarg
                            additive_lod=dict(n_lods=4))
```

Composition with `partition=` is "partition-outer, additive-LOD-inner": each
spatial part gets its own LOD ladder.

### 3. Points Nodes

Points nodes contain the actual point data.

**Attributes (.zattrs):**
```javascript
{
  "type": "points",
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "nd_transform": {                                        // Optional, per-dimension transforms
    "Time": {"scale": 0.001, "offset": 50.0}              //   for non-displayed dimensions
  },
  "opacity": 1.0,
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0,
  "blending_mode": "additive",  // or "normal", "max", "opaque", "luminous" — written
                           //   only when explicitly set (unset ⇒ inherited)
  "layer": false,          // Optional: if true, node appears in the viewer's Layers panel
  "visible": true,         // Optional: initial visibility when the scene loads (default true)
  "n_points": 10000,
  "max_radius": 2.5,
  "extend_to_all": ["Time", "Channel"]  // Optional: extend visibility to all values of these dimensions
}
```

**Data Arrays:**

The dtypes below describe the default `EncodingMode.AUTO`. Every array
self-describes its on-disk encoding via an `encoding` attr in its `.zattrs`
(see `luxar.encoding` and *Array Encodings* below); readers dispatch on
`encoding.name` and decode to float32 (or the array's original integer
dtype, e.g. uint8 colors stay uint8). `PRECISION` stores raw float32
everywhere; `MEMORY` quantizes more aggressively for the wide-range geolog
family (8-bit where AUTO uses 16-bit — HDR colors, wide-range positive
scalars); coordinates stay uint16 and bounded scalars pick 8-vs-16 bits from
their dynamic range identically in both modes. Uniform arrays are stored as
a single `broadcasted` value and byte-identical duplicates as an
`array_ref`, regardless of mode.

Chunking is **byte-based**, not a fixed element count: the first-dimension
chunk length is derived from the 64 KB target (`TARGET_CHUNK_BYTES` ÷
bytes-per-row for the array's *input* dtype — computed before encoding, so
float32 rows even when the stored code is uint8/uint16), or aligned to the spatial
index's `chunk_size` when spatial ordering is enabled (the default), so a
chunk-index range maps to exactly one zarr chunk.

#### positions/ (Required)
- **Shape:** `(N, D)` where N = number of points, D = dimensionality
- **Dtype/Encoding:** `uint16` per-axis fixed-point (`linear_perchannel_u16`)
  under AUTO/MEMORY — each axis quantized over its own `[min, max]` to 65536
  levels, decoded back to float32 on read (visually lossless, ~2× smaller).
  `float32` under PRECISION, or when a per-axis extent ≥ 2¹⁶ forces the
  float32 fallback (uint16 could no longer resolve a unit step).
- **Chunks:** `(chunk_rows, D)` — byte-based / spatial-index-aligned (see above)
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** Point positions in D-dimensional space (never broadcast)

#### colors/ (Optional)
- **Shape:** `(N, 3)` for RGB
- **Dtype:** `uint16` (`geolog_perchannel_u16`, HDR default) / `uint8` (SDR
  `rgb_uint8`, or HDR under MEMORY) / `float32` (PRECISION). HDR colors are
  quantized per channel on a true-log grid (uniform relative precision, code 0
  reserved for exact zeros) and decoded back to float32.
- **Chunks:** `(chunk_rows, 3)` — byte-based / spatial-index-aligned
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** HDR RGB colors in normalized range
  - **SDR Range:** 0.0-1.0 (standard dynamic range)
  - **HDR Range:** Values > 1.0 represent HDR brightness
  - **Typical HDR:** 0.0-10.0 (extreme brightness)
  - **Note:** Values are NOT in 0-255 range; use 0.0-1.0 for normal colors
- **Default:** White (1.0, 1.0, 1.0) if not provided

#### radii/ (Optional)
- **Shape:** `(N,)`
- **Dtype/Encoding:** POSITIVE_SCALAR — under AUTO, quantized to
  `bounded_scalar_uint8`/`bounded_scalar_uint16` (rescale-first, anchored at
  the array's own `[min, max]`) or, for wide dynamic range (> 65536:1),
  `geolog_scalar_uint16` (geometric-log grid, code 0 reserved for exact
  zeros). `float32` under PRECISION; `broadcasted` when uniform.
- **Chunks:** `(chunk_rows,)` — byte-based / spatial-index-aligned
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** Point radii in scene units
- **Default:** 0.5 if not provided (see `DEFAULT_POINT_RADIUS` in `core/scene.py`)
- **Validation:** All values must be positive

#### sharpnesses/ (Optional)
- **Shape:** `(N,)`
- **Dtype/Encoding:** BOUNDED_SCALAR with fixed bounds `(0.0, 1.0)` — under
  AUTO, quantized to `bounded_scalar_uint8`/`uint16`; `float32` under
  PRECISION; `broadcasted` when uniform.
- **Chunks:** `(chunk_rows,)` — byte-based / spatial-index-aligned
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** Point edge sharpness — a normalised `[0, 1]` knob. The viewer
  maps it to the super-Gaussian falloff exponent `β = 2^(6s − 2)`: `s = 0.5 → β = 2`
  (a true Gaussian), higher `s` → harder/crisper edge (β up to 16), lower `s` →
  peakier cusp (β down to 0.25).
- **Default:** 0.5 (→ β = 2, Gaussian) if not provided
- **Validation:** All values must be in `[0, 1]`

### 4. Lines Nodes

Lines nodes contain polyline/segment data. All four user-facing line types
(`segments`, `polyline`, `loop`, `indexed`) are converted to a unified
**indexed representation** at write time — a `segments` array of vertex-index
pairs — so the on-disk layout is identical for every type; the user's original
choice is recorded in `original_line_type`.

**Attributes (.zattrs):**
```javascript
{
  "type": "lines",
  "n_vertices": 10000,
  "n_segments": 9999,
  "ndim": 3,
  "original_line_type": "polyline",  // "segments" | "polyline" | "loop" | "indexed"
  "has_colors": true,
  "has_sharpness": false,
  "max_width": 1.5,
  "position_bounds": {"min": [...], "max": [...]},
  "ordering": "hilbert",             // or "morton" / "none"
  "vertex_ordering": { ... },        // Vertex spatial-index metadata (D-space)
  "segment_ordering": { ... },       // Segment spatial-index metadata (2×D-space)
  /* transform, nd_transform, opacity, gamma, intensity, offset,
     blending_mode, layer, visible — same as Points */
}
```

Lines use **dual spatial indexing**: vertices are curve-ordered in D-space
(like Points) and segments are independently curve-ordered in (2×D)-space
(concatenating both endpoints), each with its own chunk-bounds array
(`vertex_chunk_bounds` / `segment_chunk_bounds`, both
`(num_chunks, D, 2)` float32 — segment *bounds* are deliberately D-space
even though the segment *ordering* sorts in 2×D, so both support view-frustum
intersection tests directly).

**Data Arrays** (same AUTO/PRECISION/MEMORY conventions as Points; all
per-vertex arrays are reordered by the vertex sort):

#### vertices/ (Required)
- **Shape:** `(N, D)` — vertex positions
- **Dtype/Encoding:** COORDINATE, same as Points `positions/`:
  `linear_perchannel_u16` under AUTO/MEMORY (float32 under PRECISION or the
  ≥ 2¹⁶-extent fallback). Never deduplicated to an `array_ref` and never
  LUT-encoded — the lines spatial-index loader reads it as raw chunked zarr
  with no structural-encoding dispatch (grid-snapped vertices would otherwise
  store as LUT indices).

#### segments/ (Required, auto-generated)
- **Shape:** `(M, 2)` — vertex-index pairs, indices local to this node
- **Dtype/Encoding:** INDEX — stored as the smallest unsigned integer dtype
  that fits the max index (`uint8`/`uint16`/`uint32`), read raw (never
  LUT-encoded or deduplicated).

#### widths/ (Required)
- **Shape:** `(N,)` — per-vertex line widths (scene units, must be positive)
- **Dtype/Encoding:** POSITIVE_SCALAR, same rules as Points `radii/`
  (`bounded_scalar_uint8/16` or `geolog_scalar_uint16` under AUTO; float32
  under PRECISION; `broadcasted` when a scalar width is given).

#### colors/ (Optional)
- **Shape:** `(N, 3)` — per-vertex RGB, same COLOR encoding rules as Points
  (SDR → `rgb_uint8`; HDR → `geolog_perchannel_u16` under AUTO).

#### sharpnesses/ (Optional)
- **Shape:** `(N,)` — per-vertex edge sharpness, same BOUNDED_SCALAR `[0, 1]`
  rules and semantics as Points `sharpnesses/`.

#### scalars/ (Optional)
- **Shape:** `(N,)` — **per-vertex** colormap scalars (matching `widths`, not
  per-segment); declared via `has_scalars` / `scalar_data_range` / `colormap`
  attrs (see *Scalar Colormap Attributes* below).

Per-vertex labels (`label_offsets`/`label_bytes`) and image labels
(`image_label_offsets`/`image_label_bytes`) are supported with the same
CSR-style layout as Points (see *Per-Element Labels*).

## Scalar Colormap Attributes

For Points and Lines, an optional per-element `scalars` zarr array
enables colormap-driven shading. The presence and configuration are
declared through three attrs on the data node (next to `type`,
`opacity`, etc.):

```json
{
  "type": "points",
  "has_scalars": true,
  "scalar_data_range": [0.0, 1.0],
  "colormap": "viridis"
}
```

- **`has_scalars: bool`** — gates whether the loader opens the
  `scalars` zarr array. Set by the writer when a scalar array exists;
  ignored otherwise.
- **`scalar_data_range: [min, max]`** — input range used to normalize
  scalars to `[0, 1]` before the LUT lookup. Required when
  `has_scalars` is true; defaults to `[0, 1]` if omitted.
- **`colormap: 'viridis' | 'plasma' | ... | 'custom'`** — selects a
  built-in LUT (15+ available) or `'custom'` to enable a user-supplied
  LUT sibling array.

**Custom LUT**: when `colormap = 'custom'`, the scene loader looks for
a sibling array named `colormap_lut` (alongside the data node, not
nested inside `scalars`). Shape: `[256, 3]` (RGB) or `[256, 4]` (RGBA),
dtype `uint8`. The loader passes the raw bytes through to
`getColormapTexture` which builds a 256×1 DataTexture; invalid
lengths fall back to viridis with a warning. Custom LUT textures are
cached per-app with a bounded LRU (16 entries) and disposed on scene
unload (B.2 of the viewer-code-review-rerun hardening pass).

**Scalar dtype**: the `scalars` zarr array may be `float32`,
`float16`, or `uint8`. Uint8 scalars are kept in their native dtype
through the loader and accumulator and widened to Float32 at the GPU
upload boundary (A.2 of the same hardening pass).

**Per-vertex Lines**: the Lines `scalars` array is per-vertex
(matching `widths`), not per-segment. Projection interpolates between
endpoints at clipped boundaries so the LUT lookup at a slice edge
uses the correct value.

## Layers (Viewer Panel)

Any scene-graph node — `points`, `lines`, `gsplats`, or a container `group` —
may be exposed as a layer in the viewer's Layers panel by setting
`layer: true` in its zarr attrs. The panel (toggled with **L**) provides
per-layer visibility, display-range, gamma, opacity, blending mode, and
colormap controls.

```javascript
{
  "type": "points",
  "layer": true,      // Expose this node as a layer in the panel
  "visible": true,    // Optional initial visibility (default true)
  "opacity": 1.0,
  "colormap": "viridis"
}
```

### Group Layers (Composite)

A `group` node marked `layer=true` acts as a composite layer: its controls
fan out to every data descendant (points/lines/gsplats) beneath it.
Composition uses the rules described in *Rendering Attribute Composition*
below — the group's live slider value replaces its authored zarr value in
the root-to-leaf chain for each descendant.

### Initial Visibility

The `visible` attr is authoring-time only: it determines the layer's
starting state when the scene loads. Subsequent toggling is done from the
eye icon in the panel and is **not** persisted back to zarr.

### Rendering Attribute Composition

Rendering attributes compose along the scene graph (root → leaf):

- `opacity`, `gamma`, `intensity` — multiplied
- `offset` — summed
- `blending_mode` — the nearest ancestor that sets it wins

Example: a group with `opacity=0.5` and a child with `opacity=0.5` yields
an effective opacity of `0.25` for the child's material. Unset values are
identity (1.0 for multiplicative, 0.0 for additive). The viewer recomposes
on every slider change so edits to group layers flow into descendants.

### Edits Are Viewer-Only

Changes made in the panel (range, gamma, opacity, blending, colormap) are
not written back to the zarr store; reloading the page restores the
authored state.

## Overlays (Screen-Space Annotations)

Overlays are screen-space annotations (text, images, HTML) rendered over the 3D canvas.
They are stored in an `overlays/` group at the scene root. Each overlay is a zarr subgroup
with metadata in `.zattrs` and optional raw image files.

Overlays are NOT part of the 3D scene graph — they use normalized screen coordinates
`[0, 1]` with top-left origin `(0, 0)`.

### Overlay Types

**Text overlay** (`overlay_text`):
```json
{
  "type": "overlay_text",
  "text": "Scale: 10μm",
  "position": [0.05, 0.95],
  "font_size": 0.025,
  "font": "sans",
  "color": "white",
  "opacity": 1.0,
  "anchor": "top-left",
  "visible_range": {"time": [5, 10]},
  "transition": "fade",
  "transition_duration": 0.3,
  "interactive": false,
  "z_index": 0
}
```

**Image overlay** (`overlay_image`):
```json
{
  "type": "overlay_image",
  "position": [0.9, 0.05],
  "image_file": "image.png",
  "size": [0.1, 0.05],
  "blend_mode": "normal",
  "z_index": 1
}
```
The image file (PNG/JPEG/WebP) is stored directly in the overlay's zarr directory.

Note: the overlay `blend_mode` is a screen-space-overlay compositing concept
(how the 2D overlay image blends over the rendered frame) — distinct from the
scene-node attribute `blending_mode` that controls 3D geometry blending.

**HTML overlay** (`overlay_html`):
```json
{
  "type": "overlay_html",
  "position": [0.01, 0.5],
  "html": "<p>Some <strong>formatted</strong> text</p>",
  "width": 0.3,
  "interactive": true,
  "z_index": 2
}
```

### Common Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `position` | `[float, float]` | Normalized screen coords, top-left origin |
| `opacity` | `float` | 0.0 – 1.0 (default 1.0) |
| `anchor` | `string` | Positioning anchor (9 options: top-left, center, bottom-right, etc.) |
| `visible_range` | `object` | Dimension-based visibility: `{"dim_name": value_or_[min,max]}` |
| `transition` | `string` | `"none"` or `"fade"` |
| `transition_duration` | `float` | Seconds (default 0.3) |
| `interactive` | `boolean` | Whether overlay captures pointer events |
| `z_index` | `int` | Rendering order (lower = behind) |

### Dimension-Aware Visibility

The `visible_range` attribute enables overlays that appear/disappear based on dimension
slider positions. Values can be exact numbers (matched with tolerance) or `[min, max]` ranges.

## Point Spatial Index

The point spatial index enables efficient nD range queries for point visibility determination during slicing operations. Points are reordered during compilation using **Morton or Hilbert space-filling curves** to ensure spatial locality, with **chunk-based bounding boxes** for fast queries.

### Design Philosophy

The spatial index uses a simple but effective approach:
1. **Space-filling curves** (Morton/Hilbert) order points by spatial locality
2. **Compound ordering** handles mixed discrete/spatial dimensions
3. **Chunk bounding boxes** enable fast intersection queries
4. **No grid discretization** - queries use actual data bounds

### Index Structure

The spatial index stores metadata in the points group `.zattrs` and chunk bounds as a separate array:

#### Points Group .zattrs (Spatial Index Metadata)
```javascript
{
  "type": "points",
  "n_points": 100000,
  "ordering": "hilbert",          // or "morton" - space-filling curve algorithm (default: hilbert)
  "ordering_dims": [0, 1, 2],     // Indices of spatial dimensions (curve-ordered)
  "slice_dims": [3],              // Indices of discrete dimensions (lexicographic)
  "ordering_min": [-100.0, -100.0, -100.0],  // Bounds for curve normalization
  "ordering_max": [100.0, 100.0, 100.0],     // Bounds for curve normalization
  "ordering_bits_per_dim": 21,    // Bits per dimension (max 21 for uint64)
  "chunk_size": 10000,            // Points per chunk
  "max_radius": 2.5               // Maximum point radius in dataset
}
```

#### chunk_bounds/ Array
- **Shape:** `(num_chunks, D, 2)` where D = number of dimensions
- **Dtype:** `float32`
- **Chunks:** `(num_chunks, D, 2)` - stored as single chunk
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** Bounding box [min, max] for each dimension of each chunk
- **Example:** For chunk 5 in a 4D dataset: `chunk_bounds[5, :, :]` = `[[x_min, x_max], [y_min, y_max], [z_min, z_max], [t_min, t_max]]`
- **Note:** Bounds include point radii extent to ensure hyperspheres are found

#### Per-Element Labels (CSR-style)

Optional per-element string labels for hover tooltips (GPU picking). Available on all node types (points, lines, gsplats). When present, `.zattrs` includes `"has_labels": true`.

**label_offsets/** Array:
- **Shape:** `(N+1,)` where N = number of elements
- **Dtype:** `uint64`
- **Description:** CSR-style byte offsets into `label_bytes`. Label for element `i` spans bytes `[offsets[i], offsets[i+1])`.

**label_bytes/** Array:
- **Shape:** `(total_bytes,)`
- **Dtype:** `uint8`
- **Description:** Concatenated UTF-8 encoded label strings. Empty labels have `offsets[i] == offsets[i+1]` (zero-length byte range).

**Decoding:**
```
label_i = utf8_decode(label_bytes[offsets[i] : offsets[i+1]])
```

Empty strings are treated as null labels (no tooltip shown on hover). Labels are reordered to match spatial ordering if enabled.

#### Per-Element Image Labels (CSR-style)

Optional per-element **image** labels for hover thumbnails, written via the
`image_labels=` parameter of `add_points` / `add_lines` / `add_gsplats`
(accepts pre-encoded bytes, PIL images, `(H, W[, C])` uint8 numpy arrays, or
file paths; PIL images and numpy arrays are encoded to WebP, while bytes and
file contents are stored as-is — a PNG file stays PNG). When present, `.zattrs`
includes `"has_image_labels": true`.

**image_label_offsets/** Array:
- **Shape:** `(N+1,)` where N = number of elements
- **Dtype:** `uint64`
- **Description:** CSR-style byte offsets into `image_label_bytes`. Image for
  element `i` spans bytes `[offsets[i], offsets[i+1])`; empty entries have
  `offsets[i] == offsets[i+1]`.

**image_label_bytes/** Array:
- **Shape:** `(total_bytes,)`
- **Dtype:** `uint8`
- **Compression:** **None** (deliberately uncompressed — the blobs are already
  compressed JPEG/WebP/PNG; the small offsets array uses the scene default)
- **Description:** Concatenated encoded image blobs.

Image labels are reordered to match spatial ordering, like text labels.

### Hover Overlays

Overlays with `"hover": true` in their `.zattrs` act as hover tooltips. Their `text` (or `html`) field can contain template variables that are substituted by the viewer when GPU picking resolves an element:

| Variable | Description |
|----------|-------------|
| `{hover_label}` | The label string for the picked element |
| `{hover_node}` | Zarr path of the picked node (e.g., "/cells") |
| `{hover_index}` | Element index within the node |

When labels exist on any node but no hover overlay is explicitly defined, a default hover overlay is auto-injected at scene finalization time.

### Compound Ordering

For datasets with both discrete (time, channel) and spatial (x, y, z) dimensions:

1. **Primary sort**: Lexicographic on discrete dimensions
2. **Secondary sort**: Morton/Hilbert code on spatial dimensions

This ensures:
- All points for time=0 come before time=1
- Within each time slice, points are spatially ordered
- Contiguous chunks contain spatially nearby points

```
Points ordered by: (time, channel) → Hilbert/Morton(x, y, z)

time=0, channel=0: [spatially-ordered xyz points]
time=0, channel=1: [spatially-ordered xyz points]
time=1, channel=0: [spatially-ordered xyz points]
...
```

### Space-Filling Curve Encoding

**Hilbert curves** (default) provide better spatial locality with continuous traversal.
**Morton codes** (Z-order) are simpler, computed by bit-interleaving normalized coordinates:

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

### Query Algorithm

To find points visible at slice position with tolerance:

```typescript
function queryChunksForView(
  index: ChunkSpatialIndex,
  slicePosition: number[],
  tolerance: number[]
): number[] {
  const matchingChunks: number[] = [];

  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    let intersects = true;

    for (let d = 0; d < ndim; d++) {
      // Chunk bounding box (row-major layout)
      const offset = chunkIdx * ndim * 2 + d * 2;
      const chunkMin = chunkBounds[offset];
      const chunkMax = chunkBounds[offset + 1];

      // Query region
      const queryMin = slicePosition[d] - tolerance[d];
      const queryMax = slicePosition[d] + tolerance[d];

      // Non-intersection test
      if (chunkMax < queryMin || chunkMin > queryMax) {
        intersects = false;
        break;
      }
    }

    if (intersects) matchingChunks.push(chunkIdx);
  }

  return matchingChunks;
}
```

### Chunk to Point Range Conversion

```typescript
// Convert chunk indices to point ranges
function chunkIndicesToRanges(
  chunkIndices: number[],
  chunkSize: number,
  totalPoints: number
): PointRange[] {
  return chunkIndices.map(chunkIdx => ({
    start: chunkIdx * chunkSize,
    end: Math.min((chunkIdx + 1) * chunkSize, totalPoints)
  }));
}

// Merge adjacent ranges for efficient loading
// [0-100], [100-200], [300-400] → [0-200], [300-400]
function mergePointRanges(ranges: PointRange[]): PointRange[] {
  // Sort and merge overlapping/adjacent ranges
  ...
}
```

### Performance Characteristics

- **Query Time:** O(num_chunks × ndim) - linear scan of ~100-1000 chunks
- **Memory:** O(num_chunks × ndim × 2) - just bounding boxes
- **Build Time:** O(N log N) for sorting points by curve
- **Cache Efficiency:** Spatially close points are contiguous in memory

### Benefits of Chunk-Based Spatial Indexing

1. **Simple Implementation**: No complex tree structures, just bounding boxes
2. **Efficient nD Slicing**: Only load chunks intersecting the view hyperplane
3. **Radius-Aware Bounds**: Chunk bounds include point radii for hypersphere queries
4. **Memory Efficient**: Only store bounds, not per-point metadata
5. **Accurate Queries**: Uses actual data bounds, no grid discretization errors

### Backward Compatibility

Point clouds without `chunk_bounds` will load all points. The viewer detects the presence of `chunk_bounds/` and uses spatial queries when available.

### Implementation in LuxarZarrCompiler

When building points with spatial index (`enable_spatial_index=True`):

1. **Identify Dimension Types**: Classify dimensions as discrete vs spatial
2. **Compute Sort Order**: Lexsort on discrete dims, then Morton/Hilbert code
3. **Reorder All Arrays**: Apply same sort order to positions, colors, radii, sharpnesses
4. **Compute Chunk Bounds**: Calculate bounding boxes including radius extent
5. **Store Metadata**: Write ordering info to group attributes
6. **Store Bounds Array**: Write chunk_bounds array to group

## Transform System

Transforms are stored as 16-element arrays representing 4x4 homogeneous transformation matrices in **column-major order** (THREE.js format):

```
[m00, m10, m20, 0,
 m01, m11, m21, 0,
 m02, m12, m22, 0,
 tx,  ty,  tz,  1]
```

Where:
- `m00-m22`: 3x3 rotation/scale matrix (transposed)
- `tx, ty, tz`: Translation vector (at indices 12, 13, 14)
- Bottom row is always `[0, 0, 0, 1]`

**CRITICAL**: Python transposes matrices from NumPy row-major to THREE.js column-major format before storage. TypeScript consumes them directly using `Matrix4.fromArray()`.

## nD Transform System

In addition to the 4x4 spatial `transform`, nodes can carry an `nd_transform` attribute for per-dimension transforms on non-displayed dimensions. This enables time alignment, unit conversion, and category remapping without modifying stored point data.

### Storage Format

`nd_transform` is stored as a JSON object in the node's `.zattrs`, alongside the existing `transform` attribute:

```json
{
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

### Affine Entry (Continuous / Discrete Ordinal Dimensions)

```json
{
  "scale": 1.0,
  "offset": 0.0
}
```

Both fields are optional (default to `1.0` and `0.0` respectively). The effective value is computed as: `effective_value = scale * original_value + offset`. For discrete ordinal dimensions, the result is rounded to the nearest integer.

### Permutation Entry (Categorical Dimensions)

```json
{
  "permutation": [2, 1, 0]
}
```

Array of integers mapping old category indices to new indices. Length must equal the number of categories defined for that dimension. The effective index is: `effective_index = permutation[original_index]`.

### Hierarchical Composition

nD transforms compose through the parent chain, per-dimension:

- **Affine**: `composed_scale = parent_scale * child_scale`, `composed_offset = parent_scale * child_offset + parent_offset`
- **Permutation**: `composed[i] = parent_perm[child_perm[i]]` (child applied first, then parent)

Nodes without `nd_transform` (or without an entry for a specific dimension) contribute identity.

### Viewer Behavior

The viewer uses an **inverse-query** approach: instead of transforming millions of point coordinates, the slice query (position + tolerance) is inverse-transformed from world to local space once per node update. This is O(1) per dimension, not O(N) per point. No changes to loader internals are required.

See `docs/guides/specs/ND_TRANSFORMS_SPEC.md` for the full specification.

## Dimension System

### Scene Dimensions
Scene-level dimensions define the coordinate system for all objects:
- Maximum 3 dimensions can be displayed simultaneously
- Non-displayed dimensions are used for slicing/navigation
- Dimensions include metadata for units, ranges, and navigation

#### Dimension Types
Dimensions are categorized by two key properties:

1. **Spatial vs Non-Spatial:**
   - **Spatial dimensions**: Points extend through these dimensions as hyperspheres
   - **Non-spatial dimensions**: Points exist at specific values only (always discrete)
   - Displayed dimensions are always spatial by default
   - Non-displayed, non-spatial dimensions must be discrete (enforced automatically)

2. **Continuous vs Discrete:**
   - **Continuous dimensions**: Can take any value in their range
   - **Discrete dimensions**: Represent categorical data or specific values (e.g., channels, time frames)
   - Discrete dimensions use quarter-step tolerance (`step/4`) during queries

**Best Practice for Discrete Dimension Ranges:**

For discrete dimensions, the declared `range` should match the actual data extent. The viewer initializes to `rangeMin`, so if your range starts before your first data point, the initial view will show no data.

Example: If you have time frames at values [1, 2, 3, ...] but set `range=(0, N)`, the viewer initializes at 0 where no data exists. Set `range=(1, N)` instead.

The compiler will emit a warning if it detects this misalignment. For discrete dimensions with a defined step:
- Query tolerance = `step / 4` (a quarter-cell — deliberately below `step / 2`
  so chunk-bound padding plus tolerance never reaches the neighbouring
  category)
- Declared range min should be ≥ (first data point - tolerance)
- Declared range max should be ≤ (last data point + tolerance)

#### Dimension Attributes
Each dimension in `scene_dimensions` contains:
- `name`: Dimension identifier
- `unit`: Physical unit
- `range`: [min, max] values
- `display`: Whether shown in 3D view
- `discrete`: Whether dimension represents discrete values
- `spatial`: Whether points extend through this dimension (auto-determined if not specified)
- `step`: Navigation step size
- `cyclic`: (Optional) Whether dimension wraps around (e.g., angles)
- `scale`: (Optional) Physical scale factor
- `categories`: (Optional) List of string labels for categorical dimensions
- `description`: (Optional) Human-readable description

#### Categorical Dimensions
Categorical dimensions allow string labels instead of numeric coordinates, useful for channels, cell types, experimental conditions, or time-lapse phases.

**Key Features:**
- Define human-readable labels for dimension values
- Automatically enforced as `discrete=True`
- Range auto-set to `(0, len(categories)-1)` if not provided
- Step auto-set to `1.0` if not provided
- Categories must be unique, non-empty strings (max 1024 characters each)

**Example:**
```json
{
  "name": "channel",
  "unit": "",
  "categories": ["DAPI", "GFP", "mCherry", "Cy5"],
  "display": false,
  "discrete": true,
  "range": [0, 3],
  "step": 1.0
}
```

In data arrays, use integer indices (0-based): 0='DAPI', 1='GFP', 2='mCherry', 3='Cy5'.

**Validation Rules:**
- Categories must be a non-empty list of strings
- Each category label must be unique within the dimension
- Category labels cannot be empty strings
- Maximum label length: 1024 characters
- When categories are provided, `discrete` is automatically set to `True`

### nD Point Cloud Support
- Points can have arbitrary dimensionality (not limited to 3D)
- Viewer performs slicing for dimensions > 3
- Radius-based visibility for spatial dimensions: points visible if their nD hypersphere intersects the current slice
- Exact matching for discrete dimensions: only points at the exact value are shown

## Viewer Constraints and Performance

Luxar's storage format supports arbitrary-dimensional scenes, but the web viewer
uses optimized kernels with practical limits:

- **WASM-accelerated nD kernels support up to 16 dimensions.** This covers
  spatial queries, effective-radius slicing, Mahalanobis distance, and GSplat
  attenuation paths backed by fixed-size WebAssembly workspaces.
- **Datasets with more than 16 dimensions still load**, but those operations use
  the TypeScript fallback path automatically. This preserves correctness but can
  be significantly slower for large point, line, or GSplat collections.
- **For best interactive performance**, keep exported viewer scenes at 16
  dimensions or fewer when possible. For higher-dimensional source data,
  pre-slice, aggregate, or encode rarely navigated axes as categorical subsets
  before export.

## Chunking Strategy

Optimal chunk sizes balance memory usage and access patterns:
- **Target chunk payload:** 64KB (`TARGET_CHUNK_BYTES`)
- **Minimum chunk payload:** 16KB (`MIN_CHUNK_BYTES`)
- **Maximum chunk payload:** 256KB (`MAX_CHUNK_BYTES`)
- **2D arrays (positions, colors):** Chunk along first dimension only, deriving element counts from dtype and row width
- **1D arrays (radii, sharpnesses):** Simple 1D chunking, deriving element counts from dtype

### Chunking with Spatial Index

When using spatial indices:
- **Chunk Alignment**: Zarr chunks are automatically aligned with spatial index chunks
- **Typical Strategy**: `chunk_size` is computed based on target memory per chunk (~64KB, see `TARGET_CHUNK_BYTES`)
- **Benefits**: Loading a chunk index range loads exactly that zarr chunk
- **Morton Ordering**: Points within a chunk are spatially nearby due to Morton ordering

## Array Encodings

Every data array self-describes its on-disk encoding via an `encoding` attr
in its `.zattrs` (`{"name": "<scheme>", ...}`); readers dispatch on
`encoding.name` and decode back to float32 (or the original integer dtype).
The full scheme vocabulary is single-sourced in
`format-contract/contract.yaml`. Besides the quantization schemes described
per-array above (`linear_perchannel_u16`, `rgb_uint8`,
`geolog_perchannel_u16`, `bounded_scalar_uint8/16`, `geolog_scalar_uint16`,
…), three **structural encodings** are part of the reader contract:

- **`broadcasted`** — all elements share one value: the array is stored with
  shape `(1,)` / `(1, d)` and `encoding.n_elements` records the logical count
  N. Readers expand on load.
- **`array_ref`** — content-deduplication: a byte-identical duplicate of
  another array in the same store is stored as an **empty** array (physical
  shape `(0,)` / `(0, D)`) whose `encoding` carries `target` (path of the
  original array), `hash`, `original_shape`, and `original_dtype`. Readers
  must resolve and load the target array. (Structural arrays whose consumers
  read raw zarr — line `vertices`/`segments` — are never dedup- or
  LUT-encoded.)
- **`lut_uint8` / `lut_uint16`** — look-up-table encoding for arrays with few
  unique values (or few unique color rows): the array stores indices and the
  `encoding.lut` attr carries the unique values as JSON (`lut_mode` +
  `original_shape` added for 2D/row-mode). Exact (lossless); INDEX arrays
  never LUT-encode.

## Compression

Default compression is a **width-aware per-dtype Blosc policy**
(`luxar.encoding.compression`), resolved from each array's stored dtype at
write time:
- **Multi-byte integer codes** (uint16 fixed-point / quantized): zstd level 9
  with **byte SHUFFLE**
- **Single-byte codes (uint8) and floats:** zstd level 9, **no shuffle**
  (shuffle filters are no-ops or harmful for these payloads)

Bit-shuffle is deliberately not used — Blosc silently neutralises it above
level 1 at 64 KiB chunks; byte shuffle at high level is what actually engages.
Decode speed is level-independent (natively and in wasm), so the high level is
purely a write-time budget.

Supported alternatives (pass an explicit compressor to override, or `None` to
store uncompressed):
- Blosc with lz4, blosclz, snappy, zlib
- Native: gzip, bz2, lzma
- External: zstd, lz4

## Metadata Consolidation

After scene construction, call `# Context manager handles finalization automatically` to:
- Consolidate all metadata into `.zmetadata` file
- Improve load performance by reducing metadata requests
- Enable efficient streaming from remote stores

## Version History

- **0.1** (Current): Complete format with spatial index, nD support, HDR colors, transforms, scene dimensions

## Best Practices

1. **Use context managers** with LuxarZarrCompiler for automatic finalization
2. **Use appropriate chunk sizes** based on expected access patterns
3. **Store transforms at group level** for hierarchical transformations
4. **Define scene dimensions** for consistent coordinate systems
5. **Use consolidated metadata** for remote data access
6. **Validate all data** before writing to ensure consistency

## Example Creation (Python)

### Basic Example

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create scene with dimensions
dims = Dimensions([
    Dimension("x", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("z", unit="um", display=True),
    Dimension("time", unit="ms", display=False, discrete=True)
])

with LuxarZarrCompiler("output.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add points
    positions = np.random.randn(10000, 4).astype(np.float32)  # 4D points
    colors = np.random.rand(10000, 3).astype(np.float32)  # SDR colors (0.0-1.0)
    radii = np.ones(10000, dtype=np.float32) * 0.5

    scene.add_points("my_points", positions, colors, radii=radii)
    # Context manager handles finalization automatically
```

### Example with Spatial Index

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create scene with dimensions
dims = Dimensions([
    Dimension("x", unit="um", display=True, range=[-100, 100]),
    Dimension("y", unit="um", display=True, range=[-100, 100]),
    Dimension("z", unit="um", display=True, range=[-100, 100]),
    Dimension("time", unit="ms", display=False, range=[0, 10], discrete=True)
])

# Enable spatial index for efficient nD slicing
with LuxarZarrCompiler("output.luxar.zarr", enable_spatial_index=True) as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Generate 4D points
    n_points = 100000
    positions = np.random.randn(n_points, 4).astype(np.float32) * 50
    colors = np.random.rand(n_points, 3).astype(np.float32)  # SDR colors (0.0-1.0)
    radii = np.random.uniform(0.1, 2.0, n_points).astype(np.float32)

    # Add points - Morton ordering and chunk bounds computed automatically
    scene.add_points(
        "my_points",
        positions,
        colors,
        radii=radii
    )
    # Points are reordered by (time) → Morton(x,y,z) and chunk_bounds are stored
```

## Compatibility Notes

- **Zarr Version:** Format 2 (for JavaScript compatibility)
- **NumCodecs:** Required for compression support
- **Consolidated Metadata:** Recommended for web streaming
- **Browser Support:** Via zarrita.js library

## Future Extensions (Planned)

- Support for meshes, volumes
- Material system with shading models
- Temporal interpolation for smooth animations
- Multi-resolution spatial indices for LOD
