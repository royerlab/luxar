# Luxar Zarr Format Specification

> For a version-policy and migration summary that distinguishes this scene format from the gsplats format, see [Formats & Migration](./FORMAT_AND_MIGRATION.md).

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
│   ├── key_offsets/        # Per-element key byte offsets, CSR-style (optional)
│   ├── key_bytes/          # Concatenated UTF-8 key strings (optional)
│   ├── image_label_offsets/ # Per-element image byte offsets, CSR-style (optional)
│   ├── image_label_bytes/  # Concatenated encoded image blobs (optional)
│   └── <child_nodes>/      # Nested child nodes (recursive structure)
└── overlays/               # Screen-space overlays (optional)
    └── <overlay_name>/     # Individual overlay
        ├── .zattrs         # Overlay metadata (type, position, style, visible_range, hover)
        ├── .zgroup
        └── image.<png|jpeg|webp>  # Raw image file (image overlays only; exact name matches payload; bytes folded into content_hash at compile time)
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
unchanged. Readers need the codec available: in Python it auto-registers
via the numcodecs `numcodecs.codecs` entry point whenever luxar is INSTALLED
(no import needed; `import luxar.encoding` also registers it); the viewer
registers `numcodecs.luxar_delta_v1` in its zarr facade. Readers without
luxar installed fail loudly (unknown codec), never silently. Full wire-format spec:
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

Additional JSON-serializable scene metadata may be authored through
`scene.attrs`, for example `title`, `description`, or `sample`. Mutations made
while the `LuxarZarrCompiler` is open write through to the root `.zattrs`;
mutating the live mapping after finalization only updates its in-memory cache
and emits a warning.

The structured root attrs Luxar owns — `scene_dimensions`, `viewer_config`,
`citation` — each have a dedicated validating API (`Scene.dimensions`,
`Scene.viewer_config`, `create_scene(citation=...)`). Author them there, not
through `scene.attrs`: the mapping accepts free-form keys at the root, so
writing one of those by hand skips its validation and leaves the `Scene`
object's own copy stale.

### `citation` (optional root attr)

`citation` credits whoever produced the data the scene shows. It is written to
the root attributes rather than to a companion file or a web page so the
attribution travels with the store — through `luxar export`, through a copy of
the `.luxar.zarr` handed to someone without the repo, and into any viewer that
opens it.

```javascript
{
  "citation": {
    "short": "OpenCell (Cho et al. 2022); embeddings by cytoself (Kobayashi et al. 2022)", // REQUIRED: single line, what a UI renders
    "ref": "Cho / Kobayashi et al. 2022",    // optional compact caption reference, max 40 chars
    "doi": "10.1126/science.abi6983",        // optional, bare DOI (no https://doi.org/ prefix)
    "license": "CC BY 4.0",                  // optional
    "url": "https://example.org/dataset"     // optional
  }
}
```

Only `short` is required: a citation that cannot be displayed is not a
citation. `ref` is the optional compact form used where the full byline does not
fit, notably the bundled demos' bottom-right captions; it is refused above 40
characters rather than truncated. A store carrying `ref` is intentionally a
newer format-contract payload; feeding those attrs back through an older Luxar
authoring path fails citation validation because that version does not recognise
the key.

Every field must be a single line of printable text — every line break and
control character is refused, including a bare `\r` and a bidi override, since
a credit that renders differently from the string that was stored is a spoofing
risk rather than a cosmetic one. `doi`, when present, must match the full
`10.<registrant>/<suffix>` shape, so the near-misses people paste (a URL, a
`doi:` prefix, a registrant whose suffix was lost) are refused rather than
stored. `url`, when present, must be `http://` or `https://`: it is the one field
a UI turns into a link, and a citation travels inside data that is copied and
published onward, so an executable or inline-payload scheme is not storable
here. The key is **absent** — not `null`, not an empty object — when the scene
owes no credit, which is the normal case for procedurally generated data.
Readers should therefore treat a missing `citation` as "unknown or not
applicable" and never as a claim that the data has no author.

Written by passing `citation=` to `LuxarZarrCompiler.create_scene(...)`. The
payload is validated at write time (`luxar.core.citation.validate_citation`), so
a malformed citation raises instead of being baked into every copy of the data.

### `incomplete` (optional root attr)

`incomplete` (boolean) is written to the root `.zattrs` only when the writer
aborted before `finalize()` completed — an exception or Ctrl-C propagated out of
the `LuxarZarrCompiler` context, or `finalize()` itself failed midway. A
successful `finalize()`
never leaves this marker. `LuxarScene.load` refuses to load a store carrying
`incomplete: true`, since it may be missing nodes or consolidated metadata.

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
  "absorption": 1.0,       // volumetric mode's kappa (>= 0, default 1.0; multiplicative)
  "gamma": 1.0,            // 0.1-10.0, per-node gamma correction
  "intensity": 1.0,        // 0.0-100.0, per-node linear color gain — but on a
                           //   colormapped node this is the scalar display window,
                           //   not a gain (see Scalar Colormap Attributes)
  "offset": 0.0,           // -10.0-10.0, per-node additive brightness shift (black level)
  "blending_mode": "additive", // normal, additive, max, opaque, luminous, volumetric — written
                           //   only when explicitly set; unset ⇒ inherited from the
                           //   nearest ancestor that sets it (viewer default: additive)
  "layer_order": 20,           // optional: authored cross-layer draw order
                               // (higher = nearer the camera). Never stamped —
                               // absence means "use the inferred ordering".
  "colormap": "viridis",   // Optional palette; nearest-setter-wins for rendering. GSplats
                           //   inherit it directly; Points / Lines / Mesh scalar leaves must
                           //   still author their own colormap today (see Rendering Attribute
                           //   Composition)
  "layer": false,          // Optional: if true, node appears in the viewer's Layers panel
  "visible": true,         // Optional: initial visibility when the scene loads (default true)
  "child_index": 0         // Insertion order among siblings (stamped on add). The viewer
                           //   sorts siblings by this so the scene graph / layers panel
                           //   follow napari-style add order, not zarr's alphabetical
                           //   consolidated-metadata enumeration. Absent → enumeration order.
}
```

#### Volumetric blending and `absorption`

Set `blending_mode` to `"volumetric"` for emission–absorption compositing:
each element emits light while attenuating elements behind it. The viewer
supports this mode for Points, Lines, and GSplats and depth-sorts the
order-dependent geometry back-to-front.

`absorption` is the node-level coefficient κ. It is non-negative, defaults to
`1.0`, and composes multiplicatively through the scene hierarchy. It is inert
in the other blending modes. At κ = 0, volumetric rendering reaches the
additive limit; increasing κ produces stronger self-occlusion. When colors
carry an RGBA alpha channel, the alpha is converted to optical-depth weight in
volumetric mode instead of being treated only as a linear contribution scale.

For the rendering equations, blend-state contract, and per-geometry details,
see the [Volumetric Blending specification](../specs/VOLUMETRIC_BLENDING_SPEC.md).

### 2. Group Kinds — Specialized `group` Nodes

A `Group` may carry an optional `kind` attribute that turns it into a
specialized container with viewer-aware semantics. Specialized groups
also carry a `display_type` attribute (one of `"points"`, `"lines"`,
`"gsplats"`) — the layers panel uses this for the user-facing type
label, so a layer reads as one logical entity of `display_type` rather
than as a "group".

#### `kind: "lod"` — Level-of-Detail group

Picks **one of N alternative children** at runtime based on the current
view. Each child carries a `coverage_fraction` threshold; the group's
`selector` attr names the UNITS. Under `selector: "screen-area"` (what every
auto-derived ladder stamps) a threshold is a literal **screen-area fraction**:
the viewer projects the LOD group's bbox to a screen-space rect, takes that
rect's area as a fraction of the viewport area, and renders the **finest**
child whose threshold is satisfied (with 10% asymmetric hysteresis on the
downgrade direction to suppress flicker). The derived whole-object ladder is
`[0, …, 1/8, 1/4, 1/2]` — full detail while the object occupies at least half
the screen, one level coarser per halving of occupied area — identically on
any monitor/viewport size (the metric is built from NDC fractions; the rect
is clipped to the viewport, off-screen reads 0, and sub-pixel-thin content
ramps to its clipped linear span — see the normative metric definition in
`docs/specs/GSPLATS_ZARR_FORMAT.md`). Under the
legacy `selector: "coverage"` (older stores, and explicitly authored
`coverage_fractions=[...]` lists) the thresholds are diagonal-metric units in
`[0, 4]`: the viewer compares them against the projected bbox diagonal over
`FILL_FACTOR=0.5 ×` the fitted screen axis (`min(width, height)` — the extent
the camera framing actually fits). Under a PERSPECTIVE camera, a group whose bounds
reach the camera's near plane has no meaningful projection (the homogeneous
divide degenerates), so both selectors saturate to the **finest** child; an
ORTHOGRAPHIC projection never degenerates, so the ordinary clipped metric
applies directly (a camera inside a large group still reads full coverage
naturally — see the normative rules in `docs/specs/GSPLATS_ZARR_FORMAT.md`).

`kind="lod"` is **geometry-agnostic**: children can be points, lines,
gsplats, or themselves specialized groups (e.g. a Partition group inside an
LOD group). The finest child's resolved `display_type` becomes the LOD
group's `display_type`.

**Standalone `.gsplats.zarr` root**: a `kind=lod` group is also a valid
root of a standalone `.gsplats.zarr` file — the file root IS the node
(current standalone format version: **v3.4**, see
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
  "selector": "screen-area",  // Units of the children's coverage_fraction:
                              //   "screen-area" (derived ladders) or
                              //   "coverage" (legacy diagonal metric).
  "default_level": 0,         // 0-based initial active level (coarsest→finest).
                              //   Seeds the "Active level" dropdown in the
                              //   Layers panel; does not lock the runtime
                              //   selector by itself.
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "nd_transform": { ... },    // Optional, same shape as on plain Group nodes
  "opacity": 1.0,             // Compositing — inherited by children
  "absorption": 1.0,          // volumetric mode's kappa (>= 0, default 1.0; multiplicative)
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
- Each child's `.zattrs` MUST carry a `"coverage_fraction"` in the group's
  selector units (`[0, 1]` for `screen-area`; `[0, 4]` for legacy `coverage`).
  Values must be strictly monotonic increasing in coarsest→finest order, and
  the coarsest is always `coverage_fraction: 0.0` (always applicable).
  A **whole-object** ladder (the auto-derivation: SCREEN-OCCUPANCY HALVING,
  independent of element counts) anchors its finest at `coverage_fraction:
  0.5` — full detail while the object occupies at least half the screen — with
  each coarser level halving the threshold (`…, 1/8, 1/4, 1/2`).
  The `1.0` fills-screen ceiling (the tile alone occupying the whole screen)
  holds a level until the node is larger still.
  That fills-screen anchor is the right one whenever a **spatial partition** is
  part of the switch, because a tile's projected rect is intrinsically a
  fraction of the whole object's; a whole-object-anchored ladder there would put
  every tile on its finest level while the object is merely full-frame. Every
  producer emits it automatically once it can see the binding: the `adaptive` /
  `overview` gsplat recipes; the two gsplat writers' topology-aware fallback (a
  `kind=partition` crossed on the way down); scene-side adders that detect a
  `kind=partition` ancestor; and `add_points(partition=...,
  substitutive_lod=...)`, which verifies a multi-part fine branch directly. In
  that overview topology the group's bbox is the whole object, so the anchor is
  the recipe contract (coarse at opening frame, fine on zoom), not tile geometry.
  An **explicitly authored**
  `coverage_fractions=[...]` list always wins over all of them. The rule assumes
  >= 2 parts. Every producer that can see the final sibling count excludes a
  **one-part** partition and falls back to the whole-object `0.5` anchor —
  `--recipe adaptive`, both gsplat writers, and the Points overview composition
  do, which matters because a dataset below `--max-elements` yields exactly that
  shape. The partition-ancestor scene-adder route cannot check it (part 0's
  ladder is derived before part 1 exists);
  the compiler's finalize pass warns when it sees the result, without rewriting it.
- A child MAY carry `"lod_bounds": {"min": [...], "max": [...]}` with the
  same nD axis order and shape as `position_bounds`. This is a producer-chosen
  robust extent (for example percentile bounds that exclude a sparse tail), used
  **only** to size the node for either LOD selector metric. Frustum gating,
  eviction, framing, clipping, and scene ranges continue to use the complete
  `position_bounds`, so excluded outliers remain visible and resident when they
  should. The robust bound must stay within `position_bounds`; a bound that is
  not contained is rejected and the child falls back to `position_bounds`.
  Missing or malformed `lod_bounds` fall back to that child's `position_bounds`;
  producers should therefore stamp every child in a ladder when they want one
  consistent robust extent. Decimation, culling, and filtering must recompute or
  remove the derived bound. Producer-side authoring policy is tracked in #1655.
- Children themselves are standard nodes — they retain their own
  `type` (`gsplats` / `points` / `lines` / `group`, possibly with their
  own `kind` attr) and full attr set.

**Builder API (Python):**
```python
# Manual (hand-authored thresholds default to the legacy
# selector="coverage" diagonal units; pass selector="screen-area" to author
# literal screen-area fractions):
lod = scene.add_lod_group("multires")
lod.add_gsplats_from_data("child_0", coarse_data, coverage_fraction=0.0)
lod.add_gsplats_from_data("child_1", medium_data, coverage_fraction=0.5)
lod.add_gsplats_from_data("child_2", fine_data, coverage_fraction=1.0)

# Convenience (auto-derives coverage_fractions by screen-occupancy halving —
# selector="screen-area": coarsest 0.0, halving up to the finest's
# half-screen-area anchor 0.5):
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
that intersect the camera frustum simultaneously. A frustum-only
per-frame selector also gates fetch and eviction for off-screen parts;
unlike an LOD selector, it never substitutes one child for another.

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
  "bsp_tree": {                 // Optional recursive tree; axis is a position-column
    "axis": 0,                  //   index mapped through displayDims by the viewer;
                                //   any unmapped split axis rejects the whole tree.
    "split": 0.0,
    "left": { "part": 0 },
    "right": { "part": 1 }
  },
  "position_bounds": {           // Union of children's bboxes — lets
    "min": [-10, -10, -10],     //   picking / framing / scene-bounds-cache
    "max": [10, 10, 10]          //   treat the layer as one logical entity.
  },
  "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],
  "nd_transform": { ... },      // Optional, same shape as on plain Group nodes
  "opacity": 1.0,
  "absorption": 1.0,            // volumetric mode's kappa (>= 0, default 1.0; multiplicative)
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0,
  "blending_mode": "additive",  // Only when explicitly set (unset ⇒ inherited)
  "layer": false,               // Optional: expose in the Layers panel with
                                //   a "N parts" badge
  "visible": true
}
```

For points and Gaussian splats, each split plane exactly separates the child
bounds. Lines and mesh are partitioned atomically by polyline and face centroid,
respectively, so their vertices may cross a split plane; their stored tree is a
stable approximate order rather than an exact painter's-order separation.

**Children**:
- Subgroup naming is **not** enforced; the convenience kwarg writes
  `part_0`, `part_1`, … in BSP recursion order. Names may have gaps when an
  empty region is omitted; the contiguous `child_index` attr is the identity
  used by `bsp_tree` leaves and the viewer.
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

`max_elements` and `rule` are the only accepted keys in a `partition` dict;
unknown keys raise an error before any node is written.

#### Multi-additive LOD (progressive loading) — Points / Lines / GSplats

All three leaf types (Points, Lines, GSplats) support a uniform
**multi-additive-LOD** layout for progressive loading. A leaf node
with `n_additive_sublods > 1` carries `additive_<i>/` subgroups under
its path; each subgroup is a fully-formed leaf node of the same type,
carrying a subset of the parent's data plus its own spatial index.
The viewer's progressive loader concatenates loaded levels for render
and refines toward the full data over `requestAnimationFrame()`
frames once initial paint commits.

Mesh uses the same subgroup layout but is documented separately (see the
*Mesh Node* section): its levels are a partition of the FACES, each
re-indexing its own vertex table, so a level is not a subset of the parent's
arrays and carries no spatial index — and the ladder is a *reveal*, restricted
to orderings whose every prefix is one connected patch.

**Parent attributes (.zattrs):**
```javascript
{
  "type": "points",                 // or "lines" / "gsplats"
  "n_points": 10000,                // (or n_segments / n_splats — total across levels)
  "n_additive_sublods": 4,
  "position_bounds": { "min": [...], "max": [...] },
  /* compositing attrs ride here */
}
```

**Subgroup naming:** `additive_<i>/` for `i = 0..n-1`, in **coarsest →
finest** order. The convenience writer (`additive_lod=` kwarg on
`add_points` / `add_lines` / `add_gsplats_from_data`) emits the
subgroups in this convention.

**Ladder provenance:** the ordering method that built the ladder is not a
top-level attr — it rides in the quality-stamp dicts, as `lod_method` inside
the parent's `level_stats` and inside each subgroup's `lod_stats` (alongside
`lod_level` / `lod_n_lods` / `lod_breakpoints_kind`). Readers treat absence as
"unstamped".

**Per-type unit:**

- **Points** — per-element. Each subgroup contains a subset of
  `positions` + per-element attrs (`colors` / `radii` / `sharpnesses` /
  `scalars`).
- **GSplats** — per-element. Each subgroup contains a subset of
  `centers` / `amplitudes` / `cholesky_factors_diag` (+ `cholesky_factors_offdiag`) / `colors`.
  (Since format v3.1 the in-memory packed `cholesky_factors` is stored on disk split
  into `_diag` + `_offdiag` so each can be encoded independently; `_offdiag` is absent
  for 1D splats. Legacy v3.0 files store a single packed `cholesky_factors`, read via a
  presence-detect fallback. Format **v3.2** renamed the `kind=lod` selector
  attrs to the coverage semantics described above — `selector: "coverage"` +
  per-child `coverage_fraction`; **v3.3** added the optional `luxar_delta_v1`
  filter on quantized code arrays. The current format is **v3.4**, which adds
  the `selector: "screen-area"` mode (literal screen-area-fraction thresholds,
  stamped by every derived ladder); see
  `docs/specs/GSPLATS_ZARR_FORMAT.md`, the authoritative gsplats format spec.)
- **Lines** — per-polyline. Each subgroup contains WHOLE polylines
  (vertices + their segments). Segment indices are local to the
  subgroup so topology stays valid during partial loads; the viewer
  offset-adjusts on concatenation. Supports all four `line_type`
  variants (`segments` / `polyline` / `loop` / `indexed`).

**Labels (Points / Lines):** per-element string labels are NOT stored per
subgroup. No single level's array is what a pick index addresses, since the
viewer's loader concatenates the levels it has loaded into one buffer — so the
label CSR (`label_offsets` + `label_bytes`) lives on the **parent** node, which
consequently carries `has_labels: true`, and the `additive_<i>` subgroups carry no
label arrays and no `has_labels`. Index `k` of the parent CSR is the `k`-th
element of the concatenation of `additive_<i>` in order (coarsest → finest), each
level in its own stored (spatially reordered) order — the same on-disk index space
a flat labelled leaf's CSR uses, just spanning the levels. Labels are
all-or-nothing across a ladder: a partially-labelled ladder is rejected at write
time.

The viewer commits levels coarsest-first, so a fully-loaded ladder maps straight
through — index `k` is committed slot `k`. The **committed buffer** is not in
general a prefix of this union, though: the per-level loader compacts out elements
culled by the current nD slice and fetches only the chunk ranges a query
intersects, so slots shift. For **Points** that shift is corrected: the viewer
composes each level's own visible-slot → on-disk-index map (the map issue #1421 /
PR #1425 introduced for **flat** nodes) into this union index space, offsetting
level `i` by the preceding levels' on-disk counts, so a labelled Points ladder
resolves exactly under an nD slice too (issue #1439). Where the levels' metadata is
inconsistent the viewer publishes no map at all and falls back to the raw committed
slot, rather than composing an index it cannot trust. For **Lines** no map is
composed across the levels, so a laddered lines node still resolves at the raw
committed slot.

Two further notes. Under the documented `partition=`-outer +
`additive_lod=`-inner composition the CSR lands on each `part_<i>` ladder parent,
which is exactly where the viewer looks it up (the hit leaf scene node — the
outermost `kind=partition` wrapper is the reported path only), so that composition
resolves too, through the same per-geometry path as an unpartitioned ladder. And for Lines the CSR is
per-vertex, matching the flat Lines writer, while the viewer's Lines pick id is a
per-segment storage slot. Issue #1424 bridged that granularity for **flat** Lines
nodes — a labelled flat lines node resolves the picked segment's slot back to that
segment's start vertex row in the stored ordering — but it does so through the same
visible-slot → on-disk-index map a **lines** ladder does not publish, so across a lines
ladder the hover only lands on the right string when every element carries the same one
(there is no broadcast label form — `labels` is always one entry per element). #1439
carried that map over the levels for Points only; a laddered Lines node is still open.

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
spatial part gets its own LOD ladder. **Points and Lines only.** On gsplats the
combination is refused: the writer every laddered gsplats leaf goes through
(`add_gsplats_multi_lod_impl`) has no partition step at all, so
`add_gsplats_from_data(..., additive_lod=..., partition=...)` raises rather than
silently dropping one of the two, and so does `add_gsplats_from_file` on a store
whose leaves already carry ladders (`gsplat lod --recipe tiles|overview|adaptive`,
`batch-fit merge --recipe stream`). Collapse those with `gsplat flatten` if you
need to re-partition them — or pass `additive_lod=False`, which flattens each
level to a single sub-LOD in memory and partitions normally, on either door.
Partitioned gsplats ladders are still *readable*:
the paragraph above describes what the pipeline itself produces, which the viewer
resolves the same way it resolves a Points one.

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
  "absorption": 1.0,       // volumetric mode's kappa (>= 0, default 1.0; multiplicative)
  "gamma": 1.0,
  "intensity": 1.0,
  "offset": 0.0,
  "blending_mode": "additive",  // or "normal", "max", "opaque", "luminous", "volumetric" —
                           //   written only when explicitly set (unset ⇒ inherited)
  "layer": false,          // Optional: if true, node appears in the viewer's Layers panel
  "visible": true,         // Optional: initial visibility when the scene loads (default true)
  "n_points": 10000,
  "max_radius": 2.5,
  "extend_to_all": ["Time", "Channel"]  // Optional: extend visibility to all values of these dimensions
}
```

The on-disk `extend_to_all` value is **always a resolved list of dimension
names**, never the `"all"` sentinel the Python `add_*` API accepts: the sentinel
is expanded to the scene's non-displayed dimension names before anything is
written (including onto each `additive_<i>/` sub-LOD of a laddered node). The
viewer relies on that invariant and reads the attr as a `string[]`.

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
float32 rows even when the stored code is uint8/uint16). When spatial
ordering is enabled (the default), each per-element array's first-axis chunk is
sized to its own dtype byte budget rounded down to a **multiple** of the
spatial index's `chunk_size` atom (never below one atom) — so a chunk-index
range always falls inside a whole zarr chunk, and a large scene issues far
fewer requests because most arrays pack several index chunks per zarr chunk.
This applies to Points, Lines and GSplats arrays alike, and to the *standalone*
`.gsplats.zarr` tree writer too — it shares the same writer, which is what keeps
a scene leaf and a standalone leaf byte-identical (see
`tests/test_scene_leaf_parity.py`). GSPLATS_ZARR_FORMAT.md §8 specifies the same
per-array sizing.

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
- **Shape:** `(N, 3)` for RGB or `(N, 4)` for RGBA
- **Dtype:** `uint16` (`geolog_perchannel_u16`, HDR default) / `uint8` (SDR
  `rgb_uint8`, or HDR under MEMORY) / `float32` (PRECISION). HDR colors are
  quantized per channel on a true-log grid (uniform relative precision, code 0
  reserved for exact zeros) and decoded back to float32.
- **Chunks:** `(chunk_rows, 3|4)` — byte-based / spatial-index-aligned
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** HDR RGB colors in normalized range
  - **SDR Range:** 0.0-1.0 (standard dynamic range)
  - **HDR Range:** Values > 1.0 represent HDR brightness
  - **Typical HDR:** 0.0-10.0 (extreme brightness)
  - **Note:** Values are NOT in 0-255 range; use 0.0-1.0 for normal colors
  - **Alpha (optional 4th channel):** per-point opacity α ∈ [0, 1] (never
    HDR; the SDR/HDR autodetect scans RGB only). Every blending mode scales a
    point's contribution by α; `volumetric` maps it into optical depth
    w = −ln(1−α) — see VOLUMETRIC_BLENDING_SPEC.md §5.4.1. No format-version
    bump: readers key off the array shape, and codecs are channel-agnostic.
- **Default:** White (1.0, 1.0, 1.0) if not provided

#### radii/ (Optional)
- **Shape:** `(N,)`
- **Dtype/Encoding:** POSITIVE_SCALAR — under AUTO, quantized to
  `bounded_scalar_uint8`/`bounded_scalar_uint16` (rescale-first, anchored at
  the array's own `[min, max]`) or, for wide dynamic range (> 65536:1),
  `geolog_scalar_uint16` (geometric-log grid, code 0 reserved for exact
  zeros). `float32` under PRECISION; `broadcasted` when uniform.
- **Deduplication:** `false` — chunk bounds depend on this array's own
  positive-scalar quantization grid.
- **Chunks:** `(chunk_rows,)` — byte-based / spatial-index-aligned
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** Point radii in scene units
- **Default:** 0.5 if not provided (`DEFAULT_POINT_RADIUS` in
  `typing_utils/constants.py`, mirrored in the viewer's `config/constants.ts`).
  The same value is what the viewer draws a radii-less node with and what the
  spatial index expands a no-radii chunk's bounds by.
- **Validation:** All values must be positive
- **Shader contract:** radii do NOT scale with the node's `transform` — a
  node-level scale repositions point centers but leaves the rendered disc
  size unchanged (size attributes are applied after the model transform).
  This is deliberate and shared with Lines `widths/`; gsplats differ
  (their covariances transform with the node). Bake the desired world
  size into the radii themselves when scaling a node.

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

Joint continuity is defined by shared **indices**, not equal coordinates. Two
segment endpoints stored as separate vertex rows remain independent even when
their coordinates match, so connected thick curves should use `polyline` or
`indexed` authoring with every joint referenced through one shared vertex row.

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
  "join": "miter",                   // Optional, LINES ONLY: join style at
                                     //   degree-2 polyline joints —
                                     //   "miter" (default) or "none".
                                     //   Unset ⇒ inherited, then the viewer
                                     //   default. Compositing: set it on the
                                     //   layer, not on internal children.
  /* transform, nd_transform, opacity, absorption, gamma, intensity, offset,
     blending_mode, layer, visible — same as Points */
}
```

`join` selects what the vertex stage does where two segments of a polyline
meet. Without join geometry the turn leaves an uncovered circular sector on the
outside of the bend and a double-covered lens inside — dark ticks along the
convex edge of a thick curve, bright ticks along the concave one. `"miter"`
rotates each quad's end edge onto the shared miter edge so the two TILE:
coverage becomes a partition, so there is nothing to sum and every blending
mode is correct by construction. It is gated in-shader by a rendered-width
threshold (a sub-pixel wedge is invisible, and thin lines are the
million-segment ones) and by a miter limit, so `"none"` is rarely worth
authoring. A session-wide `?lineJoin=none|miter` overrides whatever the file
says. Unrecognised values are rejected at write time rather than silently
meaning "no joins".

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
- **Shader contract:** like Points `radii/`, widths do NOT scale with the
  node's `transform` — a node-level scale repositions vertices but leaves
  the rendered line width unchanged.

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

Per-vertex labels (`label_offsets`/`label_bytes`), keys
(`key_offsets`/`key_bytes`), and image labels
(`image_label_offsets`/`image_label_bytes`) are supported with the same
CSR-style layout as Points (see *Per-Element Labels* and *Per-Element Keys*).
Because the string channels are per-vertex while the viewer picks whole
*segments*, hover and selection on a lines node report the picked segment's
**start** vertex. Two consequences follow from that convention: on a segment
that the current slice clips only partially the reported start vertex may lie
entirely outside the visible slab (what is drawn starts at the clipped position,
not at the stored vertex), and **any vertex that is never a segment's start is
unreachable by hovering** — its label, key, or image label can never be read.

Which vertices those are depends on `original_line_type` (the segment pairs are
built by `luxar.io._ordering.lines.convert_to_indexed`):

- **`segments`** — the pairs are consecutive disjoint vertices `(0,1)`, `(2,3)`,
  …, so **every odd-numbered vertex** (in authored order) is only ever an end:
  half of the label array is unreachable. Author the label you want shown on the
  even-numbered vertex of each pair.
- **`polyline`** — the pairs are `(0,1)`, `(1,2)`, …, so only the final vertex is
  unreachable.
- **`loop`** — the last vertex connects back to the first, so every vertex starts
  a segment and all labels are reachable.
- **`indexed`** — whichever subset the supplied `indices` never place first,
  plus any vertex no segment references at all.

### 5. Mesh Nodes

Mesh nodes contain triangle-surface data — isosurfaces, segmentation boundaries,
organ and cortical meshes. They are the only node type that describes a
*connected, opaque surface* rather than a set of soft per-element primitives.

✅ **Writable and renderable.** Mesh nodes are written, read and reported by
`luxar info`, and the viewer loads, shades and picks them too — the whole
vertical ships today (see `docs/specs/MESH_NODE_SPEC.md` §11). The format contract
still distinguishes the two: `geometry_types` (the writable leaf vocabulary) and
`loader_types` (the viewer-drawable subset) — and both now include `mesh`.

Two structural differences from the other three types:

- **No per-element size.** A triangle's extent comes from its own vertices, so
  there is no `radii` / `widths` / `cholesky_factors` analogue — and a mesh
  contributes **zero extent padding** to `position_bounds`.
- **Topology is load-bearing.** `faces` is an index array, so unlike a
  coordinate or colour array it cannot tolerate lossy or aliasing encoding: it is
  written with dedup and LUT encoding **disabled** (see below).

**Attributes (.zattrs):**
```javascript
{
  "type": "mesh",
  "n_vertices": 10000,
  "n_faces": 19996,
  "ndim": 3,
  "has_normals": true,
  "normal_dims": [0, 1, 2],          // REQUIRED iff has_normals — see below
  "has_colors": true,
  "has_scalars": false,
  "has_uvs": true,
  "has_texture": true,
  "texture_encoding": "raw",         // "raw" | "png" | "webp" | "jpeg" | "ktx2"
  "texture_width": 2048,
  "texture_height": 1024,
  "texture_channels": 3,
  "texture_color_space": "srgb",     // "srgb" | "linear"
  "shading": "smooth",               // "smooth" | "flat" | "none"
  "double_sided": true,
  "position_bounds": {"min": [...], "max": [...]},
  "ordering": "none",                // always "none" in v1 (no spatial index)
  // ... plus the standard render attrs (opacity, gamma, intensity, offset,
  //     absorption, blending_mode, colormap, layer, transform, nd_transform,
  //     extend_to_all) and mesh-only appearance attrs (ambient, shade_exponent,
  //     specular, shininess, alpha_cutoff, texture_filter, texture_wrap),
  //     plus slab_tolerance, plus the opt-in material family:
  "material": "physical",            // "luxar" (default when absent) | "physical"
  "roughness": 0.4,                  // physical knobs, each in [0, 1], written
  "metalness": 1.0,                  //   only when authored; absent = three's
  "clearcoat": 1.0,                  //   own default
  "clearcoat_roughness": 0.1,
  "iridescence": 0.0,
  "sheen": 0.0,
  "sheen_color": "#ffffff",          // "#rrggbb"
  "transmission": 1.0,               // the glass family: [0, 1]
  "ior": 1.5,                        //   [1, 2.333]
  "thickness": 0.4,                  //   >= 0, scene units
  "attenuation_color": "#f6d148",    //   "#rrggbb"
  "attenuation_distance": 0.3,       //   > 0, scene units; absent = none
  "dispersion": 0.5                  //   >= 0
}
```

The seven house-shader appearance attrs control shading and texture sampling;
`slab_tolerance` controls nD membership loading; `material` selects the material
family and unlocks the thirteen physically based knobs
(`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md`). All twenty-two mesh-only
authored attrs are rejected on points, lines, Gaussian splats, and groups. A
physical knob without `material: "physical"` is refused at authoring; a
physical mesh refuses `ambient` / `shade_exponent` / `specular` / `shininess`,
`blending_mode`, `colormap`, a texture and `shading: "none"`; and `thickness`,
`attenuation_color`, `attenuation_distance` and `dispersion` are refused without
a `transmission` above zero — none of these pairings has a meaning.

#### vertices/ (Required)
- **Shape:** `(V, D)` — nD vertex positions, exactly like `Lines.vertices`.
- **Encoding:** `COORDINATE`, written with `deduplicate=false` and
  `allow_lut=false`. The loader reads it as raw chunked zarr and does not resolve
  `array_ref`, so dedup would silently drop geometry for a byte-identical sibling,
  and LUT encoding of grid-snapped coordinates would decode as garbage.

#### faces/ (Required)
- **Shape:** `(F, 3)` — triangle vertex indices.
- **On-disk dtype:** ⚠️ **any unsigned integer width — a reader must not assume
  `uint32`.** `uint32` is the writer's canonical logical dtype, but the `INDEX`
  encoder narrows integer arrays losslessly by observed value range, so a
  4-vertex mesh stores `uint8` and a 60k-vertex one `uint16`. The narrowing is
  reversible (the original dtype is recorded in the array's `encoding` attr) and
  verified exact in every encoding mode, but a consumer that hardcodes `uint32`
  will reject valid stores. Accept any integer dtype and widen on read.
- **Encoding:** `INDEX`, `deduplicate=false`, `allow_lut=false` — same reasoning
  as `Lines.segments`. Note `allow_lut=false` matters here beyond the raw-read
  argument: grid-structured index values are exactly what LUT encoding targets, so
  without it a regular mesh would be a prime candidate for it.
- **Winding:** counter-clockwise as seen with the mesh's authored spatial triple
  in ascending index order. For a 3D mesh that triple is `[0,1,2]`; for an nD mesh
  it is `sorted(normal_dims)` when normals are present. No winding can be
  counter-clockwise under *every* 3D projection of an nD mesh, so the viewer
  restores front-facing winding only when the displayed set matches that frame.
- Writers must keep every index in `[0, n_vertices)`. An out-of-range index is
  not a rendering artefact: it reads past the vertex buffer, which aborts the
  whole WASM module in the Rust culling kernel.

#### normals/ (Optional)
- **Shape:** `(V, 3)` — **always** 3-component, even for an nD mesh.
- **Encoding:** `COORDINATE` (per-axis `uint16` over each component's own
  `[-1, 1]` range — a free 2× over float32 — and it correctly blocks
  broadcasting, since a normal is always per-vertex).
- **Paired with a required `normal_dims` attr.** Normals are a *display-space*
  quantity, meaningful only for the three displayed dimensions, so the store must
  record which three they describe. ⚠️ Do **not** store normals against an
  implicit "first three dimensions": for a `(t, x, y, z)` mesh those are
  `(t, x, y)` and such a normal is meaningless. The viewer uses stored normals
  only when `shading == "smooth"` **and** `normal_dims` equals the active
  `displayDims`, and otherwise computes flat face normals from the projected
  triangle — so a wrong-but-well-formed triple degrades shading rather than
  corrupting it, while an ill-formed one is rejected at write time.
- Zero-length normals are **warned about, not rejected**: degenerate triangles
  legitimately produce them and the renderer epsilon-guards its `normalize`.

#### colors/ (Optional)
- **Shape:** `(V, 3)` or `(V, 4)` — per-vertex RGB or RGBA, same encoding rules as
  Points. The optional 4th component is per-vertex **opacity**.

#### scalars/ (Optional)
- **Shape:** `(V,)` — per-vertex colormap scalars; declared via `has_scalars` /
  `scalar_data_range` / `colormap` (see *Scalar Colormap Attributes* below).

Per-vertex labels (`label_offsets`/`label_bytes`), keys
(`key_offsets`/`key_bytes`), and image labels
(`image_label_offsets`/`image_label_bytes`) use the same CSR-style layout as
Points (see *Per-Element Labels* and *Per-Element Keys*).

**Not written for a mesh node:** no spatial index (`ordering` is always `"none"`).

#### nD slicing: whole-triangle cull

A mesh slices differently from the other three geometry types, and the difference
is visible. Points, Lines and GSplats are collections of independent elements, so
slicing keeps or drops each element on its own — and Lines goes further, *clipping*
a segment that straddles the slice and interpolating its attributes at the cut. A
triangle cannot be handled that way cheaply: cutting one against an nD slab yields
a polygon that has to be re-triangulated, with new vertices and interpolated
attributes, every frame the slice moves.

Luxar does not do that. The rule is:

> A triangle is drawn **iff all three of its vertices** fall inside the slice slab.

Two consequences follow, and both are worth knowing before you author a mesh with
hidden dimensions.

**A cut surface has a ragged edge.** Because whole triangles are kept or dropped,
the boundary follows triangle edges rather than the slice plane. On a
well-tessellated surface sliced with a slab comparable to its edge length this
reads as a slightly jagged edge. On a *coarse* mesh with a thin slab it can drop
whole regions — if no triangle has all three vertices inside, nothing is drawn.

**On a continuous hidden dimension you get a slab, not a section.** There is no
interpolation, so there is no such thing as an exact cross-section: what you see is
"the surface near this slice", of finite thickness. The viewer says so once per
node, by name, in an `info` log line.

`slab_tolerance` is the control over that thickness:

```python
scene.add_mesh(
    "surface", vertices, faces,
    normals=normals, normal_dims=[0, 1, 2],
    slab_tolerance=2.5,   # vertices within ±2.5 cells (a 5-cell slab); default 1.0
)
```

It is a half-width measured in cells of the hidden dimension's own `step`, must
be strictly positive, and defaults to one cell. Raising it thickens the slab
(more surface shown, more of it away from the slice); lowering it thins the slab
toward the degenerate case above. It applies **only** to *continuous* hidden dimensions — a
*discrete* one (time, channel, or any axis with `categories`) uses a half-cell
membership rule instead and ignores the attr. Discrete hidden dimensions are the
dominant real case for a mesh, and they have none of the problems in this section:
a timepoint either matches or it does not.

Mesh is the only geometry type whose slab is tunable, and the reason is that it has
nothing to measure. The other three derive their tolerance from a per-element
extent — a point's `radii`, a line's `widths`, a splat's truncated `sigma` — that a
mesh vertex simply does not have, so the thickness is chosen rather than read off
the data.

If your mesh's hidden dimensions are continuous *and* spatial, and you need a true
planar section, a mesh node is the wrong representation today — fit the volume as
Gaussian splats instead, which slice exactly. Exact nD triangle clipping is a
deliberate non-goal for now; see `docs/specs/MESH_NODE_SPEC.md` §5 and §9.


**Additive sub-LOD subgroups (`additive_<i>/`) ARE written, but only for a reveal.**
A prefix of an index buffer is a holed surface, not a coarse one, so the ladder is
not a level of detail for a mesh and `add_mesh(additive_lod=…)` accepts only
`method="radial"` — a concentric-shell *reveal*, whose every prefix is a
contiguous partial surface. Three consequences are visible on disk: each level
re-indexes its own gathered vertex table (so the parent's `n_vertices` exceeds the
source count by the boundary duplication, exactly as `kind=partition` parts do);
no level carries `energy_fraction_cum` and the parent carries no
`reference_energy`, because a reveal prefix is a partial object at full brightness
rather than a dim version of the whole; and `has_labels`, `has_keys`, and
`has_image_labels` are all **unset**, because one source vertex maps to a slot in
every level that touches it, so a union annotation CSR spanning levels has no
well-defined index space. Annotated meshes take `substitutive_lod=` or
`partition=`, both of which keep their per-element annotations. A mesh **may**
be a child of a `kind=partition` group;
`add_mesh(partition=…)` writes exactly that, with each part carrying its own
gathered-and-renumbered vertex table (vertices on a cut are duplicated between
neighbouring parts). A mesh may equally be a child of a `kind=lod` group —
`add_mesh(substitutive_lod=…)` writes that shape, with each coarse level a decimated
copy of the surface. The two cannot be combined in one call.

## Scalar Colormap Attributes

For Points, Lines and Mesh, an optional per-element `scalars` zarr array
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

**`intensity`/`offset` on a colormapped node are the display window, not a
gain.** When a colormap is active, an authored `intensity`/`offset` defines
the scalar display *window* (the value→LUT mapping) exactly as the Layers
panel's range control does — the post-LUT color gain stays at identity, so
the value is never applied twice. A *non-identity* leaf-authored
`intensity`/`offset` pair therefore *replaces* `scalar_data_range` as the
window (`window = [-offset/intensity, (1 - offset)/intensity]`). The
decision is by value, matching the Layers panel: an explicitly authored
identity pair (`intensity: 1.0`, `offset: 0.0`) behaves exactly like an
unauthored one and keeps the `scalar_data_range` window. An ancestor-only
gain is instead folded onto the declared `scalar_data_range`. This keeps the
load-time render identical to the post-interaction (Layers-panel) render, and
mirrors how gsplat nodes treat their `amplitude_data_range`. A direct-color
node with no colormap still treats `intensity`/`offset` as an ordinary
post-shading gain.
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

## Environment (baked scene lighting)

A `material: "physical"` mesh is lit by the viewer's scene environment
(`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3). Two root-level
things describe it:

**`viewer_config.environment`** (optional; Python `EnvironmentConfig`):

```json
"environment": {
  "source": "scene",           // "room" (default) | "scene" | "hdri"
  "probe": "auto",             // "auto" | "node:<path>" | [x, y, z]
  "resolution": 128,           // cube face size for a "scene" capture, 16-1024
  "intensity": 1.0,            // scene.environmentIntensity, >= 0
  "url": "env/studio.hdr"      // "hdri" only; store-relative or absolute HTTP(S)
}
```

`"scene"` makes the viewer capture the environment from the scene itself (an
exact `CubeCamera` render from the probe), so metals and glass reflect the data
they sit in. House-shaded meshes, points, lines and splats never read the
environment, so the block changes nothing about them.
An HDRI URL is limited to 2048 characters; absolute URLs must use HTTP(S) and
must not contain embedded credentials. Protocol-relative and active-content
schemes such as `data:` or `javascript:` are refused.

**`environment/` group** (optional; written by `luxar env bake` / `luxar env
attach`, never by the compiler): the six captured cube faces, prefiltered by the
viewer at load in milliseconds so a published scene pays no live capture.

```
environment/                        # a SIDECAR: no `type`, no `kind` attr
├── zarr.json                       # attrs, below
└── faces-3f9a1c02/                 # (6, H, W, 4) uint16 — IEEE half-float bits, RGBA
```

```json
{
  "format": "cube-faces-half",
  "faces": "faces-3f9a1c02",        // the LIVE array; a re-bake is a NEW name
  "sample_format": "half-float-bits",
  "shape": [6, 128, 128, 4],
  "face_order": ["px", "nx", "py", "ny", "pz", "nz"],   // three's CubeTexture order
  "coordinate_system": "webgl",     // the backend that captured it
  "probe": {"spec": "auto", "position": [0.0, 0.0, 0.0]},
  "resolution": 128,
  "scene_content_hash": "…",        // the root content_hash it was baked against
  "appearance": {"viewer_config": {}},
  "baked_at": "2026-09-06T00:00:00Z",
  "viewer_version": "…",
  "content_hash": "…"               // the GROUP's own digest (tooling only)
}
```

Three rules make it safe. The group carries neither `type` nor `kind`, so the
viewer's node discovery skips it as a metadata sidecar; `LuxarScene.nodes` and
`luxar info` consult `RESERVED_ROOT_GROUPS`, while `luxar optimise` and scene
hashing skip `ENVIRONMENT_GROUP` directly. The compiler refuses a user node named
`environment`. The group is **excluded from the scene `content_hash`**, so
attaching a map never changes the root digest: the `scene_content_hash` guard is
exact (the viewer ignores a map whose digest is not the root's, saying so in the
console), a visitor's warm cache survives a bake, and attaching the same map
twice writes nothing. And the faces array is named by its own digest, so a
re-bake is a new path a caching viewer cannot serve stale. `uint16` rather than
`float16` because the viewer's zarr reader needs a `Float16Array` for `<f2`
while the GPU readback and three's half-float cube texture already speak half
bits. `luxar optimise` copies the array verbatim and restamps its
`scene_content_hash` to the output scene's new digest; `luxar restamp-lod`
likewise updates the stamp after changing the scene digest (`luxar info` does
not list the group).

## Layers (Viewer Panel)

Any scene-graph node — `points`, `lines`, `gsplats`, `mesh`, or a container
`group` — may be exposed as a layer in the viewer's Layers panel by setting
`layer: true` in its zarr attrs. The panel (toggled with **L**) provides
per-layer visibility, display-range, gamma, opacity, absorption (volumetric
mode's κ), blending mode, and colormap controls, plus mesh shading controls
(ambient, shade falloff, specular, shininess, alpha cutoff) — or, for a
`material: "physical"` mesh, live sliders for its physical knobs in place of
those sliders (the two colour knobs are read-only rows). The mesh-only appearance attrs also include `texture_filter`
and `texture_wrap`; all of them are valid only on mesh leaves and do not inherit
through groups.

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
fan out to every data descendant (points/lines/gsplats/mesh) beneath it.
Composition uses the rules described in *Rendering Attribute Composition*
below — the group's live slider value replaces its authored zarr value in
the root-to-leaf chain for each descendant.

### Initial Visibility

The `visible` attr is authoring-time only: it determines the layer's
starting state when the scene loads. Subsequent toggling is done from the
eye icon in the panel and is **not** persisted back to zarr.

### Rendering Attribute Composition

Rendering attributes compose along the scene graph (nearest data/group root → leaf):

- `opacity`, `absorption`, `gamma`, `intensity` — multiplied (`absorption`
  has identity 1.0, is floored at 0, and has no upper clamp)
- `offset` — summed
- `blending_mode`, `join`, `colormap`, `layer_order` — the nearest ancestor that
  sets it wins (`join` is lines-only; a `colormap='custom'` carries its sibling
  `colormap_lut` bytes down with the name, and a leaf that names a different
  palette does *not* inherit those bytes)

The scene root is a carrier and is excluded from this composition chain. Put a
scene-wide palette on a Group containing the geometry rather than on the scene
root. A standalone `.gsplats.zarr` root is a data node, not a scene root, so its
palette does compose into its children.

Example: a group with `opacity=0.5` and a child with `opacity=0.5` yields
an effective opacity of `0.25` for the child's material. Unset values are
identity (1.0 for multiplicative, 0.0 for additive). The viewer recomposes
on every slider change so edits to group layers flow into descendants.

**Producers: set `blending_mode` on the layer, never on its internal
children.** Unlike the multiplicative attrs, it is *nearest-setter-wins*, so a
copy stamped on a `kind=partition` part or a `kind=lod` child **shadows** the
layer that contains it. The layers panel exposes one Blend control per layer,
so a shadowed layer's control silently does nothing. The Python writers route
it (with the other compositing attrs) onto the wrapper only — see
`COMPOSITING_ATTRS` in `core/group/compositing.py`. Correspondingly, within a
layer's own subtree the panel treats the layer's mode as authoritative and
ignores a mode authored on a non-layer descendant.

**`layer_order` — the authored cross-layer draw order.** An integer stating
where a layer draws relative to the layers it overlaps: **higher = nearer the
camera = drawn later**, the CSS `z-index` / Illustrator convention. Optional and
**never stamped** — its absence on disk is genuine silence, which is
load-bearing: the viewer treats an unset level as band 0, so an authored `0`
orders identically, but the panel's blank `auto` state and authored-band
diagnostics must still distinguish those states. Different band values, not
explicitness itself, override bounding-sphere containment inference.

Layers with different levels never interleave, whatever the camera does, which is
the point: it converts an inferred, geometry-dependent order into a stated one.
Sparse values (10/20/30) leave room to insert a layer later. Negative values are
fine. Two things it deliberately does not do: it cannot reorder across the
opaque/transparent render split (an `opaque` mesh always draws before any
transparent layer), and it buys *stability* rather than correctness — for two
concave interpenetrating layers no single order is right from every viewpoint.

Like `blending_mode`, set it on the **layer**, never on a `kind=partition` part
or `kind=lod` child. Unlike `blending_mode` this is not merely discouraged but
**refused** by the Python writers, through both the adder kwarg and a post-hoc
`node.attrs[...]` assignment: a partition's parts are ordered exactly against
each other from their stored BSP planes, and splitting the wrapper across bands
would destroy that. A level on the wrapper moves the whole block while the part
order travels with it intact. Full design:
`docs/guides/specs/LAYER_ORDER_SPEC.md`.

An inherited `colormap` is offered to every descendant at render time, but the
current Python adders still require Points / Lines / Mesh scalar leaves to
author a `colormap` themselves. Those types require a scalar channel
(`has_scalars`) and otherwise keep rendering direct colours, while a
GSplats leaf is always colormap-capable — its amplitude *is* the scalar — so an
ancestor's palette overrides even per-splat colours there. This matches what the
layers panel's colormap dropdown already does when it fans a palette out over a
group. Note the corollary: the writers stamp the implicit `colormap="gray"` on a
colorless gsplats leaf *only* when no ancestor authored a palette, since a
manufactured value nearer the leaf would shadow the authored one.

For GSplats, author the group palette before adding its children (normally by
passing `colormap=...` to `add_group`). The writer consults ancestors while each
leaf is created; assigning `group.attrs["colormap"]` afterwards does not
retroactively remove a gray already stamped on existing leaves.

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
The image file is stored directly in the overlay's zarr directory. Compiler-written
overlays use exactly `image.png`, `image.jpeg`, or `image.webp`, matching the PNG,
JPEG, or WebP payload bytes. These canonical names avoid case-insensitive metadata
collisions for compiler-written overlays; hand-authored stores must still follow the
payload-name rules below. The image bytes are folded into the `content_hash` at
compile time, so two builds differing only in the image get different hashes; editing
the file inside an already-finalized store restamps nothing, since nothing re-hashes
on the fly.
For a payload name that differs from a zarr metadata document only by case
(`Zarr.json`, `.ZATTRS`, ...), the authored spelling must appear exactly in the
store's immediate-child listing before any read. Otherwise it is not treated as
a payload and contributes the deterministic `absent:` hash term, with no bytes
read. Ordinary non-colliding names retain the store's native lookup semantics.

The viewer fetches `overlays/<name>/<image_file>` directly. A case-insensitive
static host may therefore return the real metadata document for a dangling
case-shifted name even though the compiler hashes it as absent; the result is a
broken image response, not a valid overlay. Producers should avoid all payload
names that collide case-insensitively with zarr metadata documents.

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

Note: the viewer sanitizes `html` against both a tag allowlist and an attribute
allowlist at render time, so hand-authored values are still constrained. A tag
outside the allowlist is unwrapped — it disappears while its children are kept
(the one exception is `<template>`, whose payload lives in an inert `.content`
fragment the sanitizer never walks, so its subtree is dropped rather than kept).
Only the attributes `style`, `href`, `src`, `alt`, `class`, `target`, `title`,
`rel`, `colspan`, `rowspan`, `width`, and `height` survive; everything else is
dropped, including `on*` event handlers, `id`/`name`, `data-*`, `ping`,
`srcset`, and `download`. On the kept attributes, `href`/`src` values using the
`javascript:`, `vbscript:`, or `data:` schemes are removed, and a `style` value
carrying `javascript:`, `vbscript:`, `expression(`, a backslash (CSS escapes
can smuggle those tokens past a text check), or a `/*` comment opener is
dropped. Author overlay markup with basic formatting, links, lists, tables,
and images.

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
  "ndim": 3,                      // dimensionality of `positions`
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

#### Ordering-metadata shape: flat vs namespaced

A geometry type with **one** ordering writes the ordering keys **flat** on the
group (`ordering`, `ordering_dims`, `slice_dims`, `ordering_min`,
`ordering_max`, `ordering_bits_per_dim`, `chunk_size`). Points and GSplats both
do this and share that set.

On a **gsplats** group `slice_dims` is not only descriptive: the viewer reads it to
classify each hidden dimension's chunk-fetch tolerance (its barrier dims got a tight
epsilon pad in `chunk_bounds` rather than the `truncation_radius · σ` expansion),
instead of inferring that from the scene dimensions' `discrete` flags. The on-disk
format is unchanged — a store stamping no `slice_dims` gets the old inference.

A type with **more than one** ordering namespaces each into its own nested
object instead, keeping a flat top-level `ordering` naming the curve. Lines is
the only such type today: it indexes vertices in D-space and segments in
(2×D)-space, so it writes `vertex_ordering` and `segment_ordering`.

Follow the same rule for any new geometry type — flat for a single ordering,
namespaced objects for several. `ordering` itself always stays flat, so a
reader can identify the curve without knowing the type's index count.

An **absent** `ordering` attr means the same as `"none"`: the node is
unordered. Producers may omit it rather than writing `"none"` explicitly, so
consumers must treat missing and `"none"` identically.

#### chunk_bounds/ Array
- **Shape:** `(num_chunks, D, 2)` where D = number of dimensions
- **Dtype:** `float32`
- **Chunks:** `(num_chunks, D, 2)` - stored as single chunk
- **Compression:** Blosc with zstd, level 9 (width-aware shuffle policy)
- **Description:** Bounding box [min, max] for each dimension of each chunk
- **Example:** For chunk 5 in a 4D dataset: `chunk_bounds[5, :, :]` = `[[x_min, x_max], [y_min, y_max], [z_min, z_max], [t_min, t_max]]`
- **Note:** Bounds include point radii extent to ensure hyperspheres are found. A node with no `radii/` array is bounded by `DEFAULT_POINT_RADIUS` (0.5) — the radius it will be drawn at — not by zero. Discrete/barrier axes get no radius extent at all, only a tiny float-boundary epsilon, so a categorical value never bleeds into its neighbour. **A stored interval is never tighter than the chunk's footprint of the AUTHORED coordinates, at any coordinate magnitude**: every pad is a small *absolute* quantity that would fall under half a float32 ULP past `|x| ~ 2**23`, so producers accumulate the interval in float64 and narrow it to this float32 array by rounding each end *away* from the interval (a bound moves one ULP outward only when the cast moved it the wrong way, so a padless axis still stores its coordinates exactly). Consumers may rely on containment; they may not assume the bound is tight.
- **Note (quantised coordinates — points and lines):** for **points and lines** the containment guarantee above holds against the coordinates *as decoded*, not merely as authored, and that takes an extra pad. `chunk_bounds` is always float32, but the coordinate arrays themselves are stored as **per-axis uint16 fixed point** under the default AUTO encoding, so a decoded coordinate can land up to half a quantum (`extent/131070`) away from the authored one on a **non-gridded** axis — 7.6e-3 at an axis extent of 1000, far above the float32 ULP the outward store closes, and enough for a reader to skip the chunk at that edge. Points and lines therefore widen every chunk bound outward by that per-axis half-quantum (on top of the radius/width footprint on a spatial axis, and on top of the float-boundary epsilon on a barrier axis); an axis the encoder stores exactly gets nothing extra, which covers a **gridded** axis (a stacked integer time/channel axis, snapped so its values round-trip bit-exactly), a constant axis and the ≥2¹⁶-extent float32 fallback. An array that is *actually* stored as a LUT (verbatim values, ≤256 distinct) is exempt too — but eligibility is not enough: **lines `vertices/` never store a LUT** (the spatial-index loader reads that array as raw chunked zarr, so the writer blocks LUT there), so a LUT-eligible lines node is quantised like any other and its vertex *and* segment bounds still get the pad. Stores written before this pad existed (2026-08) keep their old, occasionally-too-tight bounds.
- **Note (quantised coordinates — gsplats):** gsplat containment comes from the encoder's per-axis round-trip slack alone: every spatial and barrier bound is widened by the full possible center displacement, so it contains decoded `centers/` without relying on the splat's σ. On a non-gridded uint16 axis that slack is the half-quantum (`extent/131070`), added on top of the geometric footprint or float-boundary epsilon; a gridded time/channel axis, LUT encoding, or float32 store answers zero and leaves the bound unchanged. The separate sigma rail is a **fidelity** guard: when a uint16 grid can move too many centers beyond their own cores it escalates the whole array to float32, but its population tolerance is not part of the containment contract. Stores written before this pad existed (2026-08) keep their old, occasionally-too-tight bounds.
- **Note (quantised radii and widths):** a point's `radii/` and a line's `widths/` are themselves quantised (`bounded_scalar_uint8` under AUTO for a typical range), so their decoded value can exceed the authored value by roughly half a quantum. Points and segment bounds therefore add the positive-scalar encoder's single-array round-trip slack to the radius/width pad on **spatial dimensions only**. Barrier dimensions still receive no geometric footprint pad. Stores written before this scalar pad existed (2026-08) keep their old, occasionally-too-tight bounds.
- **Note (quantised Cholesky factors):** a gsplat's packed `cholesky_factors` input is split on disk into `cholesky_factors_diag/` and `cholesky_factors_offdiag/`, while its footprint pad is still built from the **authored** values. The diagonal uses `log_perchannel_u8` under AUTO, so decoded σ can exceed the authored `coverage_sigma · σ` pad (measured decoded − authored σ up to 1.99e-2, putting 11 of 11 gsplat chunks outside their stored bound, worst 5.10e-2 after the 2.75× coverage factor). This remaining extent gap is not closed yet.

#### Per-Element Labels (CSR-style)

Optional per-element string labels for hover tooltips (GPU picking). Available on all four geometry node types (points, lines, gsplats, mesh — per-vertex for lines and mesh). When present, `.zattrs` includes `"has_labels": true`. For lines the picked unit is a *segment*, so hover/selection reports the label of that segment's **start** vertex.

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

One exception to "the CSR sits on the node carrying the elements": a
multi-additive-LOD Points / Lines node stores a single CSR on the **parent**,
spanning its `additive_<i>` subgroups (which carry none) — see the **Labels**
paragraph in the "Multi-additive LOD (progressive loading)" section above.

#### Per-Element Keys (CSR-style)

Optional per-element **machine-readable** strings, for `link` / `copy` templates
to substitute as `{hover_key}`. Same encoding, same reordering and the same
all-or-nothing ladder rule as **Per-Element Labels** above — only the array
names and the presence attr differ. When present, `.zattrs` includes
`"has_keys": true`.

- **key_offsets/** Array — uint64, shape `(N+1,)`: byte offset of each key
- **key_bytes/** Array — uint8: concatenated UTF-8 encoded key strings

**Decoding:** `key_i = utf8_decode(key_bytes[offsets[i] : offsets[i+1]])`, with
`offsets[i] == offsets[i+1]` meaning "no key for this element".

Keys exist because a label and a link key are usually *different strings*. A
label is composite prose a reader sees on hover (`"P04637 · DNA-binding
cluster"`); a URL needs the bare id (`"P04637"`). Folding one into the other
means either a degraded tooltip or an unusable link, and in several datasets the
id is not present in the label at all. Keys are also far cheaper than storing a
whole URL per element: a 6-character accession is ~7 bytes against ~45, and the
viewer decodes the entire CSR into memory on first hover.

Keys are independent of labels: a node may carry either, both, or neither. A
`link` built purely from `{hover_index}` needs no strings at all.

```python
scene.add_points(
    "proteins", positions,
    labels=[f"{acc} · {cluster}" for acc, cluster in ...],  # what the tooltip shows
    keys=accessions,                                        # what the URL uses
    link="https://www.uniprot.org/uniprotkb/{hover_key}/entry",
)
```

#### Per-Element Image Labels (CSR-style)

Optional per-element **image** labels for hover thumbnails, written via the
`image_labels=` parameter of `add_points` / `add_lines` / `add_gsplats` / `add_mesh`
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
| `{hover_key}` | The machine-readable key for the picked element (see *Per-Element Keys*). Empty when the node carries none. |
| `{hover_label}` | The label string for the picked element |
| `{hover_node}` | Zarr path of the picked layer (e.g., "/cells") |
| `{hover_index}` | Element index within the node that was hit (on-disk index or buffer slot — see below). For a **lines** node carrying per-vertex labels it is the picked segment's start-vertex row in the stored (spatially ordered) vertex arrays — line labels are per-vertex and a segment carries a single pick id, so its start endpoint is the one reported. |

When labels exist on any node but no hover overlay is explicitly defined, a default hover overlay is auto-injected at scene finalization time.

### Element Interaction Templates

The same substitution vocabulary — including `{hover_key}`, backed by the
**Per-Element Keys** CSR — drives two per-**node** attrs that make a picked
element clickable. They live on the geometry node (not on an overlay),
because a scene has effectively one hover overlay but many layers, and the
target is a property of the data:

| Attribute | Type | Description |
|-----------|------|-------------|
| `link` | `string` | URL template. Opened on left-click (no drag). Must resolve to an absolute `http`/`https` URL. |
| `copy` | `string` | Plain-text template offered as `Copy` in the right-click menu. Defaults to `{hover_label}` when absent and the node has labels. |
| `link_target` | `string` | `"_blank"` (default) or `"_self"`. |

```python
scene.add_points(
    "organs", positions, labels=organ_names,
    link="https://en.wikipedia.org/wiki/Special:Search?search={hover_label}",
    copy="{hover_label}",
)
```

Right-clicking a picked element opens a menu with `Copy "<text>"` plus, when a
`link` resolves, `Open link in new tab` and `Copy link address`. Nothing is
shown when the element offers neither.

Element actions currently require a settled hover pick, so taps on touch-only
devices do not trigger them.

**Substitution and escaping differ by consumer.** Values interpolated into a
`link` are percent-encoded, so a label may contribute *content* to the URL but
never *structure* — a label containing `/`, `?`, `#` or `&` cannot add a path
segment, query or fragment. Values interpolated into `copy` are not escaped:
plain text is the point. Tooltip escaping is unchanged.

**A referenced placeholder that resolves empty suppresses the action** rather
than leaving a hole. `https://example.org/{hover_label}` on an element with no
label would otherwise become `https://example.org/`, a valid URL to the wrong
place. This is the normal case at coarse levels of a `substitutive_lod=`
ladder, whose synthesised gsplat levels inherit the node attrs but carry no
labels.

Both the Python writer and the viewer validate a `link`: the scheme must be
`http` or `https` (an allowlist — the viewer *navigates* to this value, and
`.zattrs` is untrusted input), the URL must be absolute (a relative one would
resolve against whatever origin the viewer is served from), it must not embed
credentials (`https://good.example@evil.example/` reads as one host and goes to
another, including in `Copy link address`), and it is length-capped. `link_target` is restricted to the two keywords that imply `noopener`;
any other value would be a *named* browsing context, which the browser opens
with a live `window.opener` the destination could use to navigate the viewer
tab. Links open with `noopener,noreferrer`.

Viewers can refuse links entirely — `?no-links`, or `allowLinks: false` in the
embedder options. That suppresses navigation, the two link menu items and the
pointer cursor, while leaving `Copy` working.

Under a `kind=partition` layer, `{hover_node}` and `{hover_index}` are reported
against different nodes and are not directly joinable: `{hover_node}` is the
outermost partition wrapper (the layer the user sees, matching the layers panel),
while `{hover_index}` is local to the `part_<i>` leaf that was hit. For a
non-partitioned node both refer to the same node.

Whether `{hover_index}` is the *on-disk* element index — the one the node's
arrays and its label CSR are keyed by — depends on the geometry. It is the
on-disk index for a **flat** Points, GSplats or Lines node that declares
`has_labels`, `has_image_labels`, or `has_keys`, and for a Mesh always, since
mesh picking reports the vertex's on-disk ordinal directly. **Lines** takes the longest route to get
there: a line is drawn one instance per *segment* and picking reports that
segment's slot, while the lines label CSR is written per *vertex*, so a labelled
lines node resolves the slot all the way back to the picked segment's start
vertex in the stored ordering (#1424). A lines node with no labels publishes no
such mapping and still reports the raw visible-segment slot, which after spatial
range loading or an nD slice compacting invisible elements out is not an on-disk
row at all. A multi-additive-LOD (laddered) node carries its label CSR on the
ladder parent, which is the node that declares `has_labels`. A laddered
**Points** node resolves like a flat one: the viewer composes each level's map
into that union CSR's index space (issue #1439), so `{hover_index}` is the
on-disk row under culling and compaction too — it degrades to the raw committed
slot only when the levels' own metadata is inconsistent. A laddered **Lines**
node publishes no mapping across its levels, so `{hover_index}` there is the raw
committed slot, which is a per-*segment* one against the per-*vertex* union CSR
and so wrong at the granularity whatever the slicing (see the **Labels**
paragraph of the multi-additive-LOD section above).

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

For a discrete dimension, numeric stops are anchored at the declared range
minimum: `range[0] + k * step`. This keeps offset exports such as
`range=[1, 11], step=5` directly navigable at `1, 6, 11`; `step` is not an
absolute zero-anchored modulus. This changes the interpretation of existing
stores whose `range[0]` is not on their authored data grid: recompile those
stores, or update `range[0]` to an on-grid value, before serving them with a
viewer that uses this contract.

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
- **Chunk Alignment**: Zarr chunk boundaries always land on the spatial-index grid. Points, Lines and GSplats arrays each size their first-axis chunk to the array's own dtype byte budget, rounded down to a multiple of the `chunk_size` atom (never below one atom). One zarr chunk therefore holds several whole index chunks, and no index chunk's row range ever straddles a zarr chunk boundary. Standalone `.gsplats.zarr` shares the same writer and the same layout.
- **Typical Strategy**: `chunk_size` is computed based on target memory per chunk (~64KB, see `TARGET_CHUNK_BYTES`)
- **Benefits**: Every chunk-index row range falls inside a whole zarr chunk, and arrays pack several index chunks per zarr chunk — far fewer HTTP requests on large scenes
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
  N. For **color** arrays, `encoding.original_dtype` records the numpy dtype
  string of the stored value (e.g. `uint8`, `uint16`) so readers restore the
  native integer color dtype and normalize it — rather than reading raw 0-255
  floats. Non-color arrays (radii/sharpness/amplitudes) omit it and decode as
  float32. Readers expand on load.
- **`array_ref`** — content-deduplication: a byte-identical duplicate of
  another array in the same store is stored as an **empty** array (physical
  shape `(0,)` / `(0, D)`) whose `encoding` carries `target` (path of the
  original array), `hash`, `original_shape`, and `original_dtype`. Readers
  must resolve and load the target array. (Structural arrays whose consumers
  read raw zarr — line `vertices`/`segments` — are never dedup- or LUT-encoded;
  point `radii` and line `widths` are never deduplicated either, because their
  chunk bounds depend on their own quantization grids.)
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

- **Zarr format on disk:** **3 for new stores, 2 for existing ones — both are
  readable and neither is being rewritten.** A format-3 store carries one
  `zarr.json` per node and slash-separated chunk keys under a `c/` prefix
  (`positions/c/0/0`); a format-2 store carries `.zgroup` / `.zattrs` /
  `.zarray` documents, dot-separated chunk keys (`positions/0.0`) and a
  consolidated `.zmetadata`. Everything described above applies to both: the
  attributes, the encodings and the node structure are identical, and only the
  container differs. A tree holding both is the expected steady state, not a
  migration window.
- **Choosing the format you write:** the two axes are pinned in exactly one
  place, `luxar._zarr_compat.ZARR_FORMAT`. Set `LUXAR_ZARR_FORMAT=2` to produce
  format 2 for a tool that cannot read 3. It is an environment variable rather
  than a flag because the process that writes is often not the one you invoked
  (batch-fit workers, Slurm array tasks, parallel tile fits).
- **Zarr library:** **zarr-python 3** (`zarr>=3.2,<4`). Being on 3.x is what lets
  Luxar read zarr v3 stores produced by other tools — 2.18 could not open one at
  all — and reading is version-agnostic regardless of what is being written.
- **NumCodecs:** Required for compression support. The Luxar-owned
  `luxar_delta_v1` filter exists once per format and produces **byte-identical**
  chunks either way: at format 2 it is a numcodecs filter in the `.zarray`
  `filters` list, resolved by the viewer through zarrita's `numcodecs.<id>`
  naming; at format 3 it is an array-to-array codec in the `zarr.json` `codecs`
  chain, resolved under the bare `luxar_delta_v1`. The viewer registers both
  names because both formats are live.
- **Consolidated Metadata:** **Required in practice**, not merely recommended.
  The viewer's scene loader builds its entire node graph from the store's
  consolidated listing and has no directory-walking fallback. Format 2 writes a
  separate `.zmetadata`; format 3 embeds `consolidated_metadata` in the root
  `zarr.json`. For v3 this is a zarr-python extension rather than part of the
  spec — zarr warns about exactly that, and Luxar suppresses the warning at its
  one consolidation call site rather than pass it on, since the advice concerns
  portability and not the store's validity — but zarrita implements it, so it
  remains load-bearing for the viewer in both formats. A third-party v3 reader
  that does NOT implement it will see the arrays but not the fast listing; the
  per-node `zarr.json` documents are complete and standard either way.
- **Editing a store in place:** re-open it through `luxar._zarr_compat.open_group`,
  not `zarr.open_group`. Format 3 permits a consolidated index on *any* group, and
  re-consolidating a store that was re-opened with plain zarr writes a second,
  nested index holding the pre-edit attributes; later reads then return stale
  values although every document on disk is correct. Going through the facade
  leaves exactly one index, at the root, in both formats.
- **Browser Support:** Via zarrita.js library

## Future Extensions (Planned)

- Support for volumes
- Material system with shading models (mesh ships one deliberately minimal,
  light-free view-anchored offset key — scene lights and richer shading models are still ahead)
- Temporal interpolation for smooth animations
- Multi-resolution spatial indices for LOD
