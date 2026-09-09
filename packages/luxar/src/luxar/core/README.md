# luxar.core

The `core` module contains the fundamental data structures and classes that form the foundation of Luxar's scene graph system. This includes scene nodes, data containers (Points, Lines, GSplats, Mesh), dimensional specifications, and transformation utilities.

## Overview

The core module implements Luxar's hierarchical scene graph architecture, enabling organization of large-scale visualization data with transforms, metadata, and nD dimensional support.

## Getting Started

### Quick Example - Create Your First Scene

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Create sample data
positions = np.random.randn(1000, 3).astype(np.float32)
colors = np.random.rand(1000, 3).astype(np.float32)

# Create scene
with LuxarZarrCompiler('my_scene.luxar.zarr') as compiler:
    # Define 3D dimensions
    dims = Dimensions([
        Dimension("x", unit="um", display=True),
        Dimension("y", unit="um", display=True),
        Dimension("z", unit="um", display=True),
    ])

    compiler.create_scene(dimensions=dims)

    # Add points
    compiler.write_points(
        "MyPoints",
        positions=positions,
        colors=colors,
        radii=1.0,  # Scalar broadcasts to all points
        opacity=0.8,
        blending_mode="additive"
    )

# View with: luxar serve my_scene.luxar.zarr --viewer
```

**Key Concepts:**
- **Scene**: Root container defining dimensions
- **Nodes**: Hierarchical organization (groups can contain groups/data)
- **DataNodes**: Points, Lines, GSplats, Mesh - the actual renderable data
- **Transforms**: 4x4 matrices for positioning/rotation/scaling
- **Dimensions**: Support nD data with keyboard navigation

## Key Components

### 1. Scene (`scene/`)

The root node of the scene hierarchy. Provides builder methods for constructing complex scenes with multiple data types. `Scene` now lives in its own subpackage (`scene/scene.py` plus `validation.py`, `dim_order.py`, and an `overlays/` subpackage) — see [scene/README.md](scene/README.md) for the full breakdown. `from luxar import Scene` and `from luxar.core import Scene` resolve unchanged.

**Key Features:**
- Progressive writing through `LuxarZarrCompiler`
- Scene-level dimension definitions
- Broadcasting support for nD data
- Hierarchical organization with groups
- Support for Points, Lines, GSplats, and Mesh data

**Usage Example:**
```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension
import numpy as np

# Define 4D scene dimensions
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True, range=(0, 99))
])

# Create scene with progressive writer
with LuxarZarrCompiler('output.luxar.zarr') as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add different data types
    positions = np.random.randn(10000, 4).astype(np.float32)
    scene.add_points('my_points', positions)

    # Add lines
    vertices = np.random.randn(100, 4).astype(np.float32)
    scene.add_lines('my_lines', vertices, widths=0.1)

    # Add Gaussian splats
    centers = np.random.randn(500, 4).astype(np.float32)
    amplitudes = np.ones(500)
    cholesky = np.random.randn(500, 10).astype(np.float32)
    scene.add_gsplats('my_splats', centers, amplitudes, cholesky)
```

**Key Methods** (inherited from Group):
- `add_group(name, **attrs)` - Create child group node
- `add_points(name, positions, ..., dim_order=..., fill=...)` - Add points
- `add_lines(name, vertices, widths, ..., dim_order=..., fill=...)` - Add lines
- `add_gsplats(name, centers, amplitudes, ..., dim_order=..., fill=..., fill_sigma=...)` - Add Gaussian splats
- `add_gsplats_from_data(name, result, ..., dim_order=..., fill=..., fill_sigma=...)` - Add from GSplatData
- `add_gsplats_from_file(name, path, ...)` - Add from .gsplats.zarr file
- `add_gsplats_from_volume(name, volume, ...)` - Fit and add Gaussian splats in one step

**Scene-only methods** (not on Group):
- `add_text(text, position, ...)` - Screen-space text overlay (returns `Overlay`)
- `add_image(image, position, ...)` - Screen-space image overlay (returns `Overlay`)
- `add_html(html, position, ...)` - Screen-space sanitized-HTML overlay (returns `Overlay`)
- `to_zarr(path)` - Finalize the writer and export to its supported destination
- `dimensions` (property) - Get/set scene-level dimensions (required at construction)
- `viewer_config` (property) - Get/set ViewerConfig hints
- `overlays` (property) - List of `Overlay` descriptors added to the scene

### 2. Group (`group/`)

Container node with data-adding methods. Groups walk up the parent chain to
find the root Scene for dimension validation and writer access. Scene inherits
from Group, so all these methods work on both. `Group` lives in its own
subpackage (`group/group.py` plus `partition.py`, `auto_partition.py`,
`compositing.py`, `dim_order.py`, and the `adders/`, `gsplats_pipeline/`, and
`lod/` subpackages) — see [group/README.md](group/README.md). The leaf adders
can return a `kind=partition` or `kind=lod` wrapper `Group` for very large or
multi-resolution geometry.

**Key Features:**
- `add_points()`, `add_lines()`, `add_gsplats()` add data children directly
- `dim_order` parameter maps lower-dimensional data columns to scene dimensions by name
- `fill` parameter provides fixed values for unmapped dimensions
- `fill_sigma` parameter (gsplats only) controls Cholesky embedding for unmapped dims
- Automatically infers `extend_to_all` for unmapped dimensions

**Usage Example:**
```python
# Groups can add data directly (preferred over parent= pattern)
group = scene.add_group('my_group',
                        opacity=0.8,
                        blending_mode='additive')
group.add_points('pts', positions)  # written under my_group/pts

# dim_order: map 3D data into a 4D scene
scene.add_gsplats_from_data('splats', result_3d,
    dim_order=['Z', 'Y', 'X'],       # data cols → scene dims
    fill={'Time': 0.0},               # fixed value for unmapped dim
    fill_sigma={'Time': 0.5})          # Cholesky sigma for unmapped dim
```

### 3. Node (`node/`)

Base class for all scene graph nodes. `Node` lives in its own subpackage
(`node/node.py` plus `specialized_groups.py`) — see [node/README.md](node/README.md).

**Key Features:**
- Hierarchical parent-child relationships
- Transform support (4x4 matrices)
- Rendering properties (opacity, absorption, gamma, intensity, offset, blending mode)
- Progressive writing without keeping Zarr groups in memory

**Usage Example:**
```python
# Groups created via add_group() have full data-adding methods
group = scene.add_group('my_group',
                        opacity=0.8,
                        gamma=1.2,
                        intensity=2.0,
                        offset=0.0,
                        blending_mode='additive')

# Transforms can be set directly
group.transform = luxar.translate(5, 0, 0)

# Or chained
group.set_opacity(0.5).set_gamma(1.0).set_intensity(2.0).set_blending_mode('additive')
```

**Key Properties:**
- `transform` - 4x4 transformation matrix
- `nd_transform` - Per-dimension transforms on non-displayed dimensions (see below)
- `opacity` - Rendering opacity (0.0-1.0)
- `absorption` - Volumetric absorption κ (>= 0.0, default 1.0; read only by the 'volumetric' mode)
- `gamma` - Gamma correction (0.1-10.0)
- `intensity` - Per-node color multiplier (0.0-100.0, default 1.0)
- `offset` - Per-node color offset (-10.0 to 10.0, default 0.0)
- `blending_mode` - Blending mode ('normal', 'additive', 'max', 'opaque', 'luminous', 'volumetric')
- `colormap` - Colormap name or LUT array (string names only via property setter)
- `layer` - Whether this node appears in the viewer's Layers panel
- `children` - List of child nodes
- `parent` - Parent node reference

**Important Notes:**
- Transforms are automatically transposed for THREE.js compatibility when stored
- Nodes use writer interface for progressive writing without keeping data in memory
- All rendering attributes are validated on assignment

**nD Transforms (`nd_transform`):**

Separate from the 4x4 spatial `transform`, nodes can carry an `nd_transform` dict
that applies per-dimension affine (scale/offset) or permutation transforms on
**non-displayed dimensions** (e.g., time, channel). This enables time alignment,
unit conversion, and channel remapping between datasets in the same scene.

```python
# Affine: shift time by 5 units
group.nd_transform = {"Time": {"scale": 1.0, "offset": 5.0}}

# Permutation: remap channels
group.nd_transform = {"Channel": {"permutation": [2, 0, 1]}}
```

- Composes hierarchically: `node.world_nd_transform` collects transforms from root to leaf
- Validated by `luxar.validation.nd_transforms`
- The compiler applies world nd_transforms to position bounds during finalization,
  so scene-level bounds reflect world-space ranges for non-displayed dimensions
- See `docs/guides/specs/ND_TRANSFORMS_SPEC.md` for full specification

### 4. DataNode (`datanode.py`)

Abstract base class for all data-bearing nodes (Points, Lines, GSplats, Mesh).

**Purpose:**
Provides common interface and behavior for all nodes that contain visualization data.

**Key Features:**
- Immediate writing to Zarr (no data kept in memory)
- Type-specific metadata storage
- Unified `n_elements` property
- Support for semantic type mapping for encoding

**Subclasses Must Implement:**
- `n_elements` property - Returns count of primary elements

**Usage Example:**
```python
# DataNode is not instantiated directly, but used through subclasses
# All data nodes share common interface:
print(f"Elements: {data_node.n_elements}")
print(f"Dimensions: {data_node.ndim}")
print(f"Metadata: {data_node.metadata}")
```

**Key Properties:**
- `n_elements` - Number of primary elements (abstract, implemented by subclasses)
- `ndim` - Dimensionality of data
- `metadata` - Type-specific metadata dictionary

**Inheritance Hierarchy:**
```
Node
 └── DataNode (abstract)
      ├── Points
      ├── Lines
      ├── GSplats
      └── Mesh
```

### 5. Points (`points.py`)

Specialized node for point cloud data. Lightweight metadata container in progressive mode.

**Key Features:**
- Metadata-only in progressive writing (data written immediately to Zarr)
- Tracks data characteristics (n_elements, has_colors, has_radii, etc.)
- Inherits all Node and DataNode capabilities

**Usage Example:**
```python
# Points created via Scene.add_points()
points = scene.add_points('cloud',
                         positions=positions,
                         colors=colors,
                         radii=radii,
                         sharpness=sharpness)

# Query metadata
print(f"Points: {points.n_elements:,}")
print(f"Has colors: {points.has_colors}")
```

**Key Properties:**
- `n_elements` - Number of points (the geometry-neutral DataNode count; the
  on-disk metadata key stays `n_points`)
- `has_colors` - Whether colors are present
- `has_radii` - Whether radii are present
- `has_sharpness` - Whether sharpness is present
- `has_scalars` - Whether scalar values for colormap lookup are present
- `has_labels` - Whether per-element string labels (hover tooltips) are present
- `has_image_labels` - Whether per-element image labels (hover thumbnails) are present
- `has_keys` - Whether per-element machine-readable keys (`{hover_key}` for `link` / `copy`) are present
- `metadata` - Full metadata dictionary

### 6. Lines (`lines.py`)

Node for line and curve data (polylines, segments, loops).

**Purpose:**
Represents 1D structures like trajectories, fiber tracts, network edges, or arbitrary curves.

**Key Features:**
- Supports multiple line types (segments, polyline, loop, indexed)
- Per-vertex colors and sharpness
- Variable line widths
- Progressive writing (data written immediately to Zarr)

**Line Types:**
- `"segments"` - Independent line segments (pairs of vertices)
- `"polyline"` - Connected line strip
- `"loop"` - Closed loop (last connects to first)
- `"indexed"` - Custom connectivity via indices array

**Usage Example:**
```python
# Create polyline from trajectory
trajectory = np.random.randn(1000, 3).astype(np.float32)
widths = np.linspace(0.1, 0.5, 1000)
colors = np.random.rand(1000, 3).astype(np.float32)

lines = scene.add_lines('trajectory',
                       vertices=trajectory,
                       widths=widths,
                       colors=colors,
                       line_type='polyline')

# Query metadata
print(f"Vertices: {lines.n_elements:,}")
print(f"Segments: {lines.n_segments:,}")
print(f"Line type: {lines.line_type}")
print(f"Max width: {lines.max_width}")
```

**Key Properties:**
- `n_elements` - Number of vertices (the geometry-neutral DataNode count; the
  on-disk metadata key stays `n_vertices`)
- `n_segments` - Number of line segments
- `line_type` - Type of line connectivity
- `has_colors` - Whether per-vertex colors are present
- `has_sharpness` - Whether per-vertex sharpness is present
- `has_scalars` - Whether scalar values for colormap lookup are present
- `has_labels` - Whether per-element string labels (hover tooltips) are present
- `has_image_labels` - Whether per-element image labels (hover thumbnails) are present
- `has_keys` - Whether per-element machine-readable keys (`{hover_key}` for `link` / `copy`) are present
- `max_width` - Maximum line width
- `has_spatial_index` - Whether spatial indexing is enabled
- `ordering` - Spatial ordering method (e.g., 'morton', 'hilbert', 'none')

**Arrays:**
- `vertices` - Shape (N, D) vertex positions
- `widths` - Shape (N,) line widths (or scalar broadcast)
- `colors` - Shape (N, 3) per-vertex colors (optional)
- `sharpness` - Shape (N,) edge sharpness (optional)
- `segments` - Shape (S, 2) connectivity pairs (stored form)
- `indices` - Vertex indices for indexed line type input (optional)

### 7. GSplats (`gsplats.py`)

Node for Gaussian splat data (oriented anisotropic Gaussians).

**Purpose:**
Represents data as generalized Gaussian distributions, useful for 3D Gaussian Splatting, uncertainty visualization, or smooth field representations.

**Key Features:**
- Generalized Gaussian kernels (not limited to standard Gaussians)
- Anisotropic covariances via Cholesky factorization
- Variable amplitudes (intensities)
- Sharpness parameter for generalized Gaussian exponent
- Progressive writing (data written immediately to Zarr)

**Mathematical Representation:**
Each splat is defined by:
- Center position: μ ∈ ℝᴰ
- Cholesky factor: L (lower triangular)
- Amplitude: α (intensity/weight)

The splat function: `f(x) = α * exp(-0.5 · ||L(x − μ)||²)` — a true Gaussian
(the falloff exponent is fixed at 2; the normalized `[0, 1]` *sharpness*
attribute is a Points/Lines rendering knob and is not part of GSplats)

**Usage Example:**
```python
# Create Gaussian splats
n_splats = 1000
centers = np.random.randn(n_splats, 3).astype(np.float32)
amplitudes = np.abs(np.random.randn(n_splats))

# Cholesky factors for 3D: 6 values per splat (packed lower triangle)
# For D dimensions: D*(D+1)/2 values per splat
cholesky = np.random.randn(n_splats, 6).astype(np.float32)

# Optional: colors
colors = np.random.rand(n_splats, 3).astype(np.float32)

splats = scene.add_gsplats('gaussians',
                          centers=centers,
                          amplitudes=amplitudes,
                          cholesky_factors=cholesky,
                          colors=colors)

# Query metadata
print(f"Splats: {splats.n_elements:,}")
print(f"Amplitude range: {splats.amplitude_range}")
print(f"Center bounds: {splats.center_bounds}")
```

**Key Properties:**
- `n_elements` - Number of splats (the geometry-neutral DataNode count; the
  on-disk metadata key stays `n_splats`)
- `has_colors` - Whether splat colors are present
- `has_labels` - Whether per-element string labels (hover tooltips) are present
- `has_image_labels` - Whether per-element image labels (hover thumbnails) are present
- `has_keys` - Whether per-element machine-readable keys (`{hover_key}` for `link` / `copy`) are present
- `ordering` - Spatial ordering type (e.g., 'morton', 'none')
- `amplitude_range` - Min/max amplitude values
- `center_bounds` - Bounding box of centers

**Arrays:**
- `centers` - Shape (N, D) splat centers
- `amplitudes` - Shape (N,) intensities (or scalar broadcast)
- `cholesky_factors` - Shape (N, k) where k=D*(D+1)/2 (packed lower triangle)
- `colors` - Shape (N, 3) splat colors (optional)

**Convenience Methods for GSplats:**

In addition to `add_gsplats()` which requires explicit arrays, Scene provides convenience methods for common workflows:

**From GSplatData:**
```python
from luxar.gsplats import fit_gaussian_splats

# Fit Gaussian splats to image
image = np.random.rand(100, 100).astype(np.float32)
result = fit_gaussian_splats(image, n_iters=1000)

# Add directly to scene (no intermediate save)
with LuxarZarrCompiler('scene.luxar.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_data('fitted', result)
    print(f"Added {gsplats.n_elements} splats with colors={gsplats.has_colors}")
```

**From .gsplats.zarr File:**
```python
# Load previously saved Gaussian splats
with LuxarZarrCompiler('scene.luxar.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_file('loaded', 'path/to/fitted.gsplats.zarr')
    print(f"Loaded {gsplats.n_elements} splats")
```

These methods automatically handle:
- Extracting arrays from GSplatData objects
- Loading data from .gsplats.zarr archives
- Passing all data (centers, amplitudes, cholesky_factors, colors) to add_gsplats()
- Preserving optional attributes (colors, labels) when present — `labels` /
  `image_labels` only on a single leaf with NO additive ladder, though: a
  multi-LEAF result (a multi-level substitutive pyramid, or a grafted multi-part
  `kind=lod` / `kind=partition` subtree) refuses them, because each leaf holds
  its own set of splats and no one list has a per-element correspondence to
  slice (#1471); a single laddered leaf is refused too, because the additive
  writer has no labels channel at all (`gsplat flatten` collapses the ladder)

### 7b. Mesh (`mesh.py`)

Node for triangle-surface data (isosurfaces, segmentation boundaries, organ and
cortical meshes).

**Purpose:**
Represents 2D *surfaces* embedded in nD — the one thing the other three types
cannot express. Points, Lines and GSplats are all soft, emissive, per-element
primitives; a surface is connected, opaque and shaded.

✅ **Writable and renderable.** The Python writer, reader and `luxar info` handle
mesh, and so do the viewer's loader, shaded material and picking — the whole
vertical ships today (`docs/specs/MESH_NODE_SPEC.md` §11). The format contract
still names the two sets separately — `geometry_types` (writable) vs
`loader_types` (viewer-drawable) — because a type becomes authorable before it
becomes drawable; today both sets include `mesh`.

**Key Features:**
- nD `vertices` plus a `faces` triangle-index array (`(F, 3)` or flat `(3F,)`)
- Optional per-vertex `normals`, with a **required** `normal_dims` companion
- Per-vertex colors (RGB or RGBA) and scalars for colormap lookup
- `shading` (`"smooth"` / `"flat"` / `"none"`) and `double_sided`
- `add_mesh(partition=True | {"max_elements": N, "rule": …})` → a `kind=partition`
  wrapper; the BSP cuts face centroids (`max_elements` counts faces, no triangle
  split), and each part re-indexes its own vertices — cut ones duplicated,
  per-vertex attributes gathered. See `docs/specs/MESH_NODE_SPEC.md` §9.2.
- `add_mesh(additive_lod=True | {"n_lods": N, …})` → a **reveal** ladder of
  `additive_<i>/` levels inside the leaf, each holding one concentric shell of
  faces, innermost first, so the surface grows outward from its centre as it
  streams. The viewer loads one through its own progressive mesh loader, which
  fetches the levels in order and commits each grown prefix into the same buffers.
- Progressive writing (data written immediately to Zarr)

**What a mesh does NOT have**, and why the absences are structural rather than
gaps:
- **No per-element size.** A triangle's extent comes from its own vertices, so
  there is no radius/width/covariance analogue — and a mesh contributes *zero*
  extent padding to scene bounds.
- **No additive ladder over an *arbitrary* order** — the restriction that survived,
  now that both LOD flavours have producers. A *substitutive* level is an
  independently-authored `(vertices, faces)` pair chosen by `coverage_fraction`, and
  `luxar.mesh.decimate` produces them, so `add_mesh(substitutive_lod=…)` writes a
  `kind=lod` group of progressively decimated surfaces. Its vocabulary is shorter
  than the sibling adders' (`core/group/lod/mesh.py`): no `truncation_radius` /
  `max_aspect` / `device` / `seed`, because those exist only for geometries that
  coarsen by lifting to gsplats. An *additive* level is a prefix, and a prefix of an
  arbitrarily ordered index buffer is a surface with **holes** rather than a coarser
  one — so `add_mesh(additive_lod=…)` accepts `method="radial"` and nothing else: a
  spatially coherent **reveal**, whose every prefix is a contiguous partial surface
  at full brightness. `"random"` / `"salience"` and the element samplers stay
  refused, as do `salience_kind` and `seed`, and the ladder carries no energy stamps
  by construction (see `core/group/lod/mesh.py::MESH_ADDITIVE_METHODS`). No two of
  `partition=` / `substitutive_lod=` / `additive_lod=` can be combined yet; each
  pairing is refused by name.
- **No labels on a reveal ladder.** The three sibling ladders write one union label
  CSR on the parent spanning the levels; a mesh level re-indexes its own vertices,
  so a boundary vertex occupies a slot in several levels and that union index space
  does not exist. `additive_lod=` with `labels` / `image_labels` degrades to a plain
  leaf with a `UserWarning` (the data is kept, the ladder is not). Labels survive
  intact on the `substitutive_lod=` and `partition=` paths.
- **No spatial index** (`ordering` is always `"none"`; the viewer loads a mesh
  whole, so a chunk index has nothing to skip).

Each remaining absence is refused with an explanation — including a non-reveal
`additive_lod=` method, `blending_mode='volumetric'`, and adding a mesh under a
`kind=partition` group that declares some other `display_type` — rather than
silently degrading. A `kind=lod` parent and a `display_type='mesh'`
`kind=partition` parent are both *accepted*: those are exactly the shapes
`substitutive_lod=` and `partition=` build.

**Usage Example:**
```python
# A welded tetrahedron
vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]],
                    dtype=np.float32)
faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
normals = compute_vertex_normals(vertices, faces)  # your own helper

mesh = scene.add_mesh('tetra',
                      vertices=vertices,
                      faces=faces,
                      normals=normals,
                      normal_dims=(0, 1, 2),   # required with normals
                      colors=(0.8, 0.8, 0.9))

print(f"Vertices: {mesh.n_elements:,}")
print(f"Faces: {mesh.n_faces:,}")
print(f"Shading: {mesh.shading}")
```

**Why `normal_dims` is required, not inferred:**
Normals are a *display-space* quantity — only meaningful for the three displayed
dimensions — so they are stored `(V, 3)` and the store must record *which* three.
Storing them against an implicit "first three dimensions" is wrong for any mesh
whose leading dimension is not spatial: for a `(t, x, y, z)` mesh the first three
are `(t, x, y)`, and a normal against them is meaningless. Making the triple
explicit turns an invisible wrong-orientation render into a checkable equality.

**Key Properties:**
- `n_elements` - Number of vertices (the geometry-neutral DataNode count, matching
  `Lines`, which also counts vertices rather than segments; the on-disk key stays
  `n_vertices`)
- `n_faces` - Number of triangles
- `has_normals` / `normal_dims` - Stored normals and the dimensions they describe
- `has_colors` / `has_scalars` - Optional per-vertex appearance channels
- `has_labels` / `has_image_labels` - Hover tooltips / thumbnails
- `has_keys` - Per-vertex machine-readable keys (`{hover_key}` for `link` / `copy`)
- `shading` - `"smooth"`, `"flat"`, or unlit `"none"`
- `double_sided` - Whether back faces render
- `ordering` - Always `"none"` in v1 (no spatial index)

### 7c. Sound (`sound.py`)

Node for an audio clip placed in the scene — the one node type that is *heard*
rather than drawn (`docs/guides/specs/SOUND_SPEC.md`).

**Purpose:**
An ambient bed, a narration that plays when a story step is reached, or a
spatial source that gets louder as the camera approaches. Sound is a layer on
top of rendering and independent of it, but a spatial source has a position in
the scene's frame — which makes it a node, not an overlay.

✅ **Writable and playable.** `scene.add_sound` writes it, `luxar info` and
`LuxarScene.list_sounds()` report it, and the viewer's audio engine plays it
through Web Audio. `sound` is in the format contract's `node_types` but NOT in
`geometry_types` / `loader_types`: it carries no elements, no blending mode, no
LOD and no picking, so none of the geometry tables have a row for it.

**Key Features:**
- **Three placements.** `positions=None, hidden=None` → audible everywhere;
  `hidden={"story": 3}` → sugar for one `(1, ndim)` row bound to a
  hidden-dimension value (extended over every other hidden dimension); explicit
  `positions=[[…]]` → a spatial source, one row per place it exists.
- **Slab audibility.** The same hidden-dimension slab rule that decides which
  points are visible decides when a sound is live; `extend_to_all` works as for
  points.
- **Attached** — `attach_to="Story 3: hsp70"`: the source follows the named
  node's bounding-box centre in the viewer (spatial by default; combine with
  `hidden=` to bind it to a value). Mutually exclusive with `positions`.
- **Ambisonic field** — `ambisonic="foa"`: a 4-channel AmbiX AAC clip the
  viewer decodes to stereo and rotates against the camera. Never spatial, never
  positioned; `hidden=` still decides when it is live.
- **Triggers** `"continuous"` (looped while audible, fades on the edge),
  `"once"` (each time the node becomes audible), `"on_depart"` / `"on_arrive"`
  (when a story flight leaves / lands on the `viewer_config.waypoints` entry
  whose `when` clause the node's row satisfies), plus `delay_ms`, `gain`,
  `fade_in_ms` / `fade_out_ms`.
- **Buses** `ambient` (default) / `voice` / `effects`; the voice bus ducks
  ambient while it plays (`ViewerConfig.audio.duck_db`).
- **Spatial knobs** map one to one onto `PannerNode`: `distance_model`,
  `ref_distance`, `max_distance`, `rolloff`, cone angles/gain, `orientation`.
  Distances left `None` stay absent so the viewer defaults them from the scene
  scale.
- **Provenance is required**: `license`, `attribution`, `source_url` on every clip.
- **Formats**: MP3 or AAC (`.m4a` / ADTS), sniffed from the bytes; Ogg/Opus is
  refused because Safari cannot decode it; WAV/FLAC are refused as the wrong size
  class for a hosted store.

**What a sound does NOT have:** appearance attrs (`opacity`, `colormap`, …) are
refused — only `layer` / `visible` / `transform` / `nd_transform` pass through;
no spatial index (`ordering` is always `"none"`); no contribution to the scene
bounds (a far-off source must not push the opening framing out).

**On disk** (`docs/guides/user/LUXAR_ZARR_FORMAT.md`, *Sound Nodes*): the node
group holds an optional `positions` array and the clip as a plain store key
(`audio.mp3` / `audio.m4a`, named by `attrs["audio_file"]`). The clip bytes are
folded into `content_hash` through `finalize/hashing.py::PAYLOAD_FILE_ATTRS`.

**Usage Example:**
```python
scene.add_sound(
    "bed_overview", "assets/overview_bed.mp3",
    trigger="continuous", gain=0.4, fade_in_ms=1500, fade_out_ms=1500,
    license="CC0", attribution="Freesound user X", source_url="https://…",
)
scene.add_sound(
    "narration_hsp70", narration_bytes, hidden={"story": 3},
    trigger="once", delay_ms=800, bus="voice",
    license="CC0", attribution="Synthesised (OpenAI TTS)", source_url="https://…",
)
scene.add_sound(
    "hum_hsp70", "assets/hum.mp3", positions=[[3, 7.28, -7.41, -0.27]],
    ref_distance=2.0, max_distance=30.0, license="CC0", attribution="…", source_url="…",
)
```

### 8. Dimensions (`dimensions.py`)

Scene-level coordinate system definitions with support for categorical dimensions.

**Key Classes:**
- `Dimension` - Single dimension specification
- `Dimensions` - Complete scene dimension system

**`Dimensions` Convenience Constructors:**
- `Dimensions.default_2d()` - 2D (x, y) in px
- `Dimensions.default_3d()` - 3D (x, y, z)
- `Dimensions.default_timeseries(n_timepoints=100, time_unit="s")` - (t, x, y, z)
- `Dimensions.default_multichannel(n_channels=3)` - (c, x, y, z)
- `Dimensions.from_positions(positions, names=None)` - Infer from an (N, D) array

**`Dimensions` Properties / Methods:**
- `ndim` / `len(dims)` - Number of dimensions
- `names` - Dimension names in order
- `displayed` / `non_displayed` - Index lists of displayed / non-displayed dims
- `spatial_extend_dims` - Per-dimension spatial-extension flags
- `get_dimension(name)` - Dimension by name (or `None`)
- `get_index(name)` - Index by name (raises if missing)
- `validate_positions(positions, name="positions")` - Shape + range check
- `to_dict()` / `from_dict()` - Serialize / deserialize

**Key Features:**
- Support for arbitrary dimensionality (not limited to 3D)
- Displayed vs non-displayed dimensions
- Discrete vs continuous dimensions
- **Categorical dimensions** with string labels
- Spatial extension flags for point coverage
- Navigation properties (step sizes, ranges)

**Usage Example:**
```python
from luxar.core.dimensions import Dimension, Dimensions

# Define 5D space with categorical channel dimension
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True,
              range=(0, 99), step=1.0),
    Dimension('channel', unit='ch', display=False, discrete=True,
              categories=['DAPI', 'GFP', 'mCherry'])  # Categorical!
])

# Query properties
print(f"Total dimensions: {dims.ndim}")
print(f"Displayed: {dims.displayed}")
print(f"Non-displayed: {dims.non_displayed}")

# Check categorical dimensions
channel_dim = dims.get_dimension('channel')
print(f"Is categorical: {channel_dim.is_categorical}")
print(f"Categories: {channel_dim.categories}")
```

**Dimension Properties:**
- `name` - Dimension identifier
- `unit` - Physical unit
- `display` - Whether dimension is displayed (max 3)
- `discrete` - Whether values are discrete
- `cyclic` - Whether dimension wraps around
- `scale` - Physical scale factor
- `spatial` - Whether points extend through this dimension
- `categories` - **NEW:** Optional list of category labels for categorical dimensions
- `description` - Human-readable description
- `range` - Optional (min, max) bounds
- `step` - Navigation step size

**Categorical Dimensions:**

Categorical dimensions allow string labels instead of numeric coordinates:

```python
# Define categorical dimension
channel_dim = Dimension(
    'channel',
    categories=['DAPI', 'GFP', 'mCherry', 'Cy5'],
    display=False
)

# Categorical dimensions are automatically:
# - discrete = True (enforced)
# - range = (0, len(categories)-1) (auto-set if not provided)
# - step = 1.0 (auto-set if not provided)

# In data, use integer indices (0-based):
# 0 = 'DAPI', 1 = 'GFP', 2 = 'mCherry', 3 = 'Cy5'
positions = np.array([
    [10.0, 20.0, 5.0, 0.0],  # Channel 0 (DAPI)
    [11.0, 21.0, 5.5, 1.0],  # Channel 1 (GFP)
    [12.0, 22.0, 6.0, 2.0],  # Channel 2 (mCherry)
])
```

**Categorical Dimension Features:**
- String labels for human-readable dimension values
- Automatic validation of category indices
- Preserved in zarr metadata for viewer display
- Useful for: channels, cell types, experimental conditions, time-lapse phases
- Categories must be unique, non-empty strings

**Automatic Behaviors:**
- `spatial` flag auto-determined from `display` if not specified
- Non-spatial, non-displayed dimensions automatically marked discrete
- Step sizes auto-calculated if not provided
- Categorical dimensions auto-set discrete=True, range, and step

### 9. Transforms (`transforms.py`)

Utilities for creating and manipulating 4x4 transformation matrices.

**Key Functions:**
- `identity()` - Create identity matrix
- `translate(x, y, z)` - Translation matrix
- `rotate_x/y/z(degrees)` - Axis-aligned rotations
- `rotate(degrees, axis)` - Arbitrary axis rotation
- `scale(x, y, z, uniform)` - Scaling matrix
- `compose(*transforms)` - Combine multiple transforms
- `inverse(transform)` - Compute inverse
- `look_at(eye, target, up)` - Camera-style transform
- `to_list(transform)` - Convert to storage format (column-major for THREE.js)
- `from_list(values)` - Convert from storage format (column-major from THREE.js)
- `prepare_transform_for_zarr(transform)` - Convert any format to zarr-compatible list
- `read_transform_from_zarr(transform_list)` - Read transform from zarr attributes
- `transform_bounding_box(matrix, lo, hi)` - Transform all 8 corners of an AABB and return the enclosing axis-aligned box, omitting corners with `|w| < 1e-12` and falling back to the input box if all are omitted (correct under rotation/shear; mirrors the viewer's `transformBoundingBox`)

(`identity`, `translate`, `scale`, `rotate*`, `compose`, `inverse`, `look_at`,
`to_list`, `from_list`, and the aliases are re-exported from `luxar.core`;
`prepare_transform_for_zarr`, `read_transform_from_zarr`, and
`transform_bounding_box` are module-level helpers used internally.)

**Convenience Aliases:**
- `translation()` - Alias for `translate()`
- `scaling()` - Alias for `scale()`
- `rotation()` - Alias for `rotate()`

**Usage Example:**
```python
import luxar

# Create transformation
t1 = luxar.translate(10, 0, 0)
t2 = luxar.rotate_z(45)
t3 = luxar.scale(2, 2, 2)

# Compose (applied in order: translate, then rotate, then scale)
combined = luxar.compose(t1, t2, t3)

# Apply to node
node.transform = combined
```

**Visual Composition Example:**
```
Point → [Translate] → [Rotate] → [Scale] → Final Position
        ↑             ↑           ↑
     compose(Translate, Rotate, Scale)
        ↑             ↑           ↑
      First        Second       Last
     Applied      Applied      Applied

Example: compose(translate(5,0,0), rotate_z(90°), scale(2,2,2))
Point (1,0,0) → Translate → (6,0,0) → Rotate → (0,6,0) → Scale → (0,12,0)
```

**Important Notes:**
- All matrices are 4x4 homogeneous transforms (float32)
- Matrices are automatically transposed for THREE.js when stored
- Use `to_list()` and `from_list()` for serialization (handles transpose)
- Composition order: `compose(A, B, C)` applies A first, then B, then C (LEFT-to-RIGHT)
- With column-vector math, the resulting matrix is `C @ B @ A`, so `C @ (B @ (A @ point))` applies A first

### 10. ViewerConfig (`viewer_config.py`)

Dataclasses for viewer configuration hints stored in the zarr file.

**Key Classes:**
- `ViewerConfig` - Top-level viewer configuration (camera, rendering, bloom, effects, UI, theme)
- `CameraConfig` - Camera position, target, FOV, clipping planes, target_node, ortho `zoom`
- `UIConfig` - Panel visibility (help, rendering controls, performance, dimensions, scale bar, layers, overlays)
- `DimensionsConfig` - nD navigation state (current step, selected dimension)
- `AnimationConfig` - Per-dimension animation (playing, target_fps, loop mode, direction, step size)
- `Waypoint` - A camera pose bound to a hidden-dimension position (`when={"story": 1, "time": (10, 20)}`, the overlay `visible_range` rule; first match wins), with optional `duration_ms`, `easing`, `reveal="on_arrival"` to hold dimension-bound overlays until the flight resolves, and a `rendering` override block — how a scene's story dimension drives the camera

**Usage Example:**
```python
from luxar import ViewerConfig, CameraConfig

vc = ViewerConfig(
    camera=CameraConfig(position=(0, 5, 20), target_node="embryo"),
    bloom_strength=0.5,
    theme="dark",
)

# From JSON exported by viewer (Ctrl+Shift+S)
vc = ViewerConfig.from_file("my_view.json")

# Apply to scene
with LuxarZarrCompiler('output.luxar.zarr') as compiler:
    scene = compiler.create_scene(dimensions=dims, viewer_config=vc)
```

**Key Methods:**
- `to_dict()` / `from_dict()` - Serialize/deserialize (sparse, omitting None fields)
- `from_file(path)` / `to_file(path)` - JSON file I/O
- `from_json(string)` / `to_json()` - JSON string I/O
- `validate()` - Validate all configuration values

### 11. Overlay (`overlay.py`)

Lightweight metadata descriptor for a screen-space annotation. Overlays are
**not** part of the 3D scene graph — they exist at the Scene level only and are
positioned in normalized screen coordinates over the viewer canvas.

`Overlay` is a frozen-style dataclass returned (for optional inspection) by
`Scene.add_text()`, `Scene.add_image()`, `Scene.add_html()`, and `Scene.add_video()`.
The overlay is written immediately to zarr; the returned object is just a descriptor.

**Attributes:**
- `name` - Unique overlay name (auto-generated or user-specified)
- `overlay_type` - One of `'overlay_text'`, `'overlay_image'`, `'overlay_html'`, `'overlay_video'`
- `position` - `(x, y)` in normalized screen coordinates `[0, 1]`, top-left origin
- `attrs` - All overlay attributes as written to the zarr `.zattrs`

```python
ov = scene.add_text("Hello", position=(0.05, 0.05))
print(ov)  # <Overlay 'text_0' type=overlay_text pos=(0.05, 0.05)>
```

See [scene/README.md](scene/README.md) and [scene/overlays/README.md](scene/overlays/README.md)
for the overlay adders, hover templating, and label-driven auto-injection.

## Architecture

### Progressive Writing Design

The core module is designed to work with Luxar's progressive writing system:

1. **Scene Creation**: Scene created with a writer (LuxarZarrCompiler)
2. **Node Creation**: Nodes are lightweight metadata containers
3. **Data Writing**: Data written immediately to Zarr via writer
4. **Memory Efficiency**: Data never kept in memory after writing

`Scene.to_zarr(path)` is an export helper for this progressive model. A
directory-backed scene is copied to a new directory destination, unless `path`
is its current backing store for an explicit finalize-in-place. An
archive-backed scene is finalized at its selected archive path, replacing an
existing archive there. A directory-backed scene cannot be copied directly to
a `.zip` destination; create the archive with `LuxarZarrCompiler` or use
`luxar optimise`. Because finalization closes the writer, do not add more nodes
to a scene after calling `to_zarr()`; use a new compiler for additional writes.

### Scene Graph Structure

```
Scene (root)
├── Group "cells"
│   ├── Points "cell_1"
│   ├── Lines "cell_edges"
│   └── GSplats "cell_uncertainty"
├── Group "markers"
│   ├── Points "marker_points"
│   └── Lines "marker_connections"
└── (overlays)            # screen-space text/image/video/HTML, not 3D nodes
```

Overlays (`add_text` / `add_image` / `add_video` / `add_html`) are tracked
separately on the `Scene` and live in normalized screen space, not in the 3D
transform hierarchy.

### Transform Hierarchy

Transforms compose hierarchically:
- Each node can have a local transform
- Final transform = parent_transform @ local_transform
- Transforms are automatically applied by the viewer

### Data Node Hierarchy

All data-bearing nodes inherit from DataNode:

```
Node (base class)
 └── DataNode (abstract base for data nodes)
      ├── Points (point data)
      ├── Lines (curve/line data)
      ├── GSplats (Gaussian splat data)
      └── Mesh (triangle surface data)
```

## Dependencies

**Internal:**
- `luxar.typing_utils` - Type definitions and validation
- `luxar.io.writer` - Writer protocol for progressive writing
- `luxar.utils.array` - Array broadcasting helpers
- `luxar.validation` - Data validation utilities
- `luxar.encoding` - Data type encoding/decoding

**External:**
- `numpy` - Array operations
- `zarr` - Data storage (indirect, through writer)
- `arbol` - Structured logging

## Testing

Tests are located in `core/tests/`:
- `test_datanode_types.py` - DataNode types: cross-type parity matrix, Lines and GSplats
- `test_dimension_metadata.py` - Dimension functionality (current Dimension class)
- `test_dimensions.py` - Scene-level dimensions
- `test_extend_to_all.py` - extend_to_all functionality in Scene.add_points()
- `test_graft_compositing_attrs.py` - attr routing when a nested `.gsplats.zarr` is grafted into a scene
- `test_gsplats_extend_to_all.py` - extend_to_all functionality in Scene.add_gsplats()
- `test_hdr_colors.py` - Edge case tests for HDR color support
- `test_lines_extend_to_all.py` - extend_to_all functionality in Scene.add_lines()
- `test_mesh.py` - Mesh DataNode: add_mesh round-trips (topology, normals/normal_dims, colors, labels), shading/double_sided resolution, the `partition=` split (face conservation, vertex duplication, per-vertex attribute gathering), and the volumetric-blending / LOD refusals
- `test_node_properties.py` - Node properties and method chaining
- `test_node_rendering.py` - Rendering attributes for Node class
- `test_overlays.py` - Screen-space overlays (add_text / add_image / add_video / add_html)
- `test_physical_units.py` - Physical units support through Dimensions system
- `test_api_regressions.py` - Regression tests pinning core API invariants (ndim metadata key, property-setter persistence, cross-scene node inequality, Scene.dimensions, Scene.to_zarr export, top-level GSplatData export)
- `test_scene_advanced.py` - Advanced Scene class tests (initialization, error handling)
- `test_scene_methods.py` - Scene class methods not covered elsewhere
- `test_scene_structure.py` - Scene graph structure
- `test_spatial_dimensions.py` - Spatial dimension functionality (spatial flag)
- `test_transforms.py` - Transform utilities and functionality
- `test_transforms_properties.py` - Property-based tests for the transform composition algebra
- `test_viewer_config.py` - ViewerConfig, CameraConfig, and related dataclasses

The `Group` side lives one level down, in `core/tests/group/`:
- `test_group.py` - Group class with add_* methods
- `test_compositing.py` - Partition/LOD wrapper compositing primitives
- `test_dim_order.py` - dim_order dimension mapping on add_points / add_lines / add_gsplats
- `test_gsplats_volume_budget.py` - splat-budget resolution in add_gsplats_from_volume (int count vs float ratio)
- `lod/`, `partition/` - the two specialized group kinds

Run tests:
```bash
hatch run pytest packages/luxar/src/luxar/core/tests/
```

## Implementation Notes

### Transform Storage

Transforms are stored in THREE.js-compatible format (column-major):
```python
# NumPy (row-major) → Storage (column-major)
numpy_matrix = np.array([[...], [...], [...], [...]])  # 4x4
storage_list = numpy_matrix.T.ravel().tolist()  # Transpose for THREE.js

# Storage → NumPy
storage_list = [...]  # 16 elements
numpy_matrix = np.array(storage_list).reshape(4, 4).T  # Transpose back
```

This ensures correct interpretation by the THREE.js viewer.

### Dimension Validation

Scene dimensions are validated to ensure:
- Maximum 3 displayed dimensions
- At least 1 displayed dimension if any exist
- Unique dimension names
- Valid ranges (min < max)
- Consistent spatial extension flags
- **Valid categories** (unique, non-empty strings)
- **Valid category indices** in data

### Node Attribute Storage

Nodes cache attributes and write them immediately:
- Attributes cached in `_attrs_cache` for fast access
- Written immediately to Zarr via writer interface
- Rendering attributes validated on assignment
- No Zarr groups kept in memory (memory-efficient)

### Categorical Dimension Validation

Categorical dimensions undergo additional validation:
- Categories must be list of strings
- Each category must be unique
- Empty strings not allowed
- Category labels limited to 1024 characters each
- Position values must be integer indices (0-based)
- Indices must be in range [0, len(categories)-1]

See `luxar.validation.validate_categories()` and `luxar.validation.validate_category_indices()`.

## Best Practices

### 1. Always Use Context Manager
```python
with LuxarZarrCompiler('output.luxar.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    # ... build scene
# Automatically finalized
```

### 2. Define Dimensions Early
```python
# Define dimensions before creating scene
dims = Dimensions([...])
scene = compiler.create_scene(dimensions=dims)
```

### 3. Use Explicit Dimension Extension
```python
# Explicit is better than implicit - works for both points and lines
scene.add_points('pts', positions,
                extend_to_all=['time', 'channel'])

# Lines can also extend across dimensions (e.g., static detector geometry)
scene.add_lines('detector', vertices, widths=0.1,
               extend_to_all=['time'])  # Visible at all time values
```

### 4. Validate Data Before Writing
```python
# Positions must match scene dimensions
if scene.dimensions:
    scene.dimensions.validate_positions(positions)
```

### 5. Use Transform Utilities
```python
# Use provided functions instead of manual matrix creation
transform = luxar.compose(
    luxar.translate(5, 0, 0),
    luxar.rotate_z(45)
)
```

### 6. Choose Appropriate Data Types
```python
# Points for discrete particles
scene.add_points('particles', positions)

# Lines for trajectories/networks
scene.add_lines('trajectories', vertices, widths=0.1, line_type='polyline')

# GSplats for smooth fields/uncertainty
scene.add_gsplats('uncertainty', centers, amplitudes, cholesky)
```

### 7. Use Categorical Dimensions for Discrete Labels
```python
# Better than numeric channel indices
Dimension('channel', categories=['DAPI', 'GFP', 'mCherry'])

# Instead of
Dimension('channel', range=(0, 2), discrete=True)  # What does 0 mean?
```

## See Also

- [node/README.md](node/README.md) - `Node` base class and specialized groups
- [group/README.md](group/README.md) - `Group` and the `add_*` machinery (partition, LOD, adders)
- [scene/README.md](scene/README.md) - `Scene` root, validation, dim_order, overlays
- [io/README.md](../io/README.md) - I/O operations and writers
- [typing_utils/README.md](../typing_utils/README.md) - Type system
- [validation/README.md](../validation/README.md) - Validation utilities
- [encoding/README.md](../encoding/README.md) - Data encoding and semantic types
- [Main README](../../../../../README.md) - Project overview
