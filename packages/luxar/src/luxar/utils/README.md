# Utils Package

The `utils` package provides utility functions for common operations in Luxar, including array manipulation, atomic directory copies, robust downloads, spatial hashing, vector-field helpers, and demo data generation.

## Quick Start

Common utility operations in 3 steps:

```python
from luxar.utils import ensure_float32, validate_array_shape
from luxar.utils import create_lorenz_attractor  # Demo data generator
import numpy as np

# 1. Type-safe array conversion
data = np.array([1, 2, 3], dtype=np.float64)
data_f32 = ensure_float32(data)  # Now guaranteed float32
print(f"Converted: {data.dtype} -> {data_f32.dtype}")

# 2. Shape validation with clear errors
positions = np.random.rand(100, 3).astype(np.float32)
validate_array_shape(positions, (100, 3), name="positions")  # Passes

# 3. Generate demo data for testing
create_lorenz_attractor(
    'lorenz_demo.luxar.zarr',
    n_points=10000,
    seed=42  # Reproducible
)
print("Demo scene created at lorenz_demo.luxar.zarr")
```

**Key Use Cases**:
- Array conversion - ensure float32 before encoding
- Shape validation - catch dimension mismatches early
- Demo generation - create test data for examples and tutorials

## Overview

This package contains helper functions that simplify common tasks and provide convenient demo data generators for testing and examples.

**Note**: Scalar broadcasting (e.g., `colors=(1,0,0)`, `radii=0.5`) is handled by `ArrayEncoder` in `luxar.encoding`. The previous `broadcast_*_to_points()` functions have been removed.

## Modules

### `array.py`
Array manipulation utilities.

**Key Functions:**
- `ensure_float32()`: Convert arrays to float32
- `validate_array_shape()`: Check array dimensions

**Features:**
- Automatic type conversion with validation
- Shape validation with helpful error messages

### `atomic_copy.py`
Atomic directory-tree copy. Writes to a sibling temp dir and `os.replace`s on
success, so the destination either exists in full or not at all. Used by
`Scene.to_zarr` and the CLI `luxar export` to keep `.zarr` exports atomic.

**Key Functions:**
- `atomic_copytree(src, dst)`: Copy `src` to `dst` atomically. `dst` must not already exist (the caller clears it for overwrite, matching `shutil.copytree`). Cleans up the temp dir and re-raises on failure.

### `download.py`
Robust download utilities with retry logic, resume capability, and progress tracking.

**Key Functions:**
- `robust_download()`: Download a file from a URL with automatic retry (exponential backoff), partial download resume via HTTP Range requests, progress tracking with ETA, and file-size verification
- `verify_file_checksum()`: Verify a file's integrity against an expected MD5 and/or SHA256 hash
- `download_with_checksum()`: Combine `robust_download()` with checksum verification, deleting the file if the checksum fails
- `find_quarantined_files(target)`: Return the `.corrupt` files associated with a cache *file* (both the `foo.npy.corrupt` and `foo.corrupt` quarantine conventions) or every `*.corrupt` inside a cache *directory*
- `format_quarantine_notice(paths, ...)`: Build an actionable multi-line notice naming each quarantined path, its size, and what to do about it (empty string when there is nothing to report)
- `warn_if_quarantined(target, ...)`: Print that notice and return the paths. Called from `robust_download()` so a user about to re-fetch a multi-gigabyte artifact is told that a rejected earlier copy is sitting next to it — instead of watching a huge download silently start over
- `QUARANTINE_SUFFIX`: The `.corrupt` suffix used when a cached artifact fails validation (see `demos.cache_computed`). A quarantined file is never reused

### `fields.py`
Shared 3D vector-field helpers for flow-field demos (the PPI flow-field demo
and the zebrahub RNA-velocity-streamlines demo). Deliberately demo-agnostic —
each demo keeps its own binning, smoothing, caching, and seeding policy.

**Key Class:**
- `FlowField`: Frozen dataclass — a cubic vector field on a regular grid (`vectors` `(n,n,n,3)`, `grid_min`/`grid_max`, `spacing`, `cache_key`)

**Key Functions:**
- `cubic_bounds()`: Symmetric cubic AABB around a point cloud, with fractional padding
- `trilinear_vector()`: Trilinearly sample a `FlowField` at world points (NaN rows for out-of-bounds)
- `unit_flow()`: Direction-only sample (NaN where the field is zero or out-of-bounds)
- `rk4_step()`: Vectorized 4-stage Runge-Kutta advection step (NaN-fills streamlines that leave the domain)
- `add_reference_cube_to_scene()`: Add the field's cubic domain as a 12-edge wire cube (Lines geometry) to a Luxar scene

### `paths.py`
Path utilities for Luxar dataset generation.

**Key Functions:**
- `get_project_root()`: Find the Luxar project root directory (cached)
- `get_datasets_dir()`: Get the `datasets/` directory at project root
- `get_examples_output_dir()`: Resolve the centralized `datasets/examples/` output directory
- `get_demos_output_dir()`: Resolve the centralized `datasets/demos/` output directory

### `spatial_hash.py`
Spatial hash grids for fast nD proximity queries. Two complementary classes for two access patterns:

**Key Classes:**
- `SpatialHashGrid`: **online** insert + `has_neighbor_within` (CPU, dict-backed). Used by Poisson-disk seeding where each accept depends on previous accepts.
- `BatchedSpatialHashGrid`: **batched** build + many radius / k-NN queries. Two backends share the same hash scheme: NumPy (`np.argsort` + `np.searchsorted`) and PyTorch (CUDA / MPS / CPU; pad-and-prune GPU k-NN). With `device='auto'` (default) the GPU backend is preferred and falls back to NumPy on **out-of-memory or device-unavailable conditions only** — generic exceptions propagate so real bugs aren't masked.

**Correctness guarantee** (both classes): radius queries find all hits within the requested radius provided `cell_size >= radius`. The 3^D neighbour-cell scan is then exhaustive. k-NN queries auto-expand the cell shell until the kth-nearest candidate's distance is within the guaranteed-coverage radius (`shell_radius * cell_size`); past 3 shells the entire stored set is brute-forced.

**Quick example:**
```python
from luxar.utils.spatial_hash import BatchedSpatialHashGrid
import numpy as np

points = np.random.randn(10_000, 3).astype(np.float32)
queries = np.random.randn(1_000, 3).astype(np.float32)

# Auto-select GPU when available, else CPU
grid = BatchedSpatialHashGrid.from_points(points, cell_size=0.5, device='auto')
print(grid.backend, grid.device)  # e.g. "torch", cuda:0

# k-NN: (Q, k) distances + indices
distances, indices = grid.query_knn(queries, k=8)

# Radius: jagged list of indices per query (require radius <= cell_size)
neighbours = grid.query_radius(queries, radius=0.4)
```

### `_umap_utils.py`
Shared utilities for UMAP demo scripts (internal module).

**Key Features:**
- Color palettes and colormap functions for UMAP visualizations
- Legend generation utilities
- Attribute-to-color mapping used by multiome UMAP demos (human, mouse, zebrahub)

### `data_fetch.py`
Manifest-driven demo-dataset resolution (R17: retiring in-repo Git LFS in favour
of fetch-on-demand from Zenodo). Reads `demos/data_manifest.json` — the single
source of truth for how each dataset is obtained, its license, and its per-file
sha256.

**Key Functions:**
- `ensure_dataset(name, ...)`: Resolve a dataset's files to local paths, cache -> in-repo Git LFS -> Zenodo. The manifest sha256 is authoritative at every step: a copy that fails it is quarantined (`.corrupt`) and never returned, so a stale download can never be resumed onto corrupt bytes
- `load_dataset_gsplats(name, ...)`: Mirror of `demos.load_precomputed_gsplats` (returns `GSplatData`, `None` on recompute) sourced through `ensure_dataset` — the one-line swap for migrating a demo. Only `zenodo`-bucket datasets are eligible
- `load_manifest()` / `dataset_spec(name)`: Read the packaged manifest
- Raises `DatasetNotFound` for an unknown key and `LocalComputeDataset` for data we cannot redistribute (the caller builds it locally)

### `demos.py`
Demo scene generators, precomputed data helpers, and viewer launch utilities.

**Key Functions:**
- `create_lorenz_attractor()`: Generate Lorenz attractor visualization
- `create_random_spheres()`: Create random spherical points
- `create_time_series_demo()`: Generate time-varying data
- `launch_viewer()`: Launch the Luxar viewer for a given dataset path
- `detect_device()`: Auto-detect the best available compute device (cuda > mps > cpu)
- `warn_if_no_cuda_gpu()`: Print a warning if no CUDA GPU is available
- `load_precomputed_gsplats()`: Load precomputed GSplat data from Git LFS or cache
- `load_precomputed_bundle()`: Load a precomputed bundle zip (timelapse demos)
- `parse_demo_flags()`: Parse common demo CLI flags (--recompute, --device, etc.)
- `is_lfs_pointer()`: Check if a file is a Git LFS pointer (not actual data)

**Features:**
- Ready-to-use demo scenes
- Configurable parameters
- Git LFS data loading with local cache fallback
- Educational examples of Luxar features

## Usage Examples

### Creating Demo Scenes

```python
from luxar.utils import create_lorenz_attractor

# Generate Lorenz attractor
create_lorenz_attractor(
    store_path='lorenz.luxar.zarr',
    n_points=50000,
    seed=42
)

# Creates a beautiful 3D visualization
# with time-based coloring
```

### Array Utilities

```python
from luxar.utils import ensure_float32, validate_array_shape
import numpy as np

# Ensure float32 dtype
arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr_f32 = ensure_float32(arr)  # Now float32

# Validate array shape
positions = np.random.rand(100, 3).astype(np.float32)
validate_array_shape(positions, (100, 3), name="positions")  # OK

# Multiple acceptable shapes
validate_array_shape(colors, [(100, 3), (100, 4)], name="colors")
```

**Note**: For scalar broadcasting (e.g., `radii=0.5` for all points), use the encoding system via `LuxarZarrCompiler.write_points()` which handles this automatically.

## Demo Generators

### Lorenz Attractor
```python
def create_lorenz_attractor(
    store_path: PathLike,  # str or Path
    n_points: int = 10_000,
    seed: Optional[int] = None
) -> None:
```
Generates the famous Lorenz attractor with:
- Smooth trajectory through phase space
- Time-based coloring
- Configurable density

### Random Spheres
```python
def create_random_spheres(
    store_path: PathLike,  # str or Path
    n_spheres: int = 100,
    points_per_sphere: int = 1000,
    seed: Optional[int] = None
) -> None:
```
Creates multiple spherical points:
- Random positions and sizes
- Different colors per sphere
- Hierarchical organization

### Time Series Demo
```python
def create_time_series_demo(
    store_path: PathLike,  # str or Path
    n_timepoints: int = 10,
    n_points_per_time: int = 1000,
    seed: Optional[int] = None
) -> None:
```
Generates time-varying data:
- 4D data (x, y, z, time)
- Animated trajectories
- Useful for testing nD features

## Best Practices

### Broadcasting
1. **Be explicit**: Specify expected shapes
2. **Validate early**: Check inputs before processing
3. **Preserve precision**: Maintain float32 for GPU compatibility
4. **Handle None**: Gracefully handle optional parameters

### Demo Data
1. **Use seeds**: Make demos reproducible
2. **Document parameters**: Explain what each demo shows
3. **Educational value**: Demos should teach Luxar features
4. **Performance**: Keep demos fast for testing

## Utility Patterns

### Safe Type Conversion
```python
def ensure_type(data, dtype=np.float32):
    """Safely convert to target dtype."""
    if data is None:
        return None
    return np.asarray(data, dtype=dtype)
```

### Shape Validation
```python
def check_compatible(a, b, axis=0):
    """Check if arrays are compatible along axis."""
    if a.shape[axis] != b.shape[axis]:
        raise ValueError(
            f"Incompatible shapes: {a.shape} vs {b.shape} "
            f"along axis {axis}"
        )
```

### Optional Processing
```python
def process_optional(data, processor, default=None):
    """Process data if provided, otherwise return default."""
    if data is None:
        return default
    return processor(data)
```

## Dependencies

Internal:
- `io.compiler`: For demo scene creation
- `typing_utils`: Type definitions, aliases, and dataset-size config
- `core.dimensions`: Scene dimension definitions (demos)
- `gsplats`: `GSplatData` loading and torch-device resolution (demos)

External:
- `numpy`: Array operations
- `arbol`: Progress display in demos and downloads
- `torch`: PyTorch backend for `BatchedSpatialHashGrid` (GPU k-NN / radius queries)
- `requests` / `urllib3`: HTTP downloads with retry (lazily imported in `download.py`)
- `Pillow (PIL)`: Legend image rendering in `_umap_utils.py`
