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

### `arbol_warnings.py`
Route Python warning *display* through arbol console output, so warnings land
as `⚠️ UserWarning: ...` tree lines instead of raw stderr
`path/to/file.py:299: UserWarning: ...` text that appears out of place
mid-tree.

**Key Functions:**
- `install_arbol_warnings()`: Process-wide install for application entry points (called by the `luxar` CLI callback)
- `arbol_warnings()`: Context manager / decorator scoping the override to a block (applied to the arbol-tree-producing public API entry points: `LuxarZarrCompiler`'s write methods, `fit_gaussian_splats`, `generate_seeds`, `save_gsplats`)

**Features:**
- Display-only: `warnings.warn` machinery, filters, `-W error`, `catch_warnings`, and `pytest.warns` behave exactly as before
- Steps aside automatically when a recorder or custom `showwarning` hook owns warning display (so test harnesses keep capturing)

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
- `robust_download()`: Download a file from a URL with automatic retry (exponential backoff), partial download resume via HTTP Range requests, progress tracking with ETA, and file-size verification. Bytes are staged in a sibling `<dest>.part` and atomically promoted onto the destination only once complete and size-verified, so a file at the destination is complete by construction; resumes are validated with `If-Range` against the ETag/Last-Modified recorded in a `<dest>.part.validator` sidecar, so a remote that changed is re-fetched clean instead of spliced
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

### `lod_breakpoints.py`
Streaming-ladder breakpoint math, shared by all three geometries (Points, Lines,
GSplats). An additive (streaming) ladder cuts an importance-ordered element
sequence into cumulative prefixes so a viewer can paint a coarse prefix
immediately and refine as later chunks arrive. The cut geometry is a property of
the network and payload, not the geometry type, so this is the one place both
`luxar.core` and `luxar.gsplats` derive identical cuts from an identical spec
(stdlib-only, to avoid a new import direction).

**Key Functions:**
- `streaming_chunk_splats(target_ms, bandwidth_mbps, bytes_per_splat)`: First-chunk element count whose download takes `target_ms` at the given bandwidth
- `parse_stream_chunk(spec)`: Extract `c` from a `"stream:<c>"` spec (validated `>= 1`)
- `validate_element_breakpoints(spec)`: Size-independent resolve-time validation of a Points/Lines `counts`/`breakpoints` value (string vocabulary, stream chunk, energy fractions, list non-emptiness) — fails before any group is written
- `stream_cuts(n, chunk, max_levels=...)`: Cumulative geometric cuts `[c, 2c, 4c, …, n]` over `n` elements
- `sibling_aware_stream_breakpoints(...)`: Raise a `stream:C` ladder's first chunk for a leaf that has a coarser sibling in its lod group

### `lod_methods.py`
The additive-LOD **ordering-method registry** — the single source of truth for
which orderings exist. Sibling of `lod_breakpoints.py` and here for the same
reason: the implementation lives in `luxar.gsplats.lod.additive`, but the CLI
needs the list too, and importing anything under `luxar.gsplats` executes that
package's `__init__`, which adds ~600 ms on top of the CLI's own ~250 ms import — a 3.4x multiplier on `luxar --help`, stable over 3 runs.
A stdlib-only leaf module is free to import from either side, so the two can share
one list instead of hand-copying it — they previously held two literal tuples with
no consistency test, and the copies had already diverged.

**Key Names:**
- `GSPLAT_ADDITIVE_METHODS` / `GSPLAT_ADDITIVE_CHOICES`: the implemented GSPLAT orderings, and the same set plus the size-adaptive `auto` sentinel accepted at the API/CLI boundary. Prefixed because the ELEMENT-side registry (`core/group/lod/group.py::ADDITIVE_METHODS`) is a different list — only `random` and `radial` overlap — and both are used as `if method not in ...` gates, so an unprefixed collision would let a wrong import silently accept or reject the wrong methods
- `GSPLAT_ADDITIVE_CHOICES_HELP`: `auto|greedy|…` rendered for `--method` help strings, so help text cannot fall out of date
- `MethodName` / `AutoOrMethod`: the `Literal` types, re-exported from `gsplats.lod.additive` for its existing importers
- `REVEAL_METHODS` / `is_reveal_method()`: which orderings are a *reveal* (currently `radial`) rather than a contribution ranking. A reveal's ladder must carry no energy stamps, because the viewer brightens an incomplete ladder by `1/e(k)` — right for an approximation, backwards for a partial object rendered at full brightness. Shared by the gsplat ladder and `core.group.lod.group.additive_level_stats` so the rule is written once for all geometries.

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
- `load_manifest()` / `dataset_spec(name)`: Read the packaged manifest. The parse is memoised but each call returns an independent copy, so mutating the result (or a nested spec) cannot poison later readers; `clear_manifest_cache()` drops the parse after the manifest is rewritten on disk
- `local_fit_path(name, filename)`: Where a demo's OWN locally computed stand-in belongs — `~/.cache/luxar/<name>/local/<filename>`. The cache dir is shared with the fetch but the two namespaces are not: `<name>/<filename>` is the manifest's destination, and `ensure_dataset` quarantines anything there that fails the pinned sha256 — which a local refit never matches, so one stored under the hosted name is destroyed and recomputed on every launch (#1618)
- `load_local_fit_gsplats(name, file_names)`: Load a previous run's own local fit from that namespace, or `None` when the caller must (re)build it — `None` also when a requested file is missing or unreadable, reported loudly since the only recovery for unchecksummed local bytes is the refit. Two things raise instead: a multi-part store (`kind=partition`), because rebuilding would write the same unloadable shape and refit on every launch, and an empty `file_names`, because `[]` is neither a loaded set nor "rebuild it" and would sail through the caller's `is not None` test
- `load_local_fit_gsplats_at(paths)`: The same door for paths the caller already holds. A demo that publishes a module-level `LOCAL_FIT` constant and writes its refit through it must READ through it too, or the two halves can be pointed at different files
- Raises `DatasetNotFound` for an unknown key and `LocalComputeDataset` for data we cannot redistribute (the caller builds it locally). `DatasetUnavailable` (a `FileNotFoundError` subclass) is the narrow "not obtainable from anywhere yet" case a demo may route around by computing its own stand-in; every other `FileNotFoundError` here is a fault (unknown file name, missing packaged manifest, an in-repo copy failing its sha256) and must propagate

### `demos.py`
Demo scene generators, precomputed data helpers, and viewer launch utilities.

**Key Functions:**
- `create_lorenz_attractor()`: Generate Lorenz attractor visualization
- `create_random_spheres()`: Create random spherical points
- `create_time_series_demo()`: Generate time-varying data
- `launch_viewer()`: Launch the Luxar viewer for a given dataset path
- `demo_ports()`: Stable per-dataset (data, viewer) port pair derived from the
  dataset name — demos never contend for 8000/5173, and no two of them share a
  full port PAIR, so a browser tab left over from one demo can never silently
  front another demo's server (two demos may still share only the data port
  and shift, which is harmless — the viewer URL carries its own `?src=`);
  explicit `--port`/`--viewer-port` in `serve_args` override
- `detect_device()`: Auto-detect the best available compute device (cuda > mps > cpu)
- `warn_if_no_cuda_gpu()`: Print a warning if no CUDA GPU is available
- `load_precomputed_gsplats()`: Load precomputed GSplat data from Git LFS or cache
- `load_precomputed_bundle()`: Load a precomputed bundle zip (timelapse demos)
- Raises `BundleMemberNotFound` (a `FileNotFoundError` subclass) when the bundle itself resolved and verified but a requested per-frame member is not inside it. That is the bundle-side counterpart of `data_fetch.DatasetUnavailable`: the member names are derived from the caller's own parameters (NEXRAD's `--dbz-floor` / `--splats` / `--grid-m`), so a non-default run legitimately asks for frames the shipped bundle cannot hold and recomputing is the right answer. A bundle name the *manifest* does not list stays a plain `FileNotFoundError` — that one is a fault
- `parse_demo_flags()`: Parse the common demo CLI flags (`--recompute`, `--no-serve`, `--serve-only`)
- `parse_int_arg(name, default, argv=None)`: Parse an integer `--name=VALUE` / `--name VALUE` flag; warns and falls back to `default` on a malformed value
- `parse_path_arg(name, argv=None)`: Parse a path `--name=PATH` / `--name PATH` flag, expanding `~`; returns `None` when the flag is absent or left without a value
- `is_lfs_pointer()`: Check if a file is a Git LFS pointer (not actual data)
- `voxel_sampled_payload_agreement(centers, payload)`: Fraction of same-voxel splat pairs carrying an identical payload row — the correspondence check for a per-splat sidecar shipped alongside a `.gsplats.zarr` fit (`save()` reorders splats, so a sidecar sampled before saving is silently misindexed). `None` when too few splats share a voxel to judge

**Features:**
- Ready-to-use demo scenes
- Configurable parameters
- Git LFS data loading with local cache fallback
- Educational examples of Luxar features

### `process.py`
Deterministic teardown for long-lived child processes (stdlib-only). Owns the
lifecycle of the subprocess trees `luxar demo run` spawns so Ctrl-C (or
SIGTERM/SIGHUP) never orphans a `luxar serve` on its port.

**Key Functions:**
- `run_child_process()`: Spawn a command, wait for it, and tear it (and its
  whole process group, when isolated) down on every exit path via a
  SIGINT → SIGTERM → SIGKILL escalation; optional `on_spawn` hook receives the
  child PID (= new pgid when isolated)
- `terminate_process_group()`: The same escalation for a group discovered
  after the fact (used by `luxar demo stop`); True only once the group is
  provably finished — an unreaped zombie counts as gone, `EPERM` (someone
  else's group) never does
- `can_kill_process_groups()`: Whether POSIX process-group signalling exists
- `proc_table()`: Best-effort `(pid, pgid, state, command)` rows from `/proc`,
  with a `ps` fallback on POSIX systems such as macOS (empty means *unknown*)

### `demo_runs.py`
Discovery + kill engine behind `luxar demo stop` (stdlib-only): find every
running demo — even one forgotten in another terminal — and free its ports.

**Key Functions:**
- `register_run()` / `unregister_run()`: JSON pidfile per launch under
  `~/.cache/luxar/running/`, written by `demo run`'s `on_spawn` hook and
  removed on exit (so the registry only ever names survivors)
- `discover_runs()`: Live demo runs from the registry plus a `ps` sweep for
  strays — a process that *leads its own group* and is genuinely running
  `python -m luxar.demos.demo_*`; prunes dead/hijacked
  entries, never returns the caller's own process group. On Linux, falls back
  to `proc_table()` when `ps` is missing, so the identity check that keeps a
  recycled pgid alive-and-innocent never silently disappears. Off POSIX, where
  neither exists, a pid listing (`tasklist`) still prunes a record left behind
  by a reboot or a hard-killed owner
- `stop_run()`: Tear one run's process group down via `terminate_process_group`,
  re-validating the group at kill time; returns False without signalling
  anything off POSIX, where a recorded pid cannot be checked before a hard
  terminate
- `describe_port_holder()`: Best-effort "port N is held by demo 'X'" hint
  for `pick_port`'s busy-port warning

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
