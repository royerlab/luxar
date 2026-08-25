# Utils Package

The `utils` package provides utility functions for common operations in Luxar, including array manipulation, atomic directory copies, robust downloads, and demo data generation.

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

### `data_fetch.py`
Manifest-driven demo-dataset resolution (R17: retiring in-repo Git LFS in favour
of fetch-on-demand from Zenodo). Reads `demos/data_manifest.json` — the single
source of truth for how each dataset is obtained, its license, and its per-file
sha256.

**Key Functions:**
- `ensure_dataset(name, ...)`: Resolve a dataset's files to local paths, cache -> in-repo Git LFS -> Zenodo. A manifest checksum is authoritative at every step: bytes that fail are quarantined (`.corrupt`) and never returned, so a stale download can never be resumed onto corrupt bytes. There are TWO contracts — `sha256` describes the copy this repo ships, the optional `hosted_sha256` describes what the Zenodo record serves. Bytes already in hand are accepted if they satisfy EITHER (hosted preferred, and taking the local one is reported); only the download leg is strict, on the hosted digest. Splitting them is what keeps a truthful hosted pin from breaking a working checkout — which is what used to force Zenodo publication onto the critical path of every demo-data PR
- `load_dataset_gsplats(name, ...)`: Mirror of `demos.load_precomputed_gsplats` (returns `GSplatData`, `None` on recompute) sourced through `ensure_dataset` — the one-line swap for migrating a demo. Only `zenodo`-bucket datasets are eligible
- `load_manifest()` / `dataset_spec(name)`: Read the packaged manifest. The parse is memoised but each call returns an independent copy, so mutating the result (or a nested spec) cannot poison later readers; `clear_manifest_cache()` drops the parse after the manifest is rewritten on disk
- `local_fit_path(name, filename)`: Where a demo's OWN locally computed stand-in belongs — `~/.cache/luxar/<name>/local/<filename>`. The cache dir is shared with the fetch but the two namespaces are not: `<name>/<filename>` is the manifest's destination, and `ensure_dataset` quarantines anything there that matches neither pinned digest — which a local refit never does, so one stored under the hosted name is destroyed and recomputed on every launch (#1618)
- `load_local_fit_gsplats(name, file_names)`: Load a previous run's own local fit from that namespace, or `None` when the caller must (re)build it — `None` also when a requested file is missing or unreadable, reported loudly since the only recovery for unchecksummed local bytes is the refit. Two things raise instead: a multi-part store (`kind=partition`), because rebuilding would write the same unloadable shape and refit on every launch, and an empty `file_names`, because `[]` is neither a loaded set nor "rebuild it" and would sail through the caller's `is not None` test
- `load_local_fit_gsplats_at(paths)`: The same door for paths the caller already holds. A demo that publishes a module-level `LOCAL_FIT` constant and writes its refit through it must READ through it too, or the two halves can be pointed at different files
- Raises `DatasetNotFound` for an unknown key and `LocalComputeDataset` for data we cannot redistribute (the caller builds it locally). `DatasetUnavailable` (a `FileNotFoundError` subclass) is the narrow "not obtainable from anywhere yet" case a demo may route around by computing its own stand-in; every other `FileNotFoundError` here is a fault (unknown file name, missing packaged manifest, an in-repo copy matching neither pinned digest) and must propagate

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
- `BUILDER_FINGERPRINT_ATTR`: Scene-root attribute that identifies the demo builder
- `demo_source_fingerprint()`: Hash a demo, Luxar's writer sources, and the Zarr environment for scene-staleness checks
- `scene_is_current()`: Reuse only a completed scene written by the current demo producer
- `warn_if_no_cuda_gpu()`: Print a warning if no CUDA GPU is available
- `load_precomputed_gsplats()`: Load precomputed GSplat data from Git LFS or cache
- `load_precomputed_bundle()`: Load a precomputed bundle zip (timelapse demos)
- Raises `BundleMemberNotFound` (a `FileNotFoundError` subclass) when the bundle itself resolved and verified but a requested per-frame member is not inside it. That is the bundle-side counterpart of `data_fetch.DatasetUnavailable`: the member names are derived from the caller's own parameters (NEXRAD's `--dbz-floor` / `--splats` / `--grid-m`), so a non-default run legitimately asks for frames the shipped bundle cannot hold and recomputing is the right answer. A bundle name the *manifest* does not list stays a plain `FileNotFoundError` — that one is a fault
- `parse_demo_flags()`: Parse the common demo CLI flags (`--recompute`, `--keep-stale`, `--no-serve`, `--serve-only`)
- `parse_int_arg(name, default, argv=None)`: Parse an integer `--name=VALUE` / `--name VALUE` flag; warns and falls back to `default` on a malformed value
- `parse_path_arg(name, argv=None)`: Parse a path `--name=PATH` / `--name PATH` flag, expanding `~`; returns `None` when the flag is absent or left without a value
- `is_lfs_pointer()`: Check if a file is a Git LFS pointer (not actual data)
- `voxel_sampled_payload_agreement(centers, payload)`: Fraction of same-voxel splat pairs carrying an identical payload row — the correspondence check for a per-splat sidecar shipped alongside a `.gsplats.zarr` fit (`save()` reorders splats, so a sidecar sampled before saving is silently misindexed). `None` when too few splats share a voxel to judge

**Features:**
- Ready-to-use demo scenes
- Configurable parameters
- Git LFS data loading with local cache fallback
- Educational examples of Luxar features

### `source_fingerprints.py`
Stable fingerprints for Python sources that produce Luxar stores.

**Key Functions:**
- `fingerprint_source_files()`: Hash source paths and contents in stable, boundary-safe order
- `fingerprint_production_sources()`: Hash production Luxar Python sources without caching
- `production_source_fingerprint()`: Cache that production-source hash per package root and process
- `store_writer_environment()`: Report installed/configured inputs that affect Zarr output

### Process lifecycle

Process lifecycle support used by `demos.py` lives in the package-root
`../_process.py`; see `../README.md` for its API and teardown guarantees.

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
- `torch`: Device availability probing in `demos.py`
- `requests` / `urllib3`: HTTP downloads with retry (lazily imported in `download.py`)
