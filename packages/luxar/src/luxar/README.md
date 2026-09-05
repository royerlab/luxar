# luxar

Core Luxar Python package and public API exports.

## Scope

- Defines module behavior and public entry points.
- Keeps tests/demos aligned with package semantics.

## Subpackages

| Subpackage | Purpose |
|---|---|
| `core/` | Scene graph nodes (Scene, Group, Points, Lines, GSplats, Mesh), dimensions, transforms, viewer config |
| `io/` | Zarr compiler (`LuxarZarrCompiler`), reader (`LuxarScene`), writer utilities |
| `encoding/` | Semantic type-based encoding (quantization, broadcasting, LUT) |
| `mesh/` | Mesh validation and geometry helpers |
| `shading/` | Geometry-derived appearance bakes such as ambient occlusion |
| `colormaps/` | Colormap definitions and utilities |
| `gsplats/` | Gaussian splatting pipeline (fitting, rendering, merging, CLI, CUDA/Metal backends) |
| `cli/` | Command-line interface (`luxar` command) |
| `demos/` | Built-in demo datasets |
| `validation/` | Input validation and nD transform validation |
| `typing_utils/` | Type aliases, enums, and configuration dataclasses |
| `utils/` | Array utilities, console verbosity, LOD policy, path management, reusable scene generators |
| `tests/` | Top-level test suite |

## Package-Root Modules

### `_process.py`

Deterministic teardown for long-lived child processes. It owns the lifecycle of
the subprocess trees `luxar demo run` spawns so Ctrl-C, SIGTERM, or SIGHUP never
orphans a `luxar serve` process on its port.

**Key Functions:**
- `run_child_process()`: Spawn a command, wait for it, and tear it and its whole
  process group down on every exit path via SIGINT → SIGTERM → SIGKILL; the
  optional `on_spawn` hook receives the child PID, which is also the process
  group ID when isolated
- `terminate_process_group()`: Apply the same escalation to a group discovered
  after the fact; returns True only once the group is provably finished — an
  unreaped zombie counts as gone, while `EPERM` never does
- `can_kill_process_groups()`: Report whether POSIX process-group signalling is
  available
- `proc_table()`: Return best-effort `(pid, pgid, state, command)` rows from
  `/proc`, with a `ps` fallback on POSIX systems such as macOS; an empty result
  means the process table is unknown, not that nothing is running

## Key Exports

See `__init__.py` for the full public API. Primary classes:

- `LuxarZarrCompiler` -- progressive Zarr writer (context manager)
- `LuxarScene` -- reader for compiled Luxar Zarr archives
- `Node`, `Scene`, `Group`, `Points`, `Lines`, `GSplats` -- scene graph nodes
- `Overlay` -- screen-space text/image/HTML annotation descriptor
- `Dimensions`, `Dimension` -- nD coordinate system definitions
- `ViewerConfig`, `CameraConfig`, `UIConfig` -- viewer configuration
- `transforms` -- 4x4 matrix utilities (translate, rotate, scale, compose, etc.)
- `validate_nd_transform`, `compose_nd_transforms`, `apply_nd_transform_to_bounds` -- nD per-dimension transform helpers
- `GSplatData`, `fit_gaussian_splats` -- Gaussian splatting (optional, requires torch)
- `set_verbosity`, `get_verbosity`, `verbosity` -- console-output level (see `utils/README.md`)

Run the CLI as a module with `python -m luxar` (`__main__.py` dispatches to `luxar.cli.app`). There are no top-level re-export modules: import from the owning subpackage (`luxar.core.transforms`, `luxar.core.dimensions`, `luxar.typing_utils.config`, `luxar.io.compiler`), or use the names `__init__.py` re-exports.
