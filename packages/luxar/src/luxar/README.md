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
| `colormaps/` | Colormap definitions and utilities |
| `gsplats/` | Gaussian splatting pipeline (fitting, rendering, merging, CLI, CUDA/Metal backends) |
| `cli/` | Command-line interface (`luxar` command) |
| `demos/` | Built-in demo datasets |
| `validation/` | Input validation and nD transform validation |
| `typing_utils/` | Type aliases, enums, and configuration dataclasses |
| `utils/` | Array utilities, download helpers, path management |
| `tests/` | Top-level test suite |

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

Run the CLI as a module with `python -m luxar` (`__main__.py` dispatches to `luxar.cli.app`). There are no top-level re-export modules: import from the owning subpackage (`luxar.core.transforms`, `luxar.core.dimensions`, `luxar.typing_utils.config`, `luxar.io.compiler`), or use the names `__init__.py` re-exports.
