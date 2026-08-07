---
name: luxar-visualization
description: >-
  Build a Luxar visualization for a dataset. Use when a user wants to turn data
  (point clouds, trajectories/lines, nD images, volumes, time series, multichannel
  stacks) into a .luxar.zarr scene and view it in the web viewer. Covers the
  LuxarZarrCompiler -> create_scene -> add_points/add_lines/add_gsplats ->
  serve/export flow, Dimensions, transforms, hierarchy, and the demo/example
  patterns that are the canonical source of know-how.
---

# Build a Luxar visualization

Luxar compiles nD scientific scenes (Points, Lines, GSplats, Mesh) to a `.luxar.zarr`
archive served to a WebGL viewer. This skill builds a scene from a dataset.

**The repo's demos and examples are the canonical know-how** — when in doubt, read a
matching one before writing code:
- `packages/luxar/src/luxar/demos/demo_*.py` (~67 complete demos)
- `packages/luxar/examples/*_example.py` (~51 focused examples)

## The canonical pattern (every demo follows this)

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions
from luxar.utils.demos import launch_viewer   # convenience: shells to `luxar serve --viewer`

output_path = "my_scene.luxar.zarr"

with LuxarZarrCompiler(output_path) as compiler:        # 1. open the compiler
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())  # 2. dims REQUIRED
    scene.add_points(                                   # 3. add geometry nodes
        "MyData",
        positions,                                      # (N, 3) float32
        colors=colors,                                  # (N, 3) float32/uint8
        radii=radii,                                    # (N,) or scalar
    )
# 4. context exit finalizes the store

launch_viewer(output_path)                              # 5. view it
```

That is the whole shape. Build data → `LuxarZarrCompiler` → `create_scene(dimensions=)`
→ `add_points` / `add_lines` / `add_gsplats` → serve. See `references/scene-api.md`
for every method signature, `**attrs`, transforms, and overlays.

## Pick the geometry type

| Data | Node | Required arrays |
| --- | --- | --- |
| Point cloud, particles, cells | `add_points` | `positions` (N,D) |
| Trajectories, skeletons, vectors, meshedges | `add_lines` | `vertices` (N,D) + `widths` |
| Triangle surfaces, isosurfaces | `add_mesh` | `vertices` (V,D) + `faces` (F,3); optional normals (with `normal_dims`), colors, scalars |
| Volumes / dense nD images (fitted to Gaussians) | `add_gsplats*` | centers/amplitudes/cholesky, or a fitted `.gsplats.zarr` |

For raw volumes, fit Gaussian splats first — see the **`luxar-gsplat-pipeline`**
skill, then `add_gsplats_from_file` / `add_gsplats_from_volume` here.

## Dimensions = the coordinate system

`Dimensions` is the single source of truth for the scene's axes. Displayed dims are
the viewport; non-displayed dims (time, channel, z) become nD slider axes.

```python
from luxar import Dimensions, Dimension

# 3D spatial
dims = Dimensions.default_3d()

# 4D time series: t is a non-displayed, discrete slider axis
dims = Dimensions([
    Dimension("t", unit="s", display=False, discrete=True, range=(0, 100), step=1),
    Dimension("z", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("x", unit="um", display=True),
])
```

`positions` columns must match the dimension order. Use `extend_to_all=` on a node to
broadcast geometry across a non-displayed dim (e.g. show the same points at every t).

## Common recipes

```python
# Scalars -> colormap (instead of explicit colors)
scene.add_points("field", pts, scalars=values, colormap="viridis")

# Lines: polyline trajectory with tapering width
scene.add_lines("track", verts, widths=w, line_type="polyline", blending_mode="additive")

# Hierarchy: a group with a transform; children compose under it
from luxar import transforms
g = scene.add_group("left", transform=transforms.translate(x=-50))
g.add_points("nuclei", pts)

# Fitted gsplats from a volume, in one step
scene.add_gsplats_from_volume("embryo", volume, seeds=8000, n_iters=5000, device="cuda")

# On-screen label
scene.add_text("Embryo, t=0", position=(0.02, 0.02), font_size=0.05, anchor="top-left")
```

## Demo helpers (for scripts that fit/serve)

From `luxar.utils.demos`:
- `launch_viewer(output_path, open_browser=True)` — serve via the CLI viewer.
- `parse_demo_flags()` — standard `--recompute` / `--no-serve` / `--serve-only` flags.
- `load_precomputed_gsplats(demo_name, file_names, recompute=...)` — load cached
  (Git-LFS) fitted splats, refitting only when `--recompute`.
- `create_lorenz_attractor`, `create_random_spheres`, `create_time_series_demo`,
  `detect_device`, `warn_if_no_cuda_gpu`.

From `luxar.utils.paths`: `get_demos_output_dir()`, `get_examples_output_dir()` for
output locations (never commit `.luxar.zarr`).

## Serve & export

```bash
luxar serve my_scene.luxar.zarr --viewer --open     # interactive viewer
luxar export my_scene.luxar.zarr -o out/ --open     # standalone offline folder
luxar info my_scene.luxar.zarr --stats              # inspect a built scene
```

## Gotchas (from the codebase)

- `create_scene` REQUIRES `dimensions=`.
- GSplat `cholesky_factors` are packed lower-triangular: 3D = `(N,6)`
  `[L00, L10, L11, L20, L21, L22]`; isotropic std σ -> `[1/σ,0,1/σ,0,0,1/σ]`.
- `transforms.compose(t1, t2, t3)` applies `t1` FIRST.
- 4D/nD scenes can show 0 elements at a given slice — navigate to a populated slice,
  or use `extend_to_all` to broadcast across a non-displayed dim.
- Match `positions` column order to the `Dimensions` order.

See `references/scene-api.md` for exact signatures and `**attrs`. For volume→splats,
use the `luxar-gsplat-pipeline` skill.
