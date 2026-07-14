# Luxar Examples

Short, didactic scripts demonstrating individual Luxar features on synthetic data.
Each script writes a `*_example.luxar.zarr` you can open in the viewer.

> **Note — examples are also test fixtures.** Most of these files are
> referenced by the viewer's E2E suite (`packages/luxar-viewer/src/tests/e2e/`).
> Their generated `.zarr` outputs are part of the test contract. Before editing
> an existing example, grep for its name under `src/tests/e2e/`; any change to
> point counts, colors, RNG seeds, sharpness math, or radii will require
> coordinated updates to the affected specs and visual-regression baselines.
> Net-new examples added per `TEMPLATE.md` have no such constraint.

## Quick start

```bash
# From project root
hatch run python packages/luxar/examples/single_point_example.py
luxar serve <generated_zarr_path> --viewer
```

Each script prints the absolute output path on completion. Defaults to
`datasets/examples/` via `luxar.utils.paths.get_examples_output_dir()`.

## Index

### Getting started

| File | What it teaches |
|---|---|
| `single_point_example.py` | The minimal viable scene (one point at origin). |
| `radius_basic_example.py` | Per-point `radii` parameter. |
| `point_spacing_example.py` | Geometric rule: `spacing = 2 × radius` for touching points. |
| `build_example.py` | Two scene-construction patterns (manual vs structured). |
| `multiple_objects_example.py` | Composing multiple point clouds in one scene. |
| `dimensions_builders_example.py` | `Dimensions.default_3d()` / `default_timeseries()` / `default_multichannel()` / `from_positions()`. |

### Hierarchy and transforms

| File | What it teaches |
|---|---|
| `transform_example.py` | `translate`/`rotate`/`scale`/`compose` and right-multiply convention. |
| `hierarchy_example.py` | Parent-child node hierarchies; property and transform inheritance. |

### Lines

| File | What it teaches |
|---|---|
| `lines_basic_example.py` | `add_lines` with `line_type` of `polyline`, `segments`, `loop`. |
| `lines_indexed_example.py` | `line_type='indexed'` with an explicit `indices` array. |

### Rendering attributes

| File | What it teaches |
|---|---|
| `rendering_modes_example.py` | Blending modes side-by-side (normal vs additive). |
| `rendering_attributes_example.py` | `opacity`, `gamma`, runtime property modification, validation. |
| `sharpness_showcase_example.py` | `sharpness` parameter and edge falloff. |
| `radius_showcase_example.py` | Per-point radii patterns (gradient, distance-based, random, layered). |
| `scalars_and_colormap_example.py` | `scalars=` + `colormap=` for runtime-tunable colouring. |
| `hdr_colors_example.py` | HDR colors (> 1.0) with additive blending and ACES tone mapping. |
| `viewer_config_example.py` | `ViewerConfig` for authored camera, theme, bloom, UI panel state. |
| `overlays_example.py` | `add_text` / `add_image` / `add_html` with `{hover_label}` templates. |
| `hover_labels_example.py` | Per-point `labels=` + `image_labels=` for hover tooltips. |

### nD data

| File | What it teaches |
|---|---|
| `simple_nd_example.py` | 5D grid with visual patterns per time slice. |
| `dimension_navigation_example.py` | Discrete dimension stepping with distinct shapes per frame. |
| `dense_grid_5d_example.py` | Dense 5D grid; `extend_to_all` for spatial reference markers. |
| `dimension_sliders_5d_example.py` | 5D animated spiral with slider UI. |
| `nd_points_example.py` | Comprehensive 5D point data with channel dynamics. |
| `scene_dimensions_example.py` | Custom dimension units, ranges, and step sizes. |
| `rainbow_sphere_4d_example.py` | True 4D spatial geometry (hypersphere slicing). |
| `time_series_4d_example.py` | Rotating spiral evolving over a discrete time dimension. |
| `temporal_spiral_sphere_4d_example.py` | Large 4D temporal animation (stress test — 102M point-records). |
| `radius_slicing_example.py` | nD-hypersphere radius slicing behavior. |
| `nd_transform_example.py` | Per-dimension affine + categorical-permutation `nd_transform`. |
| `categorical_dimensions_example.py` | `Dimension(categories=[...])` for label-driven sliders. |
| `extend_to_all_example.py` | `extend_to_all=[]` vs `['dim']` for axis-marker visibility. |

### Layers and rendering tests

| File | What it teaches |
|---|---|
| `layers_test_example.py` | Layer system + `visible=False`. *(E2E fixture; do not modify.)* |
| `rainbow_sphere_spiral_example.py` | 200K Fibonacci sphere; rendering stress test. |
| `dense_cubic_gradient_example.py` | 1.5M-point cubic lattice; rendering stress test. |

### Performance and memory

| File | What it teaches |
|---|---|
| `progressive_writing_example.py` | `compiler.write_points()` for streaming large datasets. |
| `memory_optimization_example.py` | `EncodingMode.AUTO`/`PRECISION`/`MEMORY` storage trade-offs. |
| `spatial_index_demo_example.py` | Spatial index for efficient nD point loading (5D clusters). |
| `performance_benchmark_example.py` | Large-scene creation benchmark (100 nodes × 1K points). |
| `metal_acceleration_example.py` | Apple Metal/MPS device dispatch for Gaussian splat fitting. |

### LOD and progressive loading

| File | What it teaches |
|---|---|
| `progressive_points_lines_example.py` | Multi-additive LOD (PR α). |
| `partition_of_lod_example.py` | Auto-partition + nested LOD UX. |
| `partition_only_example.py` | Pure spatial partition (auto and manual), no LOD. |
| `lines_partition_and_sampling_example.py` | Polyline-atomic BSP, Poisson-disk LOD, SAH BSP. |
| `energy_breakpoints_example.py` | Energy-aware LOD ordering + explicit coverage_fraction. |
| `points_substitutive_lod_example.py` | `substitutive_lod=` for Points: coarse levels synthesised as mass-preserving Gaussian splats (vs additive decimation baseline). |
| `lines_substitutive_lod_example.py` | `substitutive_lod=` for Lines: segments lifted to gsplat "beads" for smooth, bright coarse levels (vs polyline decimation baseline). |
| `progressive_timelapse_example.py` | Additive LOD ladders across a discrete time dimension; SliceCache instant revisits when scrubbing frames. |

### Gaussian splats

| File | What it teaches |
|---|---|
| `gsplats_basic_example.py` | `add_gsplats` with hand-authored centers, amplitudes, Cholesky factors. |
| `gsplats_fit_volume_example.py` | `fit_gaussian_splats(volume, …)` → `add_gsplats_from_data(…)`. |
| `gsplats_lod_example.py` | Substitutive LOD: build a ladder with `make_substitutive_lod(compression_factor, levels)`, then hand it to `add_gsplats_from_data(..., lod_group=True)`. |

## Keyboard controls (in viewer)

- **1-9** — select dimension to navigate; **`[`** / **`]`** — step backward/forward
- **Left-drag** — pan; **Right-drag** — orbit; **Wheel** — zoom
- **L** — layers panel; **D** — dimension sliders; **H** — help overlay
- Full reference: `packages/luxar-viewer` docs.

## Conventions for new examples

See [`TEMPLATE.md`](TEMPLATE.md) for the full template. Highlights:

- File name: `<feature>_example.py`. Output: `<feature>_example.luxar.zarr`.
- Use `get_examples_output_dir()` for the output path.
- Add one house-style **explainer card** per scene via
  `from _overlay_style import add_explainer` (title → explanation →
  "look for" list). It is what tells a viewer what to see and verify; see
  `TEMPLATE.md` §E.1 for the rules.
- **snake_case** node names (`rainbow_spiral`, not `RainbowSpiral`).
- Prefer `parent_group.add_points(...)` over `scene.add_points(..., parent=parent_group)`.
- Pick **one** color convention per example: float-0-1 OR uint8-0-255, never both.
- Stay ≤ ~100K elements and ≤ ~250 lines. Larger showcases → `src/luxar/demos/`.
- Use `arbol.aprint`/`asection`; explicit `dtype=np.float32` for positions/radii/sharpness.

## Troubleshooting

- **Points not visible** — check the slice position (press 1-9 then `[`/`]`) and that
  positions are inside the dimension `range`.
- **Colors look wrong** — mixed float-0-1 and uint8-0-255 in the same scene triggers
  HDR detection for the >1.0 values. Pick one convention.
- **Performance** — close other tabs, lower density, or use one of the LOD examples
  as a starting point.

## License

Same as the parent Luxar project.
