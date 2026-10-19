#### Breaking: `add_gsplats_from_file` now builds `gsplat convert`'s scene for a fitted store

Breaking: an existing script that adds a fitted `.gsplats.zarr` with
`add_gsplats_from_file` and does not pass `fitted_defaults=False` now renders
differently. Data-unit amplitudes with an auto-exposed `display_gain` replace
per-splat normalisation, and a fresh default scene gets the fit's dimensions,
column mapping and a y-down camera. To keep the previous behaviour exactly, add
`fitted_defaults=False` to the call.

The documented `scene.add_gsplats_from_file(name, "fit.gsplats.zarr")` example
on `Dimensions.default_3d()` opened a fitted stack edge-on, saturated and missing
from the Layers panel, while `luxar gsplat convert` on the same store looked
right. For a FITTED store (one with a root `dimension_metadata`, or fitting
provenance whose `source_shape` matches the centers) the adder now applies
convert's defaults itself. It keeps the amplitudes in data units with an
auto-exposed `display_gain` and unit labels, predicted under the tone mapping
the scene has when the node is added. It also sets `layer=True` and convert's
explicit `blending_mode` / `opacity` / `absorption`, and writes 8-bit amplitudes
for a uint8 source. Node placement changes only on a fresh scene: when the node
goes on the root scene and that scene's dimensions equal `Dimensions.default_3d()`
(a scene explicitly created with it counts), with no camera and no geometry, the
adder also sets convert's scene dimensions, maps the `z, y, x` columns onto
`x, y, z` and adds the y-down XY camera. A second fitted add on that scene reuses
the mapping. On any other scene the dimensions, camera and positional column
mapping are left alone, and a log line names the `dim_order` and `create_scene`
arguments for convert's layout. On a fresh scene the result has the same
`content_hash` as `convert --no-center` (convert also centers by default). The
store is loaded once, but resolving runs convert's projected-exposure prediction;
for a very large store, `fitted_defaults=False` skips it.

An explicit argument always wins: `dim_order` / `fill` / `fill_sigma` /
`transform` / `nd_transform` keep your orientation and your scene, and `normalize_amplitudes`, `display_gain`,
`amplitude_unit`, `amplitude_offset`, `source_data_range`, `intensity` or
`offset` keep the old per-splat normalisation, since a window or gain authored
against normalised amplitudes would otherwise be read in data units. Inside a
`kind=lod` or `kind=partition` group nothing changes. Classical imports
(`.ply` / `.splat` / `.spz`) and stores without axis names or fitting provenance
keep the old behaviour. `fitted_defaults=True` applies the defaults to any
`.gsplats.zarr` store.
Every in-repo demo, the render-gate scene generator and the viewer fixture
generator now pass `fitted_defaults=False`, so their output is unchanged.

The decisions come from one function,
`luxar.gsplats.scene_defaults.resolve_fitted_scene`, which convert now calls too,
so the CLI and Python paths cannot drift. `fitted_scene_defaults(path, ...)`
returns the same scene as values (`dimensions`, `viewer_config`,
`amplitude_bits`, adder `attrs`) for a scene you lay out yourself.
`luxar.gsplats.exposure.auto_exposure_attrs` now delegates to convert's exposure
resolver: without `dimensions=` it predicts for convert's own layout, so it gives
convert's gain on a partition and accepts any column count. Convert's orientation
and exposure modules moved from `luxar.cli.gsplat_ops` to
`luxar.gsplats.scene_orientation` and `luxar.gsplats.scene_exposure`.

`--blending-mode volumetric` still has no auto exposure, and its emission is the
same density the absorption acts on, so a small `--absorption` opened as
saturated as additive. Convert now also prints the `--display-gain` additive auto
exposure would use and the `--absorption` that gives about one optical depth at
that gain. The `--display-gain` help now says the gain is absolute (brightness per
data unit), not a multiplier, and its example no longer suggests 0.1.
