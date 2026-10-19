# A fitted store in a Python scene: what the adder applies

`scene.add_gsplats_from_file(name, path)` on a FITTED store applies what
`luxar gsplat convert` applies, resolved by
`luxar.gsplats.scene_defaults.resolve_fitted_scene`, the function convert itself
calls, so the two cannot drift. On a fresh `Dimensions.default_3d()` scene the
result matches `convert --no-center` byte for byte (same `content_hash`); convert
also centers on the centroid by default (`--center`), the adder does not.

**Fitted** = the store names its columns: a complete root `dimension_metadata`,
or fitting provenance whose `source_shape` has the centers' rank
(`luxar.gsplats.scene_orientation.fitted_column_names`). Classical imports
(`.ply` / `.splat` / `.spz`) and hand-built stores keep the plain adder.
`fitted_defaults=None` (default) = fitted stores only, `True` = any store,
`False` = never.

| `convert` does | the adder's default |
|---|---|
| names the columns from the store (a fit is `z, y, x`) and maps them onto `x, y, z` scene dims | `dim_order` ONLY when this add lays out a fresh scene, or the scene's dims were laid out by an earlier fitted add (second channel); elsewhere the positional mapping stays and a log line names the `dim_order` to pass. A partition is grafted in column order |
| scene dims from the store, y-down XY camera (a camera choice; no data moves) | set on the scene ONLY when the node goes on the root scene, the scene's dims equal `Dimensions.default_3d()` (an explicit `default_3d()` counts as fresh), no camera, no geometry, and no `dim_order` / `fill` / `fill_sigma` / `transform` / `nd_transform` was passed; otherwise left alone, with one log line |
| data units + auto-exposed `display_gain` (opening view's additive p99.9 at the tone-mapping target: 2.0, or 0.9 under `None`/`Linear`), unit labels | `normalize_amplitudes=False` + those attrs, predicted for the scene the node lands in under the tone mapping set AT ADD TIME; SKIPPED when you pass `normalize_amplitudes`, `display_gain`, `amplitude_unit`, `amplitude_offset`, `source_data_range`, `intensity` or `offset` |
| Layers entry | `layer=True` unless passed |
| `blending_mode` / `opacity` / `absorption` written explicitly | same, unless passed |
| 8-bit amplitudes for a uint8/int8 source | scoped on the compiler for the add, unless the compiler got `gsplat_amplitude_bits` |

Not inside a `kind=lod` / `kind=partition` group (siblings share one exposure)
unless `fitted_defaults=True`. A refused add restores the scene's dims and camera.
The store is loaded once, but resolving runs convert's projected-exposure
prediction; for a very large store pass `fitted_defaults=False` to skip it.

```python
# Several channels: the first lays the scene out, the rest reuse it.
for name, path, cmap in [("membrane", "mem.gsplats.zarr", "bop_orange"),
                         ("nuclei", "nuc.gsplats.zarr", "bop_blue")]:
    scene.add_gsplats_from_file(name, path, colormap=cmap)
```

For a scene you lay out yourself, `fitted_scene_defaults(path, **options)` returns
convert's scene as values: `dimensions` / `viewer_config` for `create_scene`,
`amplitude_bits` for `LuxarZarrCompiler(gsplat_amplitude_bits=...)`, and the
adder kwargs in `attrs` (options as convert's flags: `colormap`, `tone_mapping`,
`blending_mode`, `opacity`, `absorption`, `dim_order`, `view`, `display_gain`,
`auto_exposure`, `target_brightness`). `luxar.gsplats.exposure.auto_exposure_attrs(path,
dimensions=..., dim_order=..., camera=...)` returns the exposure kwargs alone
(convert's resolver; without `dimensions=` it uses convert's own layout).

On the CLI, `--display-gain` is ABSOLUTE (brightness per data unit), not a
multiplier on the auto gain; the `Exposure:` line convert prints names the auto
gain. `--blending-mode volumetric` has no auto exposure (and its emission is the
density the absorption acts on, so a small `--absorption` opens saturated), so
convert prints the gain additive auto exposure would use and the `--absorption`
that gives about one optical depth at that gain.
