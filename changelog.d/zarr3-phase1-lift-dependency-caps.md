#### Lift the anndata and napari ceilings the zarr-2 pin forced

Three version ceilings existed for exactly one reason — they held back packages
that had already moved to zarr 3 while Luxar was pinned to zarr 2. With the
project on zarr 3 they are gone:

- `anndata>=0.10.0,<0.13` → `anndata>=0.10.0`. 0.13 requires `zarr>=3.1`.
- `napari[pyqt6]>=0.4.18,<0.8` → `napari[pyqt6]>=0.8` (the `tracksdata` extra).
- `napari[all]>=0.4.18,<0.8` → `napari[all]>=0.8` (the default dev environment).

The napari ceiling was the actively expensive one. napari 0.8's optional-base
extra requires `zarr>=3.0.8`; unable to satisfy that, the resolver backtracked
napari into 2018-era sdist-only releases whose `setup.py` imports the long-removed
`setuptools.Feature`, and every CI environment build failed on 2026-07-14. That
failure mode is now structurally impossible rather than pinned around.

`metpy` keeps its bounds: they are numpy-driven, not zarr-driven, and the audit
note claiming the pin had been checked against zarr is simply no longer relevant.

Two tests had to be inverted rather than deleted, since they asserted the caps
they were protecting:

- `test_anndata_ceiling_is_present` is replaced by
  `test_every_upper_bound_explains_itself`. The old test pinned one specific
  ceiling; the lesson of watching that ceiling outlive its cause is that the
  durable invariant is not "this bound exists" but "whatever bounds exist, say
  why" — which stays true as caps come and go.
- `test_deps_shows_the_constrained_requirement_not_a_bare_name` now asserts on
  metpy. Checking for `anndata>=0.10` as a substring would no longer prove a
  bound is carried at all, so the assertion moved to a bound that is still live.

The prose that restated these caps is updated in the same pass: the
`_dependencies` module rule (whose motivating example is now metpy's NumPy-2
floor), the `INSTALL_SPECS` note, `demos/README.md`, and the install hints printed
by `demo_zebrahub_velocity_streamlines`,
`demo_gsplats_3d_blastocyst_dapi_nuclei` and
`demo_gsplats_3d_blastocyst_multichannel`
— which were telling users to pin napari below 0.8 on purpose.
