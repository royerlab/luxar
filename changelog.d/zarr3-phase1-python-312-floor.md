#### Raise the Python floor to 3.12 and uncap numpy

`requires-python` moves from `>=3.10` to `>=3.12`, and the mypy target and ruff
`target-version` move with it. The floor is what the move to zarr-python 3
requires: every zarr 3 release needs Python >= 3.11, and 3.2 onward needs
>= 3.12. It is called out separately because it is the part with consequences of
its own beyond zarr — everything below follows from raising the floor, not from
changing the zarr pin, and none of it changes a byte on disk.

The `numpy>=2.0,<2.5` cap is lifted as a direct consequence. That cap existed only
because mypy checked numpy's PEP 695 stubs against `python_version = "3.10"`, where
a `type` statement is a syntax error; at 3.12 the stubs parse and numpy 2.5 is clean.
Four latent type errors that numpy 2.5's more precise stubs expose are fixed rather
than silenced: a float64 normal array assigned into a float32 list in the glTF
importer, a loop variable rebound from `int` to a numpy integer in the batched
spatial hash, an unnarrowed `np.squeeze` result in `load_volume`, and a
`min(int, np.signedinteger)` in gsplat culling.

The CI Python matrix is rebased on the new floor rather than shrunk: a pull request
runs 3.12 (the floor, and the required status context), while push-to-main and the
nightly run 3.12, 3.13 and 3.14. `requires-python = ">=3.12"` has no ceiling — 3.13
and 3.14 are supported, `install-hatch` prefers them, and a developer's Hatch
environment picks the newest interpreter on the box — so the version most people
actually run is exercised too, within 24 h rather than on every PR. The off-PR set
is exactly what the published classifiers advertise (3.12–3.14), so "declared" and
"tested" cannot drift apart; the `test` Hatch matrix carries the same three legs. The `tomli` backport is dropped from the dev
extra now that `tomllib` is always stdlib.

Reworking the matrix exposed a second, older bug: the `test` extra never declared
`httpx`, which `fastapi.testclient` needs. It reached the default environment
transitively through a `dev` dependency, so it was invisible where CI runs; but the
per-version `test` matrix environments got none, and `test_cli_integration.py` died
during collection — which pytest reports with a **zero** exit status, so the run
looked green. `httpx` is now declared next to the pytest dependencies.

One real gap closed on the way: the demo entry-point preflight scanner matched only
`ast.Try`, skipping `except*` handlers on the grounds that they were a syntax error
under the old floor. `ast.TryStar` is a sibling of `ast.Try` rather than a subclass,
so at 3.12 that exemption would have become a live blind spot; both node types are
now matched.
