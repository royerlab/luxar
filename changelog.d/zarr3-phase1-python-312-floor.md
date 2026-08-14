#### Raise the Python floor to 3.12 and uncap numpy

`requires-python` moves from `>=3.10` to `>=3.12`, and the mypy target and ruff
`target-version` move with it. This is the first step of the migration off
zarr-python 2: every zarr 3 release requires Python >= 3.11, and 3.2 onward
requires >= 3.12, so the floor has to move before the library can. Nothing about
zarr itself changes here — the project still resolves `zarr>=2.16,<3.0` and still
writes zarr format 2 stores, byte for byte.

The `numpy>=2.0,<2.5` cap is lifted as a direct consequence. That cap existed only
because mypy checked numpy's PEP 695 stubs against `python_version = "3.10"`, where
a `type` statement is a syntax error; at 3.12 the stubs parse and numpy 2.5 is clean.
Four latent type errors that numpy 2.5's more precise stubs expose are fixed rather
than silenced: a float64 normal array assigned into a float32 list in the glTF
importer, a loop variable rebound from `int` to a numpy integer in the batched
spatial hash, an unnarrowed `np.squeeze` result in `load_volume`, and a
`min(int, np.signedinteger)` in gsplat culling.

The CI Python matrix collapses to a single 3.12 leg. The nightly cron stays, because
its remaining value was never the matrix: a scheduled event has no PR base, so the
change-detection job cannot path-filter and selects the whole suite plus the
documentation gate. The `tomli` backport is dropped from the dev extra now that
`tomllib` is always stdlib.

Collapsing the matrix exposed a second, older bug: the `test` extra never declared
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
