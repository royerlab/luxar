#### Core scene authoring no longer needs the `gsplats` extra

`scene.add_gsplats(...)` is part of the core authoring API, but on a plain
`pip install luxar` it died with `ModuleNotFoundError: No module named 'torch'`.
The unconditional trigger is in the compiler: `write_gsplat_arrays`
(`io/_compiler/gsplat_assembly.py`) does `from ...gsplats.utils.trils import
split_tril` on every `add_gsplats`, to split the Cholesky diagonal from the
off-diagonal before writing. `trils` is pure NumPy, but importing that submodule
executes the parent package's `__init__`, which eagerly pulled the torch-only
`device` module. (`core/group/dim_order.py` reaches the same package for
`embed_cholesky_packed` — a second route, but only when the caller passes
`dim_order=`.) The two torch-backed exports (`resolve_torch_device`,
`is_mps_available`) are now resolved lazily through a PEP 562 module
`__getattr__`, with a matching `__dir__` so they stay visible to `dir()` and to
the Sphinx API reference. `from luxar.gsplats.utils import resolve_torch_device`
still works for the fitting code that needs it, while the core path pays
nothing.

`GSplatData` had the same shape of problem one level up: the container and its
zarr writer are pure NumPy, but `luxar/gsplats/__init__.py` imported them inside
the guard that degrades the whole subsystem to install-hint stubs when the extra
is absent, so a core-only install could not construct or save a `.gsplats.zarr`
even though nothing about doing so requires the extra. `GSplatData`,
`AdditiveSubLOD` and `SubstitutiveLevel` are now imported unguarded. The scope
of that exception is constructing, saving, loading, geometrically transforming,
and grafting into a scene. Content-EDITING still needs the extra: intensity
rescaling, a reducing `filter`/`filter_by` and the heuristic `cull` methods
(hence a bare `cull()`, whose `auto` resolves to `cumulative`) recompute stats
through `luxar.gsplats.lod`, which imports `scipy.sparse`, while the
rendering-based `cull` methods `error_budget`/`redundancy` hit `import torch`
sooner still. The package docstring now states exactly where that line falls.

The `gsplats` extra is `scipy` *and* `torch`, so the subprocess regression test
blocks both: it compiles a real four-geometry scene plus a standalone
`.gsplats.zarr` with the whole extra made unimportable by a `sys.meta_path`
finder, then reads the store back and asserts the arrays. The contract is "core
authoring works without the extra", not merely "the module imports".

While here, `gsplats/interop/tracksdata.py` drops its inlined `_unpack_tril`
copy — it existed only to avoid the eager torch import this change removed — and
calls `luxar.gsplats.utils.trils.unpack_tril` directly.
