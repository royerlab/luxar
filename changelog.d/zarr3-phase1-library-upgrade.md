#### Move to zarr-python 3, still writing zarr format 2

The core pin goes from `zarr>=2.16,<3.0` to `zarr>=3.2,<4`. **Nothing on disk
changes.** Luxar still writes zarr *format 2* — `.zgroup` / `.zattrs` / `.zarray`
documents, dot-separated chunk keys, a consolidated `.zmetadata`, the same
Blosc/zstd-9 width-aware shuffle policy, and the same `luxar_delta_v1` v2 filter.
Existing `.luxar.zarr` and `.gsplats.zarr` stores are read and rewritten
unchanged, and the TypeScript viewer is untouched.

The library version and the on-disk format are two independent axes, and until now
only one of them was modelled anywhere. Both are now pinned in a single new module,
`luxar._zarr_compat` (`ZARR_FORMAT = 2`), which is the only place in the package
that names a zarr format. Writers go through its helpers — `open_group`,
`create_array`, `create_root_group`, `consolidate`, `close`, `open_store`,
`memory_group`, `is_missing_error` — instead of calling `zarr.*` directly. That
mirrors the viewer's `src/data/zarr.ts`, whose single-import design is why the
TypeScript side needed almost no work here; the Python side had 53 production
modules importing `zarr` (171 counting tests), which is what made this worth doing
as its own step.

What the upgrade buys, and why it was worth doing now: zarr-python 2.18 cannot open
a zarr **v3** store at all, so any v3 input — a GEFF graph from the tracking
ecosystem, an OME-Zarr written by a newer tool — was simply unreadable. Reading is
version-agnostic on 3.x, so those now open. It also retires two transitive
ceilings the old pin forced: `anndata<0.13` and `napari<0.8` (in two places), the
latter having broken every CI environment build once already when the resolver
backtracked napari into 2018-era sdists.

The most consequential behaviour change is one that has nothing to do with the
format: **zarr 3 trusts consolidated metadata on read, and zarr 2 did not.** zarr 2
only consulted `.zmetadata` through the separate `zarr.open_consolidated`, so a
plain `open_group` always reported the arrays that were actually on disk. Under
zarr 3's default, deleting an array directory from a consolidated store leaves it
still *appearing* present — `"x" in group` is True and `array_keys()` still lists
it — because the answer comes from a stale index. That silently disables Luxar's
detection of a partially written store, which is precisely the failure mode the
writers' crash-safety machinery exists to bound: a killed merge used to leave a
partial store that the existence-gated batch-merge resume then treated as
complete, and the readers' "this required array is missing" guards are the
backstop. Reads therefore pass `use_consolidated=False`, restoring zarr-2
semantics. The viewer is unaffected — it fetches `.zmetadata` itself over HTTP,
which is what consolidated metadata is actually for.

Four more zarr-3 behaviour changes were neutralised in the facade rather than left
to call sites, because each fails *silently* or breaks a default:

- **An omitted compressor is not "no compressor".** zarr 3 defaults `compressors`
  to `"auto"`, which is Blosc/lz4/clevel-5. Luxar stores its packed label blobs
  RAW and everything else under a measured zstd-9 policy, so `None` maps to an
  explicit `compressors=None`, and an AST test fails the build if a production
  call site stops naming its compressor. (`"auto"` remains available and matches
  zarr 2's implicit default exactly, so test fixtures that never named one keep
  their old bytes.)
- **`data=` and `shape=` became mutually exclusive.** zarr 2 accepted both and
  several writers pass both; `create_array` accepts both and forwards only what
  zarr 3 permits, rejecting a genuinely contradictory pair.
- **`zarr.group()` now means format 3.** Around 700 test call sites construct
  stores directly; an autouse session fixture in `luxar/conftest.py` pins the
  ambient default to 2 so a fixture looks like real Luxar output. This cannot mask
  a production regression — the facade passes `zarr_format` explicitly and never
  reads that config, and a test flips the global default to 3 and asserts the
  facade still writes 2.
- **`chunks=True` is rejected outright** ("True is not a valid chunk input"). zarr 2
  spelled "choose chunks for me" that way; `luxar.typing_utils.ChunkSpec` still
  carries it and `create_resizable_dataset` *defaults* to it, so every resizable
  dataset would have failed. Mapped to `"auto"` (and `False` to a single chunk
  spanning the array).

Mechanical API churn along the way: `create_dataset` → `create_array` (109 call
sites; `create_dataset` was removed in zarr 3), `DirectoryStore` →
`zarr.storage.LocalStore`, `zarr.MemoryStore` → `zarr.storage.MemoryStore`, and
`Group.close()` (gone in zarr 3) behind `_zarr_compat.close`. The `maxshape=`
kwarg is no longer forwarded to zarr: it was an h5py-compatibility argument that
zarr 2 ignored — zarr arrays are always resizable — and it stays on Luxar's own
`create_resizable_dataset` signature, where it has never constrained anything.

Two claims worth correcting for anyone reading the migration notes: `zarr.group`,
`zarr.open`, `zarr.open_consolidated` and `zarr.consolidate_metadata` all still
exist in zarr 3, and of the two error names Luxar caught only
`PathNotFoundError` was removed — `GroupNotFoundError` survives and now subclasses
`FileNotFoundError`, which is why a single `except FileNotFoundError` correctly
replaces the old two-name tuple.
