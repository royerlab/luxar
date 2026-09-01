# Demo Dataset Support

This private package owns how demo datasets are named, located, cached, and
loaded. It resolves the packaged manifest, distinguishes hosted artifacts from
locally computed fallbacks, validates content identities, loads precomputed
GSplat bundles, and records cache metadata. The implementation is deliberately
behind the `luxar.demos` barrel so demo authors get one stable import spelling
while manifest and storage policy remain free to evolve.

## Quick Start

Resolve a published dataset through the public demo API:

```python
from luxar.demos import ensure_dataset

(volume_path,) = ensure_dataset("example_dataset")
```

## Key Functions

- `ensure_dataset()` resolves cache, in-repo Git LFS, then hosted data. Existing
  bytes prefer `hosted_sha256`, then `sha256`; the newest
  `superseded_sha256` is accepted only when no current copy can replace it. A
  declared `positional_pair` additionally requires every partner to resolve to
  that same generation, otherwise the whole dataset is refused. Unrecognized
  bytes are quarantined as `.corrupt` and never returned or used as a
  download-resume base. `file_names` selects an exact subset in manifest order;
  unknown names and selections that split a positional pair are rejected.
- `load_dataset_gsplats()` loads a manifest-backed Zenodo GSplat dataset, or
  returns `None` when recomputation is requested.
- `load_manifest()` and `dataset_spec()` return independent copies of the
  memoized manifest; `clear_manifest_cache()` refreshes it after an on-disk
  rewrite.
- `local_fit_path()` stores a demo's own refits under
  `~/.cache/luxar/<name>/local/`, separate from manifest destinations. Mixing
  those namespaces makes `ensure_dataset()` quarantine every unpinned refit.
- `load_local_fit_gsplats()` and `load_local_fit_gsplats_at()` return `None` for
  missing or unreadable local fits so callers rebuild them. They raise for an
  empty file list or a partition store, because neither can be repaired by
  repeatedly taking the same refit path.
- `load_precomputed_gsplats()` loads precomputed GSplat data from Git LFS or
  cache.
- `load_precomputed_bundle()` loads a precomputed bundle ZIP for timelapse
  demos. It raises `BundleMemberNotFound` when a verified bundle does not
  contain the requested per-frame member; see the exception docstring for the
  distinction from manifest lookup failures.
- `is_lfs_pointer()` checks whether a file is a Git LFS pointer rather than the
  materialized payload.
- `voxel_sampled_payload_agreement()` measures whether same-voxel splat pairs
  carry identical payload rows, catching sidecars that became misindexed when
  `save()` reordered splats. It returns `None` when too few pairs exist to
  judge.

`data_fetch.py` owns manifest resolution and local-fit namespaces. `cache.py`
owns cache paths, metadata, and quarantine behavior. `bundles.py` loads shipped
GSplat stores and archive members. `lfs.py` recognizes packaged Git LFS pointers,
while `payload_agreement.py` checks whether fitted sidecars remained aligned.
Actual HTTP and ZIP mechanics live in the sibling `downloads` package. Modules
inside this directory may collaborate, but callers outside `_support` should
continue to import the exported names from `luxar.demos`.
