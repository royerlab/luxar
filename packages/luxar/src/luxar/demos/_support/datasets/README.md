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

paths = ensure_dataset("example_dataset")
volume_path = paths["volume.npy"]
```

`data_fetch.py` owns manifest resolution and local-fit namespaces. `cache.py`
owns cache paths, metadata, and quarantine behavior. `bundles.py` loads shipped
GSplat stores and archive members. `lfs.py` recognizes packaged Git LFS pointers,
while `payload_agreement.py` checks whether fitted sidecars remained aligned.
Actual HTTP and ZIP mechanics live in the sibling `downloads` package. Modules
inside this directory may collaborate, but callers outside `_support` should
continue to import the exported names from `luxar.demos`.
