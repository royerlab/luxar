#### Hosted gsplat demos fetch through the manifest

Thirteen gsplat demos still resolved their precomputed data with
`load_precomputed_gsplats` / `load_precomputed_bundle`, which read
`demos/data/<dir>/` and the local cache and consult the manifest at no point. Their
checksums were pinned and never verified, the Zenodo record could not be reached
even once its URL was populated, and the moment the git-LFS payload leaves the
repository each of them would simply stop working. Only four gsplat demos were on
the manifest-driven path.

They now call `load_dataset_gsplats`, which was written for exactly this swap:
same argument order, same "list of `GSplatData`, or `None` if you must build it
yourself" contract, but resolution goes through `ensure_dataset` — checksum-verified
cache, then the in-repo copy, then Zenodo.

Timelapse demos ship one outer zip holding many per-frame stores and had no
manifest-driven equivalent, so `load_dataset_bundle` is new. It verifies the outer
bundle, which is the unit that gets downloaded and therefore the thing worth
checksumming, then reuses the existing extraction path rather than growing a second
copy of it — that logic has to reject archive members that name `../` and has to
re-extract when the bundle itself changes, and neither property survives being
reimplemented alongside. On the manifest-driven path the "has it changed" question
is answered with the sha256 that was just verified rather than the in-repo path's
`(size, mtime)` guess, so a re-upload cannot leave stale frames extracted behind it.

Six demos deliberately stay on the in-repo loader: `gsplats_tribolium`,
`gsplats_acto3d_heart` and `gsplats_tng_cosmic_web` are `local-compute`, so the
manifest-driven helper returns `None` for them by design and migrating would send
those demos into a from-scratch GPU refit instead of loading the file that is
sitting right there. A test names that set exactly, and asserts every member of it
is still `local-compute`, so it cannot quietly grow.

The gate that holds the rest checks the direction that matters: a `zenodo`-bucket
dataset must not be reached by a loader that never consults the manifest.
