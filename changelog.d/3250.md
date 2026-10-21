#### Organoid and EASI-FISH scenes carry their CC BY 4.0 attribution

The budding-organoid and EASI-FISH hypothalamus demos now state their data
attribution inside the scene, so it travels with the store wherever it is
hosted. Each scene description gains a sentence giving the source data DOI, the
CC BY 4.0 licence with its link, and the changes made to the data: for the
organoid, membrane unmixing, the frame selection, splat fitting and the lineage
overlay; for EASI-FISH, tile stitching, block-mean downsampling, background
subtraction, splat fitting and rescaling to pre-expansion micrometres. The size
sentence stays last and is still restated by the build. ACKNOWLEDGMENTS.md
carries the same licence link and changes, and both demos are now gallery tiles.

The organoid docstring now describes its frame selection exactly. It shows
every fourth acquisition slot, but the slot of the empty T0245 draws T0244, so
the frames actually drawn step by 4 everywhere except 3 then 5 around it.

#### Organoid and EASI-FISH fits are hosted on Zenodo

Both demos now load finished fits from the Luxar cc-by Zenodo record (v2.1.0,
doi:10.5281/zenodo.23524429) instead of refitting on first run. The organoid
demo resolves `organoid_budding_frames.gsplats.zarr.zip` (98.8 MB, 334
per-frame stores) and still fetches only the 0.9 MB lineage from Zenodo 6828906;
the EASI-FISH demo resolves its four per-channel archives (83.2 MB) instead of
downloading the 34.9 GB CZI. Neither needs a GPU on the hosted path. A
complete local fit cache from an earlier build still wins, with no network;
the hosted fits come next, and the local build last, when nothing is cached and
the download fails (offline, timeout, HTTP error). A hosted file that fails its
checksum still stops the run. `--recompute` forces the local build (now also on
the EASI-FISH demo, which warns that `--czi` is ignored without it). Cached
hosted files are verified by hash and reused without the network. The EASI-FISH
scene records which inputs it was built from, so switching between hosted and
local fits rebuilds it. `luxar.demos.is_download_failure` tells a failed
download apart from a checksum mismatch or a local fault. `load_dataset_bundle` gains an opt-in `include_stats`, which the organoid
demo uses so its hosted scene keeps the same per-part fit provenance as a local
build.

The manifest points the cc-by record at v2.1.0; every earlier file carries over
byte-identical, so no existing pin moved. The record text snapshots were
re-captured, and the README's data table now counts 27 of the 33 datasets as
hosted.
