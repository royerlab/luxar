#### Fetch the neuromast timelapse from its record instead of a machine-local store

`demo_gsplats_4d_neuromast_2ch` was the last demo that read a hosted dataset
without going through `ensure_dataset`. Its two fitted channels were already on
the `cc-by` Zenodo record and already pinned by SHA-256 in
`data_manifest.json`, but the record was an unpublished draft — and an
unpublished record builds no download URL — so the demo read
`~/luxar_demo_data/gsplats_neuromast_2ch/` and ran only on a machine where
somebody had put the files there by hand. Its own docstring recorded the
follow-up: publish the record, then switch to `ensure_dataset` and drop the
local store. The record was published on 2026-09-02, so that is what this does.

`resolve_channel_paths` now fetches the pair through `ensure_dataset`,
SHA-256-verified and cached under `~/.cache/luxar/`, and the zipped stores load
directly with nothing unpacked. The demo's declared requirements change with it:
`local_data` was `manual-file`, which is one of the two values that make
`luxar demo run-all` skip a demo outright, so clearing it is what actually makes
the demo reachable. The download figure was the 220 MB unzipped size and is now
the 136 MB that is really transferred, measured from the manifest.

Channels are paired to archives by NAME, not by position. `ensure_dataset`
returns the manifest's file order, which matches `CHANNELS` today, but a
reordering of either list would otherwise swap the two markers' colormaps and
opacities — and a swapped-channel render looks entirely plausible, so nothing
would catch it.

`HOSTED_DATASET_EXCEPTIONS` in `test_demo_fetch_path.py` is now empty. That
list is asserted for equality in both directions, so it could not have been
left populated: the test fails if a hosted dataset goes unreached, and equally
if an entry claims an exception that no longer applies.
