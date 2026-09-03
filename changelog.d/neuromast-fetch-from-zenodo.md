#### Route the neuromast timelapse through its manifest entry

`demo_gsplats_4d_neuromast_2ch` was the last demo that did not reach its hosted
dataset through `ensure_dataset`. Its two fitted channels sit on the published
`cc-by` record, pinned by SHA-256 in `data_manifest.json`, so the demo now
downloads and verifies them like every other hosted one.

`resolve_channel_paths` tries the manifest first and pairs the returned archives
to channels by name rather than by position, so a manifest reordering cannot
silently swap the membranes and nuclei styles. Only the specific
`DatasetUnavailable` absence — a manifest that can build no download URL — falls
back to the `$LUXAR_NEUROMAST_DATA_DIR` store, which is what keeps the
acquisition machine and any hand-placed copy working; other fetch or manifest
faults remain errors. Nothing has to be provisioned by hand any more, so the
demo's `local_data` requirement is `None` and `run-all` and gallery generation
stop skipping it.

The 130 MB cache under `~/.cache/luxar/gsplats_4d_neuromast_2ch/` is declared, so
cache inventory and cleanup claim it correctly. The zips stay in that cache and
are expanded to a temporary directory when loaded, which is why a fetched path
ends `.gsplats.zarr.zip` while the local fallback names the unzipped store.

`HOSTED_DATASET_EXCEPTIONS` in `test_demo_fetch_path.py` is now empty because the
demo reaches the checksum-gated resolver. Focused tests pin name-based pairing,
manifest divergence, the fallback and its error path, that the record takes
precedence over a complete local copy, and the cache/download metadata.
