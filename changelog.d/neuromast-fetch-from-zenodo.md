#### Route the neuromast timelapse through its manifest entry

`demo_gsplats_4d_neuromast_2ch` was the last demo that did not attempt its
hosted dataset through `ensure_dataset`. Its two fitted channels are already on
the `cc-by` Zenodo record and pinned by SHA-256 in `data_manifest.json`, but the
packaged manifest still marks that record unpublished and therefore builds no
download URL.

`resolve_channel_paths` now tries the manifest path first and pairs the returned
archives to channels by name rather than position. A manifest reordering cannot
silently swap the membranes and nuclei styles. Until the manifest publication
switch lands, the specific `DatasetUnavailable` absence falls back to the
existing `$LUXAR_NEUROMAST_DATA_DIR` store; other fetch or manifest faults remain
errors. The demo consequently stays classified as `manual-file` for `run-all`
and gallery generation in the interim.

The future 136 MB cache is declared under
`~/.cache/luxar/gsplats_4d_neuromast_2ch/`, so cache inventory and cleanup claim
it correctly. The zip files remain in that cache and are expanded to a temporary
directory when loaded.

`HOSTED_DATASET_EXCEPTIONS` in `test_demo_fetch_path.py` is now empty because the
demo reaches the checksum-gated resolver. Focused tests pin name-based pairing,
manifest divergence, the unpublished-record fallback, its error path, and the
cache/manual-data metadata.
