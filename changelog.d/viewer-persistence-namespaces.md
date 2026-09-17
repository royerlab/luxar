#### Everything the viewer leaves in a browser, or exposes to a host page, is namespaced

Dataset caches in OPFS now live under a `luxar/` directory instead of sitting
as bare `zarr-cache-…` entries at the origin root, so a host page's own OPFS
content and Luxar's never share a level; `listDatasets` tolerates a cold
origin and `clearAll` removes only from the namespace. Per-scene rendering
settings are stored in a versioned envelope (`RENDERING_SETTINGS_VERSION`),
matching the global settings that already carried a version: a foreign or
missing version resets to defaults with a log line rather than being
half-applied. `StorageKeys` is now genuinely the complete registry — the
control-rail keys that bypassed it are in, and a source-scan test rejects any
`localStorage` call that does not go through it. The one unprefixed window
event (`open-dataset-browser`) and the two bare DOM ids in the dataset
browser carry the `luxar-` prefix, with a scan test on every `id=` literal
under `src/ui`. Directory navigation normalises the folder URL once, so a
static host's `<dir>/.luxar-index.json` manifest (and the zarr probes) are
requested at the right path; the manifest format is documented in the viewer
guide.

Pre-release, none of this carries a read-old-key fallback: a development
profile will see its dismissed hint and per-scene tweaks reset once.
