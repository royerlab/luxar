#### Demos keep the provenance of the fit they ship

Four demos saved their fitted splats with `include_fitting_info=False`. That flag
reads like a request to tidy the metadata, but `split_fitting_info` leaves
`fitting_info` as `None` when it is set, so the `_FITTING_INFO_KEYS` whitelist
never runs — and because those same keys are excluded from `pipeline_info`, they
are dropped from the store *entirely* rather than relocated.

What goes missing is the canonical `fitting/` group: `fitter_name`, `n_splats`,
`time_seconds`, the achieved `psnr_db`, and the record of what the final cull
removed (`culled` / `culling_method` / `n_original` / `n_removed` /
`amplitude_retention`). The reduction/topology stats in `pipeline/` are
unaffected, so this is the fit's own report card going missing rather than every
trace of it — but `fitting/` is one of the three groups the loader harvests back
into `stats`, so nothing downstream can quote a figure that was never written.
Any source-grid stamp that lands in the same whitelist (#1614) would go the same
way, so these demos could not have reported a compression ratio either.

For three of the four — `cryoem_virus`, `ct_totalsegmentator` and
`visible_human_head` — the cost reaches past a scratch cache, because the file
they write is the name that gets packaged: `cryoem_virus.gsplats.zarr.zip`,
`ct_atlas.gsplats.zarr.zip`, `vh_head.gsplats.zarr.zip`. `tng_cosmic_web` is
local-compute and ships nothing, so there the loss stops at the user's cache.

This is about what a demo run writes from here on, not a rewrite of bytes already
in the repo: the three packaged `.zarr.zip` files were restamped out of band
(their store root is still `restamp_*`) and carry the fit numbers as a root
`lod_stats` attr rather than a `fitting/` group — two of the three also have a
`pipeline/` group, `vh_head` has neither — so they line up with the format the
loader reads only when they are next regenerated.

The NEXRAD sentinel deliberately keeps the flag. That call site saves a synthetic
single-splat placeholder for a radar frame with nothing above the dBZ floor —
a store the fitter never saw, and therefore one with no provenance to keep.
A new gate enumerates every suppressing call site in the demo package (the demos
plus the shared helpers they delegate to) by AST and holds it against that single
documented exemption, so the next one has to justify itself; the exemption is
keyed to a stated reason and an allowed call-site *count* rather than a bare file
name, so a file cleared for one sentinel save cannot quietly become a file-wide
opt-out.
