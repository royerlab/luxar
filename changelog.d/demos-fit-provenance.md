#### Demos keep the provenance of the fit they ship

Four demos saved their fitted splats with `include_fitting_info=False`. That flag
reads like a request to tidy the metadata, but `split_fitting_info` leaves
`fitting_info` as `None` when it is set, so the `_FITTING_INFO_KEYS` whitelist
never runs — and because those same keys are excluded from `pipeline_info`, they
are dropped from the store *entirely* rather than relocated.

For these demos the consequence is not confined to a cache. Each one saves to the
file that is then packaged and shipped: `cryoem_virus.gsplats.zarr.zip`,
`ct_atlas.gsplats.zarr.zip` and `vh_head.gsplats.zarr.zip` are the same names the
demo writes. What those published datasets lost is the top-line record of the fit
that produced them — `fitter_name`, `n_splats`, `time_seconds` and the achieved
`psnr_db`, plus (for `tng_cosmic_web`, which culls) the `culled` /
`culling_method` / `n_original` / `n_removed` / `amplitude_retention` provenance
of what the cull removed. The reduction/topology stats in `pipeline/` survived, so
this is the fit's own report card going missing, not every trace of it — and once
`fit` stamps the source grid into `fitting/` too (#1614), the same flag would have
kept dropping that, so these demos could never have quoted a compression ratio no
matter what else changed upstream.

`cryoem_virus`, `ct_totalsegmentator`, `visible_human_head` and `tng_cosmic_web`
now keep it.

The NEXRAD sentinel deliberately does not. That call site saves a synthetic
single-splat placeholder for a radar frame with nothing above the dBZ floor —
a store the fitter never saw, and therefore one with no provenance to keep.
A new gate enumerates every suppressing call site by AST and holds it against
that single documented exemption, so the next one has to justify itself; the
exemption is keyed to a stated reason and an allowed call-site *count* rather
than a bare file name, so a file cleared for one sentinel save cannot quietly
become a file-wide opt-out.
