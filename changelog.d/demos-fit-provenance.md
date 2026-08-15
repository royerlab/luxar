#### Demos keep the provenance of the fit they ship

Four demos saved their fitted splats with `include_fitting_info=False`. That flag
reads like a request to tidy the metadata, but `split_fitting_info` leaves
`fitting_info` as `None` when it is set, so the `_FITTING_INFO_KEYS` whitelist
never runs — and because those same keys are excluded from `pipeline_info`, they
are dropped from the store *entirely* rather than relocated.

For these demos the consequence is not confined to a cache. Each one saves to the
file that is then packaged and shipped: `cryoem_virus.gsplats.zarr.zip`,
`ct_atlas.gsplats.zarr.zip` and `vh_head.gsplats.zarr.zip` are the same names the
demo writes. So the published datasets carried no record of the fit at all, and
in particular no record of the volume they are a representation *of* — which is
what makes a compression figure computable.

`cryoem_virus`, `ct_totalsegmentator`, `visible_human_head` and `tng_cosmic_web`
now keep it.

The NEXRAD sentinel deliberately does not. That call site saves a synthetic
single-splat placeholder for a radar frame with nothing above the dBZ floor —
a store the fitter never saw, and therefore one with no provenance to keep.
A new gate enumerates every suppressing call site by AST and holds it against
that single documented exemption, so the next one has to justify itself; the
exemption is keyed to a stated reason rather than a bare file name, which is how
such a list would otherwise decay into a general opt-out.
