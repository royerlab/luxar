#### EASI-FISH spatial transcriptomics demo: from cell types to single mRNA molecules

A new demo, `gsplats_3d_easifish_hypothalamus`, shows Janelia's public EASI-FISH
example data (Wang et al., *Cell* 2021; Tillberg and Sternson labs): an expanded
slab of mouse lateral hypothalamus imaged in four channels, DAPI nuclei plus the
mRNA of *Slc17a6*, *Gpr83* and *Bdnf*. At the opening view the cells expressing
each gene stand out in that gene's colour against grey nuclei. Zoom in on one and
the colour resolves into individual transcripts, which is the scale change a
partitioned scene with per-part level of detail is built for.

The demo builds from the original source. It downloads the 34.9 GB CZI from
figshare (DOI 10.25378/janelia.13622819, CC BY 4.0, sha256-pinned and resumable),
stitches its two tiles from the recorded stage positions refined by normalised
cross-correlation, and fits each channel with `gsplat batch-fit run`: content
tiling for the punctate mRNA channels, uniform tiles for DAPI, and a per-part
`levels` ladder from the streaming merge. Every leaf is then re-laddered with
equal-energy rungs, so each part's first rung is the few splats carrying the
most light, and the fits are scaled by the tissue's 2x linear expansion so the
scene reads in biological micrometres. The mRNA channels pass `--min-extent 0`
so no punctum is pruned, each channel subtracts a background floor resolved on
its own array, and the scene is packed with `luxar optimize --pack`. Brightness
and colormap windows are set per layer, because four additive channels at the
default exposure sum to a beige wash, and the cinematic lens is turned off so
puncta stay round at the frame edges.

No fitted copy is hosted yet, so the first run needs a CUDA GPU, at least 32 GB
of RAM, about 40 GB of disk and a few hours; `--czi PATH` reuses a copy of the
file you already have. Every step is cached, and the scene is rebuilt only when
the demo or a fit changed. Reading the CZI needs `czifile`, now part of the
`demos` extra. The fit quality assumes the pedestal-seeding fix in #3226.
