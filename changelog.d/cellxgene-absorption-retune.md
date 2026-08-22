#### Let the cellXGene census UMAP read through its own depth

The cellXGene census UMAP demo now uses `absorption=2.12` instead of `6.5`.
The previous setting screened cells behind the first dense shell and suppressed
most of each splat's own emission; the lower value preserves volumetric depth
without turning the atlas into an opaque surface.
