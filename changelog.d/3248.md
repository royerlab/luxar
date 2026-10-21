#### Budding-organoid demo states its compression against the source frames

The budding intestinal organoid demo's caption and description now give the
served scene's on-disk size and how many times smaller it is than the source
it was built from: the 167 shown VTI frames as extracted from the Zenodo
archive (T0244 standing in for the empty T0245) plus the lineage XML, 462 MB
in all, against an 88.6 MB scene (≈5.2x). The source size is pinned beside the
source digest and checked against the files whenever they are fetched, so a
run built from cached fits states it without the source on disk. The ratio is
modest because the published frames are already deconvolved, cropped, 8-bit
and zlib-compressed; the ~700 GB raw acquisition, available only on request,
is not the baseline. The scene is written into a scratch store, its two text
attrs are restated with the measured size (restamping the content hash) until
the figure stops changing, and only then is it renamed onto the served path.
The scratch, measure, restate and rename steps moved into
`luxar.demos._scene_size`, which the EASI-FISH demo now uses too.
