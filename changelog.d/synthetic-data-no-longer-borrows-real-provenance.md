#### Synthetic stand-in data no longer carries someone else's provenance

Both blastocyst gsplat demos caught *every* exception from the remote load and
returned procedurally generated Gaussian blobs in its place. The scene then
went on, unchanged, to apply Blin et al.'s DOI, a CC BY 4.0 notice, the title
"Mouse Blastocyst, DAPI-Stained Nuclei" and a description of a Leica SP8
confocal acquisition — to fifteen random blobs. A dropped connection, a missing
`fsspec` or a decode error was enough; only a console line said anything, and
nothing in the published scene did.

Measured on the unmodified code with the source unreachable: it returns a
128³ volume and the scene publishes it under doi:10.1371/journal.pbio.3000388.

Acquisition failures now raise `DatasetUnavailable` and stop the demo rather
than fabricating a replacement. Procedural data is reachable only through an
explicit `--synthetic` flag, which switches the whole identity of the run
together: its own cache file, its own scene name, a title and caption that say
SYNTHETIC, a description stating nothing is measured, and **no citation** — a
DOI asserts provenance rather than decorating a scene. The separate cache
matters on its own: sharing one let a single offline run poison every later
online run.

The DAPI demo's on-screen caption also read "Light-sheet microscopy" while six
other places in the same file, and the sibling demo, correctly said confocal
(IDR records a Leica SP8 point-scanning confocal for this image). The caption
a viewer actually reads was the wrong one; it now matches.
