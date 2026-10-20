#### EASI-FISH demo states its compression against the raw CZI

The EASI-FISH hypothalamus demo's caption and description now give the served
scene's on-disk size and how many times smaller it is than the 34.9 GB raw CZI
it was compiled from (≈315x on the reference build). The scene's size is
measured after the pack, as the sum of its file sizes; the CZI's is its pinned,
sha256-verified published size. Because the packed scene's size is only known
after `luxar optimize --pack`, the build packs into a scratch store with an
estimate, rewrites the two text attrs of the unpacked scene with the measured
size, packs again, and only then renames the result onto the served path, so
the stated figure is the size of the store it ends up in and an interrupted
build never leaves a scene stating a wrong one. The
ratio and size wording live in `luxar.demos._caption` (`compression_ratio`,
`format_ratio`, `format_bytes`, `compression_caption`, `compression_note`) for
other demos to reuse. The module docstring also explains why the mRNA channels
are fitted at xy/2 rather than full resolution.
