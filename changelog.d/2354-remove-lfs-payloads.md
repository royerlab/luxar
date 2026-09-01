#### The demo payloads leave the repository

27 Git-LFS files — 409.9 MB — are gone from `demos/data/`, and the manifest's
dual checksum contract goes with them. Every clone has been paying that
bandwidth to carry a second copy of data the Zenodo records already serve.

With no file in the tree for a repo digest to describe, `hosted_sha256` becomes
`sha256` and the hosted-specific fields are dropped. The direction matters and is
the only safe one: `download_sha = hosted_sha or sha`, so keeping the repo digest
and dropping the hosted one would make every fresh download match nothing and be
quarantined. Nothing about how the fetch reads a pin changes.

Four payloads carried only a repo `sha256` and were previously thought to block
this. They do not: their hosted copies were measured byte-identical, so that one
digest already describes the record copy and needed no re-pin —
`3d_umap_coords_human.parquet`, `3d_umap_coords_mouse.parquet`,
`census_umap_1m.npz` and `zebrafish_4d.gsplats.zarr.zip`.

`dipc_genome/dipc_gm12878.npz` stays. Its bucket is `regenerate` and it is on no
record, so deleting it would leave a pin nothing can satisfy.

A cache populated from the old in-repo copies no longer matches the collapsed pin
and is re-downloaded once — 23 of 36 files on a developer machine. Recording each
outgoing repo digest in `superseded_sha256` would avoid that, at the cost of
spending the single honoured fallback slot on the older generation; that trade is
left open rather than assumed.
