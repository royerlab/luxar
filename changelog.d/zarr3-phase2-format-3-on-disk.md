#### Write zarr format 3 on disk (Phase 2)

Luxar has run on zarr-python 3 since the Phase 1 migration while still writing
zarr format 2. It now writes **format 3** by default. Reading is unchanged and
version-agnostic, so existing `.luxar.zarr` and `.gsplats.zarr` stores stay
format 2 and are never rewritten — a tree holding both formats is the expected
steady state rather than a migration window, and that is what makes this cheap:
no dataset regeneration, no Git-LFS churn, no republishing.

`LUXAR_ZARR_FORMAT=2` still produces format 2 for a tool that cannot read 3.
That is an environment variable rather than a CLI flag because the process doing
the writing is frequently not the one you invoked — `batch-fit run` spawns
per-GPU workers, `batch-fit submit` writes an sbatch script whose array tasks run
hours later on other nodes, and `fit -j N` forks tile workers. An exported
variable reaches all of them; a flag parsed by one command would have to be
threaded through every spawn site.

The consequence worth internalising is that **every reader is now bi-format**.
The two formats disagree about where metadata lives: format 2 writes `.zgroup` /
`.zattrs` / `.zarray` documents plus a consolidated `.zmetadata`, while format 3
writes one `zarr.json` per node with attributes nested under `attributes` and
consolidated metadata embedded in the root document. Anything that inspected a
store by naming those files was silently wrong for the other format, and the
failures were all quiet ones rather than errors: `batch-fit validate` passed a
tile whose attributes had been stripped, the viewer's cache-validation probe
returned no token and fell back to a null TTL (serving stale OPFS data
indefinitely), the scene-identity watchdog reported *every* poll as a change
because a 404 is not an inconclusive status, and the dataset browser could not
see a format-3 store at all. `luxar._zarr_compat` now offers `read_array_meta`,
`read_node_attrs` and `is_consolidated`, and nothing outside it names a document.

Two translations were needed that the plan had not anticipated. numcodecs
objects are format-2 currency and a format-3 array rejects them outright, so the
facade converts compressors and filters to their `zarr.codecs` equivalents —
keyed on the format of the group being written rather than the global default,
since adding an array to a legacy format-2 store while the default is 3 is
routine. The measured compressor policy survives intact: uint16 codes still get
`typesize=2` with byte shuffle, because `typesize` is left tunable and zarr
evolves it from the dtype. Note that `compressor="auto"` is NOT the same
compressor in both formats (Blosc/lz4/5 versus zstd), which matters only for
fixtures that never named one.

The `luxar_delta_v1` filter now exists once per format and produces
**byte-identical chunks** either way — verified across 18 combinations of rows,
columns and code width, including partial last chunks. Format 2 keeps the
numcodecs filter; format 3 adds a `zarr.codecs` array-to-array codec under the
same name, registered through the `zarr.codecs` entry point so a vanilla
`zarr.open_group` in a process that never imports Luxar still decodes it. The
viewer registers both `numcodecs.luxar_delta_v1` and the bare `luxar_delta_v1`,
because zarrita looks the two formats up in different registry namespaces.

One incidental hazard closed on the way: `validate_node_name` rejected the whole
dot-prefixed namespace, which covered every reserved key at format 2. Format 3's
`zarr.json` is not dot-prefixed, so a node could take that name and collide with
its own parent's metadata.

Sharding is deliberately not part of this. It needs `getRange` on the viewer's
`MultiLevelCachingStore`, which is cache-architecture work rather than a format
change, and keeping it out means this lands as a change you can verify by
diffing stores.
