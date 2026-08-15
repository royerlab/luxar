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
because a 404 is not an inconclusive status, the dataset browser could not see a
format-3 store at all, `detect_store_encoding` answered "unclassifiable" so
`gsplat lod` stopped detecting its input's encoding, batch-fit's bytes/splat
measurement counted zero splats and fell back to its analytic estimate, the
overlay loader's directory-listing fallback offered `zarr.json` itself as an
overlay, and the demo re-encoder classified a `kind=partition` baseline as a
flat leaf — which selects the `flatten` path, so with `--apply` it would have
collapsed a partitioned dataset's structure and written the result over the
committed Git-LFS file.
`luxar._zarr_compat` now offers `read_array_meta`, `read_node_attrs`,
`is_consolidated` and `read_consolidated_attrs`, and nothing outside it names a
document. The shared shape of all eight is worth stating: **none of them raised**
— each returned an ordinary value that a caller had a reasonable interpretation
for, which is why they have to be found by grepping for the document names.

Two translations were needed that the plan had not anticipated. numcodecs
objects are format-2 currency and a format-3 array rejects them outright, so the
facade converts compressors and filters to their `zarr.codecs` equivalents —
keyed on the format of the group being written rather than the global default,
since adding an array to a legacy format-2 store while the default is 3 is
routine. The measured compressor policy survives intact: uint16 codes still get
`typesize=2` with byte shuffle, because `typesize` is left tunable and zarr
evolves it from the dtype. Keeping it intact needs one dependency change, so
`numcodecs>=0.16` is now a direct dependency rather than only zarr's transitive
one. zarr's format-3 `BloscCodec` hands numcodecs the *serialized byte buffer*
rather than the typed array, so the element width is lost unless it can forward
`typesize` explicitly — which it only does above that floor. Below it the byte
shuffle silently degrades to a no-op while the metadata still records
`typesize: 2, shuffle: shuffle`; measured on a 200k-point scene, that cost 12.5%
of the chunk bytes, more than the delta filter was introduced to win. With the
floor in place a format-3 store's chunks are byte-identical in size to a
format-2 one's, and the regression test asserts on the stored bytes rather than
on the recorded configuration, which is exactly what a lost shuffle leaves
looking correct. Note also that `compressor="auto"` is NOT the same compressor
in both formats (Blosc/lz4/5 versus zstd), which matters only for fixtures that
never named one.

The `luxar_delta_v1` filter now exists once per format and produces
**byte-identical chunks** either way — verified across 18 combinations of rows,
columns and code width, including partial last chunks. Format 2 keeps the
numcodecs filter; format 3 adds a `zarr.codecs` array-to-array codec under the
same name, registered through the `zarr.codecs` entry point so a vanilla
`zarr.open_group` in a process that never imports Luxar still decodes it. The
viewer registers both `numcodecs.luxar_delta_v1` and the bare `luxar_delta_v1`,
because zarrita looks the two formats up in different registry namespaces.

Consolidated metadata needed two accommodations that only exist at format 3.
zarr-python warns that it is a zarr-python extension rather than part of the v3
spec — advice about portability, not about the store's validity, since zarrita
implements it and the member is additive. Left alone that warning fired on
*every* save, and turned saving into a hard failure for anyone running with
`-W error` (`LuxarZarrCompiler.finalize` catches it and re-raises "Could not
finalize Zarr store"). It is now suppressed at the single facade call site;
dropping consolidation instead was never an option, because the viewer builds
its whole scene graph from that index and has no directory-walk fallback.

The second is subtler and is why in-place attribute edits must go through the
facade. Format 3 allows a consolidated index on ANY group, and bypassing the
root one does not bypass a nested one. Re-opening an already-consolidated store
with plain `zarr.open_group` yields nodes built from the root index, so
re-consolidating writes that stale in-memory tree back out as a nested index —
after which reads return pre-edit attributes although every document on disk is
correct, silently, as usual. Re-opening through the facade carries no index to
re-serialize and leaves exactly one, at the root: the format-2 invariant the
rest of the codebase already assumes.

One incidental hazard closed on the way: `validate_node_name` rejected the whole
dot-prefixed namespace, which covered every reserved key at format 2. Format 3's
`zarr.json` is not dot-prefixed, so a node could take that name and collide with
its own parent's metadata.

Sharding is deliberately not part of this. It needs `getRange` on the viewer's
`MultiLevelCachingStore`, which is cache-architecture work rather than a format
change, and keeping it out means this lands as a change you can verify by
diffing stores.
