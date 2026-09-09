# `chunk-source/`

Implementations of the `ChunkSource` port declared in `../chunk-source.ts`.

A source answers one question for `MultiLevelCachingStore`: **give me the bytes for
this key.** Everything above it — L1, L2/OPFS, the segmented LRU, the prefetcher —
keys on whole objects and does not care where the bytes came from.

| module                 | what it reads                                                             |
| ---------------------- | ------------------------------------------------------------------------- |
| `http-chunk-source.ts` | a directory-backed zarr store: one retrying, gated HTTP request per chunk |
| `zip-chunk-source.ts`  | a zipped store through an injected archive byte reader                    |

## Why the port lives one directory up

`chunk-source.ts` sits in `src/cache/`, not in `src/data/` next to the stores it
abstracts, because `src/cache` precedes `src/data` in the layer order enforced by
`.dependency-cruiser.cjs`. A source therefore cannot import a concrete store to read
through it — the cache declares the port and the data layer supplies the adapter.

## Two shapes worth knowing before adding a source

**Return materialized bytes, not a `Response`.** The store used to hold a
`FetchResponseScope` and cancel unread bodies in a `finally`. A non-HTTP container has
no response to cancel, so that could not survive as a shared contract; response
lifetime is now each source's own business.

**`bytesOverWire` is separate from `data.byteLength`.** They are equal over plain
HTTP. They are not for a container whose transport compresses, and collapsing them
would make the bandwidth meter over-report by the compression ratio.

## The contract that is easy to break

`get()` **must not throw.** Failures come back as a `ChunkFetchOutcome`. A throw
escapes into `getResult` and is flattened to `NetworkError`, losing the source's
classification and diagnostic. Aborts are the sharp edge here: check the signal before
reading a body, and map a rejecting body read to `aborted` rather than letting it
propagate. The caching store rethrows both network and fatal failures so zarrita cannot
silently replace them with fill values; `fatal` preserves the more actionable diagnostic
for a container that is wholly unreadable.
