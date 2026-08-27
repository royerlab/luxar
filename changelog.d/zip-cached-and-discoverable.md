#### Cache zipped scenes, and let the dataset browser find them

A `.luxar.zarr.zip` now reads through the full L1/L2 chunk cache instead of bypassing it,
and appears in `luxar serve`'s listing and the dataset browser as the dataset it is. Until
now a zipped scene loaded only if you hand-typed its URL, and re-read the archive from
scratch on every visit.

Caching earns more on an archive than on a directory store, not less. Reading one costs
about two HTTP requests per member — the zip format puts a local file header immediately
before each member's data, and it has to be read first — and a repeat read cannot fall back
to the browser's own HTTP cache the way a repeat GET of a per-chunk URL can, because every
member read is a `Range` request against a single URL. The chunk cache is what makes those
repeats free.

Because an archive keeps its root metadata inside itself, it cannot answer "is this still
the scene I cached?" by re-fetching a document. A zipped store is instead identified by a
`HEAD` on the archive — its `ETag`, or failing that its modification time and size — which
covers the whole store at once rather than one document, and shows in the cache monitor as
"Archive ETag". A zipped store and its unzipped twin keep separate caches: they hold the
same decoded chunks but different key namespaces, and sharing a bucket would let one serve
the other's members undetected.

Image overlays and local drag-and-drop of an archive remain unsupported; both need a path
that does not exist yet (reading an overlay through the store rather than by URL, and
loading a dataset from a file rather than a URL).

Reading an archive also stops re-downloading part of its own index. The zip reader is
given a fixed 65,557-byte window to locate the end-of-central-directory record, and then
asks for the central directory itself — which sits immediately before that window and is
mostly inside it. Those bytes were being transferred twice on every load. The reader now
retains what the directory read touched and splices, fetching only the part it is missing:
measured 152 kB → 88 kB per visit at ~1000 members. The identity probe and the archive
length are also one request now instead of two, since they were asking the same server the
same question.
