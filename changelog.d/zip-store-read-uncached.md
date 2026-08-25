#### Read a zipped scene (`.zarr.zip`) in the viewer, uncached

A `.luxar.zarr.zip` can now be opened directly with `?src=`, without unpacking it
first: each chunk is one member of the archive, so the viewer reads it in place over
HTTP range requests. This is deliberately narrow for now — a zipped store bypasses the
L1/L2 chunk cache, because `MultiLevelCachingStore` holds a base URL and builds its own
chunk URLs, which cannot address a member inside an archive. It reads uncached rather
than 404ing every chunk against `archive.zip/<key>`, and directory stores are unaffected.

Two failure modes that would otherwise be silent are now loud. A server that ignores
`Range` answers `200` with the whole file; the upstream reader accepts that as if it
were the requested slice, so the zip parser reads the wrong bytes at every offset and a
perfectly good archive looks corrupt. Every ranged read here requires `206` and
cross-checks `Content-Range`, and says to serve the dataset with `luxar serve` (which
does answer `206`). Likewise an archive built with `zip -r` wraps the store in a
directory, and since entry lookup is verbatim that would miss on every key and render an
empty scene; a single wrapping directory is now detected and stripped, and an archive
holding no store — or two — is refused by name rather than guessed at.
