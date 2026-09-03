#### Let a demo opt out of the gallery orbit clip

Some subjects should not be rocked. `gsplats_lod_embryo_line` is a *row* of
embryos, so the ±20° turntable foreshortens the row and its mean luminance
swings **13×, twice per loop** — on the live site that read as violent flashing
rather than motion. `gsplats_2d_codex_pancreas` is planar and behaves the same
way. Measured at ±6° the swing is still 8.2×, so no amplitude setting rescues
either: the answer is not to orbit them at all, which is the convention already
recorded for `exotic_surfaces` ("near face-on only, NO orbit video").

`noOrbitVideo: true` in the gallery manifest now captures a **static
single-frame** tile and skips the `.webm` entirely.

The trap worth naming: the `.webp` *is* the animated loop. `build_gallery_data`
uses it as the tile's `still` and sets `has_media` from `video or still`, so
skipping only the `.webm` would have left the pulsing exactly as it was while
looking like a fix. The flag therefore encodes a genuinely static webp, and the
tile survives having no video.

When the flag is set, capture also removes any stale staged `.webm` before the
disk-based summary and publish sweep can mistake it for a fresh encode. The two
previously published objects were retired from the deploy tree and R2.
