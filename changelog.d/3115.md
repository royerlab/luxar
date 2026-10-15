#### Viewer polish: overlays clear the rail, no console 404s, the loop rests when it should

Scene and demo text overlays anchored at the left edge used to paint under the
control rail's icons on desktop. They now take the same `--luxar-rail-gutter` as
the docked panels, as a floor (`max(authored x, gutter)`), so an overlay already
right of the rail is untouched, and an explicit-width overlay is capped to the
room left on screen; right- and centre-anchored overlays never move.

A normal scene load logged two to five `GET … 404` lines in the console, each a
speculative read of something the store's consolidated index had already
answered. `openStore` now places a metadata gate under zarrita's consolidated
wrapper: once the store's format is known, the other format's documents (and a
format-2 `.zattrs` the index settles) are answered without a request. A scene
without overlays no longer probes `overlays/`, and the `chunk_packs/` and
`environment/` sidecars are checked against an index fetched from the network
that load — on a warm load the load-time root-document fetch, which for a
format-3 store is the consolidated index — so the stale-cache guarantees both
sidecars rely on are unchanged. When a probe is still needed (format 2, a
presigned or zipped source) it asks for one document of the known format. The
identity watchdog's first poll asks for the document the load was served from
even when the server sends no `ETag`.

The window `focus` handler no longer restarts rendering while the document is
hidden. While the data server cannot be reached at all (a refused or dropped
connection; an HTTP status such as an expired presigned credential, a rate
limit or a 5xx shows the identity banner but does not count), the render loop
stops sustaining itself: cinematic noise, the auto-rotate turntable and
auto-dolly, dimension playback and loader polls pause, while user input still
renders the loaded scene. Recording is exempt, so a video or offline turntable
never films a frozen canvas. The loop resumes when the server answers again.
