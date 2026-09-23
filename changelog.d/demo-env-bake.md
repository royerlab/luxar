#### Demos bake their scene environment

A scene whose environment source is a live capture had it taken by the viewer
on load: six renders of the whole scene into a cube map, repeated whenever the
geometry, slice or appearance changed. What a bubble reflected therefore
depended on how much data had streamed in when the capture fired, so it varied
between machines and between runs, and every visitor paid for it.

The two demos with `material="physical"` bubbles now bake that environment into
their store as they build, through `luxar.demos.bake_scene_environment`. The
viewer prefers the stored faces over any capture, so the reflections are
identical everywhere and there is nothing to recompute at load. The map is
excluded from the scene `content_hash`, so attaching one leaves warm viewer
caches valid, and it costs about 136 KB.

The step is best-effort and self-gating: it uses the scene's own probe and
resolution, skips a scene that cannot gain from it or already carries a map,
and a checkout without a built viewer simply keeps the live capture.
