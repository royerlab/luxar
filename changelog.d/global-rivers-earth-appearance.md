#### Rivers of Earth: a solid globe with depth-aware river glow

The `global_rivers_earth` demo drew its terrain shell near-transparent (`normal`
blending at opacity 0.05) and its rivers `additive`. Because `additive` ignores
depth entirely, every river on the *far* side of the planet was drawn over the
near side, so the globe read as a flat tangle of overlapping networks rather than
a sphere.

The terrain is now `opaque` at full opacity and the rivers `luminous`. Luminous
keeps additive's glow but respects depth occlusion, and opaque is the only mode
that leaves the viewer's sorted-transparent set and unconditionally writes depth
— so the terrain finally gives the rivers a surface to be hidden behind, and only
the visible hemisphere's network is drawn. The river display window is narrowed
to `0 – 0.38` as well (authored as `RIVER_DISPLAY_MAX` and inverted into the
shader gain), keeping the brighter trunks off the clip point now that they are
composited against a lit globe instead of near-black space.
