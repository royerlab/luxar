#### Global "Zoom Sensitivity" knob for the mouse wheel

Some mice and drivers deliver far larger wheel deltas than others, so on those
machines a single notch flew through the scene and nothing in the viewer could
be set once to fix it: the Navigation popover's "Zoom Speed" is a per-scene
setting that bottoms out at 0.2. The Settings popover's Input folder now has a
**Zoom Sensitivity** slider (0.05 to 2, default 1) next to FOV Sensitivity. It
is a per-machine preference persisted with the other global settings, applies to
the very next wheel notch, and scales every wheel zoom — orbit and ortho dolly
as well as fly forward/back — while multiplying with, not replacing, the
per-scene zoom speed. Pointer-drag dolly and Shift+scroll roll are untouched:
a drag is pixels the user controls directly, and roll is not a zoom.
