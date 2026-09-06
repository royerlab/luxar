#### Global "Zoom Sensitivity" knob for the mouse wheel

Some mice and drivers deliver far larger effective wheel steps than others, so
on those machines a single notch flew through the scene and the only existing
adjustment was the Navigation popover's per-scene "Zoom Speed", which bottoms
out at 0.2. (Part of that spread was the browser rather than the mouse; line-
and page-mode wheel deltas are now normalized to pixel equivalents upstream of
this knob, so the slider is genuinely a per-mouse preference.) The Settings
popover's Input folder now has a **Zoom Sensitivity** slider (0.05 to 2,
default 1) next to FOV Sensitivity. It is a per-machine preference persisted
with the other global settings, applies to the very next wheel notch, and
scales every wheel zoom — orbit and ortho dolly as well as fly forward/back —
while multiplying with, not replacing, the per-scene zoom speed. Pointer-drag
dolly and Shift+scroll roll are untouched: a drag is pixels the user controls
directly, and roll is not a zoom.
