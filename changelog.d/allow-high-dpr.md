#### High-DPI rendering is now opt-in

On a HiDPI display the viewer rendered at the full device pixel ratio from the very
first frame — 2.0 on a Retina Mac, which is four times the fragment work. For geometry
that is overwhelmingly soft-edged and emissive (points, gsplats, lines) those extra
pixels buy very little, and nothing capped them: the adaptive-resolution loop only ever
walked the ratio back down *reactively*, after the frame rate had already suffered.

A new **Allow High DPR** toggle in the Performance panel governs this, off by default
and authorable per scene as `viewer_config.allow_high_dpr`. While it is off, 1.0 is a
hard ceiling on everything interactive: it is the starting resolution, the ceiling the
adaptive loop may scale up to, the top of the Manual DPR slider, and the resolution the
idle resting frame settles at. Adaptive resolution keeps working exactly as before
*beneath* that ceiling, so a struggling scene still gets its reduction.

This is not a new mechanism so much as an existing one promoted to a default. The
manager already demoted its own ceiling to 1.0 once it had watched a scene fail to
sustain HiDPI; the setting applies that same conclusion up front instead of after the
evidence.

Two explicit requests still override the ceiling. The `?dpr=` URL parameter raises it
for the session, so `?dpr=2` renders at 2.0 and deterministic visual runs are
unaffected. Captures get a real knob: the recording panel's *Max Resolution* checkbox
becomes a **Capture DPR** slider that defaults to whatever is on screen — so a
screenshot now matches the viewport rather than silently exporting at twice its
resolution — and can be raised for a higher-resolution export.

Two fixes fell out of the work. Restoring from a scaled recording re-applied
`window.devicePixelRatio` directly, behind the scene manager's own override, which left
the renderer drawing at the wrong resolution until the next window resize snapped it
back. And the adaptive manager skipped re-applying the pixel ratio after a display
change whenever its current value happened to equal the old ceiling — on the assumption
that the renderer was tracking the ceiling by itself. It is not always: a reduction
applied at a higher ceiling leaves an explicit override behind, and the manager cannot
see it. Property-based fuzzing produced the sequence where the manager reported 3.0
while the renderer kept drawing at 1.0.

Note that a few appearance constants are expressed in *device* pixels rather than CSS
pixels — the minimum drawn size for points and lines, the line-join gates, and the
Gaussian-splat 2D dilation. At a 1.0 ceiling on a HiDPI display, sub-pixel dots are
drawn wider and dimmer, splats are softer, and very thin polylines can drop their
mitered joins. This is pre-existing behaviour (adaptive resolution already reached
these ratios under load); what changes is that it becomes the steady state. Making
those constants resolution-relative is tracked separately.
