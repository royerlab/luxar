#### Every demo now opens in cinematic mode

The gallery is what a first-time visitor sees, and it opened flat: `cinematic_mode`
was set by exactly one demo (the FlyLight MCFO neurons, which spelled out the
individual effects because at the time the flag alone did nothing). All 83
scene-building sites across the 86 demos now pass
`ViewerConfig(cinematic_mode=True)`, so each opens with ACES, a subtle wide
bloom, detector noise, a vignette and a 35 mm chromatic lens — the same look the
`C` key applies interactively.

It is one keyword per scene rather than a default in the write path because
`create_scene` serves library users too, and a rendering look enabled there would
restyle every scene anyone writes with Luxar. Forty-three scenes gained a
`viewer_config` they previously did not pass at all; the rest gained the flag
beside what they already pinned. Nothing else about their look changes: the
preset only fills fields a scene left unset, so every explicit `tone_mapping`,
`exposure` and bloom value in the demos still wins — including the interop
family's `"Neutral"` pin, which is deliberate for baked sRGB and is left alone.

The preset widens the field of view from 47° to 63°, and the viewer applies that
after its first-load auto-frame without re-framing, so auto-framed scenes open
0.71x smaller linearly — half the screen area. That ordering is viewer-side and
documented in `VIEWER_GUIDE.md`; a demo cannot correct it, and closing it means
applying the expanded fov before the fit. Filed as a follow-up.

Demos with an authored camera position keep their composition, in one of two
ways. Sixteen pin `camera.fov` (38°–50°, computed in five) — framing preserved,
but they then take the preset's 35 mm barrel distortion at a longer lens's
framing, and those pins all predate the cinematic look. The five poses derived
from a data extent or a fitted radius instead compose for 63° through the new
`demos/_cinematic_camera.py`, so the lens stays whole and the framing is
preserved exactly: `CINEMATIC_FOV_DEG` where the distance comes from the lens
(quantum orbitals' `asin(R/D)` rule, which therefore just moves closer,
79.97 → 60.65), and `pull_in()` where an empirically tuned pose is carried over
(the cells3d isosurface, the ChromaTrace sequence, and the two interop scans).
`pull_in` scales about the pose's target, not the world origin, so the Mip-NeRF
garden camera keeps pointing at the table. Only the framing carries over, not the
image — a wider lens at a shorter distance foreshortens more — so those five are
worth a look on the next gallery pass.

A new lint (`test_demos_cinematic_mode.py`) keeps this true for demo 87: every
`create_scene` must pass a `viewer_config`, and every `ViewerConfig` built in the
demos package must set a literal `cinematic_mode=True`.

Existing generated stores under `datasets/demos/` predate the flag — the look
appears when a demo is re-run and its scene rewritten.
