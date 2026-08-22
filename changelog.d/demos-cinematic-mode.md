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

The lens comes through whole, and framing is still unchanged. The preset expands
a 35 mm field of view (63° vertical, against the viewer's 47° default) as well as
the 35 mm barrel distortion, and `fov` / `fov_preset` are one unit — so pinning
either would have kept a 50 mm framing while still taking 35 mm distortion, which
is two different lenses in one image. No demo pins them. Instead the five demos
that state their own camera distance now compose it for 63°, because such a pose
specifies a distance rather than a framing: at a fixed distance the wider lens
scales the subject to 0.71x linear, or HALF its screen area. (An auto-framed
scene needs nothing — the bounding-sphere fit divides by `tan(fov / 2)` and moves
the camera in by itself.)

The new `demos/_cinematic_camera.py` holds that arithmetic once, so no demo
carries a bare 0.709: `CINEMATIC_FOV_DEG` for a demo that derives its distance
from the lens (quantum orbitals, whose `asin(R/D)` rule now reads 63° and which
therefore just moves closer), and `pull_in()` for one carrying an empirically
tuned pose (the cells3d isosurface, the ChromaTrace sequence, and the two interop
scans whose close pose exists to escape a sparse environment shell). `pull_in`
scales about the pose's TARGET, not the world origin, so the Mip-NeRF garden
camera keeps pointing at the table rather than swinging off it. Only the framing
carries over, not the image: a wider lens at a shorter distance foreshortens
more, so these five are worth a look on the next gallery pass.

A new lint (`test_demos_cinematic_mode.py`) keeps this true for demo 87: every
`create_scene` must pass a `viewer_config`, and every `ViewerConfig` built in the
demos package must set a literal `cinematic_mode=True`.

Existing generated stores under `datasets/demos/` predate the flag — the look
appears when a demo is re-run and its scene rewritten.
