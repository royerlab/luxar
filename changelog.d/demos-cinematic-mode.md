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

The preset widens the field of view from 47° to 63° after the viewer performs its
first-load auto-frame, so auto-framed scenes open 0.71x smaller linearly. Demos
with authored camera positions preserve their deliberate composition instead:
all 21 pin `camera.fov`, including the five extent- or radius-derived poses that
would otherwise have opened looser. The post-process lens still applies unless a
scene overrides it. The two quantitative ortho demos disable distortion and
detector noise so their projection-derived scale bars and intensities remain
meaningful; the biodiversity globe disables both to preserve categorical hues.

A new lint (`test_demos_cinematic_mode.py`) keeps this true for demo 87: every
`create_scene` must pass a `viewer_config`, and every `ViewerConfig` built in the
demos package must set a literal `cinematic_mode=True`.

Existing generated stores under `datasets/demos/` predate the flag — the look
appears when a demo is re-run and its scene rewritten.
