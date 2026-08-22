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

The preset widens the field of view from 47° to 63°. The viewer resolves that FOV
before automatic framing, so auto-framed scenes keep the fitted subject
occupancy intended for the lens. A returning visitor's stored FOV remains
authoritative by design.

All demos with an authored camera position compose for 63° through the new
`demos/_cinematic_camera.py`, so the lens stays whole (#1862):
`CINEMATIC_FOV_DEG` where the distance comes from the lens, and
`framing_scale()` / `pull_in()` for poses carried over from an authored 38°–50°
FOV. Most preserve their framing exactly; the biodiversity globe preserves its
silhouette, while the forest and embryo-line poses preserve camera clearance.
`pull_in` scales about the pose's target, not the world origin, so the Mip-NeRF
garden camera keeps pointing at the table. For those pulled-in poses only the
framing carries over, not the image — a wider lens at a shorter distance
foreshortens more — so they are worth a look on the next gallery pass.

A new lint (`test_demos_cinematic_mode.py`) keeps this true for demo 87: every
`create_scene` must pass a non-`None` `viewer_config`, every `ViewerConfig` built
in the demos package must set a literal `cinematic_mode=True`, every authored
camera must leave its FOV unpinned and demonstrate composition for 63°, and the
scientific-fidelity overrides must remain explicit.

Four fidelity-sensitive demos deliberately opt parts of the preset back out:
the CMU-1 pathology and CODEX pancreas ortho demos disable bloom, vignette, lens
distortion and detector noise, while the biodiversity globe and nD transform
bench disable lens distortion and detector noise to preserve their colour
encodings.

Existing generated stores under `datasets/demos/` predate the flag — the look
appears when a demo is re-run and its scene rewritten.
