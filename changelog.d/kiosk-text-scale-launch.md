#### Overlay text scale, and machine settings for an exported kiosk

A scene can now scale the type of its overlays without moving them:
`viewer_config.text_scale` (or `?textScale=` on the URL, which wins) multiplies
every text overlay's font size, and authored HTML follows it by writing its
sizes as `calc(1.3vh * var(--luxar-text-scale, 1))`. Positions, widths and
anchors stay where they were authored. A title that belongs to the layout keeps
its size with `add_text(..., scale_text=False)`. An exported folder's `serve.py`
takes `--text-scale 0.8` for the same thing at launch.

`ViewerConfig.launch` (`LaunchConfig(renderer=, workers=, prefetch=)`) records
the machine settings a package built for known hardware should run with. The
viewer resolves these before it reads the scene, so `luxar export` bakes them
into the URL its `serve.py` opens, as the new `?workers=` and `?prefetch=`
parameters next to `?renderer=`. They win over the machine's saved Settings for
the session without rewriting them.

The protein universe tour gains `--kiosk` for its exhibit display: reading text
at 0.8x, a third of the screen at most for each story panel, short structure
captions placed just under the molecule, no Density Guard, and WebGPU with the
full worker pool and prefetch limit. Detector noise in both builds is now shot
noise only.
