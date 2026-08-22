#### The IllustrisTNG cosmic web renders volumetric, in turbo

The `cosmic_web` layer shipped as an additive, magma-mapped, fully-opaque leaf.
Additive is the wrong compositing mode for this dataset: it is a genuine 3D
density field about 300 Mpc deep, so every ray sums straight through the box and
the nearest clusters stop reading as nearer — the one depth cue a cosmic web
most needs. The layer is now `volumetric`.

Absorption carries that cue, and the default kappa of 1.0 is far too thin at
this scale: the far wall of the box still shows through the densest halos. It is
now 10.0, which puts the extinction length well inside the box so foreground
structure occludes background structure. Opacity drops from 1.0 to 0.66 as a
consequence rather than a separate taste call — volumetric compositing
accumulates alpha along the ray, so splats that were tuned to be fully opaque
under additive now over-fill, and 0.66 keeps filament interiors translucent
enough to see the structure behind them.

The colormap moves from magma to turbo. The field is read as density, and
turbo's full hue sweep separates void / sheet / filament / halo, where a
single-hue luminance ramp saturates the filaments and the halos they feed to the
same white.

Nothing pins the display window. The panel reading `0 – 0.084` is the writer's
own `amplitude_data_range` — `[min, p99.9]` of the fit, currently
`[0.0004, 0.0847]` — not a hand-set value, and `gamma` was already at its 1.0
default. Writing either into the demo would freeze a data-dependent number that
a refit legitimately moves, so both are left to the writer; `SCENE_INTENSITY`
stays at 0.15 for the same reason, since it scales the amplitudes the window is
derived from and a change there would move both together and re-tone nothing.

The four tuned values are pinned by a test that reads them off the
`add_gsplats_from_data` call, because three of them only make sense as a set: a
later edit reverting the blend mode alone would silently strip `absorption` of
any meaning while leaving the reduced opacity looking like a deliberate choice.
