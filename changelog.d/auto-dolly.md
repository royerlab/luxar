#### The camera can breathe as well as spin

The Navigation popover gains **Auto Dolly**, the turntable's radial sibling:
instead of going around the subject the camera moves toward and away from it
on a sine, with its own amplitude and period. Combined with Auto Rotate it is
the slow approach-and-retreat hero shot; on its own the parallax reads depth
in a way a still cannot. Off by default, so nothing existing changes.

The amplitude is a percent of the viewing distance rather than a length, which
makes it the sinusoidal mousewheel it is described as: the wheel scales
distance by a constant factor per click, so equal swings in and out are equal
in ratio, and one setting means the same thing on a micron-scale volume and a
galaxy catalogue. Zoom stays yours while it runs — both the wheel and the
dolly only ever multiply the distance, so a scroll moves the centre the camera
is breathing around instead of fighting the animation, exactly as a drag still
rotates a running turntable. Because it is gated on zoom rather than rotation
it also works in ortho mode, where it breathes the orthographic zoom and the
turntable is inert.

The amplitude slider runs to 95% — nearly halving and doubling the viewing
distance each cycle — but the default is a modest 15%, because the cost is
steeply non-linear. Screen area goes as the inverse square of distance, so a
swing of `A` moves projected area by `(1 + A)⁴`, and the LOD ladder answers by
loading finer levels at the near extreme. Measured over one cycle on a
100-group demo: 15% keeps 118k elements resident, 50% keeps 526k, 95% keeps
2.29M — a 19× resident set for a 6× bigger swing. A local warm cache absorbs
that (144 → 129 fps); a hosted scene, where cost is requests, does not. Nothing
about a large amplitude is unsafe — the distance clamps sit orders of magnitude
away — so it is a budget decision, not a safety one.

Scenes can author it from Python as `ViewerConfig(auto_dolly=True,
auto_dolly_amplitude_percent=20, auto_dolly_period=8)`, and the choice
persists per scene with the rest of the rendering settings. Turntable
recording bakes it in frame-indexed, at a whole number of cycles per turn, so
an exported clip breathes like its preview and still loops seamlessly.
