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

Keep the amplitude modest. Screen area goes as the inverse square of distance,
so a swing of `A` moves the subject's projected area by `(1 + A)⁴` — 1.75× at
the default 15%, but 2.9× at 30% and 5× at 50%. The LOD ladder steps on
halvings of screen area, so a large amplitude walks up and down it every
cycle, re-fetching chunks each time on a hosted scene where the cost is
requests. The 15% default keeps the whole oscillation inside one level.

Scenes can author it from Python as `ViewerConfig(auto_dolly=True,
auto_dolly_amplitude_percent=20, auto_dolly_period=8)`, and the choice
persists per scene with the rest of the rendering settings. Turntable
recording bakes it in frame-indexed, at a whole number of cycles per turn, so
an exported clip breathes like its preview and still loops seamlessly.
