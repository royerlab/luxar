#### Kidney toggle demo opens with all three channels on

`gsplats_6d_kidney_multichannel_toggles` encodes channel visibility as three
boolean dimensions, and a non-displayed discrete dimension starts at its range
minimum — which for these is `Off`. The demo therefore opened on an empty view
and only showed the tissue once the visitor found the dimension sliders and
flipped all three. The scene now authors `dimensions.current_step` so every
toggle starts at `On`; the entries for x/y/z are 0.0, exactly the viewer's own
default for camera-controlled dimensions, so only the toggles move.

The scene also opens auto-rotating. The three stains overlap heavily in
projection, and the parallax of a slow orbit is what separates them into one 3D
structure on first sight.
