#### Bird plumage colour space, with ultraviolet as the fourth dimension

A new demo, `bird_plumage_colorspace`, plots every reading in
[BirdColorBase](https://github.com/BirdColorBase/home) — roughly half a million
spectrophotometer measurements of plumage across ~2,500 species — as a spike in
CIELAB: direction is hue, length is chroma, height is lightness, and each spike
is painted the colour it actually is.

The reason it is a Luxar demo rather than a figure is the axis a figure cannot
hold. The measurements run from 300 nm because birds are tetrachromats and
ultraviolet is part of a plumage patch's colour to the bird looking at it, but
the CIE observer is blind below ~380 nm, so a human colour-space plot has to
integrate it away. Here it becomes the scene's non-displayed fourth dimension:
UV chroma (R300–400 / R300–700, the standard avian-plumage metric) in ten
deciles, walked with the dimension slider against an `extend_to_all` ghost of
the whole corpus. Two patches sitting on the same CIELAB spike can be as far
apart in UV as red is from green, and scrubbing the axis is what shows it.

The colorimetry needs no data file and no new dependency: the CIE 1931 observer
comes from the Wyman–Sloan–Shirley multi-lobe fit already used by
`demo_galaxy_simulation`, the illuminant is a white-balanced 6500 K Planckian,
and the `.xlsx` sources are streamed with the standard library. Readings are
reduced block by block, so the 640 MB corpus never lands in memory whole.
