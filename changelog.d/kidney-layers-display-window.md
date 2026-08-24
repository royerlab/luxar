#### Kidney multichannel layers demo opens on a window you can see

The three-channel kidney demo rendered too dark to read: it left every channel
on the writer's own robust display window, `[min, p99.9]` of that channel's
amplitudes. That default is the right one for a single additive layer, but on
this scene it puts the MEDIAN splat at ~15% of the window — so with three
volumetric channels composited together, most of the tissue lands in the bottom
fifth of the LUT and the nuclei channel is barely present at all.

Each channel now authors its own display window at 40% of that range (roughly
its 92nd percentile), derived per channel at build time so a refit moves the
window with the data instead of stranding it on stale numbers. Measured against
the alternatives on this dataset: the old full window is the dim render being
replaced, and pulling the max down to 25% washes the red/blue overlap out to
magenta and merges structure — 40% lifts the mid-tones while only the brightest
cores begin to clip. The Layers panel (press **L**) still moves the range live;
this only decides where it opens.

The lever is `intensity`/`offset` on the colormapped node, which the viewer
consumes as the scalar display WINDOW rather than a post-LUT gain. Scaling the
amplitudes cannot do this: the writer derives its window FROM the amplitudes, so
a global rescale cancels out on screen and the render does not change.
