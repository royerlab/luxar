#### The monitor's Depth Sort row counts "elements", not "splats"

The depth sorter serves all three geometry types, but the Data Loading
Monitor's Depth Sort row still labelled whatever it sorted as "splats" — a
points-only scene in normal blending showed "3.0K splats" with no splat in
sight. The sort rows (and the completion stream) now report a
geometry-neutral element count, with the per-type "pts"/"segs"/"splats"
tags unchanged on the geometry rows themselves; the row's tooltip also now
names all three geometry types.
