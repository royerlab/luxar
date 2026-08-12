#### The monitor's Depth Sort row counts "elements", not "splats"

The depth sorter serves all four geometry types, but the Data Loading
Monitor's Depth Sort row still labelled whatever it sorted as "splats" — a
points-only scene in normal blending showed "3.0K splats" with no splat in
sight. The sort rows (and the completion stream) now report a
geometry-neutral element count (points, line segments, splats or mesh
triangles), with the per-type "pts"/"segs"/"splats" tags unchanged on the
geometry rows themselves; the row's tooltip also now names all four
geometry types.
