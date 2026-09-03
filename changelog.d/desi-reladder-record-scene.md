#### Re-ladder the DESI record scene instead of warning about it

`desi_galaxies` is the only demo whose Zenodo record carries a pre-built
*scene* — every other record holds raw fits or component arrays that the demos
structure at authoring time — so it is the only one that can ship a stale
STRUCTURE. It does. The record's finest level is a single unpartitioned node of
9,751,955 points, 2.4x the demo's own `SCENE_MAX_POINTS_PER_NODE` ceiling, and
the demo already knew: `warn_if_scene_is_stale` fires on both layers with
"larger nodes can silently lose their tail on a 4096-class GPU", and `main()`
printed it and rendered the scene anyway.

The fetch path now repairs instead of reporting. A new
`scene_exceeds_node_capacity` answers the same question the warning describes
(sharing its threshold rather than restating it), and when it is true
`restructure_scene` decodes the record's own points and colours and re-runs the
same `add_points(partition=…, substitutive_lod=…, additive_lod=…)` call that
`--recompute` uses. No record changes: the hosted bytes and their pinned digest
are untouched, and the re-ladder happens after download.

This had to land before the git-LFS payloads are removed. Resolution runs
cache → in-repo → hosted, so today the good in-repo scene wins and the record is
never reached — which is also why the live demo site is currently fine. Once the
payload is gone the record becomes the only source, and one of the warning's own
two remedies ("delete and re-run to unpack a current shipped asset") becomes
circular. Both the local default path and the next site rebuild would ship the
over-capacity scene.

Verified against the in-repo scene as ground truth rather than against
expectations. The re-laddered output reproduces it: coverage `[0.0, 0.5, 1.0]`,
finest level a partition of 4 parts, 9,751,955 points total, largest part
2,438,037 against the 4,000,000 ceiling, bit-identical scene bounds, and both
stale-checks silent. Colours survive exactly — they are a uint8 index into a
float32 row LUT, so the decode is lossless (measured max channel delta 0.0).
Positions are re-quantised onto the new per-part grids, which costs about what
one quantisation round already costs: per-axis sorted marginals agree to
≤0.16 Mpc, RMS ≤0.05 Mpc, on a cloud spanning ~14,000 Mpc.

The guard closes the hole that let this through. The existing record-reading
test skips unless the archive happens to be in the local cache, so it never ran
in CI — which is how a non-compliant scene reached a published record. The new
assertions build from scratch into a temp directory and check the AUTHORED
scene, with no record and no cache: a partition is present, every part is within
the ceiling, and the point and colour content survives a restructure. They also
never read `datasets/demos/`, because running the demo over an existing scene
reuses it without compiling, so a guard pointed there could pass while the
authoring code was broken.

Reading the scene goes through `luxar.encoding`'s decoder directly, as a
stopgap: there is no general "extract a node's data from a compiled scene" API
yet (#2482), and this should move onto it when it lands. Note the demo's own
`dequantize_positions` is NOT that tool — it inverts the demo's `.npz` scheme
(one global offset/scale), while a compiled scene uses per-chunk bounds, so
reaching for it yields plausible-looking wrong coordinates.

The capacity check fails CLOSED: a scene it cannot inspect is treated as needing
a re-ladder, not as fine. The asymmetry decides it — re-laddering an
already-compliant scene wastes minutes and changes nothing, while skipping a
non-compliant one renders a node that can lose its tail with no further signal.
This defect existed in the first place because a check that could not tell said
nothing and carried on.
