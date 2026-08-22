#### DESI DR1 opens on the observer even when the cached scene says otherwise

The DESI demo has framed its camera on the origin since #813 — the observer, the
one point every sightline in the survey radiates from — and the shipped
`desi_dr1_cosmic_web.luxar.zarr.zip` carries that camera. A user could still open
the demo swinging the local universe around a point out in the ELG shell, because
`main()` prefers an existing `datasets/demos/desi_galaxies.luxar.zarr` over the
shipped asset and never looks at what its camera says. A copy built before the
pivot moved keeps a bounding-box target — `(-35, -114, 1159)` Mpc on the one this
was found on, the ~1.2 Gpc down-`+z` centre the two asymmetric caps produce — and
keeps it forever.

`warn_if_scene_is_stale` could not catch this. It checks the streaming ladder and
the finest-level point count, and a scene framed on a bounding box passes both:
its geometry is entirely current, only its camera is not.

`ensure_origin_framing` now runs on every path that reuses or unpacks a scene
rather than building one, and re-pins the orbit target to the origin when it
finds one elsewhere. Only `target` is rewritten. Position, fov and the clipping
planes stay as the older build computed them — they are mutually consistent and
consistent with the cloud, and a bounding-box pivot is a complaint about the
orbit CENTRE, not the distance — which also keeps this a one-attribute metadata
touch rather than a rebuild. `--recompute` remains the way to get the current
framing in full, and the repair is idempotent, so a correctly framed scene is
read and left alone.

A scene carrying no authored camera at all is reported rather than repaired: the
viewer auto-frames on the bounding box in that case, which is the same wrong
result, but the distance the demo would need to author instead comes from the
95th percentile of the catalog's radial extent and the reuse path deliberately
never loads the catalog. Naming both remedies is honest; inventing a distance
would not be.
