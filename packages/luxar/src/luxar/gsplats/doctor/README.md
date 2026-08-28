# `luxar.gsplats.doctor`

Diagnoses — and on request repairs — partition metadata in an existing
`.gsplats.zarr` dataset or `.luxar.zarr` scene.

The problems this exists for are the ones you cannot see. A dataset written by
an older Luxar loads fine and renders fine; it is just missing something a later
version learned to record, or is carrying metadata that went stale under an
edit. Nothing errors, so nothing tells you. Doctor is where that class of
condition gets named, costed, and fixed in place — no re-fitting.

## Using it

```bash
luxar gsplat doctor data.gsplats.zarr              # report (also prints `info`)
luxar gsplat doctor data.gsplats.zarr --fix        # repair in place
luxar gsplat doctor data.gsplats.zarr --full-provenance
luxar gsplat doctor scene.luxar.zarr --no-info     # every partition in a scene
luxar gsplat doctor data.gsplats.zarr --no-info --json report.json
```

Read-only unless you pass `--fix`; exits non-zero while a problem is still
standing, so it can gate a pipeline. A `.zip`/`.tar.gz` archive can be
DIAGNOSED (it is extracted to a temp directory and read from there) but not
repaired — there is nothing to write back to in place, so `--fix` needs an
uncompressed directory store, the same rule `gsplat annotate-quality` follows.
Diagnosing archives matters in practice: most bundled demo datasets ship as
`.zip`, and refusing them would put the common case out of reach of a sweep.
For a scene, findings name the store-relative partition path and `--fix`
re-stamps the root `content_hash` before consolidating metadata.

```python
from luxar.gsplats.doctor import diagnose_store

report = diagnose_store("scene.luxar.zarr")
if not report.healthy:
    for finding in report.unresolved:
        print(finding.severity, finding.path, finding.summary)
```

## What it currently checks

| check | condition | repair |
|---|---|---|
| `split-planes` | a `kind=partition` records no `bsp_tree` | recover the planes from the part boxes, when those are disjoint |
| `split-planes` | the stored `bsp_tree` does not separate the parts it names (stale after a transform, or written against a different part set) | rebuild from the part boxes, or remove the tree so ordering falls back honestly |
| `split-planes` | overlapping parts carry planes outside their measured overlap bands and per-axis tolerance floor | recover the band-bounded cuts; report a coordinate-frame scale only when repeated planes support the same factors |
| `split-planes` | the parts overlap, or lines/mesh parts are split by polyline/face centroid, so no stored plane separates them exactly | none when the parts overlap; when they are disjoint, `--fix` replaces the approximate planes with exact cuts recovered from the boxes — reported as a note |

Why it matters: the viewer orders partition parts back-to-front by traversing
those planes, which is exact for any camera pose including inside the volume.
Without them it sorts parts by content centroid — not a valid painter's order.
It flips discretely as the camera moves, so `normal` and `volumetric` blending
pop at the seams on every orbit. A *stale* tree is worse than a missing one: the
traversal still yields a plausible permutation, so the ordering is confidently
wrong instead of falling back.

Not every partition can be repaired. A uniform-tiled fit keeps each tile's
apodization halo, so its parts genuinely intersect and no exact ordering exists
to recover; that is reported, with the remedy (re-fit), and left alone.
Centroid-split lines and mesh can carry approximate planes even when their boxes
are disjoint; doctor can tighten those planes from the boxes under `--fix`. When
a partition carries its producer's approximate planes, the separation test
fails by construction. Doctor still checks that each plane lies between the two
sides' part-box centers. The largest measured interpenetration on each axis
supplies a tolerance floor, so sparse content cannot erase the halo scale. A
grosser violation is an error. Disjoint points and splats must still separate
exactly, even when stale cuts remain between the child centers. When one
constant multiplier per axis moves every
plane back into the measured overlap bands, doctor can rescale the tree; it names
a coordinate-frame scale only when at least two raw ratios on
each changed axis agree closely, or a single ratio agrees with a factor already
proven by repeated planes on another changed axis. Otherwise it describes the
repair as recovering the measured band cuts. If no safe factors exist, it removes
the misleading tree rather than guessing. A geometrically plausible approximate
tree is reported as a note and kept rather than condemned.

A repair is not always a cure — removing a misleading tree from parts that
cannot be ordered exactly leaves the lesser "no split planes" condition behind.
A `--fix` run therefore re-runs every check afterwards and reports what is still
standing (`DoctorReport.residual`); the exit code keys off that, not off the
fixes having been called.

## Adding a check

Write a function of the registry shape and append it to `ALL_CHECKS` in
`checks.py`. Nothing else in the doctor knows about any particular condition.

```python
def check_my_condition(root: zarr.Group) -> list[Finding]:
    ...
    return [Finding(check="my-condition", severity="error", path=..., summary=...,
                    detail=..., remedy=..., fix=lambda: ...)]
```

Three rules the existing check follows, and the next one should:

1. **Report what you cannot fix.** A finding with no `fix` and a concrete
   `remedy` beats silence.
2. **Never repair on a guess.** If the correct value is not recoverable from the
   store, say so and name what produces it.
3. **Do not finalize.** A `fix` closure writes attrs and nothing else — the
   runner re-stamps the root `content_hash` and re-consolidates metadata once
   for the whole run. That order matters: consolidated metadata *shadows* the
   per-node `.zattrs` a fix just wrote, so skipping it would leave every repair
   invisible to readers while looking applied on disk.

A check must not write during diagnosis; it attaches the closure and the runner
decides. Split `checks.py` into a package once the list outgrows one file.
