# `luxar.gsplats.doctor`

Diagnoses — and on request repairs — an existing `.gsplats.zarr` store.

The problems this exists for are the ones you cannot see. A dataset written by
an older Luxar loads fine and renders fine; it is just missing something a later
version learned to record, or is carrying metadata that went stale under an
edit. Nothing errors, so nothing tells you. Doctor is where that class of
condition gets named, costed, and fixed in place — no re-fitting.

## Using it

```bash
luxar gsplat doctor data.gsplats.zarr              # report (also prints `info`)
luxar gsplat doctor data.gsplats.zarr --fix        # repair in place
luxar gsplat doctor data.gsplats.zarr --no-info --json report.json
```

Read-only unless you pass `--fix`; exits non-zero while a problem is still
standing, so it can gate a pipeline. A `.zip`/`.tar.gz` archive can be
DIAGNOSED (it is extracted to a temp directory and read from there) but not
repaired — there is nothing to write back to in place, so `--fix` needs an
uncompressed directory store, the same rule `gsplat annotate-quality` follows.
Diagnosing archives matters in practice: most bundled demo datasets ship as
`.zip`, and refusing them would put the common case out of reach of a sweep.

```python
from luxar.gsplats.doctor import diagnose_store

report = diagnose_store("data.gsplats.zarr")
if not report.healthy:
    for finding in report.unresolved:
        print(finding.severity, finding.path, finding.summary)
```

## What it currently checks

| check | condition | repair |
|---|---|---|
| `split-planes` | a `kind=partition` records no `bsp_tree` | recover the planes from the part boxes, when those are disjoint |
| `split-planes` | the stored `bsp_tree` does not separate the parts it names (stale after a transform, or written against a different part set) | rebuild from the part boxes, or remove the tree so ordering falls back honestly |
| `split-planes` | overlapping parts carry planes in a different coordinate frame | recover one scale per axis from the parts and rescale the tree, but only when the same factors explain every plane |
| `split-planes` | the parts OVERLAP, so no tree separates them and the stored one cannot be checked exactly | none — reported as a note; this is what a uniform-tiled fit's approximate planes look like, and deleting them would be a downgrade |

Why it matters: the viewer orders partition parts back-to-front by traversing
those planes, which is exact for any camera pose including inside the volume.
Without them it sorts parts by content centroid — not a valid painter's order.
It flips discretely as the camera moves, so `normal` and `volumetric` blending
pop at the seams on every orbit. A *stale* tree is worse than a missing one: the
traversal still yields a plausible permutation, so the ordering is confidently
wrong instead of falling back.

Not every partition can be repaired. A uniform-tiled fit keeps each tile's
apodization halo, so its parts genuinely intersect and no exact ordering exists
to recover; that is reported, with the remedy (re-fit), and left alone. When
such a partition already carries its producer's approximate planes, the
separation test fails by construction. Doctor still checks that each plane lies
between the two sides' part-box centers, allowing the measured overlap width as
tolerance. A grosser violation is an error. When one constant multiplier per
axis moves every plane back into the measured overlap bands, doctor names those
factors and can rescale the tree; otherwise it removes the misleading tree
rather than guessing. A geometrically plausible approximate tree is reported as
a note and kept rather than condemned.

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
