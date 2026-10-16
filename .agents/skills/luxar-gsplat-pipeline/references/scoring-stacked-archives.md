# Scoring a STACKED or TRANSFORMED archive

Loaded from the `luxar-gsplat-pipeline` skill ("Scoring a fit"). Read this before
running `luxar gsplat compare` on a time-stacked, multi-channel, or spatially
transformed archive.

Five separate wrong numbers were produced in one day by the same mistake:
**comparing two things held under different conventions.** None looked like an
error — each produced a plausible dB that a reader would simply believe. Check
all six failure modes before you trust a `compare`; the measured cases and full
controls are in [scoring-convention-failures.md](scoring-convention-failures.md).

**1. Score each timepoint before stacking, and retain its stamps.** A stacked
axis is the LAST centre column in a `.gsplats.zarr`; the source movie is usually
time-FIRST, so comparing both whole silently misaligns them. One archive measured
17.57 dB foreground whole versus 50.26 dB per timepoint. There is currently no
CLI route to recover that per-timepoint score from the stacked store:
`compare --timepoint` slices only the reference, and `gsplat slice` preserves the
archive rank. Newly built stacks may carry `fitting/part_provenance`; follow the
format spec's nested part → channel → timepoint levels to find per-timepoint
coordinates, and quote dB only when every `fit_reference.kind` is `acquisition`.
Missing, `preprocessed`, and `synthetic` references are deliberately not quotable.
Do not present either CLI command as an archive slicer.

**2. Keep the input archive's column order separate from the compiled scene's.**
In an `add_gsplats`/`add_points` authoring call, `dim_order[i]` names input data
column `i`; the compiler permutes those columns into scene `Dimensions` order
before storing them. `dim_order` is not persisted. Therefore the upstream
`.gsplats.zarr` follows the authoring argument, while the compiled scene node
follows `Dimensions`. Also check what the stacked column HOLDS: it is often a
physical value (minutes, µm) rather than an index, so `frame = round(value / step)`.

**3. Score before any spatial transform. "It was only a scale" is not a defence.**
Fitted centres are usually voxel indices; a shipped archive has often been scaled
to physical units and recentred. Comparing the transformed archive against an
untransformed reference cost **16 dB** even for a pure diagonal scale. Score at
fit time. If you must score later, invert the transform explicitly and prove the
result against an independent control; do not assume a scale is harmless.
`compare` maps a fit's own `fitting/voxel_size` stamp back to voxels, but a
`gsplat transform` scale or rotation drops that stamp. A global intensity edit is
different: `transform --scale-intensity` / `--normalize-intensity` record their
cumulative factor as `intensity_scale`; `compare` and `render` divide the render
by it and `cull --target` multiplies the target by it (the log prints
`intensity_scale=`), so a pure rescale scores like the original. An offset, clamp
or per-splat reweight drops the amplitude basis (`compare` warns): score before it.

**4. Fixing ONE convention mismatch and getting a different wrong number is not
progress.** One radar comparison successively exposed a mismatched grid, a −999
no-data fill that poisoned Otsu, and centres in km against a voxel-indexed
reference. Fixing the first two still left 6.70 dB foreground. **Carry a prior for
what the number should be, and keep digging while it is violated** — a
wrong-but-less-wrong number is the most expensive state to stop in.

**5. Never assign an axis ROLE from a data-dependent property.** A levelling
rotation derived in the plane of "the two widest axes" is correct only until the
tilt passes ~45°, at which point the second axis becomes the wider one, the roles
swap, and the angle comes out 90° off: a synthetic 70° tilt was measured as −20°
and "levelling" made the cloud LESS elongated (in-plane ratio 2.30 → 1.15). It
passed ruff, mypy, the complexity ratchet and the existing tests. Fix roles by
the convention the consumer uses, and exercise any such derivation against
synthetic inputs whose answer you already know.

**6. Never compare a preprocessed fit against its own preprocessed input.** It is
excellent by construction. Score against the original — see [denoising-before-a-fit.md](denoising-before-a-fit.md) for the subtler version, where the mask itself is the thing that lies.

Quality stamps are the way out of most of this, but **check whether they exist
before relying on them**: they are written per fit by `fit`, and coverage is
uneven downstream. Both `combine_as_new_dimension` and the `batch-fit` merge load
or concatenate inputs without carrying their per-fit quality to the merged root.
The component fits and current per-tile stores do carry the numbers, so inspect
and preserve them before merging. Older physical-coordinate fits may lack them
because metrics were skipped before #1668. Absent stamps mean re-measurement, and
re-measurement means every trap above. For post-fit command context, also see the
`luxar-gsplat-edit` skill.
