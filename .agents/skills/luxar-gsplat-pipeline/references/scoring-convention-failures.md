# Scoring convention failures: measured cases

These measurements support the rules in "Scoring a STACKED or TRANSFORMED
archive". Prediction and reference must share axis order, coordinate units,
spatial transform, grid, and intensity basis.

## Whole-stack axis mismatch

The stacked axis was the last splat centre column while the source movie was
time-first. Comparing both whole produced a plausible but false result:

| comparison | PSNR | foreground PSNR | SSIM |
| --- | ---: | ---: | ---: |
| stacked 4D, compared whole | 28.60 | 17.57 | 0.651 |
| component fits, per timepoint | **50.92** | **50.26** | **0.992** |

This is a 33 dB foreground understatement. Preserve the component-fit stamps
before stacking. `compare --timepoint` selects a timepoint from the reference
volume only; it does not slice the splat archive. `gsplat slice` filters splats
by a coordinate range but keeps the archive rank.

## Spatial transform mismatch

A drosophila fit was scored before and after a pure diagonal scale while the
reference remained on its voxel grid:

| comparison | PSNR | foreground PSNR | SSIM |
| --- | ---: | ---: | ---: |
| pre-transform fit vs voxel-grid reference | 44.94 | 40.94 | 0.968 |
| post-transform fit vs same reference | 28.78 | 24.33 | 0.541 |
| hosted post-transform archive vs same reference | 28.77 | 24.22 | 0.536 |

The final two rows agree within 0.01 dB, which is the control showing that the
16 dB collapse comes from the convention mismatch rather than the fit. Rotated
and centred data can be harder to recover; one flylight case reached only
correlation 0.17/0.07. An exact inverse has not been established as a general
recovery path.

## Compound radar mismatch

One radar comparison contained three independent faults:

1. The archive and reference came from two different spatial grids because the
   reference was selected with `ls | head -1`.
2. Raw dBZ used −999 as no-data. Global PSNR was 3.5 dB and Otsu selected
   −996.9, classifying roughly half the empty sky as foreground.
3. After matching the grid and applying the demo mapping
   `clip(dBZ - 20, 0, 50) / 50`, foreground PSNR was still 6.70 dB because the
   archive centres were in km and the reference remained voxel-indexed.

Fixing one mismatch can reveal another. Stop only when the result is consistent
with an independently justified quality range.

## Data-dependent axis roles

A levelling rotation assigned roles to "the two widest axes". Beyond roughly
45 degrees of tilt those axes swapped: a synthetic 70-degree tilt was measured
as −20 degrees, and levelling reduced the in-plane elongation ratio from 2.30 to
1.15. Assign axis roles from the consumer's convention and test the derivation
against synthetic inputs with known answers.
