---
name: luxar-gsplat-edit
description: >-
  Edit, transform, and inspect an existing .gsplats.zarr dataset with the Luxar
  CLI. Use when a user has already-fitted Gaussian splats and wants to crop/slice,
  spatially transform (scale/rotate/translate/center), rescale intensity, cull or
  filter splats, partition into spatial parts, flatten an LOD/partition tree to a
  single leaf, ladder existing leaves (additive), re-quantize the Cholesky
  encoding (reencode), retrofit LOD quality stamps (annotate-quality), merge
  datasets (e.g. multichannel or as a new time dimension), convert to a web
  scene, migrate a legacy format, or inspect quality
  (info / render / compare / view / napari). This is the post-fit
  toolbox — for FITTING a volume use the gsplat-pipeline skill instead.
---

# Edit & inspect a fitted .gsplats.zarr

These `luxar gsplat` subcommands operate on an **already-fitted** dataset (the output
of `fit` / `batch-fit`). For fitting a volume, calibration, and LOD recipes, use the
**`luxar-gsplat-pipeline`** skill; for whole-timelapse fitting, **`luxar-hpc-batch-fit`**.

The read-only inspection commands (`info` / `render` / `compare` / `view` /
`napari`) accept flat, partition, or nested LOD stores. `partition`, `cull`,
`filter`, `slice`, `decimate`, and `merge` require a flat (matrix-shaped) store;
run `luxar gsplat flatten` first for partition or nested inputs. Most *editing*
commands accept `--encoding`/`-e` and usually `--compress`, and write a new
dataset (non-destructive). Exception: `annotate-quality` stamps the input IN
PLACE (no `-e`).

## Pick the operation

| Goal | Command |
| --- | --- |
| Crop to a coordinate box | `slice` (numpy-style ranges) |
| Move/scale/rotate/recenter in space | `transform` |
| Rescale or normalize brightness | `transform --scale-intensity / --normalize-intensity` — but see below: you usually do NOT need this |
| Drop low-value splats (shrink file) | `cull` |
| Reduce to a target splat count | `decimate --target N` / `-f 0.1` |
| Keep splats matching property thresholds | `filter` |
| Split into spatial parts (frustum culling) | `partition` |
| Collapse LOD/partition tree to one flat leaf | `flatten` |
| Give every leaf a streaming ladder (inverse of flatten) | `additive` |
| Combine datasets (channels, timepoints) | `merge` |
| Make a web-viewer scene | `convert` |
| Re-quantize the on-disk Cholesky encoding (smaller/exact) | `reencode` |
| Retrofit Q·e LOD quality stamps in place | `annotate-quality` |
| Upgrade an old-format file | `migrate-format` |
| Look at stats / quality | `info` / `render` / `compare` / `view` / `napari` |

## Amplitudes: normalisation now happens on INSERTION, not here

A fitted `.gsplats.zarr` stores amplitudes in **raw source units** — the fitter
multiplies its `[0,1]` working copy back out by the volume's intensity range, so
a fit from a uint16 detector stack carries detector counts, in the hundreds or
thousands. Those units cannot be corrected at display time: the colormap window
only feeds the LUT index (`t = clamp((A-min)*scale, 0, 1)`, clamped, so it picks
a *colour*), while emitted radiance and volumetric optical depth are both LINEAR
in the raw stored amplitude and nothing windows them.

**Since #2211 the scene adders normalise by default** — `add_gsplats_from_data`,
`add_gsplats_from_file` and the graft path scale a robust p99.9 to 1.0, with ONE
factor for the whole structure, recorded as `amplitude_normalization_factor`.
Data already in range is untouched.

So `transform --normalize-intensity` is now for the cases the default does not
cover: pinning an explicit target across SEVERAL archives that must share an
exposure (multichannel, or a timelapse fitted in separate batches), or preparing
an archive for a consumer that is not the Luxar scene compiler. **`--scale-intensity`
on `gsplat convert` implies opting out of the automatic normalisation**, since
normalising straight after an explicit scale would cancel it exactly.

## `cull --method cumulative` ranks by PEAK, and that is not what you see

`--retention R` keeps splats carrying a fraction R of the total **peak
amplitude**. But a splat's contribution to the render is amplitude x volume, and
the dim splats a cumulative cull discards first are the LARGE, diffuse ones. So
the retention number systematically understates what the cull costs.

Measured on one Drosophila timepoint at `--retention 0.960`:

| | splats | peak amplitude | integrated mass | rendered intensity |
|---|---|---|---|---|
| pre-cull | 256,000 | 1.804e7 | 6.881e7 | 43.54 |
| post-cull | 165,739 | 1.738e7 | 5.749e7 | 32.25 |
| **retained** | 64.7% | **96.3%** | **83.5%** | **74.1%** |

The cull is honest about its own contract — 96.3% of peak, as asked. But it
removed **16.5% of the mass and 26% of the rendered intensity**, because the
discarded splats carry ~4.4x more mass per unit peak than the kept ones. Against
the raw volume that frame scored **28.7 dB foreground post-cull vs 37.2 dB
pre-cull** — an 8.5 dB drop for a nominal 4% amplitude loss.

Two consequences:

* **`retention` is not a fidelity dial.** 0.96 does not mean "96% as good"; on
  diffuse data it can mean a quarter of the light. If you need to bound the
  quality loss, score the culled result — do not infer it from R.
* **It is still the right tool when the diffuse component IS haze**, which is
  the usual case in light-sheet: the removed intensity was background, the
  render looks unchanged, and the authored `opacity` absorbs the difference.
  That is a judgement about the data, not something R guarantees.

## `cull --method cumulative`: report the COUNT, not the retention

`--retention R` is an *input*; the surviving splat count is the *outcome*, and it
depends entirely on the amplitude distribution. **Always report and check the
count** — R alone tells a reader nothing about what they will get.

This was worth more than a style note. The same `-r 0.960` returned **65%** of
splats on a 20-frame store and **12%** on the 500-frame one, from provably
identical amplitude distributions (both need exactly 65.0% of splats for 96% of
amplitude). Cause: the cumulative sum accumulated in **float32** and saturated on
the larger set — fixed in #2260, filed as #2258. Ladder depth was the obvious
suspect and was wrong: 1, 2, 4, 8 and 13 rungs all gave 65.3% on the same data.

So the check that would have caught it, and is worth keeping:

* **Compare the resulting count against the CDF.** If `-r 0.96` does not keep
  roughly the fraction `np.cumsum(np.sort(amps)[::-1], dtype=np.float64)` says
  it should, something other than the documented rule is acting. Here `amps`
  means alpha-effective amplitude (`A·α` for RGBA), as returned by
  `luxar.gsplats.utils.alpha.effective_amplitudes`, not raw amplitude.
* **Check per-slice survival on a timelapse** (`np.unique` on the stacked centre
  column). On this data all 500 frames survived within a 2.4x spread, so the loss
  was uniform rather than gutting dim timepoints — worth confirming, not assuming.

**There is deliberately no occlusion/shading command here.** If a fit renders as a
flat glow with no visible shape, the fix is `luxar.shading.bake_ambient_occlusion`
applied at SCENE-authoring time (pass `mass=amplitudes`) — see the
`luxar-visualization` skill. A `.gsplats.zarr` is a *reconstruction*; colormaps,
tone mapping and occlusion are all authored on the way into a scene. Baking a
per-splat shading sidecar into the store instead would give `reencode`, `lod`,
`decimate` and every refit one more array to reorder or silently invalidate.

## Common recipes

```bash
# Crop, then recenter + anisotropic z-scale
luxar gsplat slice in.gsplats.zarr crop.gsplats.zarr "0:50, :, 10:90"
luxar gsplat transform crop.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center

# Shrink: error-budget cull against the original volume (most principled)
luxar gsplat cull in.gsplats.zarr culled.gsplats.zarr --target vol.npy -p 95
# ...or simple cumulative-amplitude retention
luxar gsplat cull in.gsplats.zarr culled.gsplats.zarr -m cumulative -r 0.90

# Property filters (all AND-combined). Any min/max accepts a number OR a
# percentile written 'pNN' / 'NN%' (robust on heavy-tailed attributes).
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1 --eccentricity-max 5
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --bbox "0,50,0,50,0,50" --volume-max 100
# --scale-min/max = characteristic size (geometric-mean SPATIAL sigma), the
# recommended "remove large diffuse background" knob (timelapse-safe).
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --scale-max p90            # drop largest 10%
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --scale-max p90 --dry-run  # preview impact only
# Remove spatially-isolated noise splats (NN distance / local density).
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --isolation-max p99
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --min-neighbors 3 --neighbor-radius 5
# Soft reweighting (no popping; count unchanged): attenuate by scale.
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --soft-highpass p90 --soft-width 1.0
# --spatial-dims 0,1,2 overrides auto spatial-axis detection.

# Spatial partition for viewer frustum culling
luxar gsplat partition in.gsplats.zarr part.gsplats.zarr --parts 4 --rule sah

# Collapse any tree (partition/LOD/nested) to a single matrix-shaped leaf.
# Useful for compatibility with tools that expect a flat .gsplats.zarr, or before
# rebuilding a fresh global LOD topology from partitioned output.
luxar gsplat flatten part.gsplats.zarr flat.gsplats.zarr

# Add a per-leaf streaming ladder WITHOUT flattening (inverse companion of
# flatten): every leaf of an existing partition/LOD/nested tree gains an
# additive prefix-sum ladder, keeping the tree structure intact. This is the
# tool for "give partitioned output a streaming ladder" — the per-leaf
# counterpart of `lod --recipe stream` (which needs a flat input). Same
# streaming knobs (--n-lods / --method / --breakpoints / --target-ms).
luxar gsplat additive part.gsplats.zarr streamed.gsplats.zarr --target-ms 200
luxar gsplat additive in.gsplats.zarr out.gsplats.zarr --n-lods 4
# This is also the no-refit REMEDY for a store whose ladder is too deep for a
# node the viewer slices: it rebuilds each leaf's ladder from its union, so a
# starved first rung can be fixed without re-fitting. Use `--n-lods 3..4` for
# that, NOT `--target-ms` — an absolute rung is divided by the slice count, a
# share is not (see the gsplat-pipeline skill's "Sliced nodes" reference).
# Check the result with `hatch run check-demo-ladders <store>`.

# Re-quantize the on-disk Cholesky encoding (structure-preserving copy; decode is
# always float32, but the STORED values are re-quantised — see below). -e memory =
# uint8 (smallest, ~93 dB); -e auto = adaptive u8→u16→f32 certificate ladder
# (near-lossless); -e precision = float32 (exact/archival). That ladder is the
# CHOLESKY policy, but the command re-encodes the CENTERS too — which its name does
# not suggest — and there both `auto` and `memory` mean uint16 over each axis's own
# extent (float32 only once an axis spans 2^16), so a degenerate
# stacked column comes back quantised: see the `merge --as-dimension` note in Notes.
luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory
luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision

# Retrofit Q·e quality stamps onto an existing LOD dataset, in place (no refit /
# re-ladder) — enables the viewer's early energy-threshold LOD upgrades.
luxar gsplat annotate-quality in.gsplats.zarr                # e(k) + w only (fast)
luxar gsplat annotate-quality in.gsplats.zarr --with-quality # + measured per-level Q
luxar gsplat annotate-quality in.gsplats.zarr --dry-run      # print stamps, write nothing

# Merge: stack two channels with colors, or stack timepoints as a new dimension
luxar gsplat merge ch0.gsplats.zarr ch1.gsplats.zarr -o multi.gsplats.zarr \
    --channel-colors "#ff0080,#00ff00"
# --as-dimension quantises the stacked coordinate under the default `auto`
# encoding. The first and last values survive exactly; an INTERIOR one does not,
# and its slice can then attenuate to nothing — so carry -e precision (see Notes).
luxar gsplat merge t0.gsplats.zarr t1.gsplats.zarr t2.gsplats.zarr \
    -o 4d.gsplats.zarr --as-dimension --values 0,1,2 -e precision

# Convert to a web scene, and upgrade a legacy file
luxar gsplat convert in.gsplats.zarr scene.luxar.zarr --center
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr
```

## Inspect & quality-check

```bash
luxar gsplat info in.gsplats.zarr                      # counts, bbox, distributions
luxar gsplat render in.gsplats.zarr out.npy --shape 128,128,128   # rasterize to volume
luxar gsplat compare fitted.gsplats.zarr original.tiff --output-json metrics.json
luxar gsplat view in.gsplats.zarr                      # quick web viewer
luxar gsplat napari in.gsplats.zarr                    # napari + centers overlay
```

Before scoring a stacked or transformed archive, read "Scoring a STACKED or
TRANSFORMED archive" in the `luxar-gsplat-pipeline` skill; `compare --timepoint`
slices the reference, not the splat archive.

### `render --shape` is in the SPLATS' coordinate space, not the volume's

If the fit was run with a physical `voxel_size`, the centres are in **microns**, not
voxel indices. Rendering into the source volume's voxel shape then packs the whole
reconstruction into a corner (a 0.44 um/voxel stack renders into the first ~44% of
each axis) — and it fails *silently*: you get a plausible-looking array, and metrics
that read as catastrophic reconstruction failure rather than as a units bug.

Check first — `centers.max(0)` against the shape — and convert with `transform`,
which rescales the centres and the covariance together:

```python
# Sigma = L L^T scales as the SQUARE of the length unit, so the Cholesky factor
# rescales along with the centres. transform() does both, per axis, and keeps
# amplitudes, colours and any LOD structure. vox is microns per voxel (Z, Y, X).
g_vox = g.transform(np.diag(1.0 / np.asarray(vox, dtype=np.float64)))
```

Do not hand-roll it as `cholesky_factors / vox`: the factors are packed
lower-triangular (6 wide in 3D, not 3), so a per-axis `vox` — the way
`voxel_size` is spelled everywhere else — raises a broadcast error, and
substituting a scalar mean gets an anisotropic stack quietly wrong.

Scoring the result has its own two traps — global PSNR on a sparse stack is
flattered by the empty part (one fit: 47 dB global vs 23 dB foreground), and a
floored fit must be scored against the ORIGINAL, not against its own preprocessed
input. Both are written up under **"Traps that cost real time"** in the
`luxar-gsplat-pipeline` skill; they are not repeated here so the two cannot drift
apart. Expect a total energy ratio below 1.0 against the raw data — with `--floor`
the pedestal is gone by design.

## Notes

- **Full flag tables** for every command (cull methods, filter criteria, transform
  order, merge modes, partition/flatten rules, inspect options) are in
  `references/edit-commands.md`.
- The old `split` command is gone — use `partition`.
- `transform` applies operations in a fixed order: scale → rotate → translate → center
  → scale-intensity → normalize-intensity. It composes to `p → R·S·p + (t − c)`, so **a
  per-axis scale is applied in the PRE-rotation frame** — which is what you want
  for a voxel-pitch correction on the stored axes, but wrong if the `--scale`
  vector is written in the frame you end up in. `--scale 0.19,0.19,0.38
  --rotate-y 90` in one call applies 0.38 to the stored third axis, not to the
  one the rotation brings into that slot; when the two are expressed in
  different frames, split them into two invocations. A partition's `bsp_tree`
  split planes are RE-MAPPED through translation, per-axis scale and quarter-turn
  rotations (a mirror also swaps each node's halves). What invalidates them is an
  affine under which one of the tree's own split axes has no axis image — no
  single nonzero in both its column and that row, or an image axis past the third
  the stored format admits. The common case is a rotation that is not a multiple
  of 90°, which shears the cells out of axis-alignment; a quarter turn that sends
  a split axis to a fourth dimension (`--spatial-dims 1,2,3`) and a zero `--scale`
  factor land in the same place. That tree is then dropped and the
  viewer's back-to-front ordering downgrades to centroid order, which the command
  warns about — re-partition afterwards to restore exact ordering.
- **`merge --as-dimension` needs `-e precision` if the stacked coordinates must
  be exact.** The default `auto` encoding quantises centers to uint16 over the
  column's own `[min, max]`, which puts the ENDPOINTS on exact codes and leaves
  every interior value to rounding: `--values 0,1,2` comes back as
  `[0, 1.0000153, 2]` (reproduced; `-e precision` gives exact `[0, 1, 2]`),
  whereas a two-timepoint `--values 0,1` is bit-exact. Harmless on an axis that
  carries real extent, fatal on a stacked one — and that is the default: `--sigma
  0` gives each splat σ = 1e-7 in the new axis, so a coordinate off by 1.5e-5
  sits ~150 σ from the slice, far past the truncation radius, and the whole
  middle slice attenuates to nothing. Carry `-e precision` through *every* stage
  that rewrites the store, not just the merge — `lod`, `transform`, `convert`,
  `cull`, `filter`, `partition`, `flatten`, `decimate`, `slice` and `additive`
  all default to `auto`, and `reencode` defaults to `memory`.
- `cull -m auto` picks error_budget (if `--target`), else redundancy (if `--shape`),
  else cumulative.
- `denoise` lives here too but acts on a raw VOLUME (pre-fit), not on splats.
