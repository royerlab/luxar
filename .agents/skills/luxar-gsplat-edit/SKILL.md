---
name: luxar-gsplat-edit
description: >-
  Edit, transform, and inspect an existing .gsplats.zarr dataset with the Luxar
  CLI. Use when a user has already-fitted Gaussian splats and wants to crop/slice,
  spatially transform (scale/rotate/translate/center), rescale intensity, cull or
  filter splats, partition into spatial parts, merge datasets (e.g. multichannel
  or as a new time dimension), convert to a web scene, migrate a legacy format, or
  inspect quality (info / render / compare / view / napari). This is the post-fit
  toolbox — for FITTING a volume use the gsplat-pipeline skill instead.
---

# Edit & inspect a fitted .gsplats.zarr

These `luxar gsplat` subcommands operate on an **already-fitted** dataset (the output
of `fit` / `batch-fit`). For fitting a volume, calibration, and LOD recipes, use the
**`luxar-gsplat-pipeline`** skill; for whole-timelapse fitting, **`luxar-hpc-batch-fit`**.

All commands take a `.gsplats.zarr` (flat, partition, or nested LOD), accept
`--encoding`/`-e` and usually `--compress`, and write a new dataset (non-destructive).

## Pick the operation

| Goal | Command |
| --- | --- |
| Crop to a coordinate box | `slice` (numpy-style ranges) |
| Move/scale/rotate/recenter in space | `transform` |
| Rescale or normalize brightness | `transform --scale-intensity / --normalize-intensity` |
| Drop low-value splats (shrink file) | `cull` |
| Keep splats matching property thresholds | `filter` |
| Split into spatial parts (frustum culling) | `partition` |
| Combine datasets (channels, timepoints) | `merge` |
| Make a web-viewer scene | `convert` |
| Upgrade an old-format file | `migrate-format` |
| Look at stats / quality | `info` / `render` / `compare` / `view` / `napari` |

## Common recipes

```bash
# Crop, then recenter + anisotropic z-scale
luxar gsplat slice in.gsplats.zarr crop.gsplats.zarr "0:50, :, 10:90"
luxar gsplat transform crop.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center

# Shrink: error-budget cull against the original volume (most principled)
luxar gsplat cull in.gsplats.zarr culled.gsplats.zarr --target vol.npy -p 95
# ...or simple cumulative-amplitude retention
luxar gsplat cull in.gsplats.zarr culled.gsplats.zarr -m cumulative -r 0.90

# Property filters (all AND-combined)
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1 --eccentricity-max 5
luxar gsplat filter in.gsplats.zarr out.gsplats.zarr --bbox "0,50,0,50,0,50" --volume-max 100

# Spatial partition for viewer frustum culling
luxar gsplat partition in.gsplats.zarr part.gsplats.zarr --parts 4 --rule sah

# Merge: stack two channels with colors, or stack timepoints as a new dimension
luxar gsplat merge ch0.gsplats.zarr ch1.gsplats.zarr -o multi.gsplats.zarr \
    --channel-colors "#ff0080,#00ff00"
luxar gsplat merge t0.gsplats.zarr t1.gsplats.zarr -o 4d.gsplats.zarr --as-dimension --values 0,1

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

## Notes

- **Full flag tables** for every command (cull methods, filter criteria, transform
  order, merge modes, partition rules, inspect options) are in
  `references/edit-commands.md`.
- The old `split` command is gone — use `partition`.
- `transform` applies operations in a fixed order: scale → rotate → translate → center
  → scale-intensity → normalize-intensity.
- `cull -m auto` picks error_budget (if `--target`), else redundancy (if `--shape`),
  else cumulative.
- `denoise` lives here too but acts on a raw VOLUME (pre-fit), not on splats.
