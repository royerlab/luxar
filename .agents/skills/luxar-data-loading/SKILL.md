---
name: luxar-data-loading
description: >-
  Load an nD image / volume into Luxar and select the right slice. Use when a user
  needs to feed a .zarr / .zarr.zip / OME-Zarr / .tiff / .npy / .npz into a gsplat
  fit/cal/compare, pick a channel or timepoint, choose a nested array key, or fix
  axis-order detection (TCZYX/CZYX/ZYX) with explicit --axes labels. Covers
  supported formats, the load_volume API, OME-Zarr multiscale/voxel-size discovery,
  and the common "wrong axes / huge movie loaded into RAM" pitfalls.
---

# Load nD images & volumes into Luxar

The fitting commands (`luxar gsplat fit` / `cal` / `compare`, and `denoise`) load a
volume through one loader. Getting the **format, slice selection, and axis order**
right up front avoids fitting the wrong data or blowing up RAM.

## Supported formats

| Format | Notes |
| --- | --- |
| `.npy` | NumPy array (base dependency) |
| `.npz` | NumPy archive — first array, or pick with `--array-key` |
| `.zarr` / `.zarr.zip` | Zarr array or group, incl. OME-Zarr 5D (base dependency) |
| `.tiff` / `.tif` | needs `pip install "luxar[io]"` (tifffile) |
| other | imageio fallback, needs `pip install "luxar[io]"` |

A missing optional dependency raises an actionable error (e.g. "Install with: `pip
install luxar[io]`").

## Selecting the slice to fit (CLI flags, shared by fit/cal/compare/denoise)

| Flag | Meaning |
| --- | --- |
| `--channel` / `-c` | channel index (C in TCZYX), 0-based |
| `--timepoint` | timepoint index (T), 0-based |
| `--array-key` | array within a `.npz` or **nested zarr group**, e.g. `h2afva/fused` |
| `--axes` | explicit per-dim labels overriding the heuristic, e.g. `z,c,y,x` |

```bash
luxar gsplat fit data.zarr.zip out.gsplats.zarr --timepoint 0 --channel 0
luxar gsplat fit data.zarr.zip out.gsplats.zarr --array-key h2afva/fused
luxar gsplat fit data.zarr.zip out.gsplats.zarr --axes time,camera,channel,z,y,x --timepoint 5
```

## Axis-order detection (and when to override)

Without `--axes`, the loader infers axes from the number of dimensions:

| ndim | Assumed axes | Default slice |
| --- | --- | --- |
| 5D | `TCZYX` | T=`--timepoint` (def 0), C=`--channel` (def 0) → 3D |
| 4D | `CZYX` (or `TZYX` if `--timepoint` given) | index the leading axis |
| 3D | `ZYX` | kept as-is |
| 2D | `YX` | kept as-is |
| >5D | T=first, extra leading dims folded into channel, last 3 = spatial | |

**Override with `--axes`** when the layout differs (e.g. a camera axis, or `ZCYX`).
Recognized labels: `time`/`t`, `channel`/`c`/`ch`/`camera`/`cam`, `z`/`y`/`x` (plus
`depth`/`height`/`width`); unknown labels are treated as spatial. Time/channel axes are
indexed (default 0) and dropped; spatial axes are kept in the given order.

## OME-Zarr metadata

For OME-Zarr the loader also discovers structure (axes, shape, `n_timepoints`,
`n_channels`, `spatial_shape`, and physical `voxel_size` + `unit` from NGFF
`coordinateTransformations`, plus the multiscale resolution-level count). Detection
order: NGFF v0.4+ multiscales → a custom `axes` attribute (e.g. Keller-lab
`time,camera,channel,z,y,x`) → the ndim heuristic above. `batch-fit` uses this to plan
tiles over T×C — see the **`luxar-hpc-batch-fit`** skill.

## Python API

```python
from luxar.cli.gsplat_config import load_volume   # used by fit/cal/compare

vol = load_volume(
    path,                 # Path to .npy/.npz/.zarr/.zarr.zip/.tiff/...
    channel=None,         # int: channel index (C)
    timepoint=None,       # int: timepoint index (T)
    array_key=None,       # str: array within .npz / nested zarr (e.g. "h2afva/fused")
    axes=None,            # str: explicit labels, e.g. "z,c,y,x"
)                         # -> float32 ndarray (>=2D), squeezed unless --axes used
```
Zarr arrays are **lazily sliced** before materialization — selecting a timepoint/channel
reads only that slice, so a huge nD movie is never pulled into RAM whole.

## Common pitfalls

- **Wrong axes** → fitting a transposed/garbage volume. If `info`/`compare` looks wrong,
  pass `--axes` explicitly. A camera axis is the usual culprit on light-sheet data.
- **Whole movie into RAM** → happens if you materialize a 4D/5D array yourself instead
  of letting `load_volume` (or `--timepoint`/`--channel`) slice it lazily.
- **Nested zarr group** → `--array-key path/within/group` (also disambiguates a `.npz`
  with multiple arrays).
- **TIFF/other won't load** → `pip install "luxar[io]"`.

## Next step

Once the slice loads correctly, fit it with the **`luxar-gsplat-pipeline`** skill
(`cal → fit → lod`) or build a scene with **`luxar-visualization`**.
