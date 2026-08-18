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

### Microscopy vendor containers

The dispatch is by **file suffix**, and anything not `.npy` / `.npz` / zarr /
`.tiff`/`.tif` falls through to `imageio.imread` — which either works whole-file or
raises. Two cases worth knowing before you plan a fit:

- **`.lsm` (Zeiss)** loads, via the imageio fallback (tifffile plugin). But that
  path reads the WHOLE file eagerly: no lazy slice, no `--array-key`, and nothing
  is sliced at all unless you pass `--axes` — with which the slicing happens
  *after* the whole file has been read. Budget RAM from the decoded array, not
  the file size: the loader returns float32, so the resident cost is
  `prod(shape) × 4` bytes whatever the file's compression — 2× an *uncompressed*
  16-bit acquisition, and an unbounded multiple of a compressed one, transiently
  more again while the cast is in flight.
  Convert to zarr first if the file is large.
- **`.h5j` (Janelia FlyLight) is NOT supported.** Despite the HDF5 extension it
  is a container of per-channel **H.265 elementary streams**, so no array reader
  can open it. Decode with ffmpeg, then **crop the macroblock padding** — the
  streams are padded up to macroblock bounds and the real extent is in the
  file's `pad_right` / `pad_bottom` attributes — and write zarr.

That is the general escape hatch for any unsupported container: decode it to
zarr yourself and feed Luxar the zarr. Doing so also buys the lazy slicing the
imageio path does not have.

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

**That table is the zarr path only.** For `.npy` / `.npz` / `.tiff` / imageio inputs
the loader squeezes the decoded array and returns it as-is, so `--channel` /
`--timepoint` are silently ignored — `fit movie.tiff --timepoint 3` fits the whole
5D stack. On any non-zarr nD input, pass `--axes` (which does the indexing itself).

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
  of letting `load_volume` (or `--timepoint`/`--channel`) slice it lazily — and
  unavoidably on any non-zarr container, where `load_volume` itself reads the whole
  file (see "Microscopy vendor containers").
- **Nested zarr group** → `--array-key path/within/group` (also disambiguates a `.npz`
  with multiple arrays).
- **TIFF/other won't load** → `pip install "luxar[io]"`.

## Next step

Once the slice loads correctly, fit it with the **`luxar-gsplat-pipeline`** skill
(`cal → fit → lod`) or build a scene with **`luxar-visualization`**.
