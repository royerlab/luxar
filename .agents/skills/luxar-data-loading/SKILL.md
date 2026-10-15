---
name: luxar-data-loading
description: >-
  Load an nD image / volume into Luxar and select the right slice. Use when a user
  needs to feed a .zarr / .zarr.zip / OME-Zarr / .tiff / .npy / .npz into a gsplat
  fit/cal, pick a channel or timepoint, choose a nested array key, or fix
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
| `.tiff` / `.tif` | needs `pip install "luxar[io]"` (tifffile + imagecodecs; compressed LZW/deflate/zstd TIFF and OME-TIFF need imagecodecs) |
| other | imageio fallback, needs `pip install "luxar[io]"` |

A missing optional dependency raises an actionable error naming the `pip install`
to run — including a compressed TIFF read without imagecodecs (`pip install
'luxar[io]'` or `pip install imagecodecs`).

**Physical voxel size is not read from TIFF.** The TIFF branch returns the pixel
array only (OME-TIFF `PhysicalSizeX/Y/Z` are ignored), and `fit`/`cal` work in
voxel index space either way: an anisotropic stack fits as if its voxels were
cubes. On this version the only `fit`-level route to physical spacing is
`voxel_size:` in a `--config` YAML (`luxar gsplat fit --dump-config` lists it
under "Anisotropic Voxels"); `batch-fit --physical` reads it from OME-Zarr NGFF
metadata.

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

## Selecting the slice to fit (CLI flags)

| Flag | Meaning | Accepted by |
| --- | --- | --- |
| `--channel` / `-c` | channel index (C in TCZYX), 0-based | fit, cal, compare (reference side), denoise |
| `--timepoint` | timepoint index (T), 0-based | fit, cal, compare (reference side), denoise |
| `--array-key` | array within a `.npz` or **nested zarr group**, e.g. `h2afva/fused` | fit, cal, denoise |
| `--axes` | explicit per-dim labels overriding the heuristic, e.g. `z,c,y,x` | fit, cal (and `batch-fit`) |

`compare` has **no** `--axes` and no `--array-key`: its `--channel`/`--timepoint`
slice only the reference, through the same heuristic below, so a non-zarr nD
reference (a 4D TIFF) cannot be sliced there. Extract the 3D reference yourself
(e.g. save the slice as `.npy`) and compare against that.

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
| 4D | `CZYX` if `--channel` given, else `TZYX` if `--timepoint` given | index the leading axis; with NEITHER flag the whole 4D array is used as-is |
| 3D | `ZYX` | kept as-is |
| 2D | `YX` | kept as-is |
| >5D | T=first, extra leading dims folded into channel, last 3 = spatial | |

**That table is the zarr path only.** For `.npy` / `.npz` / `.tiff` / imageio inputs
the loader squeezes the decoded array and returns it as-is, so `--channel` /
`--timepoint` are silently ignored — `fit movie.tiff --timepoint 3` fits the whole
5D stack. On any non-zarr nD input, pass `--axes` (which does the indexing itself).

**Override with `--axes`** (on `fit` / `cal`) when the layout differs (e.g. a
camera axis, or `ZCYX`). Give exactly one label per array dimension; recognized
labels are `time`/`t`, `channel`/`c`/`ch`/`camera`/`cam`, `z`/`y`/`x` (plus
`depth`/`height`/`width`), case-insensitive. **Any other label is an error**, not a
spatial axis. At most one time axis, indexed by `--timepoint` (default 0). Several
channel-like axes are allowed: `--channel` is then ONE flat row-major index across
all of them (e.g. `--axes time,camera,channel,z,y,x` with 2 cameras x 3 channels:
`--channel 4` = camera 1, channel 1). Time/channel axes are dropped; spatial axes
are kept in the given order and NOT squeezed, so `--axes z,y,x` keeps a single
z-plane. On zarr input `--axes` slices lazily, before anything is read.

## OME-Zarr metadata

For OME-Zarr the loader also discovers structure (axes, shape, `n_timepoints`,
`n_channels`, `spatial_shape`, and physical `voxel_size` + `unit` from NGFF
`coordinateTransformations`, plus the multiscale resolution-level count). Detection
order: NGFF v0.4+ multiscales → a custom `axes` attribute (e.g. Keller-lab
`time,camera,channel,z,y,x`) → the ndim heuristic above. `batch-fit` uses this to plan
tiles over T×C — see the **`luxar-hpc-batch-fit`** skill.

## Python API

```python
from luxar.io.volume import load_volume   # the loader behind fit/cal/compare/denoise

vol = load_volume(
    path,                 # pathlib.Path to .npy/.npz/.zarr/.zarr.zip/.tiff/...
    channel=None,         # int: channel index (C); flat across channel axes with axes=
    timepoint=None,       # int: timepoint index (T)
    array_key=None,       # str: array within .npz / nested zarr (e.g. "h2afva/fused")
    axes=None,            # str: explicit labels, e.g. "z,c,y,x"
    info=None,            # optional dict: receives "source_dtype" (pre-cast dtype)
    region=None,          # optional tuple of spatial slices (lazy on zarr)
)                         # -> float32 ndarray (>=2D), squeezed unless axes= used
```
(`luxar.cli.gsplat_config.load_volume` is the CLI wrapper that turns loader
errors into clean CLI exits.)
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
- **TIFF/other won't load** → `pip install "luxar[io]"`. "requires the
  'imagecodecs' package" on an older install → `pip install imagecodecs`.
- **Anisotropic stack looks squashed** → the fit is in voxel units (see "Physical
  voxel size" above); the scene's transform or a `--config` `voxel_size:` fixes it.

## Next step

Once the slice loads correctly, fit it with the **`luxar-gsplat-pipeline`** skill
(`cal → fit → lod`) or build a scene with **`luxar-visualization`**.
