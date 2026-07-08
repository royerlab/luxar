# `luxar gsplat` editing & inspection commands — full reference

Verified from `cli/gsplat_ops/transforms.py`, `scene.py`, `inspect.py`. All operate on
a `.gsplats.zarr` (flat, partition, or nested LOD) and accept `--encoding`/`-e`
(`auto`/`precision`/`memory`) and usually `--compress` (`zip`/`tar.gz`). NOTE: the old
`split` command is gone — use `partition`.

---

## transform IN OUT
Spatial then intensity, applied in this order: **scale → rotate → translate → center →
scale-intensity → normalize-intensity**.
| Flag | Meaning |
| --- | --- |
| `--scale` / `-s` | per-axis scale, e.g. `4,1,1,1` |
| `--rotate-x` / `--rotate-y` / `--rotate-z` | degrees |
| `--translate` / `-t` | per-axis offset, e.g. `0,100,0` |
| `--center` | recenter at amplitude-weighted centroid (default off) |
| `--scale-intensity` | multiply amplitudes |
| `--normalize-intensity` | scale so max amplitude = given value (e.g. `1.0`) |

## cull IN OUT
Remove low-value splats. `--method`/`-m` (default `auto`):
- `auto` → `error_budget` if `--target`, else `redundancy` if `--shape`, else `cumulative`.
- `error_budget` (most principled, needs `--target VOL`): `--error-percentile`/`-p` (99),
  `--error-tolerance` (1.0), `--max-iters` (8).
- `redundancy` (needs `--shape` e.g. `41,512,512`): `--redundancy-threshold` (0.01).
- `cumulative`: `--retention`/`-r` (0.95) — keep top fraction of cumulative amplitude.
- also `amplitude_percentile`, `combined`.
Shared: `--truncate`, `--device`/`-d`, `--channel`/`-c`, `--timepoint`.

## filter IN OUT
Keep splats matching ALL given criteria (unspecified = skipped):
`--bbox "x0,x1,y0,y1,z0,z1"`, `--amplitude-min/--amplitude-max` (+ `--amplitude-normalized`),
`--volume-min/--volume-max` (+ `--volume-normalized`), `--eccentricity-min/--eccentricity-max`
(1.0 = sphere), `--mass-min/--mass-max` (amplitude × volume), `--sigma-axis` + `--sigma-min/--sigma-max`.

## slice IN OUT RANGES
Numpy-style, comma-separated per spatial axis; float coords allowed:
`"0:50, :, 10:90"`, `":50, 20:80, :"`. Each is `lo:hi` / `lo:` / `:hi` / `:`.

## merge IN... -o OUT
Combine datasets. Modes (mutually exclusive):
- default: concatenate.
- `--as-dimension`: stack along a new dim; `--values` (default 0,1,2,…), `--sigma` (0 = discrete).
- `--channel-colors "#ff0080,#00ff00"`: per-input hex colors.

## partition IN OUT
Spatial BSP into one `kind=partition`. `--max-elements`/`-m` OR `--parts`/`-n`
(mutually exclusive); `--rule` (`median` default / `midpoint` / `sah`).

## flatten IN OUT
Collapse any gsplat tree (leaf, LOD/matrix-shaped tree, partition, or nested
partition/LOD) into one flat matrix-shaped leaf. Use it when a downstream tool
expects a flat `.gsplats.zarr`, or to rebuild a fresh global LOD after tiled or
partitioned fitting. Options: `--overwrite`, `--encoding`/`-e`, `--compress`.

## convert IN.gsplats.zarr OUT.luxar.zarr
Wrap a fitted dataset as a web scene. `--center`/`--no-center` (default on),
`--scale-intensity`, `--opacity` (1.0), `--blending-mode` (`additive` default /
`normal` / `max` / `opaque`).

## migrate-format IN OUT
Upgrade legacy layouts (v1.0 / v1.1 / v2.0 / substitutive dir) → v3.1.
`--overwrite`, `--lossless` (preserve float32 Cholesky), `--quiet`/`-q`.

---

## Inspection (read-only)

| Command | Key options | Output |
| --- | --- | --- |
| `info IN` | `--histograms/--no-histograms`, `--bins`/`-b` (40) | counts, ndim, bbox, amplitude/volume/color distributions, metadata |
| `render IN OUT.npy` | `--shape` (auto from bbox), `--device`/`-d`, `--truncate`/`-t` | rasterize splats → volume (.npy/.tiff) |
| `compare GS REF` | `--shape`, `--device`/`-d`, `--channel`/`-c`, `--timepoint`, `--output-json`/`-j`, `--quiet` | PSNR / SSIM / MSE / rel-L2 / max-abs-err |
| `view IN` | `--port`/`-p` (8000), `--viewer-port` (5173), `--open/--no-open`, `--cors-origin` | quick web viewer (handles partition/nested; extracts archives) |
| `napari IN` | (none) | render to volume + centers overlay in napari |

## denoise IN OUT (volume preprocessing, not a gsplat op)
NLM-denoise a raw volume before fitting. `--h` (manual, skip auto-calib),
`--patch-size` (3), `--search-distance` (5), `--backend` (auto), `--device`/`-d`,
`--denoise-2d`; input selection `--channel`, `--timepoint`, `--array-key`.
