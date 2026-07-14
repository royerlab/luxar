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
(1.0 = sphere; spatial by default), `--mass-min/--mass-max` (amplitude × volume),
`--sigma-axis` + `--sigma-min/--sigma-max`.

**Percentiles:** any min/max threshold accepts a plain number OR a percentile of
that attribute written `pNN` / `NN%` (e.g. `--scale-max p90`) — robust on
heavy-tailed attributes.

Additional criteria:
- `--scale-min/--scale-max` — characteristic size = geometric-mean SPATIAL sigma;
  timelapse-safe (auto-ignores a zero-variance time axis). The recommended
  "remove large diffuse background" knob (e.g. `--scale-max p90`).
- `--isolation-max` — remove spatially-isolated noise splats by nearest-neighbour
  distance (e.g. `--isolation-max p99`).
- `--min-neighbors` + `--neighbor-radius` — remove splats with too few neighbours
  within a radius (local density). Grouped by the non-spatial axis so timepoints
  never count as neighbours.
- `--soft-highpass` / `--soft-lowpass` (+ `--soft-width` octaves) — SOFT
  reweighting: attenuate amplitude by a smooth function of scale instead of
  hard-removing (no popping; splat count unchanged). High-pass suppresses
  large/diffuse background.
- `--spatial-dims 0,1,2` — override the auto spatial-axis detection used by
  scale / eccentricity / isolation.
- `--dry-run` — preview impact (splats / mass / amplitude removed); write nothing.

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
Upgrade legacy layouts (v1.0 / v1.1 / v2.0 substitutive dir / v3.0/v3.1) → v3.2.
`--overwrite`, `--lossless` (preserve float32 Cholesky), `--quiet`/`-q`. (For a
structure-preserving re-quantization of a current-format file, use `reencode`
instead — it exposes the full encoding ladder.)

## reencode IN OUT
Re-quantize a fitted dataset's Cholesky factors to another encoding (a
structure-preserving copy: the whole node tree — leaf / additive ladder /
`kind=lod` / partition / nested — plus `fitting`/`provenance`/`pipeline` groups
are carried over verbatim; only the on-disk Cholesky encoding changes). Splat
count and geometry are unchanged; decode is always to float32, so
viewer/GPU/WASM paths are unaffected. Unlike `migrate-format` (legacy → current,
float32-vs-AUTO-uint16 only), this exposes the full ladder including `memory`
(uint8) and works on already-current files.
| Flag | Default | Meaning |
| --- | --- | --- |
| `--encoding` / `-e` | `memory` | `memory` = uint8 (smallest, ~93 dB); `auto` = adaptive u8→u16→f32 ladder (near-lossless); `precision` = float32 (exact/archival) |
| `--ordering` | `hilbert` | output spatial chunk ordering (`hilbert` / `morton` / `none`) |
| `--quiet` / `-q` | off | suppress trailing summary |

## additive IN OUT
Give every leaf of an existing tree a per-leaf additive (streaming) ladder,
structure-preservingly — `kind=lod` levels, partition parts, adaptive/overview
groups all keep their shape — WITHOUT recomputing the expensive
substitutive/partition structure. The per-leaf counterpart of
`lod --recipe stream` (which needs a flat input) and the inverse companion of
`flatten`. Its exact use case: giving tiled/partitioned output a streaming
ladder. Same streaming knobs as `lod --recipe stream`: `--n-lods`,
`--method`, `--breakpoints` (incl. `stream:C`), `--target-ms`,
`--bandwidth-mbps`, `--bytes-per-splat`, `--encoding`/`-e`, `--compress`,
`--overwrite`.

## annotate-quality IN
Retrofit Q·e quality stamps onto an existing `.gsplats.zarr`, **in place** (no
refit / re-ladder): per-additive-sub-LOD cumulative energy `e(k)` + per-leaf
reference energy `w` (cheap O(N); enables the viewer's early energy-threshold
LOD upgrades on legacy datasets). Re-stamps the root `content_hash` so viewer
caches invalidate automatically. Directory stores only (unpack `.zip` first).
| Flag | Meaning |
| --- | --- |
| `--with-quality` | also measure per-level mixture-L² Q vs each lod group's finest content |
| `--dry-run` | compute and print the stamps; write nothing |

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
