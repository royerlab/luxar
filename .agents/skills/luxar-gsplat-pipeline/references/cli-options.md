# Luxar gsplat CLI — full option reference

Exhaustive flags for `fit`, `cal`, and `lod`. Run `luxar gsplat <cmd> --help` for
the authoritative live list; this file mirrors it for offline reference. Defaults
shown are the code defaults.

---

## `luxar gsplat fit INPUT OUTPUT`

Inputs: `.npy`, `.npz`, `.tiff`/`.tif`, `.zarr`, `.zarr.zip` (TIFF/other need `pip install "luxar[io]"`).

### Core
| Flag | Default | Meaning |
| --- | --- | --- |
| `--seeds` / `-s` | auto | int = splat count; float in (0,1) = compression ratio; `auto` |
| `--iters` / `-n` | preset | max optimization iterations |
| `--preset` | none | `draft` / `standard` / `hifi` / `ultra` / `n2s` (see preset table) |
| `--config` | none | YAML config file (overrides preset) |
| `--dump-config` | false | print resolved default config and exit |
| `--device` / `-d` | auto | `auto` / `cpu` / `cuda` / `mps` |
| `--loss` | l1 | `l1` / `mse` / `poisson` |
| `--lr` | preset | learning rate |
| `--seed-method` | auto | `auto` / `edges` / `grid` / `decomposition` / `peaks` (comma-combinable) |
| `--compress` / `-c` | none | compress output: `zip` / `tar.gz` |
| `--verbose` / `--quiet` | verbose | output verbosity |

### Input selection (nD volumes / OME-Zarr)
| Flag | Meaning |
| --- | --- |
| `--channel` | channel index (C in TCZYX) |
| `--timepoint` | timepoint index (T in TCZYX) |
| `--array-key` | array key in `.npz` or nested zarr group (e.g. `h2afva/fused`) |
| `--axes` | explicit per-dim labels (e.g. `z,c,y,x`), bypassing the TCZYX/CZYX/ZYX heuristic |
| `--downscale` | downsample factor(s) before fitting (scalar or per-axis comma list) |

### Tiling (large volumes)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--tiling` | auto | `auto` / `none` / `uniform` / `content` |
| `--flat` | false | emit a single leaf instead of a `kind=partition` |
| `--tile-size` | 256 | tile edge in voxels |
| `--overlap` | 32 | inter-tile overlap voxels (Hann-stitched) |
| `--tile` | none | fit a single tile `N/M` (e.g. `3/16`) — Slurm-ready |
| `--jobs` / `-j` | 1 | concurrent tiles on one GPU; `auto` sizes from free VRAM |
| `--keep-tiles` | false | keep per-tile temp outputs after merge |

### Content-adaptive tiling (`--tiling content`) — needs a density
| Flag | Default | Meaning |
| --- | --- | --- |
| `--cal` | none | calibration JSON from `gsplat cal` (supplies density) |
| `--k-star-ref` | — | reference K* (effective splats) |
| `--n-features-ref` | — | reference feature count |
| `--saturation-exponent` | 0.44 | K ~ features^alpha |
| `--saturation-cap` | — | per-box splat cap |
| `--feature-threshold` | — | absolute feature-count threshold |
| `--feature-metric` | peaks | `peaks` / `edges` / `intensity` |
| `--cell` | 16 | content-scan cell size |
| `--target-features` | — | features per content box |
| `--min-leaf` / `--max-leaf` | 256 / 512 | content-box edge bounds |
| `--plan` | none | pre-computed FitPlan JSON |
| `--plan-only` | false | write the box plan JSON and stop (no fit) |

### Per-part LOD at fit time (tiled partition only)
`--recipe additive` → `partitioned` topology; `--recipe substitutive` → `mosaic`.
Knobs mirror `lod`: `--n-lods`, `--additive-method`/`-m`, `--breakpoints`/`-b`,
`--compression-factor`/`-K`, `--levels`/`-L`, `--substitutive-method`,
`--coarsen-dims`, `--lod-method`.

### Progressive fitting
| Flag | Default | Meaning |
| --- | --- | --- |
| `--progressive` | false | multi-pass fitting on residuals |
| `--splats-per-pass` | 5000 | max splats added per pass |
| `--psnr-patience` | 0.5 | stop when ΔPSNR < this (dB) |
| `--max-passes` | none | cap on passes |

### Post-fit culling & denoising
| Flag | Default | Meaning |
| --- | --- | --- |
| `--cull-retention` | none | keep top fraction of cumulative amplitude (e.g. 0.95) |
| `--denoise` | false | NLM-denoise the volume before fitting |
| `--denoise-h` | auto | manual NLM strength (skip auto-calibration) |
| `--denoise-2d` | false | slice-by-slice 2D NLM |
| `--denoise-patch-size` / `--denoise-search-distance` | 3 / 5 | NLM windows |
| `--denoise-backend` | auto | `auto` / `cuda` / `pytorch` / `skimage` |

### Presets (`gsplat_config.py`)
| Preset | n_iters | early_stop_patience | max_eccentricity | cull_retention |
| --- | --- | --- | --- | --- |
| `draft` | 2,000 | 200 | 10 | 0.999 |
| `standard` | 5,000 | 300 | 10 | 0.999 |
| `hifi` | 10,000 | 400 | 15 | 0.999 |
| `ultra` | 20,000 | 500 | 20 | 0.999 |
| `n2s` | 20,000 | 500 | 10 | 0.999 (manuscript blind-spot protocol) |

---

## `luxar gsplat cal INPUT OUTPUT_JSON`

Blind-spot (Noise2Self) K-sweep → K*, curve type, noise floor, PSNR ceiling.

### K grid
| Flag | Default | Meaning |
| --- | --- | --- |
| `--k-grid` | none | explicit K values (overrides n-grid/min/max) |
| `--n-grid` | 10 | number of K values |
| `--k-min` / `--k-max` | 1000 / 512000 | sweep bounds |
| `--progression` | exp | `exp` (log-spaced) / `power` |
| `--power` | 2 | exponent when `--progression power` |

### Cross-validation mask
| Flag | Default | Meaning |
| --- | --- | --- |
| `--mask-fraction` | 0.05 | held-out voxel fraction |
| `--mask-seed` | 42 | RNG seed for the mask |

### Fit config & loader pass-through
`--preset` (default `n2s`), `--config`, `--device`/`-d`, and the same input-selection
flags as `fit`: `--channel`/`-c`, `--timepoint`, `--array-key`, `--axes`.

### Regime-robust extensions
| Flag | Default | Meaning |
| --- | --- | --- |
| `--k-star-metric` | psnr_minmax | `psnr_minmax` / `psnr_foreground` / `gain` |
| `--auto-region` / `--no-auto-region` | off | calibrate on a content-rich sub-region |
| `--region-size` | 256 | auto-region edge (voxels) |
| `--region-strategy` | densest | `densest` / `median` |
| `--feature-metric` | peaks | `peaks` / `edges` / `intensity` |
| `--saturation-exponent` | 0.44 | K ~ features^alpha |
| `--rd-model` / `--no-rd-model` | on | fit parametric error-vs-K model |
| `--fit-exponent` | false | measure alpha across scales (multiplies runtime) |
| `--exponent-scales` | 128,192,256 | region edges for `--fit-exponent` |

### Outputs
`--pdf PATH` (report), `--keep-fits DIR` (persist per-K fits + slice montages), `--quiet`/`-q`.

---

## `luxar gsplat lod INPUT OUTPUT`

`--recipe` is REQUIRED. Recipes scale-ordered: `flat` < `additive` < `partitioned`
< `multiscale` / `mosaic`; primitives `substitutive`, `pyramid`.

### Additive ladder (additive / partitioned / multiscale / pyramid)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-lods` | 4 | additive LOD levels |
| `--method` / `-m` | auto | `auto` (greedy at N ≤ 5000, else self_energy) / `greedy` / `self_energy` / `mass` / `amplitude` / `spectral` / `random` |
| `--breakpoints` / `-b` | equal-count | `equal-count` / `stream:C` (geometric streaming ladder, first chunk C splats then doubling; sized per part/level) / `counts:N1,N2,...` (clamped per part) / `energy:f1,f2,...` |
| `--target-ms` | — | streaming sizing: derive `stream:<c>` so the first additive chunk downloads in ~this many ms (mutually exclusive with `--breakpoints`) |
| `--bandwidth-mbps` | 25 | assumed downlink for `--target-ms` sizing |
| `--bytes-per-splat` | measured/estimated | override the on-wire bytes/splat for `--target-ms` sizing |
| `--truncation-sigmas` | 3.0 | Mahalanobis cutoff for greedy |
| `--max-n-dense` | 2000 | greedy dense-Gram threshold |

### `luxar gsplat additive <in> <out>` — ladder every leaf of an existing tree
Structure-preserving per-leaf additive laddering: substitutive `kind=lod`
levels, partition parts, and mosaic groups keep their shape; every leaf gains
an additive ladder WITHOUT recomputing the substitutive/partition structure.
The per-leaf counterpart of `lod --recipe additive` (which needs a flat input).
Options: `--n-lods` / `--method` / `--breakpoints` (incl. `stream:C`) /
`--target-ms` / `--bandwidth-mbps` / `--bytes-per-splat` / `--encoding` /
`--compress` / `--overwrite`.
```bash
luxar gsplat additive sub.gsplats.zarr pyr.gsplats.zarr --target-ms 200   # ~200ms first paint per level
```

### Spatial partition (partitioned / multiscale / mosaic)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-elements` | 1,000,000 | per-part splat cap (BSP) |
| `--parts` | none | target part count (sets max_elements = ceil(N/parts)) |
| `--partition-rule` | median | `median` / `midpoint` / `sah` |

### Substitutive reduction (substitutive / pyramid / multiscale cap / mosaic)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--compression-factor` / `-K` | 4 | per-level coarsening factor |
| `--levels` / `-L` | 3 | substitutive coarser levels |
| `--substitutive-method` | auto | `auto` / `kmeans` / `kmeans_lloyd` / `greedy` / `greedy_lloyd` |
| `--lloyd-iters` | 5 | Lloyd refinement passes |
| `--candidate-bins-k` | 12 | Lloyd spatial-hash top-k |
| `--coverage-inflation` | 3.0 | widen merged reps' inter-center spread (mass-preserving) so coarse splats sum flat — suppresses the grid ripple; 1.0 = pure moment match |
| `--coarsen-dims` | all | center-column indices coarsening may merge over (rest = hard barriers) |

### LOD switch tuning (any kind=lod group)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--lod-method` | extent | `extent` (T·W/r) / `count` (√N proxy) |
| `--extent-percentile` | 90 | percentile of per-level radius |
| `--extent-anisotropy` / `--no-extent-anisotropy` | on | use largest principal axis |
| `--base-pixel-size` | none | LOD selector pixel anchor |

### Universal
`--ordering` (hilbert/morton/none, default hilbert), `--device` (auto), `--seed`,
`--overwrite`, `--encoding`/`-e` (auto/precision/memory), `--compress` (zip/tar.gz), `--quiet`/`-q`.
