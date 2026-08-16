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
| `--seeds` / `-s` | auto | int = splat count — a WHOLE-VOLUME budget (what a default `cal` reports as K*); a tiled fit divides it across its N tiles (`ceil(K/N)`, floored at 1) instead of giving each tile the full count, so the total tracks the request, not the tile count. Not exact: near-zero tiles are skipped, and `K < N` gives N. A non-positive int is left alone so the fitter still rejects it. float in (0,1] = compression ratio, scale-free so applied per tile unchanged; `auto`. Ignored under `--tiling content` (budgets come from the density plan). A `cal --auto-region` K* is region-scoped, NOT a whole-volume budget — transfer it via `--cal` + `--tiling content`, not `--seeds` |
| `--floor` | auto | background floor / DC-offset suppression subtracted (clip at 0) BEFORE normalization, so output amplitudes are background-relative. `auto` = histogram-mode estimate (capped at the median; a no-op on clean data). `pNN` = subtract that percentile; a plain number = fixed value; `none` = disable (legacy hard-min). Resolved against the WHOLE volume under any tiling — never against a tile or box crop, so every tile/box works from the same level. `uniform`/`content` resolve it once in the parent and pass the number down; the uniform `-j N` / `--tile k/M` workers each resolve the same spec against the same whole volume (the sampler is deterministic, so they agree) |
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
`--recipe`/`-r` `stream` → `tiles` topology; `--recipe levels` → `adaptive`.
Knobs mirror `lod`: `--n-lods`, `--add-method`/`-m`, `--breakpoints`/`-b`,
`--target-ms`, `--bandwidth-mbps` (default 25), `--bytes-per-splat`,
`--compression-factor`/`-K`, `--levels`/`-L`, `--subst-method`,
`--coarsen-dims`, `--refine`, `--refine-iters`. LOD switch thresholds are
auto-derived (`coverage_fraction`, no knob — see "LOD switch tuning" below).
`--refine volume` needs no `--target` here (the volume being fitted is already
in hand) and re-fits each tile against its own crop of it; incompatible with
`--downscale` (the tile grid and the rescaled splats would be in different
coordinate frames).

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
| `--cull-retention` | 0.95 | keep top fraction of cumulative amplitude (presets set 0.999); 0 keeps every splat |
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

**The Python API defaults are BELOW `draft`.** `fit_gaussian_splats` (no `preset=`
argument) defaults to `n_iters=1000`, `early_stop_patience=300`, `patience=15`,
`max_eccentricity=10.0`, `cull_retention=0.95`, `enable_dynamic_ops=True`. A caller
who omits `n_iters` gets a fifth of `standard`, which on thin filaments leaves splats
at their isotropic σ=1-voxel seed shape and renders axons as bead chains. See the
"API defaults are BELOW draft" section in `SKILL.md`.

### Shape-related knobs not exposed as `fit` flags (Python / `--config` YAML only)
| Knob | Default | Meaning |
| --- | --- | --- |
| `patience` | 15 | plateau LR-decay patience; at 15 the shape LR decays away long before shapes settle |
| `enable_dynamic_ops` | True | periodic splat relocation — **resets relocated splats to isotropic σ=0.5 with off-diagonals zeroed**, undoing elongation a long fit earned |
| `l1_diag` | auto (0.01·lr) | L1 on the Cholesky diagonal; its own comment says it "encourages smaller, more isotropic splats" |
| `sigma_min_diag` | sqrt(1/12) ≈ 0.289 | per-axis floor on the Cholesky diagonal |

`max_eccentricity` caps the axis ratio at `sqrt(max_eccentricity)` (so 10.0 → 3.16),
enforced as a hard clamp on the Cholesky diagonal AND off-diagonals. It is inert on a
short fit that never elongates that far, and only starts binding once the fit is
converged — measured at 14.9% of splats pinned to the ceiling, costing 2.65 dB.

---

## `luxar gsplat cal INPUT OUTPUT_JSON`

Blind-spot (Noise2Self) K-sweep → K*, `k_knee` operating point (diminishing-returns
elbow, ≤ K*), curve type, noise floor, PSNR ceiling. Also writes a `splat_density`
block (saturation exponent alpha etc.) consumed by `fit --tiling content`.

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

`cal` also honours `--floor` (default `auto`, same semantics as `fit`): the floor
is subtracted ONCE up front so K* is measured on floor-suppressed data (matching
how you fit). Pass `--floor none` to reproduce the legacy hard-min numbers.

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

`--recipe` is REQUIRED. Recipes scale-ordered by element count N:
`flat` < `stream` < `levels` < `tiles` < `overview` < `adaptive`. `stream` is
additive (refines one leaf); `levels` is substitutive (coarse↔fine swap);
`overview`/`adaptive` compose the two over spatial tiles.

### Additive ladder (stream / tiles / overview / levels)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-lods` | 4 | additive LOD levels |
| `--add-method` / `-m` | auto | `auto` (greedy at N ≤ 5000, else self_energy) / `greedy` / `self_energy` / `mass` / `amplitude` / `spectral` / `random` / `radial` (concentric-shell reveal from the bbox centre; no energy stamps) |
| `--breakpoints` / `-b` | equal-count | `equal-count` / `stream:C` (geometric streaming ladder, first chunk C splats then doubling; sized per part/level) / `counts:N1,N2,...` (clamped per part) / `energy:f1,f2,...` |
| `--target-ms` | — | streaming sizing: derive `stream:<c>` so the first additive chunk downloads in ~this many ms (mutually exclusive with `--breakpoints`) |
| `--bandwidth-mbps` | 25 | assumed downlink for `--target-ms` sizing |
| `--bytes-per-splat` | measured/estimated | override the on-wire bytes/splat for `--target-ms` sizing |
| `--truncation-sigmas` | the dataset's own truncation radius | Mahalanobis cutoff for greedy |
| `--max-n-dense` | 2000 | greedy dense-Gram threshold |
| `--reveal-centre` | dataset bbox centre | `-m radial` only: comma-separated shell centre, one coordinate per measured axis. On a partitioned recipe the default centres each part on itself — pass this to grow the whole object from one point |
| `--spatial-dims` | non-degenerate axes | `-m radial` only: comma-separated centre-column indices the shell distance spans (order pairs with `--reveal-centre`); the default keeps a stacked time/channel axis out of the shells |

### `luxar gsplat additive <in> <out>` — ladder every leaf of an existing tree
Structure-preserving per-leaf additive laddering: substitutive `kind=lod`
levels, partition parts, and adaptive groups keep their shape; every leaf gains
an additive ladder WITHOUT recomputing the substitutive/partition structure.
The per-leaf counterpart of `lod --recipe stream` (which needs a flat input).
Options: `--n-lods` / `--add-method` / `--breakpoints` (incl. `stream:C`) /
`--target-ms` / `--bandwidth-mbps` / `--bytes-per-splat` / `--encoding` /
`--compress` / `--overwrite`.
```bash
luxar gsplat additive sub.gsplats.zarr pyr.gsplats.zarr --target-ms 200   # ~200ms first paint per level
```

### Spatial partition (tiles / overview / adaptive)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--max-elements` | 1,000,000 | per-part splat cap (BSP) |
| `--parts` | none | target part count (sets max_elements = ceil(N/parts)) |
| `--partition-rule` | median | `median` / `midpoint` / `sah` |

### Substitutive reduction (levels / overview cap / adaptive)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--compression-factor` / `-K` | 4 | per-level coarsening factor |
| `--levels` / `-L` | 3 | substitutive coarser levels |
| `--subst-method` | auto | `auto` / `kmeans` / `kmeans_lloyd` / `greedy` / `greedy_lloyd` |
| `--lloyd-iters` | 5 | Lloyd refinement passes |
| `--candidate-bins-k` | 12 | Lloyd spatial-hash top-k |
| `--coverage-inflation` | 3.0 | widen merged reps' inter-center spread (mass-preserving) so coarse splats sum flat — suppresses the grid ripple; 1.0 = pure moment match |
| `--additive` / `--no-additive` | on | additive ladder in every substitutive level / adaptive tile / overview cap (streaming first paint); `--no-additive` = bare leaves |
| `--conserve-mass` / `--no-conserve-mass` | on | pin each level's mass over coarsened dims to its fine input (per barrier group) — kills the LOD brightness pop |
| `--refine` | none | `l2` = post-merge L2 refit of each level against its fine input (slower, higher fidelity, peak-preserving; mass pinned); `volume` = warm-start re-fit against the source volume given via `--target` (highest fidelity; never worse than the merge; works on levels/overview and per-tile `adaptive`, with or without barrier dims — each re-fit gets the sub-volume it owns, and a per-tile re-fit that leaves its tile is discarded) |
| `--refine-iters` | 120 / 300 | steps per refined level (120 for `l2`, 300 for `volume`; requires `--refine l2\|volume`) |
| `--target` | — | source volume for `--refine volume` (.npy/.npz/.tiff/.zarr[.zip]; with `--channel`/`--timepoint`/`--array-key` selectors) |
| `--target-axes` | — | per-dimension labels for a `--target` that KEEPS its stacked axis (e.g. `time,z,y,x`), so `--refine volume` walks it one slice per barrier group; the target is then opened lazily (only the slice is read). Contrast `--timepoint`, which slices a single timepoint out and drops the axis — mutually exclusive with this. |
| `--coarsen-dims` | all | center-column indices coarsening may merge over (rest = hard barriers) |

### LOD switch tuning (any kind=lod group)
Auto-derived, no knob: each child's `coverage_fraction` = `sqrt(N_i / N_finest)`
(a viewport-relative value; coarsest 0.0, finest 1.0). The viewer multiplies it by
half of the live viewport's fitted screen axis (the smaller of its width/height),
so the finest level shows at any normal full-frame view (projected size ≳ half
the fitted screen axis) and coarser levels step in as it shrinks below that —
self-calibrating on any monitor or aspect ratio.

The `adaptive` and `overview` recipes are the exception: their ladders are bound
to a spatial partition (a tile projects to a fraction of the whole object), so
they are scaled to anchor the finest at `4.0` = `SCREEN_FILL_DIAGONAL_RATIO /
FILL_FACTOR` — approximately the switch point a screen-filling tile needs (exact
only near aspect ratio sqrt(3) ~= 1.73; the real screen-filling metric ranges
~2.8 at 1:1 to ~7.4 at an ultrawide 32:9 — see `lod-group-registry.ts`'s
`FILL_FACTOR` doc) — and what `overview`'s "coarse overview, fine tiles on zoom"
means. An explicit `coverage_fractions=[...]` list may use the same `[0, 4]`
range. The former `extent`/`count` methods and the
`--lod-method` / `--extent-percentile` / `--extent-anisotropy` /
`--base-pixel-size` flags have been removed.

### Quality stamps
| Flag | Default | Meaning |
| --- | --- | --- |
| `--quality-stamps` / `--no-quality-stamps` | on | measure each coarse substitutive level's mixture-L² quality `Q` vs its group's finest content and stamp it (with the reference-energy weight `w`) into the level stats — the viewer folds `Q` with committed-energy `e(k)` into a recursive quality estimate |
| `--quality-max-pair-splats` | 2,000,000 | subsample cap per mixture for the quality measurement (lower = faster, noisier `Q`) |

### Universal
`--ordering` (hilbert/morton/none, default hilbert), `--device` (auto), `--seed`,
`--overwrite`, `--encoding`/`-e` (auto/precision/memory), `--compress` (zip/tar.gz), `--quiet`/`-q`.
