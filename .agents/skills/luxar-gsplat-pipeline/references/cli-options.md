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
| `--seeds` / `-s` | auto | int = splat count — a WHOLE-VOLUME budget (what a default `cal` reports as K*); a tiled fit divides it across the tiles that survive the resolved floor plus Hann window instead of giving each tile the full count. `fit --tiling uniform` (sequential AND the `-j N` parent, which forwards an exact per-worker count) and `batch-fit` with tile-local reads divide it MASS-WEIGHTED (Hann voxels × `(mean above-floor intensity / ceiling)^--saturation-exponent`, threshold-free; a tile holding >= 1/4 of the equal mass share never gets < 1/4 of the equal seed share), so a busy tile gets more and a dim structured tile is proportional rather than starved. Equal share (`ceil(K/N)`) only where nobody weighed it: a hand-run `--tile k/M`, and a `batch-fit` plan with tile-local reads off. The occupancy scan does not replay optional per-tile denoising, so denoising can still empty a counted tile. `K` below the non-empty count gives 1 per such tile; a tile weighted to zero is skipped. A non-positive int is left alone so the fitter still rejects it. float in (0,1] = compression ratio, scale-free so applied per tile unchanged; `auto`. Ignored under `--tiling content` (budgets come from the density plan). A `cal --auto-region` K* is region-scoped, NOT a whole-volume budget — transfer it via `--cal` + `--tiling content`, not `--seeds` |
| `--floor` | auto | background floor / DC-offset suppression subtracted (clip at 0) BEFORE normalization, so output amplitudes are background-relative. `auto` = histogram-mode estimate (capped at the median; a no-op on clean data). `pNN` = subtract that percentile of non-zero voxels; a plain number = fixed value; `none` or `0` = disable (legacy hard-min). Resolved against the WHOLE volume under any tiling — never against a tile or box crop, so every tile/box works from the same level. `uniform`/`content` resolve it once in the parent and pass the number down; a uniform `-j N` parent that resolves the level while weighing its tiles forwards that concrete level and the raw-input range, while a hand-run `--tile k/M` worker resolves the same spec against the same whole volume. A user numeric floor on that hand-run worker is guarded against the whole-volume maximum just as it is without tiling; a parent-forwarded level is marked resolved and applied verbatim |
| `--iters` / `-n` | preset | max optimization iterations |
| `--preset` | none | `draft` / `standard` / `hifi` / `ultra` / `n2s` (see preset table) |
| `--config` | none | YAML config file (overrides preset) |
| `--dump-config` | false | print resolved default config and exit |
| `--device` / `-d` | auto | `auto` / `cpu` / `cuda` / `mps` |
| `--loss` | l1 | `l1` / `mse` / `poisson` — `poisson` is the matching noise model for shot-noise-dominated photon counts (see `packages/luxar/src/luxar/gsplats/GLOSSARY.md`) |
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
| `--tile-size` | 256 | tile edge in voxels. One uniform grid everywhere (#2838): a trailing sliver thinner than the overlap is FOLDED into its predecessor, which then spans up to `tile_size + overlap - 1` voxels (287 at 256/32 = 1.41x the voxels in 3D). Size this for that worst case — it is the GPU-memory knob |
| `--overlap` | 32 | inter-tile overlap voxels (Hann-stitched) |
| `--tile` | none | fit a single tile `N/M` (e.g. `3/16`) — Slurm-ready |
| `--jobs` / `-j` | 1 | concurrent tiles on one GPU; `auto` accounts for GPU memory plus shared host RAM/CPU limits |
| `--keep-tiles` | false | keep per-tile temp outputs after merge |

### Content-adaptive tiling (`--tiling content`) — needs a density
| Flag | Default | Meaning |
| --- | --- | --- |
| `--cal` | none | calibration JSON from `gsplat cal` (supplies density) |
| `--k-star-ref` | — | reference K* (effective splats) |
| `--n-features-ref` | — | reference feature count |
| `--saturation-exponent` | 0.44 | K ~ features^alpha. NOT content-only: it also shapes the mass-weighted split of an integer `--seeds` budget across UNIFORM tiles (between equal-size tiles, budgets stand in the ratio of their intensity masses to this power) |
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
| `--cull-retention` | 0.95 | keep top fraction of cumulative amplitude (presets set 0.999, and so does `--tiling content` without one); 0 keeps every splat |
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

**With NO `--preset`, `fit` runs 1000 iterations — below `draft`.** `--preset` has no
default, and `load_fit_config` layers one only `if preset is not None`, so a bare
`luxar gsplat fit` falls through to the *function* defaults: `n_iters=1000`,
`early_stop_patience=300`, `patience=15`, `max_eccentricity=10.0`,
`cull_retention=0.95`, `enable_dynamic_ops=True`. Verified by resolving the config:

    load_fit_config(None, None, {})["n_iters"] == 1000     # bare CLI fit
    load_fit_config("draft", None, {})["n_iters"] == 2000

One exception to that fall-through: `--tiling content` layers a *command default* of
`cull_retention=0.999` between the preset and the function defaults, so a preset-less
content fit is near-lossless per box. `--preset`, a `cull_retention:` in a `--config`
and `--cull-retention` all still win. Nothing else is special-cased: a bare content fit
still runs 1000 iterations, and a bare `--tiling uniform` still culls each tile at
`0.95` even though its default partition merge is per-tile too.

The Python API (`fit_gaussian_splats`, which has no `preset=` argument) shares those
same defaults. Either way, a fifth of `standard` on thin filaments leaves splats at
their isotropic σ=1-voxel seed shape and renders axons as bead chains. Treat a bare
`fit` as a preview; pass `--preset`/`--iters` for anything real. See "BOTH entry
points default to 1000 iters" in `SKILL.md`.

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
| `--k-star-metric` | psnr_minmax | `psnr_minmax` / `psnr_foreground` / `psnr_fg_weighted` / `gain`; prefer `psnr_fg_weighted` for sparse or deconvolved volumes. `gain` is diagnostic and selects identically to `psnr_minmax` because its baseline is constant across K |
| `--fg-bg-ratio` | 1.0 | foreground:background total-weight ratio for `psnr_fg_weighted`; 1 gives equal total weight over the held-out subset |
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
`overview`/`adaptive` compose the two over spatial tiles. Scale is not the only
axis: when first paint is request-constrained, the eager-rung count picks the
recipe instead — see "First paint cost" in SKILL.md.

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
| `--reveal-center` | dataset bbox centre | `-m radial` only: comma-separated shell centre, one coordinate per measured axis. On a partitioned recipe the default centres each part on itself — pass this to grow the whole object from one point |
| `--spatial-dims` | non-degenerate axes | `-m radial` only: comma-separated centre-column indices the shell distance spans (order pairs with `--reveal-center`); the default keeps a stacked time/channel axis out of the shells |

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
Auto-derived, no knob, and COUNT-INDEPENDENT: thresholds come from SCREEN-AREA
occupancy halving and are stamped `selector="screen-area"`. Each
`coverage_fraction` is a literal screen-area fraction (projected bbox rect area /
viewport area): the coarsest child gets `0.0` (always-eligible floor), the finest
gets `0.5` — so a WHOLE-OBJECT ladder holds full detail while the object occupies
at least half the screen — and each level between halves once more
(…, 1/8, 1/4, 1/2). Element counts are read only for the ladder's LENGTH.

The metric is an NDC-area fraction, so it is RESOLUTION-independent — the same
framing reads the same fraction on any monitor size, and the projected rect is
clipped to the viewport first so it tops out at exactly 1.0. It is **not**
aspect-independent: `fov` is vertical, so a wider viewport shows more world
horizontally and the same object covers a smaller area fraction. Resizing between
square and ultrawide does move the switch points.

The retired `sqrt(N_i/N_finest)` derivation was a diagonal metric spaced by a
count ratio; it held the most expensive level across nearly the whole usable zoom
range on dense additive data, which is why the anchor is now occupancy, not count.

The `adaptive` and `overview` recipes are the exception: their ladders are bound
to a spatial partition, so they keep the FILLS-SCREEN anchor
(`partitioned_coverage_fractions`, finest = area 1.0 = the tile alone fills the
screen). For `adaptive` that is geometry — each lod group's bbox is one BSP tile,
so it projects to a fraction of the whole object. For `overview` it is the
recipe's contract: the coarse cap is what you see at the opening framing and the
fine partition is the zoom-in branch, so it deliberately does NOT show full detail
at a normal full-frame view. Use `levels` if you want that.

Legacy stores and explicit `coverage_fractions=[...]` lists keep the older
`selector="coverage"` diagonal metric (thresholds in `[0, 4]`); the viewer reads
both. The former `extent`/`count` methods and the `--lod-method` /
`--extent-percentile` / `--extent-anisotropy` / `--base-pixel-size` flags have
been removed.

### Quality stamps
| Flag | Default | Meaning |
| --- | --- | --- |
| `--quality-stamps` / `--no-quality-stamps` | on | measure each coarse substitutive level's mixture-L² quality `Q` vs its group's finest content and stamp it (with the reference-energy weight `w`) into the level stats — the viewer folds `Q` with committed-energy `e(k)` into a recursive quality estimate |
| `--quality-max-pair-splats` | 2,000,000 | subsample cap per mixture for the quality measurement (lower = faster, noisier `Q`) |

### Universal
`--ordering` (hilbert/morton/none, default hilbert), `--device` (auto), `--seed`,
`--overwrite`, `--encoding`/`-e` (auto/precision/memory), `--compress` (zip/tar.gz), `--quiet`/`-q`.
