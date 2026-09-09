# Gaussian Splat Calibration Package

Blind-spot cross-validation for principled Gaussian-splat model selection (Luxar manuscript Supp. Doc. 2, `splat_count_vs_quality`).

## Purpose

This package implements the Noise2Self-based calibration protocol that finds the optimal splat count K* for a dataset: sweep K, fit each against a donut-median-filled volume (held-out 5% mask; each held-out voxel is replaced by the median of its *unmasked* donut neighbours, so the fitter never sees a held-out value directly or through a neighbour's fill), measure held-out PSNR against the original unmasked values, and pick the K that maximises held-out quality. Capacity beyond K* memorises noise rather than signal.

As a free byproduct, an ensemble noise-floor estimator (Laplacian + Haar HH + background MAD) places each dataset in absolute terms (PSNR ceiling).

## Module Map

Split by phase:

- **`masking.py`** — Bernoulli mask generation (`cv_mask`) and donut-median self-supervision fill (`donut_median_fill`; held-out donors are excluded, clustered masks expand the radius, and an all-held-out volume is rejected)
- **`metrics.py`** — Held-out reconstruction metrics (`held_out_psnr`, `held_out_gain_db`, `held_out_psnr_foreground`, `held_out_psnr_fg_weighted`, `predict_zero_baseline_mse`)
- **`noise_floor.py`** — Ensemble noise-floor estimation (`estimate_noise_floor` → `NoiseFloor`) and DC-offset estimation (`estimate_floor`)
- **`content.py`** — Feature content estimation (local-maxima / edges / intensity counts) for splat-density prediction; Otsu and lightly smoothed foreground masks; auto region selection
- **`curve_analysis.py`** — K-grid construction (`build_k_grid`), peak detection (`find_k_star` → `HeldOutPeak`), transferable splat-density model (`SplatDensity`, `fit_saturation_exponent` → `ExponentFit`), parametric rate-distortion model (`fit_rd_model` → `RDModel`)
- **`result.py`** — `CalibrationResult` container with JSON serialisation (non-finite floats round-trip as `null`)
- **`driver.py`** — Top-level `calibrate` driver (wires together mask → fit → metrics → curve analysis) and `calibrate_saturation_exponent` (multi-scale α fit)
- **`__init__.py`** — Re-exports the public surface so `luxar.gsplats.calibration.X` resolves

Supporting CLI report (lazy-imported, matplotlib-free when not used):
- **`../calibration_report.py`** — Optional multi-page PDF report renderer for `luxar gsplat cal --pdf` (rate-distortion curves, blind-spot CV plot, reconstruction montages)

## Public API

### Core Functions

```python
from pathlib import Path

from luxar.gsplats.calibration import (
    calibrate,
    build_k_grid,
    cv_mask,
    donut_median_fill,
    CalibrationResult,
    estimate_noise_floor,
    estimate_floor,
    find_k_star,
    count_features,
    select_calibration_region,
)

# Build a K-grid (manuscript default: 10 log-spaced points, 1K–512K)
ks = build_k_grid(n_points=10, k_min=1_000, k_max=512_000)

# Run the calibration sweep
result = calibrate(
    volume,
    k_grid=ks,
    fit_kwargs={"device": "cuda"},  # forwarded to fit_gaussian_splats
)

print(f"Recommended K* = {result.held_out_peak.k_star:,}")
print(f"Curve type     = {result.held_out_peak.type}")  # peak | plateau | signal_limited
print(f"Noise floor σ̂ = {result.noise_floor.sigma_hat:.4f}")
print(f"PSNR ceiling   = {result.noise_floor.psnr_max_db:.1f} dB")

# Persist or reload
result.to_json(Path("cal.json"))
loaded = CalibrationResult.from_json(Path("cal.json"))
```

### Result Container

`CalibrationResult` carries the full sweep:
- K values (requested + post-cull effective)
- Per-K metrics: held-out PSNR/MSE, train PSNR, full PSNR/SSIM
- `held_out_peak` (`HeldOutPeak`): recommended `k_star`, curve `type`, `confidence_db`, operating point `k_knee`, supporting metadata
- `noise_floor` (`NoiseFloor`): ensemble `sigma_hat`, component estimators, `psnr_max_db`
- Optional regime-robust extensions: `held_out_psnr_fg_db`, `held_out_psnr_fg_weighted_db`, `foreground_mask_fraction`, `foreground_otsu_threshold`, `fg_bg_ratio`, `held_out_gain_db`, `predict_zero_baseline_mse`, `calibration_region`, `splat_density`, `rd_model`, `exponent_fit`
- Fitting provenance: `fit_times_seconds`, `fit_config`, `volume_shape`, `volume_dtype`, `timestamp`, optional `splat_paths` (when `--keep-fits`)
- JSON (de)serialisation: non-finite floats round-trip as `null` → `nan`/`inf`; additive fields hydrate with defaults so old `cal.json` files keep loading

### Primitives

```python
# Blind-spot masking
mask = cv_mask(shape=(128, 128, 128), fraction=0.05, seed=42)
V_filled = donut_median_fill(V, mask, radius=1)

# Noise floor (three-estimator ensemble)
nf = estimate_noise_floor(V)
print(f"Sigma: Laplacian={nf.sigma_laplacian:.5f}, Haar={nf.sigma_haar:.5f}, "
      f"Background={nf.sigma_background:.5f}, Ensemble={nf.sigma_hat:.5f}")

# DC offset / pedestal subtraction (applied ONCE before fitting)
floor = estimate_floor(V, method="mode")  # or "percentile"
V_suppressed = np.clip(V - floor, 0.0, None)

# Feature content estimation (drives density model + planner)
n_features = count_features(V, method="peaks")  # or "edges", "intensity"
```

## Minimal Usage Example

```python
import numpy as np
from luxar.gsplats.calibration import calibrate, build_k_grid

# Synthetic volume
volume = np.random.rand(64, 64, 64).astype(np.float32)

# Quick calibration (5-point sweep for speed)
ks = build_k_grid(n_points=5, k_min=1_000, k_max=64_000)
result = calibrate(volume, k_grid=ks, fit_kwargs={"device": "cpu", "n_iters": 50})

# Inspect result
print(f"K* = {result.held_out_peak.k_star} ({result.held_out_peak.type})")
print(f"Noise floor σ̂ = {result.noise_floor.sigma_hat:.5f}")

# Refit at K* (external call, NOT part of this package)
# from luxar.gsplats import fit_gaussian_splats
# final = fit_gaussian_splats(volume, seeds=result.held_out_peak.k_star)
```

## CLI Integration

```bash
# Basic calibration (10-point default sweep)
luxar gsplat cal volume.tiff cal.json

# Fast preview (5 points)
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000

# Explicit K grid
luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'

# Multi-page PDF report
luxar gsplat cal volume.tiff cal.json --pdf cal_report.pdf

# With reconstruction montages (requires --keep-fits)
luxar gsplat cal volume.tiff cal.json --pdf report.pdf --keep-fits fits/
```

After calibration, re-fit at K*: `luxar gsplat fit volume.zarr out.zarr --seeds <K*>`.

## Invariants & Gotchas

### Background Floor Subtraction

Calibration applies `--floor` (default `"auto"`) **once** to the volume before masking, so the fit target, render reference, AND held-out truth are all on the same floor-suppressed scale. A per-fit floor would mismatch the raw held-out reference (tanking PSNR). K* is thus measured the way you will fit.

**Implementation**: `driver.calibrate` calls `_resolve_floor` once, subtracts it (clipping at 0), then passes `floor="none"` to every per-K fit.

**Edge case**: A floor ≥ `volume.max()` would clip the whole volume to zero (non-finite held-out PSNR); the driver emits a warning, ignores the floor (sets it to `None`), and proceeds un-floored (only an explicit too-high float/percentile can trigger; `"auto"` is capped at the median).

### Held-Out PSNR Metric Regimes

- **Default (`psnr_minmax`)**: `V.max() - V.min()` range held-out PSNR — correct for raw/noisy data at manageable scale (the manuscript regime).
- **Background-dominated**: On large sparse volumes a trivial predict-zero reconstruction already scores 40-60 dB (most masked voxels are background ~0). Regime-robust alternatives:
  - `held_out_gain_db`: dB improvement over the all-zeros baseline — plateaus meaningfully (not inflated by trivially-reconstructed background).
  - `held_out_psnr_foreground`: Held-out PSNR restricted to voxels that are both held out AND foreground (Otsu thresholded).
  - `held_out_psnr_fg_weighted_db`: lightly-smoothed Otsu foreground plus controlled background weight; prefer this for sparse or deconvolved volumes. `fg_bg_ratio=1` gives equal total foreground/background weight over held-out voxels.

`held_out_gain_db` differs from `psnr_minmax` only by a constant because its predict-zero baseline does not vary with K, so it reports a useful diagnostic scale but selects the same K*.

**Default K* selection**: Still uses `psnr_minmax` (additive). The alternatives are recorded in `CalibrationResult` fields (`held_out_gain_db`, `held_out_psnr_fg_db`, `held_out_psnr_fg_weighted_db`, `foreground_mask_fraction`, `foreground_otsu_threshold`, `fg_bg_ratio`, `predict_zero_baseline_mse`, `k_star_metric`, `held_out_peak_selected`) but not used by default.

### Peak Detection Hybrid Rule

`find_k_star` (manuscript §4.2) uses a hybrid 3-regime detector:

1. **Peak**: argmax is strictly interior AND each flank's mean is ≥ 0.1 dB below the peak → return the argmax.
2. **Signal-limited**: argmax at the last K, curve rose ≥ 0.3 dB across sweep, AND still climbing at the top (mean rise over trailing run ≥ 0.1 dB/step AND final step ≥ 0.05 dB) → return the last K (budget anchor). Tail checks stop a flat-topped plateau from being misread as signal-limited.
3. **Plateau**: otherwise → return the smallest K within 0.3 dB of the maximum (onset of diminishing returns).

**Operating point fields**:
- `k_star`: Budget anchor (the regime-specific detector above).
- `k_knee`: Point of diminishing returns — the argmax for a clear peak, else the smallest K within 0.3 dB of max. Decoupled from regime label; equals `k_star` for peak/plateau, earlier knee for signal-limited. Use this when a single "reasonable operating point" is wanted regardless of regime.

**Backward compatibility**: Older `cal.json` files (pre-`k_knee`) hydrate with `k_knee` defaulting to `k_star`.

### Input Requirements

`calibrate` is a single-pass sweep driver over a raw volume: it expects a NumPy array and only guards that `V.ndim >= 2` (raising `ValueError` otherwise) and that `k_grid` is non-empty. It operates upstream of LOD construction — you pass a volume, not an already-fitted `GSplatData`.

## Tests

All tests colocated in `packages/luxar/src/luxar/gsplats/tests/test_calibration.py`:

- `cv_mask`: Determinism, fraction Binomial CI, shape/dtype, invalid bounds
- `donut_median_fill`: Constant volume unchanged, unmasked voxels untouched, gradient volume local average, 2D/3D/4D shape correctness, empty mask early return, radius-bounds validation, invariance to the values at held-out positions, masked-neighbour exclusion, clustered-mask radius expansion, genuine-NaN propagation, bounded-chunk equivalence, all-masked error
- Noise floor estimators: Laplacian/Haar/background MAD on constant/noisy volumes
- `estimate_floor`: Mode histogram vs percentile, zero-padding exclusion, high-offset float32 volumes (a background band narrower than float32 spacing) and a degenerate single-value band
- `build_k_grid`: Exponential/polynomial spacing, endpoint pinning, explicit-grid override & precedence, input validation (bad progression, too few points, `k_max ≤ k_min`)
- `find_k_star`: Peak/plateau/signal-limited regime detection, flank thresholds, tail-rise gate, `k_knee` vs `k_star`, backward-compatible hydration
- Feature content: `count_features` (peaks/edges/intensity), `feature_threshold` shared scale, `select_calibration_region` (`densest`, small-volume `whole` fallback, hot-outlier robustness, unknown-strategy rejection)
- `CalibrationResult`: JSON round-trip (non-finite floats → `null`), additive field defaults
- Full sweep integration: Calls `fit_gaussian_splats` → `render_to_volume_tensor` → held-out/train/full metrics → noise floor → `find_k_star` → `CalibrationResult` with all metadata

Run via: `hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_calibration.py -v`

## See Also

- **`luxar.gsplats.fit_gaussian_splats`** — The single-pass fitter function `calibrate` calls at each K (implemented in `fit_gsplats.py`)
- **`luxar.gsplats.lod`** — LOD post-processing (downstream of calibration; `cal → fit → lod` pipeline)
- **`luxar.gsplats.planner`** — Content-adaptive tiled fitting (consumes the `SplatDensity` model from calibration)
- **`luxar.gsplats.rendering`** — GPU-accelerated volume rendering for metric computation
- **`luxar.gsplats.metrics`** — PSNR/SSIM/MSE on GPU tensors (re-used by `calibrate` driver)
- **Parent package** — `luxar.gsplats.README.md` (§ Calibration)
- **Manuscript** — Supp. Doc. 2: `splat_count_vs_quality` protocol
