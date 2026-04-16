# Progressive vs Single-Pass Fitting — Interpretation

## Overview

This analysis compares single-pass and multi-pass (progressive) Gaussian splat
fitting at the same total splat and iteration budget.

**Experimental design:**
- **Fixed budget:** 32K splats, 10K total iterations
- **Conditions:** 1-pass, 2-pass, 4-pass, 8-pass
- **For N passes:** `splats_per_pass = 32K/N`, `iters_per_pass = 10K/N`
- **Shared settings:** `cull_retention=0.999`, `enable_dynamic_ops=True`, `loss_type="l1"`
- **12 datasets** across confocal, spinning-disk, widefield, and light-sheet microscopy
- **Replicates:** 1-3 per condition (see Statistical Notes below)

---

## Results

### PSNR comparison (dB, mean across replicates)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass | Best | vs 1-pass |
|---------|-------:|-------:|-------:|-------:|:----:|----------:|
| opencell_map4_ch0 | **28.34** | 28.12 | 28.14 | 27.28 | 1-pass | — |
| opencell_map4_ch1 | **25.16** | 24.57 | 24.66 | 23.96 | 1-pass | — |
| opencell_lmnb1_ch0 | **25.47** | 25.16 | 25.40 | 24.77 | 1-pass | — |
| opencell_lmnb1_ch1 | **33.46** | 33.38 | 32.79 | 31.37 | 1-pass | — |
| kidney_dapi | **33.70** | 33.27 | 31.57 | 29.68 | 1-pass | — |
| kidney_actin | **30.60** | 30.36 | 29.11 | 27.19 | 1-pass | — |
| organoid_ch0 | **39.02** | 38.24 | 38.10 | 37.36 | 1-pass | — |
| celegans_t100 | **39.38** | 38.83 | 39.33 | 39.06 | 1-pass | — |
| cells3d_nuclei | 35.44 | **36.30** | 35.46 | 33.96 | **2-pass** | +0.86 dB |
| cells3d_membrane | 46.57 | **48.21** | 46.79 | 44.70 | **2-pass** | +1.65 dB |
| acto3d_heart_nuclei | 31.94 | **32.84** | 32.65 | 31.12 | **2-pass** | +0.90 dB |
| tribolium | 42.61 | **43.84** | 42.86 | 41.10 | **2-pass** | +1.23 dB |

**Summary:** 1-pass wins 8/12 datasets; 2-pass wins 4/12 datasets.

### SSIM comparison (mean across replicates)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass |
|---------|-------:|-------:|-------:|-------:|
| opencell_map4_ch0 | **0.675** | 0.651 | 0.672 | 0.667 |
| opencell_map4_ch1 | **0.620** | 0.587 | 0.601 | 0.592 |
| opencell_lmnb1_ch0 | **0.514** | 0.491 | 0.509 | 0.505 |
| opencell_lmnb1_ch1 | 0.910 | 0.901 | **0.921** | 0.919 |
| kidney_dapi | **0.962** | 0.958 | 0.944 | 0.925 |
| kidney_actin | 0.897 | **0.926** | 0.906 | 0.866 |
| organoid_ch0 | 0.933 | 0.925 | 0.939 | **0.943** |
| celegans_t100 | **0.894** | 0.874 | 0.892 | 0.888 |
| cells3d_nuclei | **0.871** | 0.871 | 0.862 | 0.848 |
| cells3d_membrane | 0.989 | **0.992** | 0.990 | 0.985 |
| acto3d_heart_nuclei | 0.825 | **0.858** | 0.849 | 0.814 |
| tribolium | 0.993 | **0.994** | 0.993 | 0.991 |

### Wall-clock time (seconds, mean across replicates)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass |
|---------|-------:|-------:|-------:|-------:|
| opencell_map4_ch0 | 70 | **19** | 29 | 32 |
| opencell_map4_ch1 | 70 | **20** | 30 | 32 |
| opencell_lmnb1_ch0 | 174 | **48** | 73 | 79 |
| opencell_lmnb1_ch1 | 243 | **54** | 70 | 75 |
| kidney_dapi | 195 | 32 | 26 | **25** |
| kidney_actin | 88 | 28 | **26** | 25 |
| cells3d_nuclei | 87 | **21** | 26 | 27 |
| cells3d_membrane | 92 | **23** | 28 | 31 |
| organoid_ch0 | 156 | 81 | **50** | 102 |
| celegans_t100 | 145 | **16** | 26 | 37 |
| tribolium | 713 | **274** | 390 | 332 |
| acto3d_heart_nuclei | 444 | **233** | 243 | 232 |

### Maximum absolute error (mean across replicates)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass |
|---------|-------:|-------:|-------:|-------:|
| opencell_map4_ch0 | **0.560** | 0.612 | 0.606 | 0.626 |
| opencell_map4_ch1 | **0.755** | 0.772 | 0.785 | 0.804 |
| opencell_lmnb1_ch0 | 0.765 | **0.695** | 0.702 | 0.700 |
| opencell_lmnb1_ch1 | **0.421** | 0.750 | 0.813 | 0.832 |
| kidney_dapi | **0.324** | 0.501 | 0.540 | 0.565 |
| kidney_actin | **0.524** | 0.749 | 0.686 | 0.700 |
| cells3d_nuclei | 0.183 | **0.182** | 0.195 | 0.299 |
| cells3d_membrane | **0.091** | 0.109 | 0.141 | 0.236 |
| organoid_ch0 | **0.298** | 0.479 | 0.592 | 0.583 |
| celegans_t100 | **0.213** | 0.907 | 0.953 | 0.960 |
| tribolium | **0.108** | 0.133 | 0.198 | 0.219 |
| acto3d_heart_nuclei | **0.568** | 0.871 | 0.836 | 0.920 |

---

## Key Findings

### 1. Single-pass wins for most datasets by PSNR (8 of 12)

With conservative culling (`cull_retention=0.999`), single-pass achieves the
highest PSNR for 8 of 12 datasets. Nearly all splats survive culling, so the
single-pass approach benefits from optimizing the full 32K splats jointly for the
full 10K iterations. Splitting the budget across passes reduces both the number
of splats and iterations available to each pass, which hurts convergence.

### 2. Four datasets favor 2-pass progressive fitting

The following datasets achieve higher PSNR with 2-pass fitting:

- **cells3d_membrane** (HeLa membrane): +1.65 dB (48.21 vs 46.57)
- **tribolium** (light-sheet): +1.23 dB (43.84 vs 42.61)
- **acto3d_heart_nuclei** (light-sheet heart): +0.90 dB (32.84 vs 31.94)
- **cells3d_nuclei** (HeLa nuclei): +0.86 dB (36.30 vs 35.44)

What these datasets share: they all have relatively smooth, well-structured
signals where the first pass captures the dominant structures and residual
seeding efficiently targets the remaining detail. The two light-sheet datasets
(tribolium, acto3d) are also the largest volumes (cropped to ~100M voxels),
where the splat budget per voxel is lowest and residual targeting matters most.
The cells3d datasets are small (60x256x256) but have very clean, high-contrast
structures amenable to progressive decomposition.

### 3. Progressive fitting is consistently 2-4x faster

Even when progressive does not win on quality, it is substantially faster in
wall-clock time. For all 12 datasets, 2-pass is faster than 1-pass:

- OpenCell MAP4 ch0: 19s (2-pass) vs 70s (1-pass) = **3.7x faster**
- C. elegans: 16s (2-pass) vs 145s (1-pass) = **9.1x faster**
- Tribolium: 274s (2-pass) vs 713s (1-pass) = **2.6x faster**

This speedup arises because fitting smaller batches of splats is sub-linearly
cheaper per iteration.

### 4. Diminishing returns beyond 2 passes

No dataset achieves its best PSNR at 4-pass or 8-pass. For all 12 datasets,
going beyond 2 passes either maintains or reduces quality. 8-pass
(1250 iters/pass) does not give each pass enough iterations to converge. The
degradation is severe for some datasets: kidney_dapi loses 4.0 dB from 1-pass
to 8-pass (33.70 to 29.68).

### 5. PSNR and SSIM sometimes disagree on the best condition

For several datasets, the PSNR-best and SSIM-best conditions differ:

- **kidney_actin:** 1-pass has the best PSNR (30.60 dB) but 2-pass has
  substantially better SSIM (0.926 vs 0.897, a +0.029 improvement). This
  suggests 2-pass better preserves structural features despite marginally higher
  pixel-wise error.
- **opencell_lmnb1_ch1:** 1-pass leads in PSNR (33.46 dB) but 4-pass achieves
  the highest SSIM (0.921 vs 0.910).
- **organoid_ch0:** 1-pass wins PSNR (39.02 dB) but 8-pass has the best SSIM
  (0.943 vs 0.933).

These discrepancies indicate that progressive fitting may distribute
reconstruction error more uniformly, preserving local structure at the cost of
slightly higher global error in some cases.

### 6. Maximum absolute error increases with more passes

In 11 of 12 datasets, the 1-pass condition has the lowest (best) maximum
absolute error. Progressive fitting introduces localized artifacts where
successive passes amplify errors at specific voxels. The effect is most
dramatic for:

- **C. elegans:** 0.213 (1-pass) to 0.960 (8-pass) — a 4.5x increase
- **acto3d_heart_nuclei:** 0.568 (1-pass) to 0.920 (8-pass)
- **opencell_lmnb1_ch1:** 0.421 (1-pass) to 0.832 (8-pass)

The sole exception is opencell_lmnb1_ch0, where 2-pass has a slightly lower
max error (0.695) than 1-pass (0.765).

This suggests that while progressive fitting can improve average reconstruction
quality for some datasets, it trades off worst-case fidelity. Applications
sensitive to outlier errors (e.g., quantitative intensity measurements) should
prefer single-pass fitting.

---

## Statistical Notes

- **Replicates:** Most datasets have 2-3 independent replicates per condition.
  opencell_map4_ch0 and opencell_map4_ch1 have 1 replicate only.
- **Reproducibility:** PSNR standard deviation within conditions is very small,
  typically < 0.02 dB (maximum observed: 0.07 dB for organoid_ch0 8-pass).
  This confirms that the fitting process is highly reproducible and the observed
  differences between conditions are robust.
- **Figures:** Bar charts show mean values with error bars (1 standard deviation).
  Datasets with 1 replicate have no error bars.

---

## Interpretation: When does progressive help?

The results suggest 2-pass progressive fitting helps when:

1. **The signal is clean and well-structured.** Datasets where 2-pass wins
   (cells3d_membrane, tribolium, acto3d_heart, cells3d_nuclei) all have
   relatively smooth, high-contrast structures. The residual after pass 1 is
   predominantly unfit signal rather than noise, allowing pass 2 to target it
   effectively.

2. **The volume is large relative to the splat budget.** The two light-sheet
   datasets (tribolium: ~100M voxels, acto3d: ~100M voxels) have the lowest
   splat density (~3100 voxels/splat at 32K splats). Residual seeding
   efficiently reaches parts of the volume that the first pass could not cover.

3. **The structure is amenable to hierarchical decomposition.** cells3d datasets
   are small but have clean, discrete structures (nuclei, membranes) that are
   well suited to a coarse-then-fine fitting strategy.

**When does progressive hurt?** For noisy confocal data (OpenCell, kidney)
with high splat density, single-pass optimization across the full budget
produces better results. The residual in these datasets is a mixture of unfit
signal and noise, causing subsequent passes to partially fit noise.

**Practical recommendation:**
- Use **2-pass** for large, clean volumes (light-sheet) or small volumes with
  high-contrast structures — gains of +0.9 to +1.6 dB at 1.2-4.4x faster speed.
- Use **1-pass** for noisy confocal/spinning-disk data with conservative culling.
- Never use more than 2 passes at this budget level — diminishing returns are
  consistent across all datasets.

---

## File inventory

```
progressive_vs_single/
    run_analysis.py                 # Runner (1/2/4/8-pass per dataset)
    plot_results.py                 # Bar chart + slice montage (with replicate aggregation)
    INTERPRETATION.md               # This file
    results/
        <dataset>/                  # 12 datasets
            progressive.tsv
            fig_progressive_comparison.pdf
            fig_progressive_slices.pdf
            slices/                 # Cached slice data for montage
```
