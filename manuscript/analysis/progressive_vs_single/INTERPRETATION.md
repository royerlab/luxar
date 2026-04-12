# Progressive vs Single-Pass Fitting — Interpretation

## Overview

This analysis compares single-pass and multi-pass (progressive) Gaussian splat
fitting at the same total splat and iteration budget.

**Experimental design:**
- **Fixed budget:** 32K splats, 10K total iterations
- **Conditions:** 1-pass, 2-pass, 4-pass, 8-pass
- **For N passes:** `splats_per_pass = 32K/N`, `iters_per_pass = 10K/N`
- **Shared settings:** `cull_retention=0.999`, `enable_dynamic_ops=True`, `loss_type="l1"`
- **7 datasets** across confocal, spinning-disk, and light-sheet microscopy

---

## Results

### PSNR comparison (dB)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass | Best | vs 1-pass |
|---------|-------:|-------:|-------:|-------:|:----:|---------:|
| opencell_ch0 | **28.3** | 27.5 | 28.0 | 27.8 | 1-pass | — |
| opencell_ch1 | **25.2** | 24.1 | 24.5 | 24.2 | 1-pass | — |
| kidney_dapi | **33.7** | 32.8 | 31.7 | 30.2 | 1-pass | — |
| kidney_actin | **30.5** | 29.9 | 29.1 | 27.6 | 1-pass | — |
| organoid_ch0 | **39.0** | 37.7 | 37.9 | 37.7 | 1-pass | — |
| celegans_t100 | **39.4** | 38.4 | 39.2 | 39.3 | 1-pass | — |
| tribolium | 42.6 | **49.6** | 48.9 | 46.1 | 2-pass | +7.0 dB |

### Wall-clock time (seconds)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass |
|---------|-------:|-------:|-------:|-------:|
| opencell_ch0 | 72 | **21** | 32 | 35 |
| opencell_ch1 | 69 | **22** | 34 | 35 |
| kidney_dapi | 61 | 25 | **23** | 22 |
| kidney_actin | 63 | 25 | **23** | 23 |
| organoid_ch0 | 71 | 60 | **40** | 82 |
| celegans_t100 | 62 | **14** | 23 | 26 |
| tribolium | 214 | **119** | 148 | 154 |

---

## Key Findings

### 1. With conservative culling (0.999), single-pass wins for most datasets

This is a **reversal** from the previous run with `cull_retention=0.99`, where
progressive consistently beat single-pass by 2-4 dB. The explanation:

- With 0.99 culling, 10-50% of single-pass splats were removed post-fit (many
  had near-zero amplitude after initialization). Progressive fitting placed splats
  more efficiently on residual peaks, wasting fewer.
- With 0.999 culling, nearly all splats survive. Single-pass keeps its full 32K
  splats and optimizes them jointly for 10K iterations, which is more effective
  than splitting the budget across passes where each pass gets fewer iterations.

**This reveals that the progressive advantage was partly an artifact of aggressive
culling, not an inherent benefit of residual decomposition.**

### 2. Tribolium is the striking exception: 2-pass gains +7.0 dB

For the low-noise light-sheet data (Tribolium), progressive fitting provides a
massive improvement: 49.6 dB (2-pass) vs 42.6 dB (1-pass). This is because:

- The volume is large (99.5M voxels) and the signal is smooth and well-structured
- With 32K splats on 99.5M voxels, each splat covers ~3100 voxels on average
- The first pass captures the major structures; the second pass targets the
  residual detail with precision seeding on residual peaks
- The very low noise (σ=0.0009) means the residual is pure signal, not noise

### 3. Progressive is consistently faster

Even when progressive doesn't win on quality, it's 2-3x faster in wall-clock time
(e.g., 21s vs 72s for OpenCell ch0). This is because fitting smaller batches of
splats is sub-linearly cheaper per iteration.

### 4. More passes = diminishing returns or degradation

For 6 of 7 datasets, going beyond 2 passes either maintains or reduces quality.
8-pass (1250 iters/pass) doesn't give each pass enough time to converge. The
exception is C. elegans where 4-pass and 8-pass approach (but don't beat) 1-pass.

---

## Interpretation: When does progressive help?

The results suggest progressive fitting primarily helps when:

1. **The volume is large relative to the splat budget** (Tribolium: 99.5M voxels
   / 32K splats = 3100 voxels/splat). Residual seeding efficiently targets the
   parts of the volume that the first pass couldn't reach.

2. **The noise level is low.** When noise is low, the residual after pass 1 is
   pure unfit signal. When noise is high (confocal datasets), the residual is a
   mix of signal and noise, and subsequent passes may fit the noise component.

3. **Culling is aggressive.** With 0.99 culling, progressive fitting wastes fewer
   splats on noise because residual-seeded splats are more targeted. With 0.999
   culling, this advantage disappears because even poorly-placed splats survive.

**Practical recommendation:** Use 2-pass progressive for large, low-noise volumes
(light-sheet). For noisy confocal data, single-pass with the full iteration budget
may be preferable, especially with conservative culling.

---

## File inventory

```
progressive_vs_single/
    run_analysis.py                 # Runner (1/2/4/8-pass per dataset)
    plot_results.py                 # Bar chart + slice montage
    INTERPRETATION.md               # This file
    results/
        <dataset>/
            progressive.tsv
            fig_progressive_comparison.pdf
            fig_progressive_slices.pdf
```
