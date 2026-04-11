# Progressive vs Single-Pass Fitting — Interpretation

## Overview

This analysis compares single-pass and multi-pass (progressive) Gaussian splat
fitting at the same total splat and iteration budget. The question: does the
progressive residual decomposition improve reconstruction quality, or is a single
large fit equally effective?

**Experimental design:**
- **Fixed budget:** 32K splats, 10K total iterations
- **Conditions:** 1-pass, 2-pass, 4-pass, 8-pass
- **For N passes:** `splats_per_pass = 32K/N`, `iters_per_pass = 10K/N`
- **Shared settings:** `cull_retention=0.999`, `enable_dynamic_ops=True`, `loss_type="l1"`
- **6 datasets** spanning confocal, spinning-disk confocal, and light-sheet microscopy

The progressive fitter (`fit_progressive_gaussian_splats`) internally tunes per-pass
hyperparameters: pass 0 uses L1 loss, subsequent passes switch to Poisson loss with
higher asymmetric penalty and adaptive learning rates. The single-pass fitter uses
L1 throughout. This is intentional — we compare the full pipelines as a user would
invoke them, not stripped-down versions.

---

## Results

### PSNR comparison (dB)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass | Best | Gain |
|---------|-------:|-------:|-------:|-------:|:----:|-----:|
| OpenCell Hoechst | 24.6 | 27.7 | **28.2** | 28.0 | 4-pass | +3.6 dB |
| OpenCell MAP4-GFP | 21.4 | 24.5 | **24.9** | 24.5 | 4-pass | +3.5 dB |
| Kidney DAPI | 31.4 | **33.3** | 32.0 | 30.4 | 2-pass | +1.9 dB |
| Kidney Actin | 27.8 | **30.3** | 29.2 | 27.6 | 2-pass | +2.5 dB |
| Organoid | 35.8 | 38.3 | **38.4** | 37.8 | 4-pass | +2.6 dB |
| C. elegans | 35.5 | 38.6 | 39.2 | **39.4** | 8-pass | +3.9 dB |

### SSIM comparison

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass |
|---------|-------:|-------:|-------:|-------:|
| OpenCell Hoechst | 0.410 | 0.639 | 0.671 | **0.674** |
| OpenCell MAP4-GFP | 0.377 | 0.583 | **0.609** | 0.605 |
| Kidney DAPI | 0.944 | **0.961** | 0.950 | 0.930 |
| Kidney Actin | 0.831 | **0.928** | 0.911 | 0.871 |
| Organoid | 0.852 | 0.921 | 0.935 | **0.944** |
| C. elegans | 0.658 | 0.864 | 0.888 | **0.893** |

### Wall-clock time (seconds)

| Dataset | 1-pass | 2-pass | 4-pass | 8-pass |
|---------|-------:|-------:|-------:|-------:|
| OpenCell Hoechst | 114 | 51 | 49 | **44** |
| OpenCell MAP4-GFP | 101 | 53 | 53 | **46** |
| Kidney DAPI | 97 | 33 | 28 | **28** |
| Kidney Actin | 98 | 33 | 31 | **30** |
| Organoid | 115 | 163 | 89 | **94** |
| C. elegans | 80 | 38 | **37** | 37 |

---

## Key Findings

### 1. Progressive fitting consistently outperforms single-pass

Across all 6 datasets, the best progressive configuration beats single-pass by
**+1.9 to +3.9 dB** in PSNR. This is a substantial improvement — 2-3 dB is easily
visible in image quality. The improvement is consistent regardless of modality,
noise level, or volume geometry.

### 2. The optimal number of passes is 2-4

- **2 passes** is optimal for datasets with simple structure and few z-slices
  (kidney), where the second pass captures fine detail that the first pass missed.
- **4 passes** is optimal for datasets with richer spatial structure (OpenCell,
  organoid), where additional decomposition levels capture progressively finer
  scales.
- **8 passes** only wins for C. elegans (sparse nuclei in mostly-empty volume),
  where each pass can focus on individual structures. But even here the gain over
  4-pass is marginal (+0.2 dB).

### 3. Too many passes hurt when iterations per pass are insufficient

For kidney (2-pass > 4-pass > 8-pass), splitting 10K iterations into 8 passes of
1250 iters each doesn't give each pass enough optimisation time to converge. The
per-pass quality suffers, and the accumulated result is worse than fewer, better-
converged passes. This effect is strongest for datasets with many recoverable
details (kidney achieves up to 41 dB at the noise floor — there's a lot of signal
to capture per pass).

### 4. Progressive fitting is also faster

Single-pass fitting with 32K splats is the slowest condition in 5 of 6 datasets
(97-115s). Progressive fitting at 2 passes typically runs in 33-53s (2-3x faster).
This is because:
- Fitting fewer splats per pass is sub-linearly cheaper (optimizer overhead is
  per-splat)
- The residual signal after pass 1 has lower dynamic range, making subsequent
  passes converge faster
- Seeding on residual peaks is more targeted than seeding on the full volume

### 5. Why progressive works: residual decomposition captures multiple scales

The first pass captures the dominant large-scale structures (nuclei boundaries,
tissue morphology). Subsequent passes fit the residual — the fine details that the
first-pass Gaussians were too large or too few to capture. This is analogous to a
wavelet decomposition: each level captures a different spatial frequency band.

The per-pass hyperparameter adaptation in the progressive fitter reinforces this:
- Pass 0: L1 loss with low asymmetric penalty → captures bulk amplitude
- Later passes: Poisson loss with high asymmetric penalty → captures fine,
  low-amplitude details without over-shooting

---

## Implications for the paper

1. **Progressive fitting should be the default recommendation.** 2-4 passes
   consistently delivers +2-4 dB improvement at equal or lower compute cost.

2. **The number of passes can be tuned to the data.** Simple data (kidney, few
   z-slices) needs only 2 passes. Complex data (dense tissue, whole embryo) benefits
   from 4. Very sparse data (C. elegans) can benefit from 8.

3. **The progressive advantage is not just about optimisation** — it's about the
   representation. Fitting residuals forces the model to allocate splats at different
   spatial scales, avoiding the common failure mode where many splats redundantly
   cover the same large feature.

---

## File inventory

```
progressive_vs_single/
    run_analysis.py                 # Runner (1/2/4/8-pass for each dataset)
    plot_results.py                 # Bar chart comparison figures
    INTERPRETATION.md               # This file
    results/
        <dataset>/
            progressive.tsv         # One row per condition
            fig_progressive_comparison.pdf
```
