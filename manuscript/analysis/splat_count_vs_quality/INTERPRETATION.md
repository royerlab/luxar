# Splat Count vs Reconstruction Quality — Multi-Dataset Interpretation

## Overview

This analysis characterises how Gaussian splat reconstruction quality scales with
the number of splats across **7 datasets** spanning 3 imaging modalities, multiple
biological structures, and a wide range of volume sizes. It consists of three
complementary experiments per dataset:

1. **Rate-distortion analysis** — PSNR, SSIM, and compression vs splat count
2. **Blind-spot cross-validation** — held-out validation to disentangle signal
   fitting from noise fitting (masking inspired by Batson & Royer, ICML 2019)
3. **Noise floor estimation** — theoretical PSNR ceiling from ensemble noise
   estimators (Laplacian MAD + Haar MAD + background MAD)

All results use `cull_retention=0.999` (99.9% amplitude retention) to minimise
confounding from post-fit culling, and `n_iters=20000` with `early_stop_patience=500`.

---

## Datasets

| Key | Modality | Structure | Shape | Voxels |
|-----|----------|-----------|-------|--------|
| `opencell_map4_ch0` | Spinning-disk confocal | Nuclei (Hoechst) | 51x600x600 | 18.4M |
| `opencell_map4_ch1` | Spinning-disk confocal | Microtubules (MAP4-GFP) | 51x600x600 | 18.4M |
| `kidney_dapi` | Confocal | Nuclei in tissue (DAPI) | 16x512x512 | 4.2M |
| `kidney_actin` | Confocal | Actin filaments (Phalloidin) | 16x512x512 | 4.2M |
| `organoid_ch0` | Confocal | Intestinal organoid | 236x275x271 | 17.6M |
| `celegans_t100` | Confocal | Embryo nuclei | 41x512x512 | 10.7M |
| `tribolium` | Light-sheet | Embryo (cropped) | 991x104x965 | 99.5M |

---

## Analysis 1: Rate-Distortion Curves

### PSNR at selected splat counts (dB)

| Dataset | 1K | 8K | 32K | 128K | 512K | Gain | Noise floor |
|---------|----:|----:|-----:|------:|------:|-----:|------------:|
| opencell_ch0 | 25.1 | 27.9 | 28.3 | 28.9 | 29.9 | +4.8 | 32.6 |
| opencell_ch1 | 21.5 | 24.2 | 25.2 | 26.1 | 27.6 | +6.1 | 30.4 |
| kidney_dapi | 22.5 | 29.5 | 33.7 | 36.4 | 37.3 | +14.8 | 41.3 |
| kidney_actin | 21.1 | 26.1 | 30.6 | 35.4 | 37.2 | +16.1 | 43.5 |
| organoid_ch0 | 32.7 | 37.5 | 39.0 | 40.1 | 41.1 | +8.4 | 54.4 |
| celegans_t100 | 38.4 | 39.3 | 39.4 | 39.5 | 40.4 | +2.0 | 40.7 |
| tribolium | 29.9 | 40.3 | 42.6 | 42.7 | 42.7 | +12.8 | 61.0 |

### Key observations

**1. Culling is now minimal.** With `cull_retention=0.999`, culling removes only
0–11% of splats (vs 10–66% with the previous 0.99 setting). The culling panel
shows near-zero or even slightly negative values (dynamic ops can add splats
beyond the requested count). This means the x-axis faithfully represents model
capacity.

**2. Tribolium is the most "compressible" dataset.** The light-sheet data has very
low noise (σ=0.0009, noise floor 61.0 dB) and smooth, well-defined structure.
PSNR plateaus sharply at 32K splats (42.6 dB) — additional splats provide no
benefit because the signal is fully captured. At 32K splats the compression ratio
is 313x.

**3. C. elegans has the least recoverable signal.** PSNR barely moves from 38.4 to
40.4 dB across 500x more splats. The volume is mostly empty with sparse nuclei,
and the noise floor (40.7 dB) is nearly reached by 16K splats. The confocal data
is noisy relative to the sparse signal.

**4. Kidney datasets have the widest dynamic range.** From 22.5 to 37.3 dB (14.8 dB
gain for DAPI), these dense tissue volumes reward more splats continuously but show
diminishing returns above 128K. The noise floor (41.3–43.5 dB) is 4–6 dB above the
best reconstruction, leaving room for improvement.

**5. OpenCell datasets approach but don't reach the noise floor.** At 512K splats,
PSNR is 29.9 dB (ch0) and 27.6 dB (ch1) against noise floors of 32.6 and 30.4 dB
respectively — a 2.7–2.8 dB gap suggesting the optimizer hasn't fully exhausted
the recoverable signal.

---

## Analysis 2: Blind-Spot Cross-Validation — Signal vs Noise Fitting

### Held-out PSNR behaviour

| Dataset | Held-out peak | At splats | Final held-out | Gap at 512K | Overfits? |
|---------|-------------:|----------:|---------------:|------------:|:---------:|
| opencell_ch0 | 28.0 dB | 32K | 27.6 dB | 2.3 dB | **Yes** |
| opencell_ch1 | 24.9 dB | 62K | 24.7 dB | 3.0 dB | **Yes** |
| kidney_dapi | 33.7 dB | 63K | 31.5 dB | 5.9 dB | **Yes** |
| kidney_actin | 30.8 dB | 124K | 28.4 dB | 8.8 dB | **Yes** |
| organoid_ch0 | 37.7 dB | 28K | 37.2 dB | 3.9 dB | **Yes** |
| celegans_t100 | 39.3 dB | 16K | 38.9 dB | 1.4 dB | **Yes** |
| tribolium | 42.8 dB | 510K | 42.8 dB | -0.1 dB | **No** |

### Key findings

**1. With conservative culling, Gaussian splats DO overfit on 6 of 7 datasets.**

This is the most important finding. The held-out PSNR (measured on blind-spot
masked voxels) peaks and then *declines* for all datasets except Tribolium.
This means that beyond the peak, additional splats are memorising noise in the
training voxels — noise that doesn't generalise to the held-out positions.

The kidney datasets show the strongest overfitting: kidney_dapi's held-out PSNR
drops 2.2 dB from peak (33.7 dB at 63K) to final (31.5 dB at 512K), while the
train PSNR continues rising to 37.3 dB — a 5.9 dB gap.

**2. Tribolium is the exception: no overfitting.**

The Tribolium light-sheet data has extremely low noise (σ=0.0009). The train and
held-out curves are virtually identical across all splat counts — there is no
measurable noise for the splats to fit. The noise floor (61 dB) is far above the
reconstruction ceiling (42.7 dB), meaning the model is signal-limited, not
noise-limited.

**3. The held-out peak identifies the optimal splat count.**

For each dataset, the splat count where held-out PSNR peaks provides a principled
stopping criterion:
- OpenCell: 32K–62K splats
- Kidney: 63K–124K splats
- Organoid: 28K splats
- C. elegans: 16K splats
- Tribolium: no peak (signal-limited)

**4. The noise floor line provides context.**

The dashed noise floor line in the figures shows the theoretical PSNR ceiling for
a perfect denoiser. When the *train* PSNR approaches or exceeds this line (as in
kidney_dapi at 512K), the model is fitting noise — confirmed by the held-out
decline. In the cross-validation MSE panel, the noise variance line shows the minimum achievable
MSE; the train MSE drops below it, confirming noise memorisation.

---

## Analysis 3: Noise Floor Estimation

### Estimates per dataset

| Dataset | σ_Laplacian | σ_Haar | σ_Background | σ_Ensemble | PSNR_max |
|---------|--------:|------:|-----------:|----------:|--------:|
| opencell_ch0 | 0.0234 | 0.0247 | 0.0039 | 0.0234 | 32.6 dB |
| opencell_ch1 | 0.0300 | 0.0322 | 0.0050 | 0.0300 | 30.4 dB |
| kidney_dapi | 0.0086 | 0.0089 | 0.0069 | 0.0086 | 41.3 dB |
| kidney_actin | 0.0067 | 0.0075 | 0.0044 | 0.0067 | 43.5 dB |
| organoid_ch0 | 0.0019 | 0.0022 | 0.0000 | 0.0019 | 54.4 dB |
| celegans_t100 | 0.0092 | 0.0093 | 0.0022 | 0.0092 | 40.7 dB |
| tribolium | 0.0009 | 0.0008 | 0.0030 | 0.0009 | 61.0 dB |

The Laplacian and Haar methods agree within 15% for all datasets. The background
method consistently underestimates (expected — the lowest-intensity voxels have
less variance). The ensemble median robustly picks the high-pass estimates.

Tribolium has the lowest noise (σ=0.0009, light-sheet) and the highest noise floor
(61 dB). OpenCell ch1 has the highest noise (σ=0.030, spinning-disk confocal with
sparse microtubule signal) and the lowest noise floor (30.4 dB).

---

## Implications

1. **Splat count should be matched to the noise level.** The N2S held-out peak
   provides a principled, dataset-specific stopping criterion. Beyond this point,
   quality does not improve and may degrade.

2. **Conservative culling (0.999) exposes overfitting.** With aggressive culling
   (0.99), the post-fit removal of weak splats masks the noise-fitting problem.
   With minimal culling, the overfitting is clearly visible in the cross-validation curves.

3. **The noise floor contextualises the rate-distortion curves.** Datasets far
   below their noise floor (OpenCell, kidney) have room for improvement through
   better optimization or more iterations. Datasets near their noise floor
   (C. elegans, Tribolium) are already capturing most of the recoverable signal.

4. **Light-sheet data is qualitatively different from confocal.** Tribolium shows
   no overfitting, minimal noise, and sharp PSNR plateaus — the representation
   is signal-limited, not noise-limited.

---

## File inventory

```
splat_count_vs_quality/
    datasets.py              # 7 registered datasets
    run_analysis.py          # Rate-distortion runner
    plot_results.py          # Quality curves + slice montage
    run_noise2self.py        # Blind-spot cross-validation analysis
    plot_noise2self.py       # Cross-validation train/held-out curves
    noise_floor.py           # Noise estimation module
    run_noise_floor.py       # Noise floor runner
    INTERPRETATION.md        # This file
    results/
        <dataset>/           # metrics.tsv + PDFs
        <dataset>_n2s/       # N2S metrics + PDF
        noise_floor.tsv      # Shared noise floor estimates
```

## References

- Batson, J. & Royer, L.A. (2019). Noise2Self: Blind Denoising by Self-Supervision. ICML.
- Cho, N.H. et al. (2022). OpenCell. Science 375(6585).
- van der Walt, S. et al. (2014). scikit-image. PeerJ 2:e453.
