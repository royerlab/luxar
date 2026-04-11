# Splat Count vs Reconstruction Quality — Multi-Dataset Interpretation

## Overview

This analysis characterises how Gaussian splat reconstruction quality scales with
the number of splats across **6 datasets** spanning 3 imaging modalities, 4 biological
structures, and a 4x range in volume sizes. It consists of two complementary
experiments per dataset:

1. **Rate-distortion analysis** — PSNR, SSIM, and compression vs splat count
2. **Noise2Self analysis** — blind-spot cross-validation to disentangle signal
   fitting from noise fitting (Batson & Royer, ICML 2019)

---

## Datasets

| Key | Modality | Structure | Shape | Voxels | Source |
|-----|----------|-----------|-------|--------|--------|
| `opencell_map4_ch0` | Spinning-disk confocal | Nuclei (Hoechst) | 51x600x600 | 18.4M | OpenCell |
| `opencell_map4_ch1` | Spinning-disk confocal | Microtubules (MAP4-GFP) | 51x600x600 | 18.4M | OpenCell |
| `kidney_dapi` | Confocal | Nuclei in tissue (DAPI) | 16x512x512 | 4.2M | scikit-image |
| `kidney_actin` | Confocal | Actin filaments (Phalloidin) | 16x512x512 | 4.2M | scikit-image |
| `organoid_ch0` | Confocal | Intestinal organoid | 256x256x256 | 16.8M | IDR idr0062 |
| `celegans_t100` | Confocal | Embryo nuclei | 41x512x512 | 10.7M | Zenodo 6460303 |

---

## Analysis 1: Rate-Distortion Curves

### Summary table (PSNR at selected splat counts)

| Dataset | 1K | 8K | 32K | 128K | 512K | Total gain |
|---------|----:|----:|-----:|------:|------:|-----------:|
| opencell_ch0 | 20.7 | 23.7 | 24.2 | 25.3 | 27.1 | +6.4 dB |
| opencell_ch1 | 17.6 | 20.1 | 21.1 | 22.3 | 24.4 | +6.9 dB |
| kidney_dapi | 19.4 | 25.0 | 29.9 | 36.3 | 41.8 | +22.4 dB |
| kidney_actin | 17.7 | 22.1 | 26.8 | 33.7 | 43.9 | +26.1 dB |
| organoid_ch0 | 30.3 | 37.4 | 39.2 | 43.4 | 47.0 | +16.8 dB |
| celegans_t100 | 34.6 | 35.3 | 35.4 | 35.9 | 37.8 | +3.2 dB |

### Key findings

**1. PSNR range varies enormously across datasets (3–26 dB gain).**
The kidney datasets show the largest improvement (22–26 dB from 1K to 512K), while
C. elegans shows the smallest (3.2 dB). This reflects fundamental differences in
signal complexity and noise level.

**2. The kidney datasets are the most "compressible".**
With 16 z-slices and dense tissue signal, the kidney volumes achieve very high PSNR
(41–44 dB at 512K) and show no sign of plateauing. The thin z-extent (16 slices)
means the effective dimensionality is almost 2D, allowing splats to achieve near-
perfect reconstruction.

**3. C. elegans is the hardest to fit.**
Despite a medium-sized volume (10.7M voxels), the PSNR barely moves from 34.6 to
37.8 dB. The confocal data has high noise and very sparse signal (only a few bright
nuclei per z-slice), leaving little recoverable structure for the splats to capture.

**4. The organoid starts at high PSNR (30.3 dB) because of its smooth structure.**
The downscaled 256^3 volume has significant blur from interpolation, making it
inherently smooth and easy to reconstruct. Even 1K splats achieve 30 dB.

**5. OpenCell microtubules (ch1) are harder than nuclei (ch0).**
At every splat count, ch1 lags ch0 by ~3 dB. Filamentous structures (microtubules)
have higher spatial frequency content than blob-like structures (nuclei), requiring
more splats to capture fine detail.

**6. Compression ratios span 3 orders of magnitude.**
At the "efficient" operating point (~64K splats), compression ranges from 7x
(kidney) to 37x (OpenCell), demonstrating Gaussian splatting as a viable compact
representation for diverse microscopy data.

---

## Analysis 2: Noise2Self — Signal vs Noise Fitting

### Summary of noise behaviour across datasets

| Dataset | Held-out range | Peak noise frac | At splats | Overfits? |
|---------|---------------:|----------------:|----------:|:---------:|
| opencell_ch0 | 20.7–25.8 dB | **48.5%** | 222K | No |
| opencell_ch1 | 17.6–22.8 dB | **57.5%** | 49K | Mild |
| kidney_dapi | 19.3–29.1 dB | **>100%** | 235K | **Yes** |
| kidney_actin | 17.8–25.6 dB | **~99%** | 446K | **Near** |
| organoid_ch0 | 30.3–40.3 dB | **83.8%** | 406K | Moderate |
| celegans_t100 | 34.7–36.9 dB | **>100%** | 7K | **Yes (early)** |

### Key findings

**1. Gaussian splats CAN overfit — but it depends on the dataset.**
Contrary to what the initial single-dataset analysis suggested, the kidney datasets
show clear overfitting: the held-out PSNR peaks and then *decreases* at high splat
counts, meaning the model memorises noise that hurts generalisation. This was not
visible in the OpenCell nuclei dataset (ch0), which stays below 50% noise fraction.

**2. Volume geometry matters: thin volumes (few z-slices) overfit more easily.**
The kidney volumes (16 z-slices) show the strongest overfitting, while the isotropic
organoid (256^3) and thicker OpenCell (51 z-slices) resist it better. With only 16
z-slices, Gaussians have less room to "spread" in z, potentially allowing them to
fit plane-by-plane noise patterns that don't generalise.

**3. Sparse, noisy data overfits earliest.**
C. elegans shows noise fraction >100% at just 7K splats. The very sparse nuclear
signal in a noisy confocal volume means the model quickly exhausts the recoverable
signal and starts fitting noise. This is consistent with the near-flat rate-distortion
curve (only 3.2 dB total gain).

**4. Dense, structured data resists overfitting.**
OpenCell nuclei (ch0) — compact, bright blobs in every z-slice — never crosses the
50% noise threshold. The dense, spatially correlated signal provides ample structure
for the smooth Gaussian basis to capture, leaving little room for noise fitting.

**5. The "50% noise crossover" is a useful but dataset-dependent criterion.**
For well-conditioned data (OpenCell, organoid), the crossover occurs at high splat
counts (>200K) or not at all. For challenging data (kidney, C. elegans), it occurs
earlier and should guide the splat budget.

### Practical recommendations by dataset type

| Signal type | Recommended splats | Rationale |
|-------------|------------------:|-----------|
| Dense blobs (nuclei, confocal) | 32K–64K | Low noise fraction, diminishing returns above 64K |
| Sparse filaments (microtubules) | 64K–128K | Higher frequency content needs more splats |
| Thin volumes (few z-slices) | 16K–32K | Overfitting risk above 32K; monitor held-out loss |
| Very noisy / sparse data | 4K–16K | Signal exhausted early; more splats fit noise |
| Smooth / downscaled data | 8K–32K | Already easy to reconstruct |

---

## Cross-Dataset Comparisons

### Structure complexity determines absolute PSNR

The datasets sort naturally by structural complexity:
- **Simplest**: Organoid (smooth, downscaled, 30–47 dB)
- **Simple**: C. elegans (sparse, few bright nuclei, 34–38 dB)
- **Moderate**: Kidney (dense tissue, multiple channels, 17–44 dB depending on channel)
- **Complex**: OpenCell (fine cellular structures, 17–27 dB)

Note that "simple" data starts at higher PSNR but has less to gain from more splats
(diminishing returns set in earlier), while "complex" data starts lower but benefits
more from additional splats.

### Signal density determines noise resistance

Datasets with dense, spatially correlated signal (OpenCell nuclei, organoid interior)
resist noise fitting because the Gaussian basis functions naturally capture smooth,
extended structures. Datasets with sparse signal (C. elegans nuclei in mostly-dark
volume) or thin geometry (kidney, 16 z-slices) allow splats to "chase" individual
noise features.

### Implications for automatic splat count selection

A practical rule based on these findings:

1. **Start with splats = volume_voxels / 500** (~32K for 18M voxels) as a default
2. **For noisy data**: reduce by 2–4x
3. **For thin volumes (z < 32)**: reduce by 2x
4. **For smooth/downscaled data**: reduce by 4x
5. **For very dense, high-SNR data**: increase by 2–4x

Alternatively, the Noise2Self analysis provides an automatic criterion: fit with
blind-spot masking and stop when the held-out loss begins to plateau or rise.

---

## Figures produced

For each dataset, the analysis generates:

| Figure | Description |
|--------|-------------|
| `fig_quality_curves.pdf` | 2x2 grid: PSNR, SSIM, fitting time, culling vs splat count |
| `fig_slice_montage.pdf` | Representative z-slices at 5–6 splat counts + error map |
| `fig_noise2self.pdf` | Train vs held-out PSNR + MSE with gap shading |

All figures are in `results/<dataset_key>/` and `results/<dataset_key>_n2s/`.

---

## File inventory

```
splat_count_vs_quality/
    datasets.py              # 6 registered datasets
    run_analysis.py          # Rate-distortion runner
    plot_results.py          # Quality curves + slice montage plotter
    run_noise2self.py        # Noise2Self blind-spot analysis runner
    plot_noise2self.py       # N2S train/held-out curve plotter
    INTERPRETATION.md        # This file
    DEDUPLICATION_BOTTLENECK.md
    results/
        opencell_map4_ch0/       # Rate-distortion results + PDFs
        opencell_map4_ch0_n2s/   # Noise2Self results + PDF
        opencell_map4_ch1/
        opencell_map4_ch1_n2s/
        kidney_dapi/
        kidney_dapi_n2s/
        kidney_actin/
        kidney_actin_n2s/
        organoid_ch0/
        organoid_ch0_n2s/
        celegans_t100/
        celegans_t100_n2s/
```

---

## References

- Batson, J. & Royer, L.A. (2019). Noise2Self: Blind Denoising by Self-Supervision.
  *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
- Cho, N.H., Bhatt, D.P. et al. (2022). OpenCell: Endogenous tagging for the
  cartography of human cellular organization. *Science*, 375(6585), eabi6983.
- van der Walt, S. et al. (2014). scikit-image: image processing in Python.
  *PeerJ*, 2:e453.
- Blin, G. et al. (2019). / Williams, E. et al. (2017). Image Data Resource.
  *Nature Methods*, 14(8):775-781.
- Bao, Z. et al. (2006). Automated cell lineage tracing in *Caenorhabditis elegans*.
  *PNAS*, 103(8):2707-2712.
