#!/usr/bin/env python3
"""Comprehensive benchmark for progressive Gaussian splat fitting.

Runs progressive fitting on TWO diverse test images:
1. **3D chimera**: DAPI volume split into 4 quadrants along Z×Y axes:
   - Z-top / Y-left:  denoised DAPI (ch1) — smooth nuclei
   - Z-top / Y-right: denoised other channel (ch0) — different contrast
   - Z-bot / Y-left:  raw noisy DAPI (ch1) — noisy nuclei
   - Z-bot / Y-right: raw noisy other channel (ch0) — noisy + different
2. **2D composite**: left=mitosis, right=astronaut grayscale

Reports PSNR + SSIM for each, plus geometric mean combined scores.
Outputs a single METRIC line for autoresearch extraction.

IMPORTANT: Budget parameters are CONSTANTS — do not change them.

IMPORTANT: every fit here pins ``floor="none"``. PSNR/SSIM are scored against
the RAW volume, so the fit must not remove a pedestal the reference still
carries. The shipped default ``floor="auto"`` estimates and subtracts a
histogram-mode background level — 3.55% of range on this file's own 2D composite
— which makes output amplitudes background-relative: the original-referenced
metrics then penalise the fit for correctly dropping non-signal, and the reported
METRIC would move silently whenever the floor estimator changes. ``floor="none"``
leaves the fit on the volume's own hard-min basis (normalization still subtracts
``min(V)`` and deliberately never adds it back), which *is* the raw basis for the
2D composite, whose ``min`` is 0; for the 3D chimera it is that same basis up to a
constant, since the chimera's global min is a min over separately normalized
quadrant crops. This is a benchmark-only pin — ``gsplat fit``/``cal`` keep
``--floor auto``, which is the right default for real use (see the 2026-07-12
``--floor`` decision in docs/guides/developer/BENCHMARK_FLOOR_DECISION.md).
"""

import os  # noqa: I001 — must set env before torch import

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import zarr
from arbol import aprint, asection

from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.metrics import compute_psnr, compute_ssim
from luxar.gsplats.preprocessing import calibrate_nlm_h, denoise_nlm
from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

# ============================================================
# FIXED BUDGET — DO NOT CHANGE THESE
# ============================================================
MAX_SPLATS_3D = 10000
MAX_SPLATS_2D = 4000
MAX_SPLATS_PER_PASS = 1000
ITERS_PER_PASS = 3000
PSNR_PATIENCE = 0.2
TRUNCATE_SIG = 3.0
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
TIME_POINT = 0
# ============================================================

CACHE_DIR = Path("delme/benchmark_cache")


def _normalize_to_100(V: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 100] range."""
    vmin, vmax = V.min(), V.max()
    if vmax > vmin:
        return ((V - vmin) / (vmax - vmin) * 100.0).astype(np.float32)
    return np.full_like(V, 50.0, dtype=np.float32)


def _denoise_volume(V_raw: np.ndarray, device: str) -> np.ndarray:
    """Apply NLM denoising to a volume."""
    v_max, v_min = float(V_raw.max()), float(V_raw.min())
    v_range = max(v_max - v_min, 1e-6)
    V_norm = torch.from_numpy(((V_raw - v_min) / v_range).astype(np.float32)).to(device)
    h_opt = calibrate_nlm_h(V_norm, device=device)
    h_opt = max(h_opt, 0.03)
    V_denoised = denoise_nlm(V_norm, h=h_opt, search_distance=9, device=device)
    return (V_denoised.cpu().numpy() * v_range + v_min).astype(np.float32)


def load_3d_chimera() -> np.ndarray:
    """Build 3D chimeric volume with 4 quadrants (cached).

    Layout (Z × Y):
        denoised_ch1 | denoised_ch0
        ─────────────┼─────────────
        noisy_ch1    | noisy_ch0
    """
    cache_path = CACHE_DIR / "chimera_3d.npy"
    if cache_path.exists():
        aprint(f"Loading cached 3D chimera from {cache_path}")
        return np.load(str(cache_path))

    with asection("Building 3D chimera volume"):
        # Load both channels from IDR
        try:
            import fsspec

            aprint(f"Loading from {ZARR_URL}")
            mapper = fsspec.get_mapper(ZARR_URL)
            try:
                store = zarr.open_group(mapper, mode="r")
            except FileNotFoundError:
                store = zarr.open_array(mapper, mode="r")

            data = store["0"]
            full_shape = data.shape
            aprint(f"OME-ZARR shape: {full_shape}")

            if len(full_shape) == 5:
                n_channels = full_shape[1]
                ch1_idx = min(1, n_channels - 1)
                ch0_idx = 0
                raw_ch1 = np.array(data[TIME_POINT, ch1_idx, :, :, :], dtype=np.float32)
                raw_ch0 = np.array(data[TIME_POINT, ch0_idx, :, :, :], dtype=np.float32)
                aprint(f"Loaded ch0 shape={raw_ch0.shape}, ch1 shape={raw_ch1.shape}")
            else:
                # Fallback: duplicate if only one channel
                raw_ch1 = np.array(data[:, :, :], dtype=np.float32)
                raw_ch0 = raw_ch1.copy()
        except Exception as e:
            aprint(f"Remote load failed: {e}, using synthetic fallback")
            shape = (128, 128, 128)
            rng = np.random.RandomState(42)
            raw_ch1 = np.zeros(shape, dtype=np.float32)
            raw_ch0 = np.zeros(shape, dtype=np.float32)
            for _ in range(10):
                for V in [raw_ch1, raw_ch0]:
                    c = [rng.uniform(10, s - 10) for s in shape]
                    sig = rng.uniform(4, 8)
                    amp = rng.uniform(60, 100)
                    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
                    V += amp * np.exp(
                        -sum((g - ci) ** 2 for g, ci in zip(grids, c)) / (2 * sig**2)
                    )

        # Normalize raw volumes to [0, 100]
        raw_ch1 = _normalize_to_100(raw_ch1)
        raw_ch0 = _normalize_to_100(raw_ch0)

        # Denoise both channels
        denoise_device = "cuda" if torch.cuda.is_available() else "cpu"
        with asection("Denoising ch1"):
            den_ch1 = _denoise_volume(raw_ch1, denoise_device)
        with asection("Denoising ch0"):
            den_ch0 = _denoise_volume(raw_ch0, denoise_device)

        # Build chimera: split Z at midpoint, Y at midpoint
        Z, Y, X = raw_ch1.shape
        mz, my = Z // 2, Y // 2

        chimera = np.zeros((Z, Y, X), dtype=np.float32)
        # Top-left: denoised ch1
        chimera[:mz, :my, :] = den_ch1[:mz, :my, :]
        # Top-right: denoised ch0
        chimera[:mz, my:, :] = den_ch0[:mz, my:, :]
        # Bottom-left: noisy ch1
        chimera[mz:, :my, :] = raw_ch1[mz:, :my, :]
        # Bottom-right: noisy ch0
        chimera[mz:, my:, :] = raw_ch0[mz:, my:, :]

        aprint(
            f"Chimera shape: {chimera.shape}, range: [{chimera.min():.1f}, {chimera.max():.1f}]"
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), chimera)
        aprint(f"Cached 3D chimera to {cache_path}")

    return chimera


def load_2d_composite() -> np.ndarray:
    """Build 2D composite: left=mitosis, right=astronaut gray (cached)."""
    cache_path = CACHE_DIR / "composite_2d.npy"
    if cache_path.exists():
        aprint(f"Loading cached 2D composite from {cache_path}")
        return np.load(str(cache_path))

    with asection("Building 2D composite"):
        from skimage import color, data
        from skimage.transform import resize
        from skimage.util import img_as_float32

        # Mitosis: grayscale, crop to 256×256
        mitosis = img_as_float32(data.human_mitosis())
        if mitosis.ndim == 3:
            mitosis = color.rgb2gray(mitosis)
        # Crop center 256×256
        h, w = mitosis.shape
        ch, cw = h // 2, w // 2
        mitosis = mitosis[ch - 128 : ch + 128, cw - 128 : cw + 128]
        mitosis = mitosis * 100.0  # Scale to [0, 100]

        # Astronaut: grayscale, resize to 256×256
        astro = img_as_float32(data.astronaut())
        if astro.ndim == 3:
            astro = color.rgb2gray(astro)
        astro = resize(astro, (256, 256), anti_aliasing=True).astype(np.float32)
        astro = astro * 100.0  # Scale to [0, 100]

        # Composite: side by side
        composite = np.zeros((256, 512), dtype=np.float32)
        composite[:, :256] = mitosis
        composite[:, 256:] = astro

        aprint(
            f"2D composite shape: {composite.shape}, range: [{composite.min():.1f}, {composite.max():.1f}]"
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), composite)
        aprint(f"Cached 2D composite to {cache_path}")

    return composite


def run_fit_and_evaluate(V: np.ndarray, max_splats: int, label: str) -> dict:
    """Run progressive fitting and compute PSNR + SSIM."""
    import time as _time

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fit_start = _time.perf_counter()
    with asection(f"Fitting: {label}"):
        result = fit_progressive_gaussian_splats(
            V,
            # PSNR/SSIM below are scored against the raw V, so the fit must stay
            # on V's own hard-min basis: the shipped floor="auto" would subtract
            # a background pedestal the reference still carries (module docstring).
            floor="none",
            max_splats=max_splats,
            max_splats_per_pass=MAX_SPLATS_PER_PASS,
            iters_per_pass=ITERS_PER_PASS,
            psnr_patience=PSNR_PATIENCE,
            truncate=TRUNCATE_SIG,
            device=device,
            verbose=True,
            max_eccentricity=6.0,
        )

    fit_time = _time.perf_counter() - fit_start

    with torch.no_grad():
        recon_tensor = render_to_volume_tensor(
            result, shape=V.shape, device=device, truncate=TRUNCATE_SIG
        )
        V_tensor = torch.from_numpy(V).to(recon_tensor.device)
        psnr = compute_psnr(recon_tensor, V_tensor)
        ssim = compute_ssim(recon_tensor, V_tensor)
        recon = recon_tensor.cpu().numpy()

    return {
        "psnr": psnr,
        "ssim": ssim,
        "fit_time": fit_time,
        "splats": result.n_splats,
        "passes": result.n_lods,
        "stop_reason": result.stats.get("stop_reason", "?"),
        "result": result,
        "recon": recon,
    }


def save_comparison_pdf(
    targets: dict, results: dict, combined_psnr: float, combined_ssim: float
) -> None:
    """Save multi-panel comparison PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("delme/benchmark_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"benchmark_{timestamp}_{combined_psnr:.2f}dB.pdf"

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Progressive Fitting Benchmark — Combined PSNR={combined_psnr:.2f} dB | SSIM={combined_ssim:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    row = 0
    for label, V in targets.items():
        res = results[label]
        recon = res["recon"]

        if V.ndim == 3:
            # 3D: show mid Z-slice
            mid = V.shape[0] // 2
            target_slice = V[mid]
            recon_slice = recon[mid]
            slice_label = f"{label} (Z-mid)"
        else:
            # 2D
            target_slice = V
            recon_slice = recon
            slice_label = label

        diff = target_slice - recon_slice
        diff_absmax = max(1e-12, float(np.abs(diff).max()))
        vmin, vmax = 0, float(V.max())

        ax1 = fig.add_subplot(len(targets), 3, row * 3 + 1)
        ax1.imshow(target_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax1.set_title(f"Target — {slice_label}")
        ax1.axis("off")

        ax2 = fig.add_subplot(len(targets), 3, row * 3 + 2)
        ax2.imshow(recon_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax2.set_title(
            f"Recon — PSNR={res['psnr']:.2f} SSIM={res['ssim']:.4f} "
            f"({res['splats']} splats, {res['passes']}p)"
        )
        ax2.axis("off")

        ax3 = fig.add_subplot(len(targets), 3, row * 3 + 3)
        im = ax3.imshow(diff, cmap="RdBu_r", vmin=-diff_absmax, vmax=diff_absmax)
        ax3.set_title(f"Difference — {slice_label}")
        ax3.axis("off")
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        row += 1

    plt.tight_layout()
    fig.savefig(str(pdf_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    aprint(f"Saved comparison PDF: {pdf_path}")


def main():
    with asection("Comprehensive Progressive Fitting Benchmark"):
        # 1. Load test data
        V_3d = load_3d_chimera()
        V_2d = load_2d_composite()

        targets = {"3D chimera": V_3d, "2D composite": V_2d}

        # 2. Run fitting on each
        results = {}
        results["3D chimera"] = run_fit_and_evaluate(V_3d, MAX_SPLATS_3D, "3D chimera")
        results["2D composite"] = run_fit_and_evaluate(
            V_2d, MAX_SPLATS_2D, "2D composite"
        )

        # 3. Compute combined metrics (geometric mean)
        psnrs = [results[k]["psnr"] for k in results]
        ssims = [results[k]["ssim"] for k in results]
        combined_psnr = float(np.exp(np.mean(np.log(psnrs))))
        combined_ssim = float(np.exp(np.mean(np.log(ssims))))

        # 4. Report individual and combined metrics
        with asection("Results"):
            for label, res in results.items():
                aprint(
                    f"{label}: PSNR={res['psnr']:.2f} dB, SSIM={res['ssim']:.4f}, "
                    f"{res['splats']} splats, {res['passes']} passes, "
                    f"stop={res['stop_reason']}"
                )
            aprint(f"Combined PSNR (geomean): {combined_psnr:.4f} dB")
            aprint(f"Combined SSIM (geomean): {combined_ssim:.6f}")

        # 5. Compute timing
        time_3d = results["3D chimera"]["fit_time"]
        time_2d = results["2D composite"]["fit_time"]
        time_total = time_3d + time_2d

        # 6. Output metrics for autoresearch
        print(f"METRIC={combined_psnr:.4f}")
        print(f"PSNR_3D={results['3D chimera']['psnr']:.4f}")
        print(f"SSIM_3D={results['3D chimera']['ssim']:.6f}")
        print(f"PSNR_2D={results['2D composite']['psnr']:.4f}")
        print(f"SSIM_2D={results['2D composite']['ssim']:.6f}")
        print(f"COMBINED_SSIM={combined_ssim:.6f}")
        print(f"splats_3d={results['3D chimera']['splats']}")
        print(f"splats_2d={results['2D composite']['splats']}")
        print(f"TIME_TOTAL={time_total:.1f}")
        print(f"TIME_3D={time_3d:.1f}")
        print(f"TIME_2D={time_2d:.1f}")

        # 7. Save PDF
        save_comparison_pdf(targets, results, combined_psnr, combined_ssim)


if __name__ == "__main__":
    main()
