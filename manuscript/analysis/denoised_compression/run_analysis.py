#!/usr/bin/env python3
"""Denoised compression comparison using Noise2Self held-out evaluation.

Standard PSNR measures reconstruction of the RAW signal (noise included).
This inflates the apparent quality of methods that preserve noise (like
quantization), and penalises methods that denoise (like GSplats at the
N2S-optimal count).

This analysis applies the Noise2Self held-out framework to ALL compression
methods equally:
  1. Generate a random 5% held-out mask
  2. For GSplats: fit on the remaining 95%, evaluate on the held-out 5%
     (already computed in the N2S analysis)
  3. For quantization baselines: apply quantization+compression to the
     FULL volume, then evaluate on the held-out 5%. Since quantization
     doesn't "learn" from the data, the mask doesn't affect the compression
     — but the held-out PSNR reveals whether the preserved information
     generalises (signal) or not (noise).

The key insight: for lossless/near-lossless methods, the held-out PSNR
equals the noise floor (they preserve noise perfectly). For GSplats at
the N2S-optimal count, the held-out PSNR exceeds the noise floor at low
BPV because the representation captures signal while discarding noise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
QUALITY_DIR = ANALYSIS_DIR.parent / "splat_count_vs_quality" / "results"
RESULTS_DIR = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ANALYSIS_DIR.parent / "splat_count_vs_quality"))

DATASETS = ['kidney_dapi', 'organoid_ch0', 'celegans_t100', 'tribolium', 'opencell_map4_ch0']

MASK_FRACTION = 0.05  # Same as N2S analysis
SEED = 42


def compute_psnr(orig, recon, mask=None):
    """Compute PSNR, optionally only on masked voxels."""
    if mask is not None:
        orig = orig[mask]
        recon = recon[mask]
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_splat_bytes(n_splats, ndim=3):
    cholesky = ndim * (ndim + 1) // 2
    return n_splats * (ndim + 1 + cholesky) * 4


def generate_mask(shape, fraction, seed):
    """Generate a random held-out mask (same as N2S analysis)."""
    rng = np.random.RandomState(seed)
    mask = rng.random(shape) < fraction
    return mask


def quantize_volume(volume, bits):
    """Uniform quantization to N bits and back."""
    v_min, v_max = float(volume.min()), float(volume.max())
    rng = v_max - v_min if v_max > v_min else 1.0
    max_val = (1 << bits) - 1

    quantized = np.clip(np.round((volume - v_min) / rng * max_val), 0, max_val)
    # Dequantize
    recon = quantized / max_val * rng + v_min
    return recon


def get_gsplat_n2s_data(ds):
    """Get GSplat N2S held-out PSNR data from existing analysis."""
    n2s_path = QUALITY_DIR / f"{ds}_n2s" / "metrics_n2s.tsv"
    if not n2s_path.exists():
        return pd.DataFrame()

    nf = pd.read_csv(QUALITY_DIR / "noise_floor.tsv", sep='\t')
    nf_row = nf[nf['dataset'] == ds]
    if nf_row.empty:
        return pd.DataFrame()

    shape = tuple(int(x) for x in nf_row['volume_shape'].values[0].split('x'))
    n_voxels = int(np.prod(shape))
    ndim = len(shape)

    n2s = pd.read_csv(n2s_path, sep='\t')
    rows = []
    for _, r in n2s.iterrows():
        n_splats = int(r['n_splats_final'])
        splat_bytes = compute_splat_bytes(n_splats, ndim)
        bpv = splat_bytes * 8 / n_voxels
        rows.append({
            'method': 'gsplats',
            'knob': int(r['seeds_requested']),
            'bpv': round(bpv, 4),
            'psnr_raw': round(float(r['full_psnr_db']), 2),
            'psnr_train': round(float(r['train_psnr_db']), 2),
            'psnr_heldout': round(float(r['held_out_psnr_db']), 2),
            'renderable': True,
        })
    return pd.DataFrame(rows)


def main():
    from arbol import aprint, asection

    import datasets as ds_module

    all_rows = []

    for ds_name in DATASETS:
        with asection(f"Dataset: {ds_name}"):
            volume, meta = ds_module.DATASETS[ds_name]()
            aprint(f"Shape: {volume.shape}, Range: [{volume.min():.3f}, {volume.max():.3f}]")

            # Generate held-out mask
            mask = generate_mask(volume.shape, MASK_FRACTION, SEED)
            n_heldout = mask.sum()
            aprint(f"Held-out mask: {n_heldout} voxels ({n_heldout/volume.size*100:.1f}%)")

            # Get noise floor for context
            nf = pd.read_csv(QUALITY_DIR / "noise_floor.tsv", sep='\t')
            nf_row = nf[nf['dataset'] == ds_name]
            noise_floor_psnr = float(nf_row['psnr_max_db'].values[0]) if not nf_row.empty else None
            if noise_floor_psnr:
                aprint(f"Noise floor: {noise_floor_psnr:.1f} dB")

            n_voxels = volume.size
            raw_bytes = n_voxels * 4

            # --- Quantization baselines ---
            for bits in [4, 6, 8, 10, 12, 16]:
                recon = quantize_volume(volume, bits)

                psnr_raw = compute_psnr(volume, recon)
                psnr_heldout = compute_psnr(volume, recon, mask)
                psnr_train = compute_psnr(volume, recon, ~mask)

                # Estimate compressed size with zstd
                import zstandard as zstd
                if bits <= 8:
                    compressed_data = zstd.ZstdCompressor(level=9).compress(
                        recon.astype(np.float32).tobytes()  # Store as float for fair comparison
                    )
                else:
                    compressed_data = zstd.ZstdCompressor(level=9).compress(
                        recon.astype(np.float32).tobytes()
                    )

                # Actually, for BPV calculation, use the quantized size
                if bits <= 8:
                    q_bytes = n_voxels * 1  # uint8
                else:
                    q_bytes = n_voxels * 2  # uint16
                # After zstd compression
                import zstandard as zstd
                q_data = np.clip(
                    np.round((volume - volume.min()) / (volume.max() - volume.min()) * ((1 << bits) - 1)),
                    0, (1 << bits) - 1
                )
                if bits <= 8:
                    q_data = q_data.astype(np.uint8)
                else:
                    q_data = q_data.astype(np.uint16)
                compressed = zstd.ZstdCompressor(level=9).compress(q_data.tobytes())
                bpv = len(compressed) * 8 / n_voxels

                row = {
                    'dataset': ds_name,
                    'method': f'quant{bits}b',
                    'knob': bits,
                    'bpv': round(bpv, 4),
                    'psnr_raw': round(psnr_raw, 2),
                    'psnr_train': round(psnr_train, 2),
                    'psnr_heldout': round(psnr_heldout, 2),
                    'renderable': False,
                }
                all_rows.append(row)
                aprint(f"  quant{bits}b: BPV={bpv:.3f}  raw={psnr_raw:.1f}  held-out={psnr_heldout:.1f}")

            # --- GSplats (from existing N2S analysis) ---
            gsplat_df = get_gsplat_n2s_data(ds_name)
            if not gsplat_df.empty:
                gsplat_df['dataset'] = ds_name
                for _, r in gsplat_df.iterrows():
                    all_rows.append(r.to_dict())
                    aprint(f"  gsplats-{r['knob']//1000}K: BPV={r['bpv']:.3f}  "
                           f"raw={r['psnr_raw']:.1f}  held-out={r['psnr_heldout']:.1f}")

            # Add noise floor reference
            if noise_floor_psnr:
                all_rows.append({
                    'dataset': ds_name,
                    'method': 'noise_floor',
                    'knob': 'ref',
                    'bpv': 0,
                    'psnr_raw': noise_floor_psnr,
                    'psnr_train': noise_floor_psnr,
                    'psnr_heldout': noise_floor_psnr,
                    'renderable': False,
                })

    df = pd.DataFrame(all_rows)
    out = RESULTS_DIR / "denoised_comparison.tsv"
    df.to_csv(out, sep='\t', index=False)
    print(f"\nSaved: {out}")

    # Print summary
    for ds in DATASETS:
        sub = df[df['dataset'] == ds]
        nf = sub[sub['method'] == 'noise_floor']
        nf_val = float(nf['psnr_heldout'].values[0]) if not nf.empty else 0

        print(f"\n{'='*70}")
        print(f"  {ds} (noise floor: {nf_val:.1f} dB)")
        print(f"{'='*70}")
        print(f"  {'Method':20s}  {'BPV':>6s}  {'Raw PSNR':>9s}  {'Held-out':>9s}  {'Diff':>6s}")
        for _, r in sub[sub['method'] != 'noise_floor'].sort_values('bpv').iterrows():
            diff = r['psnr_raw'] - r['psnr_heldout']
            print(f"  {r['method']:20s}  {r['bpv']:6.3f}  {r['psnr_raw']:9.1f}  "
                  f"{r['psnr_heldout']:9.1f}  {diff:+5.1f}")


if __name__ == '__main__':
    main()
