#!/usr/bin/env python3
"""Cross-validation-based compression comparison: GSplats vs H.265 video codec.

Applies the blind-spot cross-validation framework to both GSplats and H.265:
  1. Mask 5% of voxels with their median-donut neighborhood value
  2. Compress the masked volume (GSplats: fit; H.265: encode Z-stack as video)
  3. Decompress / render
  4. Evaluate on the original held-out voxels → measures true signal recovery

This is a fair comparison because both methods are spatially aware:
  - GSplats: global mixture model that interpolates across masked regions
  - H.265: block-based prediction + DCT that spatially interpolates

The "knob":
  - GSplats: splat count (1K-512K)
  - H.265: CRF (Constant Rate Factor, 0=lossless to 51=worst)
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
QUALITY_DIR = ANALYSIS_DIR.parent / "splat_count_vs_quality" / "results"
RESULTS_DIR = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ANALYSIS_DIR.parent / "splat_count_vs_quality"))

DATASETS = [
    'kidney_dapi', 'kidney_actin', 'organoid_ch0',
    'celegans_t100', 'tribolium',
    'opencell_map4_ch0', 'opencell_map4_ch1', 'cells3d_nuclei',
]

MASK_FRACTION = 0.05
SEED = 42

# H.265 CRF values to sweep (lower = better quality, larger file)
H265_CRF_VALUES = [51, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]


def compute_psnr(orig, recon, mask=None):
    if mask is not None:
        orig = orig[mask]
        recon = recon[mask]
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def generate_cv_mask(shape, fraction, seed):
    """Generate random blind-spot mask for cross-validation."""
    rng = np.random.RandomState(seed)
    return rng.random(shape) < fraction


def apply_median_donut(volume, mask):
    """Replace masked voxels with median of their 3D donut neighborhood.

    The donut excludes the center pixel — this is the cross-validation blind-spot.
    Uses a 3x3x3 neighborhood (26-connected, center excluded).
    """
    from itertools import product

    # Compute true donut median: 26 neighbors, excluding center
    shifts = list(product([-1, 0, 1], repeat=3))
    shifts.remove((0, 0, 0))

    neighbors = []
    for dz, dy, dx in shifts:
        neighbors.append(np.roll(np.roll(np.roll(volume, -dz, 0), -dy, 1), -dx, 2))
    donut_median = np.median(np.stack(neighbors, axis=0), axis=0).astype(volume.dtype)

    # Replace masked voxels with their donut-median
    masked_volume = volume.copy()
    masked_volume[mask] = donut_median[mask]
    return masked_volume


def compute_splat_bytes(n_splats, ndim=3):
    cholesky = ndim * (ndim + 1) // 2
    return n_splats * (ndim + 1 + cholesky) * 4


def encode_h265(volume, crf, tmpdir):
    """Encode a 3D volume as H.265 video (Z-slices = frames).

    Converts float32 [0,1] → 16-bit grayscale → H.265.
    Returns: compressed file size in bytes, decoded volume.
    """
    nz, ny, nx = volume.shape

    # Convert to 16-bit (H.265 supports 10-bit and 12-bit; we use 16-bit gray)
    # Actually ffmpeg gray16 input works best with rawvideo
    vol_uint16 = np.clip(volume * 65535, 0, 65535).astype(np.uint16)

    # Write raw frames
    raw_path = os.path.join(tmpdir, 'input.raw')
    vol_uint16.tofile(raw_path)

    # Encode with H.265
    h265_path = os.path.join(tmpdir, f'encoded_crf{crf}.mp4')
    cmd_encode = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'gray16le',
        '-s', f'{nx}x{ny}',
        '-r', '1',  # 1 fps (doesn't matter for quality)
        '-i', raw_path,
        '-c:v', 'libx265',
        '-preset', 'medium',
        '-x265-params', f'lossless={1 if crf == 0 else 0}',
        '-crf', str(crf),
        '-pix_fmt', 'gray12le',  # 12-bit internal for better quality
        h265_path,
    ]

    result = subprocess.run(cmd_encode, capture_output=True, timeout=120)
    if result.returncode != 0:
        # Try simpler encoding if gray12le not supported
        cmd_encode_simple = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'gray16le',
            '-s', f'{nx}x{ny}',
            '-r', '1',
            '-i', raw_path,
            '-c:v', 'libx265',
            '-preset', 'medium',
            '-crf', str(max(crf, 1)),  # crf=0 not supported without lossless
            '-pix_fmt', 'yuv420p',
            h265_path,
        ]
        result = subprocess.run(cmd_encode_simple, capture_output=True, timeout=120)
        if result.returncode != 0:
            return None, None

    compressed_size = os.path.getsize(h265_path)

    # Decode back
    decoded_path = os.path.join(tmpdir, 'decoded.raw')
    cmd_decode = [
        'ffmpeg', '-y',
        '-i', h265_path,
        '-f', 'rawvideo',
        '-pix_fmt', 'gray16le',
        decoded_path,
    ]
    result = subprocess.run(cmd_decode, capture_output=True, timeout=120)
    if result.returncode != 0:
        return compressed_size, None

    # Read decoded frames
    try:
        decoded_raw = np.fromfile(decoded_path, dtype=np.uint16)
        expected_size = nz * ny * nx
        if decoded_raw.size >= expected_size:
            decoded_raw = decoded_raw[:expected_size]
            decoded_vol = decoded_raw.reshape(nz, ny, nx).astype(np.float32) / 65535.0
        else:
            # yuv420p decode gives different size - need to handle
            return compressed_size, None
    except Exception:
        return compressed_size, None

    return compressed_size, decoded_vol


def get_gsplat_cv_data(ds):
    """Get existing GSplat cross-validation results."""
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
            'compressed_bytes': splat_bytes,
            'psnr_raw': round(float(r['full_psnr_db']), 2),
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
            nz, ny, nx = volume.shape
            n_voxels = volume.size
            aprint(f"Shape: {volume.shape}")

            # Generate cross-validation blind-spot mask
            mask = generate_cv_mask(volume.shape, MASK_FRACTION, SEED)
            aprint(f"Held-out: {mask.sum()} voxels ({mask.sum()/n_voxels*100:.1f}%)")

            # Create masked volume (blind-spot filled with median-donut)
            masked_volume = apply_median_donut(volume, mask)
            aprint("Masked volume ready (median-donut fill)")

            # Noise floor
            nf = pd.read_csv(QUALITY_DIR / "noise_floor.tsv", sep='\t')
            nf_row = nf[nf['dataset'] == ds_name]
            nf_psnr = float(nf_row['psnr_max_db'].values[0]) if not nf_row.empty else None

            # --- H.265 sweep ---
            with asection("H.265 CRF sweep"):
                with tempfile.TemporaryDirectory() as tmpdir:
                    for crf in H265_CRF_VALUES:
                        t0 = time.time()
                        comp_size, decoded = encode_h265(masked_volume, crf, tmpdir)
                        elapsed = time.time() - t0

                        if decoded is None or comp_size is None:
                            aprint(f"  CRF={crf}: FAILED")
                            continue

                        bpv = comp_size * 8 / n_voxels
                        psnr_raw = compute_psnr(volume, decoded)
                        psnr_heldout = compute_psnr(volume, decoded, mask)

                        row = {
                            'dataset': ds_name,
                            'method': 'h265',
                            'knob': crf,
                            'bpv': round(bpv, 4),
                            'compressed_bytes': comp_size,
                            'psnr_raw': round(psnr_raw, 2),
                            'psnr_heldout': round(psnr_heldout, 2),
                            'renderable': False,
                        }
                        all_rows.append(row)
                        aprint(f"  CRF={crf:2d}: BPV={bpv:.3f}  raw={psnr_raw:.1f}  "
                               f"held-out={psnr_heldout:.1f}  size={comp_size/1e3:.0f}KB  "
                               f"time={elapsed:.1f}s")

            # --- GSplats (from existing cross-validation analysis) ---
            with asection("GSplats (existing CV)"):
                gsplat_df = get_gsplat_cv_data(ds_name)
                if not gsplat_df.empty:
                    gsplat_df['dataset'] = ds_name
                    for _, r in gsplat_df.iterrows():
                        all_rows.append(r.to_dict())
                    aprint(f"  {len(gsplat_df)} splat counts loaded")

            # Noise floor reference
            if nf_psnr and nf_psnr < 100:
                all_rows.append({
                    'dataset': ds_name,
                    'method': 'noise_floor',
                    'knob': 'ref',
                    'bpv': 0,
                    'compressed_bytes': 0,
                    'psnr_raw': nf_psnr,
                    'psnr_heldout': nf_psnr,
                    'renderable': False,
                })

    df = pd.DataFrame(all_rows)
    out = RESULTS_DIR / "n2s_baselines.tsv"
    df.to_csv(out, sep='\t', index=False)
    print(f"\nSaved: {out}")

    # Summary
    for ds in DATASETS:
        sub = df[df['dataset'] == ds]
        nf = sub[sub['method'] == 'noise_floor']
        nf_val = float(nf['psnr_heldout'].values[0]) if not nf.empty else 0

        print(f"\n{'='*75}")
        print(f"  {ds} (noise floor: {nf_val:.1f} dB)")
        print(f"{'='*75}")
        print(f"  {'Method':15s}  {'Knob':>6s}  {'BPV':>7s}  {'Raw':>6s}  {'Held-out':>8s}  {'Render?':>7s}")
        methods = sub[sub['method'] != 'noise_floor'].sort_values('bpv')
        for _, r in methods.iterrows():
            knob = str(r['knob'])
            render = 'YES' if r.get('renderable', False) else 'no'
            print(f"  {r['method']:15s}  {knob:>6s}  {r['bpv']:7.3f}  "
                  f"{r['psnr_raw']:6.1f}  {r['psnr_heldout']:8.1f}  {render:>7s}")


if __name__ == '__main__':
    main()
