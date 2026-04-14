#!/usr/bin/env python3
"""Compression comparison: Gaussian splats at N2S-optimal count vs raw volume.

For each dataset, finds the Noise2Self-optimal splat count (held-out PSNR peak),
then computes file sizes and compression metrics at that operating point.
For signal-limited datasets (no N2S peak), uses the count at which quality plateaus.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
QUALITY_DIR = ANALYSIS_DIR.parent / "splat_count_vs_quality" / "results"
RESULTS_DIR = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = [
    'kidney_dapi', 'kidney_actin',
    'opencell_map4_ch0', 'opencell_map4_ch1',
    'organoid_ch0', 'celegans_t100', 'tribolium',
    'cells3d_nuclei', 'cells3d_membrane',
    'opencell_lmnb1_ch0', 'opencell_lmnb1_ch1',
    'acto3d_heart_nuclei',
]


def parse_shape(shape_str):
    return tuple(int(x) for x in shape_str.split('x'))


def compute_splat_size_bytes(n_splats, ndim=3):
    cholesky_params = ndim * (ndim + 1) // 2
    params_per_splat = ndim + 1 + cholesky_params
    return n_splats * params_per_splat * 4


def find_n2s_optimal(ds):
    """Find the N2S-optimal splat count (held-out PSNR peak).

    Returns (seeds_requested, held_out_psnr, is_genuine_peak).
    If no genuine peak (last point is max), returns last point with is_genuine_peak=False.
    """
    n2s_path = QUALITY_DIR / f"{ds}_n2s" / "metrics_n2s.tsv"
    if not n2s_path.exists():
        return None, None, False

    n2s = pd.read_csv(n2s_path, sep='\t').sort_values('seeds_requested')
    peak_idx = n2s['held_out_psnr_db'].idxmax()
    last_idx = n2s.index[-1]

    seeds = int(n2s.loc[peak_idx, 'seeds_requested'])
    psnr = float(n2s.loc[peak_idx, 'held_out_psnr_db'])
    is_genuine = peak_idx != last_idx

    return seeds, psnr, is_genuine


def main():
    nf_path = QUALITY_DIR / "noise_floor.tsv"
    if not nf_path.exists():
        print(f"ERROR: {nf_path} not found")
        sys.exit(1)
    nf_df = pd.read_csv(nf_path, sep='\t')

    rows = []
    for ds in DATASETS:
        metrics_path = QUALITY_DIR / ds / "metrics.tsv"
        if not metrics_path.exists():
            print(f"  Skipping {ds} (no metrics)")
            continue

        metrics = pd.read_csv(metrics_path, sep='\t')

        nf_row = nf_df[nf_df['dataset'] == ds]
        if nf_row.empty:
            continue

        shape_str = nf_row['volume_shape'].values[0]
        shape = parse_shape(shape_str)
        n_voxels = int(np.prod(shape))
        ndim = len(shape)

        raw_bytes = n_voxels * 4
        raw_mb = raw_bytes / 1e6

        # Find N2S-optimal splat count
        opt_seeds, opt_held_out_psnr, is_genuine = find_n2s_optimal(ds)

        if opt_seeds is None:
            print(f"  Skipping {ds} (no N2S data)")
            continue

        # Get metrics at the N2S-optimal count
        row = metrics[metrics['seeds_requested'] == opt_seeds]
        if row.empty:
            # Find closest available count
            closest_idx = (metrics['seeds_requested'] - opt_seeds).abs().idxmin()
            row = metrics.loc[[closest_idx]]

        n_splats = int(row['n_splats_final'].values[0])
        psnr = float(row['psnr_db'].values[0])
        ssim = float(row['ssim'].values[0])
        fit_time = float(row['fit_time_s'].values[0])

        splat_bytes = compute_splat_size_bytes(n_splats, ndim)
        splat_mb = splat_bytes / 1e6

        cr_vs_raw = raw_mb / splat_mb if splat_mb > 0 else 0
        bpv_splat = (splat_bytes * 8) / n_voxels

        rows.append({
            'dataset': ds,
            'shape': shape_str,
            'n_voxels_M': round(n_voxels / 1e6, 1),
            'n2s_optimal_seeds': opt_seeds,
            'n2s_genuine_peak': is_genuine,
            'n_splats': n_splats,
            'psnr_db': round(psnr, 1),
            'ssim': round(ssim, 3),
            'held_out_psnr': round(opt_held_out_psnr, 1) if opt_held_out_psnr else None,
            'raw_MB': round(raw_mb, 1),
            'splat_MB': round(splat_mb, 2),
            'cr_vs_raw': round(cr_vs_raw, 1),
            'bpv_splat': round(bpv_splat, 3),
            'fit_time_s': round(fit_time, 1),
        })

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "compression_at_n2s_optimal.tsv"
    df.to_csv(out_path, sep='\t', index=False)
    print(f"Saved: {out_path}")

    # Also save the multi-count data for the bits-per-voxel curves
    bpv_rows = []
    for ds in DATASETS:
        metrics_path = QUALITY_DIR / ds / "metrics.tsv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path, sep='\t')
        nf_row = nf_df[nf_df['dataset'] == ds]
        if nf_row.empty:
            continue
        shape = parse_shape(nf_row['volume_shape'].values[0])
        n_voxels = int(np.prod(shape))
        ndim = len(shape)

        for _, r in metrics.iterrows():
            n_splats = int(r['n_splats_final'])
            splat_bytes = compute_splat_size_bytes(n_splats, ndim)
            bpv_rows.append({
                'dataset': ds,
                'seeds_requested': int(r['seeds_requested']),
                'n_splats': n_splats,
                'psnr_db': float(r['psnr_db']),
                'bpv_splat': round((splat_bytes * 8) / n_voxels, 4),
                'cr_vs_raw': round((n_voxels * 4) / splat_bytes, 1) if splat_bytes > 0 else 0,
            })

    bpv_df = pd.DataFrame(bpv_rows)
    bpv_path = RESULTS_DIR / "bits_per_voxel_all.tsv"
    bpv_df.to_csv(bpv_path, sep='\t', index=False)
    print(f"Saved: {bpv_path}")

    # Print summary
    print(f"\n{'='*90}")
    print("  Compression at N2S-optimal splat count")
    print(f"{'='*90}")
    print(f"{'Dataset':25s}  {'N2S opt':>7s}  {'Peak?':>5s}  {'Splats':>7s}  "
          f"{'Raw MB':>7s}  {'Splat MB':>8s}  {'CR':>6s}  {'PSNR':>5s}  {'BPV':>6s}")
    print(f"{'-'*90}")
    for _, r in df.iterrows():
        peak = 'Yes' if r['n2s_genuine_peak'] else 'No'
        print(f"  {r['dataset']:23s}  {r['n2s_optimal_seeds']:>7d}  {peak:>5s}  "
              f"{r['n_splats']:>7d}  {r['raw_MB']:>7.1f}  {r['splat_MB']:>8.2f}  "
              f"{r['cr_vs_raw']:>5.1f}x  {r['psnr_db']:>5.1f}  {r['bpv_splat']:>6.3f}")


if __name__ == '__main__':
    main()
