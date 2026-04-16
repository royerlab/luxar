#!/usr/bin/env python3
"""Compression baseline rate-distortion comparison.

For each method, varies the quality "knob" to generate rate-distortion curves
comparable to the GSplat splat-count sweep.

Methods and their knobs:
- GSplats: splat count (1K-512K) → already computed
- Quantization+zstd: bit depth (4, 6, 8, 10, 12, 16 bits)
- Blosc+bitshuffle: lossless (single point, perfect quality)

The key architectural distinction: GSplats are RENDERABLE at their compressed
size (no decompression). All other methods require full decompression to voxels
before any rendering can happen.
"""

import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
QUALITY_DIR = ANALYSIS_DIR.parent / "splat_count_vs_quality" / "results"
RESULTS_DIR = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ANALYSIS_DIR.parent))
sys.path.insert(0, str(ANALYSIS_DIR.parent / "splat_count_vs_quality"))
from _shared import compute_psnr, compute_splat_bytes  # noqa: E402

DATASETS = ['kidney_dapi', 'organoid_ch0', 'celegans_t100', 'tribolium', 'opencell_map4_ch0']


def sweep_quantization(volume):
    """Vary quantization bit depth: 4, 6, 8, 10, 12, 16 bits → zstd level 9."""
    import zstandard as zstd

    results = []
    v_min, v_max = float(volume.min()), float(volume.max())
    rng = v_max - v_min if v_max > v_min else 1.0

    for bits in [4, 6, 8, 10, 12, 16]:
        max_val = (1 << bits) - 1

        # Quantize
        t0 = time.time()
        quantized = np.clip(
            np.round((volume - v_min) / rng * max_val), 0, max_val
        )
        if bits <= 8:
            quantized = quantized.astype(np.uint8)
        else:
            quantized = quantized.astype(np.uint16)

        data = quantized.tobytes()
        header = struct.pack('ffI', v_min, v_max, bits)  # 12 bytes

        cctx = zstd.ZstdCompressor(level=9)
        compressed = cctx.compress(header + data)
        compress_time = time.time() - t0

        # Decompress
        t0 = time.time()
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)
        hdr = raw[:12]
        v_min_r, v_max_r, bits_r = struct.unpack('ffI', hdr)
        dtype = np.uint8 if bits_r <= 8 else np.uint16
        recon_q = np.frombuffer(raw[12:], dtype=dtype).reshape(volume.shape)
        recon = recon_q.astype(np.float32) / max_val * (v_max_r - v_min_r) + v_min_r
        decompress_time = time.time() - t0

        psnr = compute_psnr(volume, recon)

        results.append({
            'method': f'quant{bits}b-zstd',
            'knob': bits,
            'type': 'lossy',
            'compressed_bytes': len(compressed),
            'compress_time_s': round(compress_time, 3),
            'decompress_time_s': round(decompress_time, 3),
            'psnr_db': round(psnr, 1),
            'renderable': False,
        })

    return results


def sweep_blosc_lossless(volume):
    """Blosc with different codecs and filters - lossless baseline."""
    import blosc2

    results = []
    data = volume.tobytes()

    configs = [
        ('blosc-lz4-shuffle', blosc2.Codec.LZ4, blosc2.Filter.SHUFFLE, 5),
        ('blosc-zstd-shuffle', blosc2.Codec.ZSTD, blosc2.Filter.SHUFFLE, 5),
        ('blosc-zstd-bitshuffle', blosc2.Codec.ZSTD, blosc2.Filter.BITSHUFFLE, 9),
    ]

    for name, codec, filt, clevel in configs:
        t0 = time.time()
        compressed = blosc2.compress(data, typesize=4, clevel=clevel,
                                      filter=filt, codec=codec)
        compress_time = time.time() - t0

        t0 = time.time()
        _ = blosc2.decompress(compressed)
        decompress_time = time.time() - t0

        results.append({
            'method': name,
            'knob': 'lossless',
            'type': 'lossless',
            'compressed_bytes': len(compressed),
            'compress_time_s': round(compress_time, 3),
            'decompress_time_s': round(decompress_time, 3),
            'psnr_db': float('inf'),
            'renderable': False,
        })

    return results


def get_gsplat_curve(ds):
    """Get the full GSplat rate-distortion curve from existing analysis."""
    metrics_path = QUALITY_DIR / ds / "metrics.tsv"
    if not metrics_path.exists():
        return []

    nf = pd.read_csv(QUALITY_DIR / "noise_floor.tsv", sep='\t')
    nf_row = nf[nf['dataset'] == ds]
    if nf_row.empty:
        return []

    shape = tuple(int(x) for x in nf_row['volume_shape'].values[0].split('x'))
    ndim = len(shape)

    metrics = pd.read_csv(metrics_path, sep='\t')
    results = []
    for _, row in metrics.iterrows():
        n_splats = int(row['n_splats_final'])
        splat_bytes = compute_splat_bytes(n_splats, ndim)
        results.append({
            'method': 'gsplats',
            'knob': int(row['seeds_requested']),
            'type': 'lossy+renderable',
            'compressed_bytes': splat_bytes,
            'compress_time_s': round(float(row['fit_time_s']), 1),
            'decompress_time_s': 0.0,
            'psnr_db': round(float(row['psnr_db']), 1),
            'renderable': True,
        })

    return results


def main():
    import datasets as ds_module

    all_rows = []

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        volume, meta = ds_module.DATASETS[ds_name]()
        raw_bytes = volume.nbytes
        print(f"  Shape: {volume.shape}, Raw: {raw_bytes/1e6:.1f} MB")

        # Quantization sweep (the "knob" is bit depth)
        print("  Quantization sweep (4-16 bits)...")
        quant_results = sweep_quantization(volume)

        # Lossless baselines (single points at infinite PSNR)
        print("  Lossless baselines...")
        lossless_results = sweep_blosc_lossless(volume)

        # GSplat curve (from existing data, "knob" is splat count)
        print("  GSplat curve...")
        gsplat_results = get_gsplat_curve(ds_name)

        all_results = quant_results + lossless_results + gsplat_results
        for r in all_results:
            r['dataset'] = ds_name
            r['raw_bytes'] = raw_bytes
            r['compression_ratio'] = round(raw_bytes / r['compressed_bytes'], 1) if r['compressed_bytes'] > 0 else 0
            r['bpv'] = round(r['compressed_bytes'] * 8 / (raw_bytes / 4), 4)  # bits per voxel
            all_rows.append(r)

        # Summary
        print(f"\n  {'Method':25s}  {'CR':>6s}  {'PSNR':>7s}  {'BPV':>6s}  {'Comp':>6s}  {'Decomp':>7s}  {'Render?':>7s}")
        for r in sorted(all_results, key=lambda x: -x['compression_ratio']):
            p = f"{r['psnr_db']:.1f}" if r['psnr_db'] < 200 else "lossless"
            print(f"  {r['method']:25s}  {r['compression_ratio']:5.1f}x  {p:>7s}  {r['bpv']:>6.3f}  "
                  f"{r['compress_time_s']:5.2f}s  {r['decompress_time_s']:6.3f}s  "
                  f"{'YES' if r['renderable'] else 'no':>7s}")

    df = pd.DataFrame(all_rows)
    out = RESULTS_DIR / "compression_baselines.tsv"
    df.to_csv(out, sep='\t', index=False)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
