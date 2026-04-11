# Seeding Performance Issue: Edge Detection Scales Superlinearly

## Summary

Edge-based seed generation becomes the dominant bottleneck for large splat counts, consuming up to **89%+ of total fitting time** at 128K seeds and growing worse. The optimization (gradient descent) step is comparatively fast and scales sublinearly thanks to early stopping. This makes the seeding algorithm the practical ceiling for fitting performance.

## Evidence

Measured on OpenCell MAP4 volume (51 x 600 x 600, float32) with `seed_method="auto"` (60% edge + 40% grid), `n_iters=10000`, `early_stop_patience=500`:

| Seeds Requested | Edge Seeds | Edge Detection Time | Optimization Time | Total Time | Seeding % |
|----------------:|-----------:|--------------------:|------------------:|-----------:|----------:|
| 1,000 | 600 | 0.6s | 15s | 17s | 12% |
| 2,000 | 1,200 | 0.4s | 12s | 14s | 12% |
| 4,000 | 2,400 | 0.8s | 14s | 16s | 14% |
| 8,000 | 4,800 | 2.3s | 20s | 24s | 17% |
| 16,000 | 9,600 | 8.4s | 57s | 68s | 16% |
| 32,000 | 19,200 | 34.5s | 69s | 107s | 36% |
| 64,000 | 38,400 | 2.5 min | 93s | 4.1 min | 63% |
| 128,000 | 76,800 | 10.8 min | 81s | 12.5 min | 89% |

Extrapolated: 256K seeds (~153K edges) and 512K seeds (~307K edges) will be **95%+** seeding time.

## Scaling Analysis

Edge detection time vs edge seed count grows **superlinearly** (~O(n^1.5) or worse):

- 600 edges → 0.6s
- 9,600 edges (16x) → 8.4s (14x) — roughly linear here
- 38,400 edges (4x from 9.6K) → 150s (18x) — superlinear
- 76,800 edges (2x) → 645s (4.3x) — superlinear

Meanwhile optimization time grows slowly: 15s → 93s across 1K→64K seeds (128x more seeds, only 6x slower), and actually *decreases* from 64K to 128K (93s → 81s) because more seeds = better initial coverage = faster convergence.

## Where to Look

The seeding code lives in:

```
packages/luxar/src/luxar/gsplats/seeds/
```

Key files:
- `edge_seeds.py` — The edge-based seed generator (this is the bottleneck)
- `grid_seeds.py` — Grid-based seeds (fast, O(n), not an issue)
- `auto_seeds.py` — The "auto" method that combines edge (60%) + grid (40%)
- `__init__.py` — `generate_seeds()` entry point

The edge seeding pipeline is roughly:
1. **Sobel gradient computation** — Compute gradient magnitude across the volume (GPU-accelerated, fast)
2. **Peak detection / local maxima** — Find local maxima in the gradient field (potentially slow for dense volumes)
3. **Top-K selection** — Select the top-K gradient peaks as seed locations
4. **Interpolation** — Subpixel refinement of seed positions
5. **Deduplication** — Remove seeds that are too close to each other

The bottleneck is likely in steps 2-3 (peak detection + top-K selection), which may be using CPU-bound operations (scipy `argpartition`, `ndi.maximum_filter`, etc.) on large arrays, or step 5 (deduplication) which could involve O(n^2) pairwise distance checks.

## Hardware Context

- GPU: NVIDIA RTX 3090 Ti (24GB)
- Volume: 51 x 600 x 600 = 18.36M voxels (float32, ~70MB)
- The volume fits easily in GPU memory, so any CPU-bound step is a missed opportunity

## What a Fix Should Achieve

1. **Target**: Edge detection for 128K seeds should take <30s (currently 645s — need ~20x speedup)
2. **Constraint**: Seed quality must remain comparable — same or better PSNR at same splat count
3. **Approach ideas** (non-exhaustive, the agent should investigate the actual code):
   - Move peak detection / top-K to GPU (torch or cupy)
   - Replace pairwise deduplication with spatial hashing or KD-tree
   - Use strided/downsampled gradient field for initial candidate generation, then refine
   - Batch the peak detection into tiles to reduce memory pressure
   - Profile first (`cProfile` or `line_profiler`) to confirm which substep dominates

## How to Benchmark

```bash
# Run the seeding benchmark directly
hatch run python -c "
import numpy as np
from luxar.gsplats.seeds import generate_seeds
import time

# Load test volume
import tifffile
data = tifffile.imread('$HOME/.cache/luxar/gsplats_opencell_map4/opencell_map4_stack.tif')
V = data[:, 0].astype(np.float32)
V = (V - V.min()) / (V.max() - V.min() + 1e-8)

for n in [1000, 4000, 16000, 64000, 128000]:
    t0 = time.time()
    seeds = generate_seeds(V, n_seeds=n, method='edges', device='cuda')
    t1 = time.time()
    print(f'  {n:>7,d} edge seeds: {t1-t0:.2f}s  ({seeds.shape[0]} returned)')
"
```

## How to Validate

After any fix, re-run the analysis to confirm seed quality is preserved:

```bash
# Clear old results and re-run
rm -rf manuscript/analysis/splat_count_vs_quality/results/opencell_map4_ch0/
hatch run python manuscript/analysis/splat_count_vs_quality/run_analysis.py
```

Compare PSNR/SSIM at each splat count — they should be within ~0.5 dB of the baseline values listed in the evidence table above.
