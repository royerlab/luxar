# Deduplication Bottleneck in Seed Generation

## Problem

After the edge detection seeding was optimized (~12x speedup), deduplication has become the new bottleneck for high seed counts. The deduplication step removes seeds that are too close together, but its runtime grows superlinearly with seed count.

## Observed Timings (OpenCell MAP4, 51x600x600 volume)

### With optimized edge detection (current state):

| Seeds Requested | Edge Seeds | Edge Detection | Deduplication | Seeding Total | Optimization | Total |
|----------------:|-----------:|---------------:|--------------:|--------------:|-------------:|------:|
| 256,000 | 153,600 | 50.8s | 1.36 min | 2.2 min | 160.3s | ~5 min |
| 512,000 | 201,171* | 1.78 min | 3.30 min | 5.1 min | 157.4s | ~8 min |

*Volume saturated — only 201K of the requested 307K edge seeds could be generated.

### Deduplication scaling:

| Input Seeds | Dedup Time | Scaling Factor |
|------------:|-----------:|---------------:|
| 889 | 32 ms | — |
| 2,082 | 75 ms | 2.3x seeds → 2.3x time |
| 3,858 | 162 ms | 1.9x seeds → 2.2x time |
| 8,268 | 369 ms | 2.1x seeds → 2.3x time |
| 16,656 | 856 ms | 2.0x seeds → 2.3x time |
| 254,724 | 1.36 min | — |
| 398,687 | 3.30 min | 1.6x seeds → 2.4x time |

The scaling is approximately **O(N^1.5) to O(N^2)** — each doubling of seed count roughly 2.3x the deduplication time.

## Where to Look

The deduplication code is called during seed generation. Key locations:

- **Seed generation entry point**: `packages/luxar/src/luxar/gsplats/seeds/` — look for `generate_seeds()` or the `auto` seed method
- **Deduplication function**: likely in `packages/luxar/src/luxar/gsplats/seeds/` — search for `deduplication`, `deduplicate`, or `min_distance`
- **Call pattern**: Edge seeds + grid seeds are merged, then deduplication removes points within a minimum distance threshold

## Potential Optimizations

1. **KD-tree based deduplication**: Replace brute-force pairwise distance with scipy's `cKDTree.query_pairs()` or `query_ball_tree()` — reduces from O(N^2) to O(N log N)
2. **Grid-based spatial hashing**: Bin seeds into a voxel grid, only compare within neighboring bins — O(N) expected time
3. **GPU-accelerated deduplication**: The edge detection was sped up by moving to GPU; deduplication could benefit similarly
4. **Skip deduplication for large counts**: At 512K seeds in a 18.4M voxel volume, the average inter-seed distance is already ~3.3 voxels. Deduplication may remove very few seeds, making it near-unnecessary

## Impact

At 256K+ seeds, deduplication takes longer than edge detection. For the paper analysis (manuscript/analysis/splat_count_vs_quality/), this is a one-time cost and tolerable. But for interactive/CLI use cases (`luxar gsplat fit --seeds 256000`), the 3+ minute deduplication overhead is significant.

## Related

- Edge detection speedup was done by another agent in this session
- The seeding pipeline is in `packages/luxar/src/luxar/gsplats/seeds/`
