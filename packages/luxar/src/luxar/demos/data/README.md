# Demo Data Files

This directory contains precomputed data files for Luxar demos. Some files are stored using **Git Large File Storage (Git LFS)** to keep the repository size manageable.

## Files in This Directory

### Precomputed Gaussian Splats (Git LFS)

These `.gsplats.zarr.zip` files contain pre-fitted Gaussian splats from real microscopy data in compressed zarr format:

- **organoids_gsplats_ch0.gsplats.zarr.zip** (~609 KB)
  - Channel 0 from IDR dataset 6001240 (intestinal organoid)
  - Fluorescent marker channel (magenta in demo)
  - Pre-fitted using full compute pipeline
  - ~20,000 splats compressed with Hilbert ordering

- **organoids_gsplats_ch1.gsplats.zarr.zip** (~630 KB)
  - Channel 1 from IDR dataset 6001240 (intestinal organoid)
  - DAPI nuclear stain channel (cyan in demo)
  - Pre-fitted using full compute pipeline
  - ~20,000 splats compressed with Hilbert ordering

**Data Source:** Image Data Resource (IDR) study idr0062, Image 6001240
**Original Authors:** Prisca Liberali lab, FMI
**Citation:** Blin et al. (2019) + Williams et al. (2017) Nature Methods 14(8):775-781

### Milky Way Data (Git LFS)

- **milky_way_gaia_3m.zarr.zip** (~133 bytes)
  - Gaia DR3 star catalog subset (3 million stars)

- **milky_way_gaia_8m.zarr.zip** (~134 bytes)
  - Gaia DR3 star catalog subset (8 million stars)

## Git LFS Setup

### What is Git LFS?

Git Large File Storage (LFS) replaces large files with lightweight pointers in the Git repository, storing the actual file contents on a remote server. This keeps the repository fast to clone while still providing access to large files.

### Installing Git LFS

If you don't have Git LFS installed, you'll see small pointer files instead of the actual data.

**Install Git LFS:**

```bash
# macOS (Homebrew)
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Other systems: https://git-lfs.github.com/
```

**Initialize Git LFS** (one-time setup):

```bash
git lfs install
```

### Pulling LFS Files

After installing Git LFS, pull the actual data files:

```bash
# Pull LFS files for the entire repository
git lfs pull

# Or pull specific files only
git lfs pull --include="packages/luxar/src/luxar/demos/data/*.gsplats.zarr.zip"
```

### Verifying LFS Files

Check if you have the actual files (not pointers):

```bash
# Should show ~609-630 KB for .gsplats.zarr.zip files
ls -lh packages/luxar/src/luxar/demos/data/*.gsplats.zarr.zip

# Check LFS status
git lfs ls-files
```

If the files are very small (< 1 KB), they're pointer files and you need to run `git lfs pull`.

## Using Precomputed Data

### Quick Start Demo (Recommended)

Use precomputed data for fast demos without network/compute overhead:

```bash
cd packages/luxar/src/luxar/demos
python demo_gsplats_3d_organoid_multichannel_precomputed.py
```

This demo:
- Loads precomputed gsplats from `.gsplats.zarr.zip` files
- No network required (after Git LFS pull)
- No GPU computation required
- Fast visualization (~2-3 seconds)

### Full Compute Pipeline

To recompute gsplats from scratch (requires network + GPU):

```bash
python demo_gsplats_3d_organoid_multichannel_from_idr.py
```

This demo:
- Fetches raw data from IDR (requires internet)
- Performs full GSplat fitting (requires GPU, ~5-15 min)
- Saves results to cache for reuse

## Troubleshooting

### "Precomputed gsplat file not found"

You need to pull Git LFS files:

```bash
git lfs install
git lfs pull
```

### "git: 'lfs' is not a git command"

Install Git LFS first (see installation instructions above).

### Files are Small Pointers (~133 bytes)

You have pointer files, not the actual data. Run:

```bash
git lfs pull
```

### Large Clone Size

If cloning is slow, you can clone without LFS files initially:

```bash
# Clone without LFS files
GIT_LFS_SKIP_SMUDGE=1 git clone <repo>

# Later, pull only what you need
cd luxar
git lfs pull --include="packages/luxar/src/luxar/demos/data/*.gsplats.zarr.zip"
```

## Adding New LFS Files

If you're adding new large data files to this directory:

1. **Update `.gitattributes`** (at repository root):
   ```
   packages/luxar/src/luxar/demos/data/*.npz filter=lfs diff=lfs merge=lfs -text
   ```

2. **Add the file**:
   ```bash
   git add packages/luxar/src/luxar/demos/data/my_data.npz
   git commit -m "Add precomputed data for demo"
   ```

3. **Verify LFS tracking**:
   ```bash
   git lfs ls-files  # Should show your file
   ```

## File Format Details

### `.gsplats.zarr.zip` Files

Compressed Gaussian splat datasets in Luxar's native zarr format. Contains:

- Splat centers, amplitudes, covariance matrices (Cholesky factors), sharpness
- Spatially ordered for efficient streaming (Hilbert curve ordering)
- Quantized encoding for compact storage
- Metadata including fitting statistics and provenance

**Load programmatically:**
```python
from luxar.gsplats.gsplat_data import GSplatData

# Automatically extracts and loads compressed archive
data = GSplatData.load("organoids_gsplats_ch0.gsplats.zarr.zip")

print(f"Loaded {len(data.amplitudes):,} splats")
print(f"Dimensions: {data.centers.shape[1]}D")
```

**Quick view with CLI:**
```bash
# View in Luxar web viewer
luxar gsplat view organoids_gsplats_ch0.gsplats.zarr.zip

# Inspect in napari
luxar gsplat napari organoids_gsplats_ch0.gsplats.zarr.zip

# Prune to reduce size
luxar gsplat prune input.gsplats.zarr.zip output.gsplats.zarr.zip \
    --method cumulative --retention 0.95
```

### `.zarr.zip` Files (Point Clouds)

Compressed Zarr stores for large point cloud datasets. Extract and serve with:

```bash
luxar serve my_data.zarr.zip --viewer
```

## License & Attribution

The organoid microscopy data is from:
- **Source:** Image Data Resource (IDR), study idr0062, Image 6001240
- **Original Research:** Prisca Liberali lab, FMI
- **Citation:** Please cite both Blin et al. (2019) and Williams et al. (2017) Nature Methods

See individual demo scripts for full citation information.
