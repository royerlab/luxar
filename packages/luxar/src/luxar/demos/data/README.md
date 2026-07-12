# Demo Data Files

This directory contains precomputed data files for Luxar GSplat demos. All files are stored using **Git Large File Storage (Git LFS)** to keep the repository size manageable.

## Directory Structure

### Precomputed Gaussian Splats (Git LFS)

Each subdirectory contains pre-fitted `.gsplats.zarr.zip` files for one demo:

| Directory | Demo | Contents |
|-----------|------|----------|
| `gsplats_multichannel/` | 3D organoid multi-channel | 2 channel files (~1.1 MB total) |
| `gsplats_dapi/` | 3D organoid DAPI nuclei | 1 file (~365 KB) |
| `gsplats_tribolium/` | 3D Tribolium embryo | 1 file (~1.7 MB) |
| `gsplats_cells3d/` | 4D cells3d multi-channel | 2 channel files (~1.0 MB total) |
| `gsplats_kidney/` | 4D kidney multi-channel | 3 channel files (~4.9 MB total) |
| `gsplats_zebrafish/` | 4D zebrafish timelapse | 1 bundle zip with 64 frames (~11 MB) |
| `gsplats_acto3d_heart/` | 3D mouse embryo heart (Acto3D) | 3 channel files (~3.5 MB total) |
| `gsplats_opencell_map4/` | 3D OpenCell MAP4 (cytoskeleton) | 2 channel files (~4.0 MB total) |
| `gsplats_cmu1_pathology/` | 2D CMU-1 pathology (H&E) | 3 channel files (pending) |
| `gsplats_celegans/` | 4D C. elegans tracking | 1 bundle zip with 400 timepoints (~64 MB) |
| _(not bundled)_ `gsplats_neuromast_2ch` | 4D two-channel neuromast timelapse | 2 channel `.gsplats.zarr` (~220 MB) — **local-only, not in Git LFS yet** |

> **Note — `demo_gsplats_4d_neuromast_2ch.py` data is not hosted yet.** Its two
> pre-fit channels (~220 MB total) are too large for the current tiny-fit LFS
> bundles and are kept in a local store on the author's machine
> (`~/luxar_demo_data/gsplats_neuromast_2ch/`, override with
> `$LUXAR_NEUROMAST_DATA_DIR`). Outstanding follow-up: upload them to the demo
> data host and switch the demo to `load_precomputed_gsplats` like the others.

### Other Data Files (top level)

- `milky_way_gaia_3m.zarr.zip` — Gaia DR3 star catalog (3M stars)
- `milky_way_gaia_8m.zarr.zip` — Gaia DR3 star catalog (8M stars)
- `3d_umap_coords_human.parquet` — Human cell UMAP coordinates
- `3d_umap_coords_mouse.parquet` — Mouse cell UMAP coordinates

## How Demos Use This Data

All GSplat demos follow a unified pattern:

1. **Default**: Load precomputed data from this directory (fast, no GPU needed)
2. **`--recompute`**: Fetch raw data and fit from scratch (slow, needs GPU)

The precomputed data is automatically copied to `~/.cache/luxar/` on first use, so subsequent runs are even faster.

## Git LFS Setup

### Installing Git LFS

```bash
# macOS (Homebrew)
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs
```

### Pulling LFS Files

```bash
git lfs install          # One-time setup
git lfs pull             # Download all LFS files
```

### Verifying LFS Files

If demo files are very small (< 1 KB), they're pointer files — run `git lfs pull`.

```bash
# Should show actual file sizes, not ~130 bytes
ls -lh packages/luxar/src/luxar/demos/data/gsplats_tribolium/
```

## File Formats

### `.gsplats.zarr.zip`
Compressed Gaussian splat datasets containing centers, amplitudes, Cholesky factors, and metadata. Load with:

```python
from luxar.gsplats.gsplat_data import GSplatData
data = GSplatData.load("file.gsplats.zarr.zip")
```

### Bundle `.zip` (timelapse demos)
Outer zip containing many per-frame `.gsplats.zarr.zip` files. Automatically extracted to cache on first use.

## Data Sources & Citations

- **Organoid data**: IDR study idr0062, Image 6001240 (Liberali lab, FMI)
  - Blin et al. (2019) + Williams et al. (2017) Nature Methods 14(8):775-781
- **Tribolium data**: Cell Tracking Challenge, Zenodo record 5270323
  - Yin et al. (2022) J. Cell Sci. 135(5), jcs259022
- **C. elegans data**: Zenodo record 6460303
  - Hirsch et al. (2022) DOI: 10.5281/zenodo.6460303

See individual demo scripts for full citation information.
