# Luxar Scripts

This directory contains standalone, run-by-hand scripts for generating data,
calibrating and refitting the gsplat demos, building the CUDA extension on HPC,
and validating project documentation. These are developer/maintenance tools,
not part of the importable `luxar` package.

Subdirectories:

```
scripts/
├── benchmarks/            # Performance benchmark scripts (seeding, WASM, etc.)
├── calibration_results/   # Output JSON from calibrate_gsplat_demos.py
└── gallery/               # Gallery dataset generation, capture manifest, and scoring tools
```

| Script | Purpose |
|--------|---------|
| `check_documentation.py` | Check top-level package READMEs plus Python docstring and TypeScript JSDoc coverage |
| `check_version_consistency.py` | Verify the zero-padded Python CalVer and npm-normalized viewer version describe the same release |
| `set_version.py` | Update the Python and viewer release versions together |
| `release.sh` | Run release preflight checks, then create and push the release tag |
| `gen_format_contract.py` | Generate the Python and TypeScript format-contract projections from `format-contract/contract.yaml` |
| `gen_data_manifest.py` | Regenerate the demo-data manifest (`demos/data_manifest.json`); `--check` is the CI drift gate |
| `generate_galaxy_simple.py` | Fetch Gaia DR3 stars → raw zarr table for demos |
| `gen_census_umap.py` | Build the large CELLxGENE Census scVI/UMAP cache on a CUDA/RAPIDS environment |
| `generate_builtin_colormaps.py` | Regenerate built-in colormap LUTs (Python + TS) |
| `build_cuda_slurm.py` | Submit a CUDA extension build job to Slurm |
| `test_hpc_setup.py` | Smoke-test the HPC/venv-fallback dev environment |
| `calibrate_gsplat_demos.py` | Run `luxar gsplat cal` on every gsplat demo's volume(s) |
| `update_demo_max_splats.py` | Apply calibrated K\* to each demo's `MAX_SPLATS` constant |
| `add_additive_lod_to_demos.py` | Add an additive LOD ladder to each gsplat demo baseline |
| `reencode_gsplat_demos.py` | Re-encode and rebuild LOD ladders for committed gsplat demo baselines without refitting |
| `benchmark_progressive_psnr.py` | Benchmark progressive gsplat fitting (PSNR/SSIM) |
| `refit_gsplat_demos.sh` | Force-refit every gsplat demo (sequential) |
| `run_demo_recompute.sh` | Sequential demo recompute from scratch |
| `test_batch_plan_fixes.py`, `test_cholesky_fix.py` | Ad-hoc regression check scripts |

## Documentation Quality Checker

### `check_documentation.py`

Checks documentation coverage for the top-level Python and TypeScript packages.
It is a read-only heuristic checker; it does **not** parse Markdown syntax, walk
all nested subpackages, or modify files.

**Purpose:**
- Require a README for each top-level package under `luxar/` and viewer `src/`
- Require Quick Start/Getting Started headings and code examples in Python package READMEs
- Flag low Python docstring and TypeScript JSDoc coverage
- Surface documentation debt before it is promoted into a CI quality gate

**Usage:**

```bash
# Check all documentation
hatch run python scripts/check_documentation.py

# With verbose output
hatch run python scripts/check_documentation.py --verbose

```

**What it checks:**
- Top-level Python packages have README.md files with minimum content markers
- Top-level TypeScript packages have README.md files
- Public Python definitions have nearby docstrings (heuristic)
- Exported TypeScript declarations have nearby JSDoc (heuristic)

The current repository has known failures, so `make check-docs` is an audit
report rather than a green required gate. See GitHub issue #776 before
tightening or enabling it in CI.

---

## Gaia DR3 Data Generator

### `generate_galaxy_simple.py`

Fetches real star data from ESA's Gaia DR3 archive and saves it as a **raw zarr table** (NOT Luxar format).

**Purpose:**
- Generates raw astronomical data for use in demos
- Output is consumed by `demo_gaia_milky_way_3m.py`

**Additional Dependencies:**
```bash
pip install astroquery astropy
```

These are **NOT** part of luxar's core dependencies because they're large astronomy-specific packages only needed for data generation.

**Usage:**

```bash
# Install dependencies (one-time)
hatch run pip install astroquery astropy

# Generate 3M stars (used for demo, ~90 minutes)
hatch run python scripts/generate_galaxy_simple.py --count 3000000 --output packages/luxar/src/luxar/demos/data/milky_way_gaia_3m.zarr

# Test with smaller datasets
hatch run python scripts/generate_galaxy_simple.py --count 10000    # 10k stars (~30 sec)
hatch run python scripts/generate_galaxy_simple.py --count 100000   # 100k stars (~2 min)
```

**Output Format:**

Raw zarr table (NOT Luxar format) with arrays:
- `x_kpc` - Galactocentric X coordinate (float32)
- `y_kpc` - Galactocentric Y coordinate (float32)
- `z_kpc` - Galactocentric Z coordinate (float32)
- `phot_g_mean_mag` - G-band magnitude (float32)
- `bp_rp` - BP-RP color index (float32)

Plus metadata: `num_stars`, `magnitude_range`, `description`, `data_source`

**Query Parameters:**
- `parallax > 0.1 mas` → distances up to ~10 kpc from Sun
- `parallax_over_error > 5` → high-quality measurements only
- `ORDER BY phot_g_mean_mag ASC` → sorted by brightness

**Coordinate Transform:**
- Uses Astropy's Galactocentric frame
- R₀ = 8.122 kpc (Sun-GC distance, GRAVITY 2018)
- Filters to stars within 30 kpc of Galactic Center

**To visualize the data:**
```bash
# The raw data CANNOT be viewed directly
# Use the demo which converts to Luxar format:
python packages/luxar/src/luxar/demos/demo_gaia_milky_way_3m.py
```

---

**Data Attribution:**

ESA/Gaia/DPAC - Gaia Data Release 3 (2022)

Citation:
Gaia Collaboration, Vallenari et al. (2023)
"Gaia Data Release 3: Summary of the content and survey properties"
Astronomy & Astrophysics, 674, A1
DOI: 10.1051/0004-6361/202243940

---

**See also:**
- `demo_gaia_milky_way_3m.py` - Converts raw data to Luxar format and visualizes
- `demo_gaia_milky_way_8m.py` - Larger 8M star dataset demo
- ESA Gaia Archive: https://gea.esac.esa.int/archive/

---

## Built-in Colormap Generator

### `generate_builtin_colormaps.py`

Regenerates Luxar's built-in colormap lookup tables (256×3 `uint8` LUTs) as
source files for **both** the Python package and the TypeScript viewer, keeping
the two in sync. Requires matplotlib for generation only (not at runtime).

**Usage:**
```bash
hatch run python scripts/generate_builtin_colormaps.py
```

**Outputs (overwritten in place):**
- `packages/luxar/src/luxar/colormaps/builtins.py`
- `packages/luxar-viewer/src/rendering/colormap-data.ts`

Both files carry an `Auto-generated by scripts/generate_builtin_colormaps.py`
header. Includes linear ramps (`green`, `magenta`, `cyan`, `red`, `blue`,
`yellow`, `gray`), a `fire` colormap, and matplotlib-derived maps.

---

## HPC / CUDA Tooling

### `build_cuda_slurm.py`

Submits a CUDA extension build job to Slurm. Invoked by `make build-cuda SLURM=1`
but also runnable directly. It detects the PyTorch CUDA version in the hatch env,
finds the best-matching `cuda/X.Y...` module, detects the active virtualenv,
generates a self-contained sbatch script with step-by-step diagnostics, and
submits it (or previews with `--dry-run`).

**Usage:**
```bash
# Via make (recommended)
make build-cuda SLURM=1
make build-cuda SLURM=1 SLURM_PARTITION=gpu CUDA_MODULE=cuda/12.8.0_570.86.10

# Direct
python scripts/build_cuda_slurm.py --partition gpu
python scripts/build_cuda_slurm.py --partition gpu --dry-run
```

### `test_hpc_setup.py`

Smoke-tests the HPC / no-sudo dev environment created by `make setup-dev`
(venv-fallback detection, tool locations, PATH). Run after bootstrapping a
cluster login node:

```bash
python scripts/test_hpc_setup.py
```

---

## GSplat Demo Calibration & Refitting

These scripts implement the canonical `cal → fit → lod` pipeline for the bundled
gsplat demos. The usual order is: `calibrate_gsplat_demos.py` →
`update_demo_max_splats.py` → `refit_gsplat_demos.sh` →
`add_additive_lod_to_demos.py`.

### `calibrate_gsplat_demos.py`

Runs `luxar gsplat cal` on every gsplat demo's preprocessed volume(s). For each
demo it imports the demo module by file path, calls the demo's own data loader to
reproduce the exact preprocessed volume, persists it as `.npy`, then subprocesses
the published `luxar gsplat cal` CLI (so any CLI regression surfaces here). For 4D
demos (celegans, zebrafish) it calibrates 3 distributed timepoints and takes the
median K\*. Results are written under `scripts/calibration_results/`.

```bash
hatch run python scripts/calibrate_gsplat_demos.py              # all demos
hatch run python scripts/calibrate_gsplat_demos.py --only dapi  # one demo
hatch run python scripts/calibrate_gsplat_demos.py --list       # show targets
hatch run python scripts/calibrate_gsplat_demos.py --skip-existing
```

### `update_demo_max_splats.py`

Reads `scripts/calibration_results/_all_summaries.json` and rewrites the
`MAX_SPLATS = N` (or `SEEDS_PER_TILE = N`) assignment in each demo file in place,
also bumping `MAX_SPLATS_PER_PASS` proportionally so the progressive fitter runs
in roughly 4–8 passes regardless of the new total.

```bash
hatch run python scripts/update_demo_max_splats.py
```

### `refit_gsplat_demos.sh` / `run_demo_recompute.sh`

Sequentially force-refit the gsplat demos (`hatch run python <demo>
--no-napari --no-serve --recompute`) after updating their splat counts.
`refit_gsplat_demos.sh` logs each demo under a private per-run temporary
directory, prints that directory at startup, and finishes with an OK/FAILED
summary; `run_demo_recompute.sh` is a simpler variant.

```bash
bash scripts/refit_gsplat_demos.sh
nohup bash scripts/refit_gsplat_demos.sh > /tmp/refit.log 2>&1 &  # background
```

Both shell scripts resolve the repository root from their own location, so they
can be launched from any working directory.

### `add_additive_lod_to_demos.py`

Adds an additive LOD ladder to every gsplat demo's LFS baseline using the
supp-doc additive-LOD algorithm (cumulative count breakpoints `1500, 8000,
40000`, sized so L0 fits in a single 64 KB zarr chunk). Uses `greedy` ordering up
to N ≈ 200K, falling back to `self_energy` above that. Defaults to writing
`<file>.lod_added.gsplats.zarr.zip` sidecar files; originals are untouched unless
`--in-place`.

```bash
hatch run python scripts/add_additive_lod_to_demos.py             # sidecar files
hatch run python scripts/add_additive_lod_to_demos.py --dry-run   # plan only
hatch run python scripts/add_additive_lod_to_demos.py --only kidney
hatch run python scripts/add_additive_lod_to_demos.py --in-place  # replace originals
```

### `benchmark_progressive_psnr.py`

Benchmarks progressive Gaussian splat fitting on a 3D DAPI chimera (4 Z×Y
quadrants mixing denoised/raw channels) and a 2D composite, reporting PSNR + SSIM
plus a single `METRIC` line for autoresearch extraction. Budget parameters are
fixed constants by design. Requires a GPU (torch).

```bash
hatch run python scripts/benchmark_progressive_psnr.py
```

See also `scripts/benchmarks/` for the seeding/WASM performance benchmarks.
