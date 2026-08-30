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
| `check_documentation.py` | Baseline-driven ratchet over package README paths/content plus Python docstring and TypeScript JSDoc coverage (JSON output; fails only on new findings) |
| `check_complexity.py` | Baseline-driven ratchet over ruff's `C901` cyclomatic-complexity rule (fails only on newly over-complex, or newly worse, functions) |
| `check_demo_ladders.py` | Audit built demo scenes for missing or degenerate additive streaming ladders |
| `check_demo_links.py` | Report whether canonical demo click-through destinations still discriminate known-good and known-bad identifiers without gating on third-party availability |
| `check_scene_credits.py` | Verify built demo stores carry the `short`, `doi`, and `license` their registry citation declares |
| `check_open_issue_pr.py` | Report whether an open PR already claims an issue (advisory, run by hand); inventory unique/shared paths before closing a duplicate |
| `ci_queue_scan.py` | Classify obsidian-labelled Actions jobs and perform bounded repository run scans for CI routing/watchdog policy |
| `ci_queue_redispatch.py` | Hand off, cancel, and fully rerun one first-attempt CI run stranded behind a saturated obsidian queue |
| `check_fixture_env.py` | Assert the viewer-fixture Hatch environment is CPU-only and free of default-env tooling |
| `check_version_consistency.py` | Verify the Python, viewer, and citation metadata describe the same release |
| `set_version.py` | Update the Python, viewer, and citation release versions together |
| `release.sh` | Run release preflight checks, then create and push the release tag |
| `gen_format_contract.py` | Generate the Python and TypeScript format-contract projections from `format-contract/contract.yaml` |
| `gen_data_manifest.py` | Regenerate the demo-data manifest (`demos/data_manifest.json`); `--check` is the CI drift gate |
| `gen_zenodo_records.py` | Generate Zenodo record descriptions from the manifest and committed `demo_archive_characteristics.json`; `--refresh` updates measurements, optionally preferring a namespaced `--archives-root`, exits 1 after writing if a pinned fit hashes but cannot be parsed, and warns without failing when fit bytes cannot be opened for hashing because those bytes cannot be checked against the pin; `--check` lists incomplete rows and fails on a pinned fit that is present but unreadable (never contacts Zenodo) |
| `zenodo_migration_audit.py` | Audit migration readiness offline by default; `--live` with `ZENODO_TOKEN` also compares manifest pins with Zenodo depositions, and `make check-zenodo-live` is the lean report-only entry point |
| `verify_cold_fetch.py` | Cold-fetch every hosted demo dataset with the cache and in-repo payload hidden; `make check-cold-fetch` is the opt-in pre-removal gate |
| `run_external_reference_audits.py` | Run the documentation, demo click-through, and live Zenodo-pin audits independently; normalize them to PASS/NOTICE/WARNING/ERROR and write a non-gating GitHub job summary |
| `generate_galaxy_simple.py` | Fetch Gaia DR3 stars → raw zarr table for demos |
| `gen_census_umap.py` | Build the large CELLxGENE Census scVI/UMAP cache on a CUDA/RAPIDS environment |
| `generate_builtin_colormaps.py` | Regenerate built-in colormap LUTs (Python + TS) |
| `build_cuda_slurm.py` | Submit a CUDA extension build job to Slurm |
| `check_hpc_setup.py` | Smoke-test the HPC/venv-fallback dev environment |
| `calibrate_gsplat_demos.py` | Run `luxar gsplat cal` on every gsplat demo's volume(s) |
| `update_demo_max_splats.py` | Apply calibrated K\* to each demo's `MAX_SPLATS` constant |
| `add_additive_lod_to_demos.py` | Add an additive LOD ladder to each gsplat demo baseline |
| `reencode_gsplat_demos.py` | Re-encode and rebuild LOD ladders for committed gsplat demo baselines without refitting |
| `benchmark_progressive_psnr.py` | Benchmark progressive gsplat fitting (PSNR/SSIM) |
| `refit_gsplat_demos.sh` | Force-refit every gsplat demo (sequential) |
| `run_demo_recompute.sh` | Sequential demo recompute from scratch |
| `run_examples.py` | Regenerate only the `datasets/examples/` producers whose source or statically imported Python modules changed, or whose recorded outputs are missing; dynamic imports, non-Python inputs, and `pyproject.toml` are not fingerprinted, so use `--force` after those change; `--check` is the E2E freshness gate |

## Demo Ladder Structural Gate

### `check_demo_ladders.py`

Audits every built `*.luxar.zarr` demo (or explicitly supplied scenes) and
fails when a large Points, Lines, or GSplats leaf has no additive ladder, a
single increment exceeds the relative `--max-share` limit, or an increment
exceeds the absolute `--max-level-elements` commit budget. For sliced nodes it
also histograms rung 0 by hidden coordinate across every partition part and
fails when the largest visible slice is below `--min-slice-first-rung` (default
2,000), or when a sliced survey is empty. The existing demo output directory is
inventoried read-only; the check does not create it.

```bash
hatch run check-demo-ladders
hatch run check-demo-ladders path/to/scene.luxar.zarr
hatch run check-demo-ladders --max-share 0.6 --max-level-elements 1000000
hatch run check-demo-ladders --min-slice-first-rung 2000
```

The command is included in `hatch run check`. A checkout without generated demo
scenes reports that none were found and succeeds; unit tests still exercise the
gate logic in CI.

A second, independent pass lives behind `--screen` (`--screen-only` to skip the
gate above): the LOD **opening-shot screen** from `luxar.io.lod_screening`. Per
`kind=lod` group, would re-deriving the ladder onto the `screen-area` selector
make the OPENING framing land on a coarser level? It is a REPORT ONLY — no
verdict it produces can change the exit code, so `--screen-only` exits 0 for any
store it can read (only a malformed flag still fails, as argparse).

```bash
hatch run check-demo-ladders --screen-only datasets/examples/*.luxar.zarr
hatch run check-demo-ladders --screen-only --screen-verdict win --screen-render-fov 63 datasets/demos/*.luxar.zarr
hatch run check-demo-ladders --screen --screen-aspect 4:3=1.3333
```

| flag | meaning |
|---|---|
| `--screen` | append the screen to the streaming-ladder gate |
| `--screen-only` | run ONLY the screen; the gate is skipped |
| `--screen-aspect` | aspects to measure at, as `label=value`, `W:H` or a bare number (default 1:1, 16:9, 21:9) |
| `--screen-viewport-long` | pixels on the viewport's LONG axis; affects the legacy diagonal metric only |
| `--screen-fit-fov` | vertical FOV the fitted camera DISTANCE uses (default 47, the viewer's own) |
| `--screen-render-fov` | fallback vertical FOV where the store does not author `camera.fov` (pass 63 for the cinematic/35mm preset) |
| `--screen-verdict` | show only these buckets (repeatable): `win`, `fragile`, `no-op`, `off-screen`, `already-current`, `skipped` |

---

## Documentation Quality Checker

### `check_documentation.py`

Checks documentation coverage for the top-level Python and TypeScript packages.
It is a read-only checker; it does **not** fully parse Markdown syntax or walk
all nested subpackages. It parses Python with the AST and validates backticked
path-like references in tracked package READMEs.

**Purpose:**
- Require a README for each top-level package under `luxar/` and viewer `src/`
- Require Quick Start/Getting Started headings and code examples in Python package READMEs
- Flag low Python docstring and TypeScript JSDoc coverage
- Fail the required PR documentation gate on any new finding

**Usage:**

```bash
# Check all documentation (ratchet mode: fails only on NEW findings)
hatch run docs:python scripts/check_documentation.py

# With verbose output
hatch run docs:python scripts/check_documentation.py --verbose

# Machine-readable JSON report (includes a `ratchet` block)
hatch run docs:python scripts/check_documentation.py --json

# (Re)write the debt baseline from the current state, then exit 0
hatch run docs:python scripts/check_documentation.py --update-baseline

# Point at a non-default baseline file
hatch run docs:python scripts/check_documentation.py --baseline path/to/baseline.json

# Legacy strict mode: ignore the baseline and fail on ANY finding
hatch run docs:python scripts/check_documentation.py --no-baseline
```

**What it checks:**
- Top-level Python packages have README.md files with minimum content markers
- Top-level TypeScript packages have README.md files
- Python modules, functions, classes and methods carry docstrings (AST-parsed, exact)
- Exported TypeScript declarations have nearby JSDoc (heuristic)
- A Python file that cannot be parsed is reported as a `Python syntax` finding (the run continues rather than crashing)

Existing documentation debt is captured in `scripts/docs_baseline.json`. A
flagless run tolerates every baselined finding and fails (exit 1) only on new
missing READMEs/docstrings/JSDoc or broken README path references. It is the
completeness stage of `make check-docs` and the required `docs-quality` CI job.
As debt is paid down, regenerate/tighten the baseline with `--update-baseline`
and commit the smaller file. See
`docs/guides/developer/DOCUMENTATION_QUALITY.md` for the full model.

---

## Complexity Ratchet

### `check_complexity.py`

Enforces `[tool.ruff.lint.mccabe] max-complexity` (10) as a baseline-driven
ratchet, the same shape as the documentation ratchet above.

**Purpose:**
- Run `ruff check --select C901` over the same paths as `hatch run lint`
- Tolerate the pre-existing over-limit functions recorded in
  `scripts/complexity_baseline.json` (228 at the time of writing)
- Fail (exit 1) when a function is newly over the limit, or when a baselined
  one gets *more* complex
- Report paid-down debt as advisory (exit 0) so the baseline can be tightened
- Report a *move* (a baselined function reappearing under a new path at no
  greater complexity, with or without a tidy-up) as advisory too, itemised
  old-key-to-new-key, so a module-move series is not a false red. Full runs
  only — see the restricted-scan note below
- Fail closed (exit 2) rather than green whenever the scan cannot be trusted: a
  ruff that did not run, a target ruff could not read (its `Failed to lint`
  warning otherwise leaves a partial scan behind a normal exit code), or a FULL
  run that found nothing while the baseline is populated (a mistyped target, a
  wrong `--project-root`, a partial checkout).
  A *restricted* run finding nothing is legitimate — a subtree may simply be
  clean — so that only warns

`C901` is deliberately not in `[tool.ruff.lint] select`: ruff has no baseline
mechanism, and its only native suppression (`per-file-ignores`) is
file-granular, so it would blind the guard to new offenders in the 151 files
that already hold a violation.

**Usage:**

```bash
# Check the tree against the baseline (the flagless, gating mode)
hatch run check-complexity
make check-complexity

# (Re)write the baseline from the current state, then exit 0
hatch run check-complexity --update-baseline

# Point at a non-default baseline
hatch run python scripts/check_complexity.py --baseline path/to/baseline.json

# Restrict the scan to some paths. Baselined keys OUTSIDE them then look
# vanished, so the run says so instead of inviting --update-baseline (writing a
# baseline from a restricted scan would drop the rest of the tree's debt; it
# warns, and refuses outright if the restricted scan found nothing at all).
# Move detection is switched OFF for a restricted run: an unscanned key must
# never be allowed to absorb a genuinely new function as a "move".
hatch run python scripts/check_complexity.py packages/luxar/src
```

Baseline keys are `<repo-relative-path>::<function-name>` mapping to the
descending-sorted complexities of the over-limit functions with that name in
that file — no line numbers, so an unrelated edit above a function never churns
the baseline. Because a move pairs on the function name alone, a genuinely new
function can in principle be absorbed by a same-named one vanishing in the same
run; what the ratchet always guarantees is the bound, not the identity — a pair
can never increase total debt. The checker runs as part of `hatch run lint` and
`hatch run check`, and the Python test suite
(`packages/luxar/src/luxar/tests/test_check_complexity.py`) asserts the real
tree is regression-free, so the ratchet gates every PR.

---

## Gaia DR3 Data Generator

### `generate_galaxy_simple.py`

Fetches real star data from ESA's Gaia DR3 archive and saves it as a **raw zarr table** (NOT Luxar format).

**Purpose:**
- Generates raw astronomical data for use in demos
- Output is consumed by `demo_gaia_milky_way_3m.py`

**Additional Dependencies:**
```bash
pip install 'luxar[demos]'
```

These are not part of Luxar's core dependencies; `astroquery` and `astropy` are
declared by the `demos` extra because the installed demo can build the catalog.

**Usage:**

```bash
# Preferred installed-demo path (cached, resumable, correct member name)
luxar demo deps --install
luxar demo run gaia_milky_way -- --build-catalog

# Source-checkout compatibility CLI
hatch run python scripts/generate_galaxy_simple.py --count 3000000 --output ~/.cache/luxar/milky_way_gaia_3m/milky_way_gaia_3m.zarr

# Test with smaller datasets
hatch run python scripts/generate_galaxy_simple.py --count 10000    # 10k stars (~30 sec)
hatch run python scripts/generate_galaxy_simple.py --count 100000   # 100k stars (~2 min)
```

The 3M output goes to the demo's cache, **not** into the repository: the Gaia
catalog is CC BY-NC and the derived point cloud inherits that, so do not commit
it (the former in-repo `demos/data/` copy was removed for exactly that reason).
The demo path fixes the cache stem, promotes the zip atomically, and can resume
from a completed raw zarr without repeating the TAP query.

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

### `check_hpc_setup.py`

Smoke-tests the HPC / no-sudo dev environment created by `make setup-dev`
(venv-fallback detection, tool locations, PATH). Run after bootstrapping a
cluster login node:

```bash
python scripts/check_hpc_setup.py
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

---

## Fitting Benchmarks Must Declare Their Floor Basis

Every gsplat fit called here through a module-level fitting *function* passes an
explicit `floor=`, and the in-repo harnesses that score PSNR/SSIM against the raw
volume pin `floor="none"` — the shipped default `floor="auto"` subtracts a
background pedestal the reference still carries, which penalises the fit for
correctly dropping non-signal and lets the reported number drift with the floor
estimator. The guard `packages/luxar/src/luxar/tests/test_benchmark_floor_pin.py`
enforces the declaration: it parses every `*.py` under `scripts/` and fails on a
fitter call that declares no floor — a `floor=` keyword, or a kwargs dict
carrying the key that is visible at the call site — so declare one when you add a
fitting script (`floor="auto"` is a valid answer — the gate wants a stated basis,
not a particular value).

Two shapes stay outside the gate's reach, so pin them by hand. *The class API*:
`GaussianSplatFitter().fit(V)` is a Python-function fit the guard cannot see, as
an AST call-name check would have to flag every `.fit(` to catch it. *Argv-driven
fits*: `calibrate_gsplat_demos.py` runs many fits by subprocessing `luxar gsplat
cal` with no `--floor`, and that one needs no pin because `cal` is
self-consistent — it subtracts the floor from the volume once up front and pins
its own per-K fits to `floor="none"`.

One caveat when changing a fit path: the two Pareto benchmarks compare against a
*local, uncommitted* `scripts/benchmarks/data/*baseline*.json`, so a baseline
recorded on a different floor basis will read as a regression — delete it once
and let the next run become the baseline.
