# CLAUDE.md

Guidance for Claude Code when working with this repository.

**Luxar** is a high-performance system for compiling and visualizing arbitrary-sized nD scientific scenes. It renders four first-class geometry types — **Points**, **Lines**, **Gaussian Splats**, and **Mesh** (triangle surfaces). Mesh is the newest and the only *shaded* one — the other three are purely emissive — via a light-free view-anchored headlight, and it is now feature-complete at the UI level: picking (at VERTEX granularity, keyed on `gl_VertexID` rather than an element-texture texel), the Layers-panel appearance controls, monitor/stats/debug counts. The docs pass is done and real WebGPU is verified pixel-identical to WebGL (see `docs/specs/MESH_NODE_SPEC.md` §11 and the CHANGELOG A/B notes). The contract still names the writable and drawable sets separately — `geometry_types` and `loader_types` — because a type becomes authorable before it becomes drawable; they simply agree on all four today.

## Quick Reference

### Python (use Hatch)
```bash
hatch run test              # Run tests
hatch run test-cov          # Tests with coverage
hatch run python script.py  # Run script
hatch run python -m ruff check .  # Lint
hatch run mypy packages/luxar/src/luxar/  # Type check
```

### TypeScript (use pnpm, from packages/luxar-viewer/)
```bash
pnpm dev          # Dev server (port 5173)
pnpm build        # Build
pnpm test --run   # Unit tests
pnpm test:e2e     # E2E tests (Playwright)
pnpm typecheck    # Type check
pnpm lint         # Lint
pnpm format       # Format
```

### Make Commands (from project root)
```bash
# Development Setup
make setup-dev    # Complete development environment setup (auto-installs dependencies)
make check-deps   # Check what dependencies are installed/missing
make install-rust # Install Rust + wasm-pack for viewer builds
make clean-setup  # Remove ALL dev tools to simulate fresh machine
# Installs the demos+gsplats+io extras, then reports the result via
# `luxar demo deps`. NOT part of setup-dev: the extras are heavy (torch,
# cellxgene-census, esm) and most work needs none of them.
make install-demo-deps  # Install the demo extras (demos + gsplats + io)

# Quality & Testing
make test-all     # All tests (Python incl. CUDA + WASM/Rust + TypeScript + Go launcher)
make test-cov-all # Coverage: Python (minus `-m slow`) + TypeScript
make test-python  # Python tests only
make test-e2e     # Full Playwright E2E suite (~17 min)
make test-e2e-smoke  # E2E smoke subset (the specs CI would run)
make test-perf-e2e   # Opt-in Playwright performance suite
# check-all is NOT read-only: `hatch run check` begins with `format`, so it
# REWRITES packages/luxar/src and scripts. When other agents/people are editing
# the same tree, use the read-only scoped targets instead (listed right below it).
make check-all    # All quality checks (Python, TypeScript, Rust, Go) — reformats
make lint-python        # read-only: ruff check
make type-check-python  # read-only: mypy
make security           # read-only: bandit
make check-typescript   # read-only: typecheck + lint + unit tests
make check-rust   # Rust type/lint checks (cargo check + clippy)
make check-knip   # REPORT only (non-gating): unused viewer files/exports/deps
make format-all   # Format all code (Python, TypeScript, Rust, Go, CUDA)

# Viewer
make viewer       # Start viewer dev server (port 5173)
make build-viewer # Build viewer for production (requires Rust)
make build-viewer-lib  # Build + verify the npm LIBRARY bundle (publish-npm.yml)
make build-wasm   # Build WASM module only
make test-wasm    # Run Rust unit tests
make benchmark-wasm  # Run WASM vs TypeScript performance benchmarks

# Data & Examples
make run-examples              # Generate example datasets
make demo                      # Generate the Lorenz demo dataset (use 'luxar demo run lorenz' to also serve)
make generate-gallery-datasets # Generate demo datasets used by the gallery harness
make generate-gallery          # Capture gallery stills + orbit videos to docs/images/gallery/

# Native launchers (luxar export --native)
# Run `make build-launchers` BEFORE `luxar export --native` — the bundlers
# look for the host-platform binary in cli/_launchers/ and raise a clear
# "run make build-launchers" error if missing. Binaries also ride along
# into wheel builds when present at build time.
make install-go       # Install Go toolchain (no sudo: brew on macOS, official tarball on Linux)
make build-launchers  # Build launcher binaries for the host platform (CGO blocks pure cross-compile)
make clean-launchers  # Clean built launcher binaries
# Runtime override: LUXAR_LAUNCHER_NO_WEBVIEW=1 ./luxar-launcher
#   Opens the system default browser instead of the embedded WebView —
#   useful for headless smoke tests. NOT a rescue for a missing
#   libwebkit2gtk: cgo links WebKit at build time, so the binary has a hard
#   DT_NEEDED on libwebkit2gtk-4.0.so and the loader aborts before main()
#   ever reads this variable. A 4.1-only distro (Ubuntu 24.04+) needs the
#   4.0 runtime installed, or a separate browser-only build. See #998.
# Runtime override: LUXAR_CACHE_BUDGET_MB=<N> ./luxar-launcher
#   Total in-memory cache pool (L0+L1+S-cache) the launcher passes to the
#   viewer via ?cacheBudgetMB=. WKWebView has no performance.memory, so the
#   viewer can't auto-size caches from the heap; the launcher supplies a
#   generous default (2048). Lower it on a memory-constrained machine.

# CUDA (Gaussian Splatting)
make setup-cuda       # Install CUDA deps + build extension (may need sudo)
make check-cuda-deps  # Check CUDA dependencies (nvcc, PyTorch CUDA, etc.)
make build-cuda       # Build CUDA splatting extension (local GPU required)
make build-cuda SLURM=1              # Build on a GPU node via Slurm (HPC)
make build-cuda SLURM=1 SLURM_PARTITION=gpu  # Specify partition
make build-cuda-slurm                # Alias for make build-cuda SLURM=1
make test-cuda        # Run CUDA tests
make benchmark-cuda   # Run performance benchmarks
make clean-cuda       # Clean CUDA build artifacts

# Utilities
make clean-all    # Clean all artifacts
make clean-viewer # Clean viewer artifacts only
make help         # Show all available commands
```

### Development Environment Setup

The build system is designed to work on **fresh Linux/macOS machines** with minimal pre-installed tools, including **HPC/Slurm login nodes** (no sudo, no GPU on login node).

**Prerequisites:**
- Python 3.10+ (usually pre-installed; `python3.12` or `python3.11` work on HPC)
- Git and curl
- **Git LFS** (optional, required for demo data files): `brew install git-lfs` (macOS) or `sudo apt-get install git-lfs` (Ubuntu)
- **Ubuntu/Debian only**: `sudo apt-get install -y pipx && pipx ensurepath`
- **HPC/no-sudo**: no extra prerequisites — the Makefile auto-detects and uses venv fallback

**What `make setup-dev` installs (no sudo needed):**
- **Node.js 22.22+** (installs the 22 LTS by default): via nvm (Linux) or Homebrew (macOS). The floor is jsdom 30 (dev/test only), whose undici 8 dependency crashes on Node older than 22.16; Vite 8 alone only needs 20.19. `engines.node` in `packages/luxar-viewer/package.json` deliberately stays at the library's runtime floor (`>=20.19.0`) — that manifest is published to npm, and a dev-only jsdom constraint there would break installs for consumers.
- **pnpm**: TypeScript package manager (via npm global or `--prefix ~/.local` fallback on HPC)
- **Hatch**: Python environment manager (via pipx, or venv fallback on HPC)
- **Pre-commit hooks**: ruff (lint + format), bandit, and mypy — see `.pre-commit-config.yaml`

**Key tools and their locations:**
| Tool | Installation | Location |
|------|--------------|----------|
| nvm | Auto-installed | `~/.nvm/` |
| Node.js | Via nvm | `~/.nvm/versions/node/` |
| Hatch | Via pipx (or venv on HPC) | `~/.local/bin/hatch` |
| pnpm | Via npm (or `--prefix ~/.local` on HPC) | `~/.local/bin/pnpm` |
| Rust/wasm-pack | `make install-rust` | `~/.cargo/` |
| Go (launchers) | `make install-go` | `~/.local/go/` (Linux) or Homebrew (macOS) |
| Launcher binaries | `make build-launchers` | `packages/luxar/src/luxar/cli/_launchers/` |
| CUDA toolkit | Manual install or module load | `/usr/local/cuda/` (typical) |
| CUDA extension | `make build-cuda` | `packages/luxar/.../cuda/*.so` |
| CUDA build info | Auto-generated by build | `packages/luxar/.../cuda/cuda_build_info.json` |

**Troubleshooting:**
```bash
# Check what's installed
make check-deps

# If pipx/hatch issues on Ubuntu
pipx reinstall hatch
pipx ensurepath
source ~/.bashrc

# If Node.js not found after nvm install
source ~/.nvm/nvm.sh
# or restart terminal

# Full reset and reinstall
make clean-setup
make setup-dev
```

**HPC / Slurm cluster setup (no sudo, no GPU on login node):**
```bash
# 1. Bootstrap the environment (detects HPC, uses venv fallback automatically)
make setup-dev

# 2. Ensure ~/.local/bin is in PATH (needed for hatch/pnpm on HPC)
export PATH="$HOME/.local/bin:$PATH"   # Add to ~/.bashrc to persist
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# 3. Verify the environment works
python scripts/check_hpc_setup.py

# 4. Build the CUDA extension on a GPU node via Slurm
make build-cuda SLURM=1                           # Auto-detect CUDA + GCC modules
make build-cuda SLURM=1 SLURM_PARTITION=gpu      # Specify partition
make build-cuda SLURM=1 CUDA_MODULE=cuda/12.8.0_570.86.10  # Pin CUDA module

# 5. Monitor the build job
tail -f build-cuda-logs/build_<JOB_ID>.out

# 6. Verify the extension after the job completes
make test-cuda
```

**What `make build-cuda SLURM=1` does:**
1. Detects the PyTorch CUDA version (e.g., 12.8) and finds a matching `cuda/` module
2. Detects and loads the highest available GCC >= 9 module (required by PyTorch 2.x)
3. Captures the current hatch virtual environment path
4. Generates a self-contained sbatch script (`build-cuda-logs/build_cuda_job.sh`)
5. Submits it to Slurm and prints monitoring commands
6. The build job loads CUDA + GCC modules, activates the venv, preserves `LD_LIBRARY_PATH`, and compiles with full output merged into the main log file for easy debugging

**Git LFS (Large File Storage):**

Some demo data files (`.npz`, `.zip`) are stored using Git LFS to keep the repository size manageable.

```bash
# Install Git LFS (one-time)
brew install git-lfs        # macOS
sudo apt-get install git-lfs  # Ubuntu/Debian

# Initialize Git LFS (one-time)
git lfs install

# Pull LFS files (when needed)
git lfs pull

# Verify LFS files (should show actual sizes, not ~100 bytes)
ls -lh packages/luxar/src/luxar/demos/data/*.npz
```

If demos fail with "file not found" errors, you likely need to pull LFS files.
See `packages/luxar/src/luxar/demos/data/README.md` for details.

See `docs/guides/developer/BUILD_SYSTEM_SPEC.md` for complete documentation.

### Luxar CLI
```bash
luxar demo                       # List the 80 bundled demos (table)
luxar demo run lorenz            # Run a demo by key/index (forwards -- args)
luxar demo cache list            # Inventory / clear demo caches (cache clear …)
# Demos keep heavyweight packages OUT of the core install, so a fresh checkout
# lists every demo but cannot run them all. Report-only `deps` exits 1 for an
# unmet row; `--install` installs the extras that provide it, while `--only`
# installs one exact constrained requirement. The table is
# `luxar.demos.INSTALL_SPECS`, which also drives the runtime `require_module`
# gate — so a package can't be advertised without being installable.
luxar demo deps                         # Report missing/outdated optional demo deps
luxar demo deps --install               # Install missing extras
luxar demo deps --extra io              # Restrict to one extra
luxar demo deps --only scipy --install  # Install one constrained requirement
luxar serve <data.luxar.zarr> --viewer # Serve with viewer
luxar info <data.luxar.zarr> --stats   # Dataset info
luxar profiles                   # Network simulation profiles
luxar export scene.luxar.zarr -o my_export/             # Export scene + viewer as standalone offline folder
luxar export scene.luxar.zarr -o my_export/ --open      # Export and serve in browser
luxar export scene.luxar.zarr -o my_export/ --overwrite # Overwrite existing export
luxar export scene.luxar.zarr -o out/ --native macos    # Native macOS .app bundle (requires `make build-launchers`)
luxar export scene.luxar.zarr -o out/ --native macos,linux-amd64,linux-arm64 --name MyScene
```

### GSplat CLI (fitting, converting, rendering, merging)
```bash
# Fit Gaussian splats to a volume (presets: draft/standard/hifi/ultra)
# Supported input formats: .zarr, .zarr.zip, .tiff, .npy, .npz
luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000
luxar gsplat fit volume.npy splats.gsplats.zarr --config params.yaml
luxar gsplat fit data.zarr.zip splats.gsplats.zarr --timepoint 0 --channel 0
luxar gsplat fit data.zarr.zip splats.gsplats.zarr --array-key h2afva/fused  # Nested zarr group
luxar gsplat fit volume.tiff splats.gsplats.zarr --preset hifi --iters 8000  # Override iterations
luxar gsplat fit --dump-config --preset hifi > config.yaml  # Generate config template

# Background floor / DC-offset suppression: `--floor` is ON by default (`auto`).
# A constant pedestal (camera offset, autofluorescence) is the worst case for a
# localized-Gaussian basis, so the floor is subtracted (clip at 0) before
# normalization; output amplitudes are background-relative. auto = histogram-mode
# estimate (capped at the median; a no-op on clean data with no pedestal).
luxar gsplat fit volume.tiff splats.gsplats.zarr                 # --floor auto (default)
luxar gsplat fit volume.tiff splats.gsplats.zarr --floor p10     # subtract 10th percentile
luxar gsplat fit volume.tiff splats.gsplats.zarr --floor 110     # subtract a fixed value
luxar gsplat fit volume.tiff splats.gsplats.zarr --floor none    # disable (hard-min, legacy)

# Tiling: `--tiling auto` (default) picks none/uniform/content automatically
# (none if the volume fits one tile — a total voxel budget, so small/medium
# stacks stay whole and avoid tile seams; content if a density is supplied;
# else uniform). A tiled fit (`--tiling uniform` or `--tiling content`) emits a
# `kind=partition` by default (one part per tile/box, for viewer frustum
# culling); pass `--flat` for a single flat leaf. Whole-volume fits
# (`--tiling none`/small auto) stay a single leaf.

# Uniform tiled fitting for large volumes (Hann cosine apodization, seamless stitching)
luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32
luxar gsplat fit large.zarr tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32  # Single tile (Slurm-ready)
# Parallel tiles on ONE GPU (no Slurm): spawn N `fit --tile` worker subprocesses,
# then merge. Default -j 1 = sequential. `-j auto` sizes N from free VRAM.
# Saturates the GPU when a single tile under-utilizes it (the local counterpart
# of `batch-fit submit --parallel`). --keep-tiles keeps the per-tile temp outputs.
luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32 -j 4
luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform -j auto

# Content-adaptive (density-driven) tiled fitting: variable-size boxes that pack
# more splats where the volume is busy. Drive it from a calibration (`--cal`) or
# the density knobs (--k-star-ref, --n-features-ref, --saturation-exponent,
# --saturation-cap, --feature-threshold, --feature-metric, --cell, --min-leaf,
# --max-leaf, --target-features, --overlap, -j/--jobs).
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json -j 8   # parallel content fit
# Emit the content plan WITHOUT fitting (writes the box plan JSON instead):
luxar gsplat fit vol.zarr plan.json --tiling content --cal cal.json --plan-only

# Per-part LOD AT FIT TIME (`--recipe`, tiled partition only): each tile/box-part
# gets its own LOD without a separate `lod` pass (which rejects a partition).
# stream -> `tiles` topology (prefix-sum ladder); levels -> `adaptive`.
# Requires a tiled fit (--tiling uniform/content) and a partition (not --flat);
# rejected with --flat / --tiling none / --tile / --plan-only / --plan-box, and
# rejects cross-recipe knobs (like `gsplat lod`). Knobs mirror `lod`:
# additive: --n-lods/--additive-method/--breakpoints; substitutive:
# --compression-factor/--levels/--substitutive-method/--coarsen-dims.
luxar gsplat fit large.zarr out.gsplats.zarr --tiling uniform -j 4 --recipe stream --n-lods 6
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json --recipe levels --compression-factor 4 --levels 3

# Whole-timelapse fitting at scale lives under `batch-fit` (scheduler-agnostic):
#   `batch-fit run`    = LOCAL multi-GPU (one box, no Slurm) — saturates all GPUs
#   `batch-fit submit` = Slurm cluster array job
# Both plan once (uniform tiles or a shared content box plan over T×C), then run a
# memory-safe STREAMING merge to one kind=partition. status/validate/merge/cancel
# are shared. (Renamed from the former `slurm-fit` group.)

# LOCAL multi-GPU whole-timelapse fit (no Slurm; the local sibling of submit).
# --gpus auto = every visible CUDA card above a free-VRAM floor (skips small
# cards; override LUXAR_GPU_VRAM_FLOOR_GB); 'all' forces every card; 'cpu' = CPU;
# '0,1,3' = explicit. Per-GPU concurrency from --jobs-per-gpu (auto sizes from
# each card's free VRAM). Resumable: re-running skips tiles already on disk.
luxar gsplat batch-fit run vol.zarr out/ --gpus all --tile-size 256            # uniform, all GPUs
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --gpus auto   # content plan
luxar gsplat batch-fit run vol.zarr out/ --gpus auto --merge-recipe stream --merge-n-lods 4  # per-part LOD at merge
luxar gsplat batch-fit run vol.zarr out/ --gpus 0,1 --jobs-per-gpu 2 --timepoints ::10   # subset, 2 workers/GPU
luxar gsplat batch-fit run vol.zarr out/ --gpus cpu                            # CPU fallback
luxar gsplat batch-fit run vol.zarr out/ --tiling content --cal cal.json --dry-run  # plan only

# HPC Slurm fitting (plans + submits Slurm array jobs). `batch-fit submit`
# submits by default; pass --dry-run to plan without submitting.
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu                    # Submit to Slurm
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --dry-run          # Dry-run plan (no submit)
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --preset draft     # Fast preview
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --tile-size 256    # Manual tile size (skips GPU profile)
# Content-aware cluster fan-out: build ONE content-balanced box plan and reuse it
# for every (t,c) — the cluster sibling of `fit --tiling content`. By default the
# plan is scanned from a temporal MAX-PROJECTION over up to --plan-samples
# (default 16) evenly-spaced timepoints, so boxes cover any region with signal at
# ANY timepoint (no holes where content moves over time); --plan-timepoint N pins
# a single timepoint instead. Needs a density (--cal or --k-star-ref/
# --n-features-ref); no GPU profile required. Each array task fits one box; merge
# streams a kind=partition. Density knobs match `fit`: --cal / --k-star-ref /
# --n-features-ref / --saturation-exponent / --feature-metric / --cell /
# --min-leaf / --max-leaf / --target-features.
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --tiling content --cal cal.json
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --tiling content \
    --k-star-ref 60000 --n-features-ref 5000 --plan-samples 24   # density knobs + 24-timepoint max-proj plan
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --parallel         # Concurrent tasks per GPU
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --tasks-per-job 5  # Manual packing
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --axes time,camera,channel,z,y,x                                    # Override axis labels
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --array-key h2afva/fused --axes time,z,y,x                          # Nested zarr group
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --timepoints '::10' --channels '0:2'                                # Subset selection
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --iters 8000 --seeds 100000                                         # Override fit params
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --merge-recipe levels --merge-compression-factor 4 --merge-levels 3   # per-part LOD at merge
luxar gsplat batch-fit status output/                                       # Check job status
luxar gsplat batch-fit merge output/                                        # Merge completed tiles → kind=partition
luxar gsplat batch-fit merge output/ --recipe stream --n-lods 6           # + per-part additive ladder (tiles)
luxar gsplat batch-fit merge output/ --recipe levels -K 4 -L 3        # + per-part coarse↔fine lod (adaptive)
# `--recipe` gives each spatial tile-part its own LOD ladder AS IT STREAMS — the
# memory-safe way to add LOD to tiled output (the `lod` command rejects a
# partition, so cal→fit→lod can't otherwise LOD a tiled merge). stream →
# tiles topology; levels → adaptive. The stacked-timepoint axis stays a
# hard coarsening barrier. `batch-fit submit --merge-recipe ...` bakes it into the
# merge Slurm job. Without `--recipe`, parts are bare leaves (frustum culling only).
luxar gsplat batch-fit validate output/                                     # Validate tile integrity
luxar gsplat batch-fit validate output/ --fix                               # Delete corrupt/stale tiles for re-fitting
luxar gsplat batch-fit cancel output/                                       # Cancel all Slurm jobs for a batch run

# GPU benchmark (required for auto tile-size; --tile-size bypasses this)
luxar gsplat benchmark --slurm --partition gpu        # Submit benchmark to Slurm
luxar gsplat benchmark --list                         # Show profiled GPUs

# Convert .gsplats.zarr to Luxar scene for web viewer
luxar gsplat convert splats.gsplats.zarr scene.luxar.zarr --center
# Appearance is baked at convert time: --colormap (builtin/matplotlib/colorcet,
# default gray), --tone-mapping (None/Linear/Reinhard/Cineon/ACES/AgX/Neutral;
# default = viewer default ACES), --gamma, --intensity, --absorption (volumetric kappa), --layer/--no-layer.
# ACES is the right choice for almost every scene (its filmic rolloff keeps
# bright structure from clipping flat) — prefer it, and set it EXPLICITLY so the
# compiler's LUT notice (which only fires when nothing was chosen) stays quiet:
luxar gsplat convert splats.gsplats.zarr scene.luxar.zarr --colormap plasma --tone-mapping ACES
# Reach for Neutral only in the narrower case where the colormap carries an exact
# scientific color encoding that must survive to the screen (ACES shifts hues):
luxar gsplat convert splats.gsplats.zarr scene.luxar.zarr --colormap plasma --tone-mapping Neutral

# Render gsplats back to volume for quality comparison
luxar gsplat render splats.gsplats.zarr rendered.npy --shape 128,128,128

# Compare reconstruction quality against original (PSNR, SSIM, MSE)
luxar gsplat compare fitted.gsplats.zarr original.tiff
luxar gsplat compare fitted.gsplats.zarr original.npy --output-json metrics.json --device cuda

# Calibrate splat count K via blind-spot cross-validation
# Sweeps K, identifies the held-out PSNR peak (K*), and reports the noise floor.
# Uses the manuscript's Noise2Self protocol: 5% donut-median masking; held-out
# evaluation against the original (pre-mask) values at masked voxels.
luxar gsplat cal volume.tiff cal.json                            # 10-point sweep, [1K, 512K]
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000  # Faster: 5-point sweep
luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'  # Explicit
luxar gsplat cal volume.tiff cal.json --pdf cal_report.pdf       # Multi-page PDF report
luxar gsplat cal volume.tiff cal.json --pdf rep.pdf --keep-fits fits/  # Slice montages too
luxar gsplat cal volume.zarr cal.json --progression power --power 2  # Polynomial K spacing
# Measure the saturation exponent alpha (K ~ features^alpha) instead of assuming
# the default 0.44: calibrates K* at several region scales and regresses
# log K* vs log n_features. The fitted alpha is written into the cal.json
# splat_density (consumed by `fit --tiling content`). WARNING: one K-sweep per
# scale, so runtime is multiplied by the number of scales.
luxar gsplat cal volume.zarr cal.json --fit-exponent --exponent-scales 128,192,256
# `cal` applies the SAME `--floor` (default auto) as `fit`, subtracting the floor
# ONCE up front so K* is measured on floor-suppressed data (matching how you fit).
# Pass `--floor none` to reproduce the legacy hard-min manuscript numbers.
luxar gsplat cal volume.tiff cal.json --floor none               # legacy (no floor)
# Output: K* + curve type {peak | plateau | signal_limited} + noise-floor σ̂ + PSNR ceiling.
# Then re-run fit at the recommended K: luxar gsplat fit volume.zarr out.zarr --seeds <K*>

# Canonical end-to-end pipeline: cal → fit (at K*) → lod (--recipe flat|stream|tiles|overview|adaptive|levels)
# `lod` operates on a pre-fitted .gsplats.zarr (output of `fit`); use `cal` upstream
# to pick K* in a principled way. .gsplats.zarr is format v3.3 (a node tree —
# a detached scene gsplat-node subtree the viewer loads directly) — see
# docs/specs/GSPLATS_ZARR_FORMAT.md.

# Build a representation topology from a fitted gsplat dataset — one command,
# one `--recipe` flag (REQUIRED). Intent-first names, scale-ordered; EVERY
# recipe carries streaming (additive prefix) ladders by default (--no-additive
# for bare leaves):
#   flat      one bare leaf                                        — tiny N / debug
#   stream    one leaf + progressive ladder (fast first paint)     — small/medium N
#   levels    coarse→fine replacement levels (zoom across scales)  — medium/large N
#   tiles     spatial BSP tiles, culled + streamed per tile        — large N
#   overview  instant coarse overview level + fine tiles on zoom   — huge N
#   adaptive  tiles where EVERY tile picks its own detail level    — largest N
# Renamed 2026-07 (old → new): additive→stream, substitutive/pyramid→levels,
# partitioned→tiles, multiscale→overview, mosaic→adaptive. Old names error with
# a pointer; stored batch manifests translate silently.
# Output is a standalone .gsplats.zarr (loadable with `luxar gsplat info`);
# graft it into a scene from Python via `add_gsplats_from_file` or `gsplat convert`.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe flat
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --n-lods 6
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream \
    --breakpoints energy:0.5,0.9,0.99,1.0                                    # cumulative energy fractions
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream \
    -m mass -b counts:500,2000,10000                                         # mass order, explicit counts
# additive default method `auto`: greedy (provably (1-1/e)-optimal at every
# prefix) at N <= 5000, else `self_energy` (cheap O(N log N)); override with -m.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --method self_energy
# STREAMING breakpoints: `-b stream:C` = geometric ladder (first chunk C splats,
# then doubling), sized per part/level. Or derive C from a download budget with
# `--target-ms` (+ `--bandwidth-mbps`, default 25; `--bytes-per-splat` override;
# bytes/splat measured from the input store, logged). First chunk ≈ target-ms of
# download → fast first paint; the viewer streams additive sub-LODs progressively.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --target-ms 200
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream -b stream:14000

# Give every leaf of an EXISTING tree an additive ladder, structure-preservingly
# (substitutive kind=lod levels, partition parts, adaptive groups all keep their
# shape) — WITHOUT recomputing the expensive substitutive/partition structure.
# The per-leaf counterpart of `lod --recipe stream` (which needs a flat input)
# and the inverse companion of `gsplat flatten`. Same streaming knobs; explicit
# `counts:` are clamped per leaf; an existing ladder is rebuilt from its union.
luxar gsplat additive sub.gsplats.zarr pyr.gsplats.zarr --target-ms 200        # ~200ms first paint/level
luxar gsplat additive in.gsplats.zarr out.gsplats.zarr -b stream:14000
luxar gsplat additive in.gsplats.zarr out.gsplats.zarr --n-lods 4              # classic equal-count

# Collapse ANY gsplat tree (leaf, LOD/matrix tree, partition, nested) into one
# flat matrix-shaped leaf. Use for compatibility with tools that expect a flat
# .gsplats.zarr, or before rebuilding a new global LOD from a tiled/partitioned
# result.
luxar gsplat flatten partitioned.gsplats.zarr flat.gsplats.zarr

# tiles / overview (the large-data topologies): each tile carries its own
# stream ladder; `--max-elements` (or `--parts`) caps per-tile splats (median
# BSP by default; --partition-rule midpoint|sah). `overview` adds a single
# coarse merged level (`--compression-factor`/-K) above the tiles branch.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe tiles --parts 8 --partition-rule sah
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe overview --compression-factor 8 --max-elements 250000

# adaptive: spatial tiles where EACH tile is its own levels group (per-tile
# coarse↔fine swap — locally adaptive; the per-tile-levels sibling of tiles).
# Partition knobs + the level-merge ones (--compression-factor/-K, --levels/-L,
# --substitutive-method); per-tile levels are stream-laddered by default.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe adaptive --max-elements 250000
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe adaptive --parts 8 -K 4 -L 2

# Stream ladders inside a lod group are SIBLING-AWARE: every level with a
# coarser sibling starts its `stream:C` ladder at max(C, ceil(n/(2·K))) so an
# upgrade's committed prefix passes the sibling within 1-2 chunks (the
# coarsest keeps the small user base = fast first paint). The viewer releases
# upgrade swaps at committed energy e(k) >= 0.6 (quality stamps) instead of
# waiting for the count crossover.
# STREAM LADDERS ARE ON BY DEFAULT everywhere: levels, adaptive per-tile
# levels, and the overview coarse cap all carry a progressive ladder (fast
# first paint) unless --no-additive is passed. Ladder knobs
# (--n-lods/-m/-b/--target-ms) therefore apply to those recipes too.
# levels: each coarser level has ceil(N/K^L) merged representative splats that
# REPLACE the previous level. Recommended workhorse `kmeans_lloyd` (O(N log N)
# Morton warm-start + cost-increment Lloyd); `greedy`/`greedy_lloyd` (lazy-heap
# Runnalls) is quality-leading at small N.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels            # K=4, L=3, method=auto
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels -K 4 -L 3 \
    --substitutive-method kmeans-lloyd --lloyd-iters 5 --device cpu
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels -K 4 -L 3 --n-lods 4
# Coverage inflation (any substitutive reduction): merged representatives get
# their inter-center spread widened x`--coverage-inflation` (default 3.0,
# mass-preserving) so neighbouring coarse splats sum flat — suppresses the
# axis-aligned grid ripple pure moment matching shows at coarse levels.
# Pass 1.0 for the historical pure moment match.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels --coverage-inflation 1.0
# Mass conservation (default ON): each level's total mass over the coarsened dims
# is pinned to its fine input's, per barrier group — no brightness pop at LOD
# switches in additive rendering. --no-conserve-mass restores raw a* amplitudes.
# L2 refinement (any substitutive reduction, opt-in): `--refine l2` Adam-optimizes
# each merged level against its fine input under the closed-form mixture L2
# (never worse than the merge; total mass pinned so brightness never pops across
# levels; barrier dims stay frozen under --coarsen-dims). Slower, higher fidelity,
# peak-preserving on sparse structures. `--refine-iters` (default 120) is the
# one time/quality knob.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels --refine l2
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe overview --refine l2 --refine-iters 200
# Volume re-fit (the highest-fidelity rung, opt-in): `--refine volume --target
# <vol>` warm-start re-fits each merged level against the SOURCE VOLUME itself
# (full fit seeded by the merge; +5-12 dB over the merge on real microscopy;
# each level keeps whichever of merge/re-fit renders closer — never worse).
# Needs the volume in hand: lod --target only (fit-time/batch are follow-ups);
# levels/overview recipes, no barrier dims. `--refine-iters` default 300 here.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels --target vol.tiff --refine volume
# Barrier-aware coarsening (levels/overview/adaptive): --coarsen-dims
# lists the center-column indices coarsening may merge over; the rest become hard
# barriers (a categorical/time/channel axis), so coarse splats never blend across
# them. Default = all dims. (The Python scene API defaults to Auto = coarsen
# displayed dims, group by non-displayed; standalone gsplats have no display info
# so the CLI takes explicit indices and warns on >3D input without the flag.)
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels --coarsen-dims 1,2,3
# LOD switch thresholds (ANY recipe with a kind=lod group — overview,
# levels, adaptive) are auto-derived as viewport-relative
# `coverage_fraction` = sqrt(N_i/N_finest): the finest level shows when the object
# fills the screen and coarser levels step in as it shrinks (the viewer anchors to
# the live viewport, so it self-calibrates on any monitor — no threshold knob).

# Import classical (photogrammetric) Gaussian-splat files → .gsplats.zarr.
# Dialects (auto-sniffed): INRIA point_cloud.ply, antimatter15 .splat,
# Niantic/Scaniverse .spz (legacy gzip v1-3), SuperSplat compressed .ply.
# SH color is reduced to the DC band (baked per-splat RGB); opacity → amplitudes.
# Orientation: Y-down dialects get a 180°-about-X fix by default (SPZ is already
# Y-up and is left alone); --no-reorient / --flip override. Result is a normal
# single-leaf .gsplats.zarr — pipe through `gsplat lod` / `gsplat convert`.
# Python: `from luxar.gsplats.interop import import_gsplats` (→ GSplatData), or
# one-line scene embed: scene.add_gsplats_from_file("garden", "garden.splat").
luxar gsplat import garden.splat garden.gsplats.zarr
luxar gsplat import point_cloud.ply scene.gsplats.zarr --no-reorient
luxar gsplat import capture.spz capture.gsplats.zarr -e precision

# Export .gsplats.zarr → classical INRIA PLY (opens in SuperSplat/PlayCanvas/
# gsplat.js). Cholesky → eigh → log-scales + quaternion; --opacity
# normalized|amplitude|constant maps amplitudes → opacity (normalized default;
# amplitude = lossless for imported data); --color auto|colors|colormap|white
# (+ --colormap NAME bakes scalar amplitudes); --sh-degree 0 default (DC only).
# Import-time orientation is auto-inverted (--keep-orientation to skip); >3D
# needs --timepoint (slices the LAST stacked dim) or --slice-dim/--slice-index;
# partitions must be `gsplat flatten`ed first. Distinct from top-level
# `luxar export` (scene → offline viewer folder).
luxar gsplat export fit.gsplats.zarr fit.ply --colormap viridis
luxar gsplat export imported.gsplats.zarr back.ply --opacity amplitude
luxar gsplat export timelapse.gsplats.zarr t42.ply --timepoint 42

# Migrate legacy .gsplats.zarr layouts (v1.0 / v1.1 / pre-v2.0 substitutive dir / v2.0 matrix /
# v3.0-v3.1 with pre-v3.2 pixel_size lod selector attrs) → v3.3
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr               # single file
luxar gsplat migrate-format old_pyr/ v3.gsplats.zarr                          # substitutive directory

# Re-quantize a fitted (current-format) .gsplats.zarr's Cholesky encoding (writes a copy).
# Structure-preserving round-trip (leaf/lod/partition/nested + fitting/pipeline
# groups kept); only the on-disk Cholesky encoding changes; decode is always
# float32 so viewer/GPU/WASM are unaffected. Unlike migrate-format (legacy→current,
# float32 vs AUTO-uint16 only) this exposes the full ladder incl. memory=uint8.
luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory      # uint8 (smallest, ~93 dB)
luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision  # float32 (exact/archival)

# Partition into a single kind=partition file via spatial BSP (--indices removed)
luxar gsplat partition splats.gsplats.zarr part.gsplats.zarr --parts 4               # target part count
luxar gsplat partition splats.gsplats.zarr part.gsplats.zarr --max-elements 100000   # per-part cap
luxar gsplat partition splats.gsplats.zarr part.gsplats.zarr --parts 4 --rule sah    # median|midpoint|sah

# Merge multiple datasets
luxar gsplat merge a.gsplats.zarr b.gsplats.zarr -o merged.gsplats.zarr
luxar gsplat merge t0.zarr t1.zarr -o 4d.zarr --as-dimension --values 0,1
luxar gsplat merge ch0.zarr ch1.zarr -o multi.zarr --channel-colors "#ff0080,#00ff00"

# Slice by coordinate ranges (numpy-style)
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr "0:50, :, 10:90"
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr ":50, 20:80, :"

# Apply spatial and intensity transforms to a gsplat dataset
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center
# --rotate-x/y/z acts on the 3 center dims given by --spatial-dims (default 0,1,2 —
# first-3-spatial / stack-last convention; listed order assigns the X/Y/Z roles,
# unlike filter's order-insensitive --spatial-dims); other dims stay unrotated.
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --rotate-z 90
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --rotate-z 90 --spatial-dims 1,2,3
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --normalize-intensity 1.0
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --translate 0,100,0 --scale-intensity 0.5

# Denoise a volume using Non-Local Means (auto-calibrates h)
luxar gsplat denoise volume.zarr denoised.zarr
luxar gsplat denoise volume.zarr denoised.npy --h 0.03
luxar gsplat denoise data.zarr.zip out.zarr --channel 0 --timepoint 5 --denoise-2d

# Open gsplat dataset in napari for visual inspection
luxar gsplat napari splats.gsplats.zarr

# Retrofit Q·e quality stamps onto an EXISTING .gsplats.zarr, in place (no
# refit / re-ladder): energy_fraction_cum e(k) per additive sub-LOD +
# reference_energy w per leaf (cheap O(N), enables the viewer's early
# energy-threshold LOD upgrades on legacy datasets); --with-quality also
# measures per-level mixture-L² Q vs each lod group's finest content.
# Re-stamps the root content_hash so viewer caches invalidate automatically.
# Directory stores only (unpack .zip first). New builds stamp by default
# (`RecipeParams.quality_stamps` / `--no-quality-stamps` on `gsplat lod`).
luxar gsplat annotate-quality splats.gsplats.zarr                # e(k) + w only (fast)
luxar gsplat annotate-quality splats.gsplats.zarr --with-quality # + measured Q per level
luxar gsplat annotate-quality splats.gsplats.zarr --dry-run      # print stamps, write nothing

# Inspect, cull, and filter
luxar gsplat info splats.gsplats.zarr          # Dataset statistics
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr                            # Auto (cumulative, keep 95%)
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m cumulative -r 0.90      # Keep 90% amplitude
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m redundancy --shape 41,512,512  # GPU, no target
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr --target vol.npy           # Error-budget (most principled)
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr --target vol.npy -p 95     # More aggressive error-budget
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1 --eccentricity-max 5
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --bbox "0,50,0,50,0,50" --volume-max 100
# Any min/max threshold accepts a number OR a percentile 'pNN' / 'NN%'
# (robust on heavy-tailed attributes; how GSIP background-cut sweeps are driven).
# --scale-min/max: characteristic size = geometric-mean SPATIAL sigma. Timelapse-
# safe (auto-ignores a zero-variance time axis) — the recommended "remove large
# diffuse background" knob. --eccentricity is likewise spatial by default.
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --scale-max p90         # drop largest 10% (diffuse background)
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --scale-max p90 --dry-run  # preview impact (splats/mass/amplitude removed), write nothing
# --isolation-max / --min-neighbors+--neighbor-radius: remove spatially-isolated
# noise splats (nearest-neighbour distance / local density; grouped by the
# non-spatial axis so timepoints never count as neighbours).
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --isolation-max p99      # strip the 1% most-isolated (noise)
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --min-neighbors 3 --neighbor-radius 5
# --soft-highpass/--soft-lowpass (+ --soft-width octaves): SOFT reweighting —
# attenuate amplitude by a smooth function of scale instead of hard-removing (no
# popping; splat count unchanged). High-pass suppresses large/diffuse background.
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --soft-highpass p90 --soft-width 1.0
# --spatial-dims 0,1,2 overrides the auto axis detection for scale/eccentricity/isolation.
luxar gsplat view splats.gsplats.zarr          # Quick web viewer
```

### GPU Acceleration (Seeding & Fitting)
```python
from luxar.gsplats.seeds import generate_seeds
from luxar.gsplats import fit_gaussian_splats

# GPU-accelerated seeding (much faster for large volumes; speedup depends on GPU)
seeds = generate_seeds(volume, device='auto')   # Auto-detect GPU
seeds = generate_seeds(volume, device='cuda')   # Explicit NVIDIA GPU
seeds = generate_seeds(volume, device='mps')    # Apple Metal GPU

# GPU-accelerated fitting with automatic seed generation
result = fit_gaussian_splats(
    volume,
    device='cuda',  # Propagates to seeding automatically
    seed_method='edges',
)

# Benchmark GPU performance
hatch run python scripts/benchmarks/benchmark_seeding_gpu.py
```

**GPU Support**:
- Sobel gradients: All dimensions (1D-nD)
- Interpolation: 2D/3D only (auto-fallback for others)
- Deduplication: All dimensions
- Expected speedup: substantial on large volumes (>100³), often orders of magnitude depending on GPU

### Make Command Nomenclature

The Makefile follows consistent naming conventions with **action-first** pattern:

| Pattern | Purpose | Examples |
|---------|---------|----------|
| `install-<tool>` | Install external tool on system | `install-node`, `install-rust`, `install-hatch` |
| `install-<component>-deps` | Install dependencies from manifest | `install-viewer-deps` (node_modules) |
| `install-dev` | Install Luxar package in editable mode | `install-dev` (pip install -e .) |
| `setup-<component>` | Orchestrated multi-step setup | `setup-dev`, `setup-cuda` |
| `enable-<feature>` | Activate/enable a feature | `enable-pre-commit` (activate hooks) |
| `build-<component>` | Compile/build artifacts | `build-viewer`, `build-wasm`, `build-cuda` |
| `test-<scope>` | Run tests | `test-all`, `test-python`, `test-e2e` |
| `check-<aspect>` | Verify/check something | `check-deps`, `check-all`, `check-cuda-deps` |
| `clean-<scope>` | Clean build artifacts | `clean-all`, `clean-viewer`, `clean-cuda` |
| `format-<language>` | Format code | `format-python`, `format-typescript` |
| `run-<script>` | Run scripts/examples | `run-examples`, `run-demos` |
| `serve-<target>` | Start a server | `serve-docs`, `serve-dataset`, `serve-examples` (serves the `datasets/` directory) |

**Key distinctions:**
- `install-<tool>` vs `install-<component>-deps`: Tools are executables (node, rust); deps are project dependencies (node_modules)
- `install-*` vs `setup-*`: Install is for single components; setup orchestrates multiple steps
- `install-dev` vs `install-<tool>`: install-dev is for Luxar package itself; install-<tool> is for external tools
- `enable-*` vs `install-*`: Enable activates already-installed features; install adds new software

---

## Project Structure

```
/packages/luxar/           # Python package
  /src/luxar/              # Source (core/, io/, utils/, validation/, typing_utils/,
                           #         encoding/, colormaps/, gsplats/, cli/, demos/)
  /src/luxar/**/tests/     # Python tests (colocated with each subpackage)
  /examples/               # Example scripts (*_example.py naming)

/packages/luxar-viewer/    # TypeScript/WebGL viewer
  /src/                    # Source with per-package READMEs

/docs/                     # Documentation
  /guides/                 # Organized guides by purpose
    /user/                 # User guides (format, HDR, testing)
    /developer/            # Developer guides (style, console, network)
    /specs/                # Technical specs (cache prefetching, nD transforms, subpixel jitter/TAA)
  /specs/                  # Format specs (GSPLATS_ZARR_FORMAT.md, GSPLATS_DIMENSION_MAPPING.md)
  /api/                    # Sphinx API reference files (.rst)
  /concepts/               # Architecture and concepts (.rst)
  /tutorials/              # Step-by-step tutorials (.rst)
  index.rst, conf.py       # Sphinx configuration
```

### Documentation Requirements

**Every Python subpackage MUST have**:
- `README.md` - Purpose, key classes, usage examples

**Every TypeScript package has**:
- `README.md` in `/src/{package}/` - Keep in sync with code changes

---

## Code Standards

### Python
- Use type hints for all parameters and return values
- Format with ruff (88 char line length)
- Use PyTest (not unittest), mock only as last resort
- **Use Arbol for console output**: Replace `print()` with `aprint()`, use `asection()` for hierarchical output

```python
from arbol import aprint, asection

with asection("Processing"):
    aprint("Step 1...")
    aprint("Step 2...")
```

### TypeScript
- Format with prettier
- Use JSDoc comments
- Unified config in `src/config/` (camelCase, not UPPER_SNAKE_CASE)
- Console logging: `import { log } from '../utils/log'` with format `[emoji] [Module] message`
- Prefix unused variables with underscore

---

## Testing

### Strategy
- **Minimum coverage**: 80%
- **NEVER skip tests** - fix them or create proper mocks
- **Run before committing**: `make test-all && make check-all`

### Python Tests
```bash
hatch run test                    # All tests
hatch run pytest path/to/test.py  # Single file
hatch run test-cov                # With coverage
```

### TypeScript Unit Tests
```bash
cd packages/luxar-viewer
pnpm test --run                   # All unit tests
pnpm test path/to/test.ts         # Single file
```

### E2E Tests (Playwright)
```bash
cd packages/luxar-viewer
pnpm test:e2e                     # All E2E tests (~17 min)
pnpm test:e2e:ui                  # Interactive mode
pnpm agent:debug                  # AI debugging (see console logs)
pnpm agent:debug:visible          # AI debugging with visible browser
```

**Running E2E tests in chunks (RECOMMENDED):**
Instead of running all E2E tests at once (which can timeout or be overwhelming), run them by topic:
```bash
# Basic functionality
npx playwright test basic-rendering.spec.ts viewer-initialization.spec.ts

# Scene & transforms
npx playwright test transform-hierarchy.spec.ts

# nD navigation & dimensions
npx playwright test nd-navigation.spec.ts dimension-initialization.spec.ts dimension-animation.spec.ts

# Worker & WASM
npx playwright test worker-wasm-integration.spec.ts

# Test fixtures (run generate-fixtures first!)
pnpm test:generate-fixtures
npx playwright test test-fixtures-rendering.spec.ts

# Keyboard & input
npx playwright test keyboard-input-system.spec.ts controls-interaction.spec.ts

# Visual regression
npx playwright test visual-regression.spec.ts theme-visual-regression.spec.ts

# Geometry & rendering (blending-modes needs generate-fixtures!)
pnpm test:generate-fixtures
npx playwright test geometry-types.spec.ts blending-modes.spec.ts colormap-system.spec.ts \
  post-processing-pipeline.spec.ts rendering-controls.spec.ts ortho-mode.spec.ts

# Data & I/O
npx playwright test data-integrity.spec.ts dataset-switching.spec.ts real-dataset-loading.spec.ts \
  url-parameters.spec.ts python-typescript-integration.spec.ts luxar-serve-integration.spec.ts

# Spatial, cache & nD transforms
npx playwright test spatial-index-accuracy.spec.ts cache-system.spec.ts nd-transforms.spec.ts \
  position-bounds-clipping.spec.ts

# UI panels & mouse
npx playwright test custom-gui-library.spec.ts layers-panel.spec.ts recording-panel.spec.ts \
  mouse-interactions.spec.ts

# Errors & robustness
npx playwright test error-recovery.spec.ts webgl-errors.spec.ts

# Performance & monitoring
# (performance-tracking now lives in the opt-in perf suite — see `pnpm test:perf:e2e`)
npx playwright test data-monitor-metrics.spec.ts

# Demos & first-time UX
npx playwright test all-examples-smoke-test.spec.ts demo-validation.spec.ts first-time-ux.spec.ts
```

**Key E2E rules**:
- Use `?src=<dataset>&debug` URL format (NOT `?data=`)
- Use 3D datasets for general tests (4D/nD slicing may show 0 points)
- Wait for `window.__luxarDebug` before assertions
- Run `pnpm test:generate-fixtures` before test-fixtures tests
- See `docs/guides/user/E2E_TESTING_GUIDE.md` and `docs/guides/developer/PLAYWRIGHT_GUIDE.md` for details

### Cross-Language E2E Testing
Python encoder and TypeScript decoder must stay in sync:
1. **Fixture-based**: Python generates zarr, TypeScript unit tests verify (fast, no browser)
2. **Playwright**: Full pipeline through browser (catches WebGL/rendering bugs)

When to run E2E:
- After changing encoding format
- After changing decoder
- Before PR/merge (always run full suite)

### Test Fixture Auto-Generation
Unit tests (`pnpm test`) auto-generate missing zarr fixtures via `globalSetup` in `vitest.config.ts`. The fixture list is parsed directly from `tests/fixtures/generate_test_data.py` (the single source of truth) — adding a new fixture to the Python script is sufficient; no separate manifest needs updating.

---

## AI-Assisted Debugging

When debugging viewer issues, use the Playwright agent driver:

```bash
cd packages/luxar-viewer
pnpm agent:debug
```

**Output includes**:
- `[BROWSER-CONSOLE-*]` - All browser console logs
- JSON state dump - Three.js scene, point counts, camera
- `test-results/debug/debug-view.png` - Screenshot

**Debug workflow**:
1. Run `pnpm agent:debug` to see current state
2. Add `console.log()` if needed
3. Run again to verify fix
4. Remove debug logging when done

**Available at `window.__luxarDebug`** (when `?debug` in URL):
- `scene`, `camera`, `renderer`, `controls`
- `getState()`, `renderOnce()`, `app`, `consoleInterceptor`

---

## Critical Gotchas

### Matrix Storage: NumPy vs THREE.js
NumPy uses row-major, THREE.js uses column-major. **Always transpose when serializing**:
```python
# Writing to zarr for THREE.js
matrix.T.ravel().tolist()

# Reading back in Python
np.array(flat_list).reshape(4, 4).T
```
Translation is at `[3,7,11]` in NumPy but `[12,13,14]` in THREE.js.

The viewer loader **refuses to load** scenes whose 4x4 transforms look
row-major (translation at indices [3,7,11] with [12,13,14] zero) — see
`packages/luxar-viewer/src/rendering/node-factory/validation.ts::validateTransformFormat`.
A producer that forgets to transpose now fails the load instead of
silently rendering in the wrong place.

### Constructor Initialization Order
When subclass and parent both set the same attribute, **parent must initialize first**:
```python
def __init__(self):
    super().__init__()  # First!
    self._metadata = {...}  # Then subclass sets it
```

### Transform Composition Order
`compose(T1, T2, T3)` applies T1 first, T3 last (right-multiply):
```python
result = result @ transform  # Correct
# NOT: result = transform @ result
```

### nD Datasets in Tests
- 4D/nD datasets may show 0 points depending on slice position
- Use 3D datasets for general-purpose loading tests
- For nD tests, navigate to slices known to have points

### Data Source URLs Normalize Trailing Slashes
The viewer trims trailing slashes from dataset base URLs before appending zarr
metadata paths, so both forms are accepted:
```bash
http://localhost:5173/?src=http://127.0.0.1:8005
http://localhost:5173/?src=http://127.0.0.1:8005/
```
Prefer the no-trailing-slash form in examples and logs as the canonical spelling.

### WASM 16-Dimension Limit (with automatic >16D fallback)
The compiled WASM kernels use fixed-size arrays (for performance) and support a
**maximum of 16 dimensions** on the fast path. `validate_ndim` **panics** (crate
is `panic = "abort"`) for `ndim > 16`, so those kernels must never be called above 16D.
- Functions affected: `calculate_effective_radii`, `mahalanobis_distance`, `compute_gsplats_attenuation`, etc.
- **>16D is fully supported (slower but works), automatically.** The TypeScript
  reference implementations in `wasm/typescript/` are uncapped, and the worker's
  `pickBackend(ctx, ndim)` (`workers/data-worker/state.ts`) transparently routes
  any `ndim > 16` operation to the TS backend instead of WASM. No caller action
  is needed — high-dimensional datasets just run on the TS path.
- Implication: the TS reference is not only a WASM-missing fallback, it is the
  production >16D backend — keep it in 1:1 sync with the Rust kernels (parity tests).

### GSplats with Fewer Than 3 Display Dimensions (2D/1D scenes)
The renderer's per-splat Cholesky buffer is **always** the 6-element packed-3D
layout, no matter how many dimensions are displayed. So a 2D scene
(`displayDims.length === 2`) produces only a 2×2 marginal and the third row must be
**synthesized** — see `compute_display_cholesky_3d` (Rust) / `computeDisplayCholesky3D`
(TS) in `wasm/*/gsplats_processing`.
- **Never pad the phantom diagonal with an epsilon.** In sum projection (additive,
  luminous, volumetric) the shader scales amplitude by the Gaussian's extent along
  the view ray, `sigmaRay = 1/√(rᵀΣ⁻¹r)`; an ε-thin splat viewed face-on is scaled
  by ~1e-5 and discarded, so the whole scene renders **black**. The diagonal is the
  geometric mean of the real Cholesky pivots (= `(det Σ_S)^(1/2n)`, rotation-invariant),
  giving the phantom axis the splat's own in-plane scale. `luxar.gsplats.lift`
  depends on this: its `opacity / (rayIntegralFactor · σ)` calibration holds for a 2D
  lift only because `√(σ·σ) == σ`.
- The dimension hazard is **two-sided**: >16D panics (above), and <3 *display* dims
  used to panic too (a hardcoded sub-ndim of 3 read `display_dims[2]` out of bounds).
  When touching these kernels, test `displayDims.length` of 1 and 2, not just 3.
- 2D gsplats are a first-class authoring path end to end (see the
  `demo_gsplats_2d_*` demos), spatial tiling included: BSP splitting needs only
  **2** spatial axes, so `luxar gsplat partition`, `lod --recipe
  tiles|overview|adaptive`, and `add_gsplats(partition=…)` all work on planar
  data. Only 1D input is rejected. Note the serialized BSP `axis` is a
  center-column index, which the viewer must map through `displayDims` to reach
  its own x/y/z (`render-order.ts`) — the two coincide only for `[0, 1, 2]`.

### ViewState.dimensions for extend_to_all
The `dimensions` field in ViewState is **required** for `extend_to_all` to work:
```typescript
// If extend_to_all is set but dimensions is undefined, the optimization is silently skipped
const viewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 5],
  dimensions: dims,  // REQUIRED for extend_to_all!
};
```

### nD Transforms on Non-Displayed Dimensions
`nd_transform` is separate from the 4x4 `transform`. It operates per-dimension on non-displayed dims:
- Continuous/discrete: `{"scale": float, "offset": float}` (affine)
- Categorical: `{"permutation": [int, ...]}` (relabeling)

**Viewer design**: Uses **inverse-query** approach — the query (slicePosition + tolerance) is inverse-transformed from world to local space ONCE (O(1)), rather than transforming millions of point coordinates (O(N)). No loader internals change.

See `docs/guides/specs/ND_TRANSFORMS_SPEC.md` for full details.

---

## Common Pitfalls and Solutions

### TypeScript: Event Listener Memory Leaks
**Problem**: Creating new bound function references on each call prevents proper cleanup.

```typescript
// ❌ WRONG - Creates new reference, removeEventListener won't work
addEventListener('resize', this.handleResize.bind(this));
removeEventListener('resize', this.handleResize.bind(this));  // Different reference!

// ✅ CORRECT - Store bound reference for cleanup
this.boundHandleResize = this.handleResize.bind(this);
addEventListener('resize', this.boundHandleResize);
removeEventListener('resize', this.boundHandleResize);  // Same reference
```

### TypeScript: Async Initialization Race Conditions
**Problem**: Multiple callers triggering async initialization concurrently.

```typescript
// ❌ WRONG - Non-atomic check
if (!this.initPromise) {
  this.initPromise = this.initialize();  // Race: two callers can both enter
}

// ✅ CORRECT - Atomic lock with cleanup
if (this.initLock) return this.initPromise;  // Return existing promise
this.initLock = true;
try {
  this.initPromise = this.initialize();
  await this.initPromise;
} finally {
  this.initLock = false;  // Always clear lock
}
```

### TypeScript: Over-Mocking in Tests
**Problem**: Mocking entire classes defeats the purpose of testing.

```typescript
// ❌ WRONG - Tests verify mock behavior, not real code
vi.mock('../rendering/point-material', () => ({
  PointMaterial: vi.fn().mockImplementation(() => ({
    uniforms: { fov: { value: 60 } },
    dispose: vi.fn()
  }))
}));

// ✅ CORRECT - Mock only external dependencies, test real code
import { PointMaterial } from '../rendering/point-material';
// Let PointMaterial run real shader generation code
// Only mock THREE.ShaderMaterial if absolutely necessary
```

**Testing Principle**: Mock external dependencies (network, file system), not your own code. If code is hard to test without mocking, refactor for testability (dependency injection, pure functions).

---

## Luxar Conventions

### Physical Units
Support: nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au

### Geometry Types & Attributes
- **Points**: positions (Float32, nD, required), colors (Uint8/Float32 HDR), radii (Float32), sharpness (Float32)
- **Lines**: vertices (Float32, nD, required), widths (Float32, required), segments (Uint32, auto-generated), colors (Uint8/Float32), sharpness (Float32)
- **GSplats**: centers (Float32, nD, required), amplitudes (Float32, required), cholesky_factors (Float32, required), colors (Uint8/Float32, RGB or RGBA — the optional alpha is per-splat opacity, consumed by every blending mode; mapped to optical depth in `volumetric`)
- **Mesh** (renderable, shaded): vertices (Float32, nD, required), faces (Uint32 `(F,3)`, required), normals (Float32 `(V,3)`) + a required `normal_dims` companion attr naming which three dimensions they describe, colors (Uint8/Float32, RGB or RGBA), scalars (Float32). No per-element size — a triangle's extent comes from its own vertices, so a mesh adds zero extent padding to scene bounds. No LOD, no `kind=partition`, no spatial index, no `volumetric` blending; each is refused with an explanation rather than silently degraded.

### Transforms
- 4x4 matrices stored as 16-element lists
- Transpose for THREE.js compatibility (see Critical Gotchas)
- Use `luxar.transforms` module (translate, rotate, scale, compose)

### Dimensions
- Define at Scene level using `Dimensions` and `Dimension` classes
- Include: name, unit, range, step, display status
- Step sizes used for keyboard navigation in viewer

### nD Navigation
- Keyboard: 1-9 selects dimension, `[`/`]` navigates
- Radius-based slicing: geometry visible based on nD hypersphere intersection

---

## Pre-commit Checklist

```bash
make test-all                    # All tests pass
make check-all                   # Linting, type checking
pnpm run format                  # Format TypeScript (from luxar-viewer/)
```

Before PR/merge:
- Full E2E suite: `cd packages/luxar-viewer && pnpm test:e2e`
- Update READMEs if functionality changed
- Update LUXAR_ZARR_FORMAT.md if data format changed

---

## Development Philosophy

1. **No backwards compatibility burden** - Early-stage project, just change it
2. **Complete before perfect** - Avoid over-engineering
3. **Minimum viable solution** - Don't add features/refactoring beyond what's asked
4. **Test everything** - Never skip tests, fix or mock them properly
5. **Keep docs in sync** - Update READMEs with code changes
6. **Use existing patterns** - Follow codebase conventions
7. **Multiple Agents at Work** - Other agents are likely at work on the same codebase and files, be mindful and careful to not delete/destroy/stash the work of the other agents.
8. **Ask Questions when Unsure** - Ask the user questions when you are genuinely unsure about a course of action. **ALWAYS use the `AskUserQuestion` interactive tool** for any decision point — never pose choices as inline prose. If the tool isn't loaded, load it via `ToolSearch` first.

### Naming Conventions
- Example files: `*_example.py` or `*_example.luxar.zarr`
- Temp files: Put in `delme/` directory
- Example outputs: Generated to `datasets/examples/` (via `get_examples_output_dir()`)
- Demo outputs: Generated to `datasets/demos/` (via `get_demos_output_dir()`)
- Never commit `.zarr` directories (in .gitignore)

---

## Detailed Documentation

| Topic | Location |
|-------|----------|
| Build System & Dev Setup | `docs/guides/developer/BUILD_SYSTEM_SPEC.md` |
| E2E Testing Quick Ref | `docs/guides/user/E2E_TESTING_GUIDE.md` |
| Playwright Full Guide | `docs/guides/developer/PLAYWRIGHT_GUIDE.md` |
| Data Format Spec | `docs/guides/user/LUXAR_ZARR_FORMAT.md` |
| HDR Color Guide | `docs/guides/user/HDR_GUIDE.md` |
| Network Simulation | `docs/guides/developer/NETWORK_SIMULATION_SPEC.md` |
| Console Logging Style | `docs/guides/developer/CONSOLE_OUTPUT_STYLE.md` |
| Documentation Quality | `docs/guides/developer/DOCUMENTATION_QUALITY.md` |
| Changelog | `CHANGELOG.md` |

---

## Architecture Overview

```
Python Data -> Luxar Core -> Zarr Archive -> Luxar Viewer -> WebGL -> Display
```

### Scene Graph
- Scene (root) contains Groups, Points, Lines, GSplats, and Mesh
- Groups can nest (hierarchical)
- Transforms compose hierarchically (parent -> child)
- Four geometry types: Points (soft-edged spheres), Lines (width-tapered curves), GSplats (oriented Gaussians), Mesh (shaded triangle surfaces)

### Performance Targets
- 100K-10M elements for smooth interaction
- Chunk size: 16KB-256KB (target 64KB; see `TARGET_CHUNK_BYTES` in `typing_utils/constants.py`)
- Compression: Blosc zstd level 9, width-aware shuffle by dtype (see `encoding/compression.py`)
