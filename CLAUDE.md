# CLAUDE.md

Guidance for Claude Code when working with this repository.

**Luxar** is a high-performance system for compiling and visualizing arbitrary-sized nD scientific scenes. It renders four first-class geometry types — **Points**, **Lines**, **Gaussian Splats**, and **Mesh** (triangle surfaces). Mesh is the newest and the only *shaded* one — the other three are purely emissive — via a light-free view-anchored offset key, and it is now feature-complete at the UI level: picking (at VERTEX granularity, keyed on `gl_VertexID` rather than an element-texture texel), the Layers-panel appearance controls, monitor/stats/debug counts. The docs pass is done and real WebGPU is verified pixel-equivalent to WebGL (see `docs/specs/MESH_NODE_SPEC.md` §11 and the CHANGELOG A/B notes). The contract still names the writable and drawable sets separately — `geometry_types` and `loader_types` — because a type becomes authorable before it becomes drawable; they simply agree on all four today.

## Issue / PR Coordination

Before starting work for a numbered issue, run
`hatch run python scripts/check_open_issue_pr.py <issue-number>`. Re-run it
immediately before `gh pr create`; when checking work already attached to a PR,
pass `--exclude-pr <your-pr>` so it does not match itself. Exit status 1 means an
open PR already declares that it closes the issue; continue or coordinate on
that PR instead of opening another. Exit status 3 means the GitHub query failed,
so do not treat it as either claimed or unclaimed. This is a manual coordination
convention, not a CI gate. Before closing a duplicate PR, use the script's
`--exclude-pr ... --compare-pr ...` mode, inspect shared paths for unique hunks,
transfer every duplicate-only change, and carry any materially different review
conclusion onto the surviving PR.

## Quick Reference

### Python (use Hatch)
```bash
hatch run test              # Run tests
hatch run test-cov          # Tests with coverage
hatch run python script.py  # Run script
hatch run python -m ruff check .  # Lint
hatch run mypy packages/luxar/src/luxar/ scripts/ci_queue_scan.py  # Type check
```

The hatch env pins `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` (see the note in
`pyproject.toml`) — the test suite is thousands of tiny tensor ops and loses
several-fold to fork-join overhead otherwise. That pin also applies to real work
run through hatch, so for a CPU fit, a demo, or a benchmark, export the width
you want — an explicit value wins. Set `MKL_NUM_THREADS` too, not just
`OMP_NUM_THREADS`: torch takes its intra-op count from MKL here, so overriding
OMP alone still leaves `torch.get_num_threads() == 1`.
```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
  hatch run luxar gsplat fit vol.tiff out.gsplats.zarr --device cpu
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
make test-fast    # INNER LOOP: Python (-m 'not slow', xdist) + TS units, no coverage.
                  # ~6-7 min whole-suite; SCOPE it for a real loop:
                  #   make test-fast PYTEST_ARGS='packages/luxar/src/luxar/encoding'
                  # Not a gate — no slow tests, no coverage thresholds, no E2E.
                  # LUXAR_PYTEST_JOBS=12 to tune (I/O-bound, so it buys little).
make test-all     # All tests (Python incl. CUDA + WASM/Rust + TypeScript + Go launcher)
make test-cov-all # Coverage: Python (minus `-m slow`) + TypeScript
make test-python  # Python tests only
make test-e2e     # Full Playwright E2E suite (~17 min)
make test-e2e-smoke  # E2E smoke subset (the specs CI would run)
make test-perf-e2e   # Opt-in Playwright performance suite
# check-all is NOT read-only: `check-static` begins with `format`, so it
# REWRITES packages/luxar/src and scripts. When other agents/people are editing
# the same tree, use the read-only scoped targets instead (listed right below it).
# It runs NO tests — `make test-all` is the single place those execute.
make check-all    # All quality checks (Python, TypeScript, Rust, Go) — reformats
make lint-python        # read-only: ruff check
make check-complexity   # read-only: ruff C901 ratcheted against scripts/complexity_baseline.json
make type-check-python  # read-only: mypy
make security           # read-only: bandit
make check-typescript   # read-only: typecheck + lint + unit tests
make check-rust   # Rust type/lint checks (cargo check + clippy)
make check-docs   # REQUIRED gate mirror: completeness + TypeDoc ratchets +
                  # warning-fatal Sphinx. A new file under docs/ must be listed
                  # in a docs/index.rst toctree or this goes red.
make check-docs-external-links  # opt-in external HTTP link audit (not a gate)
make check-demo-links  # opt-in demo click-through audit (reports only; not a gate)
make check-zenodo-live          # opt-in live Zenodo manifest-pin audit (not a gate)
make check-cold-fetch           # opt-in hosted demo cold-fetch gate before payload removal
make check-external-references  # aggregate external audits (report-only, non-gating)
make check-knip   # REPORT only (non-gating): unused viewer files/exports/deps
make check-gallery-staleness  # REPORT only; requires full Git history
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
#   viewer via ?cacheBudgetMB=. One third also becomes the auto GPU-geometry
#   residency signal. WKWebView has neither performance.memory nor deviceMemory,
#   so the launcher default (2048) raises that budget from 512 to 716 MB; lower
#   it on a constrained machine.

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
- Python 3.12+ (usually pre-installed; `python3.12` is the usual HPC module)
- Git and curl
- **Git LFS** (optional, required for the Dip-C demo payload): `brew install git-lfs` (macOS) or `sudo apt-get install git-lfs` (Ubuntu)
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

The Dip-C demo payload, `dipc_genome/dipc_gm12878.npz`, is stored using Git LFS.
Other hosted demo payloads are fetched from their checksum-pinned records.

```bash
# Install Git LFS (one-time)
brew install git-lfs        # macOS
sudo apt-get install git-lfs  # Ubuntu/Debian

# Initialize Git LFS (one-time)
git lfs install

# Pull LFS files (when needed)
git lfs pull

# Verify LFS files (should show actual sizes, not ~100 bytes)
ls -lh packages/luxar/src/luxar/demos/data/dipc_genome/dipc_gm12878.npz
```

If the Dip-C demo reports a tiny or missing payload, pull the LFS file. For
other hosted demos, inspect the manifest/record fetch error instead.
See `packages/luxar/src/luxar/demos/data/README.md` for details.

See `docs/guides/developer/BUILD_SYSTEM_SPEC.md` for complete documentation.

### Luxar CLI
```bash
luxar demo                       # List the 89 bundled demos (table)
luxar demo run lorenz            # Run a demo by key/index (forwards -- args)
luxar demo stop                  # Stop running demos and free their ports (--dry-run lists)
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
luxar info <data.luxar.zarr> --stats   # Dataset info (--stats also reports the chunk layout)
luxar profiles                   # Network simulation profiles
# Re-chunk a store that is ALREADY on disk so it streams well — one
# structure-preserving pass, no refit/source volume/GPU. Only zarr chunk shapes
# change: values stay bit-identical and the spatial-index grid (`chunk_size` /
# `chunk_bounds`) never moves, since every new chunk is a whole multiple of its
# node's atom. Nothing is chunked SMALLER than it already is. The output gets a
# fresh `content_hash` + a `chunk_layout` attr, because chunk keys now cover
# different rows and a warm viewer cache validating on an unchanged hash would
# serve stale chunks — so prefer publishing under a NEW URL prefix.
# `--profile hosting` (256 KB) trades PARTIAL-QUERY bytes for full-load
# requests: a Points/GSplats node the viewer SLICES into pays 4.5x the bytes per
# partial hit vs `local`. Size up only when the access pattern is "load whole".
luxar optimise scene.luxar.zarr out.luxar.zarr             # 64 KB default
luxar optimise scene.luxar.zarr --dry-run                  # report the plan, write nothing
luxar optimise scene.luxar.zarr out.luxar.zarr --profile hosting  # hosting 256 KB / local 64 KB / archive 1 MB
luxar optimise scene.luxar.zarr out.luxar.zarr --verify    # re-read the output, compare every array
luxar optimise arbitrary.zarr out.zarr --generic           # a plain (non-Luxar) zarr store
# Re-derive a store's LOD switch thresholds IN PLACE — attrs only, no chunk data
# moves. Every `kind=lod` group still on the legacy `coverage` diagonal metric
# (or carrying no `selector`) gets screen-occupancy-halved thresholds and a
# `screen-area` stamp; the fills-screen anchor only under a REAL (>1 part)
# partition. An EXPLICIT opt-in and nothing else may trigger it: an authored
# `coverage_fractions=[...]` list and a legacy derived one are indistinguishable
# on disk, so this may override a deliberate choice — hence the printed old→new
# audit line, `--dry-run`, and `--group`. A group already on `screen-area` is
# skipped, so a second run changes nothing, `content_hash` included. Exits 1 when
# a ladder was left alone (unsupported selector → `gsplat migrate-format` first;
# unresolvable finest element count).
luxar restamp-lod scene.luxar.zarr                         # every legacy ladder
luxar restamp-lod scene.luxar.zarr --dry-run               # report the old→new ladders
luxar restamp-lod scene.luxar.zarr --group tiled/part_0    # one ladder (repeatable)
luxar export scene.luxar.zarr -o my_export/             # Export scene + viewer as standalone offline folder
luxar export scene.luxar.zarr -o my_export/ --open      # Export and serve in browser
luxar export scene.luxar.zarr -o my_export/ --overwrite # Overwrite existing export
luxar export scene.luxar.zarr -o out/ --native macos    # Native macOS .app bundle (requires `make build-launchers`)
luxar export scene.luxar.zarr -o out/ --native macos,linux-amd64,linux-arm64 --name MyScene
```

### GSplat CLI (fitting, converting, rendering, merging)
```bash
# Diagnose an existing store; --histograms enables the optional info histograms.
luxar gsplat doctor splats.gsplats.zarr --histograms --bins 40

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
#
# STAY ON `auto` UNLESS YOU HAVE MEASURED OTHERWISE. A `pN` floor subtracts the
# Nth percentile OF NON-ZERO VOXELS, which on sparse data lands wherever the sparsity
# puts it, not where the noise ends. On a 96x640x640 crop of a sparse light-sheet
# brain (1.01% of voxels foreground = >10% of max; 12.9% in the dim band 1-10%),
# p99 sat at 1.34% of THAT CROP's max — squarely inside signal. (Over the whole
# stack the same percentile is 0.05% of peak: a pN floor moves with whatever you
# point it at, which is the problem.) Every arm at a fixed seed budget, scored
# against the UNFLOORED original:
#
#   floor   splats   global   foreground   dim-band mass recovered
#   none    47,172   41.90    28.49 dB     42.0%
#   auto    46,020   41.76    28.24 dB     40.7%   <- within 0.25 dB of none
#   p95     39,859   40.33    27.04 dB     23.0%
#   p99     15,483   35.81    18.86 dB      0.6%   <- erases the dim band
#
# A high floor also LOOKS better in a MIP (the haze is gone, the render is
# crisper than its own source) — that is the trap. Judge a floor on foreground /
# dim-band PSNR against unfloored data, never on how the render looks. Handle
# residual haze with the display window / opacity, not by destroying data at fit
# time. And never port a floor choice between datasets without retesting.
luxar gsplat fit volume.tiff splats.gsplats.zarr                 # --floor auto (default, recommended)
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
# An integer `--seeds K` is a WHOLE-VOLUME budget (what a default `cal`
# reports): a tiled fit DIVIDES it across the tiles that survive the resolved
# floor plus Hann window instead of giving each tile the full count. Every
# worker derives the same non-empty count; K below it gives 1 per such tile.
# The share is equal, not occupancy-weighted, so uneven grids can misallocate K.

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
# additive: --n-lods/--add-method/--breakpoints; substitutive:
# --compression-factor/--levels/--subst-method/--coarsen-dims.
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
# `--merge-refine volume` re-opens THIS input at merge time and re-fits each tile
# against its own crop (and each stacked timepoint against its own slice) — the
# highest-fidelity coarse levels. Needs --axes recorded and a single channel; both
# are validated at PLAN time, so a typo costs nothing rather than surfacing after
# every tile has been fitted. Also on `batch-fit submit` (baked into the Slurm
# merge job) and on `batch-fit merge` itself as plain `--refine`.
luxar gsplat batch-fit run vol.zarr out/ --gpus auto --axes time,z,y,x \
    --merge-recipe levels --merge-levels 1 --merge-refine volume --merge-refine-iters 300
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
luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --axes time,z,y,x \
    --merge-recipe levels --merge-refine volume        # + per-tile volume re-fit at merge
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
# When the colormap carries an exact scientific color encoding that must survive
# to the screen (ACES shifts hues), pick by RANGE. Inside [0, 1] `None` is an
# exact passthrough (exposure/offset/gamma still apply — the shader runs
# them before the tone-mapping switch). Neutral is NOT a passthrough: even below
# its knee it subtracts an offset taken from the channel MINIMUM, so anything
# but a fully saturated colour moves, dulling the encoding you meant to protect:
luxar gsplat convert splats.gsplats.zarr scene.luxar.zarr --colormap plasma --tone-mapping None
# Over range NO operator is faithful, and they fail differently: Neutral keeps
# the HSV hue angle exactly but sheds chroma (at peak 100 a saturated colour
# comes out at saturation 0.06, essentially white), while a `None` clamp
# distorts BOTH — it holds full saturation only where the darkest channel is
# already 0 ((100, 0, 0) -> (1, 0, 0); (2, 0.5, 0.5) -> (1, 0.5, 0.5) drops
# saturation 0.75 -> 0.5), SHIFTS hue when channels clip unequally ((2, 1, 0)
# goes hue 30deg -> 60deg) and flattens everything above 1.0. Bring the scene
# back into [0, 1] with --intensity/exposure and use `None`, or accept ACES's
# filmic rolloff. Decide with an actual render, not from first principles.

# Render gsplats back to a background-relative volume. Partition and nested
# trees render their default-selected leaves without first flattening the store.
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
# to pick K* in a principled way. .gsplats.zarr is format v3.4 (a node tree —
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
# Scale is not the only axis: when first paint is request-constrained, eager-rung
# count picks the recipe. See "First paint cost" in the gsplat-pipeline skill.
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
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --add-method self_energy
# `-m radial` = the REVEAL: orders concentric shells around the node's own bbox
# centre (NOT the scene origin), so a streaming prefix grows outward from the
# middle. Authoring only — no viewer changes, nothing about how data is DISPLAYED.
# Available on GSplats, Points and Lines (on Lines it orders whole polylines, so
# every prefix keeps valid segment topology). A radial ladder deliberately carries
# NO energy stamps: the viewer brightens an incomplete ladder by 1/e(k), which is
# backwards for a reveal (a partial object at FULL brightness, not a dim whole).
# Knobs: reveal_centre / spatial_dims (Python), --reveal-centre / --spatial-dims.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream -m radial --n-lods 6
# STREAMING breakpoints: `-b stream:C` = geometric ladder (first chunk C splats,
# then doubling), sized per part/level. Or derive C from a download budget with
# `--target-ms` (+ `--bandwidth-mbps`, default 25; `--bytes-per-splat` override;
# bytes/splat measured from the input store, logged). First chunk ≈ target-ms of
# download → fast first paint; the viewer streams additive sub-LODs progressively.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream --target-ms 200
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe stream -b stream:14000
# ON A NODE THE VIEWER SLICES (any hidden dim), COUNT PER SLICE, NOT PER NODE.
# An explicit `-b stream:C` fixes an ABSOLUTE first rung for the whole node, but
# only one hidden coordinate is on screen, so each slice receives part of C;
# Nexrad's explicit `stream:20000` ladder has p05 = 4 splats per played coordinate.
# Before the CLI scaling fix, drosophila's `--target-ms` resolved to a 20,833-splat
# rung 0 for 500 timepoints; the measured slices had median = 45, p05 = 7, min = 1,
# and playback rendered an empty frame (#2374/#2376). Prefer `--n-lods 3..4` for
# an aggregate share that stays stable as slice count changes, but verify long or
# non-uniform axes: the global prefix can still starve sparse slices. The CLI now
# scales `--target-ms` by the observed slice count and LOGS the multiplier; read
# that line rather than assuming the number you typed is what renders.
# Count stops as distinct OCCURRING combinations over all hidden axes: not the
# product of per-axis cardinality, and not the declared Dimension range (that
# demo declares 500 timepoints and its coarsest rung has data at 499).
# `hatch run check-demo-ladders` fails a built store whose SPARSEST slices fall
# below the floor — it measures the 5th percentile, since a ladder starves at its
# sparsest slice and one busy coordinate masks hundreds of starved ones.

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
# result. Precision-only; additive ladders are dropped (rebuild with
# `lod --recipe stream`), and staging beside the destination needs roughly one
# uncompressed flat payload of temporary free space.
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
# --subst-method); per-tile levels are stream-laddered by default.
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
    --subst-method kmeans-lloyd --lloyd-iters 5 --device cpu
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
# Needs the volume in hand, and is available at all three entry points:
#   lod --target … | fit --recipe levels --refine volume (no --target: the volume
#   being fitted is already in hand) | batch-fit merge --recipe levels --refine
#   volume (re-opens the source the manifest recorded, cropping per tile).
# Works on levels/overview AND per-tile `adaptive`, with or without barrier dims:
# each re-fit is handed the sub-volume it is responsible for (a barrier group gets
# its own timepoint slice, a tile its own crop), and a per-tile re-fit that leaves
# its tile is discarded in favour of the merge. The volume is only SLICED, never
# read whole, so a lazy zarr target stays lazy. `--refine-iters` default 300 here.
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels --target vol.tiff --refine volume
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe adaptive --target vol.tiff --refine volume
# A STACKED target needs --target-axes: fitted splats put spatial dims first and
# the stacked axis LAST, while the source array is usually time-FIRST, so the
# identity map would target the wrong axis. (Contrast --timepoint, which slices
# ONE timepoint out; --target-axes keeps the axis so the re-fit walks it.)
luxar gsplat lod tl.gsplats.zarr out.gsplats.zarr --recipe levels --refine volume \
    --target movie.zarr --array-key h2afva/fused --target-axes time,z,y,x --coarsen-dims 0,1,2
# Barrier-aware coarsening (levels/overview/adaptive): --coarsen-dims
# lists the center-column indices coarsening may merge over; the rest become hard
# barriers (a categorical/time/channel axis), so coarse splats never blend across
# them. Default = all dims. (The Python scene API defaults to Auto = coarsen
# displayed dims, group by non-displayed; standalone gsplats have no display info
# so the CLI takes explicit indices and warns on >3D input without the flag.)
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr --recipe levels --coarsen-dims 1,2,3
# LOD switch thresholds (ANY recipe with a kind=lod group — overview,
# levels, adaptive) are auto-derived by SCREEN-OCCUPANCY HALVING (count-
# independent), stamped as selector="screen-area": each coverage_fraction is a
# literal screen-area fraction (projected bbox rect area / viewport area). For
# `levels` (a WHOLE-OBJECT ladder) the finest level shows while the object
# occupies at least HALF THE SCREEN (finest anchor 0.5) and each halving of
# occupied area steps one level coarser (NDC-fraction metric → identical on
# any monitor — no threshold knob). Legacy stores / explicit
# coverage_fractions=[...] lists keep selector="coverage" (diagonal metric,
# thresholds in [0,4]); the viewer supports both.
# EXCEPTION — `adaptive` and `overview` are PARTITION-BOUND and keep the
# fills-screen anchor (finest = area 1.0 = the tile alone fills the screen), via
# `partitioned_coverage_fractions`. For `adaptive` that is geometry (each lod
# group's bbox is one BSP tile, so it projects to a fraction of the whole object);
# for `overview` it is the recipe's contract — the coarse cap is what you see at
# the opening framing and the fine partition is the zoom-in branch, so it does NOT
# show full detail at a normal full-frame view. Use `levels` if you want that.

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
# v3.0-v3.1 with pre-v3.2 pixel_size lod selector attrs) → v3.4
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr               # single file
luxar gsplat migrate-format old_pyr/ v3.gsplats.zarr                          # substitutive directory

# Re-quantize a fitted (current-format) .gsplats.zarr's Cholesky encoding (writes a copy).
# Structure-preserving round-trip (leaf/lod/partition/nested + fitting/pipeline
# groups kept); auto/memory also re-quantize the CENTERS to per-axis uint16
# fixed-point, so on ordinary spatial data centers are bit-exact only under
# -e precision. Three exceptions stay exact in every mode: a GRIDDED axis (a
# stacked time/channel centers column that lands on a regular lattice) keeps
# uint16 but has its grid snapped onto the data's own spacing; a LUT-eligible
# centers array is stored verbatim as lut_uint8 (~1 B/value); and an axis that
# is NEITHER gridded nor LUT-eligible whose grid would displace splats past
# their own sigma FOR MORE THAN 0.1% OF THE SPLATS falls back to float32 (a
# smaller degenerate population is quantized away silently — see
# MAX_UNREPRESENTABLE_SPLAT_FRACTION). Decode is always float32 so
# viewer/GPU/WASM are unaffected. Unlike migrate-format (legacy→current,
# float32 vs AUTO-uint16 only) this exposes the full ladder incl. memory=uint8.
luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory      # uint8 (smallest, ~93 dB)
luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision  # float32 (exact/archival)

# Partition a flat (matrix-shaped) store into one kind=partition file via spatial
# BSP (--indices removed). A partition/nested input must be `flatten`ed first.
luxar gsplat partition splats.gsplats.zarr part.gsplats.zarr --parts 4               # target part count
luxar gsplat partition splats.gsplats.zarr part.gsplats.zarr --max-elements 100000   # per-part cap
luxar gsplat partition splats.gsplats.zarr part.gsplats.zarr --parts 4 --rule sah    # median|midpoint|sah

# Merge multiple datasets. A partition must be `flatten`ed first.
luxar gsplat merge a.gsplats.zarr b.gsplats.zarr -o merged.gsplats.zarr
luxar gsplat merge t0.zarr t1.zarr -o 4d.zarr --as-dimension --values 0,1
luxar gsplat merge ch0.zarr ch1.zarr -o multi.zarr --channel-colors "#ff0080,#00ff00"

# Slice by coordinate ranges (numpy-style). A partition must be `flatten`ed first.
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

# Open a gsplat dataset in napari. Partition/nested trees use their
# default-selected leaves.
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

# Reduce a dataset to a TARGET SPLAT COUNT (one flat result) — the "this fit is
# bigger than I need" tool, distinct from `cull` (removes by a quality threshold)
# and `lod` (builds a multi-level structure). Two families:
#   merge   cluster neighbours into representatives carrying their combined mass
#   prefix  keep the first N of an additive ordering (discards splats, dims)
# `auto` follows the MEASURED crossover: merge below 50% kept, prefix at/above.
# Foreground PSNR on a 1.65M-splat light-sheet fit (global PSNR flatters
# everything on a 97.8%-empty stack, so it is not the number to steer by):
#   kept  50%: merge 44.5 / prefix 45.5 dB   <- prefix wins, little to summarise
#   kept  25%: merge 41.7 / prefix 39.1 dB
#   kept  10%: merge 38.3 / prefix 34.5 dB   <- ~10x smaller, recommended point
#   kept   1%: merge 33.1 / prefix 29.6 dB   <- merge wins by 3.5 dB
# Quality falls ~3-4 dB per halving with NO knee, so pick from the curve.
# A partition must be `flatten`ed first (decimate returns a single flat leaf).
luxar gsplat decimate in.gsplats.zarr out.gsplats.zarr --target 165000   # absolute count
luxar gsplat decimate in.gsplats.zarr out.gsplats.zarr -f 0.1            # share of input
luxar gsplat decimate in.gsplats.zarr out.gsplats.zarr -f 0.1 -m merge   # force a family
# Python: `from luxar.gsplats.lod import decimate` (target=int count | float fraction)

# Inspect, cull, and filter. A partition must be `flatten`ed before cull/filter.
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
- **Minimum coverage**: Enforced by `pyproject.toml` for Python and `packages/luxar-viewer/coverage-thresholds.mjs` for TypeScript
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
Instead of running all E2E tests at once (which can timeout or be overwhelming), run them by topic.
Run `pnpm test:generate-fixtures` once first: the Playwright pre-flight requires the generated
zarr fixtures for EVERY chunk, not just the two that read them directly (set
`LUXAR_E2E_NO_FIXTURES=1` to skip the check for a chunk you know needs none).
Run `make run-examples` from the repository root too: the pre-flight warns and continues on a
stale example stamp, but specs that read those stores may fail against outdated data.
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
  post-processing-pipeline.spec.ts cinematic-auto-framing.spec.ts rendering-controls.spec.ts \
  ortho-mode.spec.ts
# Line joint artifacts (#780/#785/#790) — scores each joint topology as its
# own band of one frame with TWO metrics (local-median outliers + axial flux
# ripple; the first is structurally blind to the bead-notch class the second
# catches). Needs generate-fixtures.
npx playwright test line-join-artifact.spec.ts

# Data & I/O
npx playwright test data-integrity.spec.ts dataset-switching.spec.ts real-dataset-loading.spec.ts \
  url-parameters.spec.ts python-typescript-integration.spec.ts luxar-serve-integration.spec.ts \
  zipped-store-loading.spec.ts

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
- Run `pnpm test:generate-fixtures` before any Playwright run (the pre-flight enforces it)
- Run `make run-examples` from the repository root before direct Playwright runs (stale examples warn and continue, but example-reading specs may fail against outdated data)
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

### zarr library version vs zarr on-disk FORMAT (two separate axes)
Luxar runs on **zarr-python 3** (`zarr>=3.2,<4`) and writes **zarr format 3** by
default, while READING both formats. Never conflate the two axes. Both are pinned
in one place — `packages/luxar/src/luxar/_zarr_compat.py` — and all writing goes
through its helpers rather than `zarr.*` directly:

```python
from luxar._zarr_compat import open_group, create_array, consolidate, memory_group
```

**A tree holding BOTH formats is the normal steady state.** Existing
`.luxar.zarr` / `.gsplats.zarr` stores stay format 2 and are not rewritten; only
new output is format 3. Set `LUXAR_ZARR_FORMAT=2` to write format 2 for a tool
that cannot read 3 (an env var, not a flag, because the writing process is often
a batch-fit worker or Slurm task rather than the one you invoked).

Why it matters, concretely:
- **Never name a metadata document.** `.zgroup` / `.zattrs` / `.zarray` /
  `.zmetadata` exist only at format 2; format 3 has one `zarr.json` per node with
  attributes nested under `attributes`, `c/0/0` chunk keys, and consolidated
  metadata *inside* the root `zarr.json`. Use the facade's bi-format readers —
  `read_array_meta`, `read_node_attrs`, `is_consolidated`, `read_consolidated_attrs`
  — for anything that inspects a store on disk. Every bug found during the
  format-3 migration was a literal document name, and **not one of them raised**:
  a validator that passed a corrupt tile, a cache probe that served stale data
  forever, a dataset browser that could not see v3 stores, an encoding classifier
  that answered "unclassifiable", a batch measurement that silently fell back to
  its analytic estimate. The failure mode is always a plausible wrong answer, so
  grep for the document names rather than trusting the test suite to go red.
- **numcodecs objects are format-2 currency.** A format-3 array REJECTS them
  (`TypeError: 'Blosc' object is not iterable`). `create_array` translates
  compressors and filters to `zarr.codecs` equivalents, keyed on the format of
  the GROUP being written — not the global default, since writing into a legacy
  v2 store while the default is 3 is routine. There is no zlib codec in
  zarr-python 3 at all; use gzip. `numcodecs>=0.16` is a DIRECT dependency for
  this reason: below it zarr's format-3 `BloscCodec` cannot forward the evolved
  `typesize` to blosc, so the measured byte shuffle silently becomes a no-op
  (~12.5% larger chunks) while the metadata still records the shuffle that never
  happened. Assert compression on the stored BYTES, not on the recorded config.
- **An omitted compressor is not "no compressor".** zarr 3's `compressors`
  defaults to `"auto"`, which is Blosc/lz4/clevel-5 at format 2 but zstd at
  format 3 — not the same bytes. Some Luxar arrays must be RAW and the rest carry
  a measured zstd-9 policy, so `create_array` takes `compressor` explicitly; an
  AST test fails the build if a production call site omits it.
- **`data=` and `shape=` are mutually exclusive in zarr 3** (zarr 2 allowed both).
  `create_array` accepts both and forwards only what zarr 3 permits.
- **Edit a store in place through the facade, never `zarr.open_group`.** Format 3
  allows a consolidated index on ANY group, and the facade's deliberate
  `use_consolidated=False` bypasses only the ROOT one. Re-opening an
  already-consolidated store with plain zarr hands back nodes built FROM the root
  index, so re-consolidating serializes that stale tree out as a NESTED index —
  after which reads return pre-edit attributes even though every document on disk
  is correct. Silent, as usual. The facade's tree carries no index to
  re-serialize, leaving exactly one at the root (the format-2 invariant
  everything already assumes).
- **Consolidating a v3 store warns**, and `ZarrUserWarning` subclasses
  `UserWarning` — so under `-W error` (which several tests use around a whole
  compile) saving FAILS with "Could not finalize Zarr store". Suppressed inside
  `_zarr_compat.consolidate()`. Never "fix" it by not consolidating: the viewer
  builds its entire scene graph from that index and has no directory-walk
  fallback, so the store would load as an empty scene.
- Reading is version-agnostic: zarr-python 3 opens v2 *and* v3, which is the point
  of being on 3.x — 2.18 could not open a v3 store at all.

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
- Functions affected: `calculate_effective_radii`, `mahalanobis_distance`, `project_gsplats_nd_to_3d`, etc.
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
- **GSplats**: centers (Float32, nD, required), amplitudes (Float32, required), cholesky_factors (Float32, required; chol(Σ), scale-like diagonal), colors (Uint8/Float32, RGB or RGBA — the optional alpha is per-splat opacity, consumed by every blending mode; mapped to optical depth in `volumetric`)
- **Mesh** (renderable, shaded): vertices (Float32, nD, required), faces (Uint32 `(F,3)`, required), normals (Float32 `(V,3)`) + a required `normal_dims` companion attr naming which three dimensions they describe, colors (Uint8/Float32, RGB or RGBA), scalars (Float32). No per-element size — a triangle's extent comes from its own vertices, so a mesh adds zero extent padding to scene bounds. Three structural paths are supported: `kind=partition` (`add_mesh(partition=…)`, spec §9.2), *substitutive* LOD (`add_mesh(substitutive_lod=…)`, decimated by `luxar.mesh.decimate`), and a spatially coherent *reveal* additive ladder (`add_mesh(additive_lod={"method": "radial"})`) — though no two of them in the same call. No additive (prefix) LOD ladder over an *arbitrary* order — a prefix of an arbitrarily ordered index buffer is a holed surface, not a coarser one — so a non-reveal method and `volumetric` blending are both still refused with an explanation rather than silently degraded. No spatial index (`ordering="none"`): a mesh loads whole.

### Transforms
- 4x4 matrices stored as 16-element lists
- Transpose for THREE.js compatibility (see Critical Gotchas)
- Use `luxar.transforms` module (translate, rotate, scale, compose)

### Dimensions
- Define at Scene level using `Dimensions` and `Dimension` classes
- Include: name, unit, range, step, display status
- Step sizes used for keyboard navigation in viewer

### nD Navigation
- Keyboard: 1-9 selects a non-displayed dimension, `[`/`]` navigates
- Radius-based slicing: geometry visible based on nD hypersphere intersection

---

## Changelog

Do NOT edit `CHANGELOG.md` directly in a PR. Add a fragment file
`changelog.d/<PR-or-issue>.md` containing the entry in the house style (a
`#### Title` line + prose paragraphs). One file per PR means `CHANGELOG.md` is no
longer a rebase-conflict magnet; `make changelog` folds fragments into it at
release-prep. See `changelog.d/README.md`. (Not every PR needs one.)

## Pre-commit Checklist

While iterating, use the fast loop — it deselects `slow`, runs pytest under
xdist and skips coverage:

```bash
make test-fast                                                  # whole suite, ~6-7 min
make test-fast PYTEST_ARGS='packages/luxar/src/luxar/encoding'  # scoped, ~35 s
make test-fast PYTEST_ARGS='-k colormap'
```

Scope it if you want a real edit-run-edit loop: with `slow` deselected the
remaining Python suite is I/O-bound on zarr small-file writes, so more workers
barely help (`LUXAR_PYTEST_JOBS=12` buys ~11% over the default 6).

`test-fast` is NOT a gate: no `slow` tests, no coverage thresholds, no E2E.
Before pushing, run the real thing:

```bash
make test-all                    # All tests pass (Python incl. slow, Rust/WASM, TS, Go)
make check-all                   # Linting, type checking — static only, runs no tests
make check-docs                  # Documentation gate (required check in CI)
pnpm run format                  # Format TypeScript (from luxar-viewer/)
```

`check-all` deliberately runs no tests: `test-all` is the single place they
execute. For one command covering everything *including* coverage, use
`hatch run check` and `pnpm run check:ci` (what CI runs) directly.

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
| UI Visual Design (authoritative) | `docs/guides/developer/UI_DESIGN_GUIDE.md` |
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
