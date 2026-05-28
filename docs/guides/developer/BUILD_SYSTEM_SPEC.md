# Build System Specification

This document provides comprehensive documentation for Luxar's build system, development setup, and Makefile commands.

## Overview

Luxar uses a Makefile-based build system designed to work on fresh Linux and macOS machines with minimal pre-installed tools. The system automatically detects the operating system and package manager, then installs required dependencies without requiring sudo (where possible).

### Design Goals

1. **Zero-friction setup**: Run `make setup-dev` on a fresh machine
2. **No sudo required**: Use nvm for Node.js, pipx for Python tools
3. **Cross-platform**: Support Linux (apt, dnf, yum) and macOS (brew)
4. **Graceful degradation**: Clear error messages with copy-paste solutions
5. **Idempotent**: Safe to run multiple times

## Prerequisites

### Minimal Requirements

Before running `make setup-dev`, you need:

| Tool | Required Version | Notes |
|------|-----------------|-------|
| Python | 3.10+ | Usually pre-installed on Linux/macOS |
| Git | Any | For cloning the repository |
| curl | Any | For downloading installers |

### System-Specific Prerequisites

**Ubuntu/Debian:**
```bash
# Usually pre-installed, but if missing:
sudo apt-get update
sudo apt-get install -y python3 python3-venv git curl

# Required for Hatch (pipx is needed on modern Ubuntu due to PEP 668)
sudo apt-get install -y pipx
pipx ensurepath
source ~/.bashrc  # or restart terminal
```

**Fedora/RHEL:**
```bash
sudo dnf install -y python3 python3-pip git curl pipx
pipx ensurepath
```

**macOS:**
```bash
# Python 3 comes with macOS or install via:
xcode-select --install  # Command line tools
# or
brew install python@3.11
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/royerlab/luxar.git
cd luxar

# Complete setup (auto-installs all dependencies)
make setup-dev

# Verify installation
make check-deps

# Start developing
make viewer      # Start viewer dev server
make test-all    # Run all tests
```

## Development Setup Process

### What `make setup-dev` Does

The setup process has 5 steps:

#### Step 1: Python Environment

1. Verifies Python 3.10+ is available
2. Checks for Hatch (Python environment manager)
3. If Hatch is missing:
   - Checks for pipx (required on modern Ubuntu/Debian)
   - Installs Hatch via `pipx install hatch`
   - Handles edge cases (broken symlinks, already installed)

#### Step 2: Node.js Environment

1. Sources nvm if already installed (`~/.nvm/nvm.sh`)
2. Checks Node.js version (requires 20.19+ for Vite 7.x)
3. If Node.js is missing or too old:
   - **macOS**: Uses Homebrew (`brew install node@22`)
   - **Linux**: Installs nvm, then `nvm install 22`
4. Installs pnpm globally via npm

#### Step 3: Python Virtual Environment

1. Creates Hatch environment (`hatch env create`)
2. Installs pre-commit hooks (`hatch run pre-commit install`)

#### Step 4: TypeScript Dependencies

1. Runs `pnpm install` in `packages/luxar-viewer/`
2. Downloads all npm packages (~300 packages)

#### Step 5: Optional WASM Support

1. Checks for Rust/wasm-pack
2. Displays instructions for `make install-rust` if not installed
3. WASM is optional - viewer works without it (uses TypeScript fallback)

### Rust/WASM Setup Details

The Rust/WASM toolchain enables high-performance WebAssembly computations in the viewer (e.g., nD filtering, distance calculations). It's optional but recommended for best performance - without it, the viewer falls back to TypeScript implementations.

**What `make install-rust` does:**
1. Installs Rust via rustup (if not present)
2. Loads the cargo environment automatically
3. Installs wasm-pack for WASM packaging

**Key Design:**
- All Rust-related make targets source `~/.cargo/env` automatically
- The `build-wasm.sh` script also sources cargo env at startup
- No manual `source ~/.cargo/env` is required after installation

**Build flow:**
```
make build-viewer
  ├─ Checks for wasm-pack
  ├─ If missing: runs make install-rust
  └─ Runs pnpm build
       └─ pnpm build:wasm (scripts/build-wasm.sh)
            ├─ Sources ~/.cargo/env
            ├─ Verifies wasm-pack is available
            └─ Runs wasm-pack build
```

### Native Launcher Setup Details

The native launcher backs `luxar export --native macos|linux-amd64|linux-arm64`, which produces double-clickable native bundles. The launcher is a small Go program (`packages/luxar-launcher/main.go`, ~150 lines) that opens the bundled viewer inside a system WebView and serves the bundled zarr over a local HTTP server.

**What `make install-go` does:**
1. macOS: installs Go via Homebrew (no sudo)
2. Linux: downloads the official Go tarball into `~/.local/go/` (no sudo); user adds `~/.local/go/bin` to PATH
3. Skips the install if a `go` binary is already on PATH

**What `make build-launchers` does:**
1. Locates `go` (PATH or `~/.local/go/bin/go`)
2. Builds the launcher with **`CGO_ENABLED=1`** because the WebView library links against system WebKit
3. On macOS: builds `darwin-arm64` + `darwin-amd64` then `lipo`-merges into `darwin-universal`. Fails loudly if amd64 build fails (no silent rename — universal binary must actually be universal)
4. On Linux: builds `linux-<host-arch>` (requires `libwebkit2gtk-4.1-dev` + `pkg-config`)
5. Drops binaries into `packages/luxar/src/luxar/cli/_launchers/`

**Critical constraint: CGO blocks pure cross-compilation.** Unlike Rust/WASM (where pure-Go cross-compile from any host worked previously), the launcher cannot be built for Linux from a macOS host or vice-versa without a CGO cross-toolchain (Zig, etc.). For full cross-platform release artifacts, build each OS on its own CI matrix runner.

**System library dependencies (end-user runtime):**
- macOS: `WebKit.framework` — system-provided, present on every Mac, no install needed
- Linux: `libwebkit2gtk-4.1` (or `4.0` on older distros) — present on every modern desktop Linux distribution; missing only on minimal/server installs

**Wheel packaging:** `_launchers/` and `_launcher_assets/` (icons) live inside the Python package, so they ride along into wheel builds automatically when present. Run `make build-launchers` before `hatch build` to populate the binaries; without it the wheel installs but `luxar export --native` raises `LauncherNotBuiltError` with a clear "run `make build-launchers`" hint.

**Build flow:**
```
make build-launchers
  ├─ Resolves go binary (PATH or ~/.local/go/bin)
  ├─ macOS:  GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 go build → darwin-arm64
  │          GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 go build → darwin-amd64
  │          lipo -create  → darwin-universal
  │          lipo -info    → verify (refuses to ship arm64-only as "universal")
  └─ Linux:  GOOS=linux GOARCH=$(uname -m) CGO_ENABLED=1 go build → linux-<arch>
```

**Runtime fallback:** end users can set `LUXAR_LAUNCHER_NO_WEBVIEW=1` to make the launcher open the system default browser instead of the embedded WebView. Useful for headless smoke tests and minimal Linux installs without `libwebkit2gtk`.

See `packages/luxar-launcher/README.md` for source-level details and `packages/luxar/src/luxar/cli/README.md` for the full bundle output structure.

### Environment Detection

The Makefile automatically detects:

```makefile
# OS Detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OS := macos
    PKG_MANAGER := brew
else ifeq ($(UNAME_S),Linux)
    OS := linux
    # Detect: apt (Debian/Ubuntu), dnf (Fedora), yum (RHEL/CentOS)
endif

# Node.js version requirements
MIN_NODE_MAJOR := 20
MIN_NODE_MINOR := 19
```

## Makefile Commands Reference

### Setup & Installation

| Command | Description |
|---------|-------------|
| `make setup-dev` | Complete development environment setup |
| `make check-deps` | Check all dependencies and their versions |
| `make install-node` | Install/upgrade Node.js via nvm (Linux) or brew (macOS) |
| `make install-pnpm` | Install pnpm package manager |
| `make install-hatch` | Install Hatch via pipx |
| `make install-rust` | Install Rust toolchain and wasm-pack |
| `make clean-setup` | Remove ALL dev tools to simulate fresh machine |

### Quality & Testing

| Command | Description |
|---------|-------------|
| `make check-all` | Run all quality checks (Python + TypeScript) |
| `make check-typescript` | Run all TypeScript checks (typecheck, lint, test) |
| `make test-all` | Run all tests (Python + Rust + TypeScript) |
| `make test-python` | Run Python tests only |
| `make test-cov-python` | Run Python tests with coverage |
| `make test-e2e` | Run Playwright E2E tests |
| `make lint-python` | Run ruff linting on Python |
| `make lint-typescript` | Run ESLint on TypeScript |
| `make type-check-python` | Run mypy type checking |
| `make type-check-typescript` | Run TypeScript type checking |
| `make security` | Run bandit security scan |
| `make format-python` | Format Python code |
| `make format-typescript` | Format TypeScript code |
| `make format-all` | Format all code (Python + TypeScript) |

### Viewer Development

| Command | Description |
|---------|-------------|
| `make viewer` | Start viewer dev server (port 5173) |
| `make build-viewer` | Build viewer for production (requires Rust) |
| `make rebuild-viewer` | Clean rebuild of viewer |
| `make test-viewer` | Run TypeScript unit tests |
| `make test-cov-typescript` | Run TypeScript tests with coverage |

### WASM Development

| Command | Description |
|---------|-------------|
| `make build-wasm` | Build WASM module |
| `make test-wasm` | Run Rust unit tests |
| `make benchmark-wasm` | Run WASM vs TypeScript performance benchmarks |
| `make clean-wasm` | Clean WASM build artifacts |

### Data & Demos

| Command | Description |
|---------|-------------|
| `make demo` | Generate demo dataset (100k points) |
| `luxar demo` | Generate demo + serve + open browser (all-in-one) |
| `make run-examples` | Generate all example datasets |
| `make serve-examples` | Serve datasets directory |
| `make serve-dataset` | Serve a specific dataset |

### Documentation

| Command | Description |
|---------|-------------|
| `make build-docs` | Build Sphinx documentation |
| `make serve-docs` | Serve documentation locally |
| `make clean-docs` | Clean documentation artifacts |
| `make check-docs` | Check documentation quality |

### Utilities

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make clean-all` | Clean all artifacts (Python, TypeScript, WASM, CUDA, datasets) |
| `make clean-examples` | Clean generated example datasets |
| `make stats` | Generate project statistics report |
| `make shell` | Enter Hatch development shell |
| `make show-env` | Show Hatch environments |
| `make prune-env` | Remove unused Hatch environments |

## Dependency Management

### Node.js via nvm (Linux)

nvm (Node Version Manager) is used on Linux to avoid requiring sudo:

```bash
# nvm installation location
~/.nvm/

# The Makefile sources nvm before Node.js commands:
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
```

**Important**: After installing nvm, either restart your terminal or run:
```bash
source ~/.bashrc  # or source ~/.nvm/nvm.sh
```

### Python Tools via pipx

Modern Ubuntu/Debian (22.04+) uses PEP 668 "externally managed environment" which blocks `pip install --user`. We use pipx instead:

```bash
# pipx installs Python CLI tools in isolated environments
pipx install hatch

# Tools are installed to:
~/.local/bin/              # Symlinks to executables
~/.local/share/pipx/venvs/ # Isolated virtual environments
```

### Hatch for Python Environments

Hatch manages Python virtual environments for the project:

```bash
# Environments are stored in:
~/.local/share/hatch/env/virtual/luxar*/

# Common commands:
hatch shell         # Activate environment
hatch run test      # Run tests in environment
hatch env prune     # Clean unused environments
```

### pnpm for TypeScript

pnpm is used for TypeScript package management:

```bash
# Install globally via npm
npm install -g pnpm

# pnpm stores packages in:
~/.local/share/pnpm/store/

# Project dependencies in:
packages/luxar-viewer/node_modules/
```

## Troubleshooting

### Common Issues

#### "pipx not found"

Modern Ubuntu/Debian requires pipx for Python CLI tools:

```bash
# Ubuntu/Debian
sudo apt-get install -y pipx
pipx ensurepath
source ~/.bashrc

# Then retry
make setup-dev
```

#### "Node.js not found" or "version too old"

```bash
# The Makefile will auto-install via nvm, but if you need to do it manually:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.nvm/nvm.sh
nvm install 22
nvm use 22
```

#### "Hatch not found" after installation

pipx may need its path added:

```bash
pipx ensurepath
source ~/.bashrc  # or restart terminal
```

If Hatch is installed but symlink is broken:

```bash
pipx reinstall hatch
```

#### "wasm-pack not found"

WASM support is optional but enables high-performance WebAssembly computations. Install if needed:

```bash
make install-rust
```

This command:
1. Installs Rust via rustup (if not present)
2. Installs wasm-pack (if not present)
3. Sources cargo environment automatically

The Makefile commands (`make build-wasm`, `make build-viewer`, etc.) automatically source the cargo environment, so you don't need to run `source ~/.cargo/env` manually.

#### nvm not available in make commands

The Makefile sources nvm automatically, but ensure it's installed:

```bash
# Check nvm installation
ls -la ~/.nvm/nvm.sh

# If missing, install:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
```

### Verifying Installation

```bash
# Check all dependencies
make check-deps

# Expected output:
# ✅ Python: Python 3.12.3
# ✅ pipx: 1.4.3
# ✅ Hatch: Hatch, version 1.16.2
# ✅ Node.js: v22.21.1 (via nvm)
# ✅ npm: 10.9.4
# ✅ pnpm: 11.4.0
# ⚪ Rust not installed (optional)
# ⚪ wasm-pack not installed (optional)
```

### Clean Slate Recovery

If something goes wrong, reset everything:

```bash
# Remove all dev tools (interactive, confirms before proceeding)
make clean-setup

# Then start fresh
make setup-dev
```

---

## HPC / Slurm Cluster Setup

HPC login nodes typically lack sudo, pipx, and GPU access. The Makefile handles these constraints automatically.

### Overview of HPC limitations and solutions

| Limitation | Solution |
|------------|----------|
| No sudo / no pipx | `install-hatch` tries `pip install --user`, then venv fallback |
| No global npm | `install-pnpm` tries global npm, then `npm install --prefix ~/.local` fallback |
| No GPU on login node | `make build-cuda SLURM=1` submits the build to a GPU node |
| Old system GCC (< 9) | `build_cuda_slurm.py` auto-detects a `gcc/` module >= 9 to load |
| CUDA modules vs PATH | `build_cuda_slurm.py` auto-selects the matching `cuda/` module |

### Step-by-step HPC first-time setup

```bash
# 1. Clone and enter the project
git clone <repo> luxar && cd luxar

# 2. Bootstrap dev tools (auto-detects HPC, uses venv fallback for hatch)
make setup-dev

# 3. Add ~/.local/bin to PATH (required on most HPC systems)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# 4. Verify everything works
python scripts/test_hpc_setup.py   # 8 smoke tests

# 5. Build the CUDA extension on a GPU node
make build-cuda SLURM=1                            # Auto-detect everything
make build-cuda SLURM=1 SLURM_PARTITION=gpu        # Choose partition
make build-cuda SLURM=1 SLURM_PARTITION=gpu \
    SLURM_ACCOUNT=myaccount SLURM_TIME=02:00:00    # Full options

# 6. Monitor the Slurm job
tail -f build-cuda-logs/build_<JOB_ID>.out

# 7. After the job completes, verify
make test-cuda
```

### How `make build-cuda SLURM=1` works

The submission script is `scripts/build_cuda_slurm.py`. Before submitting, it:

1. **Detects PyTorch CUDA version** — queries `torch.version.cuda` from the hatch env
2. **Finds matching CUDA module** — runs `module spider cuda`, picks the highest `cuda/X.Y.z` matching the torch CUDA major.minor
3. **Finds GCC >= 9 module** — runs `module spider gcc`, picks the highest `gcc/X.Y` with X >= 9 (required by PyTorch 2.x; system GCC on RHEL 8 is 8.5.0)
4. **Captures VIRTUAL_ENV** — the hatch env path must be reachable from the compute node (shared filesystem)
5. **Generates sbatch script** at `build-cuda-logs/build_cuda_job.sh`
6. **Submits with sbatch** and prints monitoring commands

The generated sbatch script:
- Loads CUDA module, then GCC module (important order)
- Saves `CUDA_LIB_DIR` before `source activate` so `LD_LIBRARY_PATH` survives venv activation
- Runs `make build-cuda SLURM=0` (explicit `SLURM=0` prevents Slurm env var recursion)
- Merges stderr into stdout (`2>&1`) so compiler errors appear in the main `.out` log
- Checks the build exit code and prints actionable diagnostics on failure

### Configurable Slurm variables (Makefile)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SLURM_PARTITION` | `gpu` | Slurm partition for GPU jobs |
| `SLURM_ACCOUNT` | (none) | Slurm account/project |
| `SLURM_QOS` | (none) | Quality of service |
| `SLURM_TIME` | `01:00:00` | Wall-time limit |
| `CUDA_MODULE` | `auto` | CUDA module to load (auto = detect from torch) |

### Diagnosing build failures

All compiler output is in `build-cuda-logs/build_<JOB_ID>.out`. Common issues:

| Error | Cause | Fix |
|-------|-------|-----|
| `GCC version too old` | System GCC < 9 | Ensure `module spider gcc` shows gcc >= 9 on compute nodes |
| `torch.cuda.is_available() False` | CUDA/torch version mismatch | Check `torch.version.cuda` vs loaded module |
| `.so not found after build` | Build succeeded but path wrong | Run `make test-cuda` which also searches for the .so |
| `sbatch: Invalid job id` | Job already finished | Check the `.out` file — it may have succeeded |

### Multi-architecture CUDA build

The CUDA extension compiles for **all common GPU architectures** (sm_75 through
sm_120, covering Turing → Blackwell) plus PTX for forward compatibility with
future GPUs. The build script queries `nvcc --list-gpu-arch` and automatically
drops any archs the current toolkit cannot target (e.g. sm_100 / sm_120 require
CUDA 12.8+), so the default works on both older and newest toolchains. This
ensures the `.so` works on any GPU in a heterogeneous cluster (e.g. A6000 sm_86
+ H100/H200 sm_90 + B100/B200 sm_100).

Override with `CUDA_ARCHS` environment variable:
```bash
make build-cuda SLURM=1                          # Default: sm_75,80,86,89,90,100,120 + PTX
CUDA_ARCHS="86;90" make build-cuda SLURM=1       # Only sm_86 and sm_90 (faster compile)
CUDA_ARCHS="100;120" make build-cuda SLURM=1     # Blackwell-only (requires CUDA 12.8+)
```

### Build metadata (`cuda_build_info.json`)

After compilation, `build.py` writes `cuda_build_info.json` alongside the `.so` recording:
- Which modules were loaded at build time (cuda/, gcc/)
- PyTorch and CUDA versions
- Python version

At job submission time, `env_capture.py` reads this file and automatically
adds missing modules to the sbatch preamble — so users don't need to remember
to `module load gcc/14.2` before submitting fit jobs.

### Batch fitting on Slurm

After building the CUDA extension, use `luxar gsplat batch plan` to plan and
submit large-scale fitting jobs:

```bash
# Plan (dry-run by default)
hatch run luxar gsplat batch plan data.zarr.zip output/ -p gpu

# Override axis labels for non-standard zarr layouts
hatch run luxar gsplat batch plan data.zarr.zip output/ -p gpu \
    --axes time,camera,channel,z,y,x

# Submit with sequential task packing (default)
hatch run luxar gsplat batch plan data.zarr.zip output/ -p gpu --submit

# Parallel task packing (multiple fits sharing one GPU)
hatch run luxar gsplat batch plan data.zarr.zip output/ -p gpu --parallel --submit

# Manual control
hatch run luxar gsplat batch plan data.zarr.zip output/ -p gpu \
    --tile-size 256 --tasks-per-job 3 --preset draft --submit
```

**Key CLI options for batch plan:**

| Option | Purpose |
|--------|---------|
| `--axes` | Comma-separated axis labels (e.g. `time,z,y,x`) — overrides auto-detection |
| `--tile-size` | Manual tile size in voxels — skips GPU profile requirement |
| `--tasks-per-job` | Number of tasks per Slurm job (auto-calculated from GPU capacity) |
| `--parallel` / `--sequential` | Run packed tasks concurrently or one-by-one (default: sequential) |
| `--preset` | Fitting preset: `draft` (500 iter), `standard` (3000), `hifi` (6000) |
| `--gpu` | GPU profile name when auto-detect unavailable (login node) |

**Auto-tiling**: compares total spatial voxels against the GPU's benchmarked
capacity. Small volumes (e.g. 108×1352×532 = 78M voxels on H100 max 453M) get
no tiling at all. Only volumes exceeding GPU capacity are tiled.

**Task packing**: when volumes are small relative to GPU capacity, multiple
fitting tasks are grouped into each Slurm job to reduce scheduling overhead.

### Running smoke tests

`scripts/test_hpc_setup.py` verifies the HPC environment:

```bash
python scripts/test_hpc_setup.py
```

Tests: Python 3.10+ available, hatch installed and functional, hatch env show works, hatch uses Python >= 3.10, pnpm installed and functional, `~/.local/bin` in PATH, npm `--prefix` fallback works, hatch venv uses Python >= 3.10.

`scripts/test_batch_plan_fixes.py` verifies the batch planning fixes:

```bash
hatch run python scripts/test_batch_plan_fixes.py
```

Tests: zarr.zip support, custom axes parsing, axes override validation, array selection consistency, auto-tile logic, cull_retention defaults, 6D slicing, manifest serialization, LD_LIBRARY_PATH handling.

---

## Environment Variables

The build system uses these environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NVM_DIR` | nvm installation directory | `~/.nvm` |
| `DATASET` | Dataset path for `make serve-dataset` | `datasets/demos/demo.zarr` |
| `PORT` | Server port for data serving | `8000` |
| `SLURM` | Set to `1` to build CUDA on a GPU node via Slurm | `0` |
| `SLURM_PARTITION` | Slurm partition for GPU builds | `gpu` |
| `SLURM_ACCOUNT` | Slurm account/project for GPU builds | (none) |
| `SLURM_QOS` | Slurm QOS for GPU builds | (none) |
| `SLURM_TIME` | Wall-time limit for Slurm build jobs | `01:00:00` |
| `CUDA_MODULE` | CUDA module to load on compute node (`auto` = detect) | `auto` |
| `CUDA_ARCHS` | CUDA architectures to compile for (e.g. `86;90`) | all common (75-120) |

## CI/CD Integration

For automated environments (GitHub Actions, etc.):

```yaml
# Example GitHub Actions setup
- name: Install pipx
  run: |
    sudo apt-get update
    sudo apt-get install -y pipx
    pipx ensurepath
    echo "$HOME/.local/bin" >> $GITHUB_PATH

- name: Setup development environment
  run: make setup-dev

- name: Run checks
  run: |
    make check-all
    make test-all
```

**Note**: In CI environments, the shell doesn't reload between steps, so we explicitly add `~/.local/bin` to `$GITHUB_PATH` to ensure pipx-installed tools are available.

## Architecture Notes

### Why nvm Instead of System Node.js?

1. **No sudo required**: nvm installs to `~/.nvm`
2. **Version control**: Easy to pin Node.js versions
3. **Isolation**: Doesn't affect system-wide Node.js

### Why pipx Instead of pip?

1. **PEP 668 compliance**: Modern Ubuntu blocks `pip install --user`
2. **Isolation**: Each tool gets its own virtual environment
3. **Clean upgrades**: No dependency conflicts between tools

### Why Hatch Instead of venv/poetry?

1. **Environment management**: Multiple environments (dev, test, docs)
2. **Script running**: `hatch run` without activation
3. **Build system**: Standards-compliant package building
4. **Configuration**: All in `pyproject.toml`

## Version Requirements

| Tool | Minimum Version | Reason |
|------|----------------|--------|
| Python | 3.10 | Type hints, dataclasses, match statements |
| Node.js | 20.19 | Vite 7.x requirements |
| Rust | stable | WASM compilation |
| wasm-pack | latest | WASM packaging |

## Related Documentation

- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - Contributing guidelines
- [CLAUDE.md](../../../CLAUDE.md) - AI assistant instructions
- [TESTING_GUIDELINES.md](./TESTING_GUIDELINES.md) - Testing best practices
- [PLAYWRIGHT_GUIDE.md](./PLAYWRIGHT_GUIDE.md) - E2E testing guide
