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
| Python | 3.12+ | Usually pre-installed on Linux/macOS |
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
brew install python@3.12
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

1. Verifies a `python3` binary is present (any version — the version itself
   is not checked here)
2. Checks for Hatch (Python environment manager)
3. If Hatch is missing:
   - Checks for pipx (required on modern Ubuntu/Debian)
   - Installs Hatch via `pipx install hatch`
   - Without pipx, falls back to `pip install --user` / a venv — this
     fallback is where a Python 3.12+ interpreter is scanned for
     (`python3.14` → `python3.13` → `python3.12` → `python3`)
   - Handles edge cases (broken symlinks, already installed)

#### Step 2: Node.js Environment

1. Sources nvm if already installed (`~/.nvm/nvm.sh`)
2. Checks Node.js version (requires 22.22+ — jsdom 30's declared floor; Vite 8.x alone needs only 20.19)
3. If Node.js is missing or too old:
   - **macOS**: Uses Homebrew (`brew install node@22`)
   - **Linux**: Installs nvm, then `nvm install 22`
4. Installs pnpm globally via npm

#### Step 3: Python Virtual Environment

1. Creates Hatch environment (`hatch env create`)
2. Installs pre-commit hooks (`hatch run pre-commit install`) — hooks are defined in `.pre-commit-config.yaml` (ruff lint + format, bandit, mypy)

#### Step 4: TypeScript Dependencies

1. Runs `pnpm install` in `packages/luxar-viewer/`
2. Downloads all npm packages (~300 packages)

#### Step 5: Optional Accelerators (WASM + CUDA)

1. Checks for Rust/wasm-pack
2. Displays instructions for `make install-rust` if not installed
3. WASM is optional - viewer works without it (uses TypeScript fallback)
4. Reports CUDA status: whether the CUDA toolkit (`nvcc`) is installed and
   whether PyTorch CUDA is available (pointing at `make check-cuda-deps` /
   `make build-cuda` as next steps)

### Rust/WASM Setup Details

The Rust/WASM toolchain enables high-performance WebAssembly computations in the viewer (e.g., nD filtering, distance calculations). It's optional but recommended for best performance - without it, the viewer falls back to TypeScript implementations.

**What `make install-rust` does:**
1. Installs Rust via rustup (if not present)
2. Loads the cargo environment automatically
3. Installs the pinned wasm-pack (`WASM_PACK_VERSION`) for WASM packaging

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

### Editable Installs vs. Release Wheels

A fresh clone or git worktree does **not** contain
`packages/luxar-viewer/dist/`; it is a gitignored production build artifact.
Hatch editable installs (`hatch run ...`, `make install-dev`) therefore skip the
wheel target's viewer `force-include` through `hatch_build.py`. Development uses
the viewer from the source tree, so no production viewer build is required just
to run Python tooling or generate viewer test fixtures.

Standard wheel builds remain strict. `hatch build -t wheel` requires
`packages/luxar-viewer/dist/index.html` and fails with an actionable
`make build-viewer` message when it is absent. This prevents publishing a wheel
without its bundled viewer while keeping clean-worktree development usable.
The source distribution remains provenance-only: it includes the build hook but
not the viewer artifact, so end users should install the published wheel.

### Native Launcher Setup Details

The native launcher backs `luxar export --native macos|linux-amd64|linux-arm64`, which produces double-clickable native bundles. The launcher is a small Go program (`packages/luxar-launcher/main.go`) that opens the bundled viewer inside a system WebView and serves the bundled zarr over a local HTTP server.

**What `make install-go` does:**
1. macOS: installs Go via Homebrew (no sudo)
2. Linux: downloads the official Go tarball into `~/.local/go/` (no sudo); user adds `~/.local/go/bin` to PATH
3. Skips the install if a `go` binary is already on PATH

**What `make build-launchers` does:**
1. Locates `go` (PATH or `~/.local/go/bin/go`)
2. Builds the launcher with **`CGO_ENABLED=1`** because the WebView library links against system WebKit
3. On macOS: builds `darwin-arm64` + `darwin-amd64` then `lipo`-merges into `darwin-universal`. Fails loudly if amd64 build fails (no silent rename — universal binary must actually be universal)
4. On Linux: builds `linux-<host-arch>` (requires `libwebkit2gtk-4.0-dev` + `pkg-config` — the pinned `webview_go` declares `#cgo pkg-config: gtk+-3.0 webkit2gtk-4.0`, which is why CI builds the launcher on ubuntu-22.04; 24.04 ships only the 4.1 package)
5. Drops binaries into `packages/luxar/src/luxar/cli/_launchers/`

**Critical constraint: CGO blocks pure cross-compilation.** Unlike Rust/WASM (where pure-Go cross-compile from any host worked previously), the launcher cannot be built for Linux from a macOS host or vice-versa without a CGO cross-toolchain (Zig, etc.). For full cross-platform release artifacts, build each OS on its own CI matrix runner.

**System library dependencies (end-user runtime):**
- macOS: `WebKit.framework` — system-provided, present on every Mac, no install needed
- Linux: SONAME `libwebkit2gtk-4.0.so.37`, packaged on Debian/Ubuntu as `libwebkit2gtk-4.0-37` — the runtime counterpart of the `webkit2gtk-4.0` pkg-config module the pinned `webview_go` links. Missing on minimal/server installs **and on distros that ship only 4.1** (verified: Ubuntu 24.04 offers only `libwebkit2gtk-4.1-0`), where the prebuilt launcher cannot start at all — see the launcher README

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

**Runtime fallback:** end users can set `LUXAR_LAUNCHER_NO_WEBVIEW=1` to make the launcher open the system default browser instead of the embedded WebView. Useful for headless smoke tests. It does *not* let the prebuilt Linux binary run without `libwebkit2gtk`: WebKit is linked at build time (cgo), so the loader aborts before `main()` on a system missing the `webkit2gtk-4.0` runtime.

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
MIN_NODE_MAJOR := 22
MIN_NODE_MINOR := 22
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
| `make install-go` | Install Go toolchain for native launcher builds (no sudo) |
| `make build-launchers` | Build native launchers for the host platform (requires Go + CGO) |
| `make install-viewer-deps` | Install viewer dependencies (node_modules) |
| `make install-dev` | Install Luxar Python package in editable mode |
| `make install-demo-deps` | Install the demo extras (demos + gsplats + io), then report via `luxar demo deps` |
| `make enable-pre-commit` | Enable and activate pre-commit hooks |
| `make check-wasm-deps` | Check WASM dev dependencies (Rust, wasm-pack) |
| `make clean-setup` | Remove ALL dev tools to simulate fresh machine |

### Quality & Testing

| Command | Description |
|---------|-------------|
| `make check-all` | Run all quality checks (Python, TypeScript, Rust, Go) — **static only, runs no tests** (`hatch run check-static` + `pnpm check:static`; `make test-all` is the single place the suites execute). **Not read-only** — `check-static` begins with `format`, so this rewrites `packages/luxar/src` and `scripts`. Use the scoped `lint-*` / `type-check-*` / `check-typescript` / `check-rust` targets for a read-only verdict. |
| `make check-typescript` | Run all TypeScript checks (typecheck, lint, test) |
| `make check-rust` | Run Rust type/lint checks (cargo check + clippy) |
| `make check-knip` | **Report only, non-gating** — full knip (unused files/exports/types + `@internal` tag hints). The enforced subset (`files,dependencies`) runs inside `make check-all`; the full run has a standing backlog, so it never fails the build. |
| `make test-fast` | Inner loop, **not a gate**: Python under xdist with `-m 'not slow'` plus the TypeScript units, no coverage. Scope it with `PYTEST_ARGS` — which *replaces* the default paths (`make test-fast PYTEST_ARGS='packages/luxar/src/luxar/encoding'`); options-only args fall through to pytest's `testpaths`. `LUXAR_PYTEST_JOBS` sets the worker count. |
| `make test-all` | Run all tests (Python + Rust/WASM + TypeScript, plus Go launcher tests when that toolchain is available). CUDA extension tests are part of the Python suite — they live under pytest's `testpaths` and skip themselves without a GPU. |
| `make test-cov-all` | Run all tests with coverage (Python, minus `-m slow`, + TypeScript) |
| `make test-python` | Run Python tests only |
| `make test-cov-python` | Run Python tests with coverage |
| `make test-fixtures` | Generate test fixtures for TypeScript tests |
| `make test-viewer-fixtures` | Generate fixtures + run TypeScript tests |
| `make test-e2e` | Run the full Playwright E2E suite (~17 min) |
| `make test-e2e-smoke` | Run the E2E smoke subset (the interaction-focused specs CI would run) |
| `make test-perf-e2e` | Run the opt-in Playwright performance suite |
| `make lint-python` | Run ruff linting on Python |
| `make check-complexity` | Ratchet cyclomatic complexity (ruff C901) against `scripts/complexity_baseline.json` |
| `make lint-typescript` | Run ESLint on TypeScript |
| `make type-check-python` | Run mypy type checking |
| `make type-check-typescript` | Run TypeScript type checking |
| `make security` | Run bandit security scan |
| `make format-python` | Format Python code |
| `make format-typescript` | Format TypeScript code |
| `make format-rust` | Format Rust code with cargo fmt |
| `make format-go` | Format the Go launcher with gofmt |
| `make format-cuda` | Format CUDA/C++ code with clang-format |
| `make format-all` | Format all code (Python, TypeScript, Rust, Go, CUDA) |
| `make run-pre-commit` | Run pre-commit hooks on all files |
| `make gen-contract` | Regenerate Python + TS format-contract projections from contract.yaml |
| `make benchmark-metal` | Run Metal (MPS) performance benchmarks (M-series only) |
| `make benchmark-metal-stress` | Run Metal RSS leak-check |

### Viewer Development

| Command | Description |
|---------|-------------|
| `make viewer` | Start viewer dev server (port 5173) |
| `make build-viewer` | Build viewer for production (auto-installs Rust/wasm-pack via `install-rust` if missing) |
| `make build-viewer-lib` | Build + verify the viewer's npm **library** bundle (`pnpm ci:release`) — the artifact `publish-npm.yml` ships, distinct from the web app bundled into the wheel |
| `make rebuild-viewer` | Clean rebuild of the viewer bundle — clears the JS/TS artifacts (`dist/`, the vite dep-optimizer cache, tsbuildinfo); leaves `public/wasm/` and the cargo target dir alone, so the Rust step is a cache hit unless its sources changed. Chain `make clean-wasm rebuild-viewer` for everything from source |
| `make test-viewer` | Run TypeScript unit tests |
| `make test-cov-typescript` | Run TypeScript tests with coverage |

### WASM Development

| Command | Description |
|---------|-------------|
| `make build-wasm` | Build WASM module |
| `make test-wasm` | Run Rust unit tests |
| `make benchmark-wasm` | Run WASM vs TypeScript performance benchmarks |
| `make clean-wasm` | Clean WASM build artifacts |

### CUDA & GPU Extensions

| Command | Description |
|---------|-------------|
| `make setup-cuda` | Install CUDA deps + build extension |
| `make check-cuda-deps` | Check CUDA dependencies (nvcc, PyTorch CUDA, etc.) |
| `make build-cuda` | Build CUDA splatting extension (`SLURM=1` to build on a GPU node) |
| `make build-cuda-slurm` | Submit CUDA extension build as a Slurm job (alias for `make build-cuda SLURM=1`) |
| `make test-cuda` | Run CUDA tests |
| `make benchmark-cuda` | Run CUDA performance benchmarks |
| `make clean-cuda` | Clean CUDA build artifacts |
| `make build-nlm-cuda` | Build the NLM CUDA denoising extension |
| `make test-nlm-cuda` | Run NLM CUDA extension tests |
| `make clean-nlm-cuda` | Clean NLM CUDA build artifacts |

### Data & Demos

| Command | Description |
|---------|-------------|
| `make demo` | Generate demo dataset (100k points) |
| `luxar demo` | List the bundled demos; `luxar demo run <key>` generates + serves one |
| `make run-examples` | Generate all example datasets |
| `make run-demos` | Generate ALL demo datasets (output to `datasets/demos/`) |
| `make serve-examples` | Serve datasets directory |
| `make serve-dataset` | Serve a specific dataset |
| `make generate-readme-demos` | Generate only the demo datasets needed for README screenshots |
| `make generate-readme-images` | Generate README screenshots using Playwright |

### Documentation

| Command | Description |
|---------|-------------|
| `make build-docs` | Build documentation: first runs `generate-doc-images` (generates demo datasets + Playwright screenshots — needs pnpm and Playwright browsers), then Sphinx, then `build-typedoc` |
| `make generate-doc-images` | Generate documentation screenshots using Playwright (depends on `generate-readme-demos`) |
| `make serve-docs` | Serve documentation locally |
| `make clean-docs` | Clean documentation artifacts |
| `make check-docs` | Run the PR documentation gate locally: pnpm-override guard, completeness ratchet, TypeDoc warning ratchet, and the warning-fatal Sphinx build (mirrors the required `docs-quality` CI job) |
| `make check-docs-verbose` | Same gate with verbose completeness output |
| `make check-docs-external-links` | Opt-in external HTTP link audit (`sphinx-build -b linkcheck`); deliberately not a required CI gate |
| `make check-demo-links` | Opt-in demo click-through destination audit; reports request failures and human-only checks without failing the command |
| `make check-zenodo-live` | Opt-in live Zenodo manifest-pin audit using the system Python; requires `ZENODO_TOKEN` and is deliberately not a required CI gate |
| `make check-cold-fetch` | Opt-in pre-removal gate that downloads hosted demo datasets into a throwaway cache with in-repo payloads hidden, then verifies their hosted SHA-256 pins |
| `make check-external-references` | Run all network-backed reference audits and emit one PASS/NOTICE/WARNING/ERROR report; always non-gating |
| `make build-typedoc` | Generate TypeScript API documentation with TypeDoc |

### Utilities

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make clean-all` | Clean all artifacts: Python, TypeScript, WASM, CUDA, launcher binaries (`clean-launchers`), generated datasets (`clean-examples`) and the `~/.cache/luxar` user cache (`clean-cache`, which keeps hand-placed demo inputs — see its row) |
| `make clean-examples` | Clean generated example datasets |
| `make clean-python` | Clean Python build artifacts and caches |
| `make clean-viewer` | Clean viewer build artifacts (node_modules, dist, `.vite`, coverage, playwright-report, test-results) |
| `make clean-launchers` | Clean native launcher binaries |
| `make clean-cache` | Clear the Luxar user cache (`~/.cache/luxar`), except the hand-placed demo inputs listed in `luxar.demos.registry.PROTECTED_INPUT_DIRS` (`milky_way_gaia_3m/` — a CC BY-NC catalog with no download path, so it is kept) |
| `make stats` | Generate project statistics report |
| `make stats-fast` | Generate project statistics without running tests (file counts only) |
| `make shell` | Enter Hatch development shell |
| `make show-env` | Show Hatch environments |
| `make prune-env` | Remove ALL Hatch environments |

### Release & Publishing

| Command | Description |
|---------|-------------|
| `make build` | Build wheel + sdist (builds the viewer first so it is bundled) |
| `make changelog-draft` | Preview the `changelog.d/` fold into `CHANGELOG.md`; changes nothing |
| `make changelog` | Fold `changelog.d/*.md` fragments into `CHANGELOG.md` and delete them (`MONTH="August 2026"` pins the heading) |
| `make set-version` | Set release version in code (`DATE=YYYY.MM.DD`, default today) |
| `make changelog-release-draft` | Preview cutting `## [Unreleased]` into the current version; changes nothing |
| `make changelog-release` | Cut `## [Unreleased]` into the current version after `make changelog` → `make set-version` |
| `make release-check` | Dry-run release: run ALL preflight checks, tag/push nothing |
| `make release` | Cut release: validate main + CI green, tag `v<version>`, push (triggers PyPI publish) |
| `make publish` / `make publish-test` | Disabled — use `make release` (tag-triggered OIDC publish via CI) |

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
hatch env remove <name>  # Remove one environment
```

Viewer fixture generation uses a separate `fixtures` environment so ordinary
`pnpm test` and `make test-fixtures` do not build the CUDA-heavy development
environment. Its first use creates roughly 1.2 GB alongside any existing
`default` environment; `hatch env remove fixtures` reclaims that space without
removing the default environment.

**Which Python does `hatch run` use?** The `default` environment declares no
`python`, so Hatch builds it with whatever interpreter **Hatch itself** runs
under — not necessarily one the project claims to support. Check before trusting
a local pass:

```bash
hatch run python -V     # the interpreter your tests actually used
```

The `test` matrix carries one leg per version the wheel's classifiers advertise —
3.12 (the floor, `requires-python = ">=3.12"`, which is what zarr 3.2+ requires),
3.13, and 3.14 (the newest). All three matter: `>=3.12` has no ceiling,
`install-hatch` prefers the newest interpreter it can find, and a developer's
`hatch env` therefore usually runs something newer than the floor. Run any of them
explicitly:

```bash
hatch run test.py3.12:cov   # the floor, and the required CI context
hatch run test.py3.13:cov
hatch run test.py3.14:cov   # the newest supported
```

CI runs each version it tests as its own parallel job and asserts the interpreter
matches the matrix leg, so a mismatch fails loudly rather than silently testing
one version three times (see issue #839). Which versions that is depends on the
event — a pull request runs the floor alone; see "Which Python versions CI runs"
below.

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
2. Installs the pinned wasm-pack (`WASM_PACK_VERSION` in the Makefile), replacing the
   copy in cargo's install root — then re-probes PATH and fails if some other copy
   (Homebrew, a distro package) still wins there
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
python scripts/check_hpc_setup.py   # 8 smoke tests

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

**Caveat with `SLURM=1`**: `CUDA_ARCHS` is not baked into the generated
sbatch script (`scripts/build_cuda_slurm.py` never references it). It only
reaches the compute node through sbatch's default environment propagation
(`--export=ALL`), so it must be set in the shell that runs the `make
build-cuda SLURM=1` submission.

### Build metadata (`cuda_build_info.json`)

After compilation, `build.py` writes `cuda_build_info.json` alongside the `.so` recording:
- Which modules were loaded at build time (cuda/, gcc/)
- PyTorch and CUDA versions
- Python version

At job submission time, `env_capture.py` reads this file and automatically
adds missing modules to the sbatch preamble — so users don't need to remember
to `module load gcc/14.2` before submitting fit jobs.

### Batch fitting on Slurm

After building the CUDA extension, use `luxar gsplat batch-fit submit` to plan and
submit large-scale fitting jobs. It submits by default; pass `--dry-run` to plan
without submitting:

```bash
# Plan without submitting
hatch run luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --dry-run

# Override axis labels for non-standard zarr layouts
hatch run luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --axes time,camera,channel,z,y,x

# Submit with sequential task packing (default)
hatch run luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu

# Parallel task packing (multiple fits sharing one GPU)
hatch run luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu --parallel

# Manual control
hatch run luxar gsplat batch-fit submit data.zarr.zip output/ -p gpu \
    --tile-size 256 --tasks-per-job 3 --preset draft
```

**Key CLI options for `batch-fit submit`:**

| Option | Purpose |
|--------|---------|
| `--axes` | Comma-separated axis labels (e.g. `time,z,y,x`) — overrides auto-detection |
| `--tile-size` | Manual tile size in voxels — skips GPU profile requirement |
| `--tasks-per-job` | Number of tasks per Slurm job (auto-calculated from GPU capacity) |
| `--parallel` / `--sequential` | Run packed tasks concurrently or one-by-one (default: sequential) |
| `--preset` | Fitting preset: `draft` (2000 iter), `standard` (5000), `hifi` (10000), `ultra` (20000), `n2s` (= ultra; canonical Noise2Self protocol name) |
| `--gpus-per-task` | GPUs to request per task, emitted as `#SBATCH --gpus-per-task` (default 1). A COUNT — unlike `batch-fit run --gpus`, which SELECTS local devices |
| `--gpu` | GPU profile name when auto-detect unavailable (login node) |

**Auto-tiling**: compares total spatial voxels against the GPU's benchmarked
capacity. Small volumes (e.g. 108×1352×532 = 78M voxels on H100 max 453M) get
no tiling at all. Only volumes exceeding GPU capacity are tiled.

**Task packing**: when volumes are small relative to GPU capacity, multiple
fitting tasks are grouped into each Slurm job to reduce scheduling overhead.

### Running smoke tests

`scripts/check_hpc_setup.py` verifies the HPC environment:

```bash
python scripts/check_hpc_setup.py
```

Tests: Python 3.12+ available, hatch installed and functional, hatch env show works, hatch uses Python >= 3.12, pnpm installed and functional, `~/.local/bin` in PATH, npm `--prefix` fallback works, hatch venv uses Python >= 3.12.

The batch-planning regression tests (zarr.zip support, custom axes parsing,
axes override validation, array selection consistency, auto-tile logic,
cull_retention defaults, 6D slicing, manifest serialization, LD_LIBRARY_PATH
handling) live in the main pytest suite as
`gsplats/tests/test_batch.py::TestBatchPlanRegression` and run with the rest of
`hatch run test`.

---

## Environment Variables

The build system uses these environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NVM_DIR` | nvm installation directory | `~/.nvm` |
| `DATASET` | Dataset path for `make serve-dataset` | `datasets/demos/lorenz.luxar.zarr` |
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

### Domain-scoped CI: a suite runs only when its domain changed

The `changes` job classifies a pull request's diff into four language domains
and each job runs its expensive steps only for the domain(s) it covers:

| Domain | Set by | Gates |
|--------|--------|-------|
| `dom_py` | `*.py`, `Makefile`, `pyproject.toml`, `*.pyx/*.pxd`, CUDA `*.cu/*.cuh`, plus cross-language gate inputs listed below | `python-tests`, `wheel-viewer` |
| `dom_ts` | anything under `packages/luxar-viewer/`, root `tsconfig*.json`, `vitest*.{ts,js,mjs}`, plus gallery-selection inputs listed below | `typescript-tests`, `release-readiness`, `wheel-viewer` |
| `dom_rust` | `*.rs`, `Cargo.toml/lock` | `typescript-tests`, `release-readiness`, `wheel-viewer` |
| `dom_go` | `*.go`, `go.mod/sum`, `cli/_launchers/` | `go-launcher` |

The mapping is a hierarchy, not a partition. `typescript-tests` builds the WASM
module and runs `cargo test`, so Rust changes select it too; `wheel-viewer`
bundles the built viewer into the wheel, so any of Python, TypeScript or Rust
selects it. The whole viewer package — not just its `*.ts` — is `dom_ts`,
because its Python fixture generators feed the TypeScript tests.

`dom_py` additionally owns the **cross-language** gates that happen to live in
`python-tests`, so their non-Python inputs are classified as Python: the
format-contract source (`format-contract/contract.yaml`) and its generated
TypeScript half, the viewer `package.json` (the other end of the version
consistency check), the root `Makefile` and viewer fixture-generation entry
points guarded by `test_fixture_environment.py`, and the `demos/data` tree with
its manifest. It owns, for the same reason, the data and documentation files
the pytest suite itself reads: `scripts/complexity_baseline.json` (the C901
ratchet's only input that carries no Python extension),
`scripts/gallery/manifest.json` (cross-validated against the demo registry) and
`docs/guides/user/CLI_REFERENCE.md` (drift-guarded against the live Typer app),
plus the root `README.md`, `CLAUDE.md`, and
`.agents/skills/luxar-visualization/SKILL.md` guarded against the live demo and
example inventories, `packages/luxar/src/luxar/demos/README.md` guarded against
the exported helper inventory, and the gallery capture spec whose `DemoEntry`
interface defines the manifest field contract. The Cholesky documentation guard
reads that same `CLAUDE.md` and skill page, and adds
`.agents/skills/luxar-visualization/references/scene-api.md`,
`.agents/skills/luxar-gsplat-pipeline/SKILL.md` and
`docs/specs/GSPLATS_DIMENSION_MAPPING.md` to the Python-owned set.
Consequently, every `CLAUDE.md` edit runs the Python matrix.
Two workflow files and `.gitattributes` are `dom_py` for the same reason:
`test_docs_workflow.py` reads `docs.yml` and `.gitattributes`, and
`test_run_external_reference_audits.py` asserts the schedule, permissions and
token wiring of `external-reference-audits.yml`. A workflow file matches no
other domain on its own, so each has to be named or its guard never runs.
Viewer TypeScript sources read by Python contract tests are also `dom_py`.
Those tests resolve files through the shared `viewer_source()` helper, and
`test_ci_diff_classifier.py` statically scans every literal helper call: each
must have a Python `GATE_INPUTS` row, while every `NON_PYTHON_DOMAIN_PATHS`
control must remain unread. Five viewer inputs are consumed without opening a
named path in a test: the version and generated-format checks run through their
scripts, while `test_fixture_environment.py` matches its three fixture files via
`git grep`. The classifier test keeps those explicit exceptions disjoint from
the scanned readers and requires every `dom_py` viewer row to be in one set or
the other. `GATE_INPUTS` is therefore the exact declaration; the workflow ERE is
its checked copy rather than a second unchecked inventory. Ownership stays
file-narrow so unrelated viewer changes do not pull in the Python matrix. The
docs gate has no corresponding hole: it already owns every viewer TypeScript
source under `src/`, while viewer tools outside `src/` are outside both the
documentation checker's viewer scan and TypeDoc's entry points. `dom_ts`
explicitly owns the root `README.md` and gallery manifest because the
gallery-selection unit test resolves and validates the README capture set from
them.
A check whose own inputs are unclassified is a check that skips for exactly the
change it exists to catch. `.github/workflows/ci.yml` selects **all four**
domains: it defines how every suite is invoked, so an edit that breaks a command
or a condition is caught by the run that contains it.

A change that touches no domain at all — most Markdown, `docs/`, and
`CHANGELOG.md`, except for the explicitly classified gate inputs noted above —
runs no language suite. Those jobs still *run* (checkout plus skipped steps),
so their required contexts (`python-tests (3.12)`, `typescript-tests`,
`release-readiness`, `wheel-viewer`) report an explicit green in seconds
instead of a grey "skipped", which is what keeps strict branch protection from
wedging. `docs-quality`, the fifth required context, is gated separately on
`docs_relevant` — a docs-only change is documentation-relevant by definition,
so it runs the full Sphinx and TypeDoc gate, which is the point.

A push to `dev`, a scheduled run, or an empty diff has no PR base and selects
every domain. The gate **fails safe**: each condition is written
`dom_x != 'false'`, so if the `changes` job itself dies its outputs read empty
and every suite runs. (Writing them `== 'true'` would invert that — a broken
classifier would report an all-green run with no CI behind it.) Rename
detection is disabled in the classifier (`git diff --no-renames`) so moving a
file never hides a deletion from the documentation gate.

What scoping gives up is *latency*, not coverage. A break that only shows
across a domain boundary — a Python encoder change the TypeScript decoder
cannot read, say, since the viewer's fixture tests are generated by
`packages/luxar-viewer/tests/fixtures/generate_test_data.py` — is not caught by
the pull request that introduces it, because that diff selects only `dom_py`.
It is caught by the merge's own push run, which has no PR base and therefore
runs everything. The same trade as the per-PR Python matrix below: found on
`dev` within minutes rather than in the PR.

### Which Python versions CI runs

`python-tests` is a matrix whose legs depend on the event:

| Event | Python legs |
|-------|-------------|
| `pull_request` | `3.12` — the floor, and the one required status context |
| `push` to `dev` | `3.12`, `3.13`, `3.14` |
| daily `schedule` (`09:17` UTC) | `3.12`, `3.13`, `3.14` |
| remaining `:17` windows (`00,03,06,12,15,18,21` UTC) | `3.12` — the promotion-required context |

3.12 is the FLOOR (`requires-python = ">=3.12"`, what zarr 3.2+ requires) and is
what the required `python-tests (3.12)` status context names, so it runs on every
event. `>=3.12` has no ceiling, though: 3.13 and 3.14 are supported, `install-hatch`
explicitly prefers them, and a developer's `hatch env` picks the newest interpreter
on the box. Every merge push and the daily 09:17 UTC schedule therefore run exactly
the versions the wheel's classifiers advertise — "declared" and "tested" are kept
identical by construction, because a claimed-but-never-exercised version is the
same species of lie as an untested 3.10 claim would be. All eight scheduled windows
land every three hours; the seven other than 09:17 carry only the required 3.12 leg.
On obsidian, `max-parallel: 2` prevents one run's Python matrix from monopolising all
three shared slots; repository-wide queue order may still put other work ahead of
that run's `typescript-tests`. The final Python leg follows. A successful TypeScript
attempt ran 17m53s of real steps; at the documented 3.3x `SCHED_IDLE` extreme, a
healthy starved run projects to roughly 59 minutes, leaving the former 60-minute
budget no headroom for any pre-step dispatch latency. A dispatch-lost leg can spend
the same budget without starting a step. `typescript-tests` now carries 120 minutes,
at the cost of a doubled time-to-red for that leg. Every scheduled window also runs
`changes`, `pick-runner`, `docs-quality`, `release-readiness`, `wheel-viewer`,
`go-launcher`, and the 10-minute `queue-watchdog` window on GitHub-hosted runners.
Together those jobs were about a 20-minute hosted wall-clock floor on the initial PR
run for this policy. The watchdog now runs on every same-repo run because all such
runs select `obsidian`. The long Python/TypeScript legs always join the obsidian queue
unless the operator has created/set the Actions repository variable
`LUXAR_CI_FORCE_HOSTED` to `1` under **Settings → Secrets and variables → Actions →
Variables**; schedules have no automatic paid exception. Sustained contention can
therefore cancel successive promotion windows. During an extended outage or promotion
stall, set the variable before the next window, then clear it after capacity recovers.
(If newer interpreters ever become deliberately unsupported, the honest fix is a
`requires-python` upper bound, not a quiet single-leg matrix.)

This is also why the version-equality assertion in the job matters: it proves each
leg really ran the interpreter it claims, rather than whatever pipx picked — the
defect behind issue #839, where all three legs silently ran the same version.

`pick-runner` is deliberately not a capacity router. Its first step sends fork PRs and
the explicit `LUXAR_CI_FORCE_HOSTED=1` break-glass to `ubuntu-latest`; its second step
sends every remaining event, including schedules and full-workflow reruns, to
`obsidian`. It reads no heartbeat, repository activity, or backlog state and checks out
no repository code. The output retains an `ubuntu-latest` fallback only for a selector
job failure, so required checks do not receive an empty `runs-on` value.

`queue-watchdog` still uses the stdlib-only `scripts/ci_queue_scan.py` helper to detect
an obsidian-routed run whose jobs remain queued while no obsidian work is active. It
sparse-checks out `scripts/` with credentials disabled and treats unreadable liveness
data as a reason not to cancel. If two consecutive scans find queued work but no active
obsidian jobs, it cancels the run and instructs the operator to create/set the Actions
repository variable `LUXAR_CI_FORCE_HOSTED` to `1`, rerun all jobs, and clear it after
capacity recovers. The former scheduled queue redispatcher was removed: under
obsidian-only routing, cancelling a queued run and creating a fresh attempt merely
returns it to the same queue.

Scheduled and push runs differ from a PR run in *scope* as well: neither has a PR
base, so the `changes` job cannot path-filter and selects the whole suite plus the
documentation gate. On a scheduled run, `changes` checks out the immutable event SHA
and captures that commit once; every downstream suite and repair checkout uses the
captured SHA. The run's check contexts attach to that same event SHA regardless of
what the jobs check out, so the pin keeps the tested tree and its contexts aligned.

Scheduled runs sit in their own `concurrency` group. While `dev` is the default,
they share `refs/heads/dev` with merge-triggered runs; after the default flips,
their event ref becomes `refs/heads/main`, but `github.event_name` still keeps the
groups separate. Under one shared group `cancel-in-progress` let whichever started
second cancel the other. A merge landing mid-schedule killed the scheduled run; a
cron firing over an in-flight merge killed that merge's push run, which is the only
place the new `dev` commit gets the full matrix at all. Scheduled runs still share a
group with each other, so a window that remains in flight three hours later is
cancelled by its successor — which is itself a promotion window, so a lost window
costs three hours rather than the cadence. *Unloaded*, a floor-only window is one
Python leg beside `typescript-tests`, about 83–91 minutes against the 180 minutes of
spacing; the 09:17 full-matrix window is three legs at `max-parallel: 2`, so two
waves, roughly 166–182 minutes plus the hosted `changes`/`pick-runner` preamble — at
the spacing rather than under it. Both are unloaded figures on a deliberately
`SCHED_IDLE` box, so neither window is guaranteed to finish. The full-matrix one is
simply the first to lose, and the 3.13/3.14 coverage it carries then waits for the
next day. A floor-only window needs about half as much quiet and is therefore the
last to be lost; sustained contention at the documented 2.4–3.3x stretch can cancel
both until the box quiets. The cron fires the whole workflow rather than
`python-tests` alone — a schedule event has no PR base, so change detection selects
the full suite and the documentation gate as well.

GitHub branch protection does not necessarily replace a cancelled push check with a
later successful scheduled check of the same name on the same SHA. After a scheduled
run has completed the five protected contexts successfully,
`repair-cancelled-push-checks` resolves dev's tip and enumerates every commit still in
`main..dev`, newest first, then inspects each completed push run. This covers commits
skipped by the three-hour schedule instead of repairing only the scheduled tip. The
promotion daemon fast-forwards `main` to the newest green ancestor, so the walk stops
after successfully enqueueing two candidates: the second is a hedge against a
genuinely red newest candidate, while repairing still-older commits cannot advance the
same promotion. Cancelled jobs for any of the five protected contexts are rerun. A
single cancelled context uses a job-level rerun. Two or more use one failed-jobs
rerun because GitHub returns `403` once the first job-level rerun has moved the run
into a new attempt; the run-level path also re-enqueues cancelled or failed
non-required matrix legs such as Python 3.13/3.14. A schedule whose own protected
contexts are not all green performs no repair. Candidate API read failures and
rejected reruns are reported as warnings; one candidate cannot abort the remaining
walk. A failed repaired job is terminal for that SHA unless it is included in the
multi-job failed-jobs rerun; otherwise recovery requires a manual rerun.

Fresh runs keep coalescing by event and ref, while reruns use their original run id in
the concurrency key. A later merge therefore cannot cancel a repaired attempt, and
repairs for different backlog commits cannot cancel each other. A later attempt of the
same original run now supersedes its earlier attempt because both use the same run id;
the previous attempt-number key kept them apart. That exemption applies to ordinary PR
reruns too, and neither rerun path restarts `queue-watchdog`. An obsidian-routed job
left undispatched after its runners disappear can therefore remain queued until
GitHub's 24-hour ceiling; one dispatched before its slot recycles can instead reach its
timeout before any step starts or runner name is recorded. The repair job has only
`actions: write` and `contents: read` permissions and runs on GitHub-hosted Linux;
recovered long legs reuse their original runner-routing decision and repay work that
the scheduled run already performed. A multi-job candidate can add up to four long
obsidian-routed legs alongside the next push run, so the two-candidate cap permits up
to eight per window. Do not widen that cap without re-measuring queue pressure.
Reruns execute the workflow definition from their original SHA, so commits predating
the run-id key retain the older attempt-only collision behavior; the scheduled SHA
itself carries the new policy and provides the forward promotion candidate that clears
that rollout backlog.

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
| Python | 3.12 | zarr 3 requires >=3.12 from 3.2 on; also stdlib `tomllib`, PEP 695 type stubs |
| Node.js | 22.22 | jsdom 30 engines `^22.22.2 || ^24.15.0 || >=26.0.0` (undici 8 crashes on older Node); Vite 8.x needs only 20.19 |
| Rust | stable | WASM compilation |
| wasm-pack | 0.15.0 (pinned) | WASM packaging — `install-rust` installs exactly `WASM_PACK_VERSION` (see the Makefile) with `cargo install --locked --force`, then fails unless PATH answers with that version |

## Related Documentation

- `CONTRIBUTING.md` (repo root) - Contributing guidelines
- `CLAUDE.md` (repo root) - AI assistant instructions
- [TESTING_GUIDELINES.md](./TESTING_GUIDELINES.md) - Testing best practices
- [PLAYWRIGHT_GUIDE.md](./PLAYWRIGHT_GUIDE.md) - E2E testing guide
