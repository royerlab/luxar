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
2. Checks Node.js version (requires 22.22+ — jsdom 30's declared floor; Vite 8.x supports `^20.19.0 || >=22.12.0`)
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
| `make test-e2e-browsers` | Run the cross-browser Playwright subset |
| `make test-e2e-mobile` | Run the mobile/touch Playwright suite used by PR CI |
| `make test-e2e-smoke` | Run the E2E smoke subset |
| `make test-e2e-smoke-strict` | Run the smoke subset with strict browser-console handling |
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
| `make check-record-attribution` | Opt-in offline audit comparing each Zenodo record's captured description with the manifest `attribution`; reports where a record asserts a publication describes *the imaging* while the imaging is unpublished (one describing the instrument or method is the correct framing and is never flagged). Report-only because the wording is authored on Zenodo, which is also why its live-repo test asserts only repository-controlled properties — `scripts/tests` runs in the required `python-tests` job |
| `make check-external-references` | Run every external-reference audit — network-backed ones plus the offline record-attribution comparison — and emit one PASS/NOTICE/WARNING/ERROR report; always non-gating |
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

#### Native backend release verification

The Linux compile gate cannot exercise the Objective-C++ binding, the Metal
shader compiler, or the Metal parity suite. Hosted macOS CI and a dedicated Mac
runner are not used today — they are deferred until a Mac runner exists.
Every release candidate must therefore be checked manually on Apple silicon
before it is tagged:

```bash
hatch run check-native --require cxx --require metal
LUXAR_REQUIRE_METAL=1 hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests -v -rs
```

Without a CUDA toolkit, the compile check must report successful checks for
`nlm/bindings.cpp`, `metal/bindings.mm`, and `metal/kernels.metal`, skip
`cuda/bindings.cpp` because its headers are unavailable, report the `nvcc` arm
as `SKIP`, and finish with `3/3 translation unit(s) compile-checked`. The parity
command fails during pytest configuration if the Metal backend or MPS interop is
unavailable, so a release check cannot pass with the Metal tests silently
skipped.

CUDA compile and parity coverage runs on a separate low-priority, dispatch-only
cadence rather than in pull-request CI, because the device compile takes minutes
and the workstation GPUs are shared with interactive work. The systemd timer in
`royerlab/luxar-ci` dispatches `.github/workflows/cuda-nightly.yml` with
`--ref dev`; a newly added dispatch workflow is unavailable until promotion
first carries it to the default branch (`main`). The two-GPU job compile-checks
both `nvcc` translation units, builds the splatting and NLM extensions, and runs
both parity suites with `LUXAR_REQUIRE_CUDA=1`, so a missing backend fails during
pytest configuration instead of silently skipping. A failure opens or updates
the `CUDA native cadence failure` issue assigned to @royerloic. The daily
`.github/workflows/cadence-liveness.yml` job separately fails when successful
dispatches stop arriving within the cadence table's staleness window.

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

Demo renderer tests use a separate `demos` environment, which creates roughly
7 GB on first use. Run the focused GL suite with
`hatch run demos:pytest packages/luxar/src/luxar/demos/tests/test_clay_renderer.py`;
`hatch env remove demos` reclaims it afterwards.

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
| Old system GCC (< 10) | `build_cuda_slurm.py` auto-detects a `gcc/` module >= 10 to load |
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
3. **Finds GCC >= 10 module** — runs `module spider gcc`, picks the highest `gcc/X.Y` with X >= 10 (required for the shipped C++20 build; system GCC on RHEL 8 is 8.5.0)
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
| `GCC version too old` | System GCC < 10 | Ensure `module spider gcc` shows gcc >= 10 on compute nodes |
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
| `dom_ts` | anything under `packages/luxar-viewer/`, root `tsconfig*.json`, `vitest*.{ts,js,mjs}`, plus gallery-selection and E2E-wiring inputs listed below | `typescript-tests`, `release-readiness`, `wheel-viewer` |
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
Five workflow files, `.gitattributes`, and `.gitignore` are `dom_py` for the
same reason: `test_docs_workflow.py` reads `docs.yml` and `.gitattributes`,
`test_run_external_reference_audits.py` asserts the schedule, permissions and
token wiring of `external-reference-audits.yml`, the classifier test parses
`coverage.yml` and `cuda-nightly.yml`,
`test_daily_workflow_has_the_permissions_and_token_to_enforce_the_table` parses
`cadence-liveness.yml`, and the wheel-completeness guard reads `.gitignore`. A
workflow file matches no other domain on its own, so each has to be named or its
guard never runs.
Viewer TypeScript sources read by Python contract tests are also `dom_py`.
Those tests resolve files through the shared `viewer_source()` helper, and
`test_ci_diff_classifier.py` statically scans every literal helper call: each
must have a Python `GATE_INPUTS` row, while every `NON_PYTHON_DOMAIN_PATHS`
control must remain unread. A second scan checks whole tracked non-Python path
literals in pytest test modules, `conftest.py` files, and helpers under `tests/`,
while a third resolves module-level, repo-rooted `Path` chains that flow into
`read_text`, `read_bytes`, or read-only `open` calls. Every discovered path that
lacks `dom_py` needs a Python `GATE_INPUTS` row or a justified exclusion, even if
another language already owns it. Documentation relevance is also independent:
Markdown and RST inputs read by pytest still need `dom_py` even though they
select `docs-quality`.
Ten viewer inputs are consumed without a literal `viewer_source()` call: the
version and generated-format checks run through their scripts, direct readers
include the viewer README and two `CURRENT_VERSION_CLAIMS` sources, while
`test_fixture_environment.py` matches its three fixture files via `git grep` and
the two repo-rooted `readFileSync` reader files are scanned by the classifier.
The classifier test keeps those explicit exceptions disjoint from the scanned
readers and requires every `dom_py` viewer row to be in one set or the other.
`GATE_INPUTS` is therefore the exact declaration; the workflow ERE is its
checked copy rather than a second unchecked inventory. Ownership stays
file-narrow so unrelated viewer changes do not pull in the Python matrix. The
docs gate has no corresponding hole: it already owns every viewer TypeScript
source under `src/`, while viewer tools outside `src/` are outside both the
documentation checker's viewer scan and TypeDoc's entry points. `dom_ts`
explicitly owns the root `README.md` and both gallery manifests because the
gallery-selection unit test resolves and validates the README capture set from
them. It also owns the root `Makefile` because the generated-fixture freshness
test checks its E2E fixture prerequisite wiring, plus
`scripts/generate_builtin_colormaps.py` because the viewer's third-party notices
test scrapes its colormap tables. A narrow static scan over viewer `src/**/*.test.ts`
files finds literal `readFileSync` inputs rooted through `join(REPO_ROOT, ...)` or
`resolve(REPO_ROOT, ...)` and requires each to have a TypeScript `GATE_INPUTS` row.
The matched reader source set must exactly equal the named Python inputs so their
edits run the classifier and stale ownership rows are rejected. This scan does not
cover `import.meta`-rooted reads, `*.spec.ts`, or `scripts/*.test.mjs`; those
existing inputs are already owned by broader TypeScript patterns, while a
brand-new `*.test.ts` reader is reported the next time another Python-relevant
change runs the repository-wide classifier.
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

A push to `dev`, a dispatched run, or an empty diff has no PR base and selects
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

| Event | Python legs (`python-tests`) |
|-------|-------------------------------|
| `pull_request` | `3.12` — the floor, and the one required status context |
| `push` to `dev` | `3.12`, `3.13`, `3.14` |
| `workflow_dispatch` | `3.12` by default; `3.12`, `3.13`, `3.14` with `full_python_matrix=true` |

Coverage instrumentation plus the 89% `fail_under` gate is the dominant cost of
`python-tests`, so it is off the per-PR critical path: PRs run the `-m 'not slow'`
suite plain (`hatch run test-nocov`). Pushes to `dev` still run `test-cov`; the
protected `python-tests (3.12)` context is the promotion-visible enforcement path
and is load-bearing even though ci.yml's cancel-in-progress policy means some
superseded dev runs never finish.

A **separate workflow, `.github/workflows/coverage.yml`**, also runs `test-cov` on
every push to `dev` (and on `workflow_dispatch`). Its per-commit concurrency group
(`coverage-${{ github.sha }}`, `cancel-in-progress: false`) ensures a newer dev push
never cancels an older coverage run. An `obsidian` outage can still leave a run queued
until GitHub expires it; `LUXAR_CI_FORCE_HOSTED=1` is the recovery path. That
`coverage` context is currently advisory because it is not one of main's protected
contexts; adding it to repository protection is the known settings gap. The workflow
defaults to `obsidian` and keeps hosted runs serial. A `schedule` trigger was
deliberately not used: scheduled checks attach to the default branch's tip, not the
dev commit tested. Dispatches of ci.yml also execute `test-cov`. The `python-tests`
context name is unchanged, so no required status is orphaned.

3.12 is the floor (`requires-python = ">=3.12"`) and names the required
`python-tests (3.12)` context. Merge pushes exercise every supported interpreter,
so the wheel classifiers and tested versions stay aligned. The promotion service
requests a repair window at most once every three hours while cancelled contexts block
promotion. A missed dispatch leaves promotion stale until a later request or a naturally
green merge push; the service warns rather than failing its promotion pass. A manually
dispatched repair window defaults to the required leg; the full-matrix input is available
for diagnostics without restoring a redundant daily cron. Every dispatch also runs the
cancelled-push repair walk. Dispatch from `dev`, not the Actions UI's default `main`
selection. On obsidian,
`max-parallel: 2` prevents one Python matrix from monopolising all three shared
slots. The TypeScript timeout remains 120 minutes to cover both dispatch latency
and the measured `SCHED_IDLE` slowdown.

The version-equality assertion in the job proves each leg really ran the interpreter
it claims, rather than whatever pipx selected — the defect behind issue #839.

`pick-runner` is deliberately not a capacity router. Fork PRs and the explicit
`LUXAR_CI_FORCE_HOSTED=1` break-glass use `ubuntu-latest`; every other event,
including workflow dispatches and reruns, uses `obsidian`. It reads no heartbeat,
repository activity, or backlog state. During an extended outage or promotion stall,
open **Settings → Secrets and variables → Actions → Variables**, create or set
`LUXAR_CI_FORCE_HOSTED` to `1`, rerun all jobs, and clear it after capacity recovers.

`queue-watchdog` uses the stdlib-only `scripts/ci_queue_scan.py` helper to detect an
obsidian-routed run whose jobs remain queued while no obsidian work is active. It
sparse-checks out `scripts/` with credentials disabled, fails open on unreadable
liveness data, and only cancels when jobs are still queued and two consecutive scans
find no active obsidian job. A dispatched run checks out the scanner from `dev`. The
former scheduled queue redispatcher was removed because cancelling a queued run and
creating a fresh attempt merely returns it to the same queue.

The promotion service requests repair windows with `workflow_dispatch --ref dev`.
That makes `github.sha`, the check-run attachment, the tree checked out by `changes`,
and the captured `dev_sha` the same immutable dev commit. Every downstream suite and
repair checkout reuses it. Dispatches have no PR base, so they select the whole suite
and documentation gate. Their `workflow_dispatch` concurrency group is separate from
push runs; a newer dispatch can supersede an older dispatch without cancelling the
merge push that produced the candidate commit. A dispatch requested on any other ref
fails `changes`, but the fail-safe suite jobs still run against and report on that same
dispatched commit. The operator error is therefore loud and self-consistent, but wastes
a repair window rather than aborting it.

After a dispatched window completes the five protected contexts successfully,
`repair-cancelled-push-checks` resolves dev's current tip and enumerates commits in
`main..dev`, newest first. This is necessary because GitHub branch protection does not
necessarily replace a cancelled push check with a later successful check of the same
name on the same SHA. It reruns cancelled required jobs from up to two completed push
suites. The second candidate hedges against a genuinely red newest candidate; repairing
still-older commits cannot advance the same promotion. One cancelled context uses a
job-level rerun. Multiple contexts use one failed-jobs rerun because GitHub rejects a
second job-level rerun after the attempt changes; that run-level path also re-enqueues
cancelled or failed non-required legs such as Python 3.13/3.14. A failed repaired job is
terminal for that SHA unless a multi-job failed-jobs rerun includes it; otherwise it
needs a manual rerun. The repair checkout must remain on dev's ancestry, and an
unresolvable dev ref or off-dev checkout fails loudly. Candidate API failures and
rejected reruns are warnings so one candidate does not abort the remaining walk.

Fresh runs coalesce by event and ref; reruns use the original run id so later merges
cannot cancel repaired attempts and repairs for different commits do not collide.
Reruns execute the workflow definition from their original SHA, and neither rerun path
restarts `queue-watchdog`. A job left undispatched after its runners disappear can remain
queued until GitHub's 24-hour ceiling; one dispatched before its slot recycles can reach
its timeout before any step starts or runner name is recorded. `pick-runner`'s
`ubuntu-latest` output is only a fallback if the selector job itself fails, ensuring the
required jobs never receive an empty `runs-on`. A multi-job candidate can add up to four
long obsidian legs, so the two-candidate cap permits up to eight per window. Do not widen
that cap without re-measuring queue pressure.

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
| Node.js | 22.22 | jsdom 30 engines `^22.22.2 || ^24.15.0 || >=26.0.0` (undici 8 crashes on older Node); Vite 8.x supports `^20.19.0 || >=22.12.0` |
| Rust | stable | WASM compilation |
| wasm-pack | 0.15.0 (pinned) | WASM packaging — `install-rust` installs exactly `WASM_PACK_VERSION` (see the Makefile) with `cargo install --locked --force`, then fails unless PATH answers with that version |

## Related Documentation

- `CONTRIBUTING.md` (repo root) - Contributing guidelines
- `CLAUDE.md` (repo root) - AI assistant instructions
- [TESTING_GUIDELINES.md](./TESTING_GUIDELINES.md) - Testing best practices
- [PLAYWRIGHT_GUIDE.md](./PLAYWRIGHT_GUIDE.md) - E2E testing guide
