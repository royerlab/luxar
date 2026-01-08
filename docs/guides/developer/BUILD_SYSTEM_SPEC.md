# Build System Specification

This document provides comprehensive documentation for Luxar's build system, development setup, and Makefile commands.

## Overview

Luxar uses a Makefile-based build system designed to work on fresh Linux and macOS machines with minimal pre-installed tools. The system automatically detects the operating system and package manager, then installs required dependencies without requiring sudo (where possible).

### Design Goals

1. **Zero-friction setup**: Run `make dev-setup` on a fresh machine
2. **No sudo required**: Use nvm for Node.js, pipx for Python tools
3. **Cross-platform**: Support Linux (apt, dnf, yum) and macOS (brew)
4. **Graceful degradation**: Clear error messages with copy-paste solutions
5. **Idempotent**: Safe to run multiple times

## Prerequisites

### Minimal Requirements

Before running `make dev-setup`, you need:

| Tool | Required Version | Notes |
|------|-----------------|-------|
| Python | 3.9+ | Usually pre-installed on Linux/macOS |
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
make dev-setup

# Verify installation
make check-deps

# Start developing
make viewer      # Start viewer dev server
make test-all    # Run all tests
```

## Development Setup Process

### What `make dev-setup` Does

The setup process has 5 steps:

#### Step 1: Python Environment

1. Verifies Python 3.9+ is available
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
2. Displays instructions for `make setup-rust` if not installed
3. WASM is optional - viewer works without it (uses TypeScript fallback)

### Rust/WASM Setup Details

The Rust/WASM toolchain enables high-performance WebAssembly computations in the viewer (e.g., nD filtering, distance calculations). It's optional but recommended for best performance - without it, the viewer falls back to TypeScript implementations.

**What `make setup-rust` does:**
1. Installs Rust via rustup (if not present)
2. Loads the cargo environment automatically
3. Installs wasm-pack for WASM packaging

**Key Design:**
- All Rust-related make targets source `~/.cargo/env` automatically
- The `build-wasm.sh` script also sources cargo env at startup
- No manual `source ~/.cargo/env` is required after installation

**Build flow:**
```
make viewer-build
  ├─ Checks for wasm-pack
  ├─ If missing: runs make setup-rust
  └─ Runs pnpm build
       └─ pnpm build:wasm (scripts/build-wasm.sh)
            ├─ Sources ~/.cargo/env
            ├─ Verifies wasm-pack is available
            └─ Runs wasm-pack build
```

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
| `make dev-setup` | Complete development environment setup |
| `make check-deps` | Check all dependencies and their versions |
| `make install-node` | Install/upgrade Node.js via nvm (Linux) or brew (macOS) |
| `make install-pnpm` | Install pnpm package manager |
| `make install-hatch` | Install Hatch via pipx |
| `make setup-rust` | Install Rust toolchain and wasm-pack |
| `make deep-clean-dev-setup` | Remove ALL dev tools to simulate fresh machine |

### Quality & Testing

| Command | Description |
|---------|-------------|
| `make check` | Run all quality checks (Python + TypeScript) |
| `make test` | Run Python tests |
| `make test-all` | Run all tests (Python + Rust + TypeScript) |
| `make test-cov` | Run Python tests with coverage |
| `make lint` | Run ruff linting |
| `make type-check` | Run mypy type checking |
| `make security` | Run bandit security scan |
| `make format` | Format Python code |
| `make format-all` | Format all code (Python + TypeScript) |

### Viewer Development

| Command | Description |
|---------|-------------|
| `make viewer` | Start viewer dev server (port 5173) |
| `make viewer-build` | Build viewer for production (requires Rust) |
| `make viewer-rebuild` | Clean rebuild of viewer |
| `make viewer-test` | Run TypeScript unit tests |
| `make viewer-test-cov` | Run TypeScript tests with coverage |
| `make viewer-lint` | Run TypeScript linting |
| `make viewer-typecheck` | Run TypeScript type checking |
| `make viewer-format` | Format TypeScript code |
| `make viewer-check` | Run all TypeScript checks |

### WASM Development

| Command | Description |
|---------|-------------|
| `make wasm-build` | Build WASM module |
| `make wasm-test` | Run Rust unit tests |
| `make wasm-clean` | Clean WASM build artifacts |

### Data & Demos

| Command | Description |
|---------|-------------|
| `make demo` | Generate demo dataset (100k points) |
| `make demo-and-serve` | Create demo and start both servers |
| `make run-examples` | Generate all example datasets |
| `make serve-examples` | Serve datasets directory |
| `make serve-data` | Serve a specific dataset |

### Documentation

| Command | Description |
|---------|-------------|
| `make docs-build` | Build Sphinx documentation |
| `make docs-serve` | Serve documentation locally |
| `make docs-clean` | Clean documentation artifacts |
| `make check-docs` | Check documentation quality |

### Utilities

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make clean` | Clean temporary files and caches |
| `make clean-examples` | Clean generated example datasets |
| `make stats` | Generate project statistics report |
| `make shell` | Enter Hatch development shell |
| `make env-show` | Show Hatch environments |
| `make env-prune` | Remove unused Hatch environments |

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
make dev-setup
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
make setup-rust
```

This command:
1. Installs Rust via rustup (if not present)
2. Installs wasm-pack (if not present)
3. Sources cargo environment automatically

The Makefile commands (`make wasm-build`, `make viewer-build`, etc.) automatically source the cargo environment, so you don't need to run `source ~/.cargo/env` manually.

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
# ✅ pnpm: 10.27.0
# ⚪ Rust not installed (optional)
# ⚪ wasm-pack not installed (optional)
```

### Clean Slate Recovery

If something goes wrong, reset everything:

```bash
# Remove all dev tools (interactive, confirms before proceeding)
make deep-clean-dev-setup

# Then start fresh
make dev-setup
```

## Environment Variables

The build system uses these environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NVM_DIR` | nvm installation directory | `~/.nvm` |
| `DATASET` | Dataset path for `make serve-data` | `datasets/demos/demo.zarr` |
| `PORT` | Server port for data serving | `8000` |

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
  run: make dev-setup

- name: Run checks
  run: |
    make check
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
| Python | 3.9 | Type hints, dataclasses |
| Node.js | 20.19 | Vite 7.x requirements |
| Rust | stable | WASM compilation |
| wasm-pack | latest | WASM packaging |

## Related Documentation

- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - Contributing guidelines
- [CLAUDE.md](../../../CLAUDE.md) - AI assistant instructions
- [TESTING_GUIDELINES.md](./TESTING_GUIDELINES.md) - Testing best practices
- [PLAYWRIGHT_GUIDE.md](./PLAYWRIGHT_GUIDE.md) - E2E testing guide
