# Makefile for Luxar development tasks
# Uses Hatch for on-demand environment management
#
# This Makefile is designed to work on fresh Linux/macOS machines with minimal
# pre-installed tools. Run 'make setup-dev' to automatically install all dependencies.
#
.PHONY: help install-dev install-demo-deps format-python format-typescript format-rust format-cuda format-all gen-contract gen-data-manifest \
        lint-python lint-typescript type-check-python type-check-typescript security \
        test-all test-python test-cov-python test-cov-typescript test-cov-all test-fixtures test-wasm test-viewer test-viewer-fixtures test-e2e \
        clean-all clean-python clean-viewer clean-examples clean-cache clean-setup enable-pre-commit run-pre-commit \
        check-all check-typescript check-rust check-wasm-deps setup-dev \
        check-docs check-docs-verbose clean-docs build-docs build-typedoc serve-docs \
        demo run-demos run-examples serve-examples serve-dataset install-viewer-deps viewer build-viewer rebuild-viewer \
        install-rust build-wasm clean-wasm generate-readme-demos generate-readme-images generate-doc-images generate-readme-videos \
	generate-gallery-datasets generate-gallery \
        stats stats-fast show-env prune-env shell build publish-test publish \
        check-deps install-node install-pnpm install-hatch \
        setup-cuda check-cuda-deps build-cuda build-cuda-slurm clean-cuda test-cuda benchmark-cuda \
        benchmark-metal benchmark-metal-stress \
        build-nlm-cuda clean-nlm-cuda test-nlm-cuda \
        benchmark-wasm \
        install-go build-launchers clean-launchers

# ============================================================================
# Shell hardening
# ============================================================================
# Run every recipe under bash with errexit + pipefail so a failing command in
# a `;`-joined sequence or a piped stage aborts the recipe instead of being
# silently swallowed. `.DELETE_ON_ERROR` removes half-written targets on
# failure. (nounset/-u is intentionally NOT set: this Makefile relies on many
# conditionally-set shell vars.)
SHELL := bash
.SHELLFLAGS := -e -o pipefail -c
.DELETE_ON_ERROR:

# Resolve `hatch` once: prefer one on PATH, else the ~/.local/bin fallback used
# on HPC/no-sudo machines. Use $(HATCH) for all hatch invocations in recipes.
HATCH ?= $(shell command -v hatch 2>/dev/null || echo $(HOME)/.local/bin/hatch)

# Shared find-prune prefix: skip VCS/dependency/venv trees when cleaning so we
# never recurse into node_modules/.git/.venv (slow + can delete the wrong dirs).
FIND_PRUNE := -name node_modules -prune -o -name .git -prune -o -name .venv -prune -o

# ============================================================================
# OS Detection and Configuration
# ============================================================================
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OS := macos
    PKG_MANAGER := brew
else ifeq ($(UNAME_S),Linux)
    OS := linux
    # Detect package manager (apt, dnf, or yum)
    ifneq ($(shell command -v apt-get 2>/dev/null),)
        PKG_MANAGER := apt
    else ifneq ($(shell command -v dnf 2>/dev/null),)
        PKG_MANAGER := dnf
    else ifneq ($(shell command -v yum 2>/dev/null),)
        PKG_MANAGER := yum
    else
        PKG_MANAGER := unknown
    endif
else
    OS := unknown
    PKG_MANAGER := unknown
endif

# Minimum Node.js version required by Vite 8.x
MIN_NODE_MAJOR := 20
MIN_NODE_MINOR := 19

# ============================================================================
# Dependency Checking and Installation Helpers
# ============================================================================

# Helper function to check Node.js version
define check_node_version
	@if command -v node >/dev/null 2>&1; then \
		NODE_VERSION=$$(node -v | sed 's/v//'); \
		NODE_MAJOR=$$(echo $$NODE_VERSION | cut -d. -f1); \
		NODE_MINOR=$$(echo $$NODE_VERSION | cut -d. -f2); \
		if [ "$$NODE_MAJOR" -lt $(MIN_NODE_MAJOR) ] || \
		   ([ "$$NODE_MAJOR" -eq $(MIN_NODE_MAJOR) ] && [ "$$NODE_MINOR" -lt $(MIN_NODE_MINOR) ]); then \
			echo "❌ Node.js $$NODE_VERSION is too old. Vite requires Node.js $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"; \
			echo "   Please upgrade Node.js:"; \
			if [ "$(OS)" = "macos" ]; then \
				echo "   brew install node@22"; \
			elif [ "$(PKG_MANAGER)" = "apt" ]; then \
				echo "   # Using NodeSource for latest Node.js:"; \
				echo "   curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -"; \
				echo "   sudo apt-get install -y nodejs"; \
			else \
				echo "   Visit https://nodejs.org/ for installation instructions"; \
			fi; \
			exit 1; \
		else \
			echo "✅ Node.js $$NODE_VERSION (meets $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+ requirement)"; \
		fi; \
	else \
		echo "❌ Node.js not found"; \
		exit 1; \
	fi
endef

check-deps:  ## Check all development dependencies and their versions
	@echo "🔍 Checking development dependencies..."
	@echo ""
	@echo "System: $(OS) (package manager: $(PKG_MANAGER))"
	@echo ""
	@echo "=== Required Dependencies ==="
	@echo ""
	@# Python
	@if command -v python3 >/dev/null 2>&1; then \
		echo "✅ Python: $$(python3 --version)"; \
	else \
		echo "❌ Python3 not found"; \
	fi
	@# pipx (required for installing Hatch on modern systems)
	@if command -v pipx >/dev/null 2>&1; then \
		echo "✅ pipx: $$(pipx --version 2>/dev/null)"; \
	elif [ "$(OS)" = "macos" ]; then \
		echo "❌ pipx not found (run: brew install pipx)"; \
	elif [ "$(PKG_MANAGER)" = "dnf" ]; then \
		echo "❌ pipx not found (run: sudo dnf install pipx)"; \
	elif [ "$(PKG_MANAGER)" = "yum" ]; then \
		echo "❌ pipx not found (run: sudo yum install pipx)"; \
	else \
		echo "❌ pipx not found (run: sudo apt-get install pipx)"; \
	fi
	@# hatch (primary Python environment manager)
	@if command -v hatch >/dev/null 2>&1; then \
		echo "✅ Hatch: $$(hatch --version)"; \
	elif [ -x "$$HOME/.local/bin/hatch" ]; then \
		echo "✅ Hatch: $$($$HOME/.local/bin/hatch --version) (in ~/.local/bin)"; \
	else \
		echo "❌ Hatch not found (run 'make install-hatch')"; \
	fi
	@# Source nvm if available for Node.js checks
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if command -v node >/dev/null 2>&1; then \
		NODE_VERSION=$$(node -v | sed 's/v//'); \
		NODE_MAJOR=$$(echo $$NODE_VERSION | cut -d. -f1); \
		NODE_MINOR=$$(echo $$NODE_VERSION | cut -d. -f2); \
		NODE_PATH=$$(which node 2>/dev/null); \
		if echo "$$NODE_PATH" | grep -q ".nvm"; then \
			NODE_SOURCE="via nvm"; \
		elif echo "$$NODE_PATH" | grep -q "brew\|Homebrew\|Cellar"; then \
			NODE_SOURCE="via Homebrew"; \
		else \
			NODE_SOURCE="system"; \
		fi; \
		if [ "$$NODE_MAJOR" -lt $(MIN_NODE_MAJOR) ] || \
		   ([ "$$NODE_MAJOR" -eq $(MIN_NODE_MAJOR) ] && [ "$$NODE_MINOR" -lt $(MIN_NODE_MINOR) ]); then \
			echo "⚠️  Node.js v$$NODE_VERSION ($$NODE_SOURCE) - UPGRADE NEEDED: requires $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"; \
		else \
			echo "✅ Node.js: v$$NODE_VERSION ($$NODE_SOURCE)"; \
		fi; \
	elif [ -d "$$HOME/.nvm" ]; then \
		echo "⚠️  nvm installed but Node.js not found. Run: nvm install 22"; \
	else \
		echo "❌ Node.js not found (run 'make install-node')"; \
	fi; \
	if command -v npm >/dev/null 2>&1; then \
		echo "✅ npm: $$(npm --version)"; \
	else \
		echo "❌ npm not found"; \
	fi; \
	if command -v pnpm >/dev/null 2>&1; then \
		echo "✅ pnpm: $$(pnpm --version)"; \
	else \
		echo "❌ pnpm not found (run: npm install -g pnpm)"; \
	fi
	@echo ""
	@echo "=== Optional Dependencies (for WASM builds) ==="
	@echo ""
	@# Rust
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v rustc >/dev/null 2>&1; then \
		RUST_VERSION=$$(rustc --version 2>&1); \
		if echo "$$RUST_VERSION" | grep -q "rustup could not choose"; then \
			echo "⚠️  Rust: rustup installed but no default toolchain (run: rustup default stable)"; \
		else \
			echo "✅ Rust: $$RUST_VERSION"; \
		fi; \
	elif command -v rustup >/dev/null 2>&1; then \
		echo "⚠️  Rust: rustup installed but no toolchain (run: rustup default stable)"; \
	else \
		echo "⚪ Rust not installed (run 'make install-rust' if needed)"; \
	fi
	@# wasm-pack
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v wasm-pack >/dev/null 2>&1; then \
		echo "✅ wasm-pack: $$(wasm-pack --version)"; \
	else \
		echo "⚪ wasm-pack not installed (run 'make install-rust' if needed)"; \
	fi
	@echo ""
	@echo "=== Optional Dependencies (for native launchers) ==="
	@echo ""
	@# Go toolchain (used by `make build-launchers`)
	@GO_BIN=""; \
	if command -v go >/dev/null 2>&1; then \
		GO_BIN=go; \
	elif [ -x "$$HOME/.local/go/bin/go" ]; then \
		GO_BIN="$$HOME/.local/go/bin/go"; \
	fi; \
	if [ -n "$$GO_BIN" ]; then \
		echo "✅ Go: $$($$GO_BIN version | sed 's/^go version //')"; \
	else \
		echo "⚪ Go not installed (run 'make install-go' if you need 'make build-launchers')"; \
	fi
	@# Launcher binaries
	@LAUNCHER_DIR="packages/luxar/src/luxar/cli/_launchers"; \
	BUILT=""; \
	for tgt in darwin-universal linux-amd64 linux-arm64; do \
		if [ -x "$$LAUNCHER_DIR/$$tgt" ]; then \
			BUILT="$$BUILT $$tgt"; \
		fi; \
	done; \
	if [ -n "$$BUILT" ]; then \
		echo "✅ Native launchers built:$$BUILT"; \
	else \
		echo "⚪ Native launchers not built (run 'make build-launchers')"; \
	fi
	@echo ""
	@echo "=== Optional Dependencies (for CUDA builds) ==="
	@echo ""
	@# CUDA toolkit (nvcc)
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "✅ CUDA: $$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"; \
	else \
		echo "⚪ CUDA toolkit not installed (nvcc not found)"; \
	fi
	@# PyTorch CUDA support
	@if $(HATCH) run python -c "import torch; print('✅ PyTorch CUDA:', torch.version.cuda if torch.cuda.is_available() else 'not available')" 2>/dev/null; then \
		:; \
	else \
		echo "⚪ PyTorch CUDA not available"; \
	fi
	@# CUDA extension build status
	@if ls packages/luxar/src/luxar/gsplats/models/gsplats/cuda/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		echo "✅ CUDA extension: built"; \
	else \
		echo "⚪ CUDA extension: not built (run 'make build-cuda')"; \
	fi
	@echo ""

install-node:  ## Install or upgrade Node.js to required version (no sudo needed)
	@echo "📦 Installing Node.js $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+..."
	@echo ""
ifeq ($(OS),macos)
	@# macOS: use Homebrew (no sudo needed)
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "📥 Installing Homebrew first..."; \
		/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
	fi
	brew install node@22 || brew upgrade node
	@echo "✅ Node.js installed via Homebrew"
	@echo "Installed version: $$(node --version)"
else
	@# Linux: use nvm (no sudo needed)
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ ! -s "$$NVM_DIR/nvm.sh" ]; then \
		echo "📥 Installing nvm (Node Version Manager)..."; \
		curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash; \
	fi; \
	echo "📥 Installing Node.js 22 via nvm..."; \
	export NVM_DIR="$$HOME/.nvm"; \
	. "$$NVM_DIR/nvm.sh" && nvm install 22 && nvm use 22 && nvm alias default 22; \
	echo ""; \
	echo "✅ Node.js installed via nvm"; \
	. "$$NVM_DIR/nvm.sh" && echo "   Version: $$(node --version)"; \
	echo ""; \
	echo "⚠️  Note: nvm is a shell function. For this terminal session, run:"; \
	echo "   source ~/.bashrc   # or source ~/.nvm/nvm.sh"; \
	echo ""; \
	echo "   Or simply restart your terminal."
endif

install-pnpm:  ## Install pnpm package manager
	@echo "📦 Installing pnpm..."
	@if ! command -v npm >/dev/null 2>&1; then \
		echo "❌ npm not found. Install Node.js first with: make install-node"; \
		exit 1; \
	fi
	@if command -v pnpm >/dev/null 2>&1; then \
		echo "✅ pnpm already installed: $$(pnpm --version)"; \
	elif [ -x "$$HOME/.local/bin/pnpm" ]; then \
		echo "✅ pnpm already installed: $$($$HOME/.local/bin/pnpm --version) (in ~/.local/bin)"; \
	elif npm install -g pnpm 2>/dev/null; then \
		echo "✅ pnpm installed: $$(pnpm --version)"; \
	else \
		echo "   Global install failed (no sudo), installing to ~/.local ..."; \
		npm install --prefix "$$HOME/.local" -g pnpm; \
		mkdir -p "$$HOME/.local/bin"; \
		if [ -x "$$HOME/.local/bin/pnpm" ]; then \
			echo "✅ pnpm installed: $$($$HOME/.local/bin/pnpm --version) (in ~/.local/bin)"; \
			echo "⚠️  Add ~/.local/bin to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
		else \
			echo "❌ pnpm installation failed"; \
			exit 1; \
		fi; \
	fi

install-hatch:  ## Install Hatch for Python environment management
	@echo "📦 Installing Hatch..."
	@# Find a suitable Python 3.10+ interpreter
	@PYTHON_CMD=""; \
	for py in python3.13 python3.12 python3.11 python3.10 python3; do \
		if command -v $$py >/dev/null 2>&1; then \
			PY_MAJOR=$$($$py -c "import sys; print(sys.version_info.major)" 2>/dev/null); \
			PY_MINOR=$$($$py -c "import sys; print(sys.version_info.minor)" 2>/dev/null); \
			if [ "$$PY_MAJOR" = "3" ] && [ "$$PY_MINOR" -ge 10 ] 2>/dev/null; then \
				PYTHON_CMD=$$py; \
				break; \
			fi; \
		fi; \
	done; \
	if command -v hatch >/dev/null 2>&1; then \
		echo "✅ Hatch already installed: $$(hatch --version)"; \
	elif [ -x "$$HOME/.local/bin/hatch" ]; then \
		echo "✅ Hatch already installed: $$($$HOME/.local/bin/hatch --version) (in ~/.local/bin)"; \
		echo "⚠️  Add ~/.local/bin to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
	elif command -v pipx >/dev/null 2>&1; then \
		echo "Installing via pipx..."; \
		pipx install hatch; \
		echo "✅ Hatch installed"; \
		if command -v hatch >/dev/null 2>&1; then \
			echo "   Version: $$(hatch --version)"; \
		elif [ -x "$$HOME/.local/bin/hatch" ]; then \
			echo "   Version: $$($$HOME/.local/bin/hatch --version)"; \
			echo "⚠️  Run 'pipx ensurepath' and restart terminal to add to PATH"; \
		fi; \
	elif [ -n "$$PYTHON_CMD" ]; then \
		echo "📥 Installing Hatch (no pipx, using $$PYTHON_CMD)..."; \
		HATCH_INSTALLED=0; \
		if $$PYTHON_CMD -m pip install --user hatch 2>/dev/null; then \
			HATCH_INSTALLED=1; \
		else \
			echo "   pip --user failed, trying venv approach..."; \
			HATCH_VENV="$$HOME/.local/hatch-env"; \
			$$PYTHON_CMD -m venv "$$HATCH_VENV" 2>/dev/null && \
			"$$HATCH_VENV/bin/pip" install hatch 2>/dev/null && \
			mkdir -p "$$HOME/.local/bin" && \
			ln -sf "$$HATCH_VENV/bin/hatch" "$$HOME/.local/bin/hatch" && \
			HATCH_INSTALLED=1; \
		fi; \
		if [ "$$HATCH_INSTALLED" = "1" ] && [ -x "$$HOME/.local/bin/hatch" ]; then \
			echo "✅ Hatch installed: $$($$HOME/.local/bin/hatch --version)"; \
			echo "⚠️  Add ~/.local/bin to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
		else \
			echo "❌ Hatch installation failed."; \
			exit 1; \
		fi; \
	else \
		echo "❌ No suitable Python 3.10+ found and pipx not available."; \
		echo ""; \
		echo "Please install pipx or ensure Python 3.10+ is in PATH:"; \
		echo ""; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "  sudo apt-get install -y pipx"; \
			echo "  pipx ensurepath"; \
			echo "  source ~/.bashrc  # or restart terminal"; \
		elif [ "$(PKG_MANAGER)" = "dnf" ]; then \
			echo "  sudo dnf install -y pipx"; \
			echo "  pipx ensurepath"; \
		elif [ "$(OS)" = "macos" ]; then \
			echo "  brew install pipx"; \
			echo "  pipx ensurepath"; \
		else \
			echo "  python3 -m pip install --user pipx"; \
			echo "  pipx ensurepath"; \
		fi; \
		echo ""; \
		echo "Then run 'make install-hatch' again."; \
		exit 1; \
	fi

# ============================================================================
# Main Targets
# ============================================================================

# Default target
help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start (for a fresh machine):"
	@echo "  make setup-dev      - Set up development environment (auto-installs dependencies)"
	@echo "  make check-deps     - Check what dependencies are installed/missing"
	@echo ""
	@echo "Common workflows:"
	@echo "  make test-all       - Run all tests"
	@echo "  make check-all      - Run all quality checks"
	@echo "  make clean-all      - Clean all artifacts"
	@echo "  make format-all     - Format all code"
	@echo "  make viewer         - Start the viewer dev server"
	@echo "  luxar demo          - Generate demo + serve + open browser"
	@echo ""
	@echo "Demos:"
	@echo "  make install-demo-deps - Install the demo extras (demos + gsplats + io)"
	@echo "  luxar demo deps        - Report which demo dependencies are missing"
	@echo ""
	@echo "Optional accelerators:"
	@echo "  make install-rust     - Install Rust/WASM for viewer builds"
	@echo "  make setup-cuda     - Install CUDA dependencies + build extension"
	@echo "  make install-go       - Install Go for native launcher builds"
	@echo "  make build-launchers  - Build native launchers (luxar export --native)"
	@echo ""
	@echo "System: $(OS) (package manager: $(PKG_MANAGER))"
	@echo "Node.js requirement: $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"

# Installation
install-dev:  ## Install Luxar Python package in editable mode for development
	pip install -e .

install-demo-deps:  ## Install the demo extras (demos + gsplats + io)
# The gsplats extra carries torch. pip leaves an ALREADY-satisfied torch alone,
# so a CUDA build put in place by `make setup-cuda` (or a custom --index-url
# wheel) survives this target; only a torch-less env gets the PyPI default.
	@echo "📦 Installing optional demo dependencies (demos + gsplats + io extras)..."
	$(HATCH) run pip install -e ".[demos,gsplats,io]"
	@echo ""
	@$(HATCH) run luxar demo deps || true
	@echo ""
	@echo "ℹ️  Some demos need credentials or a manual download instead of a"
	@echo "   package — see 'luxar demo info <key>' (NEEDS column in 'luxar demo')."

# Code formatting (using Hatch)
format-python:  ## Format Python code with ruff
	$(HATCH) run format

format-typescript:  ## Format TypeScript code with prettier
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run format

format-rust:  ## Format Rust code with cargo fmt
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v cargo >/dev/null 2>&1; then \
		echo "⚠️  cargo not found - skipping Rust formatting"; \
		echo "   Run 'make install-rust' to install Rust"; \
	else \
		echo "🦀 Formatting Rust code..."; \
		cd packages/luxar-viewer/src/wasm/rust && cargo fmt; \
		echo "✅ Rust code formatted"; \
	fi

format-cuda:  ## Format CUDA/C++ code with clang-format (skips if not installed)
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "⚠️  clang-format not found - skipping CUDA formatting"; \
		echo "   Install with:"; \
		echo "     Ubuntu/Debian: sudo apt install clang-format"; \
		echo "     macOS: brew install clang-format"; \
	else \
		echo "🔧 Formatting CUDA/C++ code..."; \
		find $(CUDA_EXT_DIR) \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.hpp" \) -exec clang-format -i {} +; \
		echo "✅ CUDA/C++ code formatted"; \
	fi

format-go:  ## Format the Go launcher with gofmt
	@GO_BIN=$$(command -v go || echo "$(HOME)/.local/go/bin/go"); \
	if [ -x "$$GO_BIN" ] || command -v go >/dev/null 2>&1; then \
		echo "🐹 Formatting Go launcher..."; \
		(cd $(LAUNCHER_SRC_DIR) && "$$GO_BIN" fmt ./...); \
		echo "✅ Go code formatted"; \
	else \
		echo "⚠️  go not found - skipping Go formatting (run 'make install-go')"; \
	fi

format-all:  ## Format all code (Python, TypeScript, Rust, Go, CUDA)
	@echo "🐍 Formatting Python code..."
	$(MAKE) format-python
	@echo ""
	@echo "📘 Formatting TypeScript code..."
	$(MAKE) format-typescript
	@echo ""
	$(MAKE) format-rust
	@echo ""
	$(MAKE) format-go
	@echo ""
	$(MAKE) format-cuda

gen-contract:  ## Regenerate the Python + TS format-contract projections from contract.yaml
	@echo "📄 Regenerating format-contract projections (Python + TypeScript)..."
	$(HATCH) run gen-contract

gen-data-manifest:  ## Regenerate demos/data_manifest.json from the demos/data tree
	@echo "📄 Regenerating the demo-data manifest..."
	$(HATCH) run gen-data-manifest

# Code quality checks (using Hatch)
lint-python:  ## Run ruff linting on Python code
	$(HATCH) run python -m ruff check packages/luxar/src/luxar/

lint-typescript:  ## Run ESLint on TypeScript code
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run lint

type-check-python:  ## Run mypy type checking on Python code
	$(HATCH) run mypy packages/luxar/src/luxar/

type-check-typescript:  ## Run TypeScript type checking
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run typecheck

security:  ## Run bandit security checks
	$(HATCH) run bandit -r packages/luxar/src/luxar/ -c pyproject.toml

# Testing (using Hatch)
test-all:  ## Run all tests (Python, Rust/WASM, and TypeScript with fresh fixtures)
	@echo "🐍 Running Python tests..."
	$(HATCH) run test
	@echo ""
	@echo "🦀 Checking Rust/WASM tests..."
	@# Source cargo env to find cargo/wasm-pack
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v cargo >/dev/null 2>&1; then \
		echo "Running Rust unit tests..."; \
		(cd packages/luxar-viewer && pnpm test:wasm) || exit $$?; \
		echo ""; \
		if command -v wasm-pack >/dev/null 2>&1; then \
			echo "Building WASM module for TypeScript comparison tests..."; \
			(cd packages/luxar-viewer && pnpm build:wasm) || exit $$?; \
		else \
			echo "⚠️  wasm-pack not found - WASM comparison tests will be skipped"; \
			echo "   Run 'make install-rust' to enable full WASM testing"; \
		fi; \
	else \
		echo "⚠️  cargo not found - Rust/WASM tests will be skipped"; \
		echo "   Run 'make install-rust' to enable full WASM testing"; \
	fi
	@echo ""
	@echo "🔬 Generating TypeScript test fixtures..."
	$(HATCH) run python packages/luxar-viewer/tests/fixtures/generate_test_data.py
	@echo "📘 Running TypeScript tests..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	@# When WASM artifacts are present (build above succeeded, or a prior
	@# build is cached), require the WASM-vs-TypeScript artifact-presence
	@# meta-test to run instead of being silently skipped via runIf().
	@if [ -f "packages/luxar-viewer/public/wasm/luxar_wasm.js" ] \
	   && [ -f "packages/luxar-viewer/public/wasm/luxar_wasm_bg.wasm" ]; then \
		export LUXAR_REQUIRE_WASM_TESTS=1; \
		echo "🦀 WASM artifacts detected; LUXAR_REQUIRE_WASM_TESTS=1"; \
		cd packages/luxar-viewer && LUXAR_REQUIRE_WASM_TESTS=1 pnpm test --run; \
	else \
		cd packages/luxar-viewer && pnpm test --run; \
	fi
	@echo ""
	@echo "🎮 Checking CUDA tests..."
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		if $(HATCH) run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
			echo "Running CUDA extension tests..."; \
			$(HATCH) run pytest $(CUDA_EXT_DIR)/tests/ -v; \
		else \
			echo "⚠️  PyTorch CUDA not available - CUDA tests skipped"; \
			echo "   Run 'make check-cuda-deps' for details"; \
		fi; \
	else \
		echo "⚠️  CUDA extension not built - CUDA tests skipped"; \
		echo "   Run 'make setup-cuda' to enable CUDA testing"; \
	fi
	@echo ""
	@echo "🐹 Checking Go launcher tests..."
	@GO_BIN=$$(command -v go || echo "$(HOME)/.local/go/bin/go"); \
	if [ -x "$$GO_BIN" ] || command -v go >/dev/null 2>&1; then \
		echo "Running Go launcher unit tests..."; \
		(cd $(LAUNCHER_SRC_DIR) && "$$GO_BIN" test ./...) || exit $$?; \
	else \
		echo "⚠️  go not found - Go launcher tests skipped"; \
		echo "   Run 'make install-go' to enable launcher testing"; \
	fi

# GPU contention warning: do not run `test-python` and `test-cuda` in
# parallel processes — both invoke PyTorch/CUDA on the same device, which
# produces non-deterministic test failures (observed: ~200 spurious CUDA
# extension comparison failures when run concurrently). Run sequentially.
test-python:  ## Run Python tests only
	$(HATCH) run test

test-cov-python:  ## Run Python tests with coverage report
	$(HATCH) run test-cov

test-cov-all:  ## Run all tests with coverage (Python + TypeScript)
	@echo "🐍 Running Python tests with coverage..."
	$(HATCH) run test-cov
	@echo ""
	@echo "📘 Running TypeScript tests with coverage..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run test:coverage
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Coverage reports generated!"
	@echo ""
	@echo "📊 Python coverage: See terminal output above or run '$(HATCH) run coverage html'"
	@echo "📊 TypeScript coverage: packages/luxar-viewer/coverage/"

# Pre-commit
enable-pre-commit:  ## Enable and activate pre-commit hooks
	$(HATCH) run pre-commit install

run-pre-commit:  ## Run pre-commit on all files
	$(HATCH) run pre-commit run --all-files

# Quality checks (run all using Hatch)
check-all:  ## Run all quality checks (Python, TypeScript, Go)
	@echo "🐍 Running Python checks (ruff, mypy, import-linter, version, bandit, tests)..."
	$(HATCH) run check
	@echo "📘 Running TypeScript checks (CI: typecheck + lint + layers + coverage)..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run check:ci
	@echo "🐹 Running Go launcher checks (go vet)..."
	@GO_BIN=$$(command -v go || echo "$(HOME)/.local/go/bin/go"); \
	if [ -x "$$GO_BIN" ] || command -v go >/dev/null 2>&1; then \
		(cd $(LAUNCHER_SRC_DIR) && "$$GO_BIN" vet ./...) || exit $$?; \
		echo "✅ Go launcher vet passed"; \
	else \
		echo "⚠️  go not found - skipping Go vet (run 'make install-go')"; \
	fi

# Documentation checks
check-docs:  ## Check documentation quality and coverage
	@echo "📚 Checking Python documentation..."
	$(HATCH) run python scripts/check_documentation.py
	@echo "📚 Checking TypeScript JSDoc coverage..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && npx tsx scripts/check-jsdoc-coverage.ts --threshold=70

check-docs-verbose:  ## Check documentation with detailed output
	@echo "📚 Checking documentation (verbose mode)..."
	$(HATCH) run python scripts/check_documentation.py --verbose
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && npx tsx scripts/check-jsdoc-coverage.ts --threshold=70 --verbose

clean-docs:  ## Clean built documentation
	@echo "🧹 Cleaning documentation build artifacts..."
	rm -rf docs/_build/
	rm -rf docs/_autosummary/
	rm -rf packages/luxar-viewer/docs/api/
	@echo "✅ Documentation artifacts cleaned (Sphinx + TypeDoc)"

# Clean up
clean-all:  ## Clean all artifacts (Python, TypeScript, WASM, CUDA, launchers, datasets, cache)
	@echo "🧹 Cleaning all artifacts..."
	@echo ""
	$(MAKE) clean-python
	$(MAKE) clean-viewer
	$(MAKE) clean-wasm
	$(MAKE) clean-cuda
	$(MAKE) clean-launchers
	$(MAKE) clean-examples
	$(MAKE) clean-cache
	@echo ""
	@echo "✅ Clean complete!"

clean-python:  ## Clean Python build artifacts and caches
	@echo "🐍 Cleaning Python artifacts..."
	@# Prune node_modules/.git/.venv (never descend) and use `rm -rf` so removal
	@# of non-empty cache dirs can't fail the way `find -delete` does.
	find . $(FIND_PRUNE) -type f -name '*.pyc' -exec rm -f {} +
	find . $(FIND_PRUNE) -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . $(FIND_PRUNE) -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find . $(FIND_PRUNE) -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . $(FIND_PRUNE) -type d -name '.mypy_cache' -prune -exec rm -rf {} +
	find . $(FIND_PRUNE) -type d -name '.ruff_cache' -prune -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf coverage/
	rm -rf .coverage*

clean-viewer:  ## Clean viewer build artifacts (node_modules, dist, etc.)
	@echo "📘 Cleaning TypeScript/Node.js artifacts..."
	rm -rf packages/luxar-viewer/dist/
	rm -rf packages/luxar-viewer/node_modules/
	rm -rf packages/luxar-viewer/.vite/
	rm -rf packages/luxar-viewer/.parcel-cache/
	rm -f packages/luxar-viewer/*.tsbuildinfo
	rm -f packages/luxar-viewer/vite.config.*.timestamp-*

clean-cache:  ## Clear the Luxar user cache (~/.cache/luxar)
	@echo "🧹 Clearing Luxar cache..."
	rm -rf ~/.cache/luxar
	@echo "✅ Luxar cache cleared!"

clean-examples:  ## Clean up generated datasets (examples, demos, zarr files)
	@echo "🧹 Cleaning generated datasets..."
	rm -rf datasets/
	rm -rf zarr_scenes/  # Remove deprecated directory
	@# NOTE: deliberately no bare `rm -rf *.zarr` here — a root-level glob would
	@# silently delete a user's exported scene (.zarr is gitignored → unrecoverable).
	@# All generated datasets live under datasets/, removed above.
	@echo "✅ Datasets cleaned!"

clean-setup:  ## Remove ALL dev tools to simulate a fresh machine (USE WITH CAUTION)
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "⚠️  DEEP CLEAN - This will remove all development tools!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "⚠️  WARNING: Some tools removed by this target may be used by other projects!"
	@echo ""
	@echo "This target will remove:"
	@echo ""
	@echo "PROJECT-SPECIFIC (safe to remove):"
	@echo "  • Build artifacts         - .pyc, .egg-info, dist/, build/, __pycache__"
	@echo "  • node_modules/           - TypeScript dependencies (this project only)"
	@echo "  • Hatch virtual envs      - Python environments (luxar project only)"
	@echo "  • pre-commit hooks        - Git hooks (this repository only)"
	@echo "  • WASM build artifacts    - Compiled WASM files (this project only)"
	@echo "  • CUDA build artifacts    - CUDA extension .so files (this project only)"
	@echo ""
	@echo "SYSTEM-WIDE TOOLS (⚠️  may affect other projects):"
	@echo "  • python3-dev             - Python development headers (system package)"
	@echo "  • pnpm                    - Package manager (may be used by other projects)"
	@echo "  • Hatch                   - Python environment tool (may be used by other projects)"
	@echo "  • Rust toolchain          - rustup, cargo, rustc (may be used by other projects)"
	@echo "  • wasm-pack               - WASM build tool (may be used by other projects)"
	@echo "  • nvm + Node.js           - Node.js version manager (may be used by other projects)"
	@echo "  • pnpm cache              - Global package cache (~/.local/share/pnpm)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "💡 To reinstall after cleaning:"
	@echo "   make setup-dev      - Reinstall Node.js, pnpm, Hatch, and project deps"
	@echo "   make install-rust     - Reinstall Rust and wasm-pack"
	@echo "   make setup-cuda     - Reinstall python3-dev, PyTorch CUDA, and build extension"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@read -p "⚠️  Remove SYSTEM-WIDE tools (may affect other projects)? [y/N] " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo ""; \
		echo "Aborted. To remove only project-specific artifacts, use:"; \
		echo "  make clean-all      - Clean project artifacts only"; \
		echo "  make clean-python   - Clean Python artifacts"; \
		echo "  make clean-viewer   - Clean TypeScript artifacts"; \
		echo "  make clean-cuda     - Clean CUDA artifacts"; \
		echo "  make clean-wasm     - Clean WASM artifacts"; \
		echo "  make clean-cache    - Clear user cache (~/.cache/luxar)"; \
		exit 1; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🧹 Starting deep clean..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🧹 [Step 0] Cleaning all build artifacts first..."
	@echo ""
	@$(MAKE) clean-python 2>/dev/null || true
	@$(MAKE) clean-viewer 2>/dev/null || true
	@$(MAKE) clean-wasm 2>/dev/null || true
	@$(MAKE) clean-cuda 2>/dev/null || true
	@echo ""
	@echo "🧹 [1/12] Removing node_modules..."
	@rm -rf packages/luxar-viewer/node_modules
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [2/12] Removing Hatch environments (includes PyTorch with CUDA)..."
	@if command -v hatch >/dev/null 2>&1; then \
		$(HATCH) env prune -y 2>/dev/null || true; \
	fi
	@rm -rf ~/.local/share/hatch/env/virtual/luxar* 2>/dev/null || true
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [3/12] Removing pre-commit hooks..."
	@if [ -d ".git/hooks" ]; then \
		if command -v hatch >/dev/null 2>&1; then \
			$(HATCH) run pre-commit uninstall 2>/dev/null || true; \
			echo "   ✓ pre-commit hooks uninstalled"; \
		elif [ -f ".git/hooks/pre-commit" ] && grep -q "pre-commit" ".git/hooks/pre-commit" 2>/dev/null; then \
			rm -f .git/hooks/pre-commit .git/hooks/commit-msg .git/hooks/pre-push 2>/dev/null || true; \
			echo "   ✓ pre-commit hook files removed"; \
		else \
			echo "   ⚪ No pre-commit hooks found"; \
		fi; \
	else \
		echo "   ⚪ Not a git repository, skipping"; \
	fi
	@echo ""
	@echo "🧹 [4/12] Removing WASM build artifacts..."
	@rm -rf packages/luxar-viewer/public/wasm
	@rm -rf packages/luxar-viewer/src/wasm/rust/target
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [5/12] Removing CUDA build artifacts..."
	@rm -rf $(CUDA_EXT_DIR)/build/
	@rm -rf $(CUDA_EXT_DIR)/*.egg-info/
	@rm -f $(CUDA_EXT_DIR)/cuda_splatting_backend*.so
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [6/12] Removing python3-dev (system package)..."
	@if [ "$(PKG_MANAGER)" = "apt" ]; then \
		if dpkg -l | grep -q python3-dev; then \
			echo "   Found python3-dev, removing with sudo..."; \
			if sudo apt remove -y python3-dev; then \
				echo "   ✓ python3-dev removed"; \
			else \
				echo "   ⚠️  Failed to remove python3-dev (may require manual removal)"; \
			fi; \
		else \
			echo "   ⚪ python3-dev not installed, skipping"; \
		fi; \
	elif [ "$(OS)" = "macos" ]; then \
		echo "   ⚪ On macOS, Python dev headers are part of Python (skipping)"; \
	else \
		echo "   ⚪ Unsupported OS, skipping"; \
	fi
	@echo ""
	@echo "🧹 [7/12] Removing wasm-pack..."
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v cargo >/dev/null 2>&1 && command -v wasm-pack >/dev/null 2>&1; then \
		cargo uninstall wasm-pack 2>/dev/null || true; \
		echo "   ✓ Done"; \
	else \
		echo "   ⚪ Not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [8/12] Removing Rust toolchain..."
	@if command -v rustup >/dev/null 2>&1; then \
		rustup self uninstall -y 2>/dev/null || true; \
		echo "   ✓ Done"; \
	else \
		echo "   ⚪ Not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [9/12] Removing Hatch..."
	@if command -v pipx >/dev/null 2>&1 && pipx list 2>/dev/null | grep -q hatch; then \
		pipx uninstall hatch 2>/dev/null || true; \
		echo "   ✓ Removed via pipx"; \
	elif [ -x "$(HOME)/.local/bin/hatch" ]; then \
		rm -f "$(HOME)/.local/bin/hatch"; \
		echo "   ✓ Removed from ~/.local/bin"; \
	elif command -v hatch >/dev/null 2>&1; then \
		HATCH_PATH=$$(which hatch 2>/dev/null); \
		if echo "$$HATCH_PATH" | grep -q "brew\|Homebrew\|Cellar"; then \
			echo "   ⚠️  Hatch installed via Homebrew"; \
			echo "   Try: brew uninstall hatch"; \
		else \
			echo "   ⚠️  Hatch found at $$HATCH_PATH but cannot auto-remove"; \
			echo "   Try: pipx uninstall hatch (if installed via pipx)"; \
		fi; \
	else \
		echo "   ⚪ Not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [10/12] Removing pnpm (global package)..."
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if command -v pnpm >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then \
		echo "   Uninstalling pnpm..."; \
		npm uninstall -g pnpm 2>/dev/null || true; \
		echo "   ✓ pnpm uninstalled"; \
	else \
		echo "   ⚪ pnpm not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [11/12] Removing nvm and Node.js..."
	@NVM_REMOVED=0; \
	HOMEBREW_NODE=0; \
	if [ -d "$(HOME)/.nvm" ]; then \
		rm -rf "$(HOME)/.nvm"; \
		echo "   ✓ Removed ~/.nvm"; \
		echo "   ⚠️  You may want to remove nvm lines from ~/.bashrc or ~/.zshrc manually"; \
		NVM_REMOVED=1; \
	fi; \
	if command -v node >/dev/null 2>&1; then \
		NODE_PATH=$$(which node 2>/dev/null); \
		if echo "$$NODE_PATH" | grep -q "brew\|Homebrew\|Cellar"; then \
			echo "   ⚠️  Node.js installed via Homebrew (not removed automatically)"; \
			echo "   To remove: brew uninstall node"; \
			HOMEBREW_NODE=1; \
		fi; \
	fi; \
	if [ "$$NVM_REMOVED" = "0" ] && [ "$$HOMEBREW_NODE" = "0" ]; then \
		echo "   ⚪ nvm not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [12.5/12] Removing Go toolchain (if installed by us)..."
	@if [ -d "$$HOME/.local/go" ]; then \
		rm -rf "$$HOME/.local/go"; \
		echo "   ✓ Removed ~/.local/go"; \
	elif command -v go >/dev/null 2>&1; then \
		GO_PATH=$$(which go 2>/dev/null); \
		if echo "$$GO_PATH" | grep -q "brew\|Homebrew\|Cellar"; then \
			echo "   ⚠️  Go installed via Homebrew (not removed automatically)"; \
			echo "   To remove: brew uninstall go"; \
		else \
			echo "   ⚠️  Go found at $$GO_PATH — not removed (system or other manager)"; \
		fi; \
	else \
		echo "   ⚪ Go not installed, skipping"; \
	fi
	@$(MAKE) clean-launchers 2>/dev/null || true
	@echo ""
	@echo "🧹 [12/12] Removing pnpm cache..."
	@if [ -d "$(HOME)/.local/share/pnpm" ]; then \
		rm -rf "$(HOME)/.local/share/pnpm"; \
		echo "   ✓ Removed ~/.local/share/pnpm"; \
	else \
		echo "   ⚪ pnpm cache not found, skipping"; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Deep clean complete!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "📋 What was removed:"
	@echo "  ✅ All development tools (Hatch, Rust, Node.js, pnpm)"
	@echo "  ✅ All build artifacts (.pyc, .egg-info, dist/, node_modules/, etc.)"
	@echo "  ✅ Python development headers (python3-dev)"
	@echo ""
	@echo "⚠️  System packages NOT removed (may be used by other projects):"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "  • NVIDIA driver: $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"; \
	fi; \
	if command -v nvcc >/dev/null 2>&1; then \
		echo "  • nvidia-cuda-toolkit: $$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"; \
	fi; \
	if [ "$(PKG_MANAGER)" = "apt" ]; then \
		if dpkg -l | grep -q "^ii  build-essential"; then \
			echo "  • build-essential (C++ compiler, make, etc.)"; \
		fi; \
	fi; \
	if ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v nvcc >/dev/null 2>&1; then \
		echo "  (No CUDA-related packages found)"; \
	fi
	@echo ""
	@echo "To remove these system packages manually (⚠️  only if not needed elsewhere):"
	@if command -v nvcc >/dev/null 2>&1 || dpkg -l 2>/dev/null | grep -q "^ii  build-essential"; then \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "  sudo apt remove nvidia-cuda-toolkit build-essential"; \
		fi; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make check-deps' to verify the cleanup"
	@echo "  2. Run 'make setup-dev' to reinstall everything"
	@echo "  3. Run 'make install-rust' to reinstall Rust/WASM (optional)"
	@echo "  4. Run 'make setup-cuda' to reinstall CUDA support (optional)"
	@echo ""

# Development setup
setup-dev:  ## Complete development setup (auto-installs missing dependencies)
	@echo "🚀 Setting up development environment..."
	@echo ""
	@echo "System: $(OS) (package manager: $(PKG_MANAGER))"
	@echo ""
	@# Step 1: Check and install Python dependencies
	@echo "=== Step 1: Python Environment ==="
	@if ! command -v python3 >/dev/null 2>&1; then \
		echo "❌ Python3 not found. Please install Python 3.9+ first:"; \
		if [ "$(OS)" = "macos" ]; then \
			echo "   brew install python@3.11"; \
		elif [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "   sudo apt-get install python3 python3-pip python3-venv"; \
		fi; \
		exit 1; \
	fi
	@echo "✅ Python: $$(python3 --version)"
	@# Install/fix hatch (use pipx or pip --user fallback for HPC/no-sudo systems)
	@PYTHON_CMD=""; \
	for py in python3.13 python3.12 python3.11 python3.10 python3; do \
		if command -v $$py >/dev/null 2>&1; then \
			PY_MAJOR=$$($$py -c "import sys; print(sys.version_info.major)" 2>/dev/null); \
			PY_MINOR=$$($$py -c "import sys; print(sys.version_info.minor)" 2>/dev/null); \
			if [ "$$PY_MAJOR" = "3" ] && [ "$$PY_MINOR" -ge 10 ] 2>/dev/null; then \
				PYTHON_CMD=$$py; \
				break; \
			fi; \
		fi; \
	done; \
	if command -v hatch >/dev/null 2>&1; then \
		echo "✅ Hatch: $$(hatch --version)"; \
	elif [ -x "$$HOME/.local/bin/hatch" ]; then \
		echo "✅ Hatch: $$($$HOME/.local/bin/hatch --version) (in ~/.local/bin)"; \
		echo "⚠️  Note: Add ~/.local/bin to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
	elif command -v pipx >/dev/null 2>&1; then \
		echo "📥 Installing Hatch via pipx..."; \
		if pipx list 2>/dev/null | grep -q "package hatch"; then \
			echo "   Hatch found in pipx but symlink missing, reinstalling..."; \
			pipx reinstall hatch; \
		else \
			pipx install hatch; \
		fi; \
		if command -v hatch >/dev/null 2>&1; then \
			echo "✅ Hatch: $$(hatch --version)"; \
		elif [ -x "$$HOME/.local/bin/hatch" ]; then \
			echo "✅ Hatch: $$($$HOME/.local/bin/hatch --version) (in ~/.local/bin)"; \
			echo "⚠️  Note: Run 'pipx ensurepath' and restart terminal to add to PATH"; \
		else \
			echo "❌ Hatch installation failed. Try: pipx reinstall hatch"; \
			exit 1; \
		fi; \
	elif [ -n "$$PYTHON_CMD" ]; then \
		echo "📥 Installing Hatch (no pipx, using $$PYTHON_CMD)..."; \
		HATCH_INSTALLED=0; \
		if $$PYTHON_CMD -m pip install --user hatch 2>/dev/null; then \
			HATCH_INSTALLED=1; \
		else \
			echo "   pip --user failed, trying venv approach..."; \
			HATCH_VENV="$$HOME/.local/hatch-env"; \
			$$PYTHON_CMD -m venv "$$HATCH_VENV" 2>/dev/null && \
			"$$HATCH_VENV/bin/pip" install hatch 2>/dev/null && \
			mkdir -p "$$HOME/.local/bin" && \
			ln -sf "$$HATCH_VENV/bin/hatch" "$$HOME/.local/bin/hatch" && \
			HATCH_INSTALLED=1; \
		fi; \
		if [ "$$HATCH_INSTALLED" = "1" ] && [ -x "$$HOME/.local/bin/hatch" ]; then \
			echo "✅ Hatch: $$($$HOME/.local/bin/hatch --version) (in ~/.local/bin)"; \
			echo "⚠️  Note: Add ~/.local/bin to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
		else \
			echo "❌ Hatch installation failed."; \
			exit 1; \
		fi; \
	else \
		echo ""; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠️  pipx not found and no Python 3.10+ available for pip install."; \
		echo ""; \
		echo "Please install pipx or load a Python 3.10+ module, then re-run 'make setup-dev':"; \
		echo ""; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "  sudo apt-get install -y pipx"; \
			echo "  pipx ensurepath"; \
			echo "  source ~/.bashrc  # or restart terminal"; \
		elif [ "$(PKG_MANAGER)" = "dnf" ]; then \
			echo "  sudo dnf install -y pipx"; \
			echo "  pipx ensurepath"; \
		elif [ "$(OS)" = "macos" ]; then \
			echo "  brew install pipx"; \
			echo "  pipx ensurepath"; \
		else \
			echo "  python3 -m pip install --user pipx"; \
			echo "  pipx ensurepath"; \
		fi; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		exit 1; \
	fi
	@echo ""
	@# Step 2: Check Node.js (auto-install via nvm if needed - no sudo required)
	@echo "=== Step 2: Node.js Environment ==="
	@# Source nvm if available
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	NODE_OK=1; \
	if command -v node >/dev/null 2>&1; then \
		NODE_VERSION=$$(node -v | sed 's/v//'); \
		NODE_MAJOR=$$(echo $$NODE_VERSION | cut -d. -f1); \
		NODE_MINOR=$$(echo $$NODE_VERSION | cut -d. -f2); \
		if [ "$$NODE_MAJOR" -lt $(MIN_NODE_MAJOR) ] || \
		   ([ "$$NODE_MAJOR" -eq $(MIN_NODE_MAJOR) ] && [ "$$NODE_MINOR" -lt $(MIN_NODE_MINOR) ]); then \
			echo "⚠️  Node.js v$$NODE_VERSION is too old (need $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+)"; \
			NODE_OK=0; \
		else \
			echo "✅ Node.js: v$$NODE_VERSION"; \
		fi; \
	else \
		echo "❌ Node.js not found"; \
		NODE_OK=0; \
	fi; \
	if [ "$$NODE_OK" = "0" ]; then \
		if [ "$(OS)" = "macos" ]; then \
			echo "📥 Installing Node.js via Homebrew..."; \
			if ! command -v brew >/dev/null 2>&1; then \
				echo "📥 Installing Homebrew first..."; \
				/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
			fi; \
			brew install node@22 || brew upgrade node; \
			echo "✅ Node.js installed: $$(node --version)"; \
		else \
			if [ ! -s "$$NVM_DIR/nvm.sh" ]; then \
				echo "📥 Installing nvm (Node Version Manager)..."; \
				curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash; \
				export NVM_DIR="$$HOME/.nvm"; \
			fi; \
			echo "📥 Installing Node.js 22 via nvm..."; \
			. "$$NVM_DIR/nvm.sh" && nvm install 22 && nvm use 22 && nvm alias default 22; \
			echo "✅ Node.js installed: $$( . $$NVM_DIR/nvm.sh && node --version)"; \
		fi; \
	fi
	@# Check/install pnpm (source nvm first if needed)
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if command -v pnpm >/dev/null 2>&1; then \
		echo "✅ pnpm: $$(pnpm --version)"; \
	elif [ -x "$$HOME/.local/bin/pnpm" ]; then \
		echo "✅ pnpm: $$($$HOME/.local/bin/pnpm --version) (in ~/.local/bin)"; \
	else \
		echo "📥 Installing pnpm..."; \
		if npm install -g pnpm 2>/dev/null; then \
			echo "✅ pnpm: $$(pnpm --version)"; \
		else \
			echo "   Global install failed (no sudo), installing to ~/.local ..."; \
			npm install --prefix "$$HOME/.local" -g pnpm; \
			mkdir -p "$$HOME/.local/bin"; \
			if [ -x "$$HOME/.local/bin/pnpm" ]; then \
				echo "✅ pnpm: $$($$HOME/.local/bin/pnpm --version) (in ~/.local/bin)"; \
				echo "⚠️  Add ~/.local/bin to PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
			else \
				echo "❌ pnpm installation failed"; \
				exit 1; \
			fi; \
		fi; \
	fi
	@echo ""
	@# Step 3: Set up Python environment with Hatch
	@echo "=== Step 3: Creating Python Environment ==="
	@# Use hatch from PATH or ~/.local/bin
	@HATCH_CMD="$$(command -v hatch 2>/dev/null || echo $$HOME/.local/bin/hatch)"; \
	echo "Creating Hatch environment..."; \
	$$HATCH_CMD env create || true; \
	echo "Installing pre-commit hooks..."; \
	$$HATCH_CMD run pre-commit install || echo "⚠️  pre-commit install skipped"
	@echo ""
	@# Step 4: Install TypeScript dependencies (source nvm first if needed)
	@echo "=== Step 4: Installing TypeScript Dependencies ==="
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	cd packages/luxar-viewer && pnpm install
	@echo ""
	@# Step 5: Optional Rust/WASM setup prompt
	@echo "=== Step 5: Optional Accelerators ==="
	@echo ""
	@echo "WASM (viewer performance):"
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v wasm-pack >/dev/null 2>&1; then \
		echo "  ✅ Rust/WASM already configured"; \
	else \
		echo "  ⚪ Not installed (run 'make install-rust' to enable)"; \
	fi
	@echo ""
	@echo "CUDA (Gaussian splatting GPU acceleration):"
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "  ✅ CUDA toolkit installed"; \
		if $(HATCH) run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
			echo "  ✅ PyTorch CUDA available"; \
		else \
			echo "  ⚠️  PyTorch CUDA not available (run 'make check-cuda-deps' for details)"; \
		fi; \
	else \
		echo "  ⚪ CUDA toolkit not installed"; \
		echo "     For GPU-accelerated splatting, install CUDA toolkit and run 'make build-cuda'"; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Development environment setup complete!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Next steps:"
	@echo "  make check-all    - Verify everything works"
	@echo "  make viewer       - Start the viewer dev server"
	@echo "  make demo         - Generate a demo dataset"
	@echo ""
	@echo "Optional accelerators:"
	@echo "  make install-rust   - Enable WASM acceleration (viewer)"
	@echo "  make setup-cuda   - Install CUDA dependencies + build extension"
	@echo "  make install-go     - Install Go (for native launcher builds)"
	@echo "  make build-launchers - Build native launchers (luxar export --native)"
	@echo ""
	@echo "💡 Use '$(HATCH) shell' to activate the Python environment"

# Demo and serving
demo:  ## Generate the Lorenz demo dataset (datasets/demos/lorenz.luxar.zarr, 100k points)
	@mkdir -p datasets/demos
	$(HATCH) run luxar demo run lorenz -- --no-serve --points=100000
	@echo "✅ Demo dataset created at datasets/demos/lorenz.luxar.zarr"

run-examples:  ## Run all examples to generate zarr files (output to datasets/examples/)
	@echo "🚀 Running all examples to generate zarr files..."
	@echo "📂 Output directory: datasets/examples/"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@total=$$(ls -1 packages/luxar/examples/*_example.py 2>/dev/null | wc -l || echo 0); \
	count=0; \
	for script in packages/luxar/examples/*_example.py; do \
		count=$$((count + 1)); \
		name=$$(basename $$script); \
		echo ""; \
		echo "[$${count}/$${total}] 📊 Running $${name}..."; \
		echo "────────────────────────────────────────────────"; \
		if $(HATCH) run python $$script; then \
			echo "✅ Success: $${name}"; \
		else \
			echo "❌ Failed: $${name}"; \
		fi; \
	done
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ All examples completed!"
	@echo ""
	@echo "📁 Generated zarr files in datasets/examples/:"
	@for zarr in datasets/examples/*.zarr; do \
		if [ -d "$$zarr" ]; then \
			size=$$(du -sh "$$zarr" | cut -f1); \
			name=$$(basename "$$zarr"); \
			echo "   • $$name ($${size})"; \
		fi; \
	done 2>/dev/null || echo "   No .zarr files found"
	@echo ""
	@echo "💡 To browse the generated datasets, run:"
	@echo "   make serve-examples"

run-demos:  ## Generate ALL demo datasets (output to datasets/demos/)
	@# Registry-driven: `demo run-all` skips demos whose declared outputs
	@# already exist and those needing manual/Kaggle data. Output stems come
	@# from each demo's DEMO_META, so the stem≠filename demos are handled too.
	@mkdir -p datasets/demos
	$(HATCH) run luxar demo run-all --skip-existing
	@echo ""
	@echo "📁 Generated zarr files in datasets/demos/:"
	@for zarr in datasets/demos/*.zarr; do \
		if [ -d "$$zarr" ]; then \
			size=$$(du -sh "$$zarr" | cut -f1); \
			name=$$(basename "$$zarr"); \
			echo "   • $$name ($${size})"; \
		fi; \
	done 2>/dev/null || echo "   No .zarr files found"

generate-readme-demos:  ## Generate only the demo datasets needed for README screenshots
	@echo "🚀 Generating README demo datasets..."
	@mkdir -p datasets/demos
	@echo "[1/5] 🌀 Lorenz Attractor..."
	@if [ -d "datasets/demos/lorenz.luxar.zarr" ]; then echo "   ✓ exists"; else $(HATCH) run python packages/luxar/src/luxar/demos/demo_lorenz.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[2/5] 🔮 Mandelbulb..."
	@if [ -d "datasets/demos/mandelbulb.luxar.zarr" ]; then echo "   ✓ exists"; else $(HATCH) run python packages/luxar/src/luxar/demos/demo_mandelbulb.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[3/5] 🌌 Spiral Galaxy..."
	@if [ -d "datasets/demos/spiral_galaxy.luxar.zarr" ]; then echo "   ✓ exists"; else $(HATCH) run python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[4/5] 🧬 Zebrahub Multiome UMAP..."
	@if [ -d "datasets/demos/zebrahub_multiome_peak_umap.luxar.zarr" ]; then echo "   ✓ exists"; else $(HATCH) run python packages/luxar/src/luxar/demos/demo_zebrahub_multiome_peak_umap.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[5/5] 🌈 Rainbow Sphere..."
	@if [ -d "datasets/demos/rainbow_sphere.luxar.zarr" ]; then echo "   ✓ exists"; else $(HATCH) run python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "✅ README demos ready!"

generate-readme-images: generate-readme-demos  ## Generate README screenshots using Playwright
	@echo "📸 Generating README screenshots..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	@# Source nvm if available
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	cd packages/luxar-viewer && pnpm readme-images
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ README screenshots generated!"
	@echo ""
	@echo "📁 Generated images in docs/images/readme/:"
	@ls -la docs/images/readme/*.png 2>/dev/null || echo "   No images found"
	@echo ""
	@echo "💡 Commit these images to include them in the README"

generate-doc-images: generate-readme-demos  ## Generate documentation screenshots using Playwright
	@echo "📸 Generating documentation screenshots..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	@# Source nvm if available
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	cd packages/luxar-viewer && pnpm doc-images
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Documentation screenshots generated!"
	@echo ""
	@echo "📁 Generated images in docs/images/docs/:"
	@ls -la docs/images/docs/*.png 2>/dev/null || echo "   No images found"

generate-readme-videos: generate-readme-demos  ## Generate README videos (GIF/WebP) using Playwright
	@echo "🎬 Generating README videos..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@# Check for ffmpeg
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "❌ ffmpeg is required for video conversion"; \
		echo "   Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"; \
		exit 1; \
	fi
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	@# Source nvm if available
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	cd packages/luxar-viewer && pnpm readme-videos
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ README videos generated!"
	@echo ""
	@echo "📁 Generated videos in docs/images/readme/:"
	@ls -la docs/images/readme/*.gif docs/images/readme/*.webp 2>/dev/null || echo "   No videos found"
	@echo ""
	@echo "💡 Commit these videos to include them in the README"

generate-gallery-datasets:  ## Generate the demo datasets for the gallery harness (idempotent)
	@echo "🖼️  Generating gallery demo datasets..."
	@mkdir -p datasets/demos
	$(HATCH) run python scripts/gallery/generate_gallery_datasets.py $(if $(ONLY),--only $(ONLY),)

generate-gallery: generate-gallery-datasets  ## Capture still + orbit video for every gallery demo → docs/images/gallery/
	@echo "🎥 Capturing gallery stills + orbit videos..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if ! command -v ffmpeg >/dev/null 2>&1; then \
		echo "❌ ffmpeg is required for video conversion"; \
		echo "   Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"; \
		exit 1; \
	fi
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then . "$$NVM_DIR/nvm.sh"; fi; \
	cd packages/luxar-viewer && $(if $(ONLY),GALLERY_ONLY=$(ONLY) ,)pnpm gallery
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Gallery media generated in docs/images/gallery/"
	@ls -la docs/images/gallery/*.png 2>/dev/null | head || echo "   No media found"
	@echo ""
	@echo "💡 Review docs/images/gallery/, then wire the best into README.md"

serve-examples:  ## Serve the datasets directory for browsing generated datasets
	@echo "🌐 Serving datasets/ directory at http://localhost:8000"
	@echo "📊 Open viewer at: http://localhost:5173/?src=http://localhost:8000"
	@echo "💡 Press 'O' in the viewer to browse available datasets"
	@echo ""
	$(HATCH) run luxar serve datasets/ -p 8000

# Default values for serve-dataset (override with: make serve-dataset DATASET=path/to/data.zarr PORT=8080)
DATASET ?= datasets/demos/lorenz.luxar.zarr
PORT ?= 8000

serve-dataset:  ## Serve a dataset (default: datasets/demos/lorenz.luxar.zarr, port: 8000)
	@if [ ! -d "datasets/demos/lorenz.luxar.zarr" ]; then \
		echo "No demo dataset found. Creating one..."; \
		$(MAKE) demo; \
	fi
	$(HATCH) run luxar serve $(DATASET) -p $(PORT)

# Web viewer
install-viewer-deps:  ## Install viewer dependencies (node_modules)
	cd packages/luxar-viewer && pnpm install

viewer:  ## Start the web viewer development server
	@# Source nvm if available
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if command -v node >/dev/null 2>&1; then \
		NODE_VERSION=$$(node -v | sed 's/v//'); \
		NODE_MAJOR=$$(echo $$NODE_VERSION | cut -d. -f1); \
		NODE_MINOR=$$(echo $$NODE_VERSION | cut -d. -f2); \
		if [ "$$NODE_MAJOR" -lt $(MIN_NODE_MAJOR) ] || \
		   ([ "$$NODE_MAJOR" -eq $(MIN_NODE_MAJOR) ] && [ "$$NODE_MINOR" -lt $(MIN_NODE_MINOR) ]); then \
			echo "❌ Node.js v$$NODE_VERSION is too old. Vite requires $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"; \
			echo "   Run 'make install-node' to upgrade, or 'make setup-dev' for full setup."; \
			exit 1; \
		fi; \
	else \
		echo "❌ Node.js not found. Run 'make setup-dev' first."; \
		exit 1; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found. Run 'make setup-dev' first."; \
		exit 1; \
	fi; \
	if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi; \
	cd packages/luxar-viewer && pnpm dev

build-viewer:  ## Build the viewer for production (auto-installs Rust/wasm-pack if needed)
	@# Source nvm and check Node.js version first
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if command -v node >/dev/null 2>&1; then \
		NODE_VERSION=$$(node -v | sed 's/v//'); \
		NODE_MAJOR=$$(echo $$NODE_VERSION | cut -d. -f1); \
		NODE_MINOR=$$(echo $$NODE_VERSION | cut -d. -f2); \
		if [ "$$NODE_MAJOR" -lt $(MIN_NODE_MAJOR) ] || \
		   ([ "$$NODE_MAJOR" -eq $(MIN_NODE_MAJOR) ] && [ "$$NODE_MINOR" -lt $(MIN_NODE_MINOR) ]); then \
			echo "❌ Node.js v$$NODE_VERSION is too old. Vite requires $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"; \
			echo "   Run 'make install-node' to upgrade, or 'make setup-dev' for full setup."; \
			exit 1; \
		fi; \
	else \
		echo "❌ Node.js not found. Run 'make setup-dev' first."; \
		exit 1; \
	fi
	@# Check for wasm-pack, install if needed (separate command to ensure Make waits)
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "⚠️  wasm-pack not found. Installing Rust/WASM toolchain..."; \
		echo ""; \
		$(MAKE) install-rust; \
	fi
	@# Build viewer (source nvm for pnpm; build-wasm.sh sources cargo env itself)
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	echo "🦀 Building viewer with Rust/WASM support..."; \
	cd packages/luxar-viewer && pnpm build

rebuild-viewer:  ## Complete clean rebuild of viewer (auto-installs dependencies as needed)
	@echo "🧹 Cleaning viewer build artifacts..."
	@rm -rf packages/luxar-viewer/dist/
	@rm -rf packages/luxar-viewer/.vite/
	@rm -f packages/luxar-viewer/*.tsbuildinfo
	@rm -f packages/luxar-viewer/vite.config.*.timestamp-*
	@# Source nvm and check Node.js version
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if command -v node >/dev/null 2>&1; then \
		NODE_VERSION=$$(node -v | sed 's/v//'); \
		NODE_MAJOR=$$(echo $$NODE_VERSION | cut -d. -f1); \
		NODE_MINOR=$$(echo $$NODE_VERSION | cut -d. -f2); \
		if [ "$$NODE_MAJOR" -lt $(MIN_NODE_MAJOR) ] || \
		   ([ "$$NODE_MAJOR" -eq $(MIN_NODE_MAJOR) ] && [ "$$NODE_MINOR" -lt $(MIN_NODE_MINOR) ]); then \
			echo "❌ Node.js v$$NODE_VERSION is too old. Vite requires $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"; \
			echo "   Run 'make install-node' to upgrade, or 'make setup-dev' for full setup."; \
			exit 1; \
		fi; \
	else \
		echo "❌ Node.js not found. Run 'make setup-dev' first."; \
		exit 1; \
	fi
	@# Check for wasm-pack, install if needed (separate command to ensure Make waits)
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "⚠️  wasm-pack not found. Installing Rust/WASM toolchain..."; \
		echo ""; \
		$(MAKE) install-rust; \
	fi
	@# Reinstall dependencies and build (source nvm for pnpm)
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	echo "📦 Reinstalling dependencies..."; \
	cd packages/luxar-viewer && pnpm install; \
	echo "🦀 Building viewer with Rust/WASM support..."; \
	cd $(CURDIR) && cd packages/luxar-viewer && pnpm build; \
	echo "✅ Viewer rebuild complete!"

# WASM/Rust setup and build
install-rust:  ## Install Rust and wasm-pack for WASM development
	@# This must be a SINGLE shell command so PATH updates persist after Rust install
	@echo "🦀 Setting up Rust/WASM development environment..."; \
	echo ""; \
	CARGO_ENV="$$HOME/.cargo/env"; \
	if [ -f "$$CARGO_ENV" ]; then \
		. "$$CARGO_ENV"; \
	fi; \
	if command -v rustc >/dev/null 2>&1; then \
		echo "✅ Rust is already installed: $$(rustc --version)"; \
		echo "🔄 Updating Rust to latest stable..."; \
		rustup update stable; \
		rustup default stable; \
	else \
		echo "📥 Installing Rust via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable; \
		echo ""; \
		echo "✅ Rust installed! Loading environment..."; \
		if [ -f "$$CARGO_ENV" ]; then \
			. "$$CARGO_ENV"; \
		else \
			echo "❌ Error: Rust installed but $$CARGO_ENV not found"; \
			exit 1; \
		fi; \
	fi; \
	echo ""; \
	echo "🔧 Checking wasm-pack installation..."; \
	if command -v wasm-pack >/dev/null 2>&1; then \
		echo "✅ wasm-pack is already installed: $$(wasm-pack --version)"; \
	else \
		echo "📥 Installing wasm-pack 0.14.0 (this may take a minute)..."; \
		cargo install wasm-pack --version 0.14.0 --locked; \
		echo "✅ wasm-pack installed successfully!"; \
	fi; \
	echo ""; \
	echo "✅ Rust/WASM development environment ready!"; \
	echo "   Rust version: $$(rustc --version)"; \
	echo "   Cargo version: $$(cargo --version)"; \
	echo "   wasm-pack version: $$(wasm-pack --version)"; \
	echo ""; \
	echo "💡 Run 'make test-wasm' to test the Rust code"; \
	echo "💡 Run 'make build-wasm' to compile the WASM module"

build-wasm:  ## Build the WASM module (requires Rust + wasm-pack)
	@# Source nvm and cargo env to ensure pnpm and wasm-pack are in PATH
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "❌ wasm-pack not found."; \
		echo ""; \
		echo "Run 'make install-rust' to install Rust and wasm-pack."; \
		echo ""; \
		exit 1; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found."; \
		echo ""; \
		echo "Run 'make setup-dev' to install Node.js and pnpm."; \
		echo ""; \
		exit 1; \
	fi; \
	echo "🦀 Building WASM module..."; \
	cd packages/luxar-viewer && pnpm build:wasm && \
	echo "✅ WASM module built successfully!"

test-wasm:  ## Run Rust unit tests for WASM module
	@# Source nvm and cargo env to ensure pnpm and cargo are in PATH
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v cargo >/dev/null 2>&1; then \
		echo "❌ cargo not found."; \
		echo ""; \
		echo "Run 'make install-rust' to install Rust."; \
		echo ""; \
		exit 1; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found."; \
		echo ""; \
		echo "Run 'make setup-dev' to install Node.js and pnpm."; \
		echo ""; \
		exit 1; \
	fi; \
	echo "🧪 Running Rust tests..."; \
	cd packages/luxar-viewer && pnpm test:wasm && \
	echo "✅ All Rust tests passed!"

benchmark-wasm:  ## Run WASM vs TypeScript performance benchmarks
	@# Source nvm and cargo env to ensure pnpm is in PATH
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found."; \
		echo ""; \
		echo "Run 'make setup-dev' to install Node.js and pnpm."; \
		echo ""; \
		exit 1; \
	fi; \
	if [ ! -f "packages/luxar-viewer/public/wasm/luxar_wasm_bg.wasm" ]; then \
		echo "⚠️  WASM module not built. Building first..."; \
		echo ""; \
		$(MAKE) build-wasm; \
		echo ""; \
	fi; \
	echo "🚀 Running WASM benchmarks..."; \
	cd packages/luxar-viewer && pnpm bench:wasm

clean-wasm:  ## Clean WASM build artifacts
	@echo "🧹 Cleaning WASM artifacts..."
	rm -rf packages/luxar-viewer/public/wasm/
	rm -rf packages/luxar-viewer/src/wasm/rust/target/
	@echo "✅ WASM artifacts cleaned!"

# ============================================================================
# Native launchers (luxar export --native)
# ============================================================================
#
# A small Go program in packages/luxar-launcher/ produces the per-OS
# binaries that back `luxar export --native macos|linux`. Source is checked
# in; binaries are built locally and gitignored under
# packages/luxar/src/luxar/cli/_launchers/.

LAUNCHER_SRC_DIR := packages/luxar-launcher
LAUNCHER_OUT_DIR := packages/luxar/src/luxar/cli/_launchers

install-go:  ## Install Go toolchain (no sudo: brew on macOS, official tarball on Linux)
	@# Single shell command so PATH updates persist within the recipe.
	@echo "🐹 Setting up Go toolchain..."; \
	echo ""; \
	if command -v go >/dev/null 2>&1; then \
		echo "✅ Go is already installed: $$(go version)"; \
		exit 0; \
	fi; \
	if [ -x "$$HOME/.local/go/bin/go" ]; then \
		echo "✅ Go is already installed: $$($$HOME/.local/go/bin/go version) (in ~/.local/go)"; \
		echo "⚠️  Add ~/.local/go/bin to PATH: export PATH=\"$$HOME/.local/go/bin:$$PATH\""; \
		exit 0; \
	fi; \
	if [ "$(OS)" = "macos" ]; then \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "📥 Installing Homebrew first..."; \
			/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		fi; \
		echo "📥 Installing Go via Homebrew..."; \
		brew install go; \
		echo "✅ Go installed: $$(go version)"; \
	elif [ "$(OS)" = "linux" ]; then \
		GO_VERSION=$${GO_VERSION:-1.22.10}; \
		ARCH=$$(uname -m); \
		case "$$ARCH" in \
			x86_64|amd64) GOARCH=amd64 ;; \
			aarch64|arm64) GOARCH=arm64 ;; \
			*) echo "❌ Unsupported Linux architecture: $$ARCH"; exit 1 ;; \
		esac; \
		TARBALL="go$${GO_VERSION}.linux-$${GOARCH}.tar.gz"; \
		echo "📥 Installing Go $$GO_VERSION for linux-$$GOARCH (no sudo, into ~/.local/go)..."; \
		mkdir -p "$$HOME/.local"; \
		rm -rf "$$HOME/.local/go"; \
		TMPDIR_GO=$$(mktemp -d); \
		trap 'rm -rf "$$TMPDIR_GO"' EXIT; \
		curl -fsSL "https://go.dev/dl/$$TARBALL" -o "$$TMPDIR_GO/$$TARBALL"; \
		tar -C "$$HOME/.local" -xzf "$$TMPDIR_GO/$$TARBALL"; \
		echo "✅ Go installed: $$($$HOME/.local/go/bin/go version)"; \
		echo "⚠️  Add ~/.local/go/bin to PATH: export PATH=\"$$HOME/.local/go/bin:$$PATH\""; \
	else \
		echo "❌ Unsupported OS: $(OS). Install Go manually from https://go.dev/dl/"; \
		exit 1; \
	fi

build-launchers:  ## Build native launchers for the host platform (requires Go + CGO)
	@# The launcher embeds the viewer in a system WebView (WKWebView /
	@# WebView2 / WebKitGTK), so it requires CGO at build time and the
	@# matching system library at runtime. CGO breaks pure-Go cross-
	@# compilation: each target OS must be built on a host of that OS
	@# (or with a CGO cross-toolchain like Zig). This target builds for
	@# the host OS only; CI will produce the other binaries on their
	@# respective runners. Set LUXAR_LAUNCHER_NO_WEBVIEW=1 at runtime if
	@# you want the launcher to open the default browser instead.
	@GO_BIN=""; \
	if command -v go >/dev/null 2>&1; then \
		GO_BIN=go; \
	elif [ -x "$$HOME/.local/go/bin/go" ]; then \
		GO_BIN="$$HOME/.local/go/bin/go"; \
	else \
		echo "❌ Go not found. Run 'make install-go' first."; \
		exit 1; \
	fi; \
	echo "🐹 Building native launcher with $$GO_BIN ($$($$GO_BIN version | sed 's/^go version //'))"; \
	mkdir -p $(LAUNCHER_OUT_DIR); \
	cd $(LAUNCHER_SRC_DIR); \
	if [ "$(OS)" = "macos" ]; then \
		echo "  • darwin/arm64 (CGO=1, WKWebView)..."; \
		GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 $$GO_BIN build -trimpath -ldflags="-s -w" -o ../../$(LAUNCHER_OUT_DIR)/darwin-arm64 .; \
		echo "  • darwin/amd64 (CGO=1, WKWebView)..."; \
		if ! GOOS=darwin GOARCH=amd64 CGO_ENABLED=1 $$GO_BIN build -trimpath -ldflags="-s -w" -o ../../$(LAUNCHER_OUT_DIR)/darwin-amd64 .; then \
			echo ""; \
			echo "❌ darwin/amd64 build failed (likely missing universal SDK)."; \
			echo ""; \
			echo "   The arm64-only launcher at $(LAUNCHER_OUT_DIR)/darwin-arm64"; \
			echo "   would NOT run on Intel Macs. Refusing to silently mislabel"; \
			echo "   it as 'darwin-universal'."; \
			echo ""; \
			echo "   Either install Xcode's full universal SDK and re-run, or"; \
			echo "   ship the arm64-only binary explicitly via your CI matrix."; \
			rm -f $(LAUNCHER_OUT_DIR)/darwin-arm64; \
			exit 1; \
		fi; \
		cd - >/dev/null; \
		echo "  • lipo darwin universal..."; \
		lipo -create -output $(LAUNCHER_OUT_DIR)/darwin-universal \
			$(LAUNCHER_OUT_DIR)/darwin-arm64 \
			$(LAUNCHER_OUT_DIR)/darwin-amd64; \
		rm -f $(LAUNCHER_OUT_DIR)/darwin-arm64 $(LAUNCHER_OUT_DIR)/darwin-amd64; \
		echo "  • verify universal..."; \
		lipo -info $(LAUNCHER_OUT_DIR)/darwin-universal | grep -q "x86_64 arm64\|arm64 x86_64" \
			|| { echo "❌ lipo verification failed"; rm -f $(LAUNCHER_OUT_DIR)/darwin-universal; exit 1; }; \
		echo ""; \
		echo "ℹ️  Linux + Windows binaries: build on a Linux/Windows host (CGO blocks pure cross-compile)"; \
	elif [ "$(OS)" = "linux" ]; then \
		ARCH=$$(uname -m); \
		case "$$ARCH" in \
			x86_64|amd64) GOARCH=amd64 ;; \
			aarch64|arm64) GOARCH=arm64 ;; \
			*) echo "❌ Unsupported Linux architecture: $$ARCH"; exit 1 ;; \
		esac; \
		echo "  • linux/$$GOARCH (CGO=1, WebKitGTK)..."; \
		echo "    Requires: libwebkit2gtk-4.1-dev (or 4.0-dev on older distros)"; \
		GOOS=linux GOARCH=$$GOARCH CGO_ENABLED=1 $$GO_BIN build -trimpath -ldflags="-s -w" -o ../../$(LAUNCHER_OUT_DIR)/linux-$$GOARCH .; \
		cd - >/dev/null; \
	else \
		echo "❌ Unsupported host OS: $(OS)"; \
		exit 1; \
	fi; \
	echo ""; \
	echo "✅ Native launchers built:"; \
	ls -lh $(LAUNCHER_OUT_DIR) | awk 'NR>1 && $$NF != "README.md" && $$NF != ".gitignore" {printf "   %-22s %s\n", $$NF, $$5}'

clean-launchers:  ## Clean native launcher binaries
	@echo "🧹 Cleaning native launcher binaries..."
	@find $(LAUNCHER_OUT_DIR) -maxdepth 1 -type f \
		! -name 'README.md' ! -name '.gitignore' -delete 2>/dev/null || true
	@echo "✅ Launcher binaries cleaned!"

# ============================================================================
# CUDA Backend (Gaussian Splatting)
# ============================================================================

# Path to CUDA extension directories
CUDA_EXT_DIR := packages/luxar/src/luxar/gsplats/models/gsplats/cuda
NLM_CUDA_DIR := packages/luxar/src/luxar/gsplats/preprocessing/cuda

# Slurm parameters for 'make build-cuda SLURM=1'
# Override any of these on the command line, e.g.:
#   make build-cuda SLURM=1 SLURM_PARTITION=gpu CUDA_MODULE=cuda/12.8.0_570.86.10
SLURM           ?= 0
SLURM_PARTITION ?= gpu
SLURM_ACCOUNT   ?=
SLURM_QOS       ?=
SLURM_TIME      ?= 01:00:00
# CUDA_MODULE: auto = detect from torch.version.cuda; or e.g. cuda/12.8.0_570.86.10
CUDA_MODULE     ?= auto

# Guard: exit early on macOS where CUDA is not supported
define CHECK_MACOS_CUDA
	@if [ "$(OS)" = "macos" ]; then \
		echo "❌ CUDA is not supported on macOS."; \
		echo ""; \
		echo "   NVIDIA dropped CUDA support for macOS after toolkit 10.2 (2020)."; \
		echo "   To build/run CUDA extensions, use a Linux machine with an NVIDIA GPU."; \
		exit 1; \
	fi
endef

setup-cuda:  ## Install CUDA dependencies (may require sudo for system packages)
	$(CHECK_MACOS_CUDA)
	@echo "🔧 Setting up CUDA development environment..."
	@echo ""
	@# Step 1: Check/install system dependencies (may need sudo)
	@echo "=== Step 1: System Dependencies ==="
	@NEED_SUDO=0; \
	MISSING=""; \
	if ! command -v nvidia-smi >/dev/null 2>&1; then \
		MISSING="$$MISSING nvidia-driver"; \
		NEED_SUDO=1; \
	else \
		echo "✅ NVIDIA driver already installed"; \
		{ nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs -I {} echo "   Driver version: {}"; } || true; \
	fi; \
	if ! command -v nvcc >/dev/null 2>&1; then \
		MISSING="$$MISSING cuda-toolkit"; \
		NEED_SUDO=1; \
	else \
		echo "✅ CUDA toolkit already installed"; \
	fi; \
	if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then \
		MISSING="$$MISSING build-essential"; \
		NEED_SUDO=1; \
	else \
		echo "✅ C++ compiler already installed"; \
	fi; \
	PYTHON_INCLUDE=$$($(HATCH) run python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo ""); \
	if [ -n "$$PYTHON_INCLUDE" ] && [ -f "$$PYTHON_INCLUDE/Python.h" ]; then \
		echo "✅ Python development headers already installed"; \
	else \
		echo "❌ Python development headers not found"; \
		MISSING="$$MISSING python-dev"; \
		NEED_SUDO=1; \
	fi; \
	if [ "$$NEED_SUDO" = "1" ]; then \
		echo ""; \
		echo "⚠️  Some system packages need to be installed (requires sudo):"; \
		echo ""; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			CMD=""; \
			NEEDS_DRIVER=0; \
			NEEDS_CUDA=0; \
			NEEDS_BUILD=0; \
			NEEDS_PYTHONDEV=0; \
			for pkg in $$MISSING; do \
				case $$pkg in \
					nvidia-driver) NEEDS_DRIVER=1;; \
					cuda-toolkit) NEEDS_CUDA=1;; \
					build-essential) NEEDS_BUILD=1;; \
					python-dev) NEEDS_PYTHONDEV=1;; \
				esac; \
			done; \
			if [ "$$NEEDS_DRIVER" = "1" ]; then \
				echo "   NVIDIA Driver:"; \
				echo "     sudo ubuntu-drivers autoinstall   # Recommended - auto-selects best driver"; \
				echo "     # OR manually: sudo apt install nvidia-driver-535"; \
			fi; \
			if [ "$$NEEDS_PYTHONDEV" = "1" ]; then \
				if [ "$(PKG_MANAGER)" = "apt" ]; then \
					echo "   Python development headers (auto-installing):"; \
					echo "     sudo apt update && sudo apt install -y python3-dev"; \
					echo ""; \
					echo "📥 Installing python3-dev..."; \
					if sudo apt update && sudo apt install -y python3-dev; then \
						echo "✅ python3-dev installed successfully"; \
					else \
						echo ""; \
						echo "❌ Failed to install python3-dev. Please run manually:"; \
						echo "   sudo apt install python3-dev"; \
						echo ""; \
						echo "Then re-run 'make setup-cuda'"; \
						exit 1; \
					fi; \
				elif [ "$(OS)" = "macos" ]; then \
					echo ""; \
					echo "⚠️  On macOS, Python dev headers are typically included with Python."; \
					echo "   If you installed Python via Homebrew:"; \
					echo "     brew reinstall python@3.12"; \
					echo "   Or install Xcode Command Line Tools:"; \
					echo "     xcode-select --install"; \
					read -p "Try reinstalling Python? [y/N] " confirm; \
					if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
						brew reinstall python@3.12 || true; \
					fi; \
				else \
					echo ""; \
					echo "⚠️  Unsupported package manager. Please install Python development headers manually."; \
					exit 1; \
				fi; \
			fi; \
			APT_PKGS=""; \
			if [ "$$NEEDS_CUDA" = "1" ] || [ "$$NEEDS_BUILD" = "1" ]; then \
				if [ "$$NEEDS_CUDA" = "1" ]; then APT_PKGS="$$APT_PKGS nvidia-cuda-toolkit"; fi; \
				if [ "$$NEEDS_BUILD" = "1" ]; then APT_PKGS="$$APT_PKGS build-essential"; fi; \
				echo "   Other packages:"; \
				echo "     sudo apt-get update && sudo apt-get install -y$$APT_PKGS"; \
			fi; \
			if [ "$$NEEDS_DRIVER" = "1" ] || [ "$$NEEDS_CUDA" = "1" ] || [ "$$NEEDS_BUILD" = "1" ]; then \
				echo ""; \
				read -p "Run these commands now? [y/N] " confirm; \
				if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
					if [ "$$NEEDS_DRIVER" = "1" ]; then \
						echo "Installing NVIDIA driver..."; \
						sudo ubuntu-drivers autoinstall; \
					fi; \
					if [ -n "$$APT_PKGS" ]; then \
						echo "Installing other packages..."; \
						sudo apt-get update && sudo apt-get install -y$$APT_PKGS; \
					fi; \
				else \
					echo ""; \
					echo "Skipped. Please install manually and re-run 'make setup-cuda'."; \
					exit 1; \
				fi; \
			fi; \
		else \
			echo "   Please install the following packages manually:"; \
			for pkg in $$MISSING; do echo "   - $$pkg"; done; \
			echo ""; \
			echo "   Then re-run 'make setup-cuda'."; \
			exit 1; \
		fi; \
	fi
	@echo ""
	@# Step 2: Install PyTorch with CUDA (no sudo needed)
	@echo "=== Step 2: PyTorch with CUDA ==="
	@if $(HATCH) run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
		TORCH_CUDA=$$($(HATCH) run python -c "import torch; print(torch.version.cuda)" 2>/dev/null); \
		echo "✅ PyTorch with CUDA $$TORCH_CUDA already installed"; \
	else \
		CUDA_VER=$$(nvcc --version 2>/dev/null | grep release | sed 's/.*release //; s/,.*//' || echo "12.8"); \
		CUDA_MAJOR=$$(echo $$CUDA_VER | cut -d. -f1); \
		CUDA_MINOR=$$(echo $$CUDA_VER | cut -d. -f2); \
		CUDA_NUM=$$(( $$CUDA_MAJOR * 10 + $$CUDA_MINOR )); \
		if [ $$CUDA_NUM -ge 128 ]; then CUDA_TAG="cu128"; \
		elif [ $$CUDA_NUM -ge 124 ]; then CUDA_TAG="cu124"; \
		elif [ $$CUDA_NUM -ge 121 ]; then CUDA_TAG="cu121"; \
		elif [ $$CUDA_NUM -ge 118 ]; then CUDA_TAG="cu118"; \
		else CUDA_TAG="cu118"; fi; \
		echo "📥 Installing PyTorch with CUDA support (system CUDA $$CUDA_VER -> PyTorch index $$CUDA_TAG)..."; \
		echo ""; \
		$(HATCH) run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$$CUDA_TAG; \
		echo ""; \
		if $(HATCH) run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
			echo "✅ PyTorch with CUDA installed successfully"; \
		else \
			echo "⚠️  PyTorch installed but CUDA not available"; \
			echo "   This may happen if NVIDIA driver is not properly installed."; \
		fi; \
	fi
	@echo ""
	@# Step 3: Build the extension
	@echo "=== Step 3: Build CUDA Extension ==="
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		echo "✅ CUDA extension already built"; \
	else \
		if command -v nvcc >/dev/null 2>&1 && $(HATCH) run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
			echo "Building CUDA extension..."; \
			$(HATCH) run pip install -q ninja 2>/dev/null || true; \
			$(HATCH) run python $(CUDA_EXT_DIR)/build.py; \
		else \
			echo "⚠️  Cannot build - prerequisites not satisfied"; \
			echo "   Run 'make check-cuda-deps' for details"; \
		fi; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		echo "✅ CUDA setup complete!"; \
		echo ""; \
		echo "Next steps:"; \
		echo "  make test-cuda      - Verify installation"; \
		echo "  make benchmark-cuda - Run performance benchmarks"; \
	else \
		echo "⚠️  CUDA setup incomplete - run 'make check-cuda-deps' for details"; \
	fi
	@echo ""

check-cuda-deps:  ## Check CUDA development dependencies
	$(CHECK_MACOS_CUDA)
	@echo "🔍 Checking CUDA dependencies..."
	@echo ""
	@echo "=== 1. CUDA Toolkit ==="
	@if command -v nvcc >/dev/null 2>&1; then \
		CUDA_VERSION=$$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//'); \
		echo "✅ CUDA toolkit: $$CUDA_VERSION"; \
		echo "   Location: $$(which nvcc)"; \
	else \
		echo "❌ CUDA toolkit not found (nvcc not in PATH)"; \
		echo ""; \
		echo "   Installation options:"; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "   Ubuntu/Debian (recommended):"; \
			echo "     sudo apt install nvidia-cuda-toolkit"; \
			echo ""; \
			echo "   Or for latest version:"; \
		fi; \
		echo "     https://developer.nvidia.com/cuda-downloads"; \
		echo ""; \
		echo "   After installing, ensure nvcc is in PATH:"; \
		echo "     export PATH=/usr/local/cuda/bin:\$$PATH"; \
	fi
	@echo ""
	@echo "=== 2. NVIDIA GPU Driver ==="
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		if nvidia-smi >/dev/null 2>&1; then \
			DRIVER_VERSION=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true); \
			GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true); \
			echo "✅ NVIDIA driver: $$DRIVER_VERSION"; \
			echo "   GPU: $$GPU_NAME"; \
		else \
			echo "⚠️  nvidia-smi found but GPU not accessible (driver not loaded?)"; \
			if command -v sbatch >/dev/null 2>&1; then \
				echo "   On HPC: load the driver module or run on a GPU node"; \
			else \
				echo "   Check that NVIDIA drivers are properly installed"; \
			fi; \
		fi; \
	else \
		echo "❌ NVIDIA driver not found (nvidia-smi not available)"; \
		echo ""; \
		echo "   Install NVIDIA drivers:"; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "     sudo ubuntu-drivers autoinstall   # Recommended - auto-selects best driver"; \
			echo "     # OR manually: sudo apt install nvidia-driver-535"; \
		else \
			echo "     https://www.nvidia.com/drivers"; \
		fi; \
	fi
	@echo ""
	@echo "=== 3. PyTorch with CUDA ==="
	@$(HATCH) run python -c "import sys; import torch; print('✅ PyTorch:', torch.__version__); cuda_available = torch.cuda.is_available(); print('✅ PyTorch CUDA:', torch.version.cuda if cuda_available else 'not available'); (print('   GPU:', torch.cuda.get_device_name(0)) if cuda_available else None); (print('   Compute capability:', str(torch.cuda.get_device_capability()[0]) + '.' + str(torch.cuda.get_device_capability()[1])) if cuda_available else None)" 2>/dev/null || \
	$(HATCH) run python -c "print('❌ PyTorch not installed'); print(''); print('   Install PyTorch with CUDA:'); print('     $(HATCH) run pip install torch --index-url https://download.pytorch.org/whl/cu128')" 2>/dev/null || \
	echo "❌ Could not check PyTorch (hatch environment issue)"
	@echo ""
	@echo "=== 4. Python Development Headers ==="
	@PYTHON_INCLUDE=$$($(HATCH) run python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo ""); \
	if [ -n "$$PYTHON_INCLUDE" ] && [ -f "$$PYTHON_INCLUDE/Python.h" ]; then \
		PYTHON_VERSION=$$($(HATCH) run python -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>/dev/null); \
		echo "✅ Python development headers: $$PYTHON_VERSION"; \
		echo "   Location: $$PYTHON_INCLUDE"; \
	else \
		echo "❌ Python development headers not found"; \
		echo ""; \
		echo "   Install development headers:"; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "     sudo apt install python3-dev"; \
		elif [ "$(OS)" = "macos" ]; then \
			echo "     brew reinstall python@3.12"; \
			echo "     # or: xcode-select --install"; \
		fi; \
	fi
	@echo ""
	@echo "=== 5. C++ Compiler ==="
	@if command -v g++ >/dev/null 2>&1; then \
		echo "✅ g++: $$(g++ --version | head -1)"; \
	elif command -v clang++ >/dev/null 2>&1; then \
		echo "✅ clang++: $$(clang++ --version | head -1)"; \
	else \
		echo "❌ C++ compiler not found"; \
		echo ""; \
		echo "   Install build essentials:"; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "     sudo apt install build-essential"; \
		elif [ "$(OS)" = "macos" ]; then \
			echo "     xcode-select --install"; \
		fi; \
	fi
	@echo ""
	@echo "=== 6. CUDA Extension Status ==="
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		SO_FILE=$$(ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so | head -1 || true); \
		echo "✅ CUDA extension built: $$(basename $$SO_FILE)"; \
		echo "   Size: $$(du -h $$SO_FILE | cut -f1)"; \
	else \
		echo "⚪ CUDA extension not built"; \
		echo "   Run 'make build-cuda' after dependencies are satisfied"; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@# Summary - check if extension can be built
	@NVCC_OK=0; DRIVER_OK=0; TORCH_OK=0; EXT_OK=0; \
	if command -v nvcc >/dev/null 2>&1; then NVCC_OK=1; fi; \
	if command -v nvidia-smi >/dev/null 2>&1; then DRIVER_OK=1; fi; \
	if ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then EXT_OK=1; fi; \
	if [ "$$EXT_OK" = "1" ]; then \
		echo "✅ CUDA extension is ready to use!"; \
		echo ""; \
		echo "   make test-cuda      - Run tests"; \
		echo "   make benchmark-cuda - Run benchmarks"; \
	elif [ "$$NVCC_OK" = "1" ] && [ "$$DRIVER_OK" = "1" ]; then \
		echo "✅ System dependencies OK. PyTorch with CUDA may need setup."; \
		echo ""; \
		echo "   Run 'make setup-cuda' to install PyTorch CUDA and build extension."; \
	else \
		echo "⚠️  Some dependencies missing - see above for installation instructions."; \
		echo ""; \
		echo "   Run 'make setup-cuda' to install dependencies."; \
	fi
	@echo ""

build-cuda:  ## Build the CUDA splatting extension  [SLURM=1 to build on a GPU node via Slurm]
	$(CHECK_MACOS_CUDA)
	@if [ "$(SLURM)" = "1" ]; then \
		echo "🚀 Submitting CUDA build to Slurm (partition: $(SLURM_PARTITION))..."; \
		echo ""; \
		HATCH_CMD="$$(command -v hatch 2>/dev/null || echo $$HOME/.local/bin/hatch)"; \
		$$HATCH_CMD run python scripts/build_cuda_slurm.py \
			--partition "$(SLURM_PARTITION)" \
			--cuda-module "$(CUDA_MODULE)" \
			$(if $(SLURM_ACCOUNT),--account "$(SLURM_ACCOUNT)") \
			$(if $(SLURM_QOS),--qos "$(SLURM_QOS)") \
			--time "$(SLURM_TIME)"; \
	else \
		echo "🔧 Building CUDA splatting extension..."; \
		echo ""; \
		if command -v sbatch >/dev/null 2>&1; then \
			echo "   💡 On an HPC cluster without a GPU on the login node, use:"; \
			echo "        make build-cuda SLURM=1"; \
			echo "        make build-cuda SLURM=1 SLURM_PARTITION=gpu"; \
			echo ""; \
		fi; \
		if ! command -v nvcc >/dev/null 2>&1; then \
			echo "❌ CUDA toolkit not found (nvcc not in PATH)"; \
			echo ""; \
			if command -v sbatch >/dev/null 2>&1 || type module >/dev/null 2>&1; then \
				echo "   On an HPC system, load the CUDA module first:"; \
				echo "     module load cuda/12.8.0_570.86.10   # match your PyTorch CUDA version"; \
				echo "     make build-cuda"; \
				echo ""; \
				echo "   Or build on a GPU node automatically:"; \
				echo "     make build-cuda SLURM=1"; \
			else \
				echo "   Install the CUDA toolkit:"; \
				echo "     https://developer.nvidia.com/cuda-downloads"; \
				echo ""; \
				echo "   After installing, ensure nvcc is in PATH:"; \
				echo "     export PATH=/usr/local/cuda/bin:\$$PATH"; \
			fi; \
			echo ""; \
			echo "   Run 'make check-cuda-deps' for a full diagnosis."; \
			exit 1; \
		fi; \
		if ! nvidia-smi >/dev/null 2>&1; then \
			echo "❌ GPU not accessible (nvidia-smi failed)"; \
			echo ""; \
			if command -v sbatch >/dev/null 2>&1; then \
				echo "   You are likely on a login node without direct GPU access."; \
				echo "   Build on a GPU node via Slurm:"; \
				echo "     make build-cuda SLURM=1"; \
				echo "     make build-cuda SLURM=1 SLURM_PARTITION=gpu"; \
			else \
				echo "   No NVIDIA GPU detected. Ensure you have:"; \
				echo "     - An NVIDIA GPU installed"; \
				echo "     - NVIDIA drivers installed (https://www.nvidia.com/drivers)"; \
			fi; \
			echo ""; \
			exit 1; \
		fi; \
		if ! $(HATCH) run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
			echo "❌ PyTorch with CUDA support not available"; \
			echo ""; \
			TORCH_CUDA=$$($(HATCH) run python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown"); \
			echo "   PyTorch CUDA version: $$TORCH_CUDA"; \
			if command -v sbatch >/dev/null 2>&1 || type module >/dev/null 2>&1; then \
				echo "   Did you load the matching CUDA module?"; \
				echo "     module load cuda/$$TORCH_CUDA.x  (find exact name with: module spider cuda)"; \
				echo ""; \
			fi; \
			echo "   Or reinstall PyTorch with CUDA:"; \
			echo "     $(HATCH) run pip install torch --index-url https://download.pytorch.org/whl/cu128"; \
			echo ""; \
			echo "   Run 'make check-cuda-deps' for more details."; \
			exit 1; \
		fi; \
		echo "✅ Prerequisites OK"; \
		echo ""; \
		$(HATCH) run pip install -q ninja 2>/dev/null || true; \
		echo "Building extension (this may take a few minutes)..."; \
		echo ""; \
		$(HATCH) run python $(CUDA_EXT_DIR)/build.py; \
		echo ""; \
		if ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
			echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
			echo "✅ CUDA extension built successfully!"; \
			SO_FILE=$$(ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so | head -1 || true); \
			echo "   Output: $$(basename $$SO_FILE)"; \
			echo ""; \
			echo "Next steps:"; \
			echo "  make test-cuda      - Run tests to verify"; \
			echo "  make benchmark-cuda - Run performance benchmarks"; \
		else \
			echo "❌ Build may have failed - .so file not found"; \
			echo "   Check the build output above for errors."; \
			exit 1; \
		fi; \
	fi

build-cuda-slurm:  ## Submit CUDA extension build as a Slurm job (alias for make build-cuda SLURM=1)
	@$(MAKE) build-cuda SLURM=1 \
		SLURM_PARTITION="$(SLURM_PARTITION)" \
		SLURM_ACCOUNT="$(SLURM_ACCOUNT)" \
		SLURM_QOS="$(SLURM_QOS)" \
		SLURM_TIME="$(SLURM_TIME)" \
		CUDA_MODULE="$(CUDA_MODULE)"

clean-cuda:  ## Clean CUDA build artifacts
	@echo "🧹 Cleaning CUDA build artifacts..."
	rm -rf $(CUDA_EXT_DIR)/build/
	rm -rf $(CUDA_EXT_DIR)/*.egg-info/
	rm -f $(CUDA_EXT_DIR)/cuda_splatting_backend*.so
	rm -rf $(CUDA_EXT_DIR)/__pycache__/
	@echo "✅ CUDA artifacts cleaned!"

# GPU contention warning: do not run `test-cuda` concurrently with
# `test-python` (Python test suite includes CUDA tests as a subset). Both
# claim the same GPU and produce non-deterministic comparison failures.
# Run sequentially.
test-cuda:  ## Run CUDA extension tests
	$(CHECK_MACOS_CUDA)
	@echo "🧪 Running CUDA extension tests..."
	@echo ""
	@# Check if extension is built and up-to-date
	@if ! ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		echo "⚠️  CUDA extension not built. Building first..."; \
		$(MAKE) build-cuda; \
		echo ""; \
	elif [ -n "$$(find $(CUDA_EXT_DIR)/src/ \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \) -newer $$(ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so | head -1) 2>/dev/null)" ]; then \
		echo "⚠️  CUDA source files changed since last build. Rebuilding..."; \
		$(MAKE) build-cuda; \
		echo ""; \
	fi
	@# Run tests
	$(HATCH) run pytest $(CUDA_EXT_DIR)/tests/ -v
	@echo ""
	@echo "✅ CUDA tests completed!"

benchmark-metal:  ## Run Metal (MPS) performance benchmarks (M-series only)
	@echo "Running Metal performance benchmarks..."
	@mkdir -p benchmarks
	$(HATCH) run python scripts/benchmarks/benchmark_metal_optimizations.py \
		--label baseline-$$(date +%Y%m%d-%H%M%S) \
		--output benchmarks/metal_$$(date +%Y%m%d-%H%M%S).json
	@echo ""
	@echo "Benchmark completed! Results in benchmarks/"

benchmark-metal-stress:  ## Run Metal RSS leak-check (validates MET-1 @autoreleasepool)
	@echo "Running Metal stress / leak-check..."
	@mkdir -p benchmarks
	$(HATCH) run python scripts/benchmarks/benchmark_metal_optimizations.py \
		--stress --label stress-$$(date +%Y%m%d-%H%M%S) \
		--output benchmarks/metal_stress_$$(date +%Y%m%d-%H%M%S).json

benchmark-cuda:  ## Run CUDA performance benchmarks
	$(CHECK_MACOS_CUDA)
	@echo "🚀 Running CUDA performance benchmarks..."
	@echo ""
	@# Check if extension is built and up-to-date
	@if ! ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so 1>/dev/null 2>&1; then \
		echo "⚠️  CUDA extension not built. Building first..."; \
		$(MAKE) build-cuda; \
		echo ""; \
	elif [ -n "$$(find $(CUDA_EXT_DIR)/src/ \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' \) -newer $$(ls $(CUDA_EXT_DIR)/cuda_splatting_backend*.so | head -1) 2>/dev/null)" ]; then \
		echo "⚠️  CUDA source files changed since last build. Rebuilding..."; \
		$(MAKE) build-cuda; \
		echo ""; \
	fi
	@# Run benchmark
	$(HATCH) run python $(CUDA_EXT_DIR)/benchmark.py
	@echo ""
	@echo "✅ Benchmark completed!"

# ============================================================================
# NLM CUDA Extension (Non-Local Means denoising)
# ============================================================================

build-nlm-cuda:  ## Build the NLM CUDA denoising extension
	$(CHECK_MACOS_CUDA)
	@echo "🔧 Building NLM CUDA extension..."
	@echo ""
	@if ! command -v nvcc >/dev/null 2>&1; then \
		echo "❌ CUDA toolkit not found (nvcc not in PATH)"; \
		exit 1; \
	fi
	@if ! nvidia-smi >/dev/null 2>&1; then \
		echo "❌ GPU not accessible (nvidia-smi failed)"; \
		exit 1; \
	fi
	$(HATCH) run pip install -q ninja 2>/dev/null || true
	$(HATCH) run python $(NLM_CUDA_DIR)/build.py
	@echo ""
	@echo "✅ NLM CUDA extension built!"

clean-nlm-cuda:  ## Clean NLM CUDA build artifacts
	@echo "🧹 Cleaning NLM CUDA build artifacts..."
	rm -rf $(NLM_CUDA_DIR)/build/
	rm -f $(NLM_CUDA_DIR)/nlm_cuda_backend*.so
	rm -f $(NLM_CUDA_DIR)/nlm_build_info.json
	rm -rf $(NLM_CUDA_DIR)/__pycache__/
	@echo "✅ NLM CUDA artifacts cleaned!"

test-nlm-cuda:  ## Run NLM CUDA extension tests
	@echo "🧪 Running NLM CUDA tests..."
	@if ! ls $(NLM_CUDA_DIR)/nlm_cuda_backend*.so 1>/dev/null 2>&1; then \
		echo "⚠️  NLM CUDA extension not built. Building first..."; \
		$(MAKE) build-nlm-cuda; \
		echo ""; \
	fi
	$(HATCH) run pytest packages/luxar/src/luxar/gsplats/preprocessing/tests/test_nlm_cuda.py -v
	@echo ""
	@echo "✅ NLM CUDA tests completed!"

test-fixtures:  ## Generate test fixtures for TypeScript tests
	@echo "🔬 Generating test fixtures..."
	$(HATCH) run python packages/luxar-viewer/tests/fixtures/generate_test_data.py

test-viewer-fixtures: test-fixtures  ## Generate fixtures + run TypeScript tests
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm test --run

test-viewer:  ## Run TypeScript tests (without regenerating fixtures)
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm test --run

test-cov-typescript:  ## Run TypeScript tests with coverage
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run test:coverage

test-e2e:  ## Run Playwright E2E tests
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm test:e2e

check-typescript:  ## Run all TypeScript checks (typecheck, lint, test)
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run check

check-rust:  ## Run Rust type/lint checks (cargo check + clippy)
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v cargo >/dev/null 2>&1; then \
		echo "❌ cargo not found."; \
		echo "   Run 'make install-rust' to install Rust."; \
		exit 1; \
	fi; \
	echo "🦀 Running Rust checks..."; \
	cd packages/luxar-viewer/src/wasm/rust && cargo check && cargo clippy -- -D warnings; \
	echo "✅ Rust checks passed!"

check-wasm-deps:  ## Check WASM development dependencies (Rust, wasm-pack)
	@echo "🔍 Checking WASM development dependencies..."
	@echo ""
	@echo "=== Rust Toolchain ==="
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v rustc >/dev/null 2>&1; then \
		echo "✅ Rust: $$(rustc --version)"; \
	elif command -v rustup >/dev/null 2>&1; then \
		echo "⚠️  rustup installed but no toolchain (run: rustup default stable)"; \
	else \
		echo "❌ Rust not installed (run 'make install-rust')"; \
	fi
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v cargo >/dev/null 2>&1; then \
		echo "✅ Cargo: $$(cargo --version)"; \
	else \
		echo "❌ Cargo not found"; \
	fi
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v wasm-pack >/dev/null 2>&1; then \
		echo "✅ wasm-pack: $$(wasm-pack --version)"; \
	else \
		echo "❌ wasm-pack not installed (run 'make install-rust')"; \
	fi
	@echo ""
	@echo "=== WASM Build Status ==="
	@if [ -f "packages/luxar-viewer/public/wasm/luxar_wasm_bg.wasm" ]; then \
		SIZE=$$(du -h packages/luxar-viewer/public/wasm/luxar_wasm_bg.wasm | cut -f1); \
		echo "✅ WASM module built ($$SIZE)"; \
	else \
		echo "⚪ WASM module not built (run 'make build-wasm')"; \
	fi
	@echo ""

# Documentation
build-typedoc:  ## Generate TypeScript API documentation with TypeDoc
	@echo "📘 Generating TypeScript API documentation..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	cd packages/luxar-viewer && pnpm typedoc
	@# Copy TypeDoc output into Sphinx build for unified docs
	@mkdir -p docs/_build/html/api/viewer
	@cp -r packages/luxar-viewer/docs/api/* docs/_build/html/api/viewer/ 2>/dev/null || true
	@echo "✅ TypeScript API docs generated at docs/_build/html/api/viewer/"

build-docs: generate-doc-images  ## Build documentation with Sphinx (auto-generates screenshots + TypeDoc)
	$(HATCH) run docs:build
	@# Build TypeDoc after Sphinx so we can copy into the output
	@$(MAKE) build-typedoc

serve-docs:  ## Serve documentation locally
	$(HATCH) run docs:serve

# Project statistics
stats:  ## Generate project statistics report (HTML + Markdown)
	@echo "📊 Analyzing project codebase..."
	$(HATCH) run python stats/generate_stats.py
	@echo "✅ Reports generated:"
	@echo "   - stats/project_stats.html  (styled, open in a browser)"
	@echo "   - stats/PROJECT_STATS.md    (GitHub-friendly, linked from README.md)"
	@echo "💡 Quick view: open stats/project_stats.html"

stats-fast:  ## Generate project statistics without running tests (file counts only)
	@echo "📊 Analyzing project codebase (no tests)..."
	$(HATCH) run python stats/generate_stats.py --no-tests
	@echo "✅ Reports generated: stats/project_stats.html, stats/PROJECT_STATS.md"

# Hatch environment management
show-env:  ## Show all Hatch environments
	$(HATCH) env show

prune-env:  ## Remove unused Hatch environments
	$(HATCH) env prune

shell:  ## Enter Hatch development shell
	$(HATCH) shell

# Building and publishing
# ------------------------
# Releases are TAG-TRIGGERED: `make release` pushes a v<version> tag and
# .github/workflows/publish.yml builds + publishes to PyPI via OIDC trusted
# publishing on a clean Linux runner. Never publish from a dev machine — that
# would ship a wheel with the wrong/missing viewer and a stray host launcher
# binary, and bypass OIDC. See scripts/release.sh for the full preflight.
.PHONY: build set-version release-check release publish publish-test

build: build-viewer  ## Build wheel + sdist (builds the viewer first so it is bundled)
	$(HATCH) build

set-version:  ## Set release version in code (DATE=YYYY.MM.DD, default today); commit via PR
	python3 scripts/set_version.py $(DATE)

release-check:  ## Dry-run release: run ALL preflight checks, tag/push nothing
	bash scripts/release.sh --dry-run

release:  ## Cut release: validate main + CI green, then tag v<version> and push (triggers PyPI publish)
	bash scripts/release.sh

publish publish-test:  ## DISABLED — use `make release` (tag-triggered OIDC publish). See scripts/release.sh
	@echo "❌ 'make $@' is disabled. Luxar publishes via a tag-triggered GitHub"; \
	echo "   Actions workflow using PyPI trusted publishing (OIDC) — not local uploads."; \
	echo "   A local 'hatch publish' would ship a wheel with NO viewer and your host"; \
	echo "   launcher binary baked in, and bypass OIDC entirely."; \
	echo; \
	echo "   To release:"; \
	echo "     make set-version        # bump CalVer, then open a PR and merge to main"; \
	echo "     make release-check      # dry-run preflight (safe)"; \
	echo "     make release            # tag + push -> CI builds & publishes"; \
	exit 1
