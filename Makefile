# Makefile for Luxar development tasks
# Uses Hatch for on-demand environment management
#
# This Makefile is designed to work on fresh Linux/macOS machines with minimal
# pre-installed tools. Run 'make setup-dev' to automatically install all dependencies.
#
.PHONY: help install-python format-python format-typescript format-rust format-cuda format-all \
        lint-python lint-typescript type-check-python type-check-typescript security \
        test-all test-python test-cov-python test-cov-typescript test-fixtures test-wasm test-viewer test-viewer-fixtures test-e2e \
        clean-all clean-viewer clean-examples clean-setup install-pre-commit run-pre-commit \
        check-all check-typescript check-rust check-wasm-deps setup-dev \
        check-docs check-docs-verbose clean-docs build-docs serve-docs \
        demo run-demos run-examples serve-examples serve-dataset install-viewer viewer build-viewer rebuild-viewer \
        setup-rust build-wasm clean-wasm generate-readme-demos generate-readme-images generate-readme-videos \
        stats show-env prune-env shell build publish-test publish \
        check-deps install-node install-pnpm install-hatch \
        setup-cuda check-cuda-deps build-cuda clean-cuda test-cuda benchmark-cuda

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

# Minimum Node.js version required by Vite 7.x
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
		echo "⚪ Rust not installed (run 'make setup-rust' if needed)"; \
	fi
	@# wasm-pack
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v wasm-pack >/dev/null 2>&1; then \
		echo "✅ wasm-pack: $$(wasm-pack --version)"; \
	else \
		echo "⚪ wasm-pack not installed (run 'make setup-rust' if needed)"; \
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
	@if hatch run python -c "import torch; print('✅ PyTorch CUDA:', torch.version.cuda if torch.cuda.is_available() else 'not available')" 2>/dev/null; then \
		:; \
	else \
		echo "⚪ PyTorch CUDA not available"; \
	fi
	@# CUDA extension build status
	@if ls packages/luxar/src/luxar/gsplats/models/gsplats/cuda/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
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
	npm install -g pnpm
	@echo "✅ pnpm installed: $$(pnpm --version)"

install-hatch:  ## Install Hatch for Python environment management
	@echo "📦 Installing Hatch..."
	@# Check if already installed
	@if command -v hatch >/dev/null 2>&1; then \
		echo "✅ Hatch already installed: $$(hatch --version)"; \
	elif [ -x "$$HOME/.local/bin/hatch" ]; then \
		echo "✅ Hatch already installed: $$($$HOME/.local/bin/hatch --version)"; \
		echo "⚠️  Run 'pipx ensurepath' and restart terminal to add to PATH"; \
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
	else \
		echo "❌ pipx not found."; \
		echo ""; \
		echo "Modern Ubuntu/Debian requires pipx for installing Python CLI tools."; \
		echo "Please install pipx first:"; \
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
	@echo "  make dev-setup      - Set up development environment (auto-installs dependencies)"
	@echo "  make check-deps     - Check what dependencies are installed/missing"
	@echo ""
	@echo "Common workflows:"
	@echo "  make test           - Run all tests"
	@echo "  make viewer         - Start the viewer dev server"
	@echo "  luxar demo          - Generate demo + serve + open browser"
	@echo "  make run-examples   - Generate all example datasets"
	@echo ""
	@echo "Optional accelerators:"
	@echo "  make setup-rust     - Install Rust/WASM for viewer builds"
	@echo "  make setup-cuda     - Install CUDA dependencies + build extension"
	@echo ""
	@echo "System: $(OS) (package manager: $(PKG_MANAGER))"
	@echo "Node.js requirement: $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"

# Installation
install-python:  ## Install the Python package in editable mode
	pip install -e .

# Code formatting (using Hatch)
format-python:  ## Format Python code with ruff
	hatch run format

format-typescript:  ## Format TypeScript code with prettier
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run format

format-all:  ## Format all code (Python and TypeScript)
	@echo "🐍 Formatting Python code..."
	hatch run format
	@echo "📘 Formatting TypeScript code..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run format

# Code quality checks (using Hatch)
lint-python:  ## Run ruff linting on Python code
	hatch run python -m ruff check packages/luxar/src/luxar/

lint-typescript:  ## Run ESLint on TypeScript code
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run lint

type-check-python:  ## Run mypy type checking on Python code
	hatch run mypy packages/luxar/src/luxar/

type-check-typescript:  ## Run TypeScript type checking
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run typecheck

security:  ## Run bandit security checks
	hatch run bandit -r packages/luxar/src/luxar/ -c pyproject.toml

# Testing (using Hatch)
test-all:  ## Run all tests (Python, Rust/WASM, and TypeScript with fresh fixtures)
	@echo "🐍 Running Python tests..."
	hatch run test
	@echo ""
	@echo "🦀 Checking Rust/WASM tests..."
	@# Source cargo env to find cargo/wasm-pack
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v cargo >/dev/null 2>&1; then \
		echo "Running Rust unit tests..."; \
		cd packages/luxar-viewer && pnpm test:wasm; \
		echo ""; \
		if command -v wasm-pack >/dev/null 2>&1; then \
			echo "Building WASM module for TypeScript comparison tests..."; \
			cd packages/luxar-viewer && pnpm build:wasm; \
		else \
			echo "⚠️  wasm-pack not found - WASM comparison tests will be skipped"; \
			echo "   Run 'make setup-rust' to enable full WASM testing"; \
		fi; \
	else \
		echo "⚠️  cargo not found - Rust/WASM tests will be skipped"; \
		echo "   Run 'make setup-rust' to enable full WASM testing"; \
	fi
	@echo ""
	@echo "🔬 Generating TypeScript test fixtures..."
	hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py
	@echo "📘 Running TypeScript tests..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm test --run

test-python:  ## Run Python tests only
	hatch run test

test-cov-python:  ## Run Python tests with coverage report
	hatch run test-cov

# Pre-commit
install-pre-commit:  ## Install pre-commit hooks
	hatch run pre-commit install

run-pre-commit:  ## Run pre-commit on all files
	hatch run pre-commit run --all-files

# Quality checks (run all using Hatch)
check-all:  ## Run all quality checks (Python and TypeScript)
	@echo "🐍 Running Python checks..."
	hatch run check
	@echo "📘 Running TypeScript checks..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run typecheck && pnpm run lint && pnpm test --run

# Documentation checks (Phase 4 automation)
check-docs:  ## Check documentation quality and coverage
	@echo "📚 Checking Python documentation..."
	hatch run python scripts/check_documentation.py
	@echo "📚 Checking TypeScript JSDoc coverage..."
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && npx tsx scripts/check-jsdoc-coverage.ts --threshold=70

check-docs-verbose:  ## Check documentation with detailed output
	@echo "📚 Checking documentation (verbose mode)..."
	hatch run python scripts/check_documentation.py --verbose
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
	@echo "✅ Documentation artifacts cleaned"

# Clean up
clean:  ## Clean up temporary files and caches
	@echo "🧹 Cleaning Python artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf coverage/
	rm -rf .coverage*
	@echo "🧹 Cleaning TypeScript/Node.js artifacts..."
	rm -rf packages/luxar-viewer/dist/
	rm -rf packages/luxar-viewer/node_modules/
	rm -rf packages/luxar-viewer/.vite/
	rm -rf packages/luxar-viewer/.parcel-cache/
	rm -f packages/luxar-viewer/*.tsbuildinfo
	rm -f packages/luxar-viewer/vite.config.*.timestamp-*
	@echo "🧹 Cleaning generated datasets..."
	rm -rf datasets/
	rm -rf *.zarr
	rm -rf zarr_scenes/  # Remove deprecated directory
	@echo "✅ Clean complete!"

clean-examples:  ## Clean up only generated example zarr files
	@echo "🧹 Cleaning generated datasets..."
	@if [ -d "datasets/examples" ]; then \
		for zarr in datasets/examples/*.zarr; do \
			if [ -d "$$zarr" ]; then \
				echo "   Removing $$zarr..."; \
				rm -rf "$$zarr"; \
			fi; \
		done; \
	fi
	@echo "✅ Example zarr files cleaned!"

clean-dev-setup:  ## Remove ALL dev tools to simulate a fresh machine (USE WITH CAUTION)
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "⚠️  DEEP CLEAN - This will remove all development tools!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "This target will remove:"
	@echo "  • node_modules/           (project-local)"
	@echo "  • Hatch virtual envs      (in ~/.local/share/hatch/)"
	@echo "  • WASM build artifacts    (public/wasm/, rust/target/)"
	@echo "  • CUDA build artifacts    (*.so, build/)"
	@echo "  • wasm-pack               (Rust tool)"
	@echo "  • Rust toolchain          (rustup, cargo, rustc)"
	@echo "  • Hatch                   (Python tool)"
	@echo "  • nvm + Node.js           (~/.nvm directory)"
	@echo "  • pnpm cache              (~/.local/share/pnpm)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@read -p "Are you sure you want to continue? [y/N] " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	@echo ""
	@echo "🧹 [1/9] Removing node_modules..."
	@rm -rf packages/luxar-viewer/node_modules
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [2/9] Removing Hatch environments..."
	@if command -v hatch >/dev/null 2>&1; then \
		hatch env prune -y 2>/dev/null || true; \
	fi
	@rm -rf ~/.local/share/hatch/env/virtual/luxar* 2>/dev/null || true
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [3/9] Removing WASM build artifacts..."
	@rm -rf packages/luxar-viewer/public/wasm
	@rm -rf packages/luxar-viewer/src/wasm/rust/target
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [4/9] Removing CUDA build artifacts..."
	@rm -rf $(CUDA_EXT_DIR)/build/
	@rm -rf $(CUDA_EXT_DIR)/*.egg-info/
	@rm -f $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [5/9] Removing wasm-pack..."
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
	@echo "🧹 [6/9] Removing Rust toolchain..."
	@if command -v rustup >/dev/null 2>&1; then \
		rustup self uninstall -y 2>/dev/null || true; \
		echo "   ✓ Done"; \
	else \
		echo "   ⚪ Not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [7/9] Removing Hatch..."
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
	@echo "🧹 [8/9] Removing nvm and Node.js..."
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
	@echo "🧹 [9/9] Removing pnpm cache..."
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
	@echo "Next steps:"
	@echo "  1. Run 'make check-deps' to verify the cleanup"
	@echo "  2. Run 'make dev-setup' to reinstall everything"
	@echo ""

# Development setup
dev-setup:  ## Complete development setup (auto-installs missing dependencies)
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
	@# Install/fix hatch (use pipx)
	@if command -v hatch >/dev/null 2>&1; then \
		echo "✅ Hatch: $$(hatch --version)"; \
	elif [ -x "$$HOME/.local/bin/hatch" ]; then \
		echo "✅ Hatch: $$($$HOME/.local/bin/hatch --version) (in ~/.local/bin)"; \
		echo "⚠️  Note: Run 'pipx ensurepath' and restart terminal to add to PATH"; \
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
	else \
		echo ""; \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "⚠️  pipx not found (required for installing Hatch on modern Ubuntu/Debian)"; \
		echo ""; \
		echo "Please install pipx first, then re-run 'make dev-setup':"; \
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
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "📥 Installing pnpm..."; \
		npm install -g pnpm; \
	fi; \
	if command -v pnpm >/dev/null 2>&1; then \
		echo "✅ pnpm: $$(pnpm --version)"; \
	else \
		echo "❌ pnpm installation failed"; \
		exit 1; \
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
		echo "  ⚪ Not installed (run 'make setup-rust' to enable)"; \
	fi
	@echo ""
	@echo "CUDA (Gaussian splatting GPU acceleration):"
	@if command -v nvcc >/dev/null 2>&1; then \
		echo "  ✅ CUDA toolkit installed"; \
		if hatch run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
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
	@echo "  make check        - Verify everything works"
	@echo "  make viewer       - Start the viewer dev server"
	@echo "  make demo         - Generate a demo dataset"
	@echo ""
	@echo "Optional accelerators:"
	@echo "  make setup-rust   - Enable WASM acceleration (viewer)"
	@echo "  make setup-cuda   - Install CUDA dependencies + build extension"
	@echo ""
	@echo "💡 Use 'hatch shell' to activate the Python environment"

# Demo and serving
demo:  ## Generate a demo dataset (datasets/demos/demo.zarr with 100k points)
	@mkdir -p datasets/demos
	hatch run luxar demo --no-serve --output datasets/demos/demo.zarr --points 100000
	@echo "✅ Demo dataset created at datasets/demos/demo.zarr"

run-examples:  ## Run all examples to generate zarr files (output to datasets/examples/)
	@echo "🚀 Running all examples to generate zarr files..."
	@echo "📂 Output directory: datasets/examples/"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@total=$$(ls -1 packages/luxar/examples/*_example.py 2>/dev/null | wc -l); \
	count=0; \
	for script in packages/luxar/examples/*_example.py; do \
		count=$$((count + 1)); \
		name=$$(basename $$script); \
		echo ""; \
		echo "[$${count}/$${total}] 📊 Running $${name}..."; \
		echo "────────────────────────────────────────────────"; \
		if hatch run python $$script; then \
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
	@echo "🚀 Generating all demo datasets..."
	@echo "📂 Output directory: datasets/demos/"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@mkdir -p datasets/demos
	@# Run all demo scripts (skip if output exists)
	@total=$$(ls -1 packages/luxar/src/luxar/demos/demo_*.py 2>/dev/null | wc -l); \
	count=0; \
	for script in packages/luxar/src/luxar/demos/demo_*.py; do \
		count=$$((count + 1)); \
		name=$$(basename $$script .py | sed 's/demo_//'); \
		echo ""; \
		echo "[$${count}/$${total}] 📊 $$name"; \
		zarr_candidates="datasets/demos/$${name}.zarr datasets/demos/$$(echo $$name | tr '_' '-').zarr"; \
		found=0; \
		for zarr in $$zarr_candidates; do \
			if [ -d "$$zarr" ]; then \
				echo "   ✓ Already exists: $$zarr"; \
				found=1; \
				break; \
			fi; \
		done; \
		if [ "$$found" = "0" ]; then \
			hatch run python $$script --no-serve 2>&1 | head -20 || echo "   ⚠️  Failed or requires manual run"; \
		fi; \
	done
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Demo generation complete!"
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
	@if [ -d "datasets/demos/lorenz.zarr" ]; then echo "   ✓ exists"; else hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[2/5] 🔮 Mandelbulb..."
	@if [ -d "datasets/demos/mandelbulb.zarr" ]; then echo "   ✓ exists"; else hatch run python packages/luxar/src/luxar/demos/demo_mandelbulb.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[3/5] 🌌 Spiral Galaxy..."
	@if [ -d "datasets/demos/spiral_galaxy.zarr" ]; then echo "   ✓ exists"; else hatch run python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[4/5] 🧬 Zebrahub Multiome UMAP..."
	@if [ -d "datasets/demos/zebrahub_multiome_peak_umap.zarr" ]; then echo "   ✓ exists"; else hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_multiome_peak_umap.py --no-serve || echo "   ⚠️  Failed"; fi
	@echo "[5/5] 🌈 Rainbow Sphere..."
	@if [ -d "datasets/demos/rainbow_sphere.zarr" ]; then echo "   ✓ exists"; else hatch run python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py --no-serve || echo "   ⚠️  Failed"; fi
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

serve-examples:  ## Serve the datasets directory for browsing generated datasets
	@echo "🌐 Serving datasets/ directory at http://localhost:8000"
	@echo "📊 Open viewer at: http://localhost:5173/?src=http://localhost:8000"
	@echo "💡 Press 'O' in the viewer to browse available datasets"
	@echo ""
	hatch run luxar serve datasets/ -p 8000

# Default values for serve-dataset (override with: make serve-dataset DATASET=path/to/data.zarr PORT=8080)
DATASET ?= datasets/demos/demo.zarr
PORT ?= 8000

serve-dataset:  ## Serve a dataset (default: datasets/demos/demo.zarr, port: 8000)
	@if [ ! -d "datasets/demos/demo.zarr" ]; then \
		echo "No demo dataset found. Creating one..."; \
		$(MAKE) demo; \
	fi
	hatch run luxar serve $(DATASET) -p $(PORT)

# Web viewer
install-viewer:  ## Install viewer dependencies
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
			echo "   Run 'make install-node' to upgrade, or 'make dev-setup' for full setup."; \
			exit 1; \
		fi; \
	else \
		echo "❌ Node.js not found. Run 'make dev-setup' first."; \
		exit 1; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found. Run 'make dev-setup' first."; \
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
			echo "   Run 'make install-node' to upgrade, or 'make dev-setup' for full setup."; \
			exit 1; \
		fi; \
	else \
		echo "❌ Node.js not found. Run 'make dev-setup' first."; \
		exit 1; \
	fi
	@# Check for wasm-pack, install if needed (separate command to ensure Make waits)
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "⚠️  wasm-pack not found. Installing Rust/WASM toolchain..."; \
		echo ""; \
		$(MAKE) setup-rust; \
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
			echo "   Run 'make install-node' to upgrade, or 'make dev-setup' for full setup."; \
			exit 1; \
		fi; \
	else \
		echo "❌ Node.js not found. Run 'make dev-setup' first."; \
		exit 1; \
	fi
	@# Check for wasm-pack, install if needed (separate command to ensure Make waits)
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "⚠️  wasm-pack not found. Installing Rust/WASM toolchain..."; \
		echo ""; \
		$(MAKE) setup-rust; \
	fi
	@# Reinstall dependencies and build (source nvm for pnpm)
	@export NVM_DIR="$$HOME/.nvm"; \
	if [ -s "$$NVM_DIR/nvm.sh" ]; then \
		. "$$NVM_DIR/nvm.sh"; \
	fi; \
	echo "📦 Reinstalling dependencies..."; \
	cd packages/luxar-viewer && pnpm install; \
	echo "🦀 Building viewer with Rust/WASM support..."; \
	cd packages/luxar-viewer && pnpm build; \
	echo "✅ Viewer rebuild complete!"

# WASM/Rust setup and build (Phase 3)
setup-rust:  ## Install/update Rust and wasm-pack for WASM development
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
		echo "📥 Installing wasm-pack (this may take a minute)..."; \
		cargo install wasm-pack; \
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
		echo "Run 'make setup-rust' to install Rust and wasm-pack."; \
		echo ""; \
		exit 1; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found."; \
		echo ""; \
		echo "Run 'make dev-setup' to install Node.js and pnpm."; \
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
		echo "Run 'make setup-rust' to install Rust."; \
		echo ""; \
		exit 1; \
	fi; \
	if ! command -v pnpm >/dev/null 2>&1; then \
		echo "❌ pnpm not found."; \
		echo ""; \
		echo "Run 'make dev-setup' to install Node.js and pnpm."; \
		echo ""; \
		exit 1; \
	fi; \
	echo "🧪 Running Rust tests..."; \
	cd packages/luxar-viewer && pnpm test:wasm && \
	echo "✅ All Rust tests passed!"

clean-wasm:  ## Clean WASM build artifacts
	@echo "🧹 Cleaning WASM artifacts..."
	rm -rf packages/luxar-viewer/public/wasm/
	rm -rf packages/luxar-viewer/src/wasm/rust/target/
	@echo "✅ WASM artifacts cleaned!"

# ============================================================================
# CUDA Backend (Gaussian Splatting)
# ============================================================================

# Path to CUDA extension directory
CUDA_EXT_DIR := packages/luxar/src/luxar/gsplats/models/gsplats/cuda

setup-cuda:  ## Install CUDA dependencies (may require sudo for system packages)
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
	if [ "$$NEED_SUDO" = "1" ]; then \
		echo ""; \
		echo "⚠️  Some system packages need to be installed (requires sudo):"; \
		echo ""; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			CMD="sudo apt update && sudo apt install -y"; \
			for pkg in $$MISSING; do \
				case $$pkg in \
					nvidia-driver) CMD="$$CMD nvidia-driver-535";; \
					cuda-toolkit) CMD="$$CMD nvidia-cuda-toolkit";; \
					build-essential) CMD="$$CMD build-essential";; \
				esac; \
			done; \
			echo "   $$CMD"; \
			echo ""; \
			read -p "Run this command now? [y/N] " confirm; \
			if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
				eval $$CMD; \
			else \
				echo ""; \
				echo "Skipped. Please install manually and re-run 'make setup-cuda'."; \
				exit 1; \
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
	@if hatch run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
		TORCH_CUDA=$$(hatch run python -c "import torch; print(torch.version.cuda)" 2>/dev/null); \
		echo "✅ PyTorch with CUDA $$TORCH_CUDA already installed"; \
	else \
		echo "📥 Installing PyTorch with CUDA 12.1 support..."; \
		echo ""; \
		hatch run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
		echo ""; \
		if hatch run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
			echo "✅ PyTorch with CUDA installed successfully"; \
		else \
			echo "⚠️  PyTorch installed but CUDA not available"; \
			echo "   This may happen if NVIDIA driver is not properly installed."; \
		fi; \
	fi
	@echo ""
	@# Step 3: Build the extension
	@echo "=== Step 3: Build CUDA Extension ==="
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
		echo "✅ CUDA extension already built"; \
	else \
		if command -v nvcc >/dev/null 2>&1 && hatch run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
			echo "Building CUDA extension..."; \
			hatch run pip install -q ninja 2>/dev/null || true; \
			hatch run python $(CUDA_EXT_DIR)/build.py; \
		else \
			echo "⚠️  Cannot build - prerequisites not satisfied"; \
			echo "   Run 'make check-cuda-deps' for details"; \
		fi; \
	fi
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
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
		DRIVER_VERSION=$$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1); \
		GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
		if [ -n "$$DRIVER_VERSION" ]; then \
			echo "✅ NVIDIA driver: $$DRIVER_VERSION"; \
			echo "   GPU: $$GPU_NAME"; \
		else \
			echo "⚠️  nvidia-smi found but GPU not detected"; \
		fi; \
	else \
		echo "❌ NVIDIA driver not found (nvidia-smi not available)"; \
		echo ""; \
		echo "   Install NVIDIA drivers:"; \
		if [ "$(PKG_MANAGER)" = "apt" ]; then \
			echo "     sudo apt install nvidia-driver-535  # or latest version"; \
		else \
			echo "     https://www.nvidia.com/drivers"; \
		fi; \
	fi
	@echo ""
	@echo "=== 3. PyTorch with CUDA ==="
	@hatch run python -c "\
import sys; \
try: \
    import torch; \
    print('✅ PyTorch:', torch.__version__); \
    if torch.cuda.is_available(): \
        print('✅ PyTorch CUDA:', torch.version.cuda); \
        print('   GPU:', torch.cuda.get_device_name(0)); \
        cap = torch.cuda.get_device_capability(); \
        print('   Compute capability:', f'{cap[0]}.{cap[1]}'); \
    else: \
        print('❌ PyTorch CUDA not available'); \
        print(''); \
        print('   Current PyTorch was built without CUDA support.'); \
        print('   Install PyTorch with CUDA (in hatch environment):'); \
        print(''); \
        print('     hatch run pip install torch --index-url https://download.pytorch.org/whl/cu121'); \
        print(''); \
        print('   Or for CUDA 12.4:'); \
        print('     hatch run pip install torch --index-url https://download.pytorch.org/whl/cu124'); \
except ImportError: \
    print('❌ PyTorch not installed'); \
    print(''); \
    print('   Install PyTorch with CUDA:'); \
    print('     hatch run pip install torch --index-url https://download.pytorch.org/whl/cu121'); \
" 2>/dev/null || echo "❌ Could not check PyTorch (hatch environment issue)"
	@echo ""
	@echo "=== 4. C++ Compiler ==="
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
	@echo "=== 5. CUDA Extension Status ==="
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
		SO_FILE=$$(ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so | head -1); \
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
	if ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then EXT_OK=1; fi; \
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

build-cuda:  ## Build the CUDA splatting extension
	@echo "🔧 Building CUDA splatting extension..."
	@echo ""
	@# Check prerequisites
	@if ! command -v nvcc >/dev/null 2>&1; then \
		echo "❌ CUDA toolkit not found (nvcc not in PATH)"; \
		echo ""; \
		echo "   Run 'make check-cuda-deps' for installation instructions."; \
		exit 1; \
	fi
	@if ! command -v nvidia-smi >/dev/null 2>&1; then \
		echo "❌ NVIDIA driver not found"; \
		echo ""; \
		echo "   Run 'make check-cuda-deps' for installation instructions."; \
		exit 1; \
	fi
	@if ! hatch run python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then \
		echo "❌ PyTorch with CUDA support not available"; \
		echo ""; \
		echo "   Install PyTorch with CUDA in hatch environment:"; \
		echo "     hatch run pip install torch --index-url https://download.pytorch.org/whl/cu121"; \
		echo ""; \
		echo "   Or run 'make check-cuda-deps' for more details."; \
		exit 1; \
	fi
	@echo "✅ Prerequisites OK"
	@echo ""
	@# Ensure ninja is installed (required by torch cpp_extension)
	@hatch run pip install -q ninja 2>/dev/null || true
	@echo "Building extension (this may take a few minutes)..."
	@echo ""
	hatch run python $(CUDA_EXT_DIR)/build.py
	@echo ""
	@if ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
		echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
		echo "✅ CUDA extension built successfully!"; \
		SO_FILE=$$(ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so | head -1); \
		echo "   Output: $$(basename $$SO_FILE)"; \
		echo ""; \
		echo "Next steps:"; \
		echo "  make test-cuda      - Run tests to verify"; \
		echo "  make benchmark-cuda - Run performance benchmarks"; \
	else \
		echo "❌ Build may have failed - .so file not found"; \
		echo "   Check the build output above for errors."; \
		exit 1; \
	fi

clean-cuda:  ## Clean CUDA build artifacts
	@echo "🧹 Cleaning CUDA build artifacts..."
	rm -rf $(CUDA_EXT_DIR)/build/
	rm -rf $(CUDA_EXT_DIR)/*.egg-info/
	rm -f $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so
	rm -rf $(CUDA_EXT_DIR)/__pycache__/
	@echo "✅ CUDA artifacts cleaned!"

test-cuda:  ## Run CUDA extension tests
	@echo "🧪 Running CUDA extension tests..."
	@echo ""
	@# Check if extension is built
	@if ! ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
		echo "⚠️  CUDA extension not built. Building first..."; \
		$(MAKE) build-cuda; \
		echo ""; \
	fi
	@# Run tests
	hatch run pytest $(CUDA_EXT_DIR)/tests/ -v
	@echo ""
	@echo "✅ CUDA tests completed!"

benchmark-cuda:  ## Run CUDA performance benchmarks
	@echo "🚀 Running CUDA performance benchmarks..."
	@echo ""
	@# Check if extension is built
	@if ! ls $(CUDA_EXT_DIR)/cuda_splatting_backend.cpython-*.so 1>/dev/null 2>&1; then \
		echo "⚠️  CUDA extension not built. Building first..."; \
		$(MAKE) build-cuda; \
		echo ""; \
	fi
	@# Run benchmark
	hatch run python $(CUDA_EXT_DIR)/benchmark.py
	@echo ""
	@echo "✅ Benchmark completed!"

test-fixtures:  ## Generate test fixtures for TypeScript tests
	@echo "🔬 Generating test fixtures..."
	hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py

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

# Documentation
build-docs:  ## Build documentation with Sphinx
	hatch run docs:build

serve-docs:  ## Serve documentation locally
	hatch run docs:serve

# Project statistics
stats:  ## Generate project statistics report (HTML)
	@echo "📊 Analyzing project codebase..."
	hatch run python stats/generate_stats.py
	@echo "✅ Report generated: stats/project_stats.html"
	@echo "💡 Open with: open stats/project_stats.html"

# Hatch environment management
show-env:  ## Show all Hatch environments
	hatch env show

prune-env:  ## Remove unused Hatch environments
	hatch env prune

shell:  ## Enter Hatch development shell
	hatch shell

# Building and publishing
build:  ## Build distribution packages
	hatch build

publish-test:  ## Publish to TestPyPI
	hatch publish -r test

publish:  ## Publish to PyPI
	hatch publish