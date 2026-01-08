# Makefile for Luxar development tasks
# Uses Hatch for on-demand environment management
#
# This Makefile is designed to work on fresh Linux/macOS machines with minimal
# pre-installed tools. Run 'make dev-setup' to automatically install all dependencies.
#
.PHONY: help install install-dev format format-all lint type-check security test test-python \
        test-cov test-all test-fixtures clean clean-examples pre-commit-install pre-commit-run check dev-setup \
        check-docs check-docs-verbose docs-clean docs-build docs-serve \
        demo run-demos run-examples serve-examples serve-data viewer-install viewer viewer-build viewer-rebuild \
        viewer-test viewer-test-fixtures viewer-test-cov viewer-lint viewer-typecheck viewer-format viewer-check \
        setup-rust wasm-build wasm-test wasm-clean readme-images readme-videos \
        demo-and-serve stats env-show env-prune shell build publish-test publish \
        check-deps install-node install-pnpm install-hatch deep-clean-dev-setup

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
	@echo "  make test-all       - Run all tests"
	@echo "  make viewer         - Start the viewer dev server"
	@echo "  make demo-and-serve - Create demo and start servers"
	@echo "  make run-examples   - Generate all example datasets"
	@echo ""
	@echo "System: $(OS) (package manager: $(PKG_MANAGER))"
	@echo "Node.js requirement: $(MIN_NODE_MAJOR).$(MIN_NODE_MINOR)+"

# Installation
install:  ## Install the package
	pip install -e .

install-dev:  ## Install with development dependencies (legacy - use hatch instead)
	@echo "⚠️  Note: Consider using 'hatch shell' for development environments"
	pip install -e ".[dev]"

# Code formatting (using Hatch)
format:  ## Format Python code with ruff
	hatch run format

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
lint:  ## Run ruff linting
	hatch run python -m ruff check packages/luxar/src/luxar/

type-check:  ## Run mypy type checking
	hatch run mypy packages/luxar/src/luxar/

security:  ## Run bandit security checks
	hatch run bandit -r packages/luxar/src/luxar/ -c pyproject.toml

# Testing (using Hatch)
test:  ## Run Python tests
	hatch run test

test-python:  ## Run Python tests (alias for test)
	hatch run test

test-cov:  ## Run Python tests with coverage report
	hatch run test-cov

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

# Pre-commit
pre-commit-install:  ## Install pre-commit hooks
	hatch run pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	hatch run pre-commit run --all-files

# Quality checks (run all using Hatch)
check:  ## Run all quality checks (Python and TypeScript)
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

docs-clean:  ## Clean built documentation
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

deep-clean-dev-setup:  ## Remove ALL dev tools to simulate a fresh machine (USE WITH CAUTION)
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "⚠️  DEEP CLEAN - This will remove all development tools!"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "This target will remove:"
	@echo "  • node_modules/           (project-local)"
	@echo "  • Hatch virtual envs      (in ~/.local/share/hatch/)"
	@echo "  • WASM build artifacts    (public/wasm/, rust/target/)"
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
	@echo "🧹 [1/8] Removing node_modules..."
	@rm -rf packages/luxar-viewer/node_modules
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [2/8] Removing Hatch environments..."
	@if command -v hatch >/dev/null 2>&1; then \
		hatch env prune -y 2>/dev/null || true; \
	fi
	@rm -rf ~/.local/share/hatch/env/virtual/luxar* 2>/dev/null || true
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [3/8] Removing WASM build artifacts..."
	@rm -rf packages/luxar-viewer/public/wasm
	@rm -rf packages/luxar-viewer/src/wasm/rust/target
	@echo "   ✓ Done"
	@echo ""
	@echo "🧹 [4/8] Removing wasm-pack..."
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
	@echo "🧹 [5/8] Removing Rust toolchain..."
	@if command -v rustup >/dev/null 2>&1; then \
		rustup self uninstall -y 2>/dev/null || true; \
		echo "   ✓ Done"; \
	else \
		echo "   ⚪ Not installed, skipping"; \
	fi
	@echo ""
	@echo "🧹 [6/8] Removing Hatch..."
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
	@echo "🧹 [7/8] Removing nvm and Node.js..."
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
	@echo "🧹 [8/8] Removing pnpm cache..."
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
	@echo "=== Step 5: Optional WASM Support ==="
	@if [ -f "$(HOME)/.cargo/env" ]; then \
		. "$(HOME)/.cargo/env"; \
	fi; \
	if command -v wasm-pack >/dev/null 2>&1; then \
		echo "✅ Rust/WASM already configured"; \
	else \
		echo "⚪ Rust/WASM not installed (optional, for production builds)"; \
		echo "   Run 'make setup-rust' to enable WASM acceleration"; \
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
	@echo "  make setup-rust   - Enable WASM acceleration (optional)"
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

readme-demos:  ## Generate only the demo datasets needed for README screenshots
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

readme-images: readme-demos  ## Generate README screenshots using Playwright
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

readme-videos: readme-demos  ## Generate README videos (GIF/WebP) using Playwright
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

# Default values for serve-data (override with: make serve-data DATASET=path/to/data.zarr PORT=8080)
DATASET ?= datasets/demos/demo.zarr
PORT ?= 8000

serve-data:  ## Serve a dataset (default: datasets/demos/demo.zarr, port: 8000)
	@if [ ! -d "datasets/demos/demo.zarr" ]; then \
		echo "No demo dataset found. Creating one..."; \
		$(MAKE) demo; \
	fi
	hatch run luxar serve $(DATASET) -p $(PORT)

# Web viewer
viewer-install:  ## Install viewer dependencies
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

viewer-build:  ## Build the viewer for production (auto-installs Rust/wasm-pack if needed)
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

viewer-rebuild:  ## Complete clean rebuild of viewer (auto-installs dependencies as needed)
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
	echo "💡 Run 'make wasm-test' to test the Rust code"; \
	echo "💡 Run 'make wasm-build' to compile the WASM module"

wasm-build:  ## Build the WASM module (requires Rust + wasm-pack)
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

wasm-test:  ## Run Rust unit tests for WASM module
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

wasm-clean:  ## Clean WASM build artifacts
	@echo "🧹 Cleaning WASM artifacts..."
	rm -rf packages/luxar-viewer/public/wasm/
	rm -rf packages/luxar-viewer/src/wasm/rust/target/
	@echo "✅ WASM artifacts cleaned!"

test-fixtures:  ## Generate test fixtures for TypeScript tests
	@echo "🔬 Generating test fixtures..."
	hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py

viewer-test-fixtures: test-fixtures  ## Generate fixtures + run TypeScript tests
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm test --run

viewer-test:  ## Run TypeScript tests (without regenerating fixtures)
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm test --run

viewer-test-cov:  ## Run TypeScript tests with coverage
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run test:coverage

viewer-lint:  ## Run TypeScript linting
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run lint

viewer-typecheck:  ## Run TypeScript type checking
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run typecheck

viewer-format:  ## Format TypeScript code
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run format

viewer-check:  ## Run all TypeScript checks (typecheck, lint, test)
	@if [ ! -d "packages/luxar-viewer/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-viewer && pnpm install; \
	fi
	cd packages/luxar-viewer && pnpm run check

# Combined workflows
demo-and-serve: demo  ## Create demo and start both servers
	@echo "Starting data server and viewer..."
	@echo "Data will be served at: http://localhost:$(PORT)"
	@echo "Viewer will be at: http://localhost:5173/?src=http://localhost:$(PORT)"
	@echo "💡 Dataset: datasets/demos/demo.zarr (100k points)"
	@$(MAKE) -j2 serve-data viewer

# Documentation
docs-build:  ## Build documentation with Sphinx
	hatch run docs:build

docs-serve:  ## Serve documentation locally
	hatch run docs:serve

# Project statistics
stats:  ## Generate project statistics report (HTML)
	@echo "📊 Analyzing project codebase..."
	hatch run python stats/generate_stats.py
	@echo "✅ Report generated: stats/project_stats.html"
	@echo "💡 Open with: open stats/project_stats.html"

# Hatch environment management
env-show:  ## Show all Hatch environments
	hatch env show

env-prune:  ## Remove unused Hatch environments
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