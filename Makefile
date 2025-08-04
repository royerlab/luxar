# Makefile for Luxar development tasks
# Uses Hatch for on-demand environment management
.PHONY: help install install-dev format lint type-check security test test-cov test-all \
        clean pre-commit-install pre-commit-run check dev-setup demo serve-data \
        viewer-install viewer viewer-build viewer-test viewer-test-cov viewer-check \
        demo-and-serve docs-build docs-serve env-show env-prune shell build \
        publish-test publish

# Default target
help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install the package
	pip install -e .

install-dev:  ## Install with development dependencies (legacy - use hatch instead)
	@echo "⚠️  Note: Consider using 'hatch shell' for development environments"
	pip install -e ".[dev]"

# Code formatting (using Hatch)
format:  ## Format code with black and isort
	hatch run format

# Code quality checks (using Hatch)
lint:  ## Run flake8 linting
	hatch run lint

type-check:  ## Run mypy type checking
	hatch run mypy packages/luxar/src/luxar/

security:  ## Run bandit security checks
	hatch run bandit -r packages/luxar/src/luxar/ -c pyproject.toml

# Testing (using Hatch)
test:  ## Run tests
	hatch run test

test-cov:  ## Run tests with coverage report
	hatch run test-cov

test-all:  ## Run all tests (Python and TypeScript)
	@echo "🐍 Running Python tests..."
	hatch run test
	@echo "📘 Running TypeScript tests..."
	cd packages/luxar-player && pnpm test --run

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
	cd packages/luxar-player && pnpm run typecheck && pnpm run lint && pnpm test --run

# Clean up
clean:  ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage*
	# Clean TypeScript/Node.js artifacts
	rm -rf packages/luxar-player/dist/
	rm -rf packages/luxar-player/node_modules/
	rm -rf packages/luxar-player/.vite/
	rm -rf packages/luxar-player/coverage/
	rm -rf packages/luxar-player/.parcel-cache/
	rm -f packages/luxar-player/*.tsbuildinfo
	rm -f packages/luxar-player/vite.config.*.timestamp-*
	# Clean pnpm store links (optional, uncomment if needed)
	# rm -rf packages/luxar-player/.pnpm-store/

# Development setup
dev-setup:  ## Complete development setup with Hatch
	@echo "🚀 Setting up development environment with Hatch..."
	@command -v hatch >/dev/null 2>&1 || { echo "❌ Hatch not found. Please install with: pip install hatch"; exit 1; }
	hatch env create
	hatch run pre-commit install
	@echo "✅ Development environment setup complete!"
	@echo "💡 Use 'hatch shell' to activate the environment"
	@echo "💡 Run 'make check' to verify everything works"

# Demo and serving
demo:  ## Generate a demo dataset (dist/demo.zarr with 100k points)
	@mkdir -p dist
	hatch run luxar random --out dist/demo.zarr --n 100000
	@echo "✅ Demo dataset created at dist/demo.zarr"

serve-data:  ## Serve a dataset (default: dist/demo.zarr, port: 8000)
	@if [ ! -d "dist/demo.zarr" ]; then \
		echo "No demo dataset found. Creating one..."; \
		make demo; \
	fi
	hatch run luxar serve $(DATASET) -p $(PORT)

# Override defaults with: make serve-data DATASET=path/to/data.zarr PORT=8080
DATASET ?= dist/demo.zarr
PORT ?= 8000

# Web viewer
viewer-install:  ## Install viewer dependencies
	cd packages/luxar-player && pnpm install

viewer:  ## Start the web viewer development server
	cd packages/luxar-player && pnpm dev

viewer-build:  ## Build the viewer for production
	cd packages/luxar-player && pnpm build

viewer-test:  ## Run TypeScript tests
	cd packages/luxar-player && pnpm test --run

viewer-test-cov:  ## Run TypeScript tests with coverage
	cd packages/luxar-player && pnpm run test:coverage

viewer-check:  ## Run all TypeScript checks (typecheck, lint, test)
	cd packages/luxar-player && pnpm run check

# Combined workflows
demo-and-serve: demo  ## Create demo and start both servers
	@echo "Starting data server and viewer..."
	@echo "Data will be served at: http://localhost:$(PORT)/data/demo.zarr/"
	@echo "Viewer will be at: http://localhost:5173/?src=http://localhost:$(PORT)/data/demo.zarr/"
	@$(MAKE) -j2 serve-data viewer

# Documentation
docs-build:  ## Build documentation with Sphinx
	hatch run docs:build

docs-serve:  ## Serve documentation locally
	hatch run docs:serve

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