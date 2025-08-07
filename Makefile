# Makefile for Luxar development tasks
# Uses Hatch for on-demand environment management
.PHONY: help install install-dev format format-all lint type-check security test test-python \
        test-cov test-all clean clean-examples pre-commit-install pre-commit-run check dev-setup \
        demo run-examples serve-examples serve-data viewer-install viewer viewer-build viewer-test \
        viewer-test-cov viewer-lint viewer-typecheck viewer-format viewer-check demo-and-serve \
        docs-build docs-serve env-show env-prune shell build publish-test publish

# Default target
help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:"
	@echo "  make dev-setup      - Set up development environment"
	@echo "  make test-all       - Run all tests"
	@echo "  make run-examples   - Generate all example datasets"
	@echo "  make serve-examples - Browse generated examples"
	@echo "  make demo-and-serve - Create demo and start servers"

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
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run format

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

test-all:  ## Run all tests (Python and TypeScript)
	@echo "🐍 Running Python tests..."
	hatch run test
	@echo "📘 Running TypeScript tests..."
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
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
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run typecheck && pnpm run lint && pnpm test --run

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
	rm -rf htmlcov/
	rm -rf .coverage*
	rm -rf packages/luxar/htmlcov/
	@echo "🧹 Cleaning TypeScript/Node.js artifacts..."
	rm -rf packages/luxar-player/dist/
	rm -rf packages/luxar-player/node_modules/
	rm -rf packages/luxar-player/.vite/
	rm -rf packages/luxar-player/coverage/
	rm -rf packages/luxar-player/.parcel-cache/
	rm -f packages/luxar-player/*.tsbuildinfo
	rm -f packages/luxar-player/vite.config.*.timestamp-*
	@echo "🧹 Cleaning example outputs..."
	find examples -name "*.zarr" -type d -exec rm -rf {} +
	rm -rf *.zarr
	rm -rf zarr_scenes/  # Remove deprecated directory
	@echo "✅ Clean complete!"

clean-examples:  ## Clean up only example zarr files
	@echo "🧹 Cleaning example zarr files..."
	@cd examples && for zarr in *.zarr; do \
		if [ -d "$$zarr" ]; then \
			echo "   Removing $$zarr..."; \
			rm -rf "$$zarr"; \
		fi; \
	done
	@echo "✅ Example zarr files cleaned!"

# Development setup
dev-setup:  ## Complete development setup with Hatch
	@echo "🚀 Setting up development environment with Hatch..."
	@command -v hatch >/dev/null 2>&1 || { echo "❌ Hatch not found. Please install with: pip install hatch"; exit 1; }
	hatch env create
	hatch run pre-commit install
	@echo "📦 Installing TypeScript/viewer dependencies..."
	cd packages/luxar-player && pnpm install
	@echo "✅ Development environment setup complete!"
	@echo "💡 Use 'hatch shell' to activate the environment"
	@echo "💡 Run 'make check' to verify everything works"

# Demo and serving
demo:  ## Generate a demo dataset (dist/demo.zarr with 100k points)
	@mkdir -p dist
	hatch run luxar random --out dist/demo.zarr --n 100000
	@echo "✅ Demo dataset created at dist/demo.zarr"

run-examples:  ## Run all examples to generate zarr files
	@echo "🚀 Running all examples to generate zarr files..."
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@total=$$(ls -1 examples/*_example.py 2>/dev/null | wc -l); \
	count=0; \
	for script in examples/*_example.py; do \
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
	@echo "📁 Generated zarr files in examples/:"
	@for zarr in examples/*.zarr; do \
		if [ -d "$$zarr" ]; then \
			size=$$(du -sh "$$zarr" | cut -f1); \
			name=$$(basename "$$zarr"); \
			echo "   • $$name ($${size})"; \
		fi; \
	done 2>/dev/null || echo "   No .zarr files found"
	@echo ""
	@echo "💡 To browse the generated datasets, run:"
	@echo "   make serve-examples"

serve-examples:  ## Serve the examples directory for browsing datasets
	@echo "🌐 Serving examples directory at http://localhost:8000/"
	@echo "📊 Open viewer at: http://localhost:5173/?src=http://localhost:8000/"
	@echo "💡 Press 'O' in the viewer to browse available datasets"
	@echo ""
	hatch run luxar serve examples/

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
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm test --run

viewer-test-cov:  ## Run TypeScript tests with coverage
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run test:coverage

viewer-lint:  ## Run TypeScript linting
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run lint

viewer-typecheck:  ## Run TypeScript type checking
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run typecheck

viewer-format:  ## Format TypeScript code
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run format

viewer-check:  ## Run all TypeScript checks (typecheck, lint, test)
	@if [ ! -d "packages/luxar-player/node_modules" ]; then \
		echo "📦 Installing TypeScript dependencies first..."; \
		cd packages/luxar-player && pnpm install; \
	fi
	cd packages/luxar-player && pnpm run check

# Combined workflows
demo-and-serve: demo  ## Create demo and start both servers
	@echo "Starting data server and viewer..."
	@echo "Data will be served at: http://localhost:$(PORT)/"
	@echo "Viewer will be at: http://localhost:5173/?src=http://localhost:$(PORT)/"
	@echo "💡 Press 'O' in the viewer to browse datasets"
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