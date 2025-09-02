# Development Tools and Standards

This document outlines the development tools used in the Luxar project for code quality, formatting, and testing.

## Python Development Tools

### Code Formatting and Linting
- **Tool**: `ruff` (v0.1.0+)
- **Purpose**: Replaces `black`, `isort`, and `flake8` with a single, fast tool
- **Configuration**: `pyproject.toml` under `[tool.ruff]`
- **Commands**:
  - Format code: `hatch run ruff format packages/luxar/src`
  - Check linting: `hatch run ruff check packages/luxar/src`
  - Fix auto-fixable issues: `hatch run ruff check --fix packages/luxar/src`

### Type Checking
- **Tool**: `mypy` (v1.5.0+)
- **Configuration**: `pyproject.toml` under `[tool.mypy]`
- **Mode**: Strict mode enabled
- **Command**: `hatch run mypy packages/luxar/src/luxar`

### Security Scanning
- **Tool**: `bandit` (v1.7.5+)
- **Configuration**: `pyproject.toml` under `[tool.bandit]`
- **Command**: `hatch run bandit -r packages/luxar/src/luxar/`

### Testing
- **Tool**: `pytest` (v7.4.0+) with `pytest-cov`
- **Configuration**: `pyproject.toml` under `[tool.pytest.ini_options]`
- **Coverage requirement**: 80% minimum
- **Commands**:
  - Run tests: `hatch run test`
  - Run with coverage: `hatch run test-cov`

### Environment Management
- **Tool**: `hatch`
- **Purpose**: Manages virtual environments and dependencies
- **Configuration**: `pyproject.toml` under `[tool.hatch.*]`
- **Commands**:
  - Enter shell: `hatch shell`
  - Show environments: `hatch env show`
  - Clean environments: `hatch env prune`

## TypeScript Development Tools

### Code Formatting
- **Tool**: `prettier` (v3.6.2+)
- **Configuration**: `packages/luxar-viewer/.prettierrc`
- **Command**: `pnpm run format`

### Linting
- **Tool**: `eslint` (v9.32.0+) with TypeScript plugins
- **Configuration**: `packages/luxar-viewer/.eslintrc.json`
- **Plugins**:
  - `@typescript-eslint/parser`
  - `@typescript-eslint/eslint-plugin`
- **Command**: `pnpm run lint`

### Type Checking
- **Tool**: `tsc` (TypeScript compiler v5.7+)
- **Configuration**: `packages/luxar-viewer/tsconfig.json`
- **Mode**: Strict mode enabled
- **Command**: `pnpm run typecheck`

### Testing
- **Tool**: `vitest` (v2.0.0+)
- **Configuration**: Built into `vite.config.ts`
- **Commands**:
  - Run tests: `pnpm test`
  - Run with UI: `pnpm test:ui`
  - Run with coverage: `pnpm test:coverage`

### Build Tool
- **Tool**: `vite` (v5.2.0+)
- **Purpose**: Development server and production builds
- **Commands**:
  - Development: `pnpm dev`
  - Build: `pnpm build`
  - Preview build: `pnpm preview`

### Package Manager
- **Tool**: `pnpm` (v8.0.0+)
- **Purpose**: Fast, disk space efficient package manager
- **Commands**:
  - Install dependencies: `pnpm install`
  - Add dependency: `pnpm add <package>`
  - Remove dependency: `pnpm remove <package>`

## Makefile Commands

The project provides convenience commands through Make:

### Python Commands
- `make format` - Format Python code with ruff
- `make lint` - Run ruff linting
- `make type-check` - Run mypy type checking
- `make test` - Run Python tests
- `make test-cov` - Run Python tests with coverage
- `make security` - Run bandit security checks

### TypeScript Commands
- `make viewer-format` - Format TypeScript code
- `make viewer-lint` - Run ESLint
- `make viewer-typecheck` - Run TypeScript type checking
- `make viewer-test` - Run TypeScript tests
- `make viewer-build` - Build for production

### Combined Commands
- `make format-all` - Format both Python and TypeScript code
- `make test-all` - Run all tests (Python and TypeScript)
- `make check` - Run all quality checks

## Pre-commit Hooks

The project uses `pre-commit` for automated checks before commits:
- Install hooks: `make pre-commit-install`
- Run manually: `make pre-commit-run`

## Standards and Conventions

### Python
- Line length: 88 characters (ruff/black standard)
- Import sorting: PEP 8 compliant with ruff
- Docstring style: Google convention
- Type hints: Required for all public functions
- Complexity: Maximum McCabe complexity of 10

### TypeScript
- Line length: 100 characters
- Quotes: Single quotes preferred
- Semicolons: Required
- Indentation: 2 spaces
- Unused variables: Must be prefixed with `_`

## Tool Selection Rationale

1. **Ruff for Python**: Chosen for its speed and ability to replace multiple tools (black, isort, flake8) with consistent configuration.

2. **Strict typing**: Both `mypy` and `tsc` run in strict mode to catch more potential issues at development time.

3. **Hatch**: Modern Python project management that handles environments, dependencies, and scripts in a standardized way.

4. **Vitest**: Fast test runner that integrates well with Vite and provides excellent TypeScript support.

5. **pnpm**: Efficient package manager that saves disk space through hard linking and provides strict dependency resolution.

6. **Unified configuration**: Most tools are configured in `pyproject.toml` (Python) or `package.json` (TypeScript) to reduce configuration file sprawl.

## Important Notes

- Always use `hatch run` prefix for Python commands to ensure correct environment
- Always use `pnpm` (not npm or yarn) for TypeScript/Node.js package management
- Run `make check` before committing to catch issues early
- Keep LUXAR_ZARR_FORMAT.md updated when making changes to the data format