# Repository Guidelines

## Project Structure & Module Organization
- `packages/luxar/src/luxar/` contains the Python compiler (core/, io/, encoding/, validation/, utils/, gsplats/).
- `packages/luxar-viewer/` is the WebGL/TypeScript viewer; `src/` holds the renderer and UI packages.
- `packages/luxar/examples/` and `packages/luxar/src/luxar/demos/` provide runnable sample scripts.
- `docs/` is Sphinx documentation (guides, tutorials, API .rst files).
- `scripts/`, `datasets/`, and `stats/` host utilities and supporting data.

## Build, Test, and Development Commands
- `make dev-setup` sets up the full dev environment (Python + viewer tooling).
- `make check` runs formatting, linting, type checks, and tests.
- `make test` runs all tests (Python + TypeScript + WASM); `make test-python` runs Python only.
- `make viewer` starts the viewer dev server at `http://localhost:5173`.
- `make test-viewer` runs TypeScript unit tests; `make test-wasm` runs Rust tests.
- For Python-only tasks, `hatch run test` / `hatch run test-cov` is the standard path.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints on public APIs, Ruff formatting (88 char lines).
- Use `arbol` for console output (`aprint`, `asection`) instead of `print()`.
- Tests use `test_*.py` names; examples use `*_example.py`.
- TypeScript: strict types, ESLint + Prettier, camelCase config keys, JSDoc for complex logic.
- Prefix unused variables with `_` and use `utils/log` for console output in viewer code.

## Testing Guidelines
- Python tests live under `packages/luxar/src/luxar/**/tests` and use pytest.
- Viewer tests live under `packages/luxar-viewer/src/tests/unit` (Vitest) and `.../e2e` (Playwright).
- Coverage target is 80%+; mark slow tests with `@pytest.mark.slow` and integration tests with `@pytest.mark.integration`.
- Typical runs: `make test-cov-python`, `pnpm test --run`, `pnpm test:e2e` (from `packages/luxar-viewer/`).

## Commit & Pull Request Guidelines
- Commit messages follow `<type>: <summary>` (examples seen: `fix: ...`, `refactor: ...`, `docs: ...`).
- PRs should describe the change, list tests run, and include screenshots for viewer/UI changes.
- Update `CHANGELOG.md`, `CLAUDE.md`, and `docs/guides/user/LUXAR_ZARR_FORMAT.md` when the change affects them.

## Agent-Specific Notes
- Keep docs in sync with code changes and follow gotchas in `CLAUDE.md` for tests and data formats.
