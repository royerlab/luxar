# Repository Guidelines

## Project Structure & Module Organization
- `packages/luxar/src/luxar/` contains the Python compiler (core/, io/, encoding/, validation/, utils/, gsplats/).
- `packages/luxar-viewer/` is the WebGL/TypeScript viewer; `src/` holds the renderer and UI packages.
- `packages/luxar/examples/` and `packages/luxar/src/luxar/demos/` provide runnable sample scripts.
- `docs/` is Sphinx documentation (guides, tutorials, API .rst files).
- `scripts/`, `datasets/`, and `stats/` host utilities and supporting data.

## Build, Test, and Development Commands
- `make setup-dev` sets up the full dev environment (Python + viewer tooling).
- `make check-all` runs formatting, linting, type checks, and tests.
- `make test-all` runs all tests (Python + TypeScript + WASM); `make test-python` runs Python only.
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

## GSplats Subsystem
- `packages/luxar/src/luxar/gsplats/` is the Gaussian splatting subsystem with these subpackages:
  - `fitting/` — Modular 6-stage fitting pipeline with `DynamicOpsConfig` for splat add/remove
  - `seeds/` — 4 seeding strategies: decomposition, edges, grid, peaks (GPU-accelerated)
  - `models/gsplats/` — PyTorch `GaussianSplatModel` with CUDA and Metal backends
  - `batch/` — HPC/Slurm batch orchestration (plan, status, merge, validate, cancel)
  - `rendering/` — Render splats back to volume arrays (`render_to_volume`)
  - `preprocessing/` — NLM denoising with CUDA/PyTorch/skimage backends
  - `io/` — Save/load `.gsplats.zarr` format
  - `optim/` — Per-splat Adam optimizer with gradient dilution compensation
  - `clahe/` — Contrast-limited adaptive histogram equalization
  - `multiscale/` — Hierarchical multiscale decomposition
  - `culling.py`, `metrics.py`, `gpu_profile.py`, `tiling.py` — Standalone utilities
- Install optional deps: `pip install "luxar[gsplats]"` (adds PyTorch, scipy, etc.)

## CUDA / Metal Backends
- CUDA: `packages/luxar/src/luxar/gsplats/models/gsplats/cuda/` — splat-centric forward/backward kernels
  - Build: `make build-cuda` (local GPU) or `make build-cuda SLURM=1` (HPC)
  - Tests: `make test-cuda`
- Metal: `packages/luxar/src/luxar/gsplats/models/gsplats/metal/` — Apple GPU splatting
- NLM CUDA: `packages/luxar/src/luxar/gsplats/preprocessing/cuda/` — CUDA NLM denoising extension

## CLI Modules
- `packages/luxar/src/luxar/cli/main.py` — Core commands: `serve`, `demo`, `info`, `profiles`, `export`
- `packages/luxar/src/luxar/cli/gsplat_commands.py` — GSplat commands: `fit`, `convert`, `render`, `compare`, `merge`, `split`, `slice`, `filter`, `cull`, `info`, `view`, `benchmark`, `batch`, `transform`, `denoise`, `napari`
- `packages/luxar/src/luxar/cli/gsplat_config.py` — Config loading/validation for gsplat commands
- `packages/luxar/src/luxar/cli/export.py` — Standalone viewer export

## Agent-Specific Notes
- Keep docs in sync with code changes and follow gotchas in `CLAUDE.md` for tests and data formats.
