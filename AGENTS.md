# Repository Guidelines for Agents

Luxar is a high-performance Python + WebGL system for compiling and visualizing arbitrary-sized nD scientific scenes. It supports first-class **Points**, **Lines**, and **Gaussian Splats**.

## Core Principles
- **Other agents may be working in this repo**: do not delete, overwrite, stash, or reformat unrelated work.
- Prefer the minimum viable change; avoid unasked-for features or broad refactors.
- Keep documentation in sync with code changes (READMEs, format docs, CHANGELOG as applicable).
- Never skip or disable tests to make them pass; fix the issue or use proper mocks.
- Early-stage project: no strong backwards-compatibility burden when a clean change is needed.
- Ask the user when genuinely unsure about an important decision; when the `AskUserQuestion` tool is available, use it instead of inline prose questions for concrete choices.

## Project Layout
- `packages/luxar/` — Python package.
  - `src/luxar/` — core source: `core/`, `io/`, `encoding/`, `validation/`, `utils/`, `typing_utils/`, `colormaps/`, `gsplats/`, `cli/`, `demos/`.
  - `src/luxar/**/tests/` — colocated pytest tests.
  - `examples/` — runnable examples; names use `*_example.py`.
- `packages/luxar-viewer/` — TypeScript/WebGL viewer; `src/` contains renderer/UI packages and per-package READMEs.
- `docs/` — Sphinx docs: `guides/user`, `guides/developer`, `guides/specs`, `api`, `concepts`, `tutorials`, `templates`.
- `scripts/`, `datasets/`, `stats/` — utilities and supporting/generated data.

## Environment Setup
- Use `make setup-dev` from the repo root for a fresh Linux/macOS machine, including HPC/Slurm login nodes with no sudo/GPU.
- Prerequisites: Python 3.10+, Git, curl; Git LFS is optional but required for demo data (`git lfs install && git lfs pull`).
- Ubuntu/Debian: install `pipx` when available (`sudo apt-get install -y pipx && pipx ensurepath`). HPC/no-sudo falls back to venv-based installs.
- `make setup-dev` installs Node.js 20.19+ (the 22 LTS by default), pnpm, Hatch, and pre-commit hooks without sudo.
- Tool locations commonly used by setup: `~/.nvm/`, `~/.local/bin/hatch`, `~/.local/bin/pnpm`, `~/.cargo/`, `~/.local/go/`, CUDA under `/usr/local/cuda/` or loaded modules.
- Troubleshooting: `make check-deps`; for full reset use `make clean-setup && make setup-dev`.
- If demos fail from missing `.npz`/`.zip`, pull Git LFS files and check `packages/luxar/src/luxar/demos/data/README.md`.

### HPC / Slurm CUDA Setup
```bash
make setup-dev
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc if needed
python scripts/test_hpc_setup.py
make build-cuda SLURM=1
make build-cuda SLURM=1 SLURM_PARTITION=gpu
make build-cuda SLURM=1 CUDA_MODULE=cuda/12.8.0_570.86.10
tail -f build-cuda-logs/build_<JOB_ID>.out
make test-cuda
```
`make build-cuda SLURM=1` detects PyTorch/CUDA and GCC modules, captures the Hatch venv, generates `build-cuda-logs/build_cuda_job.sh`, submits it, preserves `LD_LIBRARY_PATH`, and logs full build output.

## Common Commands

### Python (Hatch)
```bash
hatch run test
hatch run test-cov
hatch run pytest path/to/test.py
hatch run python script.py
hatch run python -m ruff check .
hatch run mypy packages/luxar/src/luxar/
```

### Viewer TypeScript (from `packages/luxar-viewer/`)
```bash
pnpm dev          # dev server, port 5173
pnpm build
pnpm test --run
pnpm test path/to/test.ts
pnpm test:e2e
pnpm test:e2e:ui
pnpm typecheck
pnpm lint
pnpm format
pnpm agent:debug
pnpm agent:debug:visible
```

### Make (repo root)
- Setup/tools: `make setup-dev`, `make check-deps`, `make install-rust`, `make install-go`, `make clean-setup`, `make help`.
- Quality/tests: `make test-all`, `make test-cov-all`, `make test-python`, `make test-e2e`, `make check-all`, `make check-rust`, `make format-all`.
- Viewer/WASM: `make viewer`, `make build-viewer`, `make build-wasm`, `make test-wasm`, `make benchmark-wasm`.
- Data/examples: `make run-examples`, `make demo`.
- Native launchers: run `make build-launchers` before `luxar export --native`; clean with `make clean-launchers`. Use `LUXAR_LAUNCHER_NO_WEBVIEW=1 ./luxar-launcher` for headless/minimal WebView environments.
- CUDA: `make setup-cuda`, `make check-cuda-deps`, `make build-cuda`, `make build-cuda SLURM=1`, `make build-cuda-slurm`, `make test-cuda`, `make benchmark-cuda`, `make clean-cuda`.
- Cleanup: `make clean-all`, `make clean-viewer`.

Make targets follow action-first naming: `install-<tool>`, `install-<component>-deps`, `install-dev`, `setup-<component>`, `enable-<feature>`, `build-<component>`, `test-<scope>`, `check-<aspect>`, `clean-<scope>`, `format-<language>`, `run-<script>`, `serve-<target>`.

## CLI Reference

### Core Luxar CLI
```bash
luxar demo
luxar serve <data.zarr> --viewer
luxar viewer
luxar viewer --data <data.zarr>
luxar viewer --data <data.zarr> --profile 3g
luxar info <data.zarr> --stats
luxar profiles
luxar export scene.zarr -o my_export/
luxar export scene.zarr -o my_export/ --open
luxar export scene.zarr -o my_export/ --overwrite
luxar export scene.zarr -o out/ --native macos
luxar export scene.zarr -o out/ --native macos,linux-amd64,linux-arm64 --name MyScene
```
CLI modules:
- `packages/luxar/src/luxar/cli/main.py` — `serve`, `viewer`, `demo`, `info`, `profiles`, `export`.
- `packages/luxar/src/luxar/cli/export.py` — standalone viewer export.
- `packages/luxar/src/luxar/cli/gsplat_commands.py` — gsplat subcommands.
- `packages/luxar/src/luxar/cli/gsplat_config.py` — gsplat config loading/validation.

### GSplat CLI
Supported fit inputs: `.zarr`, `.zarr.zip`, `.tiff`, `.npy`, `.npz`. Presets: `draft`, `standard`, `hifi`, `ultra`.
```bash
# Fit / config / tiled fitting
luxar gsplat fit volume.tiff splats.gsplats.zarr --preset standard --seeds 8000
luxar gsplat fit volume.npy splats.gsplats.zarr --config params.yaml
luxar gsplat fit data.zarr.zip splats.gsplats.zarr --timepoint 0 --channel 0
luxar gsplat fit data.zarr.zip splats.gsplats.zarr --array-key h2afva/fused
luxar gsplat fit volume.tiff splats.gsplats.zarr --preset hifi --iters 8000
luxar gsplat fit --dump-config --preset hifi > config.yaml
# Tiling: --tiling auto (default) picks none/uniform/content. Tiled fits emit a
# kind=partition by default (one part per tile/box); --flat for a single leaf.
luxar gsplat fit large.zarr splats.gsplats.zarr --tiling uniform --tile-size 256 --overlap 32
luxar gsplat fit large.zarr tile_3.gsplats.zarr --tile 3/16 --tile-size 256 --overlap 32
luxar gsplat fit vol.zarr out.gsplats.zarr --tiling content --cal cal.json   # content-adaptive boxes
luxar gsplat fit vol.zarr plan.json --tiling content --cal cal.json --plan-only  # emit box plan, no fit

# Slurm fitting: submit/status/merge/validation/cancel (submit submits by default)
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --dry-run
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --preset draft
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --tile-size 256
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --parallel
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --tasks-per-job 5
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --axes time,camera,channel,z,y,x
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --array-key h2afva/fused --axes time,z,y,x
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --timepoints '::10' --channels '0:2'
luxar gsplat slurm-fit submit data.zarr.zip output/ -p gpu --iters 8000 --seeds 100000
luxar gsplat slurm-fit status output/
luxar gsplat slurm-fit merge output/
luxar gsplat slurm-fit validate output/
luxar gsplat slurm-fit validate output/ --fix
luxar gsplat slurm-fit cancel output/

# Benchmark / convert / render / compare
luxar gsplat benchmark --slurm --partition gpu
luxar gsplat benchmark --list
luxar gsplat convert splats.gsplats.zarr scene.zarr --center
luxar gsplat render splats.gsplats.zarr rendered.npy --shape 128,128,128
luxar gsplat compare fitted.gsplats.zarr original.tiff
luxar gsplat compare fitted.gsplats.zarr original.npy --output-json metrics.json --device cuda

# Split / merge / slice / transform
luxar gsplat split splats.gsplats.zarr output_dir/ --parts 4
luxar gsplat split splats.gsplats.zarr output_dir/ --indices "1000,5000"
luxar gsplat merge a.gsplats.zarr b.gsplats.zarr -o merged.gsplats.zarr
luxar gsplat merge t0.zarr t1.zarr -o 4d.zarr --as-dimension --values 0,1
luxar gsplat merge ch0.zarr ch1.zarr -o multi.zarr --channel-colors "#ff0080,#00ff00"
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr "0:50, :, 10:90"
luxar gsplat slice input.gsplats.zarr output.gsplats.zarr ":50, 20:80, :"
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --scale 4,1,1,1 --center
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --rotate-z 90
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --normalize-intensity 1.0
luxar gsplat transform in.gsplats.zarr out.gsplats.zarr --translate 0,100,0 --scale-intensity 0.5

# Denoise / inspect / cull / filter / view / napari
luxar gsplat denoise volume.zarr denoised.zarr
luxar gsplat denoise volume.zarr denoised.npy --h 0.03
luxar gsplat denoise data.zarr.zip out.zarr --channel 0 --timepoint 5 --denoise-2d
luxar gsplat napari splats.gsplats.zarr
luxar gsplat info splats.gsplats.zarr
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m cumulative -r 0.90
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr -m redundancy --shape 41,512,512
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr --target vol.npy
luxar gsplat cull input.gsplats.zarr culled.gsplats.zarr --target vol.npy -p 95
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --amplitude-min 0.1 --eccentricity-max 5
luxar gsplat filter splats.gsplats.zarr out.gsplats.zarr --bbox "0,50,0,50,0,50" --volume-max 100
luxar gsplat view splats.gsplats.zarr
```

## Coding Standards

### Python
- 4-space indentation; type hints for all public parameters and return values.
- Ruff formatting, 88-character lines.
- Use pytest, not unittest. Mock only external dependencies or as a last resort.
- Console output must use Arbol, not `print()`:
  ```python
  from arbol import aprint, asection

  with asection("Processing"):
      aprint("Step 1...")
  ```

### TypeScript
- Strict types, ESLint + Prettier formatting, JSDoc for complex logic.
- Use camelCase config keys in `src/config/`, not `UPPER_SNAKE_CASE`.
- Prefix unused variables with `_`.
- Use `utils/log` (`import { log } from '../utils/log'`) for viewer logging; format messages as `[emoji] [Module] message`.

### Naming and Generated Files
- Tests: `test_*.py`; examples: `*_example.py` or `*_example.luxar.zarr`.
- Put temporary files in `delme/`.
- Example outputs go to `datasets/examples/` via `get_examples_output_dir()`.
- Demo outputs go to `datasets/demos/` via `get_demos_output_dir()`.
- Never commit `.zarr` directories (ignored by git).

## Documentation Requirements
- Every Python subpackage must have a `README.md` with purpose, key classes, and usage examples.
- Every TypeScript package under `packages/luxar-viewer/src/{package}/` has a `README.md`; keep it synced with code.
- Update `CHANGELOG.md`, `CLAUDE.md`, and `docs/guides/user/LUXAR_ZARR_FORMAT.md` when the change affects them.
- Update READMEs for functionality and algorithm changes.

## Testing
- Minimum coverage: 80%.
- Run before committing: `make test-all && make check-all`.
- Python tests live in `packages/luxar/src/luxar/**/tests` and use pytest.
- Viewer unit tests live in `packages/luxar-viewer/src/tests/unit` (Vitest); E2E tests in `src/tests/e2e` (Playwright).
- Mark slow Python tests with `@pytest.mark.slow` and integration tests with `@pytest.mark.integration`.

### E2E Testing Rules
- Prefer chunked E2E runs over the entire suite when iterating; full suite before PR/merge.
- Use URL format `?src=<dataset>&debug`, never `?data=`.
- Data-source URLs must not have a trailing slash.
- Use 3D datasets for general loading tests; 4D/nD datasets may show 0 points depending on slice.
- For nD tests, navigate to slices known to contain points.
- Wait for `window.__luxarDebug` before assertions.
- Run `pnpm test:generate-fixtures` before test-fixtures rendering tests.
- See `docs/guides/user/E2E_TESTING_GUIDE.md` and `docs/guides/developer/PLAYWRIGHT_GUIDE.md`.

Recommended E2E chunks from `packages/luxar-viewer/`:
```bash
npx playwright test basic-rendering.spec.ts viewer-initialization.spec.ts
npx playwright test transform-hierarchy.spec.ts
npx playwright test nd-navigation.spec.ts dimension-initialization.spec.ts dimension-animation.spec.ts
npx playwright test worker-wasm-integration.spec.ts
pnpm test:generate-fixtures && npx playwright test test-fixtures-rendering.spec.ts
npx playwright test keyboard-input-system.spec.ts controls-interaction.spec.ts
npx playwright test visual-regression.spec.ts theme-visual-regression.spec.ts
npx playwright test geometry-types.spec.ts blending-modes.spec.ts colormap-system.spec.ts post-processing-pipeline.spec.ts rendering-controls.spec.ts ortho-mode.spec.ts
npx playwright test data-integrity.spec.ts dataset-switching.spec.ts real-dataset-loading.spec.ts url-parameters.spec.ts python-typescript-integration.spec.ts luxar-serve-integration.spec.ts
npx playwright test spatial-index-accuracy.spec.ts cache-system.spec.ts nd-transforms.spec.ts position-bounds-clipping.spec.ts
npx playwright test custom-gui-library.spec.ts layers-panel.spec.ts recording-panel.spec.ts mouse-interactions.spec.ts
npx playwright test error-recovery.spec.ts webgl-errors.spec.ts
npx playwright test data-monitor-metrics.spec.ts  # perf-tracking moved to `pnpm test:perf:e2e` opt-in suite
npx playwright test all-examples-smoke-test.spec.ts demo-validation.spec.ts first-time-ux.spec.ts
```

### Cross-Language Fixtures
- Python encoder and TypeScript decoder must stay in sync.
- Unit tests auto-generate missing zarr fixtures via Vitest `globalSetup`.
- Keep `src/tests/global-setup.ts` `EXPECTED_FIXTURES` synchronized with `tests/fixtures/generate_test_data.py`.
- Run E2E after changing the encoding format, decoder, or browser rendering pipeline.

### AI-Assisted Viewer Debugging
```bash
cd packages/luxar-viewer
pnpm agent:debug
```
Outputs include browser console logs (`[BROWSER-CONSOLE-*]`), JSON state dump, and `test-results/debug/debug-view.png`. With `?debug`, `window.__luxarDebug` exposes `scene`, `camera`, `renderer`, `controls`, `getState()`, `renderOnce()`, `app`, and `consoleInterceptor`. Remove temporary console logging before finishing.

## GSplats Subsystem
- Install optional dependencies with `pip install "luxar[gsplats]"` (PyTorch, scipy, etc.).
- `packages/luxar/src/luxar/gsplats/` contains:
  - `fitting/` — modular 6-stage fitting pipeline; `DynamicOpsConfig` for add/remove.
  - `seeds/` — decomposition, edges, grid, peaks; GPU-accelerated.
  - `models/gsplats/` — PyTorch `GaussianSplatModel` with CUDA and Metal backends.
  - `batch/` — HPC/Slurm orchestration: plan, status, merge, validate, cancel.
  - `rendering/` — render splats to volume arrays (`render_to_volume`).
  - `preprocessing/` — NLM denoising with CUDA/PyTorch/skimage backends.
  - `io/` — save/load `.gsplats.zarr`.
  - `optim/` — per-splat Adam with gradient dilution compensation.
  - `clahe/` — contrast-limited adaptive histogram equalization.
  - `multiscale/` — hierarchical multiscale decomposition.
  - `culling.py`, `metrics.py`, `gpu_profile.py`, `tiling.py` — standalone utilities.
- CUDA kernels: `packages/luxar/src/luxar/gsplats/models/gsplats/cuda/`; build with `make build-cuda` or `make build-cuda SLURM=1`; test with `make test-cuda`.
- Metal backend: `packages/luxar/src/luxar/gsplats/models/gsplats/metal/`.
- NLM CUDA extension: `packages/luxar/src/luxar/gsplats/preprocessing/cuda/`.

### GPU Seeding and Fitting
```python
from luxar.gsplats.seeds import generate_seeds
from luxar.gsplats import fit_gaussian_splats

seeds = generate_seeds(volume, device="auto")
seeds = generate_seeds(volume, device="cuda")
seeds = generate_seeds(volume, device="mps")
result = fit_gaussian_splats(volume, device="cuda", seed_method="edges")
```
- Benchmark with `hatch run python scripts/benchmarks/benchmark_seeding_gpu.py`.
- GPU support: Sobel gradients and deduplication in all dimensions; peak detection and interpolation in 2D/3D only with automatic fallback.
- Large volumes (>100³) can see substantial, sometimes orders-of-magnitude, speedups depending on GPU.

## Luxar Data and Scene Conventions
- Physical units: `nm`, `um`, `mm`, `cm`, `m`, `meter`, `metre`, `km`, `inch`, `foot`, `px`, `au`.
- Geometry attributes:
  - Points: `positions` (Float32, nD, required), `colors` (Uint8/Float32 HDR), `radii` (Float32), `sharpness` (Float32).
  - Lines: `vertices` (Float32, nD, required), `widths` (Float32, required), `segments` (Uint32, auto-generated), `colors` (Uint8/Float32), `sharpness` (Float32).
  - GSplats: `centers` (Float32, nD, required), `amplitudes` (Float32, required), `cholesky_factors` (Float32, required), `colors` (Uint8/Float32).
- Define dimensions at Scene level with `Dimensions` and `Dimension`: name, unit, range, step, display status. Step sizes drive viewer keyboard navigation.
- Viewer nD navigation: keys `1`-`9` select dimension, `[`/`]` navigate. Geometry visibility uses radius-based nD hypersphere slicing.
- 4x4 transforms are stored as 16-element lists; use `luxar.transforms` (`translate`, `rotate`, `scale`, `compose`).

## Critical Gotchas

### Matrix Storage: NumPy vs THREE.js
NumPy is row-major and THREE.js is column-major. Always transpose when serializing for THREE.js:
```python
matrix.T.ravel().tolist()                 # write
np.array(flat_list).reshape(4, 4).T       # read back in Python
```
Translation is at `[3,7,11]` in NumPy but `[12,13,14]` in THREE.js.

### Initialization and Transform Composition
- If subclass and parent set the same attribute, call `super().__init__()` first, then set subclass attributes.
- `compose(T1, T2, T3)` applies `T1` first and `T3` last via right-multiply: `result = result @ transform`, not `transform @ result`.

### URLs, WASM, and ViewState
- Never include trailing slash in data-source URLs, e.g. use `?src=http://127.0.0.1:8005`, not `?src=http://127.0.0.1:8005/`.
- WASM functions use fixed arrays and support at most 16 dimensions: `calculate_effective_radii`, `mahalanobis_distance`, `compute_gsplats_attenuation`. For >16D, TypeScript fallback is automatic but slower.
- `ViewState.dimensions` is required for `extend_to_all`; if undefined, the optimization is silently skipped.

### nD Transforms
- `nd_transform` is separate from the 4x4 `transform` and applies to non-displayed dimensions.
- Continuous/discrete transform form: `{ "scale": float, "offset": float }`.
- Categorical transform form: `{ "permutation": [int, ...] }`.
- Viewer uses inverse-query: transform the query (slice position + tolerance) from world to local once, instead of transforming all point coordinates. See `docs/guides/specs/ND_TRANSFORMS_SPEC.md`.

## TypeScript Pitfalls
- Event listeners: store bound function references; do not call `.bind(this)` separately in `addEventListener` and `removeEventListener`.
- Async initialization: protect concurrent initialization with an atomic lock/promise and clear locks in `finally`.
- Avoid over-mocking: mock external dependencies (network/filesystem), not project classes. If code is hard to test, refactor with dependency injection or pure functions.

## Pre-Commit / PR Checklist
```bash
make test-all
make check-all
cd packages/luxar-viewer && pnpm run format
```
Before PR/merge:
- Run full E2E: `cd packages/luxar-viewer && pnpm test:e2e`.
- Update READMEs if functionality or algorithms changed.
- Update `docs/guides/user/LUXAR_ZARR_FORMAT.md` if the data format changed.
- PRs should describe the change, list tests run, and include screenshots for viewer/UI changes.
- Commit messages follow `<type>: <summary>` (for example `fix: ...`, `refactor: ...`, `docs: ...`).

## Key Documentation
| Topic | Location |
| --- | --- |
| Build system and setup | `docs/guides/developer/BUILD_SYSTEM_SPEC.md` |
| E2E testing quick ref | `docs/guides/user/E2E_TESTING_GUIDE.md` |
| Playwright guide | `docs/guides/developer/PLAYWRIGHT_GUIDE.md` |
| Data format spec | `docs/guides/user/LUXAR_ZARR_FORMAT.md` |
| HDR color guide | `docs/guides/user/HDR_GUIDE.md` |
| Network simulation | `docs/guides/developer/NETWORK_SIMULATION_SPEC.md` |
| Console logging style | `docs/guides/developer/CONSOLE_OUTPUT_STYLE.md` |
| Changelog | `CHANGELOG.md` |

## Architecture and Performance
```text
Python Data -> Luxar Core -> Zarr Archive -> Luxar Viewer -> WebGL -> Display
```
- Scene graph: Scene root contains nested Groups, Points, Lines, and GSplats; transforms compose parent-to-child.
- Geometry: Points are soft-edged spheres, Lines are width-tapered curves, GSplats are oriented Gaussians.
- Performance targets: 100K-10M elements with smooth interaction; chunk size 32KB-1MB; Blosc zstd compression level 3.
