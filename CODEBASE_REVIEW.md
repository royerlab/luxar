# Luxar Codebase Review

## Scope
- Reviewed: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `Makefile`, `pyproject.toml`, `TODO.md`.
- Source: `packages/luxar/src/luxar` (core, io, encoding, validation, gsplats), `packages/luxar-viewer/src` (data, cache, themes, utils), tests and docs.
- Tests were not executed for this review.

## Findings (ordered by severity)

### High
1. Optional dependency import failure in `luxar.gsplats`
   - `luxar.gsplats` eagerly imports `torch` (and relies on SciPy in other submodules), but these packages are only in dev/test extras. A plain `pip install luxar` will fail on `import luxar.gsplats`.
   - Recommendation: add a `luxar[gsplats]` extra and/or lazy-import with a clear error if dependencies are missing.
   - refs: `packages/luxar/src/luxar/gsplats/__init__.py:1-4`, `packages/luxar/src/luxar/gsplats/fit_gsplats.py:7-9`

### Medium
1. Public API documentation mismatch for `create_scene`
   - The top-level docstring shows `compiler.create_scene()` with no arguments, but the implementation requires `dimensions` and raises if omitted. This will mislead users.
   - Recommendation: update examples to show `Dimensions(...)` or use a default helper.
   - refs: `packages/luxar/src/luxar/__init__.py:17-27`

2. Broadcasted arrays only expand when `expectedElements` is provided
   - `ArrayDecoder.decode` only handles broadcasted arrays if `expectedElements` is passed. Any call site that omits it will return a 1-element array instead of the expanded data.
   - Recommendation: infer the target length from metadata when `expectedElements` is missing, or raise a clear error.
   - refs: `packages/luxar-viewer/src/data/array-decoder.ts:159-183`

3. `extend_to_all` accepts arbitrary dimension names without validation
   - `Scene.add_points` / `Scene.add_lines` pass `extend_to_all` through without checking against scene dimensions, which can silently produce metadata that the viewer cannot apply.
   - Recommendation: validate names against `Dimensions.names` and fail fast or warn.
   - refs: `packages/luxar/src/luxar/core/scene.py:206-236`, `packages/luxar/src/luxar/core/scene.py:357-387`

4. Documentation completeness requirements not met
   - Per `CLAUDE.md`, every Python subpackage should include `README.md` and `SPECIFICATIONS.md`, and every TS package should include a README in `src/{package}`. Multiple packages are missing these files.
   - Recommendation: either add the missing docs or document exceptions (tests/demos).
   - refs: `CLAUDE.md` requirements; see Appendix A + Appendix B.

### Low
1. Logging style violations (non-doc code uses `print` / `console.*`)
   - Python CLI uses `print`, and viewer modules use direct `console.*` calls instead of `arbol` / `utils/log`, which diverges from repo standards.
   - Recommendation: route through `aprint` or `log`, or gate debug logs behind a flag.
   - refs: `packages/luxar/src/luxar/cli/main.py:960-1045`, `packages/luxar-viewer/src/themes/theme-manager.ts:132,435-462`, `packages/luxar-viewer/src/cache/decompressed-chunk-cache.ts:101-160`, `packages/luxar-viewer/src/data/lines-spatial-index-loader.ts:1159-1260`

2. Incremental view updates are still TODO
   - `PointSpatialIndexLoader.updateView` always reloads data, which may limit interactivity for large datasets.
   - refs: `packages/luxar-viewer/src/data/point-spatial-index-loader.ts:625-628`

3. Blosc-based fixture tests are effectively blocked
   - Unit tests for broadcasted arrays note a Blosc decompression TODO in Node, leaving those test cases unverified in CI.
   - refs: `packages/luxar-viewer/src/tests/unit/data/array-decoder.test.ts:63-99`

## Observations
- Strong documentation and specifications coverage in core modules (`core/`, `io/`, `encoding/`).
- Build/test workflows are clear and well documented (Hatch + Makefile + pnpm).
- TODO list is kept current and includes several open viewer concerns (cache eviction policy, shader blending).

## Appendix A: Missing Python package docs (README/SPECIFICATIONS)
The following packages (detected via `__init__.py` presence) are missing README and/or SPECIFICATIONS files:
- `packages/luxar/src/luxar` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/cli/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/core/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/demos` (missing SPECIFICATIONS)
- `packages/luxar/src/luxar/encoding/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/clahe/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/demos` (missing SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/fitting/dynamic_ops` (missing SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/fitting/dynamic_ops/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/fitting/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/io/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/models/gsplats` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/models/gsplats/metal/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/models/gsplats/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/models/utils` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/models/utils/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/multiscale/demos` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/multiscale/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/optim/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/seeds/demos` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/seeds/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/tests` (missing SPECIFICATIONS)
- `packages/luxar/src/luxar/gsplats/utils/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/io/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/typing_utils/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/utils/tests` (missing README, SPECIFICATIONS)
- `packages/luxar/src/luxar/validation/tests` (missing README, SPECIFICATIONS)

## Appendix B: Missing TypeScript package READMEs
- `packages/luxar-viewer/src/profiling`
- `packages/luxar-viewer/src/styles`
- `packages/luxar-viewer/src/themes`
- `packages/luxar-viewer/src/wasm`
- `packages/luxar-viewer/src/workers`
