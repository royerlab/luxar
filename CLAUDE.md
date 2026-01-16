# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Quick Reference

### Python (use Hatch)
```bash
hatch run test              # Run tests
hatch run test-cov          # Tests with coverage
hatch run python script.py  # Run script
hatch run python -m ruff check .  # Lint
hatch run mypy packages/luxar/src/luxar/  # Type check
```

### TypeScript (use pnpm, from packages/luxar-viewer/)
```bash
pnpm dev          # Dev server (port 5173)
pnpm build        # Build
pnpm test --run   # Unit tests
pnpm test:e2e     # E2E tests (Playwright)
pnpm typecheck    # Type check
pnpm lint         # Lint
pnpm format       # Format
```

### Make Commands (from project root)
```bash
# Development Setup
make setup-dev    # Complete development environment setup (auto-installs dependencies)
make check-deps   # Check what dependencies are installed/missing
make install-rust # Install Rust + wasm-pack for viewer builds
make clean-setup  # Remove ALL dev tools to simulate fresh machine

# Quality & Testing
make test-all     # All tests (Python + TypeScript + WASM)
make test-python  # Python tests only
make test-e2e     # Playwright E2E tests
make check-all    # All quality checks
make check-rust   # Rust type/lint checks (cargo check + clippy)
make format-all   # Format all code (Python, TypeScript, Rust, CUDA)

# Viewer
make viewer       # Start viewer dev server (port 5173)
make build-viewer # Build viewer for production (requires Rust)
make build-wasm   # Build WASM module only
make test-wasm    # Run Rust unit tests
make benchmark-wasm  # Run WASM vs TypeScript performance benchmarks

# Data & Examples
make run-examples # Generate example datasets
make demo         # Generate demo dataset only (use 'luxar demo' to also serve)

# CUDA (Gaussian Splatting)
make setup-cuda       # Install CUDA deps + build extension (may need sudo)
make check-cuda-deps  # Check CUDA dependencies (nvcc, PyTorch CUDA, etc.)
make build-cuda       # Build CUDA splatting extension
make test-cuda        # Run CUDA tests
make benchmark-cuda   # Run performance benchmarks
make clean-cuda       # Clean CUDA build artifacts

# Utilities
make clean-all    # Clean all artifacts
make clean-viewer # Clean viewer artifacts only
make help         # Show all available commands
```

### Development Environment Setup

The build system is designed to work on **fresh Linux/macOS machines** with minimal pre-installed tools.

**Prerequisites:**
- Python 3.9+ (usually pre-installed)
- Git and curl
- **Ubuntu/Debian only**: `sudo apt-get install -y pipx && pipx ensurepath`

**What `make setup-dev` installs (no sudo needed):**
- **Node.js 22+**: via nvm (Linux) or Homebrew (macOS)
- **pnpm**: TypeScript package manager
- **Hatch**: Python environment manager (via pipx)
- **Pre-commit hooks**: Automatic code quality checks

**Key tools and their locations:**
| Tool | Installation | Location |
|------|--------------|----------|
| nvm | Auto-installed | `~/.nvm/` |
| Node.js | Via nvm | `~/.nvm/versions/node/` |
| Hatch | Via pipx | `~/.local/bin/hatch` |
| pnpm | Via npm | Global npm package |
| Rust/wasm-pack | `make install-rust` | `~/.cargo/` |
| CUDA toolkit | Manual install | `/usr/local/cuda/` (typical) |
| CUDA extension | `make build-cuda` | `packages/luxar/.../cuda/*.so` |

**Troubleshooting:**
```bash
# Check what's installed
make check-deps

# If pipx/hatch issues on Ubuntu
pipx reinstall hatch
pipx ensurepath
source ~/.bashrc

# If Node.js not found after nvm install
source ~/.nvm/nvm.sh
# or restart terminal

# Full reset and reinstall
make clean-setup
make setup-dev
```

See `docs/guides/developer/BUILD_SYSTEM_SPEC.md` for complete documentation.

### Luxar CLI
```bash
luxar demo                       # Quick demo with viewer
luxar serve <data.zarr> --viewer # Serve with viewer
luxar info <data.zarr> --stats   # Dataset info
luxar profiles                   # Network simulation profiles
```

### Make Command Nomenclature

The Makefile follows consistent naming conventions with **action-first** pattern:

| Pattern | Purpose | Examples |
|---------|---------|----------|
| `install-<tool>` | Install external tool on system | `install-node`, `install-rust`, `install-hatch` |
| `install-<component>-deps` | Install dependencies from manifest | `install-viewer-deps` (node_modules) |
| `install-dev` | Install Luxar package in editable mode | `install-dev` (pip install -e .) |
| `setup-<component>` | Orchestrated multi-step setup | `setup-dev`, `setup-cuda` |
| `enable-<feature>` | Activate/enable a feature | `enable-pre-commit` (activate hooks) |
| `build-<component>` | Compile/build artifacts | `build-viewer`, `build-wasm`, `build-cuda` |
| `test-<scope>` | Run tests | `test-all`, `test-python`, `test-e2e` |
| `check-<aspect>` | Verify/check something | `check-deps`, `check-all`, `check-cuda-deps` |
| `clean-<scope>` | Clean build artifacts | `clean-all`, `clean-viewer`, `clean-cuda` |
| `format-<language>` | Format code | `format-python`, `format-typescript` |
| `run-<script>` | Run scripts/examples | `run-examples`, `run-demos` |
| `serve-<target>` | Start a server | `serve-docs`, `serve-dataset` |

**Key distinctions:**
- `install-<tool>` vs `install-<component>-deps`: Tools are executables (node, rust); deps are project dependencies (node_modules)
- `install-*` vs `setup-*`: Install is for single components; setup orchestrates multiple steps
- `install-dev` vs `install-<tool>`: install-dev is for Luxar package itself; install-<tool> is for external tools
- `enable-*` vs `install-*`: Enable activates already-installed features; install adds new software

---

## Project Structure

```
/packages/luxar/           # Python package
  /src/luxar/              # Source (core/, io/, utils/, validation/, typing_utils/)
  /tests/                  # Python tests
  /examples/               # Example scripts (*_example.py naming)

/packages/luxar-viewer/    # TypeScript/WebGL viewer
  /src/                    # Source with per-package READMEs

/docs/                     # Documentation
  /guides/                 # Organized guides by purpose
    /user/                 # User guides (format, HDR, testing)
    /developer/            # Developer guides (style, console, network)
    /specs/                # Technical specs (cache, dimensions, lines)
  /templates/              # Templates (SPECIFICATIONS_TEMPLATE.md)
  /api/                    # Sphinx API reference files (.rst)
  /concepts/               # Architecture and concepts (.rst)
  /tutorials/              # Step-by-step tutorials (.rst)
  index.rst, conf.py       # Sphinx configuration
```

### Documentation Requirements

**Every Python subpackage MUST have**:
- `README.md` - Purpose, key classes, usage examples
- `SPECIFICATIONS.md` - Algorithms, data structures, behavior specification (use template in `docs/templates/`)

**Every TypeScript package has**:
- `README.md` in `/src/{package}/` - Keep in sync with code changes

---

## Code Standards

### Python
- Use type hints for all parameters and return values
- Format with ruff (88 char line length)
- Use PyTest (not unittest), mock only as last resort
- **Use Arbol for console output**: Replace `print()` with `aprint()`, use `asection()` for hierarchical output

```python
from arbol import aprint, asection

with asection("Processing"):
    aprint("Step 1...")
    aprint("Step 2...")
```

### TypeScript
- Format with prettier
- Use JSDoc comments
- Unified config in `src/config/` (camelCase, not UPPER_SNAKE_CASE)
- Console logging: `import { log } from '../utils/log'` with format `[emoji] [Module] message`
- Prefix unused variables with underscore

---

## Testing

### Strategy
- **Minimum coverage**: 80%
- **NEVER skip tests** - fix them or create proper mocks
- **Run before committing**: `make test-all && make check-all`

### Python Tests
```bash
hatch run test                    # All tests
hatch run pytest path/to/test.py  # Single file
hatch run test-cov                # With coverage
```

### TypeScript Unit Tests
```bash
cd packages/luxar-viewer
pnpm test --run                   # All unit tests
pnpm test path/to/test.ts         # Single file
```

### E2E Tests (Playwright)
```bash
cd packages/luxar-viewer
pnpm test:e2e                     # All E2E tests (~17 min)
pnpm test:e2e:ui                  # Interactive mode
pnpm agent:debug                  # AI debugging (see console logs)
pnpm agent:debug:visible          # AI debugging with visible browser
```

**Running E2E tests in chunks (RECOMMENDED):**
Instead of running all E2E tests at once (which can timeout or be overwhelming), run them by topic:
```bash
# Basic functionality
npx playwright test basic-rendering.spec.ts data-loading.spec.ts

# Scene & transforms
npx playwright test scene-integration.spec.ts transform-hierarchy.spec.ts

# nD navigation & dimensions
npx playwright test nd-navigation.spec.ts dimension-initialization.spec.ts dimension-animation.spec.ts

# Worker & WASM
npx playwright test worker-wasm-integration.spec.ts

# Test fixtures (run generate-fixtures first!)
pnpm test:generate-fixtures
npx playwright test test-fixtures-rendering.spec.ts

# Keyboard & input
npx playwright test keyboard-input-system.spec.ts controls-interaction.spec.ts

# Visual regression
npx playwright test visual-regression.spec.ts theme-visual-regression.spec.ts
```

**Key E2E rules**:
- Use `?src=<dataset>&debug` URL format (NOT `?data=`)
- Use 3D datasets for general tests (4D/nD slicing may show 0 points)
- Wait for `window.__luxarDebug` before assertions
- Run `pnpm test:generate-fixtures` before test-fixtures tests
- See `docs/guides/user/E2E_TESTING_GUIDE.md` and `docs/guides/developer/PLAYWRIGHT_GUIDE.md` for details

### Cross-Language E2E Testing
Python encoder and TypeScript decoder must stay in sync:
1. **Fixture-based**: Python generates zarr, TypeScript unit tests verify (fast, no browser)
2. **Playwright**: Full pipeline through browser (catches WebGL/rendering bugs)

When to run E2E:
- After changing encoding format
- After changing decoder
- Before PR/merge (always run full suite)

---

## AI-Assisted Debugging

When debugging viewer issues, use the Playwright agent driver:

```bash
cd packages/luxar-viewer
pnpm agent:debug
```

**Output includes**:
- `[BROWSER-CONSOLE-*]` - All browser console logs
- JSON state dump - Three.js scene, point counts, camera
- `test-results/debug/debug-view.png` - Screenshot

**Debug workflow**:
1. Run `pnpm agent:debug` to see current state
2. Add `console.log()` if needed
3. Run again to verify fix
4. Remove debug logging when done

**Available at `window.__luxarDebug`** (when `?debug` in URL):
- `scene`, `camera`, `renderer`, `controls`
- `getState()`, `renderOnce()`, `app`, `consoleInterceptor`

---

## Critical Gotchas

### Matrix Storage: NumPy vs THREE.js
NumPy uses row-major, THREE.js uses column-major. **Always transpose when serializing**:
```python
# Writing to zarr for THREE.js
matrix.T.ravel().tolist()

# Reading back in Python
np.array(flat_list).reshape(4, 4).T
```
Translation is at `[3,7,11]` in NumPy but `[12,13,14]` in THREE.js.

### Constructor Initialization Order
When subclass and parent both set the same attribute, **parent must initialize first**:
```python
def __init__(self):
    super().__init__()  # First!
    self._metadata = {...}  # Then subclass sets it
```

### Transform Composition Order
`compose(T1, T2, T3)` applies T1 first, T3 last (right-multiply):
```python
result = result @ transform  # Correct
# NOT: result = transform @ result
```

### nD Datasets in Tests
- 4D/nD datasets may show 0 points depending on slice position
- Use 3D datasets for general-purpose loading tests
- For nD tests, navigate to slices known to have points

### Data Source URLs Must NOT Have Trailing Slash
When loading data via URL, **never include a trailing slash**:
```bash
# ❌ WRONG - trailing slash breaks data loading
http://localhost:5173/?src=http://127.0.0.1:8005/

# ✅ CORRECT - no trailing slash
http://localhost:5173/?src=http://127.0.0.1:8005
```
The zarr loader interprets trailing slashes as path components, causing 404s.

### WASM 16-Dimension Limit
WASM functions use fixed-size arrays (for performance) and support **maximum 16 dimensions**.
- Functions affected: `calculate_effective_radii`, `mahalanobis_distance`, `compute_gsplats_attenuation`
- Error message: `"ndim=X exceeds maximum supported dimensions (16)"`
- For >16D data: TypeScript fallback is used automatically (slower but works)
- If you need >16D with WASM performance, reduce dimensions via PCA or feature selection

### ViewState.dimensions for extend_to_all
The `dimensions` field in ViewState is **required** for `extend_to_all` to work:
```typescript
// If extend_to_all is set but dimensions is undefined, the optimization is silently skipped
const viewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 5],
  dimensions: dims,  // REQUIRED for extend_to_all!
};
```

---

## Common Pitfalls and Solutions

### TypeScript: Event Listener Memory Leaks
**Problem**: Creating new bound function references on each call prevents proper cleanup.

```typescript
// ❌ WRONG - Creates new reference, removeEventListener won't work
addEventListener('resize', this.handleResize.bind(this));
removeEventListener('resize', this.handleResize.bind(this));  // Different reference!

// ✅ CORRECT - Store bound reference for cleanup
this.boundHandleResize = this.handleResize.bind(this);
addEventListener('resize', this.boundHandleResize);
removeEventListener('resize', this.boundHandleResize);  // Same reference
```

### TypeScript: Async Initialization Race Conditions
**Problem**: Multiple callers triggering async initialization concurrently.

```typescript
// ❌ WRONG - Non-atomic check
if (!this.initPromise) {
  this.initPromise = this.initialize();  // Race: two callers can both enter
}

// ✅ CORRECT - Atomic lock with cleanup
if (this.initLock) return this.initPromise;  // Return existing promise
this.initLock = true;
try {
  this.initPromise = this.initialize();
  await this.initPromise;
} finally {
  this.initLock = false;  // Always clear lock
}
```

### TypeScript: Over-Mocking in Tests
**Problem**: Mocking entire classes defeats the purpose of testing.

```typescript
// ❌ WRONG - Tests verify mock behavior, not real code
vi.mock('../rendering/point-material', () => ({
  PointMaterial: vi.fn().mockImplementation(() => ({
    uniforms: { fov: { value: 60 } },
    dispose: vi.fn()
  }))
}));

// ✅ CORRECT - Mock only external dependencies, test real code
import { PointMaterial } from '../rendering/point-material';
// Let PointMaterial run real shader generation code
// Only mock THREE.ShaderMaterial if absolutely necessary
```

**Testing Principle**: Mock external dependencies (network, file system), not your own code. If code is hard to test without mocking, refactor for testability (dependency injection, pure functions).

---

## Luxar Conventions

### Physical Units
Support: nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au

### Point Attributes
- **positions**: Required (Float32, nD)
- **colors**: Optional (Uint8 or Float32 for HDR)
- **radii**: Optional (Float32)
- **sharpness**: Optional (Float32)

### Transforms
- 4x4 matrices stored as 16-element lists
- Transpose for THREE.js compatibility (see Critical Gotchas)
- Use `luxar.transforms` module (translate, rotate, scale, compose)

### Dimensions
- Define at Scene level using `Dimensions` and `Dimension` classes
- Include: name, unit, range, step, display status
- Step sizes used for keyboard navigation in viewer

### nD Navigation
- Keyboard: 1-9 selects dimension, `[`/`]` navigates
- Radius-based slicing: points visible based on nD hypersphere intersection

---

## Pre-commit Checklist

```bash
make test-all                    # All tests pass
make check-all                   # Linting, type checking
pnpm run format                  # Format TypeScript (from luxar-viewer/)
```

Before PR/merge:
- Full E2E suite: `cd packages/luxar-viewer && pnpm test:e2e`
- Update READMEs if functionality changed
- Update SPECIFICATIONS.md if algorithms changed
- Update LUXAR_ZARR_FORMAT.md if data format changed

---

## Development Philosophy

1. **No backwards compatibility burden** - Early-stage project, just change it
2. **Complete before perfect** - Avoid over-engineering
3. **Minimum viable solution** - Don't add features/refactoring beyond what's asked
4. **Test everything** - Never skip tests, fix or mock them properly
5. **Keep docs in sync** - Update READMEs and specs with code changes
6. **Use existing patterns** - Follow codebase conventions

### Naming Conventions
- Example files: `*_example.py` or `*_example.zarr`
- Temp files: Put in `delme/` directory
- Example outputs: Generated to `datasets/examples/` (via `get_examples_output_dir()`)
- Demo outputs: Generated to `datasets/demos/` (via `get_demos_output_dir()`)
- Never commit `.zarr` directories (in .gitignore)

---

## Detailed Documentation

| Topic | Location |
|-------|----------|
| Build System & Dev Setup | `docs/guides/developer/BUILD_SYSTEM_SPEC.md` |
| E2E Testing Quick Ref | `docs/guides/user/E2E_TESTING_GUIDE.md` |
| Playwright Full Guide | `docs/guides/developer/PLAYWRIGHT_GUIDE.md` |
| Data Format Spec | `docs/guides/user/LUXAR_ZARR_FORMAT.md` |
| HDR Color Guide | `docs/guides/user/HDR_GUIDE.md` |
| Network Simulation | `docs/guides/developer/NETWORK_SIMULATION_SPEC.md` |
| Console Logging Style | `docs/guides/developer/CONSOLE_OUTPUT_STYLE.md` |
| Changelog | `CHANGELOG.md` |
| Spec Template | `docs/templates/SPECIFICATIONS_TEMPLATE.md` |

---

## Architecture Overview

```
Python Data -> Luxar Core -> Zarr Archive -> Luxar Viewer -> WebGL -> Display
```

### Scene Graph
- Scene (root) contains Groups and Points
- Groups can nest (hierarchical)
- Transforms compose hierarchically (parent -> child)
- Points have positions (nD), colors, radii, sharpness

### Performance Targets
- 100K-10M elements for smooth interaction
- Chunk size: 32KB-1MB optimal
- Compression: Blosc with zstd level 3
