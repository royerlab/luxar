# Luxar Viewer Test Suite

This directory contains the comprehensive test suite for the Luxar Viewer TypeScript/WebGL application.

**Current Status**: 696/701 tests passing (99.3%), 5 skipped

---

## Quick Start

```bash
# Unit tests (fast, mocked dependencies)
pnpm test              # Watch mode
pnpm test --run        # Single run

# E2E tests (full browser, Playwright)
pnpm test:e2e          # All E2E tests
pnpm test:e2e:ui       # Interactive UI
pnpm test:e2e:report   # View HTML report

# With test fixtures (Python-generated datasets)
pnpm test:generate-fixtures   # Generate fixtures from Python
pnpm test:with-fixtures       # Generate + run tests

# Coverage
pnpm run test:coverage        # Generate coverage report
open coverage/typescript/index.html
```

---

## Test Organization

### Directory Structure

```
tests/
├── mocks/                         # Mock infrastructure (WebGL, Browser APIs, etc.)
│   ├── index.ts                   # Central export point
│   ├── webgl.mock.ts              # WebGL2RenderingContext mock (164 LOC)
│   ├── browser-apis.mock.ts       # window.matchMedia, ResizeObserver, etc. (90 LOC)
│   ├── opfs.mock.ts               # Origin Private File System mock (24 LOC)
│   ├── three.mock.ts              # THREE.js mocks (1524 LOC)
│   └── orbit-controls.mock.ts     # OrbitControls mock (57 LOC)
│
├── unit/                          # Unit tests (30 files, fast, isolated)
│   ├── data/                      # Data loading & encoding (8 tests)
│   ├── ndim/                      # nD slicing & spatial queries (3 tests)
│   ├── cache/                     # Caching system (5 tests)
│   ├── controls/                  # Camera controls & input (4 tests)
│   ├── rendering/                 # Materials, shaders, post-processing (5 tests)
│   ├── scene/                     # Scene management (3 tests)
│   └── architecture/              # Global state, clean architecture (2 tests)
│
├── e2e/                           # End-to-end tests (21 files, full browser)
│   ├── *.spec.ts                  # Playwright E2E test files
│   └── *-snapshots/               # Visual regression snapshots (auto-generated)
│
├── builders/                      # Test data builders & helpers
│   └── test-data-builders.ts      # Fluent builders for test data
│
├── setup.ts                       # Global test setup (installs mocks)
└── test-config.ts                 # Vitest configuration

# Test fixtures (auto-generated, Python → TypeScript compatibility)
../../tests/fixtures/              # At luxar-viewer root level
├── generate_test_data.py          # Python script to generate test datasets
├── test_4d.zarr                   # 4D points dataset
├── test_broadcasting.zarr         # Broadcasting encoding test
├── test_array_refs.zarr           # Array reference resolution test
└── ... (more fixtures)
```

---

## Mock Infrastructure

The test suite uses comprehensive mocks to run tests in Node.js without a browser. All mocks are organized in the `mocks/` directory.

### Mock Files

#### `mocks/index.ts` (Central Export)

- Re-exports all mocks
- Provides `installAllMocks()` convenience function
- Used in `setup.ts` to install mocks globally

```typescript
import { installAllMocks } from './mocks';
installAllMocks(); // Sets up all mocks at once
```

#### `mocks/webgl.mock.ts` (WebGL Context Mock)

**Purpose**: Mock WebGL2RenderingContext for Three.js rendering tests

**What it mocks**:

- WebGL constants (e.g., `gl.TRIANGLES`, `gl.FLOAT`)
- Shader compilation (`createShader`, `compileShader`, `linkProgram`)
- Buffer operations (`createBuffer`, `bufferData`)
- Texture operations (`createTexture`, `texImage2D`)
- WebGL extensions (HDR extensions, debug info)
- Canvas element

**Key features**:

- 164 lines of comprehensive WebGL API surface
- Supports HDR extension detection
- Tracks shader compilation for debugging
- Mock canvas with resize support

#### `mocks/browser-apis.mock.ts` (Browser APIs Mock)

**Purpose**: Mock browser APIs not available in Node.js

**What it mocks**:

- `window.matchMedia()` - HDR/P3 color space detection
- `ResizeObserver` - Canvas resizing
- `IntersectionObserver` - Visibility tracking
- `requestAnimationFrame` - Animation loops
- `performance.now()` - High-precision timing

**Usage**: Automatically installed via `installAllBrowserMocks()`

#### `mocks/opfs.mock.ts` (Origin Private File System Mock)

**Purpose**: Mock browser-native persistent storage

**What it mocks**:

- `navigator.storage.getDirectory()` - OPFS root access
- File handle operations (read, write, remove)
- Directory iteration

**Usage**: Enables cache tests without real browser storage

#### `mocks/three.mock.ts` (THREE.js Mock)

**Purpose**: Mock THREE.js classes for rendering tests

**What it mocks**:

- Core classes: `Scene`, `Camera`, `Renderer`, `Mesh`, `Geometry`, `Material`
- Point rendering: `Points`, `PointsMaterial`, `BufferGeometry`, `BufferAttribute`
- Post-processing: `EffectComposer`, `RenderPass`, `ShaderPass`
- Controls: Orbit/fly controls (see `orbit-controls.mock.ts`)

**Key features**:

- 1524 lines of comprehensive THREE.js API surface
- Tracks scene graph changes for assertions
- Mock shaders with uniform tracking
- Dispose tracking for memory leak detection

#### `mocks/orbit-controls.mock.ts` (OrbitControls Mock)

**Purpose**: Mock Three.js OrbitControls

**What it mocks**:

- `OrbitControls` class with camera manipulation
- Event system (`addEventListener`, `removeEventListener`)
- State properties (`enabled`, `target`, `dampingFactor`)

**Usage**: Lightweight mock for control system tests

### Why Mock?

**Reasons**:

1. **Speed**: Unit tests run in <5s (vs 30s+ for E2E)
2. **Isolation**: Test logic without browser/GPU dependencies
3. **Reliability**: No flaky GPU driver issues
4. **Debugging**: Easier to debug pure JavaScript logic

**Trade-offs**:

- Mocks can't catch browser-specific bugs (WebGL errors, OPFS issues)
- Visual correctness requires E2E tests
- Mock drift: Mocks must stay in sync with real APIs

**Solution**: Comprehensive E2E test suite (21 tests) covers full browser integration

---

## Test Categories

### 1. Data Loading & Encoding (`unit/data/`)

Tests for Python-TypeScript data compatibility and the complete loading pipeline.

**Scope**: Zarr loading, scene graph parsing, encoding/decoding (broadcasting, LUT, quantization, array_ref), spatial index loading

**Key Files**:

- `array-decoder.test.ts` - **Python ↔ TypeScript encoding compatibility**
  - Broadcasting: `(1, 3) → (1000, 3)` color expansion
  - LUT encoding: Indexed color/radius lookup
  - Quantization: Float32 → Uint16 compression
  - Array references: Shared data via `array_ref` metadata

- `encoded-range-extraction.test.ts` - **Critical bug fix**: Range extraction with encoded arrays
  - Bug: LUT-encoded arrays used wrong `elementsPerPoint` (1 instead of 3)
  - Result: Only 1/3 of points rendered, 2/3 appeared black
  - Tests: Verify range extraction uses correct `elementsPerPoint`

- `zarr-loader.test.ts` - Scene graph loading from zarr
  - Hierarchical transforms (parent → child)
  - Node naming and metadata
  - Dimension specifications

- `point-spatial-index-loader.test.ts` - Spatial index-based loading
  - Efficient range queries
  - Cell-based organization
  - nD spatial indexing

- `data-loading-integration.test.ts` - **Unit integration**: Loader pipeline with mocks
  - Tests internal logic with mocked zarr/THREE.js
  - Fast feedback loop

- `data-loading-monitor.test.ts` - Loading UI monitor unit tests
  - Progress tracking
  - Error handling
  - Cancellation

- `data-monitor-integration.test.ts` - **Unit integration**: Monitor + loader interaction with mocks
  - Event system verification
  - State synchronization

**Test Boundaries**:

- **Unit tests** (`unit/data/`): Mock zarr data, focus on logic
- **E2E tests** (`e2e/data-loading.spec.ts`): Real browser + real files

---

### 2. nD Slicing & Spatial Queries (`unit/ndim/`)

Tests for multi-dimensional (4D, 5D+) visualization and slicing.

**Scope**: nD → 3D projection, hypersphere slicing, dimension navigation utilities

**Key Files**:

- `ndim-calculation-projectTo3D.test.ts` - **Critical bug fix**: ndim calculation from positions array
  - Bug: Wrong ndim calculation (default 3) for 4D+ data without spatial index
  - Result: Block artifacts in rendering
  - Tests: Verify ndim calculated from actual positions array

- `effective-radius-calculator.test.ts` - Hypersphere slicing for nD points
  - Formula: `effectiveRadius = sqrt(radius² - distanceToSlice²)`
  - 4D+ visibility calculation
  - Edge cases (point on plane, behind plane)

- `nd-navigation-utils.test.ts` - Dimension navigation helpers
  - Step size calculation
  - Value formatting
  - Keyboard navigation utilities

**Why This Matters**: 4D+ datasets require special handling. These tests prevent regressions of critical bugs that cause rendering artifacts.

---

### 3. Caching System (`unit/cache/`)

Tests for memory management and performance optimization.

**Scope**: In-memory caching (LRU, segmented LRU), persistent storage (OPFS), two-level caching

**Key Files**:

- `range-cache.test.ts` - Range-based cache for point data
  - Range intersection detection
  - Partial overlap handling
  - Memory limits

- `lru-cache.test.ts` - LRU eviction strategy
  - Least-recently-used eviction
  - Access time tracking
  - Memory limits

- `segmented-lru-cache.test.ts` - Segmented LRU for better hit rates
  - Hot/warm/cold segments
  - Promotion/demotion logic
  - Workloads with temporal/spatial locality

- `opfs-store.test.ts` - OPFS (Origin Private File System) persistent cache
  - Browser-native storage API
  - Async file operations
  - Error handling

- `two-level-caching-store.test.ts` - Memory + OPFS integration
  - Hot data in memory (fast)
  - Warm data in OPFS (persistent)
  - Automatic tier management

**Cache Strategy**:

- **lru-cache**: Simple LRU for general use
- **segmented-lru-cache**: Segmented LRU for better locality
- **two-level-caching-store**: Hot data in memory, warm data in OPFS

---

### 4. Controls & Input (`unit/controls/`)

Tests for camera controls and input management.

**Scope**: Orbit/fly controls, keyboard/mouse input, input context management, conflict prevention

**Key Files**:

- `controls-manager.test.ts` - Control switching (orbit ↔ fly)
  - Mode switching
  - State preservation
  - Event handling

- `luxar-fly-controls.test.ts` - Fly controls physics
  - Inertial movement
  - Keyboard/mouse input
  - Collision detection

- `input-context-manager.test.ts` - **Input context system**
  - Prevents WASD conflicts with text fields
  - Context stack (global, UI, text input)
  - Automatic context switching

- `input-validation.test.ts` - Input validation utilities
  - Block shortcuts when typing
  - FOV calculation
  - Key press validation

**Key Innovation**: Input context system prevents control interference (e.g., pressing 'W' in text field doesn't move camera)

---

### 5. Rendering & Materials (`unit/rendering/`)

Tests for WebGL rendering, shaders, and post-processing.

**Scope**: Material management, point shaders, HDR rendering, post-processing

**Key Files**:

- `material-manager.test.ts` - Material creation and lifecycle
  - Material caching
  - Uniform updates
  - Dispose tracking

- `point-material.test.ts` - Point shader material
  - Vertex/fragment shaders
  - World-space sizing
  - HDR color support

- `postprocessing-manager.test.ts` - HDR post-processing pipeline
  - Tone mapping
  - Bloom effects
  - Exposure control

- `postprocessing-depth-mapping.test.ts` - Depth mapping for compositing
  - Depth buffer readback
  - Multi-pass rendering

- `rendering-controls-utils.test.ts` - Rendering control helpers
  - Parameter validation
  - UI integration

**Key Feature**: World-space point sizing (physical accuracy, FOV-independent)

---

### 6. Scene Management (`unit/scene/`)

Tests for scene graph and state management.

**Scope**: Scene orchestration, hierarchical transforms, dimension management

**Key Files**:

- `scene-manager.test.ts` - Scene orchestration and lifecycle
  - Scene initialization
  - Node hierarchy
  - Transform composition

- `scene-manager-utils.test.ts` - Scene utility functions
  - Transform helpers
  - Node traversal
  - Bounds calculation

- `scene-dims-manager.test.ts` - Dimension specification management
  - Scene-level dimensions
  - Display configuration
  - Step sizes for navigation

---

### 7. Architecture (`unit/architecture/`)

Tests for clean architecture and global state management.

**Scope**: Singleton managers, no global pollution, clean separation of concerns

**Key Files**:

- `global-state.test.ts` - **Verifies no global variable pollution**
  - No global THREE.js leaks
  - No global state accumulation
  - Clean test isolation

- `directory-navigator.test.ts` - File navigation UI
  - Directory listing
  - Path navigation
  - File selection

**Why This Matters**: Global state causes test interference and memory leaks

---

### 8. End-to-End Tests (`e2e/`)

Full browser tests using Playwright that exercise the complete pipeline.

**Scope**: Real browser, real WebGL, real OPFS, real datasets (21 test files)

**Key Tests**:

- `basic-rendering.spec.ts` - Smoke tests for rendering
- `all-examples-smoke-test.spec.ts` - **Validates all example datasets**
- `python-typescript-integration.spec.ts` - **Full Python → TypeScript pipeline**
- `visual-regression.spec.ts` - Snapshot testing for visual correctness
- `spatial-index-accuracy.spec.ts` - Spatial index correctness
- `nd-navigation.spec.ts` - nD slicing in browser
- `performance-benchmarks.spec.ts` - Performance tracking
- `error-recovery.spec.ts` - Error handling and recovery
- `scene-integration.spec.ts` - Scene loading integration
- `webgl-error-detection.spec.ts` - WebGL error handling

**Why E2E Tests Matter**:

- Unit tests mock WebGL/browser APIs → can't catch browser-specific bugs
- E2E tests catch: WebGL errors, OPFS issues, rendering glitches, GPU driver issues
- Visual regression tests prevent UI breakage
- **Critical**: Many bugs only manifest in real browsers (e.g., WebGL context loss, OPFS quota errors)

**Test Artifacts** (Auto-Generated, Not Committed):

- `test-results/` - Per-test screenshots, videos, traces
- `playwright-report/` - Interactive HTML report

**Viewing Results**:

```bash
pnpm test:e2e:report  # Opens HTML report with all screenshots/videos
```

---

## Test Fixtures

Test fixtures are Python-generated zarr datasets for Python-TypeScript compatibility testing.

### Location

```
packages/luxar-viewer/tests/fixtures/
├── generate_test_data.py          # Python script to generate fixtures
├── test_4d.zarr                   # 4D points dataset
├── test_broadcasting.zarr         # Broadcasting encoding test
├── test_array_refs.zarr           # Array reference resolution test
├── test_hdr_colors.zarr           # HDR color support test
├── test_hierarchical_transforms.zarr  # Transform composition test
└── ... (more fixtures)
```

### Generating Fixtures

```bash
# From luxar-viewer directory
pnpm test:generate-fixtures

# Or from repo root
make test-fixtures

# Generate + run tests
pnpm test:with-fixtures
make viewer-test-fixtures
```

### What Fixtures Test

**Encoding Compatibility** (Python → TypeScript):

1. **Broadcasting**: Python writes `(1, 3)` color array → TypeScript expands to `(N, 3)`
2. **LUT Encoding**: Python encodes 1000 colors as 10 unique values → TypeScript decodes
3. **Quantization**: Python quantizes Float32 → Uint16 → TypeScript dequantizes
4. **Array References**: Python writes shared array with `array_ref` → TypeScript resolves

**Scene Features**: 5. **4D/5D Data**: nD slicing, hypersphere visibility 6. **Hierarchical Transforms**: Parent-child transform composition 7. **HDR Colors**: Float32 colors > 1.0

### When to Regenerate

Regenerate fixtures when:

- Python encoding format changes
- New encoding modes added
- Scene metadata schema changes
- Dimension system changes

**Important**: Fixtures are checked into git to ensure consistent test data across machines

---

## Running Tests

### Unit Tests (Fast, Isolated)

```bash
# Watch mode (recommended for development)
pnpm test

# Single run (CI/pre-commit)
pnpm test --run

# Specific test file
pnpm test array-decoder

# Specific test category
pnpm test unit/data/          # All data tests
pnpm test unit/ndim/          # All nD tests
pnpm test unit/cache/         # All cache tests
```

### E2E Tests (Full Browser)

```bash
# Run all E2E tests
pnpm test:e2e

# Interactive UI (watch mode)
pnpm test:e2e:ui

# Debug mode (visible browser)
pnpm test:e2e:debug

# View HTML report (after running tests)
pnpm test:e2e:report
```

### Coverage

```bash
# Generate coverage report
pnpm run test:coverage

# View HTML report
open coverage/typescript/index.html

# Coverage requirements
# - Minimum: 80%
# - Critical paths: 100%
```

### Makefile Integration

```bash
# From repo root
make viewer-test                # Unit tests only
make viewer-test-fixtures       # Generate fixtures + run tests
make test-all                   # Python + TypeScript tests
```

---

## Writing Tests

### When to Write Unit Tests vs E2E Tests

**Unit Tests** (`unit/`):

- ✅ Fast feedback loop (<5s)
- ✅ Easy to debug (no browser)
- ✅ Test logic and algorithms
- ❌ Can't catch browser-specific bugs
- **Use for**: Encoding, decoding, math, state management

**E2E Tests** (`e2e/`):

- ✅ Full browser integration
- ✅ Catch visual/rendering bugs
- ✅ Test user workflows
- ❌ Slower (~30s)
- **Use for**: Rendering, OPFS, WebGL, visual regression

**Rule of Thumb**: Start with unit tests, add E2E tests for visual/browser-specific behavior

---

### Test File Naming

- **Unit tests**: `*.test.ts` (e.g., `array-decoder.test.ts`)
- **E2E tests**: `*.spec.ts` (e.g., `basic-rendering.spec.ts`)
- **Descriptive names**: Describe what's being tested, not generic names

---

### Test Structure

```typescript
/**
 * Clear description of what this test file covers
 *
 * Context:
 * - What's being tested
 * - Why it matters (if fixing a bug, explain the bug)
 * - Any critical dependencies or assumptions
 */

import { describe, it, expect } from 'vitest';

describe('Module/Feature Name', () => {
  describe('specific function or behavior', () => {
    it('should do something specific', () => {
      // Arrange: Set up test data
      const input = ...;

      // Act: Execute the code under test
      const result = functionUnderTest(input);

      // Assert: Verify the result
      expect(result).toBe(expected);
    });
  });
});
```

---

### Using Test Builders

Use test builders from `builders/test-data-builders.ts` for readable test setup:

```typescript
import { PointsBuilder, DimensionsBuilder } from '../builders/test-data-builders';

// Create 1000 4D points with colors and radii
const points = new PointsBuilder()
  .withPoints(1000)
  .withDimensions(4)
  .withRandomPositions([-10, 10])
  .withColors()
  .withRadii(0.5)
  .build();

// Create dimension configuration
const dims = new DimensionsBuilder()
  .withNDimensions(5)
  .withDisplayed(0, 1, 2) // x, y, z displayed
  .withDimension(3, 'time', 's', [0, 10])
  .build();
```

**Benefits**:

- Fluent, expressive API
- Type-safe
- Centralized test data generation
- Reduces boilerplate

---

### Using Mocks

Mocks are installed globally in `setup.ts`. Just import and use:

```typescript
import { vi } from 'vitest';
import * as THREE from 'three';

// THREE.js is already mocked, just use it
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera();
const renderer = new THREE.WebGLRenderer();

// For custom mocks in specific tests
const mockZarrStore = {
  get: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3])),
  // ...
};
```

**Important**: Check `mocks/` directory for available mocks before creating new ones

---

## Test Philosophy

### What We Test

**Unit Tests Focus On**:

1. **Logic correctness**: Encoding, decoding, math, algorithms
2. **Error handling**: Invalid inputs, edge cases
3. **State management**: No global pollution, clean isolation
4. **API contracts**: Public interfaces behave as documented

**E2E Tests Focus On**:

1. **Visual correctness**: Points render correctly
2. **Browser integration**: WebGL, OPFS, network requests
3. **User workflows**: Load dataset → navigate → interact
4. **Performance**: FPS, memory usage, load times

### What We Don't Test

- **Internal implementation details**: Only test public APIs
- **Third-party libraries**: Trust THREE.js, zarrita, etc.
- **Browser APIs**: Trust browser implementations (mock for unit tests)

### Critical Bug Documentation

Several tests document and prevent regressions of critical bugs:

#### 1. nD Calculation Bug (`ndim-calculation-projectTo3D.test.ts`)

- **Bug**: Wrong ndim calculation (default 3) for 4D+ data without spatial index
- **Impact**: Block artifacts in rendering
- **Root Cause**: Assumed 3D, didn't read from positions array
- **Fix**: Calculate ndim from positions array shape
- **Test**: Verify ndim calculated from actual data

#### 2. Encoded Range Extraction Bug (`encoded-range-extraction.test.ts`)

- **Bug**: LUT-encoded arrays used wrong `elementsPerPoint` (1 instead of 3)
- **Impact**: Only 1/3 of points rendered, 2/3 appeared black
- **Root Cause**: Didn't account for per-point element count
- **Fix**: Use correct `elementsPerPoint` from metadata
- **Test**: Verify range extraction with LUT/quantization

#### 3. Array Reference Bug (`array-decoder.test.ts`)

- **Bug**: Array references not resolved correctly
- **Impact**: Duplicated point data not shared (memory waste)
- **Root Cause**: Didn't follow `array_ref` metadata
- **Fix**: Resolve `array_ref` to target array before decoding
- **Test**: Verify array_ref resolves to shared array

---

## Test Boundaries & Scope

When multiple test files test similar functionality at different levels, clarify scope:

### Example: Data Loading Tests

1. **`unit/data/data-loading-integration.test.ts`**
   - **Scope**: Unit tests for loader pipeline with mocks
   - **What**: Tests internal logic with mocked zarr/THREE.js
   - **Fast**: No browser, no real files (~100ms per test)

2. **`unit/data/data-monitor-integration.test.ts`**
   - **Scope**: Unit tests for monitor + loader interaction with mocks
   - **What**: Tests event system and state synchronization
   - **Fast**: No browser, no real files (~50ms per test)

3. **`e2e/data-loading.spec.ts`**
   - **Scope**: E2E tests with real browser and real files
   - **What**: Tests full pipeline including WebGL rendering
   - **Slow**: Real browser, real datasets, real OPFS (~5s per test)

**Add comments** at the top of each test file clarifying its scope!

---

## Debugging Tests

### Unit Tests

```bash
# Run with debugger
pnpm test --inspect-brk

# Run specific test file
pnpm test array-decoder

# Watch mode for TDD
pnpm test --watch

# Verbose output
pnpm test --reporter=verbose
```

### E2E Tests

```bash
# Debug mode (visible browser, step through)
pnpm test:e2e:debug

# AI debugging (for Claude Code) - see browser console in terminal
pnpm agent:debug

# View test artifacts (screenshots, videos, traces)
pnpm test:e2e:report
```

### Common Issues

**"ReferenceError: X is not defined"**:

- Check if X needs a mock in `mocks/`
- Add mock to `setup.ts` if needed

**"TypeError: X is not a function"**:

- Check if mock provides the required method
- Update mock in `mocks/` directory

**"Test timeout"**:

- Check for infinite loops
- Check for unresolved promises
- Increase timeout in test config

---

## Continuous Integration

Tests run automatically on:

- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Main branch merges

**CI Requirements**:

- All unit tests pass (696/701 passing)
- All E2E tests pass (21 tests)
- Coverage ≥ 80%
- No TypeScript errors
- No linting errors

**Pre-commit Checklist**:

```bash
# 1. Run all tests
pnpm test --run && pnpm test:e2e

# 2. Check types
pnpm run typecheck

# 3. Check linting
pnpm run lint

# 4. Check coverage
pnpm run test:coverage
```

---

## Contributing

When adding new tests:

1. **Choose the right category**:
   - Place in appropriate `unit/` subfolder or `e2e/`
   - Unit tests for logic, E2E tests for browser integration

2. **Use test builders**:
   - Leverage `builders/test-data-builders.ts`
   - Add new builders if needed

3. **Use existing mocks**:
   - Check `mocks/` directory first
   - Only add new mocks if absolutely necessary

4. **Document critical bugs**:
   - If fixing a bug, add detailed comments
   - Explain: what broke, why, how the test prevents it

5. **Update this README**:
   - Add new test categories
   - Document new test patterns
   - Update test counts/statistics

6. **Run all tests**:
   - `pnpm test --run && pnpm test:e2e` before committing
   - Ensure no regressions

---

## Additional Resources

- [Main README](../../../README.md) - Project documentation
- [CLAUDE.md](../../../../CLAUDE.md) - Development guidance for AI assistants
- [PLAYWRIGHT_GUIDE.md](../../docs/PLAYWRIGHT_GUIDE.md) - Playwright testing guide
- [LUXAR_ZARR_FORMAT.md](../../../../docs/LUXAR_ZARR_FORMAT.md) - Data format specification
- [UI_DESIGN.md](../../../../docs/UI_DESIGN.md) - UI design system

---

## Questions?

- Check existing tests for patterns
- See [CLAUDE.md](../../../../CLAUDE.md) for testing philosophy
- Ask in project repository issues
