# Luxar Viewer Test Suite

This directory contains the comprehensive test suite for the Luxar Viewer TypeScript/WebGL application.

## Test Organization

Tests are organized by domain and type for easy navigation and maintenance:

```
tests/
├── unit/                           # Unit tests (fast, isolated)
│   ├── data/                       # Data loading & encoding
│   ├── ndim/                       # nD slicing & spatial queries
│   ├── cache/                      # Caching system
│   ├── controls/                   # Camera controls & input
│   ├── rendering/                  # Materials, shaders, post-processing
│   ├── scene/                      # Scene management
│   └── architecture/               # Global state, clean architecture
│
├── e2e/                            # End-to-end tests (Playwright)
│   ├── *.spec.ts                   # E2E test files
│   └── *-snapshots/                # Visual regression snapshots
│
├── builders/                       # Test data builders & helpers
└── setup.ts, test-config.ts        # Test configuration
```

---

## Test Categories

### 1. **Data Loading & Encoding** (`unit/data/`)

Tests for Python-TypeScript data compatibility and the complete loading pipeline.

**Scope**: Zarr loading, scene graph parsing, encoding/decoding (broadcasting, LUT, quantization, array_ref), spatial index loading

**Key Files**:

- `array-decoder.test.ts` - Encoding compatibility (Python ↔ TypeScript)
- `encoded-range-extraction.test.ts` - Range extraction bug fix verification
- `zarr-loader.test.ts` - Scene graph loading from zarr
- `point-spatial-index-loader.test.ts` - Spatial index-based loading
- `data-loading-integration.test.ts` - **Unit integration**: Tests loader pipeline with mocks
- `data-loading-monitor.test.ts` - Loading UI monitor unit tests
- `data-monitor-integration.test.ts` - **Unit integration**: Monitor + loader interaction with mocks

**Test Boundaries**:

- Unit tests use mocked zarr data and focus on logic
- E2E tests (`e2e/data-loading.spec.ts`) test with real browser + real files

---

### 2. **nD Slicing & Spatial Queries** (`unit/ndim/`)

Tests for multi-dimensional (4D, 5D+) visualization and slicing.

**Scope**: nD → 3D projection, hypersphere slicing, dimension navigation utilities

**Key Files**:

- `ndim-calculation-projectTo3D.test.ts` - **Critical bug fix**: ndim calculation from positions array
- `effective-radius-calculator.test.ts` - Hypersphere slicing for nD points
- `nd-navigation-utils.test.ts` - Dimension navigation helpers (step size, formatting, etc.)

**Why This Matters**: 4D+ datasets require special handling. These tests prevent regressions of critical bugs.

---

### 3. **Caching System** (`unit/cache/`)

Tests for memory management and performance optimization.

**Scope**: In-memory caching (LRU, segmented LRU), persistent storage (OPFS), two-level caching

**Key Files**:

- `range-cache.test.ts` - Range-based cache for point data
- `lru-cache.test.ts` - LRU eviction strategy
- `segmented-lru-cache.test.ts` - Segmented LRU for better hit rates
- `opfs-store.test.ts` - OPFS (Origin Private File System) persistent cache
- `two-level-caching-store.test.ts` - Memory + OPFS integration

**Cache Strategy**:

- `lru-cache`: Simple LRU for general use
- `segmented-lru-cache`: Segmented LRU for workloads with temporal/spatial locality
- `two-level-caching-store`: Hot data in memory, warm data in OPFS

---

### 4. **Controls & Input** (`unit/controls/`)

Tests for camera controls and input management.

**Scope**: Orbit/fly controls, keyboard/mouse input, input context management, conflict prevention

**Key Files**:

- `controls-manager.test.ts` - Control switching (orbit ↔ fly)
- `luxar-fly-controls.test.ts` - Fly controls physics, inertial mode, input handling
- `input-context-manager.test.ts` - Input context system (prevents WASD conflicts with text fields)
- `input-validation.test.ts` - Input validation utilities (block shortcuts when typing, FOV calculation)

---

### 5. **Rendering & Materials** (`unit/rendering/`)

Tests for WebGL rendering, shaders, and post-processing.

**Scope**: Material management, point shaders, HDR rendering, post-processing

**Key Files**:

- `material-manager.test.ts` - Material creation and lifecycle
- `point-material.test.ts` - Point shader material
- `postprocessing-manager.test.ts` - HDR post-processing pipeline
- `postprocessing-depth-mapping.test.ts` - Depth mapping for compositing
- `rendering-controls-utils.test.ts` - Rendering control helpers

---

### 6. **Scene Management** (`unit/scene/`)

Tests for scene graph and state management.

**Scope**: Scene orchestration, hierarchical transforms, dimension management

**Key Files**:

- `scene-manager.test.ts` - Scene orchestration and lifecycle
- `scene-manager-utils.test.ts` - Scene utility functions
- `scene-dims-manager.test.ts` - Dimension specification management

---

### 7. **Architecture** (`unit/architecture/`)

Tests for clean architecture and global state management.

**Scope**: Singleton managers, no global pollution, clean separation of concerns

**Key Files**:

- `global-state.test.ts` - Verifies no global variable pollution
- `directory-navigator.test.ts` - File navigation UI

---

### 8. **End-to-End Tests** (`e2e/`)

Full browser tests using Playwright that exercise the complete pipeline.

**Scope**: Real browser, real WebGL, real OPFS, real datasets

**Key Tests**:

- `basic-rendering.spec.ts` - Smoke tests for rendering
- `all-examples-smoke-test.spec.ts` - Validates all example datasets
- `python-typescript-integration.spec.ts` - Full Python → TypeScript pipeline
- `visual-regression.spec.ts` - Snapshot testing for visual correctness
- `spatial-index-accuracy.spec.ts` - Spatial index correctness
- `nd-navigation.spec.ts` - nD slicing in browser
- `performance-benchmarks.spec.ts` - Performance tracking
- `error-recovery.spec.ts` - Error handling and recovery
- `scene-integration.spec.ts` - Scene loading integration

**Why E2E Tests Matter**:

- Unit tests mock WebGL/browser APIs
- E2E tests catch browser-specific bugs (WebGL errors, OPFS issues, rendering glitches)
- Visual regression tests prevent UI breakage

---

## Running Tests

### All Tests

```bash
cd packages/luxar-viewer
pnpm test              # Unit tests (watch mode)
pnpm test --run        # Unit tests (single run)
pnpm test:e2e          # E2E tests (Playwright)
```

### Specific Test Categories

```bash
# Unit tests by domain
pnpm test unit/data/          # Data loading tests
pnpm test unit/ndim/          # nD slicing tests
pnpm test unit/cache/         # Cache tests
pnpm test unit/controls/      # Controls tests
pnpm test unit/rendering/     # Rendering tests
pnpm test unit/scene/         # Scene tests

# E2E tests
pnpm test:e2e                 # All E2E tests
pnpm test:e2e:ui              # Interactive UI
pnpm test:e2e:debug           # Debug mode
pnpm test:e2e:report          # View HTML report
```

### Coverage

```bash
pnpm run test:coverage        # Unit test coverage
pnpm test:e2e                 # E2E tests (coverage in report)
open coverage/typescript/index.html  # View coverage report
```

---

## Writing Tests

### When to Write Unit Tests vs E2E Tests

**Unit Tests** (`unit/`):

- Fast, isolated tests
- Mock external dependencies (zarr, THREE.js, browser APIs)
- Focus on logic and algorithms
- Example: Testing encoding/decoding logic

**E2E Tests** (`e2e/`):

- Full browser integration
- Real WebGL, OPFS, network requests
- Focus on user workflows and visual correctness
- Example: Loading a dataset and verifying points render correctly

### Test File Naming

- **Unit tests**: `*.test.ts`
- **E2E tests**: `*.spec.ts`
- **Descriptive names**: `array-decoder.test.ts` (not `test1.test.ts`)

### Test Structure

```typescript
/**
 * Clear description of what this test file covers
 *
 * Include context about:
 * - What's being tested
 * - Why it matters
 * - Any critical bugs this prevents
 */

import { describe, it, expect } from 'vitest';

describe('Module/Feature Name', () => {
  describe('specific function or behavior', () => {
    it('should do something specific', () => {
      // Arrange
      const input = ...;

      // Act
      const result = functionUnderTest(input);

      // Assert
      expect(result).toBe(expected);
    });
  });
});
```

### Test Boundaries & Scope Clarification

When multiple test files test similar functionality at different levels:

**Example**: Data Loading Tests

1. `unit/data/data-loading-integration.test.ts`
   - **Scope**: Unit tests for loader pipeline with mocks
   - **What**: Tests internal logic of loaders with mocked zarr/THREE.js
   - **Fast**: No browser, no real files

2. `unit/data/data-monitor-integration.test.ts`
   - **Scope**: Unit tests for monitor + loader interaction with mocks
   - **What**: Tests that monitor correctly receives loader events
   - **Fast**: No browser, no real files

3. `e2e/data-loading.spec.ts`
   - **Scope**: E2E tests with real browser and real files
   - **What**: Tests full pipeline including WebGL rendering
   - **Slow**: Real browser, real datasets, real OPFS

**Add comments** at the top of each test file clarifying its scope!

---

## Test Helpers

### Test Builders (`builders/`)

Use test builders for creating test data:

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

- Readable, expressive test setup
- Centralized test data generation
- Type-safe builders

### Test Fixtures

Python-generated test datasets in `tests/fixtures/`:

- Generated by `generate_test_data.py`
- Used for Python-TypeScript integration tests
- Include various encoding modes (broadcasting, LUT, quantization, array_ref)

---

## Critical Bug Documentation

Several tests document and prevent critical bugs:

1. **`ndim-calculation-projectTo3D.test.ts`**
   - Bug: Wrong ndim calculation (default 3) for 4D+ data without spatial index
   - Result: Block artifacts in rendering
   - Tests: Verify ndim calculated from actual positions array

2. **`encoded-range-extraction.test.ts`**
   - Bug: LUT-encoded arrays used wrong elementsPerPoint (1 instead of 3)
   - Result: Only 1/3 of points rendered, 2/3 appeared black
   - Tests: Verify range extraction uses correct elementsPerPoint

3. **`array-decoder.test.ts`**
   - Bug: Array references not resolved correctly
   - Result: Duplicated point data not shared
   - Tests: Verify array_ref resolves to target array

---

## Continuous Integration

Tests run automatically on:

- Every commit (pre-commit hook)
- Pull requests
- Main branch merges

**Coverage Requirements**:

- Minimum 80% coverage
- Critical paths must be 100% covered

---

## Debugging Tests

### Unit Tests

```bash
# Run with debugger
pnpm test --inspect-brk

# Run specific test file
pnpm test ndim-calculation-projectTo3D

# Watch mode for TDD
pnpm test --watch
```

### E2E Tests

```bash
# Debug mode (visible browser)
pnpm test:e2e:debug

# AI debugging (for Claude Code)
pnpm agent:debug

# View test artifacts
pnpm test:e2e:report  # Opens HTML report with screenshots/videos
```

---

## Contributing

When adding new tests:

1. **Choose the right category**: Place tests in the appropriate `unit/` subfolder or `e2e/`
2. **Use test builders**: Leverage `builders/test-data-builders.ts` for test data
3. **Document critical bugs**: If fixing a bug, add comments explaining the issue
4. **Update this README**: If adding a new test category or important test
5. **Run all tests**: `pnpm test --run && pnpm test:e2e` before committing

---

## Questions?

- Check the [main README](../../../README.md) for project documentation
- See [CLAUDE.md](../../../../CLAUDE.md) for development guidance
- Ask in the project repository issues
