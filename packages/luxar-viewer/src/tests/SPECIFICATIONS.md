# luxar-viewer.tests - Test Suite Specification

**Version**: 1.0.0
**Last Updated**: 2025-12-07

## Purpose

The test suite ensures correctness, reliability, and maintainability of the Luxar Viewer across:

- Python-TypeScript data compatibility (encoding/decoding)
- WebGL rendering and visual correctness
- nD visualization and slicing algorithms
- Performance and memory efficiency
- Cross-browser compatibility

---

## Test Architecture

### Design Principles

1. **Pyramid Structure**: Many fast unit tests, fewer integration tests, minimal E2E tests
2. **Test Isolation**: Each test runs independently with clean state
3. **Mock Everything External**: WebGL, DOM, OPFS, Network - all mocked in unit tests
4. **Real Data in E2E**: E2E tests use actual zarr files and real browser environment
5. **Python-TypeScript Parity**: Cross-language compatibility is critical and heavily tested

### Test Categories

```
tests/
├── unit/              # Fast, isolated, mocked (696 tests, ~3s runtime)
│   ├── data/          # Data loading, encoding, zarr compatibility
│   ├── ndim/          # nD slicing, spatial queries, projections
│   ├── cache/         # Memory management, LRU/OPFS caching
│   ├── controls/      # Camera controls, input handling
│   ├── rendering/     # Materials, shaders, post-processing
│   ├── scene/         # Scene management, transforms
│   └── architecture/  # Clean architecture, state management
│
├── e2e/               # Slow, integrated, real browser (Playwright)
│   ├── basic-rendering.spec.ts      # Core rendering pipeline
│   ├── all-examples-smoke-test.spec.ts  # All example datasets render
│   ├── visual-regression.spec.ts     # Screenshot comparisons
│   └── performance-benchmarks.spec.ts   # Performance measurement
│
├── mocks/             # Centralized mock infrastructure
│   ├── three.mock.ts           # THREE.js complete mock (1525 lines)
│   ├── webgl.mock.ts           # WebGL context mock (165 lines)
│   ├── browser-apis.mock.ts    # Browser APIs (matchMedia, observers, etc.)
│   ├── opfs.mock.ts            # Origin Private File System mock
│   └── orbit-controls.mock.ts  # THREE OrbitControls mock
│
├── builders/          # Test data builders and helpers
└── fixtures/          # Python-generated zarr test datasets
    ├── generate_test_data.py  # Fixture generation script
    └── *.zarr         # Test datasets (broadcasting, LUT, quantization, etc.)
```

---

## Test Philosophy

### Unit Tests: The Foundation

**What We Test**:

- Pure logic: algorithms, calculations, data transformations
- Component behavior: how individual classes/functions work
- Error handling: edge cases, invalid inputs, boundary conditions
- Integration contracts: how components interact through interfaces

**What We Don't Test**:

- Visual appearance (E2E responsibility)
- Actual WebGL rendering (mocked)
- Real network requests (mocked)
- Browser-specific behavior (E2E responsibility)

**Mock Strategy**:

- Mock external dependencies (WebGL, DOM, network)
- Mock I/O (file system, OPFS)
- Real business logic (no mocking of our own code)
- Real math (Vector/Matrix operations actually work in mocks)

### E2E Tests: The Reality Check

**What We Test**:

- Complete user workflows end-to-end
- Visual correctness (screenshot comparisons)
- Performance on real datasets
- Cross-browser compatibility
- Real WebGL rendering in real browser
- Complete data pipeline (Python zarr → TypeScript → GPU → Canvas)

**Why E2E Matters**:
Many bugs only manifest in real browsers:

- WebGL shader compilation issues
- Canvas rendering artifacts
- Browser-specific APIs (OPFS, WebGPU, HDR)
- Performance bottlenecks
- Memory leaks

---

## Critical Test Domains

### 1. Python-TypeScript Encoding Compatibility

**Why Critical**: Python encodes data, TypeScript decodes it. Mismatches cause silent bugs.

**Test Files**:

- `unit/data/array-decoder.test.ts` (39 tests) - All encoding modes
- `unit/data/encoded-range-extraction.test.ts` (23 tests) - Range loading bugs

**Encoding Modes Tested**:

- Broadcasting (uniform values)
- LUT (≤256 unique values)
- Quantization (uint8/uint16 compression)
- Array References (deduplication)
- Scalar LUT (per-channel LUT)
- Log-space encoding (wide dynamic range)
- Direct/no encoding

**How Fixtures Are Generated**:

```bash
# Automatically before tests:
pnpm test:with-fixtures

# Or manually:
pnpm test:generate-fixtures
```

### 2. nD Visualization & Slicing

**Why Critical**: 4D+ datasets require special math. Bugs cause points to disappear or appear incorrectly.

**Test Files**:

- `unit/ndim/ndim-calculation-projectTo3D.test.ts` - Critical ndim calculation bug fix
- `unit/ndim/effective-radius-calculator.test.ts` - Hypersphere slicing
- `unit/ndim/nd-navigation-utils.test.ts` - Dimension utilities

**Key Algorithms**:

- nD → 3D projection (displayDims selection)
- Hypersphere radius-based slicing
- nD tolerance calculations
- Dimension step navigation

### 3. Spatial Index & Performance

**Why Critical**: Without spatial indexing, million-point datasets are unusably slow.

**Test Files**:

- `unit/data/point-spatial-index-loader.test.ts` - Chunk-based spatial queries
- `e2e/spatial-index-accuracy.spec.ts` - Correctness verification
- `e2e/performance-benchmarks.spec.ts` - Performance measurement

**What We Verify**:

- Correct points returned for view frustum
- Range merging optimization
- Memory efficiency
- Query time < 10ms for 1M points

### 4. Memory Management & Caching

**Why Critical**: Viewer must handle GB-scale datasets in limited browser memory.

**Test Files**:

- `unit/cache/lru-cache.test.ts` - LRU eviction strategy
- `unit/cache/segmented-lru-cache.test.ts` - Segmented cache (large values)
- `unit/cache/two-level-caching-store.test.ts` - OPFS persistent cache

**Cache Layers** (after L0 removal):

- L1: In-memory compressed chunks (100MB)
- L2: OPFS persistent storage (2GB)
- L3: Network (HTTP/fetch)

### 5. WebGL Rendering & Shaders

**Why Critical**: GPU bugs cause visual artifacts, crashes, or black screens.

**Test Files**:

- `unit/rendering/point-material.test.ts` - Shader material setup
- `unit/rendering/material-manager.test.ts` - Material caching
- `unit/rendering/postprocessing-manager.test.ts` - HDR pipeline
- `e2e/basic-rendering.spec.ts` - Visual verification

**What We Test**:

- Shader compilation (mocked in unit, real in E2E)
- Material property updates
- HDR color pipeline (float32 colors)
- Post-processing effects (bloom, tone mapping)

---

## Mock Infrastructure

### Centralized Mock Organization

All mocks are in `src/tests/mocks/` for easy maintenance and reuse.

**File Structure**:

```
mocks/
├── index.ts              # Central export, installAllMocks()
├── three.mock.ts         # Complete THREE.js mock (all classes)
├── webgl.mock.ts         # MockWebGLRenderingContext
├── browser-apis.mock.ts  # window.matchMedia, observers, etc.
├── opfs.mock.ts          # navigator.storage, OPFS API
└── orbit-controls.mock.ts # OrbitControls from three/examples
```

**Usage**:

```typescript
// In setup.ts (automatic for all tests):
import { installAllMocks } from './mocks';
installAllMocks();

// Or selective in individual tests:
import { installWebGLMock } from './mocks/webgl.mock';
installWebGLMock();
```

### Mock Design Philosophy

**1. Functional Where Possible**:

```typescript
// Vector math actually works:
const v = new Vector3(1, 2, 3);
v.add(new Vector3(1, 1, 1)); // → (2, 3, 4)
v.length(); // → sqrt(4 + 9 + 16) = 5.39
```

**2. Mocked Side Effects**:

```typescript
// I/O and rendering are vi.fn() mocks:
renderer.render(scene, camera); // vi.fn() - no actual rendering
gl.createTexture(); // vi.fn() - no GPU allocation
```

**3. Realistic Defaults**:

```typescript
// Mocks return sensible values:
gl.getParameter(gl.VERSION); // → "WebGL 2.0 (OpenGL ES 3.0)"
gl.getExtension('EXT_color_buffer_float'); // → {} (truthy = supported)
```

**4. Type Safety**:
All mocks use proper TypeScript types where possible, with `any` escape hatches for complex THREE.js types.

---

## Test Data & Fixtures

### Python-Generated Fixtures

**Script**: `tests/fixtures/generate_test_data.py` (635 lines)

**Datasets Generated** (11 total):

1. `test_broadcasting.zarr` - Uniform values (1 color → 1000 points)
2. `test_lut.zarr` - 10 unique colors (LUT encoding)
3. `test_quantization.zarr` - uint8 quantized colors/radii
4. `test_array_refs.zarr` - Shared arrays (deduplication)
5. `test_mixed.zarr` - Multiple encoding modes in one scene
6. `test_4d.zarr` - 4D positions with time dimension
7. `test_4d_scalar_lut.zarr` - 4D with scalar LUT (quantum orbitals bug fix)
8. `test_hierarchical_transforms.zarr` - Nested transforms
9. `test_hdr_colors.zarr` - HDR colors (values > 1.0)
10. `test_sharpness_range.zarr` - Full sharpness range [1, 31]
11. `test_log_scalar.zarr` - Log-space encoded radii

**Auto-Generation**:
Fixtures are automatically regenerated before tests to match current Python encoding:

```bash
pnpm test:with-fixtures  # Auto-gen + test
make test-all            # Python tests + gen fixtures + TypeScript tests
```

### Test Data Builders

**File**: `tests/builders/test-data-builders.ts`

Helper functions for creating mock zarr data in tests without Python:

- `createMockZarrLocation()` - In-memory zarr store
- `createMockSceneNode()` - Scene graph nodes
- `createMockPointsData()` - Point cloud data

---

## Running Tests

### Unit Tests (Fast, ~3s)

```bash
# All unit tests
pnpm test --run

# Specific test file
pnpm test array-decoder

# Watch mode
pnpm test:watch

# With coverage
pnpm test:coverage

# Generate fixtures first (needed after Python changes)
pnpm test:with-fixtures
```

### E2E Tests (Slow, ~60s)

```bash
# All E2E tests
pnpm test:e2e

# Interactive UI
pnpm test:e2e:ui

# Debug mode (headed browser)
pnpm test:e2e:debug

# View HTML report
pnpm test:e2e:report
```

### All Tests (Comprehensive)

```bash
# From project root:
make test-all  # Python + TypeScript (with fixture gen)

# From packages/luxar-viewer:
pnpm test:with-fixtures && pnpm test:e2e
```

---

## Test Coverage Goals

**Target Coverage**: ≥80% for all metrics

**Current Coverage**:

- Python: ~95% (via hatch run test-cov)
- TypeScript: ~85% (via pnpm test:coverage)

**Priority Areas for Coverage**:

1. Data loading & encoding (critical path)
2. nD slicing algorithms (complex math)
3. Spatial index queries (performance critical)
4. Error handling (user-facing)

**Acceptable Lower Coverage**:

- UI components (hard to test, covered by E2E)
- Debugging utilities (dev-only code)
- WebGL initialization (browser-specific)

---

## Common Test Patterns

### 1. Testing Encoding Compatibility

```typescript
// Python generates fixture with specific encoding:
// packages/luxar-viewer/tests/fixtures/generate_test_data.py

// TypeScript tests it loads correctly:
const { rootLoc } = await loadTestDataset('test_broadcasting.zarr');
const array = await zarr.open(rootLoc.resolve('points/colors'));
const data = await get(array);

expect(data.data).toEqual(expectedColors);
```

### 2. Testing nD Slicing

```typescript
// Set up 4D data with specific slice position:
const viewState: ViewState = {
  displayDims: [1, 2, 3], // Show x, y, z
  slicePosition: [5, 0, 0, 0], // At time=5
  tolerance: [0.5, 0, 0, 0], // Radius in time dimension
};

// Load points and verify only time=5 points are returned:
const result = await loader.loadPoints(viewState);
expect(result.metadata.loadedPoints).toBeLessThan(totalPoints);
```

### 3. Testing WebGL (Mocked)

```typescript
// WebGL is mocked - we test the setup, not actual rendering:
const renderer = new WebGLRenderer();
expect(renderer.getContext()).toBeDefined();
expect(renderer.render).toHaveBeenCalled();

// Shader compilation is mocked to succeed:
const material = new ShaderMaterial({ vertexShader, fragmentShader });
expect(material.needsUpdate).toBe(true);
```

### 4. Testing E2E (Real Browser)

```typescript
// Real browser, real WebGL, real zarr data:
test('should render points correctly', async ({ page }) => {
  await page.goto('http://localhost:5173/?src=http://localhost:9000/test.zarr');

  // Wait for rendering
  await page.waitForSelector('canvas');

  // Take screenshot
  await expect(page).toHaveScreenshot('points-rendered.png');

  // Query scene state via debug interface
  const pointCount = await page.evaluate(() => window.__luxarDebug.getState().totalPoints);
  expect(pointCount).toBeGreaterThan(0);
});
```

---

## Known Limitations & Trade-offs

### 1. WebGL Mocking

**Limitation**: Can't test actual GPU rendering in unit tests
**Mitigation**: E2E tests with real WebGL verify visual correctness
**Trade-off**: Fast unit tests (3s) vs slow E2E tests (60s)

### 2. OPFS Mocking

**Limitation**: OPFS not available in Node.js test environment
**Mitigation**: Mock rejects gracefully, tests verify error handling
**Trade-off**: Can't test actual persistence in unit tests

### 3. Cross-Browser Testing

**Limitation**: Unit tests run in jsdom (not real browsers)
**Mitigation**: E2E tests run in Chromium/Firefox/WebKit via Playwright
**Trade-off**: E2E suite takes ~5min for all browsers

### 4. Test Fixture Maintenance

**Limitation**: Fixtures must be regenerated when Python encoding changes
**Mitigation**: Automatic generation before tests (`pnpm test:with-fixtures`)
**Trade-off**: Fixtures are checked in (100MB), not generated on-demand

---

## Test Quality Metrics

### Current Status

**Unit Tests**:

- Total: 701 tests
- Passing: 696 (99.3%)
- Skipped: 5 (intentional - require specific setup)
- Failing: 0 ✅
- Runtime: ~3 seconds

**E2E Tests**:

- Total: ~50 tests
- Passing: ~48 (96%)
- Flaky: 2 (network-dependent)
- Runtime: ~60 seconds

**Python Tests**:

- Total: 1372 tests
- Passing: 1369 (99.8%)
- Skipped: 3
- Runtime: ~4 minutes

### Test Suite Health Indicators

**Healthy**:

- ✅ >95% pass rate
- ✅ <5s unit test runtime
- ✅ <2min E2E suite runtime
- ✅ Zero flaky tests

**Needs Attention**:

- ⚠️ <90% pass rate
- ⚠️ >10s unit test runtime
- ⚠️ >5 flaky E2E tests
- ⚠️ Coverage <80%

---

## Adding New Tests

### For New Features

```typescript
// 1. Unit test the logic
describe('MyNewFeature', () => {
  it('should handle basic case', () => {
    const result = myNewFeature(input);
    expect(result).toEqual(expected);
  });

  it('should handle edge case', () => {
    expect(() => myNewFeature(invalidInput)).toThrow();
  });
});

// 2. E2E test the user experience
test('user can use new feature', async ({ page }) => {
  await page.goto('...');
  await page.click('#new-feature-button');
  await expect(page).toHaveScreenshot();
});
```

### For Bug Fixes

```typescript
// 1. Write a failing test that reproduces the bug
it('should not crash when X happens (bug #123)', () => {
  // This test currently fails:
  const result = buggyFunction(edgeCaseInput);
  expect(result).not.toBeNull(); // Fails: returns null!
});

// 2. Fix the bug
// ... fix code ...

// 3. Verify test now passes
// The test passing proves the bug is fixed and won't regress
```

---

## Debugging Failed Tests

### 1. Read the Error Message

```bash
pnpm test failing-test
# Look for:
# - "expected X but got Y" → logic error
# - "TypeError: X is not a function" → mock issue
# - "Test timed out" → async issue or infinite loop
```

### 2. Run Single Test in Watch Mode

```bash
pnpm test:watch failing-test
# Edit code, save, test re-runs automatically
```

### 3. Add Debug Logging

```typescript
it('debugging test', () => {
  console.log('Input:', input);
  const result = myFunction(input);
  console.log('Result:', result);
  expect(result).toBe(expected);
});
```

### 4. Check Mock Setup

```typescript
// Verify mock is actually being used:
const spy = vi.spyOn(myModule, 'myFunction');
myCode();
console.log('Spy calls:', spy.mock.calls);
```

---

## Continuous Integration

### Pre-commit Checks

```bash
# Developers should run before committing:
make check  # typecheck + lint + test

# Or individually:
pnpm run typecheck
pnpm run lint
pnpm test --run
```

### CI Pipeline (GitHub Actions / etc.)

```yaml
# Recommended CI workflow:
- name: Install Python dependencies
  run: pip install -e .

- name: Install TypeScript dependencies
  run: cd packages/luxar-viewer && pnpm install

- name: Run Python tests
  run: hatch run test

- name: Generate test fixtures
  run: pnpm test:generate-fixtures

- name: Run TypeScript unit tests
  run: cd packages/luxar-viewer && pnpm test --run

- name: Run TypeScript E2E tests
  run: cd packages/luxar-viewer && pnpm test:e2e

- name: Upload test artifacts
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: |
      packages/luxar-viewer/playwright-report/
      packages/luxar-viewer/test-results/
```

---

## Changelog

### v1.0.0 (2025-12-07)

- Initial specification
- Documented test architecture after L0 cache removal
- Added mock infrastructure documentation
- Documented fixture generation workflow
- Achieved 99.3% unit test pass rate (696/701)

---

## Related Specifications

- `luxar.encoding` - Encoding specification (see `../../encoding/SPECIFICATIONS.md`)
- `luxar.io` - I/O and zarr format (see `../../io/SPECIFICATIONS.md`)
- Luxar Zarr Format - Data format spec (see `/docs/LUXAR_ZARR_FORMAT.md`)
