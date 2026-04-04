# TypeScript Testing Guidelines - Luxar Viewer

## Python Testing Note

Some environments auto-load pytest plugins from installed packages (e.g., napari/numba),
which can fail during collection. If you hit a plugin auto-load error, run pytest with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 hatch run pytest path/to/test.py
```

## Problem: Over-Mocking Antipattern

### What is Over-Mocking?

Over-mocking occurs when tests mock internal application modules instead of just external dependencies. This defeats the purpose of testing because you're testing mock behavior, not real code.

### Current Issues in Codebase

Several test files suffer from excessive mocking:
- `scene-manager.test.ts` - Mocks 7 internal modules
- `app.test.ts` - Mocks 9 internal modules
- `data-loading-integration.test.ts` - Mocks internal data loaders

### Why This is Bad

```typescript
// ❌ BAD - Mocking internal modules
vi.mock('../data/scene-loader', () => ({
  SceneLoader: vi.fn().mockImplementation(() => ({
    loadScene: vi.fn().mockResolvedValue({ /* mock data */ })
  }))
}));

// Tests verify mock behavior, not real SceneLoader logic!
```

**Problems:**
1. **False confidence**: Tests pass but real code may be broken
2. **Brittle tests**: Break when implementation details change
3. **No integration testing**: Module boundaries never tested
4. **Refactoring resistance**: Can't safely refactor without breaking tests

---

## The Solution: Mock Only External Dependencies

### What to Mock

**✅ DO Mock:**
- External libraries (THREE.js WebGL context, DOM APIs)
- Network requests (fetch, WebSocket)
- File system access
- Browser APIs not available in test environment
- Time-dependent code (Date.now, setTimeout)

**❌ DON'T Mock:**
- Your own modules and classes
- Internal utilities
- Business logic
- Data transformations
- State management

---

## Refactoring Strategy

### Step 1: Identify What MUST Be Mocked

```typescript
// ✅ GOOD - Mock only WebGL context (external dependency)
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  class MockWebGLRenderer {
    // Minimal mock to avoid WebGL context creation
    domElement = document.createElement('canvas');
    setSize() {}
    render() {}
    dispose() {}
  }

  return {
    ...actual,  // Keep everything else real!
    WebGLRenderer: MockWebGLRenderer
  };
});
```

### Step 2: Remove Internal Mocks

```typescript
// ❌ BEFORE - Mocking internal modules
vi.mock('../data/scene-loader', () => ({ ... }));
vi.mock('../controls/controls-manager', () => ({ ... }));
vi.mock('../ui/helpers', () => ({ ... }));

// ✅ AFTER - Use real implementations
import { SceneLoader } from '../data/scene-loader';
import { ControlsManager } from '../controls/controls-manager';
import * as uiHelpers from '../ui/helpers';

// Mock only external calls these modules make
vi.spyOn(uiHelpers, 'showLoadingIndicator').mockImplementation(() => {});
```

### Step 3: Use Test Fixtures Instead of Mocks

```typescript
// ❌ BAD - Mock returns fake data
vi.mock('../data/scene-loader', () => ({
  loadScene: vi.fn().mockResolvedValue({ points: [] })
}));

// ✅ GOOD - Use real loader with fixture data
import { SceneLoader } from '../data/scene-loader';

const fixtureUrl = 'test-fixtures/basic_example.zarr';
const loader = new SceneLoader();
const scene = await loader.loadScene(fixtureUrl);
```

---

## Example Refactoring

### Before: Over-Mocked Test

```typescript
// ❌ BAD - Mocks everything
vi.mock('../rendering/point-material', () => ({
  PointMaterial: vi.fn().mockImplementation(() => ({
    uniforms: { fov: { value: 60 } },
    dispose: vi.fn()
  }))
}));

vi.mock('../data/scene-loader', () => ({
  SceneLoader: vi.fn().mockImplementation(() => ({
    loadPoints: vi.fn().mockResolvedValue({
      positions: new Float32Array([0, 0, 0]),
      colors: new Uint8Array([255, 0, 0])
    })
  }))
}));

describe('SceneManager', () => {
  it('should load points', async () => {
    const manager = new SceneManager();
    await manager.init();
    await manager.loadScene('test.zarr');

    // Test passes but proves nothing about real code!
    expect(mockLoadPoints).toHaveBeenCalled();
  });
});
```

### After: Minimal Mocking

```typescript
// ✅ GOOD - Mock only WebGL
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');
  class MockWebGLRenderer {
    domElement = document.createElement('canvas');
    setSize() {}
    render() {}
    dispose() {}
  }
  return { ...actual, WebGLRenderer: MockWebGLRenderer };
});

import { SceneManager } from '../scene/scene-manager';
import { PointMaterial } from '../rendering/point-material';
import { SceneLoader } from '../data/scene-loader';

describe('SceneManager', () => {
  it('should load points and create material', async () => {
    const manager = new SceneManager();
    await manager.init();

    // Use test fixture with real loader
    await manager.loadScene('test-fixtures/basic_example.zarr');

    // Verify real interactions
    expect(manager.scene.children.length).toBeGreaterThan(0);
    const points = manager.scene.children[0];
    expect(points.material).toBeInstanceOf(PointMaterial);
    expect(points.geometry.attributes.position.count).toBe(100);
  });
});
```

**Benefits:**
- Tests real code paths
- Catches integration bugs
- Survives refactoring
- Documents actual behavior

---

## Specific Patterns

### Pattern 1: Spy on Specific Methods

```typescript
// ✅ GOOD - Spy on specific methods that have side effects
import * as uiHelpers from '../ui/helpers';

const showLoadingSpy = vi.spyOn(uiHelpers, 'showLoadingIndicator')
  .mockImplementation(() => {});

// Real code runs, but we prevent DOM manipulation
```

### Pattern 2: Dependency Injection for Testability

```typescript
// ✅ GOOD - Make dependencies injectable
class SceneManager {
  constructor(
    private loader = new SceneLoader(),
    private controls = new ControlsManager()
  ) {}
}

// In tests, inject test doubles if needed
const testLoader = new SceneLoader({ endpoint: 'test-fixtures/' });
const manager = new SceneManager(testLoader);
```

### Pattern 3: Test Fixtures Over Mocks

```typescript
// ✅ GOOD - Generate test fixtures once, reuse everywhere
// test-fixtures/generate_fixtures.py generates:
// - basic_example.zarr (100 points)
// - nd_example.zarr (4D dataset)
// - hdr_example.zarr (HDR colors)

describe('Data Loading', () => {
  it('should load 3D points', async () => {
    const loader = new SceneLoader();
    const data = await loader.loadPoints('test-fixtures/basic_example.zarr');
    expect(data.positions.length).toBe(300); // 100 points * 3
  });
});
```

---

## Testing Principles

### 1. Test Behavior, Not Implementation
```typescript
// ❌ BAD
expect(material.uniforms.fov.value).toBe(60);

// ✅ GOOD
expect(material.calculateScreenSpaceSize(distance)).toBeCloseTo(expectedSize);
```

### 2. Test Public APIs
```typescript
// ❌ BAD - Testing private methods
expect((manager as any)._calculateBounds()).toBeDefined();

// ✅ GOOD - Testing public behavior
expect(manager.getBoundingBox()).toEqual(expectedBox);
```

### 3. Keep Tests Simple
```typescript
// ❌ BAD - Complex setup
beforeEach(async () => {
  mockA.mockReturnValue(...);
  mockB.mockResolvedValue(...);
  mockC.mockImplementation(...);
  await setupComplexScenario();
});

// ✅ GOOD - Simple, focused test
it('should handle empty dataset', () => {
  const loader = new SceneLoader();
  expect(() => loader.loadPoints('empty.zarr')).toThrow();
});
```

---

## Resources

- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
- [Mock Vs Stub Vs Spy](https://martinfowler.com/articles/mocksArentStubs.html)
- [Growing Object-Oriented Software, Guided by Tests](http://www.growing-object-oriented-software.com/)

---

## Questions?

If unsure whether to mock something, ask:
1. **Is this code I own?** → Don't mock it
2. **Does this make network/file/GPU calls?** → Mock the I/O, not the wrapper
3. **Would this test catch real bugs?** → If no, you're over-mocking

**Golden Rule**: If removing the mock breaks the test for reasons unrelated to the test's purpose, you're probably over-mocking.
