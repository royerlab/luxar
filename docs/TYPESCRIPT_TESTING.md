# TypeScript Testing Strategy and Best Practices

## Current State Analysis

### Test Coverage Summary
- **Overall Coverage**: 10.65% (Critical - needs immediate attention)
- **Well-tested modules** (>70%):
  - `src/config`: 81.61%
  - `src/controls`: 79.78%
  - `src/input/input-context-manager.ts`: 94.97%
- **Untested modules** (0%):
  - `src/core`: Application bootstrap
  - `src/scene`: Scene management
  - `src/ui`: All UI components
  - `src/types`: Type definitions

### Existing Test Quality Assessment

#### Strengths
1. **Good test structure**: Tests use describe/it blocks with clear descriptions
2. **Setup/teardown**: Proper use of beforeEach/afterEach hooks
3. **Event testing**: Good coverage of event dispatching and handling
4. **State preservation**: Tests verify state transitions correctly

#### Weaknesses
1. **Heavy mocking**: Over-reliance on mocking (e.g., zarrita, memory-detector)
2. **Limited integration tests**: Most tests are unit tests with mocked dependencies
3. **Missing edge cases**: Few tests for error conditions and boundary cases
4. **No visual regression tests**: No tests for rendering output
5. **Poor test isolation**: Some tests depend on DOM manipulation

## Refactoring Recommendations for Better Testability

### 1. Extract Pure Functions from Classes

**Current Problem**: Classes with complex state are hard to test without mocking.

**Solution**: Extract business logic into pure functions.

```typescript
// Before: Hard to test class method
class SceneManager {
  calculateBoundingBox(): THREE.Box3 {
    // Complex logic intertwined with Three.js objects
  }
}

// After: Testable pure function
export function calculateBoundingBox(positions: Float32Array): BoundingBox {
  // Pure calculation logic
}

class SceneManager {
  calculateBoundingBox(): THREE.Box3 {
    const box = calculateBoundingBox(this.positions);
    return new THREE.Box3(box.min, box.max);
  }
}
```

### 2. Dependency Injection Pattern

**Current Problem**: Hard-coded dependencies require extensive mocking.

**Solution**: Inject dependencies for easier testing.

```typescript
// Before: Hard-coded dependency
class LazyDataManager {
  constructor() {
    this.memoryMonitor = new MemoryMonitor(); // Hard to test
  }
}

// After: Injected dependency
class LazyDataManager {
  constructor(memoryMonitor: IMemoryMonitor = new MemoryMonitor()) {
    this.memoryMonitor = memoryMonitor;
  }
}
```

### 3. Separate Business Logic from UI/Rendering

**Current Problem**: UI components mix logic with DOM manipulation.

**Solution**: Create logic modules that UI components consume.

```typescript
// New: Pure logic module
export class DimensionNavigationLogic {
  calculateNextStep(current: number, direction: 1 | -1, range: Range): number {
    // Pure calculation
  }
}

// UI component uses logic module
class DimensionSliders {
  private logic = new DimensionNavigationLogic();
  
  handleSliderChange(value: number) {
    const nextStep = this.logic.calculateNextStep(/*...*/);
    this.updateUI(nextStep);
  }
}
```

### 4. Strategy Pattern for Algorithms

**Current Problem**: Complex algorithms embedded in classes.

**Solution**: Extract algorithms as strategies.

```typescript
// Define strategy interface
interface SlicingStrategy {
  slice(positions: Float32Array, dims: SimpleDims): Uint32Array;
}

// Implement strategies
export class RadiusBasedSlicing implements SlicingStrategy { /*...*/ }
export class ExactMatchSlicing implements SlicingStrategy { /*...*/ }

// Use strategy
class DataNavigator {
  constructor(private strategy: SlicingStrategy) {}
  
  navigate() {
    return this.strategy.slice(this.positions, this.dims);
  }
}
```

## Testing Best Practices

### 1. Test Pyramid Approach
```
         /\
        /  \  E2E Tests (5%)
       /    \
      /------\  Integration Tests (25%)
     /        \
    /----------\  Unit Tests (70%)
```

### 2. Test File Organization
```typescript
// Group related tests logically
describe('ModuleName', () => {
  describe('initialization', () => {/*...*/});
  describe('core functionality', () => {/*...*/});
  describe('edge cases', () => {/*...*/});
  describe('error handling', () => {/*...*/});
});
```

### 3. Avoid Mocking When Possible
```typescript
// Prefer real implementations for simple dependencies
const realConfig = { ...defaultConfig, testMode: true };

// Only mock external dependencies and I/O
vi.mock('zarrita'); // External library
vi.mock('../api/fetch'); // Network calls
```

### 4. Test Data Builders
```typescript
// Create test data builders for complex objects
class PointCloudBuilder {
  private positions: number[] = [];
  private colors: number[] = [];
  
  withPoint(x: number, y: number, z: number): this {
    this.positions.push(x, y, z);
    return this;
  }
  
  build(): PointCloud {
    return {
      positions: new Float32Array(this.positions),
      colors: new Float32Array(this.colors),
    };
  }
}

// Usage in tests
const pointCloud = new PointCloudBuilder()
  .withPoint(0, 0, 0)
  .withPoint(1, 1, 1)
  .build();
```

### 5. Parameterized Tests
```typescript
// Test multiple scenarios with same logic
describe.each([
  { input: [0, 0, 0], expected: 0 },
  { input: [1, 1, 1], expected: Math.sqrt(3) },
  { input: [-1, 0, 1], expected: Math.sqrt(2) },
])('calculateDistance($input)', ({ input, expected }) => {
  it(`returns ${expected}`, () => {
    expect(calculateDistance(input)).toBeCloseTo(expected);
  });
});
```

## Priority Areas for New Tests

### Critical Path Coverage (Priority 1)
1. **Scene Loading Pipeline**
   - `zarr-loader.ts`: Scene initialization from Zarr
   - `scene-manager.ts`: Scene setup and rendering
   - `dimensions-manager.ts`: nD navigation logic

2. **Data Slicing Algorithms**
   - `slicing.ts`: Core nD slicing logic
   - `dims-navigator.ts`: Dimension navigation

3. **Core Application Flow**
   - `app.ts`: Application initialization
   - Error handling paths

### User Interaction Coverage (Priority 2)
1. **Controls System**
   - Keyboard shortcuts
   - Mouse interactions
   - Control mode switching

2. **UI Components**
   - Dimension sliders
   - Loading indicators
   - Error displays

### Performance Critical Coverage (Priority 3)
1. **Memory Management**
   - Cache eviction strategies
   - Chunk loading optimization
   
2. **Rendering Pipeline**
   - Shader compilation
   - Material updates
   - Post-processing effects

## Implementation Plan

### Phase 1: Refactor for Testability (Week 1)
- [ ] Extract pure functions from `slicing.ts`
- [ ] Add dependency injection to `LazyDataManager`
- [ ] Separate logic from UI in dimension navigation
- [ ] Create test data builders

### Phase 2: Critical Path Tests (Week 2)
- [ ] Add comprehensive tests for slicing algorithms
- [ ] Test scene loading pipeline
- [ ] Test dimension navigation logic
- [ ] Add error handling tests

### Phase 3: Integration Tests (Week 3)
- [ ] Add integration tests for data loading
- [ ] Test control system interactions
- [ ] Test UI component workflows
- [ ] Add performance benchmarks

### Phase 4: Coverage Improvement (Week 4)
- [ ] Achieve 50% overall coverage
- [ ] 80% coverage for critical modules
- [ ] Add regression tests for fixed bugs
- [ ] Document testing patterns

## Testing Commands

```bash
# Run all tests
pnpm test

# Run tests with coverage
pnpm test:coverage

# Run tests in watch mode
pnpm test:watch

# Run specific test file
pnpm test src/tests/slicing.test.ts

# Run tests with UI
pnpm test:ui
```

## Metrics and Goals

### Current Metrics
- Coverage: 10.65%
- Test execution time: ~1s
- Test count: 119

### Target Metrics (3 months)
- Coverage: 80%
- Test execution time: <10s
- Test count: 500+
- Critical path coverage: 95%

## Anti-Patterns to Avoid

1. **Testing Implementation Details**: Test behavior, not implementation
2. **Brittle Tests**: Avoid testing exact HTML structure or CSS classes
3. **Test Interdependence**: Each test should be independent
4. **Excessive Mocking**: Prefer real implementations when feasible
5. **Ignored Failures**: Fix flaky tests immediately
6. **Missing Assertions**: Every test must have clear assertions

## Resources and Tools

### Testing Libraries
- **Vitest**: Modern test runner with great DX
- **@testing-library**: For UI component testing
- **MSW**: Mock Service Worker for API mocking
- **Playwright**: E2E testing (future consideration)

### Coverage Tools
- **v8**: Native coverage provider
- **codecov**: Coverage tracking service

### Documentation
- [Vitest Documentation](https://vitest.dev/)
- [Testing Best Practices](https://github.com/goldbergyoni/javascript-testing-best-practices)
- [Three.js Testing Guide](https://threejs.org/docs/#manual/en/introduction/Testing)

## Conclusion

The current test coverage is critically low and needs immediate attention. By following this strategy:
1. Refactor code for better testability
2. Focus on critical path coverage first
3. Gradually increase coverage to 80%
4. Maintain test quality over quantity

This will result in a more maintainable, reliable, and confident codebase.