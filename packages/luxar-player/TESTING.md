# Testing Guide for Luxar Player

## Overview

The luxar-player uses Vitest for testing, which provides excellent TypeScript support, fast execution, and seamless integration with Vite.

## Test Structure

```
src/
├── tests/
│   ├── setup.ts              # Global test setup and WebGL mocks
│   ├── zarr_loader.test.ts   # Unit tests for Zarr loading
│   ├── shader-manager.test.ts # Unit tests for shader compilation
│   └── integration/          # Integration tests (future)
└── ... (source files)
```

## Running Tests

```bash
# Run all tests once
npm test

# Run tests in watch mode (re-runs on file changes)
npm run test:watch

# Run tests with UI (opens browser-based test runner)
npm run test:ui

# Run tests with coverage report
npm run test:coverage

# Run all checks (typecheck, lint, tests)
npm run check
```

## Writing Tests

### Unit Tests

Unit tests should focus on individual functions and classes in isolation:

```typescript
import { describe, it, expect, vi } from 'vitest';

describe('MyModule', () => {
  it('should do something', () => {
    // Arrange
    const input = 'test';
    
    // Act
    const result = myFunction(input);
    
    // Assert
    expect(result).toBe('expected');
  });
});
```

### Mocking WebGL

The test setup provides comprehensive WebGL mocks. When testing WebGL-dependent code:

```typescript
describe('WebGL Component', () => {
  let gl: WebGLRenderingContext;

  beforeEach(() => {
    const canvas = document.createElement('canvas');
    gl = canvas.getContext('webgl')!;
  });

  it('should create shader', () => {
    const shader = gl.createShader(gl.VERTEX_SHADER);
    expect(shader).toBeDefined();
    expect(gl.createShader).toHaveBeenCalledWith(gl.VERTEX_SHADER);
  });
});
```

### Testing Async Code

Use async/await for testing promises:

```typescript
it('should load data asynchronously', async () => {
  const data = await loadData('url');
  expect(data).toHaveProperty('metadata');
});
```

## Best Practices

1. **Test Organization**
   - One test file per source file
   - Group related tests with `describe` blocks
   - Use descriptive test names

2. **Mocking**
   - Mock external dependencies (zarrita, network requests)
   - Use `vi.mock()` for module mocking
   - Reset mocks in `beforeEach` with `vi.clearAllMocks()`

3. **WebGL Testing**
   - Test shader compilation logic, not actual GPU operations
   - Verify correct GL function calls
   - Test error handling for shader/program failures

4. **Coverage Goals**
   - Aim for 80%+ code coverage
   - Focus on critical paths (shader compilation, data loading)
   - Don't test Three.js internals

## Testing WebGL Shaders

Since actual GPU operations can't be tested in Node.js:

1. Test shader string generation
2. Test uniform/attribute setup
3. Mock GL context to verify correct API usage
4. Use snapshot testing for complex shader strings

Example:
```typescript
it('should generate correct vertex shader', () => {
  const shader = generateVertexShader({ hasRadii: true });
  expect(shader).toContain('attribute float radius');
  expect(shader).toMatchSnapshot();
});
```

## Integration Tests (Future)

For full rendering tests, consider:
- Puppeteer/Playwright for browser automation
- Visual regression testing with screenshots
- WebGL rendering to offscreen canvas

## Debugging Tests

- Use `npm run test:ui` for interactive debugging
- Add `console.log` or use VS Code debugger
- Use `it.only()` to run single test
- Use `describe.skip()` to skip test suites

## CI/CD Integration

Add to GitHub Actions:
```yaml
- name: Install dependencies
  run: cd packages/luxar-player && npm ci
  
- name: Run tests
  run: cd packages/luxar-player && npm run test:coverage
  
- name: Upload coverage
  uses: codecov/codecov-action@v3
```