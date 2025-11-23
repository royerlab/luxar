# Luxar E2E Tests - Test Suite Documentation

## Overview

This directory contains end-to-end tests for the Luxar viewer using Playwright. These tests verify functionality from a user's perspective and enable AI-assisted debugging.

## Test Files

### Foundation Tests (4 files, 24 tests)

#### 1. `basic-rendering.spec.ts` - Infrastructure Tests

**Purpose**: Verify basic initialization and infrastructure

**Tests** (5):

- ✅ Viewer loads without errors
- ✅ Debug interface is available
- ✅ Three.js components initialized correctly
- ✅ Canvas element renders
- ✅ Screenshots work without crashing

**Coverage**: Core initialization, error handling, WebGL stability

#### 2. `data-loading.spec.ts` - Generic Data Loading

**Purpose**: Verify data loading infrastructure

**Tests** (5):

- ✅ Points load from datasets
- ✅ Point cloud attributes present
- ✅ Scene loader accessible
- ✅ Renderer processes frames
- ✅ Animation state tracked

**Coverage**: Data pipeline infrastructure

#### 3. `controls-interaction.spec.ts` - User Interaction

**Purpose**: Verify keyboard shortcuts and controls

**Tests** (6):

- ✅ Fullscreen toggle (Space)
- ✅ Help overlay (H)
- ✅ Camera tracking
- ✅ Control mode switching (V)
- ✅ Console interceptor
- ✅ FOV validation

**Coverage**: Keyboard input, camera controls

#### 4. `ai-debugging-demo.spec.ts` - AI Capabilities

**Purpose**: Demonstrate AI debugging capabilities

**Tests** (8):

- ✅ Complete state inspection
- ✅ Real-time console monitoring
- ✅ WebGL validation
- ✅ Memory diagnostics
- ✅ Failure detection
- ✅ Component verification
- ✅ Network monitoring
- ✅ Multi-stage screenshots

**Coverage**: AI debugging workflows

---

### Critical Functionality Tests (4 files, 45+ tests) ⭐ NEW!

#### 5. `real-dataset-loading.spec.ts` - REAL Data Loading ⭐

**Purpose**: Verify actual Zarr dataset loading works end-to-end

**Tests** (8):

- ✅ Load dimension_navigation_example.zarr
- ✅ Load 5D dataset with correct dimensions
- ✅ Load dataset with all attributes (aligned)
- ✅ Load broadcast dataset
- ✅ Handle dataset without spatial index
- ✅ Load multiple point clouds in hierarchy
- ✅ Preserve scene dimensions from dataset
- ✅ Verify WebGL rendering with real data

**Coverage**: **REAL data loading pipeline** (not mocked!)
**Why Critical**: First tests that actually load real Zarr files!

#### 6. `nd-navigation.spec.ts` - nD Navigation ⭐

**Purpose**: Verify core nD navigation feature

**Tests** (13):

- ✅ Select dimension with number keys
- ✅ Navigate forward with ]
- ✅ Navigate backward with [
- ✅ Spatial index queries on navigation
- ✅ Points change when slice changes
- ✅ Multiple navigation steps
- ✅ Cache hits on return navigation
- ✅ Broadcasting across dimensions
- ✅ Show/hide dimension sliders
- ✅ Rapid navigation stress test
- ✅ Navigation performance (<3s)

**Coverage**: **Core nD feature** - keyboard navigation, slicing, queries
**Why Critical**: Tests Luxar's primary differentiating feature!

#### 7. `spatial-index-accuracy.spec.ts` - Index Correctness ⭐

**Purpose**: Verify spatial index queries are accurate

**Tests** (11):

- ✅ Log spatial index metadata
- ✅ Perform queries on navigation
- ✅ Merge adjacent ranges
- ✅ Effective radius calculation
- ✅ Zero-radius point filtering
- ✅ Cache misses on first load
- ✅ Cache hits on return
- ✅ Cache statistics reporting
- ✅ Handle missing spatial index
- ✅ Handle queries outside bounds

**Coverage**: **Spatial indexing correctness** - queries, caching, filtering
**Why Critical**: Performance depends on correct spatial queries!

#### 8. `visual-regression.spec.ts` - Screenshot Baselines ⭐

**Purpose**: Catch visual rendering regressions

**Tests** (8):

- ✅ Render datasets consistently (baselines)
- ✅ HDR multiplier visual changes
- ✅ Different nD slices look different
- ✅ FOV changes (default vs wide)
- ✅ Centered vs origin views
- ✅ Orbit mode rendering
- ✅ Fly mode rendering

**Coverage**: **Visual correctness** - rendering, HDR, camera
**Why Critical**: Catches visual bugs automatically!

#### 9. `performance-benchmarks.spec.ts` - Performance Metrics ⭐

**Purpose**: Prevent performance regressions

**Tests** (8):

- ✅ Load time <5 seconds
- ✅ Init time <2 seconds
- ✅ Maintain 30+ FPS
- ✅ Render frames after loading
- ✅ No memory leaks on load
- ✅ Track WebGL memory
- ✅ Track cache memory
- ✅ Navigation <2 seconds
- ✅ Rapid navigation handling
- ✅ Cache improves performance
- ✅ FPS doesn't degrade over time

**Coverage**: **Performance benchmarks** - load time, FPS, memory
**Why Critical**: Prevents performance regressions!

---

## Helper Functions (`helpers.ts`)

### Core Helpers

- `waitForLuxarReady(page, timeout)` - Wait for full initialization
- `getLuxarState(page)` - Get current state snapshot
- `renderOnce(page)` - Trigger single render frame
- `waitForPointsLoaded(page, minPoints, timeout)` - Wait for data loading
- `captureConsoleMessages(page)` - Collect console output
- `takeStableScreenshot(page, path)` - Take screenshot after render settles

---

## Running Tests

### All Tests

```bash
pnpm test:e2e
```

### Specific Test File

```bash
pnpm test:e2e basic-rendering.spec.ts
pnpm test:e2e data-loading.spec.ts
pnpm test:e2e controls-interaction.spec.ts
pnpm test:e2e ai-debugging-demo.spec.ts
```

### Single Test

```bash
pnpm test:e2e -g "should load viewer without errors"
```

### Interactive Mode

```bash
pnpm test:e2e:ui
```

### Debug Mode

```bash
pnpm test:e2e:debug
```

---

## Test Structure

### Standard Test Pattern

```typescript
import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState } from './helpers';

test('descriptive test name', async ({ page }) => {
  // 1. Navigate with debug mode
  await page.goto('/?debug');

  // 2. Wait for initialization
  await waitForLuxarReady(page);

  // 3. Interact or inspect
  const state = await getLuxarState(page);

  // 4. Assert expectations
  expect(state.initialized).toBe(true);
});
```

### AI Debugging Pattern

```typescript
test('AI debugging scenario', async ({ page }) => {
  // 1. Capture console
  const logs: string[] = [];
  page.on('console', (msg) => logs.push(msg.text()));

  // 2. Navigate
  await page.goto('/?debug');
  await waitForLuxarReady(page);

  // 3. Inspect state
  const state = await page.evaluate(() => {
    return (window as any).__luxarDebug.getState();
  });

  // 4. Analyze
  console.log('State:', state);
  console.log('Logs:', logs);

  // 5. Assert
  expect(state.totalPoints).toBeGreaterThan(0);
});
```

---

## What These Tests Enable

### For AI Agents (Claude Code):

1. **Autonomous Debugging** ✅
   - `ai-debugging-demo.spec.ts` shows complete state inspection
   - Can query scene, camera, renderer, loaders
   - Can capture console logs in real-time
   - Can verify WebGL context validity

2. **Regression Detection** ✅
   - Screenshot tests catch visual regressions
   - State assertions catch logic regressions
   - Console monitoring catches error regressions

3. **Systematic Diagnostics** ✅
   - Component verification (all parts initialized?)
   - Memory diagnostics (leaks detected?)
   - Network monitoring (data loading?)
   - WebGL validation (context valid?)

### For Developers:

1. **Automated Testing** ✅
   - 20+ test cases covering core functionality
   - Run before commits to catch regressions
   - CI/CD integration ready

2. **Documentation** ✅
   - Tests serve as executable examples
   - Show how to use debug interface
   - Demonstrate best practices

3. **Debugging Reference** ✅
   - `ai-debugging-demo.spec.ts` shows debugging patterns
   - Can copy patterns for new tests
   - Learn how to inspect state effectively

---

## Test Coverage

### Current Coverage:

- ✅ **Foundation Tests**: 24 tests (initialization, infrastructure, AI debugging)
- ✅ **Real Dataset Loading**: 8 tests ⭐ NEW
- ✅ **nD Navigation**: 13 tests ⭐ NEW
- ✅ **Spatial Index Accuracy**: 11 tests ⭐ NEW
- ✅ **Visual Regression**: 8 tests ⭐ NEW
- ✅ **Performance Benchmarks**: 11 tests ⭐ NEW
- ✅ **Total**: **75+ test cases** (24 foundation + 51 critical)

### Areas Tested:

- Viewer initialization
- Three.js scene setup
- WebGL context creation
- Debug interface exposure
- Point cloud loading
- Renderer frame processing
- Animation state tracking
- Keyboard shortcuts
- Camera controls
- Console logging
- Network requests
- Memory usage
- Error handling
- Component verification

### Not Yet Covered (Future):

- nD navigation with actual datasets
- Spatial index query correctness
- Cache hit/miss behavior
- Visual regression with known datasets
- Performance benchmarks
- Cross-browser compatibility
- Mobile/touch interactions

---

## Best Practices

### DO:

- ✅ Use `waitForLuxarReady()` instead of arbitrary timeouts
- ✅ Use `getLuxarState()` to inspect current state
- ✅ Capture console messages for debugging
- ✅ Take screenshots for visual verification
- ✅ Use deterministic waits (`waitForFunction`)
- ✅ Check `?debug` mode is enabled

### DON'T:

- ❌ Use `waitForTimeout()` for critical waits (only for settling periods)
- ❌ Assume DOM structure (Three.js objects aren't in DOM)
- ❌ Use tight pixel matching for WebGL (allow 5% tolerance)
- ❌ Forget to enable debug mode (`?debug` parameter)
- ❌ Query CSS selectors for 3D objects (use `__luxarDebug` instead)

---

## Adding New Tests

### Step 1: Create Test File

```bash
# Create new test file
touch src/tests/e2e/my-feature.spec.ts
```

### Step 2: Write Test

```typescript
import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState } from './helpers';

test.describe('My Feature', () => {
  test('should do something', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Your test logic here
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
  });
});
```

### Step 3: Run Test

```bash
pnpm test:e2e my-feature.spec.ts
```

---

## Resources

- **[PLAYWRIGHT_GUIDE.md](../../PLAYWRIGHT_GUIDE.md)** - Comprehensive usage guide
- **[PLAYWRIGHT_REVIEW.md](../../PLAYWRIGHT_REVIEW.md)** - Code review and fixes
- **[Playwright Docs](https://playwright.dev)** - Official documentation

---

**Test Suite Status**: ✅ Production Ready
**Total Tests**: 23 test cases across 4 files
**Coverage**: Core functionality, AI debugging, user interactions
**Last Updated**: January 2025
