# Luxar E2E Testing - Quick Reference Guide

> For a comprehensive Playwright setup and debugging guide, see [`PLAYWRIGHT_GUIDE.md`](../developer/PLAYWRIGHT_GUIDE.md).

## 🎯 Console Output Capture - YES, It Works!

### Three Methods to Capture Browser Console:

#### 1. Real-Time Console Listener (Most Common)
```typescript
test('my test', async ({ page }) => {
  const logs: string[] = [];

  page.on('console', (msg) => {
    console.log(`[${msg.type()}]`, msg.text());
    logs.push(msg.text());
  });

  await page.goto('/?src=/data.zarr&debug');

  // Analyze logs
  const errors = logs.filter(log => log.includes('Error'));
  expect(errors).toEqual([]);
});
```

#### 2. Console Interceptor History
```typescript
const messages = await page.evaluate(() => {
  return window.__luxarDebug.consoleInterceptor.messages;
});
```

#### 3. Agent Driver Tool (Manual Debugging)
```bash
pnpm agent:debug              # Headless - shows console in terminal
pnpm agent:debug:visible      # Visible browser
```

See: `ai-debugging-demo.spec.ts` for complete examples

---

## 🧪 Test Suite Organization

### Test Files (check `packages/luxar-viewer/src/tests/e2e/` for current count; 29 spec files as of 2026-03-31)

Key test suites include:

| File | Purpose |
|------|---------|
| **ai-debugging-demo.spec.ts** | AI debugging capabilities demo |
| **all-examples-smoke-test.spec.ts** | Comprehensive smoke tests for all examples |
| **basic-rendering.spec.ts** | Initialization, canvas, errors |
| **cache-system.spec.ts** | OPFS caching functionality |
| **controls-interaction.spec.ts** | Keyboard, mouse, camera |
| **custom-gui-library.spec.ts** | Custom GUI panel testing |
| **data-monitor-metrics.spec.ts** | Performance monitoring |
| **demo-validation.spec.ts** | Python script syntax + .zarr output checks |
| **dimension-animation.spec.ts** | Dimension animation playback |
| **dimension-initialization.spec.ts** | nD dimension setup |
| **error-recovery.spec.ts** | ⭐ Error handling |
| **first-time-ux.spec.ts** | Browser, help, error messages |
| **geometry-types.spec.ts** | Lines & GSplats rendering |
| **keyboard-input-system.spec.ts** | Fly controls, shortcuts, modifiers |
| **nd-navigation.spec.ts** | ⭐ Core nD feature! |
| **nd-transforms.spec.ts** | nD transform inverse-query |
| **performance-tracking.spec.ts** | ⭐ Regression detection + memory/FPS |
| **position-bounds-clipping.spec.ts** | Boundary testing |
| **python-typescript-integration.spec.ts** | ⭐ Cross-language E2E |
| **real-dataset-loading.spec.ts** | Real .zarr files + dataset switching |
| **recording-panel.spec.ts** | Screenshot/video capture panel |
| **rendering-controls.spec.ts** | Rendering panel, FOV, controls |
| **spatial-index-accuracy.spec.ts** | Query accuracy, caching |
| **test-fixtures-rendering.spec.ts** | Encoding/decoding compatibility |
| **theme-visual-regression.spec.ts** | Theme visual regression |
| **transform-hierarchy.spec.ts** | ⭐ Transform correctness |
| **viewer-initialization.spec.ts** | Viewer startup without data |
| **visual-regression.spec.ts** | Screenshot comparison |
| **webgl-errors.spec.ts** | WebGL error detection |
| **worker-wasm-integration.spec.ts** | Worker + WASM integration |

---

## 🚀 Common Test Commands

```bash
# Run all E2E tests
pnpm test:e2e

# Run specific category
pnpm test:e2e --grep "Python.*TypeScript"
pnpm test:e2e --grep "Transform"
pnpm test:e2e --grep "Error Recovery"
pnpm test:e2e --grep "Performance"

# Debug mode (interactive UI)
pnpm test:e2e:ui

# Run with headed browser (see it run)
pnpm test:e2e --headed

# Update visual regression baselines
pnpm test:e2e --grep "Visual" --update-snapshots

# View HTML report
pnpm test:e2e:report

# AI debugging mode
pnpm agent:debug
```

---

## 📋 Pre-Test Checklist

Before running E2E tests:

1. ✅ Generate example datasets: `make run-examples`
2. ✅ Install Playwright browsers: `pnpm exec playwright install chromium`
3. ✅ Ensure ports 5173 and 9000 are free
4. ✅ Run from project root or luxar-viewer directory

---

## 🔍 Debugging Failed Tests

### Step 1: View HTML Report
```bash
pnpm test:e2e:report
```

Shows:
- Screenshots of every test
- Videos of failures
- Trace files for debugging
- Error context

### Step 2: Check Error Context
Each failed test creates:
- `test-results/{test-name}/error-context.md` - Detailed error info
- `test-results/{test-name}/test-failed-1.png` - Screenshot at failure
- `test-results/{test-name}/trace.zip` - Full execution trace

### Step 3: Run Single Test in Debug Mode
```bash
pnpm test:e2e --grep "specific test name" --debug
```

### Step 4: Use Agent Driver for Manual Inspection
```bash
pnpm agent:debug:visible  # Watch the browser
```

---

## 🎓 Writing New E2E Tests

### Template:
```typescript
import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState, waitForSpatialQuery } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/my_dataset.zarr';

test.describe('My Feature Tests', () => {
  test('should do something', async ({ page }) => {
    // 1. Load viewer with dataset
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // 2. Get initial state
    const initialState = await getLuxarState(page);

    // 3. Perform action
    await page.keyboard.press('4');
    await page.waitForTimeout(100); // Brief wait for key processing
    await page.keyboard.press(']');

    // 4. Wait for condition (NOT arbitrary timeout!)
    await waitForSpatialQuery(page);

    // 5. Verify result
    const finalState = await getLuxarState(page);
    expect(finalState.totalPoints).toBeGreaterThan(0);
  });
});
```

### Best Practices:

1. **Always use `?debug` URL parameter** - Exposes `window.__luxarDebug`
2. **Use condition waits, not timeouts** - `waitForSpatialQuery()` not `waitForTimeout(3000)`
3. **Capture console early** - Set up `page.on('console')` before navigation
4. **Check error states** - Verify graceful handling, not just happy paths
5. **Use helper functions** - `waitForLuxarReady()`, `getLuxarState()`, etc.

---

## 🐛 Common Issues & Solutions

### Issue: Tests timeout waiting for initialization
**Solution**: Check that:
- Vite dev server started (port 5173)
- Python HTTP server started (port 9000)
- Datasets exist in `datasets/examples/`

```bash
make run-examples  # Generate datasets
lsof -ti:5173 | xargs kill -9  # Kill stuck servers
```

### Issue: Visual regression tests fail
**Solution**: GPU rendering varies. Update baselines:
```bash
pnpm test:e2e --grep "Visual" --update-snapshots
```

### Issue: "Executable doesn't exist" error
**Solution**: Install Playwright browsers:
```bash
pnpm exec playwright install chromium
```

### Issue: Tests flaky/intermittent failures
**Solution**: Replace `waitForTimeout` with condition waits:
```typescript
// BEFORE (flaky):
await page.keyboard.press(']');
await page.waitForTimeout(3000);

// AFTER (reliable):
await page.keyboard.press(']');
await waitForSpatialQuery(page);
```

---

## 📊 Performance Baselines

Location: `packages/luxar-viewer/performance-baselines.json`

Tracks:
- `loadTime` - Dataset load time (ms)
- `initTime` - Viewer initialization (ms)
- `navigationTime` - nD navigation (ms)
- `fps` - Rendering frame rate

**How it works**:
1. First run: Creates baseline
2. Subsequent runs: Fails if >30% slower
3. Auto-updates when faster (continuous improvement)

**CI Integration**:
```yaml
# .github/workflows/test.yml
- name: Run E2E Tests
  run: pnpm test:e2e

- name: Upload Performance Metrics
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: performance-baselines
    path: packages/luxar-viewer/performance-baselines.json
```

---

## 🎯 Test Coverage Checklist

✅ **Core Features**:
- [x] Basic rendering & initialization
- [x] nD navigation (dimension selection, slicing)
- [x] Data loading (points, attributes, hierarchy)
- [x] Controls (orbit, fly, keyboard shortcuts)
- [x] Spatial index queries
- [x] Visual rendering (screenshot regression)

✅ **Integration**:
- [x] Python→TypeScript encoding/decoding
- [x] Transform hierarchy (parent→child)
- [x] Multi-level hierarchies
- [x] nD dataset handling

✅ **Performance**:
- [x] Load time tracking
- [x] Frame rate measurement
- [x] Memory usage monitoring
- [x] Navigation responsiveness
- [x] Regression detection

✅ **Error Handling**:
- [x] Invalid datasets
- [x] Network failures
- [x] WebGL context issues
- [x] Data validation
- [x] Memory limits

✅ **User Experience**:
- [x] First-time user flow
- [x] Error messages
- [x] Help overlay
- [x] Dataset browser

---

## 🔬 Advanced Testing Techniques

### Accessing Internal State
```typescript
const internalState = await page.evaluate(() => {
  const debug = window.__luxarDebug;

  return {
    scene: debug.scene.children.length,
    camera: debug.camera.position.toArray(),
    renderer: debug.renderer.info.render,
    controls: debug.controls.getControlType(),
    // ... any internal state you need
  };
});
```

### Triggering Manual Renders
```typescript
await page.evaluate(() => {
  window.__luxarDebug.renderOnce();
});
```

### Inspecting Spatial Index
```typescript
const spatialIndex = await page.evaluate(async () => {
  const loader = await window.__luxarDebug.getSceneLoader();
  const defaultLoader = loader.getDefaultLoader();

  return {
    gridShape: defaultLoader.spatialIndex.gridShape,
    occupiedCells: defaultLoader.spatialIndex.occupiedCells.length,
  };
});
```

### Taking Custom Screenshots
```typescript
// Specific element
await page.locator('canvas').screenshot({
  path: 'test-results/my-canvas.png'
});

// Full page
await page.screenshot({
  path: 'test-results/full-page.png',
  fullPage: true
});
```

---

## 🎉 Summary

The Luxar E2E test suite is now:
- ✅ **Comprehensive** - 226 tests covering all features
- ✅ **Reliable** - Condition-based waits, not arbitrary timeouts
- ✅ **Fast** - Optimized waits save 40%+ test time
- ✅ **Robust** - Error recovery prevents crashes
- ✅ **Regression-Proof** - Performance tracking catches slowdowns
- ✅ **AI-Ready** - Full console access for autonomous debugging

**Test Count**: 226 tests across 23 test suites

**Ready for Production!** 🚀
