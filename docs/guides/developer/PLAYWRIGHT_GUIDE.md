# Playwright Testing & AI-Assisted Debugging Guide

> For a quick-reference E2E testing cheat sheet, see [`E2E_TESTING_GUIDE.md`](../user/E2E_TESTING_GUIDE.md).

This guide explains how to use Playwright for testing and AI-assisted development with the Luxar viewer.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [For AI Agents (Claude Code)](#for-ai-agents-claude-code)
- [Available Scripts](#available-scripts)
- [Writing Tests](#writing-tests)
- [Debugging](#debugging)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Luxar viewer uses **Playwright** for end-to-end testing and AI-assisted debugging. This allows:

1. **Automated Testing**: Run tests that verify rendering, data loading, and UI behavior
2. **AI Debugging**: Claude Code can "see" and debug the app without a physical monitor
3. **Visual Regression**: Catch visual bugs with screenshot comparisons
4. **Performance Monitoring**: Track FPS, memory usage, and loading times

### Why Playwright for Three.js/WebGL?

Standard DOM testing tools don't work for WebGL apps because they can't see inside the `<canvas>`. Playwright solves this by:
- **GPU Acceleration**: Forces hardware rendering (not software fallback)
- **State Inspection**: Accesses Three.js scene via `window.__luxarDebug`
- **Screenshot Testing**: Captures actual rendered output
- **Console Mirroring**: Pipes browser logs to terminal

---

## Quick Start

### 1. Prerequisites

Ensure you have Playwright installed (already done if you cloned the repo):

```bash
cd packages/luxar-viewer
pnpm install
```

### 2. Run the Agent Driver (AI Debugging)

This is the primary tool for AI-assisted debugging:

```bash
# Headless mode (for Claude Code)
pnpm agent:debug

# With visible browser (for manual observation)
pnpm agent:debug:visible

# Custom URL
pnpm agent:debug --url="http://localhost:5173/?src=/data/my-dataset.luxar.zarr&debug"
```

**Output**:
- Terminal: All browser console logs, errors, and Luxar state
- `test-results/debug/debug-view.png`: Screenshot of current state
- `test-results/debug/error-state.png`: Screenshot on failure (if any)

### 3. Run E2E Tests

```bash
# Run all tests
pnpm test:e2e

# Run with UI (interactive mode)
pnpm test:e2e:ui

# Debug mode (step through tests)
pnpm test:e2e:debug

# View last test report
pnpm test:e2e:report
```

---

## For AI Agents (Claude Code)

### Your New Capabilities

When debugging the Luxar viewer, you can now:

1. **See What's Happening**: Run `pnpm agent:debug` to see browser console logs
2. **Inspect Scene State**: View Three.js objects, point counts, camera position
3. **Take Screenshots**: Verify visual rendering via `debug-view.png`
4. **Monitor Performance**: Check FPS, memory usage, loading times

### How to Debug

#### Step 1: Run the Agent Driver

```bash
pnpm agent:debug
```

#### Step 2: Read the Output

Look for these key sections:

```
[BROWSER-CONSOLE-LOG] Loading dataset from /data/demo.luxar.zarr...
[BROWSER-CONSOLE-ERROR] Failed to load spatial index
[NETWORK-FAIL] http://localhost:5173/data/demo.luxar.zarr/.zarray - 404
```

**Color Coding**:
- 🔴 `[BROWSER-CONSOLE-ERROR]` - JavaScript errors
- 🟡 `[BROWSER-CONSOLE-WARN]` - Warnings
- 🔵 `[BROWSER-CONSOLE-INFO]` - Info logs
- ⚫ `[BROWSER-CONSOLE-LOG]` - Debug logs

#### Step 3: Inspect Luxar State

The driver outputs a JSON dump of the current state:

```json
{
  "timestamp": "2025-01-15T10:30:00.000Z",
  "scene": {
    "totalChildren": 5,
    "pointClouds": 3,
    "pointCloudDetails": [
      {
        "name": "cells",
        "pointCount": 125000,
        "visible": true,
        "hasColors": true,
        "hasRadii": true
      }
    ]
  },
  "camera": {
    "position": { "x": 0, "y": 0, "z": 10 },
    "fov": 47
  },
  "performance": {
    "totalPoints": 125000
  }
}
```

#### Step 4: Analyze and Fix

Based on the output, you can:
- **No errors**: Everything working correctly
- **Network errors**: Dataset path wrong or server not running
- **Zero points**: Spatial index query failing
- **Console errors**: Logic bugs in the code

#### Step 5: Verify Fix

After making changes:

```bash
# Run driver again
pnpm agent:debug

# Check that errors are gone
# Verify point count is correct
# Check screenshot looks good
```

### Debugging Workflow Example

```bash
# Scenario: User reports "Points not loading in 4D dataset"

# 1. Run driver to see current state
pnpm agent:debug

# 2. Check output (see 0 points loaded)
[BROWSER-CONSOLE-LOG] Query result: 0 cells → 0 ranges → 0 points

# 3. Add instrumentation
# Edit src/data/points/points-spatial-index-loader.ts
console.log('[DEBUG] Slice position:', slicePosition);
console.log('[DEBUG] Query tolerance:', queryTolerance);

# 4. Run again
pnpm agent:debug

# 5. Analyze output
[BROWSER-CONSOLE-LOG] [DEBUG] Slice position: [0, 0, 0, 10]
[BROWSER-CONSOLE-LOG] [DEBUG] Query tolerance: [0, 0, 0, 0]  ← BUG!

# 6. Fix the bug (tolerance should be > 0 for discrete dims)
# ... make fix ...

# 7. Verify
pnpm agent:debug
[BROWSER-CONSOLE-LOG] Query result: 50 cells → 10 ranges → 12000 points ✅
```

### Important Rules for AI Debugging

1. **Don't try to find DOM elements**: Three.js objects don't have HTML tags
2. **Use `window.__luxarDebug`**: Query the scene state via this global
3. **Check screenshots**: Visual bugs need visual verification
4. **Watch for WebGL errors**: Look for "context lost" or shader errors
5. **Test with ?debug**: Always add `?debug` to URL for state inspection

---

## Available Scripts

### Agent Driver Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `pnpm agent:debug` | Run headless browser, output logs | AI debugging, CI |
| `pnpm agent:debug:visible` | Run with visible browser | Manual verification |

### E2E Test Scripts

> **CI status**: GitHub Actions E2E job is currently disabled
> (`.github/workflows/ci.yml` `e2e-tests: if: false`). The disabled
> job calls `pnpm test:e2e:smoke` if re-enabled. Local-only.

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `pnpm test:e2e` | Run all tests | Local manual testing |
| `pnpm test:e2e:smoke` | Non-GPU smoke subset | Local CI mirror; basis for the disabled workflow |
| `pnpm test:e2e:ui` | Interactive test runner | Writing new tests |
| `pnpm test:e2e:debug` | Debug mode | Debugging failing tests |
| `pnpm test:e2e:report` | View last test report | After test run |

---

## Writing Tests

### Best Practices for Test Assertions ⭐

**DO**:
- ✅ Use specific wait helpers instead of arbitrary timeouts
- ✅ Verify actual behavior (point counts, state changes)
- ✅ Check for both success conditions AND what changed
- ✅ Use `waitForDataLoaded()` after navigation
- ✅ Use `waitForDimensionNavigation()` for nD tests

**DON'T**:
- ❌ Use `await page.waitForTimeout(3000)` - too fragile
- ❌ Only check `expect(state.initialized).toBe(true)` - too weak
- ❌ Use `expect(typeof x).toBe('number')` - always passes!
- ❌ Hide errors with `.catch(() => false)` - masks real issues

### Basic Test Structure

```typescript
// Import from `./fixtures` (NOT @playwright/test directly) so the
// shared console-error / pageerror auto-check runs after each test.
// Specs that need to allow specific noisy messages annotate at the
// test level — see `fixtures.ts:DEFAULT_ALLOWED_CONSOLE_ERRORS`.
import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  getLuxarState,
  waitForDataLoaded,
  waitForDimensionNavigation,
} from './helpers';

test('should load demo dataset', async ({ page }) => {
  // Use `?src=` with a full dataset path served by the dev server
  // (typical: http://localhost:9000/datasets/examples/...). The
  // older `/data/...` path used in some examples is not the
  // production convention.
  await page.goto('/?src=http://localhost:9000/datasets/examples/demo.luxar.zarr&debug');

  // Wait for Luxar to initialize
  await waitForLuxarReady(page);

  // Wait for data to actually load (better than arbitrary timeout!)
  await waitForDataLoaded(page);

  // Get current state
  const state = await getLuxarState(page);

  // Strong assertions
  expect(state.totalPoints).toBeGreaterThan(0);
  expect(state.pointClouds.length).toBeGreaterThan(0);
  expect(state.initialized).toBe(true);

  // Visual regression
  await expect(page).toHaveScreenshot('demo-loaded.png', {
    maxDiffPixelRatio: 0.05,  // 5% tolerance for WebGL
    threshold: 0.2            // Color tolerance
  });
});
```

### Using Helper Functions

```typescript
import {
  waitForLuxarReady,
  getLuxarState,
  waitForDataLoaded,
  waitForDimensionNavigation,
  renderOnce,
  captureConsoleMessages
} from './helpers';

test('advanced nD navigation test', async ({ page }) => {
  // Capture console messages
  const console = captureConsoleMessages(page);

  await page.goto('/?src=/examples/5d-dataset.luxar.zarr&debug');
  await waitForLuxarReady(page);
  await waitForDataLoaded(page);

  // Get initial state
  const initialState = await getLuxarState(page);
  const initialPoints = initialState.totalPoints;

  // Navigate through dimension
  await page.keyboard.press('4');  // Select dimension 4
  await page.waitForTimeout(300);  // Short delay for key processing
  await page.keyboard.press(']');  // Navigate forward

  // Wait for navigation to complete (robust!)
  await waitForDimensionNavigation(page, initialPoints, 8000);

  const finalState = await getLuxarState(page);

  // Strong assertions
  expect(finalState.initialized).toBe(true);
  expect(finalState.totalPoints).toBeGreaterThanOrEqual(0);

  // Verify something actually changed
  const pointsChanged = finalState.totalPoints !== initialPoints;
  const hasQueryLogs = console.logs.some(log => log.includes('Query result:'));
  expect(pointsChanged || hasQueryLogs).toBe(true);

  // Check for errors
  expect(console.errors).toEqual([]);
});
```

### Testing nD Navigation

```typescript
test('should navigate through dimensions', async ({ page }) => {
  await page.goto('/?src=/data/4d-dataset.luxar.zarr&debug');
  await waitForLuxarReady(page);

  // Press '1' to select dimension 0
  await page.keyboard.press('1');

  // Press ']' to navigate forward
  await page.keyboard.press(']');

  // Wait for points to update
  await page.waitForTimeout(1000);

  // Verify new points loaded
  const state = await getLuxarState(page);
  expect(state.totalPoints).toBeGreaterThan(0);
});
```

### Screenshot Testing

```typescript
test('visual regression', async ({ page }) => {
  await page.goto('/?src=/data/demo.luxar.zarr&debug');
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page, 1000);

  // Take baseline screenshot
  await expect(page).toHaveScreenshot('baseline.png', {
    maxDiffPixelRatio: 0.05,  // CRITICAL: Allow 5% pixel difference
    threshold: 0.2,            // CRITICAL: Allow color variance
    animations: 'disabled'     // CRITICAL: WebGL doesn't pause
  });
});
```

**Why Relaxed Thresholds?**
- WebGL rendering varies across GPUs
- Anti-aliasing differs between hardware
- CI servers use different GPUs than dev machines
- 0% tolerance will fail 100% of the time

---

## Debugging

### Debug Individual Tests

```bash
# Run single test file
pnpm test:e2e src/tests/e2e/basic-rendering.spec.ts

# Run single test by name
pnpm test:e2e -g "should load viewer"

# Debug mode (pause before each test)
pnpm test:e2e:debug
```

### View Traces

When tests fail, Playwright captures a trace:

```bash
# View trace from last run
pnpm test:e2e:report

# Or open specific trace
npx playwright show-trace test-results/trace.zip
```

**Trace Contains**:
- Video recording of test
- DOM snapshots at each step
- Network requests/responses
- Console logs
- Screenshots

### Common Issues

#### 1. Test Times Out

**Problem**: Test exceeds 60-second timeout

**Solutions**:
- Check if dev server is running (`pnpm dev`)
- Increase `waitTime` in agent-driver.ts
- Check for infinite loops in code

#### 2. Screenshot Doesn't Match

**Problem**: Visual regression fails with pixel differences

**Solutions**:
- First run creates baseline (expected to fail)
- Re-run to compare against baseline
- If legitimate change: Update baseline with `--update-snapshots`

```bash
pnpm test:e2e --update-snapshots
```

#### 3. GPU Acceleration Not Working

**Problem**: Tests slow or screenshots look different

**Check**: Browser flags in `playwright.config.ts`:
```typescript
args: [
  '--use-gl=egl',           // Must be present
  '--ignore-gpu-blocklist'
]
```

**Verify**: Check test output for "SwiftShader" (software renderer)

#### 4. Debug Mode Not Enabled

**Problem**: `window.__luxarDebug` is undefined

**Solution**: Add `?debug` to URL:
```typescript
await page.goto('/?debug');  // ← Must include
```

---

## Troubleshooting

### "No tests found"

**Cause**: Test files not in `src/tests/e2e/` or don't match `*.spec.ts` pattern

**Fix**:
```bash
# Rename files to match pattern
mv my-test.ts my-test.spec.ts

# Or update testDir in playwright.config.ts
```

### "Cannot find module @playwright/test"

**Cause**: Playwright not installed

**Fix**:
```bash
cd packages/luxar-viewer
pnpm install
npx playwright install chromium
```

### "Page crashed"

**Cause**: WebGL context lost or out of memory

**Fixes**:
- Close other GPU-intensive apps
- Reduce dataset size for tests
- Add GPU error handling to code

### "Browser context closed"

**Cause**: Test timeout or crash

**Fix**:
- Check for infinite loops
- Increase timeout in `playwright.config.ts`
- Check console for errors

---

## Best Practices

### For AI Agents

1. **Always run agent driver first**: Don't guess, verify actual state
2. **Add targeted logging**: Insert console.logs to narrow down issues
3. **Check screenshots**: Visual bugs need visual confirmation
4. **Test fixes immediately**: Run driver after every change
5. **Clean up debug logs**: Remove console.logs after debugging

### For Developers

1. **Run tests before committing**: `pnpm test:e2e`
2. **Update snapshots carefully**: Only when changes are intentional
3. **Write focused tests**: One behavior per test
4. **Use helper functions**: Reuse common patterns
5. **Test on real data**: Don't rely only on demo datasets

### For CI/CD

1. **Run tests in Docker**: Consistent GPU environment
2. **Retry flaky tests**: WebGL can be non-deterministic (max 2 retries)
3. **Save artifacts**: Keep screenshots and traces on failure
4. **Monitor test duration**: WebGL tests are slower (60s typical)

---

## Advanced Topics

### Coordinate-Based Interactions

Three.js objects don't have CSS selectors. To click a 3D object:

```typescript
// Helper to project 3D position to screen coordinates
const getScreenPosition = async (page, objectName) => {
  return await page.evaluate((name) => {
    const obj = window.__luxarDebug.scene.getObjectByName(name);
    const camera = window.__luxarDebug.camera;

    // Project to normalized device coordinates
    const vector = obj.position.clone();
    vector.project(camera);

    // Convert to screen pixels
    const width = window.innerWidth;
    const height = window.innerHeight;

    return {
      x: (vector.x * 0.5 + 0.5) * width,
      y: (-(vector.y * 0.5) + 0.5) * height
    };
  }, objectName);
};

// Usage
const coords = await getScreenPosition(page, 'my-mesh');
await page.mouse.click(coords.x, coords.y);
```

### Performance Testing

```typescript
test('performance benchmark', async ({ page }) => {
  await page.goto('/?debug');
  await waitForLuxarReady(page);

  // Measure FPS
  const fps = await page.evaluate(async () => {
    const start = Date.now();
    let frames = 0;

    await new Promise(resolve => {
      const id = setInterval(() => {
        frames++;
        if (Date.now() - start > 5000) {
          clearInterval(id);
          resolve(null);
        }
      }, 16);  // ~60fps
    });

    return frames / 5;  // FPS over 5 seconds
  });

  expect(fps).toBeGreaterThan(50);  // At least 50 FPS
});
```

### Memory Leak Testing

```typescript
test('no memory leaks', async ({ page }) => {
  await page.goto('/?debug');

  const initialMemory = await page.evaluate(() => {
    return (performance as any).memory?.usedJSHeapSize || 0;
  });

  // Load and unload dataset 10 times
  for (let i = 0; i < 10; i++) {
    await page.evaluate(() => window.__luxarDebug.scene.clear());
    // Load dataset again...
  }

  const finalMemory = await page.evaluate(() => {
    return (performance as any).memory?.usedJSHeapSize || 0;
  });

  // Memory should not grow unbounded
  const growthMB = (finalMemory - initialMemory) / 1024 / 1024;
  expect(growthMB).toBeLessThan(100);  // Less than 100MB growth
});
```

---

## Resources

- [Playwright Documentation](https://playwright.dev)
- [Three.js Testing Guide](https://threejs.org/docs/#manual/en/introduction/Testing)
- [WebGL Debugging Tools](https://www.khronos.org/webgl/wiki/Debugging)

---

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `pnpm agent:debug` to see detailed logs
3. Check `playwright-report/` for test traces
4. Open an issue with screenshots and error logs

**Happy Testing! 🎭**
