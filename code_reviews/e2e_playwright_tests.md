# E2E Playwright Test Suite -- Critical Review

**Date**: 2026-03-31
**Scope**: All 29 spec files in `packages/luxar-viewer/src/tests/e2e/`
**Reviewer**: Claude Code (automated review)
**Last updated**: 2026-03-31 (post PR #53 fixes)

---

## Status: Issues Fixed in PR #53

The following issues from the original review have been addressed:

### error-recovery.spec.ts (3 CRITICAL issues resolved)
- **Corrupted .zmetadata test**: Was navigating to a non-existent `http://localhost:9000/corrupted.zarr` (testing 404, not corruption). Now uses `page.route('**/.zmetadata', ...)` to intercept and return corrupted JSON `'{"this is not valid json: {{[[[' ` against a real dataset. **(CRITICAL -> FIXED)**
- **Missing positions test**: Was navigating to non-existent `no-positions.zarr` (testing 404 again). Now uses `page.route('**/positions/**', ...)` and `page.route('**/positions/.zarray', ...)` to return 404 for position data while letting metadata through. **(CRITICAL -> FIXED)**
- **Network failure mid-load test**: Was loading `/?debug` with no dataset and asserting `initialized === true` (tested nothing). Now uses `page.route()` to let metadata through and abort chunk requests after the first 2 succeed, simulating real mid-load failure. **(CRITICAL -> FIXED)**

### data-monitor-metrics.spec.ts (1 CRITICAL + 1 HIGH issue resolved)
- **No assertion on monitorVisible (line 66)**: Was `console.log(...)` only, always passed. Now asserts `expect(monitorVisible).toBe(true)`. **(CRITICAL -> FIXED)**
- **Silent if/return guards**: `if (!metrics) { return; }` patterns replaced with `expect(metrics).toBeTruthy()` and `expect(metrics!.hasMonitor).toBe(true)`, so missing monitor API now fails the test. **(HIGH -> FIXED)**

### controls-interaction.spec.ts (3 HIGH issues resolved)
- **Fullscreen no-op test**: Was always passing regardless of outcome. Now marked as `test.fixme('fullscreen is blocked in headless Chromium')` so it is explicitly skipped and visible in reports. **(HIGH -> FIXED)**
- **Camera test used JS property assignment**: Was directly mutating `camera.position.z` and reading it back (tested nothing). Now performs a real mouse drag (`page.mouse.down() -> move -> up()`) and verifies camera position changed. **(HIGH -> FIXED)**
- **Control mode switching didn't verify change**: `expect(['orbit', 'arcball', 'fly']).toContain(newType)` passed even without mode change. Now also asserts `expect(newType).not.toBe(initialType)`. **(HIGH -> FIXED)**

### Partially addressed
- **Help overlay selector (controls-interaction.spec.ts)**: The ultra-broad `body.includes('Help')` fallback was tightened, but still uses `body.includes('Keyboard Shortcuts')` which is fragile (see remaining finding #4.2). **(MEDIUM -> PARTIALLY FIXED)**

---

## Executive Summary

The E2E test suite is extensive (29 spec files, ~300+ individual tests) and covers the viewer's major subsystems: initialization, data loading, nD navigation, caching, keyboard input, geometry types, visual regression, and more. Overall quality is solid -- the helpers module is well-designed, and most tests follow the CLAUDE.md conventions (`?src=` URLs, `?debug` parameter, `window.__luxarDebug` waits).

However, there are systemic issues: **weak assertions** (many tests verify only `state.initialized === true` instead of actual behavior), **excessive `waitForTimeout` usage** despite the helpers providing condition-based waits, **significant overlap** between several spec files, and a few tests that enforce **wrong or untestable behavior**.

### Severity Ratings

- **CRITICAL**: Test is wrong, enforces incorrect behavior, or gives false confidence
- **HIGH**: Significant flaw that undermines test value (flaky, weak assertion, missing coverage)
- **MEDIUM**: Suboptimal but not harmful (redundancy, style issues)
- **LOW**: Minor improvement opportunity

---

## Per-Spec Analysis

### 1. all-examples-smoke-test.spec.ts

**Severity: MEDIUM**

**Rigor**: Good. Loads every example, checks for console errors, WebGL errors, and positive point counts. The `assertNoConsoleErrors` pattern with `allowedPatterns` is well-designed.

**Issues**:
- **`KNOWN_FLAKY_LARGE_DATASETS` uses `test.skip`** -- these are permanently skipped, which means regressions in temporal/time-series datasets are invisible. Should use `test.fixme` with a tracking issue, or investigate the root cause. (HIGH)
- **`hasLoadSuccess` assertion** (line 178) checks for `Scene loaded successfully|Loaded.*points` in console logs. This is fragile -- if the log message wording changes, all tests break silently (the assertion would fail, not be skipped). (MEDIUM)
- **Overlap with `webgl-errors.spec.ts`**: Both iterate over datasets checking WebGL errors. The smoke test inlines WebGL error checking instead of using the `getWebGLErrors` helper. (LOW)

**Missing Coverage**: No test for datasets with only Lines or GSplats geometry (only Points-based examples are listed).

---

### 2. basic-rendering.spec.ts

**Severity: MEDIUM**

**Rigor**: Adequate for initialization smoke tests.

**Issues**:
- **`should handle missing dataset gracefully`** (line 45): Uses `.catch(() => {})` on the `waitForFunction`, swallowing failures. If neither `.error-message` nor `.dataset-browser` appears, the test still passes because the catch block is entered, then the test checks for one of two selectors using `.catch(() => false)`. This is fine but obscures failures. (LOW)
- **Visual regression test** (line 126): `maxDiffPixelRatio: 0.1` and `threshold: 0.3` are extremely loose tolerances. At 10% pixel difference and 0.3 color threshold, significant rendering regressions would pass. (HIGH)
- **Overlap with `viewer-initialization.spec.ts`**: Both test `/?debug` initialization, Three.js scene components, canvas element, and renderer frames. (MEDIUM)

---

### 3. cache-system.spec.ts

**Severity: MEDIUM**

**Rigor**: Good coverage of L0/L1/L2 cache layers and debug API. Tests actual browser OPFS behavior.

**Issues**:
- **`beforeEach` OPFS cleanup** (line 25-37): The OPFS cleanup uses `(dir as any).removeEntry?.({ recursive: true })` which is not a standard OPFS API call. The correct call is `root.removeEntry('luxar-cache', { recursive: true })`. This cleanup may silently fail, causing test pollution. (HIGH)
- **Multiple `waitForTimeout(1000-2000)` calls**: Lines 131, 150, 199, 270, 292. The helpers provide `waitForDataLoaded` and `waitForNextRender` which should be used instead. (MEDIUM)
- **`should track L0 cache hits on second chunk access`** (line 177): Uses `page.mouse.wheel(0, 10)` to trigger a re-render, but this may not actually trigger a chunk re-read (zoom doesn't re-query spatial index for 3D datasets). The test may be asserting on unchanged values. (HIGH)
- **`should disable all caching with ?no-cache parameter`** (line 213): Soft assertion -- uses `if (cacheStats.l0)` guard, so if L0 stats are missing entirely, it passes. (MEDIUM)

---

### 4. controls-interaction.spec.ts

**Severity: LOW** (improved from MEDIUM after PR #53 fixes)

**Rigor**: Reasonable for basic keyboard shortcut verification.

**Remaining Issues**:
- **`should show help overlay with H key`**: The tightened selector still falls back to `body.includes('Keyboard Shortcuts')` which is fragile -- any DOM element containing that string would pass. Should use a specific CSS selector. (MEDIUM)
- **Overlap with `keyboard-input-system.spec.ts`**: Both test 'v' for control switching, 'h' for help. (MEDIUM)

---

### 5. custom-gui-library.spec.ts

**Severity: LOW**

**Rigor**: Good. Tests actual GUI widget interactions (sliders, checkboxes, dropdowns), theme switching, folder expand/collapse, memory leak prevention, visual snapshot.

**Issues**:
- **Hard-coded expected values**: FOV preset 35mm = 63 degrees, 85mm = 28 degrees (lines 132, 139). If these values change in the demo page, the test breaks. Should either read expected values from the demo or document why these are the correct values. (LOW)
- **`should have responsive layout`** (line 205): Asserts `box!.width === 350` exactly. This is fragile -- any rounding or DPI difference would fail. (MEDIUM)
- **Uses `/demo-custom-gui.html`**: This is a separate page from the main viewer. Ensure this page is actually served during E2E tests. (LOW)

---

### 6. data-monitor-metrics.spec.ts

**Severity: LOW** (improved from MEDIUM after PR #53 fixes)

**Rigor**: Tests the data monitor API with proper assertions.

**Remaining Issues**:
- **`should not grow infinitely with interactions`** (line 86): Uses `page.mouse.wheel(0, 100)` which may not trigger re-queries for a 3D dataset. The test may not actually exercise the accumulation bug it claims to catch. (MEDIUM)

---

### 7. demo-validation.spec.ts

**Severity: LOW**

**Rigor**: Validates Python syntax and checks for .zarr output existence. Good pre-flight check.

**Issues**:
- **`build_example.py` special case** (line 71): Hard-codes `build_example_manual.zarr` as the output. If `build_example.py` generates multiple outputs (it generates both `_manual` and `_structured`), only one is checked. (LOW)
- **Tolerance for missing datasets** (line 143): `expect(missing.length).toBeLessThan(DEMO_SCRIPTS.length / 2)` allows up to 13 missing datasets before failing. This is very permissive. (MEDIUM)
- **Executes `hatch run python -m py_compile`** in E2E: This spawns a Python process during Playwright testing, which adds ~30s per script. Could be a separate CI step. (LOW)

---

### 8. dimension-animation.spec.ts

**Severity: MEDIUM**

**Rigor**: Comprehensive animation testing (play/pause, FPS, loop modes, bounce, multi-dimension).

**Issues**:
- **Helper functions access `debug?.inputHandler` directly** (lines 33-86) instead of using `debug?.app?.inputHandler` (the canonical path per CLAUDE.md). Also accesses `debug?.sceneDimsManager` instead of `debug?.app?.inputHandler?.sceneDimsManager`. These non-canonical paths may break if the debug interface is refactored. (MEDIUM)
- **`should stop at end with once mode`** (line 552): Uses a 100-iteration polling loop with 100ms intervals (10 seconds total). This is a reasonable pattern but could use `waitForFunction` instead. (LOW)
- **`should increase speed with Shift+Up`** (line 372): Bypasses the keyboard and calls `animManager.increaseSpeed(3)` directly. The test title is misleading -- it tests the API, not the keyboard shortcut. (MEDIUM)
- **Multiple `waitForTimeout(200)` for context menu interactions**: Context menus may not be ready in 200ms on slow CI. (LOW)

---

### 9. dimension-initialization.spec.ts

**Severity: HIGH**

**Rigor**: Tests initialization policy but relies heavily on console log assertions.

**Issues**:
- **`DATASETS.timeAnimated` references `time_animated_example.zarr`** (line 28): This dataset is NOT in the `REQUIRED_DATASETS` list in `global-setup.ts` and NOT in the `ALL_EXAMPLES` list in `all-examples-smoke-test.spec.ts`. If this dataset doesn't exist, the test will fail with an unhelpful error. (HIGH)
- **Console log assertions** (lines 36-38, 76): Tests capture `Query position:` console logs and assert they exist. This couples tests to internal logging format. If the log message changes (e.g., `Query pos:` or removed), the test fails for the wrong reason. (HIGH)
- **`should not cause visual jump on first slider interaction`** (line 136): Dispatches a raw `Event('input')` which may not trigger the same handler chain as a real user interaction. The actual slider input event may be `InputEvent` with `data` property. (MEDIUM)
- **Weak final assertions**: `expect(finalState.initialized).toBe(true)` and `expect(finalState.totalPoints).toBeGreaterThanOrEqual(0)` don't verify the slider interaction worked. (HIGH)

---

### 10. error-recovery.spec.ts

**Severity: MEDIUM** (improved from HIGH after PR #53 fixes)

**Rigor**: Tests error conditions with proper route interception for corruption and network failure scenarios.

**Remaining Issues**:
- **`should handle empty dataset gracefully`** (line 248): Loads `/?debug` (no dataset). Tests that the viewer initializes without a dataset, not that it handles an empty dataset (a dataset with 0 points). (HIGH)
- **WebGL context loss test** (line 189): Loses and restores context via extension, but the GL context returned by `getContext('webgl2')` after loss is likely `null`, making the `restoreContext()` call fail silently. The test should get the extension before loss. (HIGH)
- **Memory limit test** (line 324): Uses `performance.memory` which is Chrome-only and non-standard. Test may not work in Firefox/Safari Playwright runs. Has a guard (`if (!perf.memory) return true`) but this means the test always passes on those browsers. (MEDIUM)

---

### 11. first-time-ux.spec.ts

**Severity: LOW**

**Rigor**: Good coverage of the onboarding flow.

**Issues**:
- **Hardcoded welcome text assertions** (lines 43-46, 60-66): Asserts specific strings like `'Interactive Scientific Data Visualization'`, `'Browse for .zarr'`, `'enter path manually'`. These will break if UX copy changes. (MEDIUM)
- **`should allow dismissing error or browser UI`** (line 127): Loads `/?src=http://localhost:9000/fail.zarr` -- the server name suggests this is a valid host returning 404, not an actual error condition. (LOW)
- **`should close dataset browser with close button`** (line 156): Has fallback that navigates to a completely different page if close button is not found. This masks the actual test failure. (MEDIUM)

---

### 12. geometry-types.spec.ts

**Severity: LOW**

**Rigor**: Good. Tests Lines and GSplats geometry rendering with specific attribute checks.

**Issues**:
- **Uses test fixtures from `FIXTURES_BASE`**: Depends on `generate_test_data.py` having been run. If fixtures are missing, errors will be confusing. (LOW)
- **No pixel-level visual verification**: Tests only check scene graph structure, not that lines/gsplats actually render visible pixels. Could add canvas pixel sampling. (MEDIUM)
- **No test for mixed geometry scenes**: Only tests Lines-only and GSplats-only scenes. A scene with Points + Lines + GSplats would catch rendering order bugs. (MEDIUM)

---

### 13. keyboard-input-system.spec.ts

**Severity: MEDIUM**

**Rigor**: Most thorough keyboard test spec. Good coverage of fly controls, modifier keys, context passthrough.

**Issues**:
- **Control mode switching** (lines 29-31): Presses 'v' twice to cycle orbit -> arcball -> fly. This assumes the initial mode is orbit and the cycle order is fixed. If the cycle order changes, all fly-mode tests break. Should verify the mode after switching. (HIGH)
- **`Cmd+R should not be blocked by R key binding`** (line 519): Presses `Control+r` or `Meta+r` and waits 1 second. Then checks if rendering controls toggled. This is a very indirect way to test browser shortcut preservation. The page may have reloaded, making the subsequent evaluate call fail or return stale state. (HIGH)
- **`WASD keys should be blocked in orbit mode`** (line 479): Good test, but uses `0.01` threshold for "essentially no movement". This could be flaky if there's any animation frame in flight. (LOW)
- **`should move faster with Shift+W`** (line 51): Recenters camera with 'f' key between measurements. The recenter position may not match the initial position exactly, making the distance comparison unfair. (MEDIUM)

---

### 14. nd-navigation.spec.ts

**Severity: MEDIUM**

**Rigor**: Core feature, good coverage of dimension selection, forward/backward navigation, spatial index queries, broadcasting.

**Issues**:
- **`should show dimension sliders for nD datasets`** (line 247): Uses class `.dimension-sliders` (no `luxar-` prefix), while `keyboard-input-system.spec.ts` uses `.luxar-dimension-sliders`. One of these selectors is wrong. (HIGH)
- **`should hide dimension sliders with N key`** (line 277): Also uses `.dimension-sliders` instead of `.luxar-dimension-sliders`. (HIGH)
- **Weak assertions in spatial index tests**: Multiple tests fall through to `expect(state.initialized).toBe(true)` if no query logs are found. This means a broken spatial index would still pass. (HIGH)
- **`should show cache hits on return navigation`** (line 190): Clears `cacheLogs` then navigates back, but never asserts anything about cache logs. The final assertion is just `expect(state.initialized).toBe(true)`. (HIGH)
- **Performance test** (line 300): Includes `waitForTimeout(300)` in the timed section, inflating the measurement by 300ms. (LOW)

---

### 15. nd-transforms.spec.ts

**Severity: LOW**

**Rigor**: Excellent. Tests the inverse-query nD transform feature with precise expected values (50 points at time=0, 50 at time=5, 0 at time=3).

**Issues**:
- **Navigation step count** (line 63): Presses `]` 5 times to reach time=5. This assumes step size of 1.0 for the time dimension. If the step size changes, the test navigates to the wrong position. (MEDIUM)
- **Hard `waitForTimeout(1000)`** after navigation (lines 73, 100): Should use `waitForNavigationComplete` or `waitForDataLoaded`. (LOW)

---

### 16. performance-tracking.spec.ts

**Severity: HIGH**

**Rigor**: Interesting approach with baseline files, but implementation has serious issues.

**Issues**:
- **Race condition in baseline file** (line 42-59): Multiple parallel test workers can read/write `performance-baselines.json` simultaneously, causing data corruption. The `saveBaselines` function does read-modify-write without locking. (CRITICAL)
- **"Always update baseline if faster"** (line 97-99): This creates a ratchet effect where baselines only decrease, making future tests increasingly likely to fail on normal variance. (HIGH)
- **FPS measurement** (line 165-186): Calls `debug.renderOnce()` inside `requestAnimationFrame`. This measures the _possible_ frame rate, not the actual render loop FPS. On a fast GPU with no data, this could report 500+ FPS. (MEDIUM)
- **`should not degrade FPS over time`** (line 291): Waits 5 seconds idle and expects FPS to stay within 20%. But `requestAnimationFrame` FPS can vary due to browser throttling, background tabs, etc. (HIGH)
- **`should not leak memory while idle`** (line 227): Uses `performance.memory` (Chrome-only). Returns 0 on other browsers, making the comparison `growthMB < 50` always pass. (MEDIUM)

---

### 17. position-bounds-clipping.spec.ts

**Severity: LOW**

**Rigor**: Thorough. Tests position_bounds metadata loading, clipping plane calculation, safety margins, and auto-adjustment.

**Issues**:
- **Safety margin calculation** (line 282-366): Duplicates the production algorithm in test code. If the algorithm changes (e.g., safety margin goes from 50% to 30%), the test will fail. Consider reading the actual safety margin constant from the app. (MEDIUM)
- **`should use 50% safety margin`**: Title hard-codes the margin value. (LOW)

---

### 18. python-typescript-integration.spec.ts

**Severity: MEDIUM**

**Rigor**: Basic smoke tests. Loads 2 datasets and checks point counts.

**Issues**:
- **Minimal assertions**: The nD test (line 61) only checks `typeof dimensionsInfo.hasDimensions === 'boolean'`. This passes even if hasDimensions is `false`. Should assert it's `true` for a 5D dataset. (HIGH)
- **Significant overlap** with `real-dataset-loading.spec.ts` and `all-examples-smoke-test.spec.ts`. (MEDIUM)
- **Missing integration checks**: Doesn't verify specific encoding formats (LUT, broadcasting, HDR). `test-fixtures-rendering.spec.ts` does this much better. (MEDIUM)

---

### 19. real-dataset-loading.spec.ts

**Severity: MEDIUM**

**Rigor**: Moderate. Covers multiple datasets with attribute verification.

**Issues**:
- **`should load rendering properties from dataset`** (line 219): Checks for `attrs.size` instead of `attrs.radius`. The attribute name in Luxar is `radius`, not `size`. The test likely always has `hasRadii === false`. (CRITICAL)
- **`should switch datasets without page reload`** (line 275): Uses `page.goto()` which IS a page reload. The test title is misleading. (HIGH)
- **`should preserve scene dimensions from dataset`** (line 174): Only checks `typeof hasDimensions === 'boolean'` -- this always passes. (HIGH)
- **Heavy overlap** with `all-examples-smoke-test.spec.ts` and `python-typescript-integration.spec.ts`. (MEDIUM)

---

### 20. recording-panel.spec.ts

**Severity: LOW**

**Rigor**: Good. Tests panel toggle, structure, screenshot capture, video confirmation dialog, recording indicator.

**Issues**:
- **`should trigger screenshot with G key`** (line 113): Depends on download event firing. May be flaky in headless mode if the browser blocks downloads. (MEDIUM)
- **`should show recording indicator during video recording`** (line 186): Directly calls `panel.startVideoRecording()` bypassing the normal UI flow. Overrides `showConfirmationDialog` with a mock. This tests the recording indicator but not the full user flow. (LOW)
- **Hardcoded filename regex** (line 125): `^luxar-capture-\d{4}-\d{2}-\d{2}-\d{6}\.\w+$` -- if the filename format changes, this breaks. (LOW)

---

### 21. rendering-controls.spec.ts

**Severity: LOW**

**Rigor**: Good API verification. Tests panel toggle, FOV, control types, post-processing, dynamic clipping.

**Issues**:
- **`should apply FOV changes`** (line 89): Calls `sceneManager.updateFOV(5)` (delta of 5 degrees) but doesn't verify the exact new value. Only checks `newFov !== initialFov`. (LOW)
- **Initial panel hidden check** (line 28): Clicks `body` at position (10, 10) to dismiss modals. This may click on a UI element at that position. (LOW)

---

### 22. spatial-index-accuracy.spec.ts

**Severity: HIGH**

**Rigor**: Claims to test spatial index "accuracy" but mostly tests "doesn't crash".

**Issues**:
- **No actual accuracy test**: Despite the name, no test verifies that the correct points are returned for a given query position. Tests just check `state.initialized === true` and `state.totalPoints >= 0`. (CRITICAL)
- **`should merge adjacent ranges for efficiency`** (line 95): Extracts chunk/range counts from console logs and asserts `ranges <= chunks`. This is a valid optimization check, but if no matching logs are found, the test passes with just `state.initialized === true`. (HIGH)
- **`should filter zero-radius points`** (line 167): Only checks `state.initialized === true`. Doesn't verify any filtering occurred. (HIGH)
- **`should handle queries outside data bounds`** (line 306): Navigates forward 20 times with only 200ms between presses. This is rapid fire that may not complete navigation between steps. Also only checks `totalPoints >= 0` which is always true. (HIGH)
- **Heavy overlap** with `nd-navigation.spec.ts` and `cache-system.spec.ts`. (MEDIUM)

---

### 23. test-fixtures-rendering.spec.ts

**Severity: LOW**

**Rigor**: Excellent. The best test file in the suite. Tests specific data values (sharpness range [1,31], HDR max > 5.0, transform world positions, broadcasting uniformity, LUT unique colors). Uses `drawRange` correctly.

**Issues**:
- **Manual screenshots** (lines 100, 179, 252, 297, 367, 435): Uses `page.screenshot({ path: ... })` instead of Playwright's built-in `toHaveScreenshot`. These screenshots are not compared to baselines -- they're just saved. (LOW)
- **Console error detection test** (line 439): Loads a non-existent dataset and checks for errors/warnings. This is testing the helper function, not the fixtures. Slightly misplaced. (LOW)

---

### 24. theme-visual-regression.spec.ts

**Severity: LOW**

**Rigor**: Good systematic coverage of all themes across UI components.

**Issues**:
- **Theme list mismatch** (line 16): `THEMES = ['dark', 'light', 'frosted-glass']` but the bottom tests (line 236) also test `liquid-glass`. The loop-based tests don't cover `liquid-glass`. (MEDIUM)
- **Hard-coded CSS values** (lines 248, 264, 273): Asserts exact CSS variable values like `'#111111'` and `'rgba(255, 255, 255, 0.15)'`. These break on any theme color adjustment. (MEDIUM)
- **`waitForTheme` helper** uses `waitForTimeout(200)` after detecting the attribute change. Should use a condition-based wait for CSS to be applied. (LOW)

---

### 25. transform-hierarchy.spec.ts

**Severity: MEDIUM**

**Rigor**: Tests hierarchy depth, transform composition, matrix correctness, consistency across renders.

**Issues**:
- **Object name matching** (line 57): Uses `p.name.includes('Galaxy')` and `p.name.includes('SolarSystem')`. These depend on example dataset naming which could change. (MEDIUM)
- **`should apply parent transform to child objects`** (line 25): If neither `parent` nor `child` is found (names don't match), the test falls through to `expect(positions.length).toBeGreaterThan(0)` -- a trivially true assertion. (HIGH)
- **`should apply rotation transforms correctly`** (line 102) and **`should apply scale transforms correctly`** (line 131): Both use `if (rotatedObject)` / `if (scaledObject)` guards. If the dataset doesn't have objects named "Rotated" or "Scaled", the test silently passes with no assertions. (HIGH)

---

### 26. viewer-initialization.spec.ts

**Severity: LOW**

**Rigor**: Solid basic initialization tests.

**Issues**:
- **Overlap with `basic-rendering.spec.ts`**: Both test `/?debug` initialization, scene/camera/renderer existence, and renderer frame info. (MEDIUM)
- **`should verify point cloud attributes`** (line 53): Only runs assertions `if (pointClouds.length > 0)`. Without a dataset, this block is skipped entirely. (LOW)

---

### 27. visual-regression.spec.ts

**Severity: MEDIUM**

**Rigor**: Good visual regression approach with reasonable tolerances.

**Issues**:
- **`should render with exposure = 0.0`** (line 57): Checks `debug.sceneManager && typeof debug.sceneManager.updateExposure === 'function'` and returns early with `console.log('Skipping')` if not available. This isn't `test.skip()`, so the test shows as passed. (HIGH)
- **Duplicate screenshot names**: `grid-5d-initial-slice.png` (line 49) and `grid-5d-slice-0.png` (line 123) are screenshots of the same view at the same slice position. (LOW)
- **`should render in fly control mode`** (line 220): Presses 'v' once -- this switches to arcball, not fly. The test name is wrong unless the initial mode cycles through to fly. (HIGH)
- **`maxDiffPixelRatio: 0.08`**: 8% is fairly permissive. For visual regression, 1-3% is more typical. (MEDIUM)

---

### 28. webgl-errors.spec.ts

**Severity: LOW**

**Rigor**: Critical safety net. Tests are well-designed.

**Issues**:
- **Per-dataset test loop** (line 267): Creates a new `page.on('console')` listener in each loop iteration without removing it. By the last dataset, there are 5 listeners, each capturing errors from previous datasets. However, since `webglErrors` is re-declared each iteration, this doesn't cause false positives -- just wasted listeners. (LOW)
- **`should render all example datasets without WebGL errors`** (line 76): Serial loop through datasets in a single test. If the 2nd dataset fails, you don't know about the 3rd-5th. Consider using `test.describe` with parameterized tests. (MEDIUM)
- **Only 5 datasets tested**: The `DATASETS` array has 5 entries, while `all-examples-smoke-test.spec.ts` has 28. Should either use the full list or document why these 5 were chosen. (MEDIUM)

---

### 29. worker-wasm-integration.spec.ts

**Severity: HIGH**

**Rigor**: Claims to test Worker and WASM integration but doesn't actually verify either is used.

**Issues**:
- **No verification of Worker usage**: Every test checks `state?.totalPoints > 0` or `state?.pointClouds?.length > 0`. This passes whether data was loaded via Workers, WASM, or main-thread fallback. (CRITICAL)
- **No verification of WASM usage**: Same issue. The test `should load WASM module successfully` (line 108) just checks that points loaded. (CRITICAL)
- **`should fallback to main thread if worker fails`** (line 56): Doesn't disable workers or inject any failure. It just loads data normally. (CRITICAL)
- **`should achieve faster queries with both enabled`** (line 190): Asserts `totalTime < 10000` (10 seconds). This threshold is so generous it would pass with 10x degradation. (HIGH)
- **`should handle rapid view updates without worker congestion`** (line 74): Presses ArrowRight 5 times rapidly. ArrowRight may not trigger spatial queries in a 3D dataset. (MEDIUM)

---

## Cross-Cutting Issues

### 1. Selector Inconsistency (HIGH)

Two different selectors are used for dimension sliders across specs:
- `.dimension-sliders` (nd-navigation.spec.ts, lines 259, 288)
- `.luxar-dimension-sliders` (keyboard-input-system.spec.ts, theme-visual-regression.spec.ts)

One of these is wrong. The correct class appears to be `.luxar-dimension-sliders` based on the theme tests that successfully interact with it.

### 2. Excessive waitForTimeout Usage (MEDIUM)

Despite the helpers module providing condition-based waits (`waitForNextRender`, `waitForDataLoaded`, `waitForNavigationComplete`, `waitForAnimationStep`), many specs still use raw `waitForTimeout` calls:
- `cache-system.spec.ts`: 6 instances
- `dimension-animation.spec.ts`: 12 instances
- `visual-regression.spec.ts`: 5 instances
- `theme-visual-regression.spec.ts`: 4 instances

These cause tests to be slower than necessary (waiting longer than needed) and flakier (not waiting long enough on slow CI).

### 3. Significant Test Overlap (MEDIUM)

Several groups of specs cover substantially the same ground:

**Initialization tests** (could be consolidated):
- `basic-rendering.spec.ts` -- Three.js init, canvas, screenshot
- `viewer-initialization.spec.ts` -- Three.js init, attributes, renderer
- `rendering-controls.spec.ts` -- Camera FOV, control types

**Dataset loading** (heavy overlap):
- `all-examples-smoke-test.spec.ts` -- All datasets, console errors, WebGL
- `real-dataset-loading.spec.ts` -- 6 datasets, attributes, switching
- `python-typescript-integration.spec.ts` -- 2 datasets, basic decode check
- `webgl-errors.spec.ts` -- 5 datasets, WebGL error checking

**Spatial index / navigation** (heavy overlap):
- `nd-navigation.spec.ts` -- Navigation, queries, sliders, performance
- `spatial-index-accuracy.spec.ts` -- Queries, cache, error handling

### 4. Tests That Can Never Fail (CRITICAL)

Several tests have assertion patterns that always pass:
- `nd-navigation.spec.ts:269` -- Falls through to "feature not implemented, test passes"
- `python-typescript-integration.spec.ts:72` -- `typeof boolean === 'boolean'` always true
- `real-dataset-loading.spec.ts:183` -- Same pattern
- `transform-hierarchy.spec.ts:123-127` -- Guarded assertions that skip on missing objects

### 5. Missing Test Fixture Dataset (HIGH)

`dimension-initialization.spec.ts` references `time_animated_example.zarr` which does not appear in `global-setup.ts:REQUIRED_DATASETS`, the examples list in `all-examples-smoke-test.spec.ts`, or the demo scripts list in `demo-validation.spec.ts`. This test likely fails when run.

### 6. Debug Interface Path Inconsistency (MEDIUM)

The canonical path per the codebase is `debug.app.inputHandler.sceneDimsManager`, but several tests use:
- `debug.inputHandler` (dimension-animation.spec.ts helpers)
- `debug.sceneDimsManager` (dimension-animation.spec.ts, keyboard-input-system.spec.ts)
- `debug.sceneManager` (visual-regression.spec.ts)

The helpers in `helpers.ts` correctly standardize this with fallbacks, but individual specs bypass the helpers and use direct access.

---

## Summary of Remaining Findings by Severity

| Severity | Count | Key Examples |
|----------|-------|-------------|
| CRITICAL | 6 | No-op tests in `worker-wasm-integration`; wrong attribute name in `real-dataset-loading` (`attrs.size` vs `attrs.radius`); race condition in performance baselines; spatial-index tests verify nothing |
| HIGH | 16 | Weak assertions that always pass; missing dataset reference; inconsistent selectors; tests that skip silently on failure; console-log-dependent assertions |
| MEDIUM | 24 | Overlapping coverage; waitForTimeout overuse; hardcoded strings; debug path inconsistency; loose visual regression tolerances |
| LOW | 15 | Minor style issues; redundant screenshots; hard-coded magic values |

---

## Recommendations (Priority Order)

1. **Fix CRITICAL no-op tests**: `worker-wasm-integration` needs to verify Worker/WASM are actually used (check console logs for WASM init, or expose worker status on debug interface).

2. **Fix wrong selector**: `real-dataset-loading.spec.ts:236` checks `attrs.size` but the attribute name is `radius`.

3. **Unify dimension slider selectors**: Determine whether the correct class is `.dimension-sliders` or `.luxar-dimension-sliders` and fix all specs.

4. **Add missing dataset**: Either add `time_animated_example.zarr` to the required datasets list and example scripts, or update `dimension-initialization.spec.ts` to use an existing dataset.

5. **Strengthen assertions in spatial-index-accuracy.spec.ts**: The spec should verify actual query results (expected point counts at known positions), not just `state.initialized === true`.

6. **Replace `waitForTimeout` with condition-based waits**: Systematic pass through all specs replacing arbitrary timeouts with the condition-based helpers already available.

7. **Consider consolidating overlapping specs**: Merge `basic-rendering` + `viewer-initialization`, merge `spatial-index-accuracy` into `nd-navigation`, and reduce the 4 dataset-loading specs to 2 (smoke test + detailed fixture test).

8. **Fix performance baseline race condition**: Use a mutex/lock file or run performance tests serially.

9. **Add real Worker/WASM verification**: Expose `wasmLoaded` and `workerPoolActive` on the debug interface so tests can verify the actual execution path.

---

## Recommended Next Batch

The following 5 issues represent the highest-impact remaining fixes, ordered by value:

### 1. worker-wasm-integration.spec.ts -- Rewrite to actually test Workers and WASM (CRITICAL)
**Impact**: 5 tests (3 CRITICAL, 1 HIGH, 1 MEDIUM) currently give complete false confidence. None verify that Workers or WASM are actually used. This is the single largest gap in the test suite.
**Fix**: Expose `wasmLoaded` and `workerPoolActive` booleans on `window.__luxarDebug`. For the worker-fallback test, use `page.route()` to intercept and break the worker script URL, then verify main-thread fallback activates. For WASM, check for WASM-specific console logs or the debug flag.

### 2. spatial-index-accuracy.spec.ts -- Add real accuracy assertions (CRITICAL)
**Impact**: 6 tests claiming to verify spatial index correctness verify nothing beyond `state.initialized === true`. A completely broken spatial index would pass all tests.
**Fix**: Use a known test fixture (e.g., `dense_grid_5d_example.zarr`) where exact point counts at specific slice positions are deterministic. Assert `state.totalPoints === expectedCount` after navigating to known positions.

### 3. real-dataset-loading.spec.ts -- Fix `attrs.size` to `attrs.radius` (CRITICAL)
**Impact**: The rendering properties test has always had `hasRadii === false` because it checks the wrong attribute name. This means radius data loading has zero E2E test coverage.
**Fix**: Change `attrs.size` to `attrs.radius` (one-line fix, high value).

### 4. nd-navigation.spec.ts -- Fix selector inconsistency and strengthen assertions (HIGH)
**Impact**: 4 HIGH-severity issues. The wrong `.dimension-sliders` selector means slider visibility tests may not be finding the real element. Weak fallback assertions mean spatial query failures are invisible.
**Fix**: Change `.dimension-sliders` to `.luxar-dimension-sliders` in both locations. Replace `expect(state.initialized).toBe(true)` fallbacks with `expect.fail('No spatial query logs found')` to surface real failures.

### 5. performance-tracking.spec.ts -- Fix baseline file race condition (CRITICAL)
**Impact**: Parallel test workers doing read-modify-write on `performance-baselines.json` without locking can corrupt the file, causing cascading flaky failures.
**Fix**: Either run performance tests with `test.describe.serial()`, use a file lock (e.g., `proper-lockfile`), or eliminate the shared baseline file entirely by using inline thresholds.
