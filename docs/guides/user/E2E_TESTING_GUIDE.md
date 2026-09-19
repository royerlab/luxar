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

  await page.goto('/?src=/data.luxar.zarr&debug');

  // Analyze logs
  const errors = logs.filter(log => log.includes('Error'));
  expect(errors).toEqual([]);
});
```

#### 2. Console Interceptor History
```typescript
// The current API exposes a buffered getter. Call
// `getBufferedMessages()` to obtain the chronological list of
// intercepted log/warn/error/info messages.
const messages = await page.evaluate(() => {
  return window.__luxarDebug?.consoleInterceptor?.getBufferedMessages?.() ?? [];
});
```

#### 3. Agent Driver Tool (Manual Debugging)
```bash
pnpm agent:debug              # Headless - shows console in terminal
pnpm agent:debug:visible      # Visible browser
```

---

## 🔗 Data Source URL Format

Use the viewer's `src` parameter for every dataset URL and add `debug` when tests
need `window.__luxarDebug`:

```text
✅ http://localhost:5173/?src=http://127.0.0.1:8000/datasets/test.luxar.zarr&debug
❌ http://localhost:5173/?data=http://127.0.0.1:8000/datasets/test.luxar.zarr&debug
⚠️ http://localhost:5173/?src=http://127.0.0.1:8000/datasets/test.luxar.zarr/&debug (accepted, non-canonical)
```

Best practices:

- Use `?src=<dataset>&debug`, not `?data=`.
- Prefer the no-trailing-slash spelling for data-source URLs. Both forms are
  accepted and normalized, but one canonical form keeps test URLs and logs
  consistent.
- Prefer explicit loopback hosts and ports in E2E tests (`127.0.0.1:<port>`) so
  tests do not depend on external DNS or network access.

---

## 🧪 Test Suite Organization

### Test Files

The authoritative list is the directory itself — run
`ls packages/luxar-viewer/src/tests/e2e/*.spec.ts` for the current set
(50+ specs). Key test suites include:

| File | Purpose |
|------|---------|
| **all-examples-smoke-test.spec.ts** | Comprehensive smoke tests for all examples |
| **basic-rendering.spec.ts** | Initialization, canvas, errors |
| **blending-modes.spec.ts** | Blending-mode rendering |
| **cache-hardening.spec.ts** | Cache health/demand/network/prefetch diagnostics + cache-monitor UI fields |
| **cache-persistence.spec.ts** | Cache persistence across reloads |
| **cache-system.spec.ts** | Three-level cache system (L0/L1/L2) incl. real browser OPFS behavior |
| **cinematic-auto-framing.spec.ts** | Cinematic FOV is resolved before first-load auto-framing |
| **colormap-system.spec.ts** | Colormap application and switching |
| **context-restore.spec.ts** | Post-processing pipeline survives WebGL context loss/restore |
| **controls-interaction.spec.ts** | Keyboard, mouse, camera |
| **custom-gui-library.spec.ts** | Custom GUI panel testing |
| **data-integrity.spec.ts** | Internal consistency of loaded geometry (aligned attribute arrays, no NaN/Inf; source-comparison lives in python-typescript-integration.spec.ts) |
| **data-monitor-metrics.spec.ts** | Data-metric accuracy (point counts) + monitor UI toggle |
| **dataset-switching.spec.ts** | Switching datasets at runtime |
| **demo-validation.spec.ts** | Python script syntax + `.luxar.zarr` output checks |
| **dimension-animation.spec.ts** | Dimension animation playback |
| **dimension-initialization.spec.ts** | nD dimension setup |
| **error-recovery.spec.ts** | ⭐ Error handling |
| **first-time-ux.spec.ts** | Browser, help, error messages |
| **geometry-types.spec.ts** | Lines & GSplats rendering |
| **glass-refraction-partition.spec.ts** | `refract_data` glass refracts the data behind it while data in front stays crisp (the depth partition), WebGL and WebGPU arms, on the lens example |
| **gsplat-rendering-visual.spec.ts** | GSplat visual rendering |
| **hover-overlay.spec.ts** | Hover-overlay DOM rendering + CSS transitions (direct OverlayManager probe) |
| **hover-tooltip.spec.ts** | Full hover pipeline: mouse → GPU pick → label → tooltip DOM |
| **keyboard-input-system.spec.ts** | Fly controls, shortcuts, modifiers |
| **layers-panel.spec.ts** | Layers panel UI |
| **line-join-artifact.spec.ts** | ⭐ #780/#785/#790 joint-artifact acceptance measurement — five joint cases as separate world-Y bands, scored on one frame with local-median outliers AND axial flux ripple (needs generate-fixtures) |
| **line-perf-bench.spec.ts** | Line-rendering frame-time benchmark, WebGL vs WebGPU (opt-in: `pnpm test:perf:e2e`) |
| **line-renderer-compare-perf.spec.ts** | WebGL vs WebGPU FPS comparison on line-heavy scenes (developer diagnostic, JSON artifact) |
| **line-rendering-visual.spec.ts** | Lines visual correctness via sampled pixels (cap intensity, near-camera artefacts) |
| **lines-nd-dimension-visibility.spec.ts** | Lines re-culled when scrubbing a non-displayed dimension (regression pin) |
| **lod-group.spec.ts** | LOD group level switching (auto + manual lock via layers panel) |
| **luxar-serve-integration.spec.ts** | `luxar serve` integration |
| **mouse-interactions.spec.ts** | Scroll-wheel/modifier behavior: Ctrl+scroll FOV, scroll zoom, Shift+scroll rotate |
| **nd-navigation.spec.ts** | ⭐ Core nD feature! |
| **nd-transforms.spec.ts** | nD transform inverse-query |
| **ortho-mode.spec.ts** | Orthographic camera mode |
| **performance-tracking-perf-bench.spec.ts** | ⭐ Regression detection + memory/FPS (opt-in: `pnpm test:perf:e2e`) |
| **points-rendering-perf.spec.ts** | Points FPS diagnostic — multi-sample min/median/max (developer diagnostic, JSON artifact) |
| **position-bounds-clipping.spec.ts** | Boundary testing |
| **post-processing-pipeline.spec.ts** | Bloom, cinematic mode (vignette/chromatic distortion), exposure, WebGL stability |
| **python-typescript-integration.spec.ts** | ⭐ Cross-language E2E |
| **real-dataset-loading.spec.ts** | Real `.luxar.zarr` files + dataset switching |
| **recording-panel.spec.ts** | Screenshot/video capture panel |
| **renderer-url-param.spec.ts** | `?renderer=` backend selection (WebGL vs WebGPU) |
| **rendering-controls.spec.ts** | Rendering panel, FOV, controls |
| **shader-material-compile.spec.ts** | Browser-real shader compile + non-black pixel-output smoke per material variant |
| **slice-cache.spec.ts** | SliceCache (per-slice decoded-geometry reuse) |
| **spatial-index-accuracy.spec.ts** | Query accuracy, caching |
| **standalone-gsplats.spec.ts** | Standalone `.gsplats.zarr` loading |
| **test-fixtures-rendering.spec.ts** | Encoding/decoding compatibility |
| **theme-visual-regression.spec.ts** | Theme visual regression |
| **transform-hierarchy.spec.ts** | ⭐ Transform correctness |
| **tsl-codegen-snapshot.spec.ts** | TSL-generated GLSL/WGSL pinned to checked-in snapshots (bloat regression alarm) |
| **tsl-shader-parity.spec.ts** | GLSL vs TSL shader parity |
| **url-parameters.spec.ts** | URL parameters control initial state (`?theme=`, `?noCache`, `?debug`, invalid params) |
| **viewer-initialization.spec.ts** | Viewer startup without data |
| **visual-regression.spec.ts** | Screenshot comparison |
| **webgl-errors.spec.ts** | WebGL error detection |
| **webgpu-native-smoke.spec.ts** | WebGPU smoke; native-only tests auto-skip without a real WebGPU adapter |
| **worker-wasm-integration.spec.ts** | Worker + WASM integration |
| **y-orientation.spec.ts** | `renderToImageData()` Y-orientation parity across WebGL/WebGPU backends |
| **zipped-store-loading.spec.ts** | STORED/DEFLATE archive parity with a directory scene (needs the range-capable data server), plus a Range-ignoring host that must show a persistent actionable failure |

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

# Update Linux visual regression baselines
pnpm test:e2e:visual:update

# View HTML report
pnpm test:e2e:report

# AI debugging mode
pnpm agent:debug
```

---

## 📋 Pre-Test Checklist

Before running E2E tests:

1. ✅ Generate example datasets: `make run-examples` — the Playwright pre-flight warns and
   continues when the stamped examples are stale. Specs using the shared page fixture repeat
   the rebuild guidance beside a failure; example-reading specs still run against the existing
   stores, so refresh them before trusting those results.
2. ✅ Generate the zarr test fixtures: `pnpm test:generate-fixtures` — 18 specs read
   `packages/luxar-viewer/tests/fixtures/`, and the Playwright pre-flight now fails the
   whole run if any of them is missing or stale rather than letting those specs time out.
   `make test-e2e` and `pnpm test` (vitest) regenerate them for you; direct `pnpm test:e2e`
   deliberately does not, because the generator takes 1–2 minutes. For a one-off run of
   specs you know read no fixtures, set
   `LUXAR_E2E_NO_FIXTURES=1` to skip the check (`pnpm test:e2e:smoke` and
   `pnpm test:perf:e2e` already do).
3. ✅ Install Playwright browsers: `pnpm exec playwright install chromium`
4. ✅ Ensure ports 5173 and 9000 are free or already serve this checkout (foreign checkout servers are rejected)
5. ✅ Run `pnpm` commands from `packages/luxar-viewer/` (from the project root, use `make test-e2e` instead — there is no root `package.json`)

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
// Import from `./fixtures` (NOT @playwright/test directly) so the
// shared console-error / pageerror checks run automatically after
// each test. Specs that need to allow specific console messages
// can annotate at the test level.
import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState, waitForSpatialQuery } from './helpers';

// Use `?src=` URL params (NOT `/data/...`) — that's the production
// convention. Datasets live under `/datasets/examples/...` on the
// dev server and are referenced via the src query parameter.
const DATASET = 'http://localhost:9000/datasets/examples/my_dataset.luxar.zarr';

test.describe('My Feature Tests', () => {
  test('should do something', async ({ page }) => {
    // 1. Load viewer with dataset
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // 2. Get initial state
    const initialState = await getLuxarState(page);

    // 3. Perform action
    await page.keyboard.press('1'); // Select the first non-displayed dimension
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

### Issue: Server identity check or port startup fails

Playwright reuses ports 5173 and 9000 only when their checkout-specific marker
matches the tree under test. Inspect a busy port before stopping it; another
active worktree may own the process:

```bash
lsof -nP -iTCP:5173 -sTCP:LISTEN
lsof -nP -iTCP:9000 -sTCP:LISTEN
```

Coordinate with the owner, stop only the stale process, and rerun the test. Do
not blindly kill every listener on a shared development machine.

### Issue: Tests timeout waiting for initialization
**Solution**: Check that:
- Vite dev server started (port 5173)
- Python HTTP server started (port 9000)
- Datasets exist in `datasets/examples/`

```bash
make run-examples  # Generate datasets
```

### Issue: Visual regression tests fail
**Solution**: GPU rendering varies. Update baselines:
```bash
pnpm test:e2e:visual:update
```

### Issue: "Executable doesn't exist" error
**Solution**: Install Playwright browsers:
```bash
pnpm exec playwright install chromium
```

### Issue: Many unrelated tests time out at once on a busy machine
**Cause**: Too many Playwright workers for the box's spare capacity — the runs
starve each other and every action hits its wall-clock timeout with the element
already visible/enabled/stable. Measured on a shared 16-core workstation at a
1-minute load of 12–24: `dimension-animation.spec.ts` failed 15 of 21 tests at 4
workers and passed 21 of 21 at `--workers=1`, with no product cause.

**Solution**: Nothing, usually — `playwright.config.ts` now scales the local
worker ceiling of 4 by the box's free fraction
(`clamp(round(4 * (cpus - load1) / cpus), 1, 4)`). The bands are fractions of the
box, so they hold at any size: 4 while at least 7/8 of it is free, 3 down to 5/8,
2 down to 3/8, 1 below that — on 16 cores, 4 up to load 2, 3 up to load 6, 2 up
to load 10, then 1. An idle box of any size still gets 4. The run prints its
worker ceiling before the pre-flight checks:

```text
[🧵] [E2E] parallelism: max 1 worker — 16 cpus, load1 22.2 (capacity)
```

`max` is literal: that is `config.workers`, and Playwright narrows it to
`min(workers, maxConcurrentTestGroups)` afterwards, so a single-file run can be
stamped `max 3` and then report "using 1 worker". When the ceiling is not the
count that was sized, the line carries both — `(capacity sized 3, run with 1)` —
and does not guess why.

Be aware of what that protection is worth: the 1-minute load average lags and is
sampled once at startup, so a run launched just after a burst can still crawl on
a machine that is already idle, and one launched into a lull can take all four
workers into the next burst. Only 4 workers (15/21 failing) and 1 worker (21/21
passing) were actually measured, at load 12–24 on 16 cores — that bottom band is
the configuration measured green; the intermediate counts are interpolation.

Pin it explicitly with `LUXAR_E2E_WORKERS=N` (an integer, clamped to `[1, cpus]`
as a typo guard), or with `--workers=N` (which overrides the config), when you
want a fixed run:

```bash
LUXAR_E2E_WORKERS=1 pnpm test:e2e
npx playwright test dimension-animation.spec.ts --workers=1
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
2. Subsequent runs: Fails if more than 2.0× slower than baseline
   (`REGRESSION_THRESHOLD` in `performance-tracking-perf-bench.spec.ts`;
   the FPS check fails below 70% of baseline)
3. Every passing run unconditionally rewrites the baseline, so it tracks
   the current machine's hardware

**CI Integration**: PR CI runs the Chromium mobile/touch suite for TypeScript
changes. The full rendering-heavy desktop corpus remains on the GPU promotion
runner because hosted software WebGL is too slow and unreliable for it. Run the
full suite locally before PR/merge. The perf-bench specs
are opt-in even locally (`pnpm test:perf:e2e`), and
`performance-baselines.json` is per-machine and gitignored — it is never
committed or uploaded anywhere.

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
Spatial-index telemetry lives on each geometry loader's metrics snapshot
(`metrics.spatialIndex`, a `SpatialIndexMetrics` from
`src/types/data-monitor-types.ts`) — not on the SceneLoader itself:
```typescript
const spatialIndex = await page.evaluate(() => {
  const manager = (window as any).__luxarDebug.getSceneLoader(); // SceneLoaderManager
  const sceneLoader = manager?.getDefaultLoader();
  // Per-geometry loaders live in the registry maps (points shown here;
  // lines/gsplats: `linesLoaders` / `gsplatLoaders`).
  const pointsLoader = sceneLoader ? [...sceneLoader.loaders.values()][0] : undefined;
  const metrics = pointsLoader?.getMetrics?.();

  // Fields: occupiedCells, totalCells, avgCellsPerQuery, avgElementsPerCell,
  // queryEfficiency.
  // Note: `occupiedCells` is a number (chunk count), not an array.
  return metrics?.spatialIndex ?? null;
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

The Luxar E2E test suite is:
- **Comprehensive** — covers all major features
- **Reliable** — condition-based waits, not arbitrary timeouts
- **Fast** — settle-based waits instead of fixed sleeps keep runs short
- **Robust** — error recovery prevents crashes
- **Regression-Proof** — performance tracking catches slowdowns
- **AI-Ready** — full console access for autonomous debugging
