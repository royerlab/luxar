/**
 * Test Fixture Rendering E2E Tests
 *
 * These tests verify that our unit test fixtures render correctly in the browser.
 * This provides visual verification of the full pipeline:
 * - Python encoding → ZARR storage → TypeScript decoding → WebGL rendering
 *
 * Tests use fixtures from packages/luxar-viewer/tests/fixtures/
 */

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  assertNoConsoleErrors,
  getConsoleMessages,
  getWebGLErrors,
  raceEvaluate,
} from './helpers';

// HTTP server (configured in playwright.config.ts) serves from project root
// Fixtures are at: packages/luxar-viewer/tests/fixtures/
const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

const FIXTURES = {
  sharpness: `${FIXTURES_BASE}/test_sharpness_range.luxar.zarr`,
  hdr: `${FIXTURES_BASE}/test_hdr_colors.luxar.zarr`,
  hierarchy: `${FIXTURES_BASE}/test_hierarchical_transforms.luxar.zarr`,
  nd4d: `${FIXTURES_BASE}/test_4d.luxar.zarr`,
  broadcasting: `${FIXTURES_BASE}/test_broadcasting.luxar.zarr`,
  lut: `${FIXTURES_BASE}/test_lut.luxar.zarr`,
  lutU16: `${FIXTURES_BASE}/test_lut_u16.luxar.zarr`,
};

// Raise this file above the config's 60 s default (`timeout` in
// playwright.config.ts). A test here can spend `navigationTimeout` 60 s on
// `page.goto` before a helper runs, then `waitForLuxarReady` 45 s,
// `waitForPointsLoaded` up to ~90 s (a 45 s loop budget plus an inner
// `getLuxarState` probe deliberately left unclamped, so one probe holds a
// further 45 s), then 45 s each for `assertNoConsoleErrors` and
// `getLuxarState` — the bounded prefix alone outruns any wall worth setting,
// so this number is a BACKSTOP, not a guarantee. What makes a failure
// attributable is the probes being bounded and throwing by name; at 60 s the
// wall beat even `goto` + `waitForLuxarReady` (105 s), so this arrived as a
// bare `Test timeout of 60000ms exceeded` with nothing to say which dataset
// or which probe was pending (the same class of wall `getLuxarState`
// documents in helpers.ts), and at 120 s those first two phases can report
// themselves. 120 s matches the two in-tree `describe.configure` precedents
// at this value, webgl-errors.spec.ts and all-examples-smoke-test.spec.ts;
// frame-pacing.spec.ts uses the same mechanism at 300 s, and
// points-rendering-perf.spec.ts reaches 120 s through `test.setTimeout`.
// File scope, so both describe blocks (8 tests) carry it. Only the timeout
// changes: this file keeps the config's `fullyParallel: true`.
test.describe.configure({ timeout: 120000 });

test.describe('Test Fixture Rendering', () => {
  test('should render sharpness range fixture correctly', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.sharpness}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for at least 1 point to load

    // CRITICAL: Check for console errors immediately after loading
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify points loaded
    expect(state.totalPoints).toBe(32); // 32 points sampling sharpness [0, 1]
    expect(state.pointClouds.length).toBe(1);

    // Verify sharpness attribute exists and has correct range
    const sharpnessData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      const texData = points?.geometry?.userData?.elementTexture?.image?.data;
      if (!points || !texData || !points.geometry?.userData?.hasSharpness) {
        return null;
      }

      const geometry = points.geometry;
      // Per-point data is texture-backed: the RGBA32F element texture holds
      // 12 floats per point, with sharpness at slot [i*12+7]. Points render
      // as instanced quads: drawRange is the 6-index base quad, while
      // instanceCount is the visible point count. The texel buffer may be
      // over-allocated by the GPU pool.
      const STRIDE = 12;
      const texelCapacity = Math.floor(texData.length / STRIDE);
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : texelCapacity;
      const actualCount = Math.min(instanceCount, texelCapacity);
      const array: number[] = [];
      for (let i = 0; i < actualCount; i++) {
        array.push(texData[i * STRIDE + 7]);
      }
      return {
        count: actualCount,
        min: Math.min(...array),
        max: Math.max(...array),
      };
    });

    expect(sharpnessData).not.toBeNull();
    expect(sharpnessData?.count).toBe(32);

    // Sharpness is a normalized [0, 1] knob (fixture = linspace(0, 1, 32)).
    // Verify the decoded range spans the full [0, 1] knob.
    expect(sharpnessData?.max).toBeGreaterThan(0.95); // reaches ~1.0
    expect(sharpnessData?.min).toBeLessThan(0.05); // reaches ~0.0

    // Check for WebGL errors (CRITICAL)
    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);

    // Log console messages for debugging
    const consoleMessages = await getConsoleMessages(page);
    console.log(`[Sharpness Test] Console logs: ${consoleMessages.logs.length}`);
    console.log(`[Sharpness Test] Console warnings: ${consoleMessages.warnings.length}`);
    console.log(`[Sharpness Test] WebGL errors: ${webglErrors.length}`);

    // Take screenshot for visual verification
    await page.screenshot({ path: 'test-results/sharpness-range-rendering.png' });
  });

  test('should render HDR colors without clamping', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.hdr}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Check for console errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify points loaded
    expect(state.totalPoints).toBe(20); // 20 points with HDR colors

    // Verify HDR color values survive into the float texel storage
    const colorData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      const texData = points?.geometry?.userData?.elementTexture?.image?.data;
      if (!points || !texData || !points.geometry?.userData?.hasColors) {
        return null;
      }

      const geometry = points.geometry;
      // Per-point data is texture-backed: the RGBA32F element texture holds
      // 12 floats per point — centers at [i*12 .. i*12+2], colors at
      // [i*12+4 .. i*12+6]. Points render as instanced quads: drawRange is
      // the 6-index base quad, while instanceCount is the visible point
      // count. The texel buffer may be over-allocated by the GPU pool.
      const STRIDE = 12;
      const texelCapacity = Math.floor(texData.length / STRIDE);
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : texelCapacity;
      const actualCount = Math.min(instanceCount, texelCapacity);
      // The fixture stores points along the X axis with index == x-position,
      // so sort by x to recover the input ordering (the loader/spatial
      // index does not preserve insertion order).
      const positions: { index: number; x: number }[] = [];
      for (let i = 0; i < actualCount; i++) {
        positions.push({ index: i, x: texData[i * STRIDE] });
      }
      positions.sort((a, b) => a.x - b.x);
      const redChannels = positions.map((p) => texData[p.index * STRIDE + 4]);

      return {
        count: actualCount,
        arrayType: texData.constructor.name,
        redMin: Math.min(...redChannels),
        redMax: Math.max(...redChannels),
        redChannels,
      };
    });

    expect(colorData).not.toBeNull();
    expect(colorData?.count).toBe(20);

    // The RGBA32F element texture stores colors as Float32 (no Uint8
    // quantization on the way to the GPU)
    expect(colorData?.arrayType).toBe('Float32Array');

    // CRITICAL: Verify HDR values preserved (max should be ~10.0, not clamped to 1.0)
    expect(colorData?.redMax).toBeGreaterThan(5.0);
    expect(colorData?.redMax).toBeLessThanOrEqual(10.5);

    // Verify monotonic increase (linspace property), reading colors in
    // x-position order rather than buffer order.
    const reds = (colorData?.redChannels as number[]) || [];
    for (let i = 1; i < reds.length; i++) {
      expect(reds[i]).toBeGreaterThanOrEqual(reds[i - 1] - 0.01); // Allow tiny float errors
    }

    // Log and check console
    const hdrConsole = await getConsoleMessages(page);
    console.log(
      `[HDR Test] Logs: ${hdrConsole.logs.length}, Warnings: ${hdrConsole.warnings.length}`
    );

    // Take screenshot
    await page.screenshot({ path: 'test-results/hdr-colors-rendering.png' });
  });

  test('should render hierarchical transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.hierarchy}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Check for console errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify scene loaded
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify transform hierarchy
    const hierarchyData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      // Find parent group - name includes leading slash from zarr path
      const parentGroup = debug.scene.getObjectByName('/parent_group');
      if (!parentGroup) return null;

      // Find child points - full path from zarr
      const childPoints = parentGroup.children.find(
        (obj: any) => obj.name === '/parent_group/child_points'
      );
      if (!childPoints) return null;

      // Local translation comes from the MATRIX, not `.position`. Authored node
      // transforms are installed as a full affine matrix so shear survives
      // (see rendering/node-factory/transforms.ts), which leaves the TRS
      // fields at their defaults — reading `.position` here reported (0,0,0)
      // for a correctly-transformed node. The sibling transform-hierarchy spec
      // was updated for this; this one was missed.
      const localTranslation = (obj: any) => ({
        x: obj.matrix.elements[12],
        y: obj.matrix.elements[13],
        z: obj.matrix.elements[14],
      });

      return {
        parentPosition: localTranslation(parentGroup),
        childLocalPosition: localTranslation(childPoints),
        // World position is read from matrixWorld by getWorldPosition, so it is
        // correct either way; the argument is only a scratch vector.
        childWorldPosition: (() => {
          const worldPos = new (childPoints.position.constructor as any)();
          childPoints.getWorldPosition(worldPos);
          return { x: worldPos.x, y: worldPos.y, z: worldPos.z };
        })(),
      };
    });

    expect(hierarchyData).not.toBeNull();

    // Parent should be at [10, 0, 0]
    expect(hierarchyData?.parentPosition.x).toBeCloseTo(10.0, 1);
    expect(hierarchyData?.parentPosition.y).toBeCloseTo(0.0, 1);
    expect(hierarchyData?.parentPosition.z).toBeCloseTo(0.0, 1);

    // Child local position should be [0, 5, 0]
    expect(hierarchyData?.childLocalPosition.x).toBeCloseTo(0.0, 1);
    expect(hierarchyData?.childLocalPosition.y).toBeCloseTo(5.0, 1);
    expect(hierarchyData?.childLocalPosition.z).toBeCloseTo(0.0, 1);

    // CRITICAL: Child world position should be [10, 5, 0] (transforms composed)
    expect(hierarchyData?.childWorldPosition.x).toBeCloseTo(10.0, 1);
    expect(hierarchyData?.childWorldPosition.y).toBeCloseTo(5.0, 1);
    expect(hierarchyData?.childWorldPosition.z).toBeCloseTo(0.0, 1);

    // Verify no errors during transform application
    const transformConsole = await getConsoleMessages(page);
    expect(transformConsole.errors.length).toBe(0);

    // Take screenshot
    await page.screenshot({ path: 'test-results/hierarchical-transforms-rendering.png' });
  });

  test('should handle 4D nD slicing dataset', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.nd4d}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: No errors in nD data loading
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 4D data loaded
    // Total: 500 points × 10 time steps = 5000
    // Note: With nD slicing, only visible points at current slice are in geometry
    // The total should be 5000 (metadata), but displayed varies by slice
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify dimension information via getState().dimensions
    const dimensionData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const state = debug.getState();

      if (!state || !state.dimensions) return null;

      return {
        numDimensions: state.dimensions.ndim,
        displayedDimensions: state.dimensions.displayed,
        currentSlice: state.dimensions.currentStep,
      };
    });

    // Dimensions may be null if dataset is 3D (not nD)
    // The test fixture is 4D, so we expect dimensions
    if (dimensionData) {
      expect(dimensionData.numDimensions).toBe(4); // time, x, y, z
      expect(dimensionData.displayedDimensions.length).toBe(3); // x, y, z displayed
    }

    // Check console for dimension-related messages
    const ndConsole = await getConsoleMessages(page);
    console.log(`[4D Test] Console output: ${ndConsole.all.length} messages`);

    // Take screenshot
    await page.screenshot({ path: 'test-results/4d-slicing-rendering.png' });
  });

  test('should render broadcasting encoded data', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.broadcasting}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Verify broadcasting decoding has no errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 1000 points loaded (broadcasted from 1 value)
    expect(state.totalPoints).toBe(1000);

    // Verify all points have same color (broadcasted)
    const colorUniformity = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      const texData = points?.geometry?.userData?.elementTexture?.image?.data;
      if (!points || !texData || !points.geometry?.userData?.hasColors) {
        return null;
      }

      const geometry = points.geometry;
      // Per-point data is texture-backed: the RGBA32F element texture holds
      // 12 floats per point, with color rgb at slots [i*12+4 .. i*12+6].
      // Points render as instanced quads: instanceCount is the visible
      // point count; the texel buffer may be over-allocated by the GPU pool.
      const STRIDE = 12;
      const texelCapacity = Math.floor(texData.length / STRIDE);
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : texelCapacity;
      const actualCount = Math.min(instanceCount, texelCapacity);
      const firstColor = [texData[4], texData[5], texData[6]];

      // Check if all colors match the first color
      let allSame = true;
      for (let i = 0; i < actualCount; i++) {
        if (
          Math.abs(texData[i * STRIDE + 4] - firstColor[0]) > 0.01 ||
          Math.abs(texData[i * STRIDE + 5] - firstColor[1]) > 0.01 ||
          Math.abs(texData[i * STRIDE + 6] - firstColor[2]) > 0.01
        ) {
          allSame = false;
          break;
        }
      }

      return { allSame, firstColor };
    });

    expect(colorUniformity).not.toBeNull();
    expect(colorUniformity?.allSame).toBe(true);

    // Optionally verify broadcast decoding in console logs (not required for test to pass)
    const broadcastConsole = await getConsoleMessages(page);
    const broadcastLog = broadcastConsole.all.find((msg) => msg.includes('Broadcasting'));
    if (broadcastLog) {
      console.log('✅ Broadcasting log found:', broadcastLog.slice(0, 100));
    } else {
      console.log('ℹ️ Broadcasting log not found (may have been optimized away)');
    }

    // Take screenshot
    await page.screenshot({ path: 'test-results/broadcasting-rendering.png' });
  });

  test('should render lut_uint16 encoded data (>256 unique colors)', async ({ page }) => {
    // The uint16 LUT tier end-to-end: Uint16Array indices through the
    // range loader -> worker -> WASM row kernel (only exercised in-browser).
    const testStartedAt = Date.now();
    await page.goto(`/?src=${FIXTURES.lutU16}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1);

    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBe(100_000);

    // `page.evaluate` carries no deadline of its own — only the test wall
    // stops it, which is precisely the unattributable failure the wall above
    // is NOT meant to be. What starves one is main-thread task starvation, so
    // no probe is exempt (see `raceEvaluate`); what exposes THIS test and not
    // its five siblings is the SCENE — it renders 100k points where they
    // render 20-1000, so only this page saturates the render loop. Bound it so
    // a starved page names this probe instead of pointing at a line number,
    // accepting the trade `changelog.d/1651.md` states: any bound can also cut
    // short a stall that would have ended, so this is not a free win.
    //
    // The budget is what is LEFT of this test's wall rather than a fixed 45 s.
    // A hard 45 s under the 120 s wall is only reachable when the bounded
    // prefix (`goto` plus the four helpers above) finishes inside 75 s; a
    // prefix between 75 s and 120 s lets the wall fire MID-scan, which is the
    // bare unattributable timeout this bound exists to remove. 10 s stays in
    // reserve for the throw, the assertions and teardown, and the 5 s floor
    // keeps a nearly-exhausted test throwing by name rather than not at all.
    const scanTimeoutMs = Math.max(
      5000,
      Math.min(45000, test.info().timeout - (Date.now() - testStartedAt) - 10000)
    );
    // Sentinel, following the argument in `getLuxarState`: it must be a value
    // the in-page function can never return, compared by identity, because
    // `null` is a LEGITIMATE answer here (no points node, no element texture,
    // or no colors on it) and would otherwise be indistinguishable from a
    // missed deadline. Collision-safety is identical to that helper's `{}` —
    // `page.evaluate` resolves with a value deserialized from the CDP
    // protocol, which can carry neither a Node-side object nor a symbol. A
    // `unique symbol` is chosen for a TYPE-level reason instead: it is a unit
    // type, so `colorStats === SCAN_TIMED_OUT` NARROWS the union and leaves
    // the object-or-null the assertions below read, where a `const timedOut =
    // {}` would type as `{}` and collapse that union.
    const SCAN_TIMED_OUT: unique symbol = Symbol('lutU16ColorScanTimedOut');

    const scanProbe = page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });
      const texData = points?.geometry?.userData?.elementTexture?.image?.data;
      if (!points || !texData || !points.geometry?.userData?.hasColors) return null;
      const geometry = points.geometry;
      // Colors live at texel slots [i*12+4 .. i*12+6] of the RGBA32F
      // element texture; instanceCount is the visible point count.
      const STRIDE = 12;
      const texelCapacity = Math.floor(texData.length / STRIDE);
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : texelCapacity;
      const actualCount = Math.min(instanceCount, texelCapacity);
      const uniqueColors = new Set<string>();
      let maxChannel = 0;
      for (let i = 0; i < actualCount; i++) {
        const r = texData[i * STRIDE + 4];
        const g = texData[i * STRIDE + 5];
        const b = texData[i * STRIDE + 6];
        maxChannel = Math.max(maxChannel, r, g, b);
        uniqueColors.add(`${r.toFixed(3)},${g.toFixed(3)},${b.toFixed(3)}`);
      }
      return { totalPoints: actualCount, uniqueColors: uniqueColors.size, maxChannel };
    });

    const colorStats = await raceEvaluate<Awaited<typeof scanProbe> | typeof SCAN_TIMED_OUT>(
      scanProbe,
      scanTimeoutMs,
      SCAN_TIMED_OUT
    );

    if (colorStats === SCAN_TIMED_OUT) {
      // Never a silent pass: the assertions below are the whole point of this
      // test, so an unanswered probe means the u16 LUT round-trip was not
      // checked — which is not the same as checking it and finding it sound.
      throw new Error(
        'lut_uint16 color scan: the page never answered the 100k-texel element-texture probe ' +
          `within ${scanTimeoutMs} ms — its main thread is saturated and starving the evaluate ` +
          'round trip, so the LUT round-trip assertions could not run. See issues #1651 and #1746.'
      );
    }

    expect(colorStats).not.toBeNull();
    expect(colorStats?.totalPoints).toBe(100_000);
    // 300 exact palette colors must survive the u16 LUT round-trip — well
    // beyond the u8 tier's 256 ceiling (the load-bearing assertion).
    expect(colorStats?.uniqueColors).toBe(300);
    // HDR values decode unclamped (palette peaks at 9.5).
    expect(colorStats?.maxChannel).toBeGreaterThan(5.0);
  });

  test('should render LUT encoded data', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.lut}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Verify LUT decoding has no errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 1000 points loaded
    expect(state.totalPoints).toBe(1000);

    // Verify LUT decoding: should have exactly 10 unique colors
    const colorStats = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      const texData = points?.geometry?.userData?.elementTexture?.image?.data;
      if (!points || !texData || !points.geometry?.userData?.hasColors) {
        return null;
      }

      const geometry = points.geometry;
      // Per-point data is texture-backed: decoded LUT colors live at texel
      // slots [i*12+4 .. i*12+6] of the RGBA32F element texture. Points
      // render as instanced quads: instanceCount is the visible point
      // count; the texel buffer may be over-allocated by the GPU pool.
      const STRIDE = 12;
      const texelCapacity = Math.floor(texData.length / STRIDE);
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : texelCapacity;
      const actualCount = Math.min(instanceCount, texelCapacity);
      const uniqueColors = new Set<string>();

      for (let i = 0; i < actualCount; i++) {
        const r = texData[i * STRIDE + 4].toFixed(2);
        const g = texData[i * STRIDE + 5].toFixed(2);
        const b = texData[i * STRIDE + 6].toFixed(2);
        uniqueColors.add(`${r},${g},${b}`);
      }

      return {
        totalPoints: actualCount,
        uniqueColors: uniqueColors.size,
      };
    });

    expect(colorStats).not.toBeNull();
    expect(colorStats?.totalPoints).toBe(1000);

    // CRITICAL: LUT encoding should preserve unique colors
    // Note: Due to float precision in color key generation, actual count may vary
    // The important thing is that LUT decoding worked and we have multiple distinct colors
    expect(colorStats?.uniqueColors).toBeGreaterThanOrEqual(7);

    // Verify LUT decoding logged in console
    const lutConsole = await getConsoleMessages(page);
    const lutLog = lutConsole.all.find((msg) => msg.includes('LUT'));
    expect(lutLog).toBeDefined(); // Should log LUT decoding operation

    // Take screenshot
    await page.screenshot({ path: 'test-results/lut-encoding-rendering.png' });
  });
});

test.describe('Console Error Detection', () => {
  test('should detect and report console errors', async ({ page }, testInfo) => {
    // This test's whole premise is a dataset that does not exist, so the load
    // failure it triggers is the subject under test, not a regression. Since
    // #2494 the viewer honestly attempts that load instead of quietly opening
    // the dataset browser, so four app-level errors now reach the fixture's
    // console gate ("[Luxar] Failed to load scene: NotFoundError…" and its
    // SceneManager / App / start-up echoes) and fail the teardown even though
    // every assertion below passes (#2548).
    //
    // Opt out per-spec rather than adding the pattern to
    // DEFAULT_ALLOWED_CONSOLE_ERRORS: "Failed to load scene" is the single most
    // important error the viewer can log, and forgiving it globally would blind
    // the ~63 specs that share this fixture to a broken scene load.
    testInfo.annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description:
        'Deliberately loads a non-existent dataset; the resulting scene-load errors are what this test asserts on.',
    });

    // Collect page-level errors as a fallback in case the console interceptor
    // isn't ready before the errors fire.
    //
    // Warnings go in their own bucket, and the verdict below counts errors from
    // both sides only. Both the listener and the assertion used to fold warnings
    // in, which meant one unrelated warning could satisfy a test named "should
    // detect and report console errors" on its own — harmless while the fixture's
    // console gate was also watching, but this test now opts out of that gate, so
    // its own assertion is the only thing left standing. Warning counts are still
    // reported below; they just cannot carry the verdict.
    const pageErrors: string[] = [];
    const pageWarnings: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        pageErrors.push(msg.text());
      } else if (msg.type() === 'warning') {
        pageWarnings.push(msg.text());
      }
    });

    // Test that our error detection actually works by loading a non-existent dataset
    await page.goto('/?src=http://localhost:9000/nonexistent.zarr&debug');

    // Return as soon as the Playwright-side capture sees the first error.
    // This path aborts before the in-page console interceptor installs.
    await expect.poll(() => pageErrors.length, { timeout: 8000 }).toBeGreaterThan(0);

    // Check both the in-app console interceptor and the Playwright-captured
    // errors. Errors only on both sides — see the listener note above.
    const consoleMessages = await getConsoleMessages(page);
    const totalErrorCount = consoleMessages.errors.length + pageErrors.length;

    console.log('[Error Detection Test] Intercepted errors:', consoleMessages.errors.length);
    console.log('[Error Detection Test] Intercepted warnings:', consoleMessages.warnings.length);
    console.log('[Error Detection Test] Page-level errors:', pageErrors.length);
    console.log('[Error Detection Test] Page-level warnings:', pageWarnings.length);

    // Verify we captured error messages from either source
    expect(totalErrorCount).toBeGreaterThan(0);
  });
});
