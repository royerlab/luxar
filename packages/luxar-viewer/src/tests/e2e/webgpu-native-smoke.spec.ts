/**
 * Best-effort native-WebGPU smoke spec.
 *
 * Headless Playwright chromium typically falls back to WebGPURenderer's
 * internal WebGL2 backend, so the bulk of CI exercises the TSL graphs
 * *through* WebGL2 — useful but doesn't catch WGSL compilation issues,
 * real WebGPU adapter limits, or the 256-byte row-padding deinterlace.
 * This spec auto-skips when no real WebGPU adapter is present, and on
 * a real-WebGPU runtime it exercises the WebGPU-specific paths.
 *
 * The suite is deliberately fast and read-only: a screenshot at an
 * unaligned width (forces `compactWebGPUReadbackRows` to engage) and
 * an HDR capture round-trip in each mode. Heavier validation lives in
 * the TSL parity harness (`tsl-shader-parity.spec.ts`).
 *
 * @see picking/PICKING_DESIGN.md — WebGPU implementation details
 * @see rendering/SPECIFICATIONS.md § 11 — Dual-stack architecture
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.zarr';

const SKIP_REASON =
  'native-WebGPU smoke spec requires a real WebGPU adapter; ' +
  'Playwright chromium falls back to WebGL2 in headless mode. ' +
  'Run manually with a WebGPU-enabled Chrome stable for full coverage.';

test.describe('WebGPU native smoke (best-effort, skips on fallback)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?renderer=webgpu&src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
  });

  test('reports caps.api === "webgpu" under real WebGPU; skips otherwise', async ({ page }) => {
    const api = await page.evaluate(
      () =>
         
        (window as any).__luxarDebug.app.sceneManager.capabilities.api
    );
    test.skip(api !== 'webgpu', SKIP_REASON);
    expect(api).toBe('webgpu');
  });

  test('screenshot at an unaligned canvas width returns a non-empty, compact buffer', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () =>
         
        (window as any).__luxarDebug.app.sceneManager.capabilities.api
    );
    test.skip(api !== 'webgpu', SKIP_REASON);

    // 853 is deliberately chosen: 853 × 4 (RGBA8) = 3412 bytes, which
    // pads to 3584 = 14 × 256 under WebGPU's `bytesPerRow` rule. If
    // `compactWebGPUReadbackRows` is wired correctly the resulting
    // ImageData wraps a compact, slant-free buffer.
    await page.setViewportSize({ width: 853, height: 480 });
    await page.waitForFunction(
       
      () => (window as any).__luxarDebug?.renderer?.domElement?.width === 853,
      null,
      { timeout: 5000 }
    );

    const { width, height, length } = await page.evaluate(async () => {
       
      const dbg = (window as any).__luxarDebug;
      const img: ImageData = await dbg.postProcessing.renderToImageData();
      return { width: img.width, height: img.height, length: img.data.length };
    });
    expect(width).toBe(853);
    expect(length).toBe(width * height * 4);
  });

  test('captureHDRPixels round-trips for visible-ldr and hdr-effects-pre-tone modes', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () =>
         
        (window as any).__luxarDebug.app.sceneManager.capabilities.api
    );
    test.skip(api !== 'webgpu', SKIP_REASON);

    for (const mode of ['visible-ldr', 'hdr-effects-pre-tone'] as const) {
      const { width, height, length } = await page.evaluate(async (m) => {
         
        const dbg = (window as any).__luxarDebug;
        const out = await dbg.postProcessing.captureHDRPixels(m);
        return { width: out.width, height: out.height, length: out.pixels.length };
      }, mode);
      expect(width).toBeGreaterThan(0);
      expect(height).toBeGreaterThan(0);
      expect(length).toBe(width * height * 4);
    }
  });
});
