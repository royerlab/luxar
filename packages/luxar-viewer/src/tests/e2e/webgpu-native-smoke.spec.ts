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
 * @see rendering/README.md — Dual-stack architecture overview
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

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

  test('reports caps.apiSurface === "webgpu" under real WebGPU; skips otherwise', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
    );
    test.skip(api !== 'webgpu', SKIP_REASON);
    expect(api).toBe('webgpu');
  });

  test('caps.framebufferYDown is true under real WebGPU (top-down framebuffer)', async ({
    page,
  }) => {
    const probe = await page.evaluate(() => {
      const dbg = (window as any).__luxarDebug;
      return {
        api: dbg.app.sceneManager.capabilities.apiSurface,
        framebufferYDown: dbg.app.sceneManager.capabilities.framebufferYDown,
        isWebGLBackend: dbg.renderer?.backend?.isWebGLBackend === true,
      };
    });
    test.skip(probe.api !== 'webgpu', SKIP_REASON);
    // Skip if the underlying backend is the WebGL2 fallback — that
    // path is exercised by the separate `webgpu-force-webgl` spec
    // path; here we want a real native WebGPU adapter.
    test.skip(probe.isWebGLBackend, SKIP_REASON);
    expect(probe.framebufferYDown).toBe(true);
  });

  test('screenshot at an unaligned canvas width returns a non-empty, compact buffer', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
    );
    test.skip(api !== 'webgpu', SKIP_REASON);

    // 853 is deliberately chosen: 853 × 4 (RGBA8) = 3412 bytes, which
    // pads to 3584 = 14 × 256 under WebGPU's `bytesPerRow` rule. If
    // `compactWebGPUReadbackRows` is wired correctly the resulting
    // ImageData wraps a compact, slant-free buffer.
    //
    // We assert against the canvas *backing-store* width (`domElement.width`)
    // because AdaptiveDPRManager can reduce DPR below 1 when FPS dips,
    // scaling the backing store away from the 853 CSS-pixel target.
    // Pinning to `width === 853` would race the adaptive controller; the
    // actual buffer round-trip is what we want to verify.
    await page.setViewportSize({ width: 853, height: 480 });
    // Wait for the CSS-pixel viewport (which DPR cannot change) to settle.
    await page.waitForFunction(
      () => (window as any).__luxarDebug?.renderer?.domElement?.clientWidth === 853,
      null,
      { timeout: 5000 }
    );

    const probe = await page.evaluate(async () => {
      const dbg = (window as any).__luxarDebug;
      const canvas = dbg.renderer.domElement;
      const img: ImageData = await dbg.postProcessing.renderToImageData();
      return {
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        canvasClientWidth: canvas.clientWidth,
        imgWidth: img.width,
        imgHeight: img.height,
        length: img.data.length,
      };
    });
    expect(probe.canvasClientWidth).toBe(853);
    // ImageData should match the actual backing-store dimensions; the
    // capture path is what `compactWebGPUReadbackRows` operates on.
    // Allow ±1 pixel: the renderer's setSize() and adaptive-DPR
    // recompute paths each apply their own `Math.round(...)` to
    // `cssSize × pixelRatio`, so the readback target can land a single
    // pixel off the canvas backing-store width when DPR isn't a clean
    // multiple. The row-padding contract we're actually pinning is
    // `length === imgWidth × imgHeight × 4`, not pixel-exact agreement
    // between two independent rounding paths.
    expect(Math.abs(probe.imgWidth - probe.canvasWidth)).toBeLessThanOrEqual(1);
    expect(Math.abs(probe.imgHeight - probe.canvasHeight)).toBeLessThanOrEqual(1);
    expect(probe.length).toBe(probe.imgWidth * probe.imgHeight * 4);
    expect(probe.length).toBeGreaterThan(0);
  });

  test('captureHDRPixels round-trips for visible-ldr and hdr-effects-pre-tone modes', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
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
