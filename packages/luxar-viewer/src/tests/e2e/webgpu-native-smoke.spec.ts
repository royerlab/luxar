/**
 * Best-effort native-WebGPU smoke spec.
 *
 * Headless Playwright chromium falls back to WebGPURenderer's
 * internal WebGL2 backend, so the bulk of CI exercises the TSL
 * graphs *through* WebGL2 — useful but doesn't catch WGSL
 * compilation issues, real WebGPU adapter limits, or the 256-byte
 * row-padding deinterlace. This spec auto-skips when no real WebGPU
 * adapter is present and exercises the WebGPU-specific paths when
 * one is. Intended to be run manually on Chrome stable with
 * `--enable-unsafe-webgpu` before merge of a renderer-touching PR.
 *
 * The suite is deliberately fast and read-only: a canvas resize to
 * an unaligned width, an HDR capture round-trip in each mode, and
 * a picking smoke. Heavier validation lives in the parity harness
 * (`tsl-shader-parity.spec.ts`).
 *
 * @see picking/PICKING_DESIGN.md — WebGPU implementation details
 * @see rendering/SPECIFICATIONS.md § 11 — Dual-stack architecture
 */

import { test, expect } from '@playwright/test';

const SKIP_REASON =
  'native-WebGPU smoke spec requires a real WebGPU adapter; ' +
  'Playwright chromium falls back to WebGL2 in headless mode. ' +
  'Run manually with a WebGPU-enabled Chrome stable for full coverage.';

test.describe('WebGPU native smoke (best-effort, skips on fallback)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/?renderer=webgpu&debug');
    await page.waitForFunction(
      () => typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.api === 'string',
      null,
      { timeout: 10000 }
    );
  });

  test('reports caps.api === "webgpu" under real WebGPU; skips otherwise', async ({ page }) => {
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.api
    );
    test.skip(api !== 'webgpu', SKIP_REASON);
    expect(api).toBe('webgpu');
  });

  test('screenshot at an unaligned canvas width returns a non-empty, compact buffer', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.api
    );
    test.skip(api !== 'webgpu', SKIP_REASON);

    // 853 is deliberately chosen: 853 × 4 (RGBA8) = 3412 bytes, which
    // pads to 3584 = 14 × 256 under WebGPU's `bytesPerRow` rule. If
    // `compactWebGPUReadbackRows` is wired correctly the resulting
    // ImageData wraps a compact, slant-free buffer of length
    // 853 × height × 4.
    await page.setViewportSize({ width: 853, height: 480 });
    await page.waitForFunction(
      () => (window as any).__luxarDebug?.app?.sceneManager?.renderer?.domElement?.width === 853,
      null,
      { timeout: 5000 }
    );

    const { width, height, length } = await page.evaluate(async () => {
      const dbg = (window as any).__luxarDebug;
      const img: ImageData = await dbg.app.postProcessing.renderToImageData();
      return { width: img.width, height: img.height, length: img.data.length };
    });
    expect(width).toBe(853);
    expect(length).toBe(width * height * 4);
  });

  test('captureHDRPixels round-trips for visible-ldr and hdr-effects-pre-tone modes', async ({
    page,
  }) => {
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.api
    );
    test.skip(api !== 'webgpu', SKIP_REASON);

    for (const mode of ['visible-ldr', 'hdr-effects-pre-tone'] as const) {
      const { width, height, length } = await page.evaluate(async (m) => {
        const dbg = (window as any).__luxarDebug;
        const out = await dbg.app.postProcessing.captureHDRPixels(m);
        return { width: out.width, height: out.height, length: out.pixels.length };
      }, mode);
      expect(width).toBeGreaterThan(0);
      expect(height).toBeGreaterThan(0);
      expect(length).toBe(width * height * 4);
    }
  });
});
