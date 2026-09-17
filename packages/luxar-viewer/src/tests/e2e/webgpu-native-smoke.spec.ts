/**
 * Best-effort native-WebGPU smoke spec.
 *
 * Headless Playwright chromium typically falls back to WebGPURenderer's
 * internal WebGL2 backend, so the bulk of CI exercises the TSL graphs
 * *through* WebGL2 — useful but doesn't catch WGSL compilation issues,
 * real WebGPU adapter limits, or the 256-byte row-padding deinterlace.
 * The native-only tests here auto-skip when no real WebGPU adapter is
 * present; the HDR capture round-trip at the bottom deliberately does
 * not (see its own comment — it makes no native claim).
 *
 * A native gate MUST probe the physical backend, not
 * `capabilities.apiSurface` (#1449): `apiSurface` is `'webgpu'` for any
 * active `WebGPURenderer` *including* one whose internal backend has
 * fallen back to WebGL2 — that is its documented meaning ("which
 * method-signature contract?", not "which GPU backend?"). Gating on it
 * alone made these tests run on the WebGL2 fallback in headless CI and
 * report "native WebGPU verified" for a run that never touched WGSL or
 * the 256-byte row padding. Hence the shared `probeWebGPUBackend`
 * helper; do not "simplify" the gate back to an `apiSurface` check.
 *
 * The suite is deliberately fast and read-only: a screenshot at an
 * unaligned width (forces `compactWebGPUReadbackRows` to engage) and
 * an HDR capture round-trip in two of its three modes
 * (`raw-scene-hdr` is not covered). Heavier validation lives in
 * the TSL parity harness (`tsl-shader-parity.spec.ts`).
 *
 * @see picking/PICKING_DESIGN.md — WebGPU implementation details
 * @see rendering/README.md — Dual-stack architecture overview
 */

import { test, expect } from './fixtures';
import { probeWebGPUBackend, waitForLuxarReady, waitForPointsLoaded } from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

const SKIP_REASON =
  'these tests require a real WebGPU adapter; ' +
  'Playwright chromium falls back to WebGL2 in headless mode. ' +
  'Run manually with a WebGPU-enabled Chrome stable for full coverage.';

const SURFACE_SKIP_REASON =
  'no WebGPURenderer surface on this page: caps.apiSurface is not "webgpu", ' +
  'so the WebGPU readback arm this test exercises does not exist here.';

test.describe('WebGPU smoke (native-only tests skip on the WebGL2 fallback)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?renderer=webgpu&src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
  });

  test('under a native WebGPU backend, caps report apiSurface "webgpu" and a top-down framebuffer', async ({
    page,
  }) => {
    // The gate is the physical backend alone, so both capability
    // fields below stay falsifiable claims about what the viewer
    // derived from it. Skipping on the WebGL2 fallback is deliberate:
    // `?webgpuForceWebgl` pins that path in `renderer-url-param.spec.ts`
    // and the first `y-orientation.spec.ts` case.
    const probe = await probeWebGPUBackend(page);
    test.skip(!probe.isNative, SKIP_REASON);
    expect(probe.apiSurface).toBe('webgpu');
    expect(probe.framebufferYDown).toBe(true);
  });

  test('screenshot at an unaligned canvas width returns a non-empty, compact buffer', async ({
    page,
  }) => {
    test.skip(!(await probeWebGPUBackend(page)).isNative, SKIP_REASON);

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

  // NOT native-gated, on purpose. This is a `?renderer=webgpu` *surface*
  // round-trip: it runs on the native adapter AND on WebGPURenderer's
  // WebGL2 backend, so it makes no WGSL / real-adapter claim. What it does
  // prove holds on either backend — `captureHDRPixels` reaches
  // `readPixelsCompactAsync` through the WebGPURenderer method-signature
  // contract (the arm that takes no destination buffer and compacts padded
  // rows, which the `?renderer=webgl` path never takes), and the returned
  // buffer honours `length === width * height * 4`. Gating it on the native
  // backend would leave `captureHDRPixels` with zero automated execution
  // anywhere.
  //
  // The gate is therefore `apiSurface`, NOT the native-backend probe: that
  // arm is selected by `caps.apiSurface !== 'webgl2'` in
  // `readPixelsCompactAsync`, so the claim above holds exactly while the
  // WebGPURenderer surface is present. A page on a plain
  // `THREE.WebGLRenderer` reports `'webgl2'` and takes the WebGL arm, which
  // would pass this test while proving the opposite of what it says. Today
  // `?renderer=webgpu` cannot land there — it is an explicit override, so
  // `renderer-setup.ts` skips its `{ fallback: true }` branch, and an
  // adapter-less browser still gets a WebGPURenderer over three's internal
  // WebGL2 backend (surface `'webgpu'`). The skip is the guard for that
  // policy changing; the `toBeDefined()` above is what catches drift in the
  // debug path itself.
  test('captureHDRPixels round-trips through the WebGPU readback surface (either backend)', async ({
    page,
  }) => {
    const { apiSurface } = await probeWebGPUBackend(page);
    // Assert BEFORE the skip that the probe actually read a surface.
    // `probeWebGPUBackend` optional-chains its way to `capabilities`, so
    // a renamed debug path yields `undefined` — which would skip on the
    // line below and quietly retire the only automated execution of
    // `captureHDRPixels` in the repo. `waitForLuxarReady` has already
    // waited for `getState().initialized`, by which point capabilities
    // are constructed, so an absent surface is drift, not a fallback.
    expect(apiSurface, 'probeWebGPUBackend read no apiSurface').toBeDefined();
    test.skip(apiSurface !== 'webgpu', SURFACE_SKIP_REASON);

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
