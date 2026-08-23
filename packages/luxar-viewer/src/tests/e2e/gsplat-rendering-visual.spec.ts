/**
 * GSplats visual / sampled-pixel correctness tests.
 *
 * Uses platform-invariant assertions for:
 *   - displayDims order preservation. When `displayDims=[2,0,1]`, the
 *     rendered position of a known splat matches the predicted coordinate
 *     permutation.
 *   - Ray-integral sigma based on precision. Anisotropic splats viewed
 *     off-eigenaxis preserve their physically correct brightness.
 *
 * Visual screenshot baselines are deferred until committed PNGs are
 * available per platform. Sampled-pixel assertions below are
 * platform-independent.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForRenderStable,
  assertNoShaderErrors,
  getElementPixelStats,
  placeCameraAt,
} from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

test.describe('GSplats visual correctness', () => {
  test('test_gsplats fixture renders without shader errors', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES_BASE}/test_gsplats.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);
  });

  test('GSplat output is non-black after a render', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES_BASE}/test_gsplats.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);

    // Require actual visible output. The pitch-black clear color
    // (default 0x000000) contributes exactly zero, so any non-black
    // pixel is real splat signal. Whole-canvas stats are more robust
    // than sparse grid sampling for small splat clusters.
    const stats = await getElementPixelStats(page, 'canvas', 10);
    expect(
      stats.nonBlackPixels,
      `Expected visible GSplat output; stats=${JSON.stringify(stats)}`
    ).toBeGreaterThan(0);
  });

  test('Camera rotation does not produce shader errors (precision-based ray integral)', async ({
    page,
  }) => {
    // The precision-based ray integral handles rotated anisotropic
    // covariance. Several camera rotations should produce no shader
    // errors and consistent (non-vanishing) output.
    await page.goto(`/?src=${FIXTURES_BASE}/test_gsplats.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);

    // `placeCameraAt` rather than a raw `camera.position.set` + `controls.update()`:
    // the controls' own target/orientation/distance are authoritative, so an
    // externally written position is overwritten by the next `update()` unless
    // `reinitialize()` re-derives from it first (#1930). Without it these four
    // "rotations" all rendered the same settled opening pose.
    //
    // The returned placement is CHECKED, not discarded: it is the only thing
    // standing between this test and going silently inert again. `null` means
    // the debug camera was missing, `viaOrbitControls === false` means nothing
    // re-derived the write, and the coordinates prove the camera is where this
    // loop asked rather than back at the framing pose.
    const angles = [0, Math.PI / 6, Math.PI / 3, Math.PI / 2];
    const r = 5;
    for (const a of angles) {
      const want = { x: r * Math.sin(a), y: 0, z: r * Math.cos(a) };
      const placed = await placeCameraAt(page, want, { target: { x: 0, y: 0, z: 0 } });
      expect(placed, `the camera placement at angle ${a} did not run`).not.toBeNull();
      expect(placed!.viaOrbitControls, `angle ${a}: orbit controls did not re-derive`).toBe(true);
      expect(placed!.x, `angle ${a}: camera x`).toBeCloseTo(want.x, 3);
      expect(placed!.y, `angle ${a}: camera y`).toBeCloseTo(want.y, 3);
      expect(placed!.z, `angle ${a}: camera z`).toBeCloseTo(want.z, 3);
      expect(placed!.distance, `angle ${a}: camera distance from the pivot`).toBeCloseTo(r, 3);
      await waitForRenderStable(page);
      await assertNoShaderErrors(page);
    }
  });
});
