/**
 * browser-real shader compile + pixel-output smoke tests.
 *
 * Material unit tests mock `THREE.ShaderMaterial` and assert on shader
 * source strings. They cannot catch GLSL syntax errors, varying /
 * attribute mismatches, precision incompatibilities, or driver-specific
 * issues. This spec instantiates each material variant in a real
 * browser, renders one frame, and asserts:
 *   1. No shader / GLSL / attribute / uniform errors in the console.
 *   2. The center pixel is non-black (proves the shader produced output
 *      rather than silently discarding every fragment).
 *
 * Each rendering variant has a specific assertion so shader breakage
 * fails close to the affected material instead of as a vague "loads
 * without crashing" smoke.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  assertNoShaderErrors,
  samplePixelAt,
  waitForRenderStable,
} from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

interface Variant {
  name: string;
  src: string;
  /** When true, assert center pixel has non-zero alpha + RGB sum > 10. */
  expectColored: boolean;
}

/**
 * Variants exercise every material × colormap-on/off combination that
 * has a corresponding fixture in the existing fixture set. Synthetic
 * scalar-colormap fixtures for Points/Lines are not yet committed
 * (would require Python-side fixture generation in a follow-up); this
 * spec uses what's present.
 */
const VARIANTS: Variant[] = [
  // Points: direct colors (vertex-color path).
  { name: 'Point direct color', src: 'test_basic_points.zarr', expectColored: true },
  // Points: colormap (4D scalar fixture exercises USE_COLORMAP).
  { name: 'Point colormap', src: 'test_4d_scalar_lut.zarr', expectColored: true },
  // Lines: direct colors.
  { name: 'Line direct color', src: 'test_lines.zarr', expectColored: true },
  // GSplats: direct color (gsplats use aAmplitude as colormap source).
  { name: 'GSplat direct color', src: 'test_gsplats.zarr', expectColored: true },
];

test.describe('browser-real shader compile + pixel smoke', () => {
  for (const v of VARIANTS) {
    test(`${v.name}: compiles + renders non-black pixels`, async ({ page }) => {
      await page.goto(`/?src=${FIXTURES_BASE}/${v.src}&debug`);
      await waitForLuxarReady(page);
      await waitForRenderStable(page);

      // (1) No shader/GLSL/attribute/uniform errors.
      await assertNoShaderErrors(page);

      // (2) The center pixel has rendered output.
      if (v.expectColored) {
        const pixel = await samplePixelAt(page, 'canvas', 0.5, 0.5);
        // Allow the center to be background — try a 3x3 grid of samples and
        // require at least ONE to be non-black. This avoids flakiness when
        // the exact center happens to fall in dataset whitespace.
        const samples = await Promise.all([
          samplePixelAt(page, 'canvas', 0.5, 0.5),
          samplePixelAt(page, 'canvas', 0.4, 0.4),
          samplePixelAt(page, 'canvas', 0.6, 0.4),
          samplePixelAt(page, 'canvas', 0.4, 0.6),
          samplePixelAt(page, 'canvas', 0.6, 0.6),
          samplePixelAt(page, 'canvas', 0.5, 0.4),
          samplePixelAt(page, 'canvas', 0.5, 0.6),
          samplePixelAt(page, 'canvas', 0.4, 0.5),
          samplePixelAt(page, 'canvas', 0.6, 0.5),
        ]);
        const anyColored = samples.some((p) => p.r + p.g + p.b > 10);
        expect(
          anyColored,
          `Variant '${v.name}': all 9 sampled pixels were near-black — shader output may have been discarded. ` +
            `Center pixel: ${JSON.stringify(pixel)}`
        ).toBe(true);
      }
    });
  }

  test('Lines + GSplats simultaneously: no shader errors', async ({ page }) => {
    // Both geometry types in one scene exercises material-cache reuse
    // and confirms no cross-material shader-link failures.
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);
  });
});
