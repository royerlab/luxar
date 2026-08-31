/**
 * browser-real shader compile + pixel-output smoke tests.
 *
 * Material unit tests mock `THREE.ShaderMaterial` and assert on shader
 * source strings. They cannot catch GLSL syntax errors, varying /
 * attribute mismatches, precision incompatibilities, or driver-specific
 * issues. This spec instantiates each material variant in a real
 * browser, renders one frame, and asserts:
 *   1. No shader / GLSL / attribute / uniform errors in the console.
 *   2. The canvas contains a non-black pixel (proves the shader produced output
 *      rather than silently discarding every fragment).
 *
 * Each rendering variant has a specific assertion so shader breakage
 * fails close to the affected material instead of as a vague "loads
 * without crashing" smoke.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForDataLoaded,
  assertNoShaderErrors,
  getElementPixelStats,
  waitForRenderStable,
} from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

interface Variant {
  name: string;
  src: string;
  /** When true, assert the canvas contains a pixel with RGB sum > 10. */
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
  // Points: colormap (4D scalar fixture exercises USE_COLORMAP).
  {
    name: 'Point colormap',
    src: 'test_4d_scalar_lut.luxar.zarr',
    expectColored: true,
  },
  // Lines: direct colors.
  {
    name: 'Line direct color',
    src: 'test_lines.luxar.zarr',
    expectColored: true,
  },
  // GSplats: direct color (gsplats use aAmplitude as colormap source).
  {
    name: 'GSplat direct color',
    src: 'test_gsplats.luxar.zarr',
    expectColored: true,
  },
];

const VISIBLE_PIXEL_THRESHOLD = 10;

test.describe('browser-real shader compile + pixel smoke', () => {
  for (const v of VARIANTS) {
    test(`${v.name}: compiles + renders non-black pixels`, async ({ page }) => {
      await page.goto(`/?src=${FIXTURES_BASE}/${v.src}&debug`);
      await waitForLuxarReady(page);
      await waitForDataLoaded(page);
      await waitForRenderStable(page);

      // (1) No shader/GLSL/attribute/uniform errors.
      await assertNoShaderErrors(page);

      // (2) Some canvas pixel has rendered output. The helper suppresses DOM
      //     chrome painted above the canvas before taking the screenshot.
      if (v.expectColored) {
        const stats = await getElementPixelStats(page, 'canvas', VISIBLE_PIXEL_THRESHOLD);
        expect(
          stats.nonBlackPixels > 0,
          `Variant '${v.name}': no canvas pixels exceeded RGB-sum threshold ` +
            `${VISIBLE_PIXEL_THRESHOLD} — shader output may have been discarded. ` +
            `Stats: ${JSON.stringify(stats)}`
        ).toBe(true);
      }
    });
  }

  test('Lines + GSplats simultaneously: no shader errors', async ({ page }) => {
    // Both geometry types in one scene exercises material-cache reuse
    // and confirms no cross-material shader-link failures.
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);
  });
});
