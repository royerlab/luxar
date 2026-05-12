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
  // Points: colormap (4D scalar fixture exercises USE_COLORMAP).
  { name: 'Point colormap', src: 'test_4d_scalar_lut.zarr', expectColored: true },
  // Lines: direct colors.
  { name: 'Line direct color', src: 'test_lines.zarr', expectColored: true },
  // GSplats: direct color (gsplats use aAmplitude as colormap source).
  { name: 'GSplat direct color', src: 'test_gsplats.zarr', expectColored: true },
];

// Sample a denser grid across most of the canvas — at Neutral tone
// mapping the scene background (`0x111111` ≈ 0.0057 linear) gets
// darkened to ~0, so we need to actually land on dataset content to
// see non-black output. Datasets aren't guaranteed to be centered;
// the previous 3×3 grid at [0.4..0.6] occasionally missed thin
// strips of points/lines.
const SAMPLE_OFFSETS: Array<[number, number]> = [];
for (let yi = 0; yi < 6; yi++) {
  for (let xi = 0; xi < 6; xi++) {
    SAMPLE_OFFSETS.push([0.15 + (xi * 0.7) / 5, 0.15 + (yi * 0.7) / 5]);
  }
}

test.describe('browser-real shader compile + pixel smoke', () => {
  for (const v of VARIANTS) {
    test(`${v.name}: compiles + renders non-black pixels`, async ({ page }) => {
      await page.goto(`/?src=${FIXTURES_BASE}/${v.src}&debug`);
      await waitForLuxarReady(page);
      await waitForRenderStable(page);

      // (1) No shader/GLSL/attribute/uniform errors.
      await assertNoShaderErrors(page);

      // (2) Some pixel on the canvas has rendered output. We sample a
      //     6×6 grid across the middle 70% of the canvas — denser than
      //     the previous 3×3 — so sparse-data fixtures (a single line
      //     strip, a thin gsplat cluster) reliably land on at least
      //     one non-black pixel.
      if (v.expectColored) {
        const samples = await Promise.all(
          SAMPLE_OFFSETS.map(([x, y]) => samplePixelAt(page, 'canvas', x, y))
        );
        const anyColored = samples.some((p) => p.r + p.g + p.b > 10);
        const brightest = samples.reduce(
          (best, p) => (p.r + p.g + p.b > best.r + best.g + best.b ? p : best),
          { r: 0, g: 0, b: 0, a: 0 }
        );
        expect(
          anyColored,
          `Variant '${v.name}': all ${SAMPLE_OFFSETS.length} sampled pixels were near-black — ` +
            `shader output may have been discarded. Brightest sample: ${JSON.stringify(brightest)}`
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
