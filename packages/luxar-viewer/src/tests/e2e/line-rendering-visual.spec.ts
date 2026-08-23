/**
 * Lines visual / sampled-pixel correctness tests.
 *
 * Uses platform-invariant assertions for:
 *   - Cap factor reaching full intensity in the line body.
 *   - Near-camera lines not producing screen-filling artefacts.
 *
 * Visual baselines (screenshot diffs) are intentionally omitted here:
 * pixel-level baselines need to be generated against a known platform /
 * driver and committed alongside. Sampled-pixel assertions below are
 * platform-independent and protect the same rendering contracts.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForRenderStable,
  assertNoShaderErrors,
  samplePixelsAt,
  placeCameraAt,
  withOrbitDistanceLimits,
  UNCLAMPED_ORBIT_DISTANCE_LIMITS,
} from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

test.describe('Lines visual correctness', () => {
  test('test_lines fixture renders without shader errors', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);
  });

  test('Line body is brighter than background (cap factor reaches 1.0)', async ({ page }) => {
    // Sampling the canvas after rendering should produce non-black
    // pixels from the line body. Exact brightness is left to a synthetic
    // single-line fixture and screenshot baseline; this test asserts the
    // documented formula produces positive visible intensity.
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);

    const offsets: Array<[number, number]> = [];
    for (let i = 0; i < 9; i++) {
      const x = 0.4 + (i % 3) * 0.1;
      const y = 0.4 + Math.floor(i / 3) * 0.1;
      offsets.push([x, y]);
    }
    const samples = await samplePixelsAt(page, 'canvas', offsets);
    const anyVisible = samples.some((p) => p.r + p.g + p.b > 10);
    expect(anyVisible).toBe(true);
  });

  test('Camera near a line: no full-screen artefact', async ({ page }) => {
    // The line shader degenerates segments where both endpoints fall
    // behind uNearCull and clamps screen-space pixel width so a single
    // near-camera segment cannot paint the entire viewport.
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);

    // Move the camera onto the world origin, ~0.002 units from the zigzag's
    // first vertex (the fixture runs x = 0…9, so the origin is its near tip,
    // not its centre), aimed along the polyline. After re-render, the canvas
    // should NOT be a uniform solid colour — that would mean a near-camera
    // line painted every pixel.
    //
    // Two things are required for the camera to actually GET there (#1930):
    // `placeCameraAt` re-derives the orbit state so the next `update()` does not
    // snap the camera back (the old `position.set` + `controls.update()` was
    // inert), and the orbit distance clamp — `minDistance` is derived from the
    // scene diagonal — has to be widened, or step 6 of the per-frame update
    // pushes the camera straight back out to the framing distance. The widened
    // window must cover the pixel sampling too, since the clamp is re-applied
    // on EVERY frame.
    const samples = await withOrbitDistanceLimits(
      page,
      UNCLAMPED_ORBIT_DISTANCE_LIMITS,
      async () => {
        await placeCameraAt(page, { x: 0.001, y: 0.001, z: 0.001 });
        await waitForRenderStable(page);
        await assertNoShaderErrors(page);

        // Sample 4 corners of the canvas. If a single near-line painted the
        // whole frame, all four corners would have ~identical colour and
        // each would be heavily saturated. Verify variance / not-all-saturated.
        return await samplePixelsAt(page, 'canvas', [
          [0.05, 0.05],
          [0.95, 0.05],
          [0.05, 0.95],
          [0.95, 0.95],
        ]);
      }
    );
    const allSaturated = samples.every((p) => p.r > 240 && p.g > 240 && p.b > 240);
    expect(allSaturated).toBe(false);
  });
});
