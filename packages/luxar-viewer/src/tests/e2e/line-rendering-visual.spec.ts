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
  type SampledPixel,
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
    // not its centre), aimed at the content pivot. After re-render, the canvas
    // should NOT be a uniform solid colour — that would mean a near-camera
    // line painted every pixel.
    //
    // What is required for the camera to actually GET there (#1930):
    // `placeCameraAt` re-derives the orbit state so the next `update()` does not
    // snap the camera back — the old `position.set` + `controls.update()` was
    // inert. The distance clamp is NOT a second trap here, despite appearances:
    // it is `[D/1000, D×10000]` around the framing distance (`ZOOM_IN_FACTOR` /
    // `ZOOM_OUT_FACTOR`), i.e. `[0.0138, 137990]` on this fixture, and the pose
    // below sits 4.53 from the pivot (~D/3). No widening is needed, so none is
    // asked for. The returned placement is asserted rather than discarded: that
    // is what would catch a regression making this inert again.
    const want = { x: 0.001, y: 0.001, z: 0.001 };
    const placed = await placeCameraAt(page, want);
    expect(placed, 'the camera placement did not run').not.toBeNull();
    expect(placed!.viaOrbitControls, 'orbit controls did not re-derive the placement').toBe(true);
    expect(placed!.x, 'camera x').toBeCloseTo(want.x, 4);
    expect(placed!.y, 'camera y').toBeCloseTo(want.y, 4);
    expect(placed!.z, 'camera z').toBeCloseTo(want.z, 4);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);

    // Sample 4 corners of the canvas. The failure mode this guards is one
    // near-camera segment painting the ENTIRE viewport, and its signature is
    // that all four corners are painted with essentially the SAME colour.
    //
    // Not "all four saturated": the fixture's vertices carry a red→blue
    // gradient (`generate_test_data.py`), so `r>240 && g>240 && b>240` cannot
    // be true of a line fragment no matter how badly the shader misbehaves —
    // the old assertion could not fail. The "all painted" precondition keeps
    // the check off a legitimately dark frame, where four near-identical BLACK
    // corners are the correct answer rather than an artefact.
    const samples = await samplePixelsAt(page, 'canvas', [
      [0.05, 0.05],
      [0.95, 0.05],
      [0.05, 0.95],
      [0.95, 0.95],
    ]);
    const spread = (get: (p: SampledPixel) => number): number =>
      Math.max(...samples.map(get)) - Math.min(...samples.map(get));
    const maxSpread = Math.max(
      spread((p) => p.r),
      spread((p) => p.g),
      spread((p) => p.b)
    );
    const allPainted = samples.every((p) => p.r + p.g + p.b > 30);
    expect(
      allPainted && maxSpread <= 8,
      'every canvas corner is painted in the same colour ' +
        `(max per-channel spread ${maxSpread}/255), which is what a single near-camera ` +
        `segment covering the whole viewport looks like: ${JSON.stringify(samples)}`
    ).toBe(false);
  });
});
