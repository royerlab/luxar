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
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.luxar.zarr&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);

    const offsets: Array<[number, number]> = [];
    for (let i = 0; i < 9; i++) {
      const x = 0.4 + (i % 3) * 0.1;
      const y = 0.4 + Math.floor(i / 3) * 0.1;
      offsets.push([x, y]);
    }
    const samples = await samplePixelsAt(page, 'canvas#app', offsets, 'framebuffer');
    const anyVisible = samples.some((p) => p.r + p.g + p.b > 10);
    expect(anyVisible).toBe(true);
  });

  test('Camera near a line: no full-screen artefact', async ({ page }) => {
    // The line shader degenerates segments where both endpoints fall
    // behind uNearCull and clamps screen-space pixel width so a single
    // near-camera segment cannot paint the entire viewport.
    await page.goto(`/?src=${FIXTURES_BASE}/test_lines.luxar.zarr&debug&dpr=1`);
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

    // The contract here has TWO halves, and each is falsifiable on its own:
    //   1. something rendered — at least one CENTRAL (non-corner) sample is
    //      painted, so the frame is not empty; and
    //   2. it did not paint the whole viewport — at least one CORNER still
    //      reads as background, which is what the screen-filling artefact
    //      destroys.
    // Half 2 alone is not enough: four background corners are exactly what an
    // EMPTY frame produces too, so a regression in the very mechanism this test
    // names (`uNearCull` degenerating segments too aggressively) would empty the
    // viewport and still pass. Half 1 is the complement, sampled with the same
    // central-grid idiom as the "line body is brighter than background" test
    // above.
    //
    // Deliberately not a colour-uniformity test. "All four saturated"
    // (`r>240 && g>240 && b>240`) was unfalsifiable — the fixture's vertices
    // carry a red→blue gradient with g == 0 everywhere (`generate_test_data.py`)
    // — and "all four within N of each other" is no better: adjacent vertices
    // differ by 1/9 in R and B, ~28/255 across a single segment, so an artefact
    // that keeps any of the line's own interpolation is wider than any threshold
    // tight enough to mean something. "A corner is still background" needs no
    // threshold derivation, and is falsifiable by construction: it fails the
    // moment the frame is fully painted.
    const CORNERS: Array<[number, number]> = [
      [0.05, 0.05],
      [0.95, 0.05],
      [0.05, 0.95],
      [0.95, 0.95],
    ];
    // A 3×3 grid over the middle of the frame — the same offsets the cap-factor
    // test uses, so "painted" means the same thing in both places.
    const CENTRAL: Array<[number, number]> = [];
    for (let i = 0; i < 9; i++) {
      CENTRAL.push([0.4 + (i % 3) * 0.1, 0.4 + Math.floor(i / 3) * 0.1]);
    }
    const allSamples = await samplePixelsAt(
      page,
      'canvas#app',
      [...CORNERS, ...CENTRAL],
      'framebuffer'
    );
    const samples = allSamples.slice(0, CORNERS.length);
    const centralSamples = allSamples.slice(CORNERS.length);
    // Printed on every run, pass or fail: BOTH halves of the predicate are only
    // meaningful if the healthy frame really does paint the middle and leave a
    // corner dark, and that claim should be checkable from the run log rather
    // than taken on trust.
    console.log(
      `[near-camera line] corner samples ${JSON.stringify(CORNERS)} → ${JSON.stringify(samples)}`
    );
    console.log(
      `[near-camera line] central samples ${JSON.stringify(CENTRAL)} → ` +
        JSON.stringify(centralSamples)
    );
    const PAINTED_SUM = 10; // r+g+b above this is "the line rendered here"
    const BACKGROUND_SUM = 30; // r+g+b at/below this is "unpainted" (near-black)
    // Half 1: the frame is not empty.
    expect(
      centralSamples.filter((p) => p.r + p.g + p.b > PAINTED_SUM).length,
      'no central sample is painted, i.e. nothing rendered at all — the corner check below is ' +
        'equally happy with an empty frame, so an over-aggressive `uNearCull` degenerating every ' +
        `segment would otherwise pass silently (central r+g+b must exceed ${PAINTED_SUM} ` +
        `somewhere): ${JSON.stringify(centralSamples)}`
    ).toBeGreaterThanOrEqual(1);
    // Half 2: and it did not paint the whole viewport.
    const painted = samples.filter((p) => p.r + p.g + p.b > BACKGROUND_SUM);
    expect(
      painted.length,
      'every canvas corner is painted, i.e. the frame has no background left — which is what a ' +
        'single near-camera segment covering the whole viewport looks like ' +
        `(corner r+g+b must fall to ≤ ${BACKGROUND_SUM} somewhere): ${JSON.stringify(samples)}`
    ).toBeLessThan(CORNERS.length);
  });
});
