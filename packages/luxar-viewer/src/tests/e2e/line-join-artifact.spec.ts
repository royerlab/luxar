/**
 * Line-joint artifact acceptance measurement (issues #780 / #785 / #790).
 *
 * Loads `test_line_joins.luxar.zarr` — five joint cases, each in its own
 * horizontal band of world Y — takes one canvas screenshot, and runs the two
 * pure metrics from `../helpers/line-join-metrics` over every band of that
 * single frame. Measuring one identical frame matters: the bands are
 * compared to each other, so re-screenshotting per band would let a stray
 * frame difference masquerade as a joint defect.
 *
 * Two things are being protected here. What is already correct must stay
 * correct while the vertex stage is rewritten for the miter-join series:
 * both straight bands carry zero dark and zero bright outliers (#785), every
 * band's flux profile is gapless, and the straight profiles are flat (#780 —
 * no bead-chain dip at interior joints). The nine-ray hub stays inside a
 * small ceiling as the never-mitered control. What is still broken is
 * recorded rather than fixed: `curve_smooth` and `zigzag_right_angle` still
 * show the uncovered outer-side wedge and the inner-side double-cover lens,
 * so their outlier fractions are only held under documented ceilings, and
 * the measured numbers are printed for the record.
 *
 * Some context before reading a number out of this spec. The local-median
 * metric only counts the part of a wedge that is still a couple of pixels
 * across, and a wedge grows from nothing at the centreline to roughly
 * `half_width x turn_angle` at the tube edge — so the gentle `curve_smooth`
 * scores far higher than the 90-degree `zigzag_right_angle`, whose wedge is
 * much worse but much wider. The full measured envelope is in the
 * `line-join-metrics.ts` module header, and the flux profile is what
 * responds on the wide-wedge band.
 *
 * Free-end cap behaviour is deliberately outside every measured region in
 * part 1: the four horizontal bands inset their X range by 1.0 world unit so
 * the end ramps never enter a rectangle. Free ends are part of the #785
 * verification set, so whoever closes #790 should not read these assertions
 * as covering them.
 *
 * The metrics run on the display-encoded luminance of the composited frame
 * rather than on linearised radiance. Both metrics are relative and the
 * encoding is monotone, so a defect still registers, though compressed by
 * roughly 2.3x near a mid-bright tube value (see the module header). The
 * remaining non-monotone terms are pinned rather than reasoned around: the
 * fixture fixes an identity tone response with bloom, AA and noise off, and
 * this spec pins the device pixel ratio with `&dpr=1`, because doubling the
 * DPR does not scale the metric's answer — it slides the wedge across the
 * window's sensitivity boundary.
 */

import type { Page } from '@playwright/test';

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForRenderStable,
  assertNoShaderErrors,
  captureCanvasRGBA,
  getLuxarState,
  type CanvasFrameRGBA,
} from './helpers';
import {
  measureAxialFlux,
  measureLocalMedianOutliers,
  rgbaToLuminance,
  type PixelRect,
} from '../helpers/line-join-metrics';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

/** A band's node name, the world-space AABB its pixels live in, and its floors. */
interface BandBox {
  /** Node name in the fixture scene. */
  name: string;
  /** World-space AABB, all at z = 0. */
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  /**
   * Inside-pixel floor, set at roughly half the value measured on
   * 2026-08-06. This is a "the band rendered at all" gate, not a
   * localisation gate — several bands clear each other's floors, so
   * `expectedWidth` / `expectedHeight` are what actually pin the camera and
   * the projection.
   */
  minInsidePixels: number;
  /**
   * Projected rectangle the band's AABB must produce, in pixels. The width
   * is exact: the four horizontal bands all span 18 world units and the hub
   * spans 3. The height is checked within `RECT_HEIGHT_TOLERANCE_PX`,
   * because its rounding depends on the canvas's exact device height and
   * subpixel phase — a mislocated or misprojected rectangle is out by tens
   * of pixels, not by one.
   */
  expectedWidth: number;
  expectedHeight: number;
  /**
   * Whether the tube spans the whole rectangle along X. True for the four
   * horizontal bands; false for the hub, whose rays legitimately leave the
   * left and right ends of its box empty.
   */
  spansRect: boolean;
}

/**
 * The one place the band rectangles live on the TypeScript side.
 *
 * These mirror the table in `generate_line_joins_test()`
 * (`tests/fixtures/generate_test_data.py`) exactly — change one and you must
 * change the other, or the spec silently measures the wrong pixels. The four
 * horizontal bands span world x [-10, 10] and their X range is inset exactly
 * 1.0 unit from those ends so the free-end cap ramps stay out of the
 * measured region; `hub_9ray` is measured whole. `curve_smooth` gets a
 * taller box than its siblings because its 28.8 px half-width plus its
 * 0.7-unit amplitude would otherwise reach the box edge exactly.
 */
const LINE_JOIN_BANDS: readonly BandBox[] = [
  {
    name: 'curve_smooth',
    xMin: -9,
    xMax: 9,
    yMin: 6.3,
    yMax: 9.7,
    minInsidePixels: 19000,
    expectedWidth: 648,
    expectedHeight: 124,
    spansRect: true,
  },
  {
    name: 'zigzag_right_angle',
    xMin: -9,
    xMax: 9,
    yMin: 2.5,
    yMax: 5.5,
    minInsidePixels: 19000,
    expectedWidth: 648,
    expectedHeight: 109,
    spansRect: true,
  },
  {
    name: 'straight_thin',
    xMin: -9,
    xMax: 9,
    yMin: -1.5,
    yMax: 1.5,
    minInsidePixels: 7000,
    expectedWidth: 648,
    expectedHeight: 109,
    spansRect: true,
  },
  {
    name: 'straight_thick',
    xMin: -9,
    xMax: 9,
    yMin: -5.5,
    yMax: -2.5,
    minInsidePixels: 25000,
    expectedWidth: 648,
    expectedHeight: 109,
    spansRect: true,
  },
  {
    name: 'hub_9ray',
    xMin: -1.5,
    xMax: 1.5,
    yMin: -9.5,
    yMax: -6.5,
    minInsidePixels: 3500,
    expectedWidth: 108,
    expectedHeight: 109,
    spansRect: false,
  },
];

/** Slack on the projected rectangle heights — see `BandBox.expectedHeight`. */
const RECT_HEIGHT_TOLERANCE_PX = 2;

/**
 * Total line segments the fixture commits: curve 120 + zigzag 16 + thin 40 +
 * thick 20 + hub 9. Hand-derived from the vertex counts in
 * `generate_line_joins_test()` and NOT computed from `LINE_JOIN_BANDS`,
 * which knows nothing about segment counts — re-derive it by hand whenever
 * the fixture geometry changes.
 */
const EXPECTED_LINE_SEGMENTS = 120 + 16 + 40 + 20 + 9;

/**
 * Dark- and bright-outlier ceilings for the two bending cases.
 *
 * Measured 2026-08-06, headless Chromium, at the fixture's pinned framing:
 * `curve_smooth` 5.07% dark / 1.35% bright, `zigzag_right_angle` 0.13% dark
 * / 0.00% bright. The ceilings sit well above those with room for GPU,
 * driver and resolution differences — they are guard rails against a
 * blow-up, not spec values, and they are geometry- and resolution-dependent,
 * so re-measure before tightening them.
 *
 * When the miter join lands the dark ceiling drops to zero: a later part of
 * the #790 series is expected to replace it with `toBe(0)`, matching what
 * the straight bands already assert.
 */
const BEND_DARK_CEILING = 0.08;
const BEND_BRIGHT_CEILING = 0.04;

/**
 * Floor on the zigzag band's axial flux p05. Measured 0.743; an ideal
 * join-free zigzag models to 0.898, and the curve barely moves either way
 * (0.860 to 0.855), so this is the one band where the flux profile
 * genuinely discriminates. Set loose in the same guard-rail spirit as the
 * outlier ceilings — this is the number the miter part of the series should
 * watch climb.
 */
const ZIGZAG_FLUX_P05_FLOOR = 0.6;

/**
 * Ceilings for the never-mitered control. A degree-9 branch point has all
 * nine quads stacked around the hub with cap suppression at 0, so it has no
 * uncovered wedge to begin with; measured 0.11% dark / 0.11% bright, and it
 * is here to stay stable.
 */
const HUB_OUTLIER_CEILING = 0.01;

/** Normalised-device-coordinate bounding box of one band. */
interface NdcBox {
  name: string;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Block until every fixture line node has committed its full segment count.
 *
 * `waitForLuxarReady` only polls `initialized` and `waitForRenderStable`
 * only watches the frame counter, so neither knows whether the data worker's
 * projection has reached the GPU. Screenshotting before it does produced a
 * bimodal `curve_smooth` dark fraction across repeat runs (~2% on a settled
 * frame, ~5% on a half-committed one) — a race, not noise.
 *
 * The 15 s budget is deliberately short: readiness already consumed up to
 * 45 s of the per-test budget, so a longer wait here would be cut off by the
 * test timeout and the operator would see a bare "test timed out" instead of
 * the diagnostic below.
 */
async function waitForLineSegmentsCommitted(
  page: Page,
  expectedSegments: number,
  expectedMeshes: number
): Promise<void> {
  try {
    await page.waitForFunction(
      ({ segments, meshes }: { segments: number; meshes: number }) => {
        const debug = (window as unknown as { __luxarDebug?: { getState?: () => unknown } })
          .__luxarDebug;
        if (typeof debug?.getState !== 'function') return false;
        const state = debug.getState() as {
          lineMeshes?: Array<{ segmentCount?: number }>;
          totalLines?: number;
        };
        return (state.lineMeshes ?? []).length >= meshes && (state.totalLines ?? 0) === segments;
      },
      { segments: expectedSegments, meshes: expectedMeshes },
      { timeout: 15000 }
    );
  } catch {
    const state = await getLuxarState(page);
    const meshes: Array<{ name?: string; segmentCount?: number }> = state?.lineMeshes ?? [];
    throw new Error(
      `test_line_joins never committed: expected ${expectedMeshes} line meshes totalling ` +
        `${expectedSegments} segments, observed ${meshes.length} meshes totalling ` +
        `${state?.totalLines ?? 0} (` +
        `${meshes.map((m) => `${m.name ?? '?'}=${m.segmentCount ?? 0}`).join(', ') || 'none'}).`
    );
  }
}

/**
 * Project each band's world AABB through the live camera and return its NDC
 * bounding box.
 *
 * Guessing fractions of the canvas would silently drift the moment the
 * fixture's framing changes; projecting through the camera the frame was
 * actually rendered with cannot.
 */
async function projectBands(page: Page, bands: readonly BandBox[]): Promise<NdcBox[]> {
  return await page.evaluate((boxes: readonly BandBox[]) => {
    const debug = (window as { __luxarDebug?: Record<string, any> }).__luxarDebug;
    const camera = debug?.camera ?? debug?.app?.sceneManager?.camera;
    if (!camera) throw new Error('line-join-artifact: debug camera unavailable');
    camera.updateMatrixWorld(true);

    // Borrow a THREE.Vector3 from the camera so no THREE import is needed
    // in the page context.
    const probe = camera.position.clone();

    return boxes.map((box) => {
      const corners: Array<[number, number]> = [
        [box.xMin, box.yMin],
        [box.xMax, box.yMin],
        [box.xMin, box.yMax],
        [box.xMax, box.yMax],
      ];
      let minX = Infinity;
      let maxX = -Infinity;
      let minY = Infinity;
      let maxY = -Infinity;
      for (const [wx, wy] of corners) {
        probe.set(wx, wy, 0);
        probe.project(camera);
        minX = Math.min(minX, probe.x);
        maxX = Math.max(maxX, probe.x);
        minY = Math.min(minY, probe.y);
        maxY = Math.max(maxY, probe.y);
      }
      return { name: box.name, minX, maxX, minY, maxY };
    });
  }, bands);
}

/**
 * Convert an NDC bounding box to a pixel rectangle in the decoded frame.
 * NDC +Y points up; image rows run down.
 */
function ndcBoxToRect(box: NdcBox, frame: CanvasFrameRGBA): PixelRect {
  const toPxX = (ndc: number) => (ndc * 0.5 + 0.5) * frame.width;
  const toPxY = (ndc: number) => (1 - (ndc * 0.5 + 0.5)) * frame.height;
  const x0 = Math.max(0, Math.floor(Math.min(toPxX(box.minX), toPxX(box.maxX))));
  const x1 = Math.min(frame.width, Math.ceil(Math.max(toPxX(box.minX), toPxX(box.maxX))));
  const y0 = Math.max(0, Math.floor(Math.min(toPxY(box.minY), toPxY(box.maxY))));
  const y1 = Math.min(frame.height, Math.ceil(Math.max(toPxY(box.minY), toPxY(box.maxY))));
  return { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) };
}

test.describe('Line-joint artifact measurement (#790)', () => {
  // Fail fast with an actionable message instead of an opaque "measured too
  // few pixels": this fixture is Python-generated and NOT covered by the
  // Playwright global-setup (which only checks datasets/examples).
  test.beforeAll(async () => {
    const { existsSync } = await import('node:fs');
    const path = await import('node:path');
    const { fileURLToPath } = await import('node:url');
    const specDir = path.dirname(fileURLToPath(import.meta.url));
    const fixtureDir = path.resolve(specDir, '../../../tests/fixtures/test_line_joins.luxar.zarr');
    if (!existsSync(fixtureDir)) {
      throw new Error(
        `Missing fixture ${fixtureDir} — run \`pnpm test:generate-fixtures\` ` +
          'from packages/luxar-viewer/ first.'
      );
    }
  });

  test('band metrics on a single frame', async ({ page }) => {
    // ?dpr=1 pins the pixel ratio: the local-median metric is non-monotone
    // in defect pixel width, so a different DPR does not scale the answer.
    await page.goto(`/?src=${FIXTURES_BASE}/test_line_joins.luxar.zarr&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForLineSegmentsCommitted(page, EXPECTED_LINE_SEGMENTS, LINE_JOIN_BANDS.length);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);

    const frame = await captureCanvasRGBA(page);
    const luminance = rgbaToLuminance(frame.rgba, frame.width, frame.height);
    const ndcBoxes = await projectBands(page, LINE_JOIN_BANDS);

    const byName = new Map<
      string,
      {
        rect: PixelRect;
        outliers: ReturnType<typeof measureLocalMedianOutliers>;
        flux: ReturnType<typeof measureAxialFlux>;
      }
    >();

    for (const ndc of ndcBoxes) {
      const rect = ndcBoxToRect(ndc, frame);
      const outliers = measureLocalMedianOutliers(luminance, frame.width, frame.height, rect);
      // Every band's tube runs along the image X axis. The hub's rays are
      // radial, so only its gap count is meaningful, not its profile shape.
      const flux = measureAxialFlux(luminance, frame.width, frame.height, rect, 'x');
      byName.set(ndc.name, { rect, outliers, flux });

      // Printed for the record — these are the #790 baseline the later parts
      // of the series are judged against.
      console.log(
        `[line-join] ${ndc.name}: rect=${rect.x},${rect.y} ${rect.width}x${rect.height} ` +
          `inside=${outliers.insidePixels} ` +
          `dark=${outliers.darkOutliers} (${(outliers.darkFraction * 100).toFixed(3)}%) ` +
          `bright=${outliers.brightOutliers} (${(outliers.brightFraction * 100).toFixed(3)}%) ` +
          `worstDeficit=${outliers.worstDeficit.toFixed(1)} ` +
          `worstExcess=${outliers.worstExcess.toFixed(1)} | ` +
          `flux samples=${flux.samples} empty=${flux.emptySamples} ` +
          `p05=${flux.p05.toFixed(3)} p95=${flux.p95.toFixed(3)} min=${flux.min.toFixed(3)}`
      );
    }

    for (const band of LINE_JOIN_BANDS) {
      const measured = byName.get(band.name);
      expect(measured, `no measurement for band ${band.name}`).toBeDefined();
      const { rect, outliers, flux } = measured!;

      // The projected rectangle pins the camera and the projection.
      expect(rect.width, `band ${band.name} rect width`).toBe(band.expectedWidth);
      expect(rect.height, `band ${band.name} rect height`).toBeGreaterThanOrEqual(
        band.expectedHeight - RECT_HEIGHT_TOLERANCE_PX
      );
      expect(rect.height, `band ${band.name} rect height`).toBeLessThanOrEqual(
        band.expectedHeight + RECT_HEIGHT_TOLERANCE_PX
      );

      // The band rendered at all.
      expect(outliers.insidePixels, `band ${band.name} measured too few pixels`).toBeGreaterThan(
        band.minInsidePixels
      );

      // A torn tube is a defect at any turn angle, so every band is gated on
      // gaps. Without this a renderer that dropped 40% of a bend band would
      // clear its inside-pixel floor, LOWER its dark fraction (fewer joints
      // left to be wrong) and pass every other assertion here.
      expect(flux.emptySamples, `band ${band.name} axial gaps`).toBe(0);
      if (band.spansRect) {
        expect(flux.samples, `band ${band.name} axial samples`).toBe(rect.width);
      }
    }

    const thin = byName.get('straight_thin')!;
    const thick = byName.get('straight_thick')!;
    const curve = byName.get('curve_smooth')!;
    const zigzag = byName.get('zigzag_right_angle')!;
    const hub = byName.get('hub_9ray')!;

    // #785: cap suppression makes a straight polyline's interior joints
    // invisible. Neither straight band may show a single tick, dark or
    // bright — both measure exactly 0 today.
    expect(thin.outliers.darkOutliers, 'straight_thin dark outliers').toBe(0);
    expect(thin.outliers.brightOutliers, 'straight_thin bright outliers').toBe(0);
    expect(thick.outliers.darkOutliers, 'straight_thick dark outliers').toBe(0);
    expect(thick.outliers.brightOutliers, 'straight_thick bright outliers').toBe(0);

    // #780: both straight tubes are uniform by construction, so their flux
    // profiles must be flat. Segment length over width is 3.3 (thin) and
    // 1.67 (thick); a notch is 2 x width long, so `thin` keeps its notches
    // separated and is the more sensitive of the pair.
    for (const [label, band] of [
      ['straight_thin', thin],
      ['straight_thick', thick],
    ] as const) {
      expect(band.flux.p05, `${label} axial p05`).toBeGreaterThan(0.9);
      expect(band.flux.p95, `${label} axial p95`).toBeLessThan(1.1);
    }

    // The never-mitered control: nine quads stacked around one hub vertex.
    expect(hub.outliers.darkFraction, 'hub_9ray dark fraction').toBeLessThan(HUB_OUTLIER_CEILING);
    expect(hub.outliers.brightFraction, 'hub_9ray bright fraction').toBeLessThan(
      HUB_OUTLIER_CEILING
    );

    // #790, recorded rather than fixed: both bending cases still leave an
    // uncovered wedge on the outside of every turn and a double-covered lens
    // on the inside. Asserting 0 here would fail today, and pinning the exact
    // buggy value would be just as wrong, so they are held under documented
    // ceilings that the miter-join part of the series is expected to replace
    // with `toBe(0)`.
    expect(curve.outliers.darkFraction, 'curve_smooth dark fraction').toBeLessThan(
      BEND_DARK_CEILING
    );
    expect(curve.outliers.brightFraction, 'curve_smooth bright fraction').toBeLessThan(
      BEND_BRIGHT_CEILING
    );
    expect(zigzag.outliers.darkFraction, 'zigzag_right_angle dark fraction').toBeLessThan(
      BEND_DARK_CEILING
    );
    expect(zigzag.outliers.brightFraction, 'zigzag_right_angle bright fraction').toBeLessThan(
      BEND_BRIGHT_CEILING
    );

    // Note for whoever closes #790: a zero dark fraction on the zigzag would
    // not prove its wedge is closed. That wedge is a 28.8 px-radius quarter
    // disc, far wider than the local-median metric's couple-of-pixels
    // envelope, so it reads 0.13% today while the far gentler curve reads
    // 5.07%. The flux dip below is the measure that responds on this band.
    expect(zigzag.flux.p05, 'zigzag_right_angle axial p05').toBeGreaterThan(ZIGZAG_FLUX_P05_FLOOR);
  });
});
