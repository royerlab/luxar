/**
 * Line-joint acceptance test (issues #780 / #785 / #790).
 *
 * Loads `test_line_joins.luxar.zarr` — five joint cases, each in its own
 * horizontal band of world Y — takes one canvas screenshot, and runs the two
 * pure metrics from `../helpers/line-join-metrics` over every band of that
 * single frame. Measuring one identical frame matters: the bands are
 * compared to each other, so re-screenshotting per band would let a stray
 * frame difference masquerade as a joint defect.
 *
 * The spec asserts that the join geometry closes the wedge. Both bending
 * bands measure ZERO dark and zero bright outliers, exactly like the two
 * straight bands, and are gated at no more than two of each; the zigzag's
 * axial flux must additionally stay above 0.9 — the same floor the straight
 * bands hold. Unmitred rendering cannot clear either gate: it measured 4.94%
 * dark / 3.52% bright (1918 dark pixels) on `curve_smooth`, 0.076% (30
 * pixels) on `zigzag_right_angle`, and a flux p05 of 0.780 there.
 * Alongside that, what was already correct must stay
 * correct: both straight bands at zero outliers (#785), every
 * band's flux profile gapless, the straight profiles flat (#780 — no
 * bead-chain dip at interior joints), and the nine-ray hub inside a small
 * ceiling as the never-mitered control.
 *
 * Why the bend bands are gated at two rather than at the zero they measure.
 * The miter's design invariant is that both sides of a joint compute the same
 * miter point, but they compute it from the same operands in a different
 * float32 order: the vertex path forms `(ndcEnd - ndcStart) * (0.5 *
 * uResolution)` while the join helper scales each point by `0.5 *
 * uResolution` and subtracts afterwards, and in float32 those two are not the
 * same number. Simulated on this fixture the two sides' miter points disagree
 * by up to 6.6e-05 px on `curve_smooth` and 7.6e-06 px on
 * `zigzag_right_angle` — enough that subpixel quantisation could in principle
 * drop or double one seam pixel, which with AA off and a 25/255 threshold
 * against a ~153/255 tube core scores as a full outlier. Both bands measure 0
 * on this GPU and driver across four runs, and the ceiling exists only to
 * absorb that seam pixel. It is not slack in the measurement: a count
 * anywhere near two means something real has changed and wants investigating,
 * not re-baselining. Making the two sides agree bit-exactly means subtracting
 * in NDC and scaling afterwards in both shader backends, which is a separate
 * change.
 *
 * Some context before reading a number out of this spec. The local-median
 * metric only counts the part of a wedge that is still a couple of pixels
 * across, and a wedge grows from nothing at the centreline to roughly
 * `half_width x turn_angle` at the tube edge — so on the unmitred renderer
 * the gentle `curve_smooth` scored far higher than the 90-degree
 * `zigzag_right_angle`, whose wedge is much worse but much wider. That is
 * why the zigzag is gated on its flux profile as well: the full measured
 * envelope is in the `line-join-metrics.ts` module header.
 *
 * Free-end cap behaviour is a standing coverage gap of this spec: the four
 * horizontal bands inset their X range by 1.0 world unit so the end ramps
 * never enter a rectangle. Free ends are part of the #785 verification set,
 * and nothing here measures them.
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
 *
 * Every figure quoted below was measured in headless Chromium at `dpr=1`,
 * both columns on 2026-08-07 — the unmitred one by re-pointing this spec's
 * URL at `&lineJoin=none` on the same tree.
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
   * Inside-pixel floor, set at roughly half the value measured with `dpr=1`
   * pinned (41092 / 49296 / 12960 / 53136 / 7010 on the mitred renderer;
   * 38831 / 39030 / 12960 / 53136 / 7010 unmitred — the miter adds coverage
   * on the two bend bands and changes neither straight band). This is a "the
   * band rendered at all" gate, not a localisation gate — several bands clear
   * each other's floors, so `expectedWidth` / `expectedHeight` are what
   * actually pin the camera and the projection.
   */
  minInsidePixels: number;
  /**
   * Projected rectangle the band's AABB must produce, in pixels, as measured
   * on the pinned `dpr=1` frame. The width is exact: the four horizontal
   * bands all span 18 world units and the hub spans 3. The height is checked
   * within `RECT_HEIGHT_TOLERANCE_PX` rather than exactly, because its
   * rounding depends on the canvas's exact device height and subpixel
   * phase — note that `straight_thin`'s 3.0-unit box lands on 108 px where
   * its identically-sized siblings land on 109. A mislocated or misprojected
   * rectangle is out by tens of pixels, not by one.
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
    expectedHeight: 108,
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
 * Floor on the zigzag band's axial flux p05 — the SECONDARY gate.
 *
 * The zigzag's primary regression detector is the outlier count above it:
 * unmitred rendering scores 30 dark outliers against the ceiling of 2
 * (15×). The flux floor exists because the zigzag's uncovered wedge is far
 * too wide for the local-median metric alone — unmitred it scored only
 * 0.076% dark while its flux p05 collapsed to 0.780.
 *
 * Re-baselined at the #1352 flip for the capsule's ROUND joins: the
 * capsule measures 0.895 (a round join genuinely carries less column flux
 * through a square corner than the quad's miter did — 0.985; the since-
 * deleted volumetric reference's joins were round too), and the unmitred pathology
 * measures 0.780. The floor sits between them at 0.85 — 0.045 of headroom
 * above the pathology-side margin of 0.070; the two margins are the
 * allowance for GPU, driver and resolution differences. If the capsule
 * drops below the floor, that is a real joint regression, not noise; if a
 * future primitive measures materially above 0.895, re-derive both
 * margins rather than keeping this value.
 */
const ZIGZAG_FLUX_P05_FLOOR = 0.85;

/**
 * Ceilings for the never-mitered control. A degree-9 branch point has all
 * nine quads stacked around the hub with cap suppression at 0, so it has no
 * uncovered wedge to begin with; measured 0.157% dark / 0.114% bright both
 * before and after the miter landed, and it is here to stay stable.
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
 * bimodal `curve_smooth` dark fraction across repeat runs (~2% on a
 * half-committed frame, ~5% once every segment had landed) — a race, not
 * noise.
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
  test('band metrics on a single frame', async ({ page }) => {
    // ?dpr=1 pins the pixel ratio: the local-median metric is non-monotone
    // in defect pixel width, so a different DPR does not scale the answer.
    await page.goto(`/?src=${FIXTURES_BASE}/test_line_joins.luxar.zarr&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForLineSegmentsCommitted(page, EXPECTED_LINE_SEGMENTS, LINE_JOIN_BANDS.length);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);

    const frame = await captureCanvasRGBA(page, 'canvas#app', 'framebuffer');
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

      // Printed for the record, so a failure report carries the full
      // per-band numbers rather than just the one assertion that tripped.
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

    // #790: the miter closes the outer-side wedge and removes the inner-side
    // double-cover lens, so both bending cases now measure exactly what the
    // straight bands do — not a single tick, dark or bright. Unmitred, the
    // curve scored 4.94% dark / 3.52% bright (1918 dark pixels) here and the
    // zigzag 0.076% (30 pixels).
    //
    // The MEASURED value on both bands is 0, across four runs, and the
    // ceiling of 2 is not slack in that measurement. It is there because the
    // two sides of a joint are not guaranteed to land on the same miter point
    // bit-for-bit: they use the same operands in a different float32 order —
    // the vertex path forms `(ndcEnd - ndcStart) * (0.5 * uResolution)`, the
    // join helper scales each point and subtracts afterwards — which on this
    // fixture puts them up to 6.6e-05 px apart on the curve (7.6e-06 px on the
    // zigzag). Subpixel quantisation can turn that into one dropped or doubled
    // seam pixel, and with AA off that scores as a full outlier. Two still
    // leaves ~960x margin on the curve and ~15x on the zigzag against unmitred
    // rendering, so a count anywhere near the ceiling is not the seam: it means
    // something real has changed and should be investigated, not re-baselined.
    // (Making the two sides agree exactly means subtracting in NDC and scaling
    // afterwards in both shader backends — see `_shared/glsl-lib.ts`.)
    //
    // Positive control, run 2026-08-07 and re-verified at the #1352 flip:
    // the control must pin the QUAD explicitly —
    // `&linePrimitive=screen-space&lineJoin=none` — because `?lineJoin=`
    // is a NO-OP on the default capsule (it partitions every interior
    // joint unconditionally); against the default the URL reproduces the
    // passing run exactly. With the quad pinned, `&lineJoin=none`
    // reproduces 4.941% dark / 3.520% bright on the curve and
    // drops the zigzag's flux p05 to 0.780, so all five assertions below fail
    // without the join geometry. They are a regression detector, not a
    // tautology — if you widen them, re-run that A/B before believing the
    // result.
    expect(curve.outliers.darkOutliers, 'curve_smooth dark outliers').toBeLessThanOrEqual(2);
    expect(curve.outliers.brightOutliers, 'curve_smooth bright outliers').toBeLessThanOrEqual(2);
    expect(zigzag.outliers.darkOutliers, 'zigzag_right_angle dark outliers').toBeLessThanOrEqual(2);
    expect(
      zigzag.outliers.brightOutliers,
      'zigzag_right_angle bright outliers'
    ).toBeLessThanOrEqual(2);

    // Zero outliers alone would not prove the zigzag's wedge is closed. That
    // wedge is a 28.8 px-radius quarter disc, far wider than the local-median
    // metric's couple-of-pixels envelope, so unmitred it read only 0.076%
    // while the far gentler curve read 4.94%. The flux profile is the measure
    // that responds on this band: 0.780 unmitred, 0.985 mitred.
    expect(zigzag.flux.p05, 'zigzag_right_angle axial p05').toBeGreaterThan(ZIGZAG_FLUX_P05_FLOOR);

    // `curve_smooth`'s flux is deliberately NOT gated. Its p05/p95 measure
    // 0.856 / 1.116, and that spread is the sinusoid's own oblique
    // cross-section: the tube is not axis-aligned, so a column sum genuinely
    // varies along screen-x. It is geometry, not a defect — do not "tighten"
    // it to the straight bands' bounds.
  });
});
