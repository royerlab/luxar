/**
 * Line-joint artifact acceptance measurement (issues #780 / #785 / #790).
 *
 * Loads `test_line_joins.luxar.zarr` — five joint cases, each in its own
 * horizontal band of world Y — takes ONE canvas screenshot, and runs the two
 * pure metrics from `../helpers/line-join-metrics` over every band of that
 * single frame. Measuring one identical frame matters: the bands are
 * compared to each other, so re-screenshotting per band would let a stray
 * frame difference masquerade as a joint defect.
 *
 * Two things are being protected here:
 *
 *   - What is ALREADY correct must stay correct while the vertex stage is
 *     rewritten for the miter-join series: the straight bands carry zero
 *     dark outliers (#785) and a flat axial flux profile (#780 — no
 *     bead-chain dip at interior joints), and the nine-ray hub stays inside
 *     a small ceiling as the never-mitered control.
 *   - What is still BROKEN is recorded, not fixed. `curve_smooth` and
 *     `zigzag_right_angle` still show the uncovered outer-side wedge, so
 *     their dark-outlier fractions are only held under a documented
 *     ceiling. The measured numbers are printed for the record.
 *
 * The metrics run on the DISPLAY-encoded luminance of the composited frame,
 * not on linearised radiance. That is deliberate — both metrics are
 * relative and the encoding is monotone, so a defect still registers (a
 * linear 50% flux dip reads as roughly 0.73 through the sRGB transfer,
 * still far under the 0.9 floor the flat-profile guard uses). The fixture
 * pins an identity tone response and turns bloom / AA / noise off precisely
 * so nothing NON-monotone sits between the geometry and the measurement.
 */

import type { Page } from '@playwright/test';

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForRenderStable, assertNoShaderErrors } from './helpers';
import {
  measureAxialFlux,
  measureLocalMedianOutliers,
  rgbaToLuminance,
  type PixelRect,
} from '../helpers/line-join-metrics';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

/** A band's node name plus the world-space AABB its pixels live in. */
interface BandBox {
  /** Node name in the fixture scene. */
  name: string;
  /** World-space AABB, all at z = 0. */
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/**
 * The one place the band rectangles live on the TypeScript side.
 *
 * These mirror the table in `generate_line_joins_test()`
 * (`tests/fixtures/generate_test_data.py`) exactly — change one and you must
 * change the other, or the spec silently measures the wrong pixels. The X
 * range is inset 1.0 unit from the geometry ends so the free-end cap ramps
 * stay out of the measured region; `hub_9ray` is measured whole.
 */
const LINE_JOIN_BANDS: readonly BandBox[] = [
  { name: 'curve_smooth', xMin: -9.0, xMax: 9.0, yMin: 6.5, yMax: 9.5 },
  { name: 'zigzag_right_angle', xMin: -9.0, xMax: 9.0, yMin: 2.5, yMax: 5.5 },
  { name: 'straight_thin', xMin: -9.0, xMax: 9.0, yMin: -1.5, yMax: 1.5 },
  { name: 'straight_thick', xMin: -9.0, xMax: 9.0, yMin: -5.5, yMax: -2.5 },
  { name: 'hub_9ray', xMin: -1.5, xMax: 1.5, yMin: -9.5, yMax: -6.5 },
];

/**
 * Dark-outlier ceiling for the two bending cases. The issue measured ~1.34%
 * on `curve_smooth`; 3% leaves real headroom for GPU / driver rasterisation
 * differences without letting a genuine blow-up through.
 *
 * WHEN THE MITER JOIN LANDS, this ceiling drops to zero — a later part of
 * the #790 series is expected to replace it with `toBe(0)`, matching what
 * the straight bands already assert.
 */
const BEND_DARK_CEILING = 0.03;

/**
 * Ceiling for the never-mitered control. A degree-9 branch point has all
 * nine quads stacked around the hub with cap suppression at 0, so it has no
 * uncovered wedge to begin with; it is here to stay byte-stable.
 */
const HUB_DARK_CEILING = 0.01;

/**
 * Minimum inside-pixel count per band. A blank frame, a lost dataset or a
 * mislocated rectangle all collapse to 0 here, so every band assertion
 * below is anchored to real measured pixels rather than an empty mask.
 */
const MIN_INSIDE_PIXELS = 200;

/** A decoded canvas frame: raw RGBA bytes plus its pixel dimensions. */
interface CanvasFrame {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

/**
 * Screenshot the canvas once and hand back its raw RGBA bytes.
 *
 * The WebGL drawing buffer cannot be read back directly (the viewer runs
 * with `preserveDrawingBuffer: false`), so the composited element
 * screenshot is the only honest source — the same reasoning as
 * `helpers.ts::samplePixelsAt`. The PNG is decoded in-page and returned as
 * base64 RGBA so the whole frame crosses the bridge exactly once.
 */
async function captureCanvasFrame(page: Page): Promise<CanvasFrame> {
  const canvas = page.locator('canvas').first();
  await canvas.waitFor({ state: 'visible' });
  const png = await canvas.screenshot({ animations: 'disabled' });
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;

  const decoded = await page.evaluate(async (url: string) => {
    const img = new Image();
    img.decoding = 'sync';
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('line-join-artifact: screenshot decode failed'));
      img.src = url;
    });

    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (w <= 0 || h <= 0) {
      throw new Error(`line-join-artifact: empty screenshot ${w}x${h}`);
    }

    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const ctx = off.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('line-join-artifact: 2D context unavailable');
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, w, h).data;

    // Chunked: String.fromCharCode.apply blows the stack on a multi-MB buffer.
    let binary = '';
    const CHUNK = 0x8000;
    for (let i = 0; i < data.length; i += CHUNK) {
      const slice = data.subarray(i, i + CHUNK) as unknown as number[];
      binary += String.fromCharCode.apply(null, slice);
    }
    return { width: w, height: h, base64: btoa(binary) };
  }, dataUrl);

  const buf = Buffer.from(decoded.base64, 'base64');
  return {
    width: decoded.width,
    height: decoded.height,
    rgba: new Uint8ClampedArray(buf.buffer, buf.byteOffset, buf.byteLength),
  };
}

/** Normalised-device-coordinate bounding box of one band. */
interface NdcBox {
  name: string;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Project each band's world AABB through the LIVE camera and return its NDC
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
function ndcBoxToRect(box: NdcBox, frame: CanvasFrame): PixelRect {
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
    await page.goto(`/?src=${FIXTURES_BASE}/test_line_joins.luxar.zarr&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);
    await assertNoShaderErrors(page);

    const frame = await captureCanvasFrame(page);
    const luminance = rgbaToLuminance(frame.rgba, frame.width, frame.height);
    const ndcBoxes = await projectBands(page, LINE_JOIN_BANDS);
    expect(ndcBoxes).toHaveLength(LINE_JOIN_BANDS.length);

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
      // Every band's tube runs along the image X axis (the hub's rays are
      // radial, but its flux profile is recorded only, never asserted).
      const flux = measureAxialFlux(luminance, frame.width, frame.height, rect, 'x');
      byName.set(ndc.name, { rect, outliers, flux });

      // Recorded for the record — these numbers are the #790 baseline the
      // later parts of the series are judged against.
      console.log(
        `[line-join] ${ndc.name}: rect=${rect.x},${rect.y} ${rect.width}x${rect.height} ` +
          `inside=${outliers.insidePixels} ` +
          `dark=${outliers.darkOutliers} (${(outliers.darkFraction * 100).toFixed(3)}%) ` +
          `bright=${outliers.brightOutliers} (${(outliers.brightFraction * 100).toFixed(3)}%) ` +
          `worstDeficit=${outliers.worstDeficit.toFixed(1)} ` +
          `worstExcess=${outliers.worstExcess.toFixed(1)} | ` +
          `flux samples=${flux.samples} p05=${flux.p05.toFixed(3)} ` +
          `p50=${flux.p50.toFixed(3)} p95=${flux.p95.toFixed(3)} min=${flux.min.toFixed(3)}`
      );
    }

    // Every band must have measured real pixels — otherwise the frame was
    // blank or a rectangle landed off the geometry, and every assertion
    // below would pass vacuously.
    for (const band of LINE_JOIN_BANDS) {
      const measured = byName.get(band.name);
      expect(measured, `no measurement for band ${band.name}`).toBeDefined();
      expect(
        measured!.outliers.insidePixels,
        `band ${band.name} measured too few pixels`
      ).toBeGreaterThan(MIN_INSIDE_PIXELS);
    }

    const thin = byName.get('straight_thin')!;
    const thick = byName.get('straight_thick')!;
    const curve = byName.get('curve_smooth')!;
    const zigzag = byName.get('zigzag_right_angle')!;
    const hub = byName.get('hub_9ray')!;

    // #785: cap suppression makes a straight polyline's interior joints
    // invisible. Neither straight band may show a single dark tick.
    expect(thin.outliers.darkOutliers, 'straight_thin dark outliers').toBe(0);
    expect(thick.outliers.darkOutliers, 'straight_thick dark outliers').toBe(0);

    // #780: the thick band is 199 collinear segments, each far shorter than
    // one line width. A per-joint flux dip (the bead chain) would drag p05
    // toward 0.7; a healthy tube holds a flat profile at 1.0.
    expect(thick.flux.samples, 'straight_thick axial samples').toBeGreaterThan(100);
    expect(thick.flux.p50, 'straight_thick axial p50').toBeCloseTo(1, 6);
    expect(thick.flux.p05, 'straight_thick axial p05').toBeGreaterThan(0.9);
    expect(thick.flux.p95, 'straight_thick axial p95').toBeLessThan(1.1);

    // The never-mitered control: nine quads stacked around one hub vertex.
    expect(hub.outliers.darkFraction, 'hub_9ray dark fraction').toBeLessThan(HUB_DARK_CEILING);

    // #790, RECORDED not fixed: both bending cases still leave an uncovered
    // wedge on the outside of every turn. Asserting 0 here would fail today,
    // and pinning the exact buggy value would be just as wrong — so they are
    // only held under a documented ceiling that the miter-join part of the
    // series is expected to replace with `toBe(0)`.
    expect(curve.outliers.darkFraction, 'curve_smooth dark fraction').toBeLessThan(
      BEND_DARK_CEILING
    );
    expect(zigzag.outliers.darkFraction, 'zigzag_right_angle dark fraction').toBeLessThan(
      BEND_DARK_CEILING
    );
  });
});
