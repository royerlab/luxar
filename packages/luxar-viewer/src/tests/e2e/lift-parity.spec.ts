/**
 * Points vs their lifted-gsplat twin must render alike in EVERY blending mode.
 *
 * `add_points(substitutive_lod=…)` builds a MIXED ladder: `luxar.gsplats.lift`
 * turns the cloud into gsplats for the coarse levels, the finest level stays
 * the original Points node. If the two families don't agree on the quantity a
 * blending mode is a functional of, the ladder visibly changes character as it
 * switches levels — which is exactly what the CELLxGENE census demo showed.
 *
 * The fixture is a 4×2 grid (four point radii × {Points, lifted twin}) with
 * IDENTICAL cluster geometry per column, all under one `layer=True` group so a
 * single Blend control drives every node. It bakes an identity tone response
 * and a pinned camera (`ViewerConfig`) so this is real photometry and not a
 * reading through ACES.
 *
 * Four distinct defects have been measured with this shape — A/B/C from
 * VOLUMETRIC_BLENDING_SPEC.md (2026-08-02), D fixed by #1994:
 *   A  τ chord factor             volumetric only    1/(R·chord)       FIXED
 *   B  uncompensated 2D dilation  all sum modes      (σ_px²+d)/σ_px²   FIXED
 *   C  peak-vs-sum lift calib.    max/normal/opaque  1/(uRIF·σ)        OPEN
 *   D  opaque drops point alpha   opaque only        1/opacity         FIXED
 *
 * So the sum modes assert parity on the crop MEAN; the peak modes assert a
 * divergence of known MAGNITUDE and known DIRECTION on the crop PEAK. Effect C
 * makes the lifted gsplat brighter for max/normal. Under post-#1994 `opaque`,
 * depth-tested alpha-over makes the points peak grow with radius while the
 * gsplat peak stays bit-identical to `max`, so their ratio crosses parity and
 * becomes strongly dimmer at the coarse radii. A per-mode DIRECTION plus a
 * per-(mode, radius) EXPECTED magnitude with a relative band bracket that
 * divergence from every side, so an effect-C change goes red and names what
 * moved — see the DELETE-WHEN-IT-LANDS note at the assertion site.
 */

import { test, expect, type Page } from '@playwright/test';
import { captureElementScreenshot, openLayersPanel } from './helpers';

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('luxar-control-rail-hint-dismissed', '1');
  });
});

// Served by the E2E data server (playwright.config.ts webServer on :9000,
// rooted at the repo root) — NOT by Vite, which transforms the zarr JSON and
// fails the decode.
const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lift_parity.luxar.zarr';

/** Radii baked into the fixture, in column order (left → right). */
const RADII = [0.02, 0.05, 0.15, 0.4] as const;

/** World-space cell centres, matching `LIFT_PARITY_*` in generate_test_data.py. */
const COLUMN_X = [-7.5, -2.5, 2.5, 7.5] as const;
const ROW_Y = { points: 3.5, gsplats: -3.5 } as const;

/** Half-size of the crop around each projected cell centre, as a fraction of canvas height. */
const CROP_HALF_FRAC = 58 / 720;

/**
 * Parity band on the crop MEAN. The measured mean ratios sit at 0.96–0.99, so
 * the residual this has to accommodate is 1–4% and 10% is tight — the defects
 * it guards were 1.4×–70×.
 *
 * Do NOT read that residual as the mean-vs-peak gap: those are different sizes
 * by an order of magnitude. The sum-mode PEAK ratio is 0.66 at r=0.02 (printed
 * in the table below), which is 8-bit quantisation on the handful of pixels at
 * the very top of the blob, not a blob-edge effect — it is exactly why the sum
 * arms assert on the mean and the peak arms do not.
 */
const PARITY_TOLERANCE = 0.1;

/** Modes whose output is a functional of the sum-projected ray mass. */
const SUM_MODES = ['additive', 'luminous', 'volumetric'] as const;
/** Modes that use peak projection — effect C is not yet calibrated. */
const PEAK_MODES = ['max', 'normal', 'opaque'] as const;

/** Radius indices whose measured divergence is strong and has one stable direction per mode. */
const PEAK_DIVERGENCE_INDICES: Record<(typeof PEAK_MODES)[number], readonly [number, number]> = {
  max: [0, 1],
  normal: [0, 1],
  opaque: [2, 3],
};

/**
 * How far apart the two families are under peak projection, per mode, at its
 * two asserted radii. The quantity is the direction-free
 * magnitude `max(ratio, 1/ratio)`:
 *   max      r=0.02 / 0.05  30.7× / 12.2×  brighter
 *   normal   r=0.02 / 0.05  29.7× /  7.8×  brighter
 *   opaque   r=0.15 / 0.40   4.57× / 18.1×  dimmer
 *
 * These are measurements, not budgets — `max` reproduces the analytic
 * 1/(uRIF·σ) to ~1% (see the cross-check at the assertion site), so the band
 * around them below can afford to be narrow.
 *
 * RE-MEASURE ALL SIX after any change to the `LIFT_PARITY_*` fixture constants
 * (generate_test_data.py:2316 — `LIFT_PARITY_OPACITY` above all, whose own
 * comment invites tuning it), to the radii or blob geometry, to the pinned
 * camera, or to the viewport. Only `max` can be re-derived on paper (analytic,
 * 1/(uRIF·σ)); `normal` and post-#1994 `opaque` are measurements because their
 * points and gsplats composite differently.
 * Raising the opacity from 0.003 to 0.01 turns four of these six cells red with
 * a message blaming a shader regression that did not happen.
 */
const PEAK_DIVERGENCE_EXPECTED: Record<(typeof PEAK_MODES)[number], readonly [number, number]> = {
  max: [30.7, 12.2],
  normal: [29.7, 7.8],
  opaque: [4.57, 18.1],
};

/**
 * Relative half-width of the band around each of those. The predicate is
 *
 *     expected / (1 + BAND)  <  magnitude  <  expected × (1 + BAND)
 *
 * i.e. a factor of 1.5 either way: `max` at r=0.02 must land in [20.5, 46.1],
 * at r=0.05 in [8.1, 18.3]; `normal` in [19.8, 44.6] and [5.2, 11.7]; `opaque`
 * at r=0.15 in [3.0, 6.9] and at r=0.40 in [12.1, 27.2].
 *
 * This replaces a single global (3, 100) window — 33× wide end to end, 97 units
 * of room at EVERY cell, around numbers this file certifies to ~1%. These bands
 * admit between 3.8 units (`opaque` r=0.15) and 25.6 (`max` r=0.02), i.e. ~8.4×
 * less room at a typical cell (geometric mean of the six widths), and unlike
 * the window they are centred on what was actually measured there. That is the
 * difference between a bracket and a sanity check. Worked example of what the
 * window let through: `shader-glsl.ts:337` is the peak
 * branch, `vAmplitude2D = aAmplitude * nearFade`, and `sigmaRay` is initialised
 * to 1.0 at :270 and only ever overwritten inside the SUM branch — so a
 * one-token edit to `aAmplitude * nearFade * sigmaRay * uRayIntegralFactor`
 * multiplies every gsplat peak by uRIF = 2.433 and touches no sum mode. Every
 * resulting magnitude stayed inside (3, 100) with its direction unchanged, so
 * all seven tests stayed green on a 2.43× regression in the exact quantity this
 * spec exists to pin. Against these bands it fails four of the six asserted
 * cells: `max` r=0.02 goes 30.7 → 74.6, outside [20.5, 46.1]; `opaque` r=0.15
 * goes 4.57 → 1.88, outside [3.0, 6.9] (and `max` r=0.05 → 29.6, `opaque`
 * r=0.40 → 7.44).
 *
 * Why a factor of 1.5 and not tighter: `normal`'s gsplat peak is already
 * saturating toward 1.0 — 0.855 at r=0.05 and 0.913 at r=0.02. A driver that
 * pushes r=0.05 the rest of the way to 1.0 moves THAT cell by up to ~17% with
 * nothing actually wrong; r=0.02 can move by ~9.5%, still under the 17% that
 * sets the band. One ~17% cell is the widest such move, and 1.5× clears it with
 * room.
 *
 * And `normal`'s UPPER edge is structurally dead — do not assume all six checks
 * are two-sided. Its points peak is 0.0307 / 0.1094 and the gsplat peak cannot
 * exceed 1.0, so the measurable ratio can never exceed 32.6× / 9.14×, inside
 * [19.8, 44.6] and [5.2, 11.7] no matter what breaks. That is exactly why the
 * uRIF mutation above survives on `normal` and is caught on the other two.
 */
const PEAK_DIVERGENCE_BAND = 0.5;

/**
 * Which way each peak mode diverges. This is NOT the sign the old assertion
 * assumed (it tested `> 1 + PARITY_TOLERANCE` for all three):
 *
 *   max, normal  effect C alone. The gsplat peak branch renders
 *                a_lift = opacity/(uRIF·σ) without the sum branch's
 *                `rayIntegrationBoost · dilationCompensation`, so the lifted
 *                twin is genuinely BRIGHTER, by 1/(uRIF·σ).
 *   opaque       #1994 restored the points' alpha-over photometry while keeping
 *                depth writes. The gsplat peak remains effect C, bit-identical
 *                to `max`; the points peak grows with radius under depth-tested
 *                alpha-over. The ratio therefore crosses 1 between r=0.02 and
 *                r=0.05, then the lifted gsplat reads increasingly DIMMER.
 */
const PEAK_DIVERGENCE_DIRECTION: Record<(typeof PEAK_MODES)[number], 'brighter' | 'dimmer'> = {
  max: 'brighter',
  normal: 'brighter',
  opaque: 'dimmer',
};

/**
 * sRGB code point (0…255) → linear channel value.
 *
 * The in-page measurement below carries its own copy — a `page.evaluate` body
 * is serialised to the browser and cannot close over module scope — so this
 * one exists purely to derive `COVERAGE_FLOOR` from the same transfer function
 * instead of hardcoding a magic float.
 */
function toLinear(v: number): number {
  const c = v / 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/**
 * A crop pixel counts as COVERED above ≈3.0e-4 — the LUMINANCE of a NEUTRAL
 * pixel one sRGB code point above black.
 *
 * The predicate is strict (`lum > floor`) and a neutral code-1 pixel evaluates
 * to EXACTLY this value (the Rec.709 weights sum to 1), so it does NOT clear
 * the bar: the smallest neutral that counts is code point 2.
 *
 * It is a per-channel linear value compared against a Rec.709 luminance, so it
 * is only "the smallest thing an 8-bit screenshot can encode" for grey: a
 * red-only pixel needs code point ≥ 5 and a blue-only one ≥ 14 to clear it.
 * This fixture is white-on-black, so neutral is the right calibration. The
 * predicate then reads as "this pixel is not background" and says nothing about
 * how bright the cell is — which is the point: it separates the geometric
 * question (how much of the crop did this family fill?) from the photometric
 * one.
 */
const COVERAGE_FLOOR = toLinear(1);

/**
 * How long the canvas gets to become screenshot-ready, for ONE measurement.
 *
 * The global `actionTimeout` is 10 s, and the first measurement after a
 * blending-mode switch follows a program recompile / TSL rebuild across all
 * eight nodes. On a loaded machine Playwright's in-page polling starves for
 * longer than that while the renderer holds the main thread — the element is
 * visible by every CSS measure the whole time.
 *
 * `cellLuminances` passes this to BOTH of its waits, and the SCREENSHOT is the
 * stricter one: `locator.screenshot()` is an action, so it runs its own
 * actionability check — Visible AND Stable, i.e. two consecutive animation
 * frames with an unchanged box — on `actionTimeout`. Budgeting only the
 * `waitFor` buys nothing, because that one clears on the first scheduling gap
 * and the screenshot then burns its un-raised 10 s waiting for two clean
 * frames: the same false failure, which has nothing to do with what this spec
 * measures.
 */
const CANVAS_READY_TIMEOUT = 30000;

/**
 * What one cell crop reports.
 *
 * `mean` is coverage × value and is what the parity ratios have always used;
 * on its own it cannot tell "the gsplat is dimmer" from "the gsplat covers
 * fewer pixels". `peak` is value alone (blind to coverage) and `covered` is
 * coverage alone (blind to value), so the three together decompose the ratio.
 */
interface CellMetrics {
  mean: number;
  peak: number;
  covered: number;
}

/** Canvas-normalised cell centre, so the crop survives DPR and element offset. */
interface Cell {
  cell: string;
  u: number;
  v: number;
}

/** Project each cell's world centre to canvas-normalised [0,1] coordinates. */
async function cellCentres(page: Page): Promise<Cell[]> {
  return page.evaluate(
    ({ columnX, rowY }) => {
      const cam = (window as any).__luxarDebug.app.sceneManager.camera;
      const mul = (e: number[], a: { x: number; y: number; z: number }) => ({
        x: e[0] * a.x + e[4] * a.y + e[8] * a.z + e[12],
        y: e[1] * a.x + e[5] * a.y + e[9] * a.z + e[13],
        z: e[2] * a.x + e[6] * a.y + e[10] * a.z + e[14],
        w: e[3] * a.x + e[7] * a.y + e[11] * a.z + e[15],
      });
      const project = (x: number, y: number, z: number) => {
        const view = mul(Array.from(cam.matrixWorldInverse.elements) as number[], { x, y, z });
        const clip = mul(Array.from(cam.projectionMatrix.elements) as number[], view);
        return { u: (clip.x / clip.w) * 0.5 + 0.5, v: 0.5 - (clip.y / clip.w) * 0.5 };
      };
      const out: Array<{ cell: string; u: number; v: number }> = [];
      columnX.forEach((cx: number, i: number) => {
        out.push({ cell: `pts_r${i}`, ...project(cx, rowY.points, 0) });
        out.push({ cell: `gsp_r${i}`, ...project(cx, rowY.gsplats, 0) });
      });
      return out;
    },
    { columnX: [...COLUMN_X], rowY: ROW_Y }
  );
}

/**
 * Open the Layers panel and wait for the Blend control to exist.
 *
 * `openLayersPanel` presses `l`, which needs canvas focus and occasionally
 * loses the race against this fixture's eight-node load. Retrying — and falling
 * back to the rail button — keeps the suite from failing on panel choreography
 * that has nothing to do with what it measures.
 */
async function openBlendControl(page: Page): Promise<void> {
  const blendVisible = async (): Promise<boolean> =>
    page.evaluate(() =>
      Array.from(document.querySelectorAll('select')).some((s) =>
        Array.from(s.options).some((o) => o.value === 'volumetric')
      )
    );

  for (let attempt = 0; attempt < 3; attempt++) {
    if (await blendVisible()) return;
    await openLayersPanel(page).catch(() => {});
    if (await blendVisible()) return;
    const rail = page.locator('[title*="Layers" i]').first();
    if (await rail.count()) await rail.click().catch(() => {});
    await page.waitForTimeout(1000);
  }
  expect(await blendVisible(), 'the Layers panel Blend control must be reachable').toBe(true);
}

/** Drive the layer's single Blend control. */
async function setBlendingMode(page: Page, mode: string): Promise<void> {
  const applied = await page.evaluate((m) => {
    const sel = Array.from(document.querySelectorAll('select')).find((s) =>
      Array.from(s.options).some((o) => o.value === 'volumetric')
    );
    if (!sel) return false;
    sel.value = m;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }, mode);
  expect(applied, 'the Layers panel Blend control must be present').toBe(true);
  // A mode switch can recompile the program / rebuild the TSL graph.
  await page.waitForTimeout(1500);
}

/**
 * LINEAR-luminance metrics for each cell crop of the current canvas.
 *
 * Decoding happens IN-PAGE off a data URL (the pattern `samplePixelsAt` in
 * helpers.ts uses) so the suite needs no Node image decoder. Screenshots are
 * sRGB-encoded — linearising before averaging is what makes the ratios mean
 * anything.
 */
async function cellLuminances(page: Page, centres: Cell[]): Promise<Map<string, CellMetrics>> {
  // By id, not by tag: the viewer also mounts a hidden 0×0 `.luxar-perf__graph`
  // canvas (the performance overlay's graph). It happens to come second in DOM
  // order today, so a bare `canvas` locator's `.first()` picks the right one by
  // accident — if the control rail ever moved ahead of `#app` this spec would
  // silently screenshot the hidden one.
  // The helper uses this as a readiness budget before the framebuffer capture;
  // the render/readback itself remains bounded by the test timeout.
  const png = await captureElementScreenshot(
    page,
    'canvas#app',
    'framebuffer',
    CANVAS_READY_TIMEOUT
  );
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;

  const measured = await page.evaluate(
    async ({ url, cells, halfFrac, floor }) => {
      const img = new Image();
      img.decoding = 'sync';
      const loaded = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('lift-parity: failed to decode screenshot'));
      });
      img.src = url;
      await loaded;

      const off = document.createElement('canvas');
      off.width = img.naturalWidth;
      off.height = img.naturalHeight;
      const ctx = off.getContext('2d', { willReadFrequently: true });
      if (!ctx) throw new Error('lift-parity: 2D context unavailable');
      ctx.drawImage(img, 0, 0);

      const toLinear = (v: number): number => {
        const c = v / 255;
        return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      };

      const half = Math.max(4, Math.round(halfFrac * off.height));
      const out: Array<[string, { mean: number; peak: number; covered: number }]> = [];
      for (const cell of cells) {
        const cx = Math.round(cell.u * off.width);
        const cy = Math.round(cell.v * off.height);
        const x0 = Math.max(0, Math.min(off.width - 1, cx - half));
        const y0 = Math.max(0, Math.min(off.height - 1, cy - half));
        const w = Math.min(half * 2, off.width - x0);
        const h = Math.min(half * 2, off.height - y0);
        const { data } = ctx.getImageData(x0, y0, w, h);
        let sum = 0;
        let peak = 0;
        let lit = 0;
        for (let i = 0; i < data.length; i += 4) {
          const lum =
            0.2126 * toLinear(data[i]) +
            0.7152 * toLinear(data[i + 1]) +
            0.0722 * toLinear(data[i + 2]);
          sum += lum;
          if (lum > peak) peak = lum;
          if (lum > floor) lit++;
        }
        const n = data.length / 4;
        out.push([cell.cell, { mean: sum / n, peak, covered: lit / n }]);
      }
      return out;
    },
    { url: dataUrl, cells: centres, halfFrac: CROP_HALF_FRAC, floor: COVERAGE_FLOOR }
  );

  return new Map(measured);
}

/** Both families' metrics for one radius, plus the two ratios they support. */
interface ParityRow {
  radius: number;
  pts: CellMetrics;
  gsp: CellMetrics;
  meanRatio: number;
  peakRatio: number;
}

/**
 * Per-radius gsplat-vs-points measurement for whatever mode is active.
 *
 * Pure measurement, no assertions: the caller logs the table BEFORE calling
 * `expectPointsRendered`, so a dead cell still gets diagnosed against the rows
 * that were measured either side of it rather than throwing mid-build and
 * printing nothing.
 */
async function parityMetrics(page: Page, centres: Cell[]): Promise<ParityRow[]> {
  const lum = await cellLuminances(page, centres);
  return RADII.map((radius, i) => {
    const pts = lum.get(`pts_r${i}`)!;
    const gsp = lum.get(`gsp_r${i}`)!;
    return { radius, pts, gsp, meanRatio: gsp.mean / pts.mean, peakRatio: gsp.peak / pts.peak };
  });
}

/**
 * The precondition every assertion below rests on: both ratios divide by the
 * points side, so a points cell that rendered nothing poisons the row.
 */
function expectPointsRendered(rows: ParityRow[]): void {
  for (const row of rows) {
    expect(row.pts.mean, `points cell r=${row.radius} must render something`).toBeGreaterThan(1e-6);
  }
}

/**
 * Dump the per-radius measurement so a failure is diagnosable from the CI log.
 *
 * A mean ratio alone cannot say WHY it moved: under `opaque` a gsplat quad
 * stamps one depth for its whole footprint, so a cluster's faint tails can
 * depth-reject the bright cores behind them — which shrinks coverage, not
 * photometry. Printing mean/peak/covered side by side separates the two
 * (issue #1929).
 */
function logMetricsTable(mode: string, rows: ParityRow[]): void {
  // Mean and peak are small linear luminances, so they only read in exponential
  // form; coverage and the ratios are O(1) and read better fixed.
  //
  // `frac` spells non-finite values out rather than printing them: a points
  // cell that rendered nothing gives meanRatio = Infinity, and 'inf' reads in
  // the table where a numeric formatting of it would not. Its padStart(8) is
  // set by the ' pts cov' header column below (nine characters including the
  // leading space), not by anything the function can emit.
  const lum = (v: number): string => v.toExponential(3).padStart(9);
  const frac = (v: number): string =>
    (Number.isFinite(v) ? v.toFixed(3) : Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf').padStart(
      8
    );
  console.log(`[lift-parity] ${mode}`);
  console.log(
    '[lift-parity]     r |  pts mean  pts peak  pts cov |' +
      '  gsp mean  gsp peak  gsp cov |    mean×    peak×'
  );
  for (const row of rows) {
    console.log(
      `[lift-parity]  ${row.radius.toFixed(2).padStart(4)} |` +
        ` ${lum(row.pts.mean)} ${lum(row.pts.peak)} ${frac(row.pts.covered)} |` +
        ` ${lum(row.gsp.mean)} ${lum(row.gsp.peak)} ${frac(row.gsp.covered)} |` +
        ` ${frac(row.meanRatio)} ${frac(row.peakRatio)}`
    );
  }
}

function expectParity(ratios: number[], mode: string, band = PARITY_TOLERANCE): void {
  ratios.forEach((ratio, i) => {
    expect(
      ratio,
      `r=${RADII[i]} in ${mode}: gsplat/points brightness ratio ${ratio.toFixed(3)} — ` +
        'the lift and the shaders disagree about the ray mass'
    ).toBeGreaterThan(1 - band);
    expect(ratio, `r=${RADII[i]} in ${mode}: ratio ${ratio.toFixed(3)}`).toBeLessThan(1 + band);
  });
}

test.describe('Lifted-gsplat / Points parity', () => {
  /**
   * Pin the viewport instead of inheriting `devices['Desktop Chrome']` from
   * playwright.config.ts. Two things here silently assume 1280×720:
   *   - `CROP_HALF_FRAC = 58/720`, whose numerator is a pixel count.
   *   - Every number in `PEAK_DIVERGENCE_EXPECTED`. At r=0.02 the points sprite
   *     is only ~1–2 px, right at the 1.5 px floor in
   *     `rendering/materials/point/shader-glsl.ts:187` (compensated by
   *     `sizeScale²` at :293). A SMALLER viewport pushes it under the floor and
   *     drops the points peak by up to 2.25×, taking the magnitude outside its
   *     band on a pure environment change with no product defect.
   */
  test.use({ viewport: { width: 1280, height: 720 } });

  /**
   * The config's 60 s per-test default is measurably too tight: in one
   * verification run the κ test took 56.6 s, and with a second worker hammering
   * the shared :9000 data server four of the seven tests died on the generic
   * `Test timeout of 60000ms exceeded` when the real cause was the scene-load
   * `waitForFunction`. The payoff of raising it is diagnostic, not capacity —
   * at 180 s those same failures surfaced as `page.waitForFunction: Timeout
   * 60000ms exceeded`, letting the INNER bounded waits (that one,
   * CANVAS_READY_TIMEOUT, `actionTimeout`) name their own cause.
   *
   * 180 s deliberately does NOT cover every inner bound saturating at once:
   * `goto` plus the two 60 s scene-load waits exhaust it exactly and the 3 s
   * settle tips it over. A run that saturates those is failing anyway and
   * should fail promptly rather than hold a worker for five minutes. 180 s
   * rather than 300 s is also a convention call — 120 s is the usual raise in
   * this suite and 300 s exists once, on a single-test describe — and across
   * seven tests at the daemon's `--retries 2` a wedged spec costs
   * 7 × 3 × 180 s = 63 min, against 105 min at 300 s.
   */
  test.describe.configure({ timeout: 180000 });

  test.beforeEach(async ({ page }) => {
    // ?dpr=1 pins the pixel ratio so the crops land on the same geometry.
    // `&no-opfs` on every load: this spec never asserts the L2 OPFS tier, and
    // automated Chromium's OPFS stalls systemically (10s per op — issue #1645),
    // starving scene readiness past the test budget. The circuit breaker only
    // helps un-flagged real sessions (it still pays ~3 timeouts per fresh page).
    await page.goto(`/?src=${FIXTURE}&debug&dpr=1&no-opfs`);
    await page.waitForFunction(() => !!(window as any).__luxarDebug, null, { timeout: 60000 });
    // All eight leaves must have materials AND committed geometry before
    // anything is measured. Waiting only for the material is not enough: a
    // placeholder mesh exists from the cheap-attach with instanceCount 0, so
    // the measurement could run against a half-loaded scene and land outside
    // the parity band. (That produced a real intermittent failure — `luminous`
    // failed in sequence but passed in isolation.)
    // `__luxarDebug.scene` and `app` attach AFTER the debug object itself, so
    // the predicate has to tolerate a partially-populated handle.
    await page.waitForFunction(
      () => {
        const dbg = (window as any).__luxarDebug;
        if (!dbg?.scene || !dbg?.app?.sceneManager?.camera) return false;
        let committed = 0;
        dbg.scene.traverse((o: any) => {
          if (o.material && o.name?.includes('_r') && (o.geometry?.instanceCount ?? 0) > 0)
            committed++;
        });
        return committed === 8;
      },
      null,
      { timeout: 60000 }
    );
    await page.waitForTimeout(3000);
    await openBlendControl(page);
  });

  for (const mode of SUM_MODES) {
    test(`${mode}: a lifted gsplat renders like the point it came from`, async ({ page }) => {
      const centres = await cellCentres(page);
      await setBlendingMode(page, mode);
      const rows = await parityMetrics(page, centres);
      logMetricsTable(mode, rows);
      expectPointsRendered(rows);
      expectParity(
        rows.map((row) => row.meanRatio),
        mode
      );
    });
  }

  for (const mode of PEAK_MODES) {
    test(`${mode}: lifted gsplat shows the known peak-projection divergence`, async ({ page }) => {
      const centres = await cellCentres(page);
      await setBlendingMode(page, mode);
      const rows = await parityMetrics(page, centres);
      logMetricsTable(mode, rows);
      expectPointsRendered(rows);
      const direction = PEAK_DIVERGENCE_DIRECTION[mode];
      // Asserted on the PEAK, not the crop mean. It is the quantity peak
      // projection calibrates (mechanism in PEAK_DIVERGENCE_DIRECTION; band in
      // PEAK_DIVERGENCE_BAND; why the sum arms use the mean instead, in
      // PARITY_TOLERANCE), and the cross-check those docblocks cite lives here:
      // under `max` the measured ratio reproduces the analytic 1/(uRIF·σ) to
      // ~1% — 30.8 / 12.3 / 4.11 / 1.54 predicted vs 30.66 / 12.15 / 4.05 /
      // 1.57 measured. The mean cannot be read that way, because `opaque`'s
      // depth-driven coverage collapse (0.34 → 0.05 over the radii, on peaks
      // bit-identical to `max`'s — see logMetricsTable) is geometry, not
      // photometry: the mean swallows it whole and the peak is immune to it.
      //
      // We assert the divergence is PRESENT rather than marking the whole test
      // `test.fail()`: an unconditional test.fail() silently accepts a blank
      // render, a shader-compile error, or a no-op mode switch — all of which
      // would leave the ratio ≈ 1 — as the "expected" failure.
      // `expectPointsRendered` above already fails loudly if the points family
      // renders nothing.
      //
      // DELETE THIS WHEN C LANDS. If the peak branch is calibrated without
      // changing the sum branch, max/normal collapse to parity and fail their
      // BAND floors; re-measure opaque, then fold each mode that reaches parity
      // into the parity loop. Direction is not sufficient at true parity: a
      // ratio of 0.999 still reads 'dimmer', so the BAND is the landing signal.
      //
      // `lift.py:347` bakes ONE amplitude that both branches consume. Changing
      // that amplitude to `opacity` also collapses max/normal, but breaks every
      // sum arm by uRIF·σ; the SUM_MODES tests and κ-response test must reject
      // that implementation rather than prompting a fold into the parity loop.
      //
      // max/normal use the two finest radii; their coarse divergence fades into
      // noise. Post-#1994 opaque does the opposite: r=0.05 is a dead 1.389× cell
      // whose band would contain parity, while r=0.15/r=0.40 are strong and
      // share one direction. Keep the selected cells explicit per mode.
      for (const [expectedIndex, i] of PEAK_DIVERGENCE_INDICES[mode].entries()) {
        const ratio = rows[i].peakRatio;
        // A gsplat cell that rendered NOTHING would sail through a `dimmer`
        // magnitude check (1/0 → ∞), so require it on screen first —
        // `expectPointsRendered` only vouches for the points side. The bar is
        // COVERAGE_FLOOR, the file's own "this pixel is not background": a
        // nominal 1e-6 is ~300× weaker and one stray code-1 pixel bleeding in
        // from a neighbouring cell (luminance 2.2e-5) would clear it.
        expect(
          rows[i].gsp.peak,
          `r=${RADII[i]} in ${mode}: gsplat cell must render something`
        ).toBeGreaterThan(COVERAGE_FLOOR);
        expect(
          ratio > 1 ? 'brighter' : 'dimmer',
          `r=${RADII[i]} in ${mode}: peak ratio ${ratio.toFixed(3)} — the lifted gsplat is ` +
            `expected to render ${direction} than its point`
        ).toBe(direction);
        // The magnitude has to sit within a factor of 1.5 either way of the
        // value measured for THIS (mode, radius) — see PEAK_DIVERGENCE_BAND
        // for why a single global window could not do this job.
        const magnitude = Math.max(ratio, 1 / ratio);
        const expected = PEAK_DIVERGENCE_EXPECTED[mode][expectedIndex];
        const low = expected / (1 + PEAK_DIVERGENCE_BAND);
        const high = expected * (1 + PEAK_DIVERGENCE_BAND);
        const message =
          `r=${RADII[i]} in ${mode}: peak ratio ${ratio.toFixed(3)} is ` +
          `${magnitude.toFixed(2)}× from parity, outside the band ` +
          `[${low.toFixed(1)}, ${high.toFixed(1)}]× around the expected ${expected}× — ` +
          'so the divergence has MOVED';
        expect(magnitude, message).toBeGreaterThan(low);
        expect(magnitude, message).toBeLessThan(high);
      }
    });
  }

  test('volumetric: both families respond to kappa identically', async ({ page }) => {
    // The stronger statement — not just "equal at the default κ" but "the whole
    // κ-response curve overlays". Before the convention fix the points row
    // needed κ ≈ 35× larger to reach the same darkening at R=0.05.
    const centres = await cellCentres(page);
    await setBlendingMode(page, 'volumetric');

    for (const kappa of [10, 100]) {
      const applied = await page.evaluate((k) => {
        let n = 0;
        (window as any).__luxarDebug.scene.traverse((o: any) => {
          if (o.material?.uniforms?.uAbsorption && o.name?.includes('_r')) {
            o.material.uniforms.uAbsorption.value = k;
            n++;
          }
        });
        (window as any).__luxarDebug.renderOnce?.();
        return n;
      }, kappa);
      expect(applied, 'κ must reach all eight probe materials').toBe(8);
      await page.waitForTimeout(1200);

      // Full decomposition here too: this is the test that measures twice, runs
      // at the widened band, and has historically flaked on a partially loaded
      // scene — so it is the one whose failures most need mean/peak/covered
      // side by side.
      const rows = await parityMetrics(page, centres);
      logMetricsTable(`volumetric κ=${kappa}`, rows);
      expectPointsRendered(rows);
      // Wider band: at high κ both rows are deep into the saturating part of
      // 1 − e^(−τ), where a small τ difference shows up amplified.
      expectParity(
        rows.map((row) => row.meanRatio),
        `volumetric κ=${kappa}`,
        2 * PARITY_TOLERANCE
      );
    }
  });
});
