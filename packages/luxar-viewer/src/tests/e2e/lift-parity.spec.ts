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
 * VOLUMETRIC_BLENDING_SPEC.md (2026-08-02), D filed as issue #1993:
 *   A  τ chord factor             volumetric only    1/(R·chord)       FIXED
 *   B  uncompensated 2D dilation  all sum modes      (σ_px²+d)/σ_px²   FIXED
 *   C  peak-vs-sum lift calib.    max/normal/opaque  1/(uRIF·σ)        OPEN
 *   D  opaque drops point alpha   opaque only        1/opacity         OPEN
 *
 * So the sum modes assert parity on the crop MEAN; the peak modes assert a
 * divergence of known MAGNITUDE and known DIRECTION on the crop PEAK — effect C
 * alone for max/normal (the lifted gsplat is brighter), C compounded with #1993
 * for opaque, where the points lose all their alpha-carried photometry and the
 * gsplat therefore comes out dimmer. Direction, floor AND ceiling bracket that
 * divergence from every side, so whichever of the two open defects lands first
 * the assertion goes red and names what moved — see the DELETE-WHEN-IT-LANDS
 * note, which enumerates the three landings and the check that catches each.
 */

import { test, expect, type Page } from '@playwright/test';
import { openLayersPanel } from './helpers';

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
 * Parity band. The residual is a couple of percent (the lift matches the
 * additive PEAK while this measures a crop MEAN, and the two differ slightly at
 * the blob edge), so 10% is tight enough — the defects it guards were 1.4×–70×.
 */
const PARITY_TOLERANCE = 0.1;

/** Modes whose output is a functional of the sum-projected ray mass. */
const SUM_MODES = ['additive', 'luminous', 'volumetric'] as const;
/** Modes that use peak projection — effect C, not yet calibrated (opaque also carries #1993). */
const PEAK_MODES = ['max', 'normal', 'opaque'] as const;

/**
 * How far apart the two families must be under peak projection, in EITHER
 * direction — `max(ratio, 1/ratio)` has to clear this.
 *
 * Measured peak ratios at the two asserted radii (r=0.02, r=0.05):
 *   max      30.7× / 12.2×  brighter
 *   normal   29.7× /  7.8×  brighter
 *   opaque   10.8× / 27.0×  dimmer  (printed as 0.093 and 0.037)
 * The smallest magnitude anywhere in that set is 7.8× — `normal` at r=0.05,
 * where the gsplat peak is already saturating at 0.855 and so UNDERSTATES the
 * divergence — which leaves ~2.6× of headroom over this floor. The `> 1.1` it
 * replaces sat inside the parity band's own width and could not tell a real
 * divergence from measurement noise.
 */
const PEAK_DIVERGENCE_FLOOR = 3;

/**
 * Upper bound on that same magnitude, because a floor alone is one-sided.
 *
 * Effect C landing on its own would leave `opaque` carrying #1993's 1/uOpacity
 * and nothing else: ratio 1 × 0.003, still 'dimmer', still miles over the
 * floor — a 10.8× → 333× move in the exact quantity this test exists to pin,
 * passing silently. The ceiling is what fails on that.
 *
 * The largest magnitude measured anywhere is 30.7× (`max` at r=0.02), so 100
 * keeps ~3.3× of headroom over today's worst case while sitting well below the
 * 333× it has to catch.
 */
const PEAK_DIVERGENCE_CEILING = 100;

/**
 * Which way each peak mode diverges. This is NOT the sign the old assertion
 * assumed (it tested `> 1 + PARITY_TOLERANCE` for all three):
 *
 *   max, normal  effect C alone. The gsplat peak branch renders
 *                a_lift = opacity/(uRIF·σ) without the sum branch's
 *                `rayIntegrationBoost · dilationCompensation`, so the lifted
 *                twin is genuinely BRIGHTER, by 1/(uRIF·σ).
 *   opaque       effect C multiplied by #1993 on the POINTS side: three.js
 *                disables blending outright for `NormalBlending` +
 *                `transparent: false`, so the emitted alpha never reaches the
 *                framebuffer. Points carry ALL their photometry in alpha and
 *                lose opacity, falloff, sub-pixel compensation and near-fade
 *                (every covered pixel lands at 1.0); gsplats premultiply theirs
 *                into RGB and lose nothing. Net: the gsplat reads DIMMER. The
 *                arithmetic closes — multiplying the opaque peak ratios by
 *                1/0.003 (the fixture's authored uOpacity) recovers `max`'s to
 *                within 3% at all four radii, i.e. one radius-independent
 *                factor and nothing density- or radius-dependent.
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
  const canvas = page.locator('canvas#app');
  // Both of these need the budget, and the screenshot — Visible AND Stable —
  // is the stricter of the two; see CANVAS_READY_TIMEOUT.
  await canvas.waitFor({ state: 'visible', timeout: CANVAS_READY_TIMEOUT });
  const png = await canvas.screenshot({ animations: 'disabled', timeout: CANVAS_READY_TIMEOUT });
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
  const lum = (v: number): string => v.toExponential(3).padStart(9);
  const frac = (v: number): string => v.toFixed(3).padStart(7);
  console.log(`[lift-parity] ${mode}`);
  console.log(
    '[lift-parity]     r |  pts mean  pts peak pts cov |' +
      '  gsp mean  gsp peak gsp cov |   mean×   peak×'
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
   * The config's 60 s per-test default does not fit this fixture, and
   * `beforeEach` runs inside it. Setup alone is up to two 60 s
   * `waitForFunction` budgets (the debug handle, then eight committed leaves)
   * plus the 3 s settle plus `openBlendControl`'s three retries (~5 s) = 128 s.
   * The κ test then measures TWICE, and each `cellLuminances` can spend
   * CANVAS_READY_TIMEOUT on the visibility wait and CANVAS_READY_TIMEOUT again
   * on the screenshot: 2 × (30 + 30 + 1.2 s settle) = 122.4 s, after a 1.5 s
   * mode switch. Worst case ≈ 252 s.
   *
   * Not a theoretical wall: on a loaded box (1-minute load average 47 on 16
   * cores) the scene load took ~45 s and the κ test 56.6 s of the 60 s default
   * — so the default is tight even on a good day, and in the pathological case
   * CANVAS_READY_TIMEOUT exists for it dies on a bare `Test timeout of 60000ms
   * exceeded` instead of the actionable locator message. 300 s covers the
   * arithmetic with ~19% of headroom and matches frame-pacing.spec.ts, the
   * in-tree `describe.configure` precedent at this value.
   */
  test.describe.configure({ timeout: 300000 });

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
      // Asserted on the PEAK, not on the crop mean, because the peak is the
      // quantity peak projection actually calibrates: the gsplat peak branch
      // renders a_lift = opacity/(uRIF·σ) unboosted — without the sum branch's
      // `rayIntegrationBoost · dilationCompensation` — and under `max` — whose
      // points peak is a flat authored opacity at every radius — the measured
      // ratio reproduces the analytic 1/(uRIF·σ) to ~1% (30.8 / 12.3 / 4.11 /
      // 1.54 predicted vs 30.66 / 12.15 / 4.05 / 1.57 measured). The mean cannot
      // be read that way: `opaque` is the only peak mode with depthWrite, and a
      // gsplat quad stamps one centre depth across its whole truncated
      // footprint, so faint tails depth-reject the bright cores behind them and
      // gsplat coverage collapses 0.34 → 0.05 over the radii on peaks that are
      // bit-identical to `max`'s. That is geometry, not photometry — the mean
      // (coverage × value) swallows it whole and the peak is immune to it.
      //
      // The SUM_MODES tests and the κ test stay on the MEAN for the mirror
      // reason: their peak ratio is 0.66 at r=0.02 — a handful of 8-bit-
      // quantised pixels, nowhere near surviving the 10% parity band — while
      // their mean ratio sits at 0.96–0.99.
      //
      // We assert the divergence is PRESENT rather than marking the whole test
      // `test.fail()`: an unconditional test.fail() silently accepts a blank
      // render, a shader-compile error, or a no-op mode switch — all of which
      // would leave the ratio ≈ 1 — as the "expected" failure.
      // `expectPointsRendered` above already fails loudly if the points family
      // renders nothing.
      //
      // DELETE THIS WHEN IT LANDS — but not all of it at once. Three landings
      // are possible and a different check catches each; `opaque`'s ratio is
      // effect C times #1993's uOpacity = 0.003, so run that product forward:
      //   C alone      max/normal collapse to ≈ 1 and fail on DIRECTION (and on
      //                the floor). `opaque` keeps #1993 alone — 1 × 0.003, so
      //                still 'dimmer' and still far over the floor — and its
      //                magnitude jumps 10.8× → 333×, which fails on the
      //                CEILING. Fold max/normal into the SUM_MODES loop.
      //   #1993 alone  `opaque` recovers its 1/0.003 and flips to 'brighter' at
      //                ≈ 30.7×, failing on DIRECTION. max/normal unchanged.
      //   both         all three reach parity and fail on direction and floor:
      //                delete these assertions and fold every peak mode into
      //                the SUM_MODES loop.
      for (const i of [0, 1]) {
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
        const magnitude = Math.max(ratio, 1 / ratio);
        expect(
          magnitude,
          `r=${RADII[i]} in ${mode}: peak ratio ${ratio.toFixed(3)} is only ` +
            `${magnitude.toFixed(2)}× from parity — the smallest divergence ever measured ` +
            'at these radii is 7.8×'
        ).toBeGreaterThan(PEAK_DIVERGENCE_FLOOR);
        expect(
          magnitude,
          `r=${RADII[i]} in ${mode}: peak ratio ${ratio.toFixed(3)} is ` +
            `${magnitude.toFixed(2)}× from parity — larger than the 30.7× worst case ever ` +
            'measured, so the divergence has MOVED (effect C landing alone puts `opaque` at 333×)'
        ).toBeLessThan(PEAK_DIVERGENCE_CEILING);
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
