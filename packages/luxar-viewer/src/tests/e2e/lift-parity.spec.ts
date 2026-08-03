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
 * Three distinct defects have been measured with this shape — see
 * VOLUMETRIC_BLENDING_SPEC.md (2026-08-02):
 *   A  τ chord factor             volumetric only    1/(R·chord)       FIXED
 *   B  uncompensated 2D dilation  all sum modes      (σ_px²+d)/σ_px²   FIXED
 *   C  peak-vs-sum lift calib.    max/normal/opaque  1/(uRIF·σ)        OPEN
 *
 * So the sum modes assert parity; the peak modes are marked `test.fail()` —
 * expected to fail until C lands, and Playwright flags them the day it's fixed.
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
/** Modes that use peak projection — effect C, not yet calibrated. */
const PEAK_MODES = ['max', 'normal', 'opaque'] as const;

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
 * Mean LINEAR luminance in each cell crop of the current canvas.
 *
 * Decoding happens IN-PAGE off a data URL (the pattern `samplePixelsAt` in
 * helpers.ts uses) so the suite needs no Node image decoder. Screenshots are
 * sRGB-encoded — linearising before averaging is what makes the ratios mean
 * anything.
 */
async function cellLuminances(page: Page, centres: Cell[]): Promise<Map<string, number>> {
  const canvas = page.locator('canvas').first();
  await canvas.waitFor({ state: 'visible' });
  const png = await canvas.screenshot({ animations: 'disabled' });
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;

  const measured = await page.evaluate(
    async ({ url, cells, halfFrac }) => {
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
      const out: Array<[string, number]> = [];
      for (const cell of cells) {
        const cx = Math.round(cell.u * off.width);
        const cy = Math.round(cell.v * off.height);
        const x0 = Math.max(0, Math.min(off.width - 1, cx - half));
        const y0 = Math.max(0, Math.min(off.height - 1, cy - half));
        const w = Math.min(half * 2, off.width - x0);
        const h = Math.min(half * 2, off.height - y0);
        const { data } = ctx.getImageData(x0, y0, w, h);
        let sum = 0;
        for (let i = 0; i < data.length; i += 4) {
          sum +=
            0.2126 * toLinear(data[i]) +
            0.7152 * toLinear(data[i + 1]) +
            0.0722 * toLinear(data[i + 2]);
        }
        out.push([cell.cell, sum / (data.length / 4)]);
      }
      return out;
    },
    { url: dataUrl, cells: centres, halfFrac: CROP_HALF_FRAC }
  );

  return new Map(measured);
}

/** gsplat/points brightness ratio per radius for whatever mode is active. */
async function parityRatios(page: Page, centres: Cell[]): Promise<number[]> {
  const lum = await cellLuminances(page, centres);
  return RADII.map((radius, i) => {
    const pts = lum.get(`pts_r${i}`)!;
    const gsp = lum.get(`gsp_r${i}`)!;
    expect(pts, `points cell r=${radius} must render something`).toBeGreaterThan(1e-6);
    return gsp / pts;
  });
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
  test.beforeEach(async ({ page }) => {
    // ?dpr=1 pins the pixel ratio so the crops land on the same geometry.
    await page.goto(`/?src=${FIXTURE}&debug&dpr=1`);
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
      expectParity(await parityRatios(page, centres), mode);
    });
  }

  for (const mode of PEAK_MODES) {
    test(`${mode}: lifted gsplat parity (KNOWN FAILING — effect C)`, async ({ page }) => {
      // EXPECTED FAILURE until effect C lands. The lift equates SUM-projection
      // brightness (`a·σ·uRIF = opacity`); under peak projection the gsplat
      // reports raw `a_lift = opacity/(uRIF·σ)`, which diverges as 1/σ — 12× at
      // R=0.05, 39× at R=0.02. Unfixable in the lift (two constraints, one
      // left-hand side); it needs a peak-branch calibration. When someone fixes
      // it Playwright reports "expected to fail but passed" — delete this block
      // then and fold the mode into the SUM_MODES loop.
      //
      // MUST stay INSIDE the test body: `test.fail()` at describe scope
      // annotates EVERY test in the suite, which silently converts real
      // failures elsewhere into reported passes (caught by a tolerance
      // mutation — the whole suite passed at a 1e-6 band).
      test.fail();
      const centres = await cellCentres(page);
      await setBlendingMode(page, mode);
      expectParity(await parityRatios(page, centres), mode);
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

      // Wider band: at high κ both rows are deep into the saturating part of
      // 1 − e^(−τ), where a small τ difference shows up amplified.
      expectParity(
        await parityRatios(page, centres),
        `volumetric κ=${kappa}`,
        2 * PARITY_TOLERANCE
      );
    }
  });
});
