/**
 * TEMPORARY verification probe (campaign 4, iteration 9) — DELETE ME.
 *
 * (a) ORTHO camera at unit extremes (×1e-6, ×1e6): switch to ortho
 *     in-page (V twice: orbit → fly → ortho), apply the in-page scale
 *     recipe (scene.scale + positionBounds scaling + boundsCache
 *     invalidate + camera/target/near/far compensation + ortho frustum
 *     scaling + radiusScale folding + uNearCull refresh via
 *     updateMaterialsForCurrentCamera), assert coverage parity with the
 *     unscaled ortho baseline and zero GL errors.
 *
 *     Coverage is measured on a CENTER CROP of the canvas that excludes
 *     the DOM UI chrome (left rail/panels, bottom dimension panel) —
 *     element screenshots include overlaying DOM, and the UI contributes
 *     ~0.28 coverage that would swamp the ~0.01-0.02 point signal.
 *
 * (b) PICKING at ×1e-6 (perspective): labelled fixture, apply scale
 *     recipe, hover a covered pixel, assert a pick result arrives via
 *     the public `selection` embedder event with valid nodeName +
 *     elementIndex, and PickingSystem diagnostics advanced.
 */

import { test, expect } from './fixtures';
import type { Page } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForNextRender,
  focusCanvas,
  getWebGLErrors,
} from './helpers';

const POINTS_DATASET =
  'http://localhost:9000/datasets/examples/scene_dimensions_example.luxar.zarr';
const LABELLED_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_labelled_points.luxar.zarr';

/** Crop that avoids the left rail/info panels, bottom nav panel and toasts. */
const CROP = { x0: 0.36, y0: 0.05, x1: 0.95, y1: 0.7 };

/** V twice: orbit → fly → ortho. */
async function switchToOrtho(page: Page): Promise<void> {
  await page.keyboard.press('v');
  await waitForNextRender(page);
  await page.keyboard.press('v');
  await waitForNextRender(page);
  const controlType = await page.evaluate(() => {
    const debug = (window as any).__luxarDebug;
    return debug.controls?.getControlType?.();
  });
  expect(controlType).toBe('ortho');
}

/**
 * In-page scale recipe: pretend the whole scene lives at `s`× the
 * original unit scale and compensate everything the renderer derives
 * from world units. A REAL dataset at the new unit scale would carry
 * proportionally scaled radii attributes too, so `s` is folded into
 * every point material's radiusScale (render + pick). GSplat covariance
 * goes through mat3(modelViewMatrix) in the shader, so it picks up the
 * scene scale automatically.
 */
async function applyScaleRecipe(page: Page, s: number): Promise<void> {
  await page.evaluate((scale) => {
    const dbg = (window as any).__luxarDebug;
    const sm = dbg.app.sceneManager;
    const scene = dbg.scene;
    const cam = dbg.camera;
    const controlsMgr = dbg.controls;

    // 1. Scale the scene graph.
    scene.scale.multiplyScalar(scale);
    scene.updateMatrixWorld(true);

    // 2. Scale positionBounds metadata (feeds boundsCache → nearCull +
    //    dynamic clipping) and invalidate the cache.
    scene.traverse((o: any) => {
      const pb = o.userData?.positionBounds;
      if (pb?.min && pb?.max) {
        pb.min = pb.min.map((v: number) => v * scale);
        pb.max = pb.max.map((v: number) => v * scale);
      }
    });
    sm.boundsCache.invalidate();

    // 3. Radii are stored in data units: fold the scale into every
    //    point material's radiusScale. Geometry may be shared between
    //    main and pick nodes — multiply each geometry's userData once.
    const seen = new Set<any>();
    const bumpGeometry = (g: any) => {
      if (!g || seen.has(g)) return;
      seen.add(g);
      g.userData = g.userData || {};
      g.userData.radiusScale = (g.userData.radiusScale ?? 1.0) * scale;
    };
    const applyRadius = (o: any) => {
      if (o?.material && typeof o.material.updateRadiusScale === 'function') {
        bumpGeometry(o.geometry);
        o.material.updateRadiusScale(o.geometry?.userData?.radiusScale ?? scale);
      }
    };
    scene.traverse(applyRadius);
    const ps = dbg.getPickingSystem?.();
    if (ps?.nodeMap) {
      ps.nodeMap.forEach((entry: any) => {
        applyRadius(entry.pick);
      });
    }

    // 4. Camera + controls compensation.
    const target = controlsMgr.getFocusTarget();
    target.multiplyScalar(scale);
    cam.position.multiplyScalar(scale);
    cam.near *= scale;
    cam.far *= scale;
    if (cam.isOrthographicCamera) {
      cam.left *= scale;
      cam.right *= scale;
      cam.top *= scale;
      cam.bottom *= scale;
    }
    cam.updateProjectionMatrix();
    controlsMgr.setTarget(target);
    controlsMgr.reinitialize();

    // 5. Controls scale-derived limits (scene-relative floors).
    const cc = controlsMgr.getControls();
    if (cc && typeof cc.minDistance === 'number') {
      if (cc.minDistance > 0) cc.minDistance *= scale;
      if (cc.maxDistance > 0 && isFinite(cc.maxDistance)) cc.maxDistance *= scale;
    }
    if (controlsMgr.storedDistanceLimits) {
      controlsMgr.storedDistanceLimits.min *= scale;
      controlsMgr.storedDistanceLimits.max *= scale;
    }
    if (typeof controlsMgr.sceneScale === 'number' && controlsMgr.sceneScale > 0) {
      controlsMgr.sceneScale *= scale;
    }

    // 6. Push uNearCull (+ ortho frustumHeight) into all materials.
    sm.updateMaterialsForCurrentCamera();
    dbg.renderOnce();
  }, s);
  await waitForNextRender(page, 4);
}

/**
 * Coverage over a fractional crop of the canvas element screenshot.
 * (The shared getElementPixelStats has no crop and the DOM UI chrome
 * overlaying the canvas swamps the point signal.)
 */
async function croppedCoverage(
  page: Page,
  savePath?: string
): Promise<{ frac: number; nonBlack: number; total: number }> {
  const buf = await page
    .locator('canvas')
    .first()
    .screenshot(savePath ? { path: savePath } : {});
  const dataUrl = `data:image/png;base64,${buf.toString('base64')}`;
  return await page.evaluate(
    async ({ url, crop }) => {
      const img = new Image();
      img.decoding = 'sync';
      const loaded = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('croppedCoverage: decode failed'));
      });
      img.src = url;
      await loaded;
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const off = document.createElement('canvas');
      off.width = w;
      off.height = h;
      const ctx = off.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      const sx = Math.floor(crop.x0 * w);
      const sy = Math.floor(crop.y0 * h);
      const sw = Math.floor((crop.x1 - crop.x0) * w);
      const sh = Math.floor((crop.y1 - crop.y0) * h);
      const pixels = ctx.getImageData(sx, sy, sw, sh).data;
      let nonBlack = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 10) nonBlack++;
      }
      return { frac: nonBlack / (sw * sh), nonBlack, total: sw * sh };
    },
    { url: dataUrl, crop: CROP }
  );
}

for (const s of [1e-6, 1e6]) {
  test(`iter9a: ortho coverage parity at scene scale ×${s}`, async ({ page }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.goto(`/?src=${POINTS_DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await focusCanvas(page);

    await switchToOrtho(page);
    await waitForNextRender(page, 4);
    const baseline = await croppedCoverage(page, `delme/zz-probe4-evidence/iter9a-baseline-${s}.png`);
    expect(baseline.frac).toBeGreaterThan(0.002);

    await applyScaleRecipe(page, s);
    const scaled = await croppedCoverage(page, `delme/zz-probe4-evidence/iter9a-scaled-${s}.png`);

    console.log(
      `[probe4-iter9a] scale=${s} baseline=${baseline.frac.toFixed(5)} (${baseline.nonBlack}px) scaled=${scaled.frac.toFixed(5)} (${scaled.nonBlack}px) ratio=${(scaled.frac / baseline.frac).toFixed(3)}`
    );

    // Coverage parity: same framing, same content → same pixels lit.
    expect(scaled.frac).toBeGreaterThan(0.002);
    const ratio = scaled.frac / baseline.frac;
    expect(ratio).toBeGreaterThan(0.7);
    expect(ratio).toBeLessThan(1.4);

    const glErrors = await getWebGLErrors(page);
    expect(glErrors).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
}

test('iter9b: picking works at scene scale ×1e-6 (perspective)', async ({ page }) => {
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));

  await page.goto(`/?src=${LABELLED_FIXTURE}&debug`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);

  // Record every public selection event.
  await page.evaluate(() => {
    (window as any).__probeSelections = [];
    (window as any).__luxarDebug.app.on('selection', (sel: unknown) => {
      (window as any).__probeSelections.push(sel);
    });
  });

  // Park the cursor away from content first.
  await page.mouse.move(5, 5);
  await page.waitForTimeout(200);

  await applyScaleRecipe(page, 1e-6);

  const cover = await croppedCoverage(page, 'delme/zz-probe4-evidence/iter9b-scaled-1e-6.png');
  expect(cover.nonBlack).toBeGreaterThan(0);

  const centre = await page.evaluate(() => {
    const canvas = (window as any).__luxarDebug.renderer.domElement as HTMLCanvasElement;
    const rect = canvas.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  });

  const before = await page.evaluate(() =>
    (window as any).__luxarDebug.getPickingSystem()?.getDiagnostics()
  );
  expect(before).toBeTruthy();

  // Hover the centre and let the settle scheduler fire the pick.
  await page.mouse.move(centre.x, centre.y);
  await page.waitForTimeout(400);

  const after = await page.evaluate(() =>
    (window as any).__luxarDebug.getPickingSystem()?.getDiagnostics()
  );
  expect(after.lastPickFiredTime).toBeGreaterThan(before.lastPickFiredTime);

  // A non-null selection with a valid nodeName + elementIndex must have arrived.
  const selections = await page.evaluate(() => (window as any).__probeSelections);
  const hits = (selections as Array<{ nodeName: string; elementIndex: number } | null>).filter(
    (sel) => sel !== null
  );
  console.log(`[probe4-iter9b] selections=${JSON.stringify(selections)}`);
  expect(hits.length).toBeGreaterThan(0);
  expect(typeof hits[0]!.nodeName).toBe('string');
  expect(hits[0]!.nodeName.length).toBeGreaterThan(0);
  expect(Number.isInteger(hits[0]!.elementIndex)).toBe(true);
  expect(hits[0]!.elementIndex).toBeGreaterThanOrEqual(0);

  const glErrors = await getWebGLErrors(page);
  expect(glErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
