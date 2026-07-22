/**
 * TEMPORARY verification probe (campaign 4, iteration 11) — DELETE ME.
 *
 * Randomized camera/controls gesture storm: load multiple_objects_example,
 * run a SEEDED random sequence of ~120 gestures (orbit drags, wheel zooms
 * in/out incl. deep zoom-outs, shift-drag pans, a V-key mode cycle
 * mid-sequence at scale 1) via page.mouse. After EVERY 20 gestures assert:
 *   - camera position/quaternion contain no NaN (__luxarDebug.camera)
 *   - controls focus target is finite
 *   - camera has not flung to >1000× the scene diagonal from the target
 *   - the scene still renders (crop coverage > 0 after a settle)
 *   - no page errors
 *
 * Runs twice: at native scale 1 and at scene scale ×1e-6 (scale recipe
 * applied first) — the orbit floors are scene-relative now; a NaN or a
 * fling here is a regression of that fix.
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

const DATASET = 'http://localhost:9000/datasets/examples/multiple_objects_example.luxar.zarr';

/** Crop that avoids the left rail/info panels, bottom nav panel and toasts. */
const CROP = { x0: 0.36, y0: 0.05, x1: 0.95, y1: 0.7 };

/** Gesture playground (viewport px) — stays over the canvas, off the DOM UI. */
const ARENA = { x0: 480, y0: 70, x1: 1150, y1: 480 };

/** Deterministic RNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function croppedCoverage(page: Page, savePath?: string): Promise<number> {
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
      return nonBlack;
    },
    { url: dataUrl, crop: CROP }
  );
}

/** Same in-page scale recipe as iter 9 (see that spec for rationale). */
async function applyScaleRecipe(page: Page, s: number): Promise<void> {
  await page.evaluate((scale) => {
    const dbg = (window as any).__luxarDebug;
    const sm = dbg.app.sceneManager;
    const scene = dbg.scene;
    const cam = dbg.camera;
    const controlsMgr = dbg.controls;

    scene.scale.multiplyScalar(scale);
    scene.updateMatrixWorld(true);
    scene.traverse((o: any) => {
      const pb = o.userData?.positionBounds;
      if (pb?.min && pb?.max) {
        pb.min = pb.min.map((v: number) => v * scale);
        pb.max = pb.max.map((v: number) => v * scale);
      }
    });
    sm.boundsCache.invalidate();

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
      ps.nodeMap.forEach((entry: any) => applyRadius(entry.pick));
    }

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

    sm.updateMaterialsForCurrentCamera();
    dbg.renderOnce();
  }, s);
  await waitForNextRender(page, 4);
}

interface CamProbe {
  pos: number[];
  quat: number[];
  target: number[];
  distance: number;
  controlType: string;
}

async function probeCamera(page: Page): Promise<CamProbe> {
  return await page.evaluate(() => {
    const dbg = (window as any).__luxarDebug;
    const cam = dbg.camera;
    const target = dbg.controls.getFocusTarget();
    return {
      pos: cam.position.toArray(),
      quat: [cam.quaternion.x, cam.quaternion.y, cam.quaternion.z, cam.quaternion.w],
      target: target.toArray(),
      distance: cam.position.distanceTo(target),
      controlType: dbg.controls.getControlType(),
    };
  });
}

/** Scene bounds diagonal (post-recipe = scaled). */
async function sceneDiagonal(page: Page): Promise<number> {
  return await page.evaluate(() => {
    const dbg = (window as any).__luxarDebug;
    const sm = dbg.app.sceneManager;
    sm.boundsCache.ensure(dbg.scene);
    const b = sm.boundsCache.getBounds();
    if (!b) return 0;
    const dx = b.max.x - b.min.x;
    const dy = b.max.y - b.min.y;
    const dz = b.max.z - b.min.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  });
}

async function runStorm(page: Page, opts: { seed: number; vToggle: boolean; label: string }) {
  const rng = mulberry32(opts.seed);
  const pick = (a: number, b: number) => a + rng() * (b - a);
  const pageErrors: string[] = [];
  page.on('pageerror', (err) => pageErrors.push(err.message));

  const diag = await sceneDiagonal(page);
  expect(diag).toBeGreaterThan(0);

  const gestures = 120;
  const blockSize = 20;
  const summary: string[] = [];

  for (let i = 0; i < gestures; i++) {
    const cx = pick(ARENA.x0 + 100, ARENA.x1 - 100);
    const cy = pick(ARENA.y0 + 60, ARENA.y1 - 60);
    const kind = rng();

    if (opts.vToggle && i === 55) {
      // One full V mode cycle: orbit → fly → ortho → orbit.
      await focusCanvas(page);
      for (let k = 0; k < 3; k++) {
        await page.keyboard.press('v');
        await waitForNextRender(page);
      }
    } else if (kind < 0.35) {
      // Orbit drag (left button).
      const dx = pick(-180, 180);
      const dy = pick(-120, 120);
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      await page.mouse.move(cx + dx / 2, cy + dy / 2, { steps: 4 });
      await page.mouse.move(cx + dx, cy + dy, { steps: 4 });
      await page.mouse.up();
    } else if (kind < 0.55) {
      // Pan (shift + left drag).
      const dx = pick(-40, 40);
      const dy = pick(-40, 40);
      await page.keyboard.down('Shift');
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      await page.mouse.move(cx + dx, cy + dy, { steps: 3 });
      await page.mouse.up();
      await page.keyboard.up('Shift');
    } else if (kind < 0.8) {
      // Wheel zoom in/out.
      await page.mouse.move(cx, cy);
      const notches = 1 + Math.floor(rng() * 4);
      const dir = rng() < 0.5 ? -1 : 1;
      for (let n = 0; n < notches; n++) {
        await page.mouse.wheel(0, dir * 120);
      }
    } else {
      // Deep zoom-out burst.
      await page.mouse.move(cx, cy);
      for (let n = 0; n < 12; n++) {
        await page.mouse.wheel(0, 240);
      }
    }

    if ((i + 1) % blockSize === 0) {
      await page.evaluate(() => (window as any).__luxarDebug.renderOnce());
      await waitForNextRender(page, 3);
      const cam = await probeCamera(page);
      const allFinite = [...cam.pos, ...cam.quat, ...cam.target, cam.distance].every((v) =>
        Number.isFinite(v)
      );
      const nonBlack = await croppedCoverage(page);
      summary.push(
        `block=${(i + 1) / blockSize} type=${cam.controlType} dist=${cam.distance.toExponential(3)} dist/diag=${(cam.distance / diag).toFixed(2)} cover=${nonBlack}px finite=${allFinite}`
      );
      expect(allFinite, `NaN/Inf in camera state after gesture ${i + 1}: ${JSON.stringify(cam)}`).toBe(
        true
      );
      expect(
        cam.distance,
        `camera flung to ${cam.distance} (diag=${diag}) after gesture ${i + 1}`
      ).toBeLessThan(diag * 1000);
      expect(nonBlack, `scene stopped rendering after gesture ${i + 1}`).toBeGreaterThan(0);
      expect(pageErrors, `page errors after gesture ${i + 1}`).toEqual([]);
    }
  }

  console.log(`[probe4-iter11] ${opts.label}\n  ${summary.join('\n  ')}`);
  await page
    .locator('canvas')
    .first()
    .screenshot({ path: `delme/zz-probe4-evidence/iter11-after-storm-${opts.label}.png` });
  const glErrors = await getWebGLErrors(page);
  expect(glErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
}

test('iter11: 120-gesture seeded storm at native scale (incl. V mode cycle)', async ({ page }) => {
  test.setTimeout(240000);
  await page.goto(`/?src=${DATASET}&debug`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  await focusCanvas(page);
  await runStorm(page, { seed: 0xc0ffee, vToggle: true, label: 'scale-1' });
});

test('iter11: 120-gesture seeded storm at scene scale ×1e-6', async ({ page }) => {
  test.setTimeout(240000);
  await page.goto(`/?src=${DATASET}&debug`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  await focusCanvas(page);
  await applyScaleRecipe(page, 1e-6);
  // No V toggle at 1e-6: swapToOrthographic carries an absolute 0.001
  // distance floor (camera-mode.ts) that mis-frames sub-milli scenes —
  // out of scope for the orbit-floor regression this run guards.
  await runStorm(page, { seed: 0xc0ffee, vToggle: false, label: 'scale-1e-6' });
});
