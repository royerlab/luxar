/**
 * A splat lands where the projection matrix puts its centre, at any canvas size.
 *
 * The drawing buffer is the CSS size times the device pixel ratio, ROUNDED,
 * while the camera's aspect is the unrounded CSS ratio: 1277×719 CSS pixels at
 * a ratio of 1.5 is a 1915×1078 buffer (aspect 1.77644) under a 1.77608 camera.
 * The rasterizer maps NDC onto the buffer, so everything the GPU projects
 * (points, lines, meshes) follows the camera's P00. Splats used to re-derive
 * their screen centre from one CPU focal length (fx = fy, the vertical scale),
 * which drifts them sideways from everything else by the aspect mismatch times
 * their distance from the screen centre. They now project through P as well.
 *
 * The check draws one splat near the left edge and compares the
 * intensity-weighted centroid of its HDR footprint with its centre projected on
 * the CPU through the camera's own matrices. The old derivation misses by about
 * 0.17 px here; the projection-matrix one by well under a hundredth of a pixel.
 */

import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';

interface Probe {
  width: number;
  height: number;
  expected: [number, number];
  centroid: [number, number];
  mass: number;
}

async function probeSplatCentre(page: Page): Promise<Probe> {
  await page.goto('/?debug&dpr=1.5&renderer=webgl');
  await page.waitForFunction(() => {
    const dbg = (window as any).__luxarDebug;
    return dbg?.app?.isInitialized === true && typeof dbg.injectSyntheticScene === 'function';
  });
  await page.evaluate(() => {
    (document.querySelector('.luxar-dataset-browser__close-btn') as HTMLElement | null)?.click();
  });
  return page.evaluate(async () => {
    const dbg = (window as any).__luxarDebug;
    const sm = dbg.app.sceneManager;
    const pp = sm.postProcessing;
    pp.setDetectorNoiseEnabled(false);
    pp.setVignetteEnabled(false);
    pp.setChromaticLensDistortionEnabled(false);
    pp.setBloomEnabled(false);
    const spec = { type: 'gsplats', count: 1, clusters: 1, seed: 3, blending: 'additive' };
    await dbg.injectSyntheticScene(spec);
    // The same deterministic generator the injector ran, for the exact centre.
    // A page-side path the dev server serves; a variable keeps tsc from
    // resolving it as a module of the test.
    const modulePath = '/src/scene/synthetic-scene.ts';
    const mod = await import(/* @vite-ignore */ modulePath);
    const centers = mod.generateSyntheticGSplats(spec).centers as Float32Array;
    const c = [centers[0], centers[1], centers[2]];

    const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
    // Closing the dataset browser re-lays the page out: wait for one buffer size.
    const canvas = dbg.renderer.domElement as HTMLCanvasElement;
    let last = '';
    for (let stable = 0, i = 0; i < 600 && stable < 10; i++) {
      await frame();
      const size = `${canvas.width}x${canvas.height}`;
      stable = size === last ? stable + 1 : 0;
      last = size;
    }
    // Look past the splat so it sits far left of the screen centre, where a
    // horizontal-scale error is largest.
    dbg.app.setCameraPose({
      ...dbg.app.getCameraPose(),
      position: [c[0], c[1], c[2] + 60],
      target: [c[0] + 40, c[1], c[2]],
      up: [0, 1, 0],
    });
    // Let the pose reach a drawn frame, then freeze the loop so the capture
    // reads that frame (the settle-then-stop the render gate uses).
    dbg.renderOnce();
    for (let i = 0; i < 10; i++) await frame();
    dbg.animationController.stopAnimation();
    const out = await pp.captureHDRPixels('raw-scene-hdr', { flipY: true });
    const W = out.width as number;
    const H = out.height as number;
    const px = out.pixels as Float32Array;

    const cam = sm.camera;
    cam.updateMatrixWorld();
    const v = new cam.position.constructor(c[0], c[1], c[2]);
    v.applyMatrix4(cam.matrixWorldInverse).applyMatrix4(cam.projectionMatrix);

    let mass = 0;
    let sx = 0;
    let sy = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        const l = px[i] + px[i + 1] + px[i + 2];
        if (!(l > 0)) continue;
        mass += l;
        sx += l * (x + 0.5);
        sy += l * (y + 0.5);
      }
    }
    return {
      width: W,
      height: H,
      // flipY: rows run top-down, so NDC +y is the top row.
      expected: [(v.x * 0.5 + 0.5) * W, (0.5 - v.y * 0.5) * H] as [number, number],
      centroid: [sx / mass, sy / mass] as [number, number],
      mass,
    };
  });
}

// The fractional ratio needs both halves: `?dpr=1.5` above (the viewer caps its
// own ratio otherwise) and a device at 1.5 (`?dpr=` is clamped to the native one).
test.use({ deviceScaleFactor: 1.5 });

test('a splat is drawn at its projected centre on an odd-size, fractional-DPR canvas', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1277, height: 719 });
  const probe = await probeSplatCentre(page);
  expect(probe.mass, 'the splat drew nothing').toBeGreaterThan(0);
  // Non-vacuous: the buffer really is off the camera's aspect, and the splat
  // really is far from the centre, where the old derivation missed by ~0.17 px.
  expect(Math.abs(probe.width / probe.height - 1277 / 719)).toBeGreaterThan(1e-4);
  expect(Math.abs(probe.expected[0] - probe.width / 2)).toBeGreaterThan(600);
  expect(Math.abs(probe.centroid[0] - probe.expected[0])).toBeLessThan(0.02);
  expect(Math.abs(probe.centroid[1] - probe.expected[1])).toBeLessThan(0.02);
});
