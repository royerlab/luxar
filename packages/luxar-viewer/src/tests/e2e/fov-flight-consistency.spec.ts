/**
 * A field-of-view change must reach the shaders whichever path writes it.
 *
 * Points and splats used to size themselves from CPU uniforms (a point size
 * factor, a splat focal length) that only `SceneManager.setFov` re-pushed. A
 * flight's intermediate frames, a snapshot restore and `setCameraPose` all
 * write `camera.fov` directly, so after any of them points and splats kept the
 * size (and splats the screen position) of the PREVIOUS fov until something
 * else happened to call `setFov`. The shaders now read the projection matrix
 * three binds per draw, so there is nothing left to go stale.
 *
 * The check moves a scene from one fov to another through `setCameraPose` (the
 * path that skipped the push) and requires the HDR frame to agree with a fresh
 * page that was at the target fov from its first frame. The frame at the
 * starting fov must differ from that reference, so an agreement cannot come
 * from a scene the fov does not affect.
 */

import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';

const START_FOV = 30;
const TARGET_FOV = 60;
const POSE = { position: [0, 0, 3.2], target: [0, 0, 0], up: [0, 1, 0] };

interface Frame {
  width: number;
  height: number;
  pixels: number[];
  peak: number;
}

async function bootSynthetic(page: Page, type: 'points' | 'gsplats'): Promise<void> {
  await page.setViewportSize({ width: 480, height: 360 });
  await page.goto('/?debug&dpr=1&renderer=webgl');
  await page.waitForFunction(() => {
    const dbg = (window as any).__luxarDebug;
    return dbg?.app?.isInitialized === true && typeof dbg.injectSyntheticScene === 'function';
  });
  await page.evaluate(() => {
    (document.querySelector('.luxar-dataset-browser__close-btn') as HTMLElement | null)?.click();
  });
  const count = await page.evaluate(async (t) => {
    const dbg = (window as any).__luxarDebug;
    const pp = dbg.app.sceneManager.postProcessing;
    pp.setDetectorNoiseEnabled(false);
    pp.setVignetteEnabled(false);
    pp.setChromaticLensDistortionEnabled(false);
    pp.setBloomEnabled(false);
    // Additive blending: the frame does not depend on a depth-sort pass
    // landing between the renders being compared.
    const result = await dbg.injectSyntheticScene({
      type: t,
      count: 20000,
      seed: 7,
      blending: 'additive',
    });
    return result.elementCount as number;
  }, type);
  expect(count).toBeGreaterThan(0);
  // Closing the dataset browser re-lays the page out; wait for the drawing
  // buffer to hold one size over consecutive frames before comparing frames.
  await page.evaluate(async () => {
    const canvas = (window as any).__luxarDebug.renderer.domElement as HTMLCanvasElement;
    const frame = () => new Promise((r) => requestAnimationFrame(() => r(null)));
    let last = '';
    let stable = 0;
    for (let i = 0; i < 600 && stable < 10; i++) {
      await frame();
      const size = `${canvas.width}x${canvas.height}`;
      stable = size === last ? stable + 1 : 0;
      last = size;
    }
  });
}

/** Render the current state once and read back the linear HDR scene buffer. */
async function capture(page: Page): Promise<Frame> {
  return page.evaluate(async () => {
    const dbg = (window as any).__luxarDebug;
    dbg.renderOnce();
    const out = await dbg.postProcessing.captureHDRPixels('raw-scene-hdr');
    const px = Array.from(out.pixels as ArrayLike<number>);
    let peak = 0;
    for (const v of px) if (Number.isFinite(v) && v > peak) peak = v;
    return { width: out.width as number, height: out.height as number, pixels: px, peak };
  });
}

/** Sum of per-pixel absolute differences, relative to the reference's total light. */
function relativeL1(a: Frame, b: Frame): number {
  expect([a.width, a.height]).toEqual([b.width, b.height]);
  let diff = 0;
  let total = 0;
  for (let i = 0; i < a.pixels.length; i++) {
    diff += Math.abs(a.pixels[i] - b.pixels[i]);
    total += Math.abs(b.pixels[i]);
  }
  return total > 0 ? diff / total : diff;
}

/** Write `fov` through `setFov` (which pushed on every build) and apply {@link POSE}. */
async function setFovAndPose(page: Page, fov: number): Promise<void> {
  await page.evaluate(
    ({ f, pose }) => {
      const dbg = (window as any).__luxarDebug;
      dbg.app.sceneManager.setFov(f);
      dbg.app.setCameraPose({ ...dbg.app.getCameraPose(), ...pose });
    },
    { f: fov, pose: POSE }
  );
}

for (const type of ['points', 'gsplats'] as const) {
  test(`${type}: an fov carried by setCameraPose reaches the shaders`, async ({
    page,
    context,
  }) => {
    await bootSynthetic(page, type);
    await setFovAndPose(page, START_FOV);
    const before = await capture(page);
    expect(before.peak, 'the synthetic scene drew nothing').toBeGreaterThan(0);

    // The stale path: fov travels with the pose, as in a flight or a restore.
    await page.evaluate(
      ({ fov, pose }) => {
        const dbg = (window as any).__luxarDebug;
        dbg.app.setCameraPose({ ...dbg.app.getCameraPose(), ...pose, fov });
      },
      { fov: TARGET_FOV, pose: POSE }
    );
    const viaPose = await capture(page);
    const fovNow = await page.evaluate(() => (window as any).__luxarDebug.camera.fov as number);
    expect(fovNow).toBeCloseTo(TARGET_FOV, 6);

    // The reference: a fresh page that is at TARGET_FOV from its first frame,
    // so no write path can have left anything behind.
    const fresh = await context.newPage();
    await bootSynthetic(fresh, type);
    await setFovAndPose(fresh, TARGET_FOV);
    const reference = await capture(fresh);

    // Non-vacuous: doubling the fov visibly changes the frame.
    expect(relativeL1(before, reference)).toBeGreaterThan(0.05);
    // The fix: the pose-carried fov draws the reference frame (rounding level only).
    expect(relativeL1(viaPose, reference)).toBeLessThan(1e-5);
  });
}
