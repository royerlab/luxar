/** A pending HDR readback must not compete with the viewer's frame loop. */

import { test, expect } from './fixtures';

test('HDR readback suspends background draws on a dense scene', async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 360 });
  await page.goto('/?debug&dpr=1&renderer=webgl');
  await page.waitForFunction(() => (window as any).__luxarDebug?.app?.isInitialized === true);
  await page.evaluate(() => {
    (document.querySelector('.luxar-dataset-browser__close-btn') as HTMLElement | null)?.click();
  });

  const result = await page.evaluate(async () => {
    const dbg = (window as any).__luxarDebug;
    const pp = dbg.app.sceneManager.postProcessing;
    pp.setBloomEnabled(false);
    // Keep the frame loop active while the readback fence is pending.
    pp.setDetectorNoiseEnabled(true);
    await dbg.injectSyntheticScene({
      type: 'points',
      count: 12000,
      seed: 7,
      blending: 'additive',
    });
    dbg.renderOnce();
    const before = dbg.renderer.info.render.frame as number;
    const capture = await pp.captureHDRPixels('raw-scene-hdr');
    const after = dbg.renderer.info.render.frame as number;
    let peak = 0;
    for (const value of capture.pixels) {
      if (Number.isFinite(value) && value > peak) peak = value;
    }
    return { frames: after - before, peak, width: capture.width, height: capture.height };
  });

  expect(result.peak).toBeGreaterThan(0);
  expect([result.width, result.height]).toEqual([480, 360]);
  // The capture's scene draw is the only renderer.render() during readback.
  expect(result.frames).toBe(1);
});
