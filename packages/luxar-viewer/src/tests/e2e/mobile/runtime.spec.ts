/**
 * Mobile: the runtime budgets are device-aware.
 *
 * On a phone the DPR cap resolves to 2 even when a scene allows high DPR, and
 * the GPU byte budget is the mobile pool share rather than the 512 MB desktop
 * fallback. Both are read through the debug surface and the console.
 */

import { test, expect } from '../fixtures';
import { captureConsoleMessages, waitForLuxarReady } from '../helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_manual.luxar.zarr';

test.describe('mobile runtime clamps', () => {
  test('allowing high DPR renders at no more than DPR 2 on a DPR-3 phone', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await page.evaluate(() => {
      const debug = (
        window as unknown as {
          __luxarDebug: { app: { setRenderingSettings: (p: Record<string, unknown>) => void } };
        }
      ).__luxarDebug;
      debug.app.setRenderingSettings({ allowHighDPR: true });
    });
    await page.waitForTimeout(800);
    const { native, renderer } = await page.evaluate(() => ({
      native: window.devicePixelRatio,
      renderer: (
        window as unknown as { __luxarDebug: { renderer: { getPixelRatio: () => number } } }
      ).__luxarDebug.renderer.getPixelRatio(),
    }));
    test.skip(native <= 2, 'only meaningful on a DPR > 2 device profile');
    expect(renderer).toBeLessThanOrEqual(2);
  });

  test('the GPU byte budget takes the mobile pool share, not the desktop fallback', async ({
    page,
  }) => {
    const captured = captureConsoleMessages(page);
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    const line = captured.logs.find((text) => /GPU byte budget/.test(text));
    expect(line, 'a GPU byte budget log line').toBeDefined();
    expect(line).toMatch(/mobile device class/);
    expect(line).not.toMatch(/no memory signal/);
  });
});
