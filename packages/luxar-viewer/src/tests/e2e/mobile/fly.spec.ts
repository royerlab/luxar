/**
 * Mobile: fly mode by touch — one finger looks, a pinch flies.
 */

import { test, expect } from '../fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from '../helpers';
import {
  cameraPose,
  canvasCentre,
  distance,
  dragFinger,
  pinch,
  quaternionAngle,
} from './touch-helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_manual.luxar.zarr';

async function enterFlyMode(page: import('@playwright/test').Page): Promise<void> {
  await page.evaluate(() => {
    const debug = (
      window as unknown as { __luxarDebug: { controls: { setControlType: (t: string) => void } } }
    ).__luxarDebug;
    debug.controls.setControlType('fly');
  });
  await page.waitForTimeout(200);
}

test.describe('mobile fly mode', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await enterFlyMode(page);
  });

  test('one-finger drag looks around (orientation changes, position does not)', async ({
    page,
  }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await dragFinger(page, c, { x: c.x + 120, y: c.y + 30 });
    await page.waitForTimeout(900);
    const after = await cameraPose(page);
    expect(quaternionAngle(before.quaternion, after.quaternion)).toBeGreaterThan(0.02);
    expect(distance(before.position, after.position)).toBeLessThan(1e-3);
  });

  test('pinch-out flies forward', async ({ page }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await pinch(page, c, 40, 130);
    await page.waitForTimeout(1200);
    const after = await cameraPose(page);
    expect(distance(before.position, after.position)).toBeGreaterThan(0.01);
  });
});
