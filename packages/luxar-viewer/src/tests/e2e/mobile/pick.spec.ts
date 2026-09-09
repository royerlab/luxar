/**
 * Mobile: tap-to-pick, long-press menus.
 *
 * Dataset: `test_linked_points.luxar.zarr` — a single labelled point at the
 * world origin carrying `link` + `copy` (see `element-actions.spec.ts`).
 */

import { test, expect } from '../fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from '../helpers';
import { cameraPose, canvasCentre, longPress, quaternionAngle } from './touch-helpers';

const DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_linked_points.luxar.zarr';
const LABEL = 'Linked point';

test.describe('mobile pick', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await page.waitForTimeout(500);
  });

  test('a tap picks the element under the finger and shows its tooltip', async ({ page }) => {
    const c = await canvasCentre(page);
    await page.touchscreen.tap(c.x, c.y);
    await expect(page.locator('[data-overlay-name="__hover_text"]')).toHaveText(LABEL, {
      timeout: 10000,
    });
  });

  test('a long-press opens the element menu; the camera does not move', async ({ page }) => {
    const c = await canvasCentre(page);
    const before = await cameraPose(page);
    await longPress(page, c, 800);
    const menu = page.locator('.luxar-context-menu');
    await expect(menu).toBeVisible({ timeout: 10000 });
    await expect(menu.getByRole('menuitem').first()).toContainText(/Copy|Open link/);
    const after = await cameraPose(page);
    expect(quaternionAngle(before.quaternion, after.quaternion)).toBeLessThan(0.01);
  });

  test('a long-press on the rail Home button opens its popover without re-framing', async ({
    page,
  }) => {
    const btn = page.locator('[data-rail-id="home"]');
    const box = await btn.boundingBox();
    expect(box).not.toBeNull();
    const before = await cameraPose(page);
    await longPress(page, { x: box!.x + box!.width / 2, y: box!.y + box!.height / 2 }, 800);
    await expect(page.locator('.luxar-control-rail__popover')).toBeVisible({ timeout: 5000 });
    const after = await cameraPose(page);
    expect(quaternionAngle(before.quaternion, after.quaternion)).toBeLessThan(0.01);
  });
});
