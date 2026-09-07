/**
 * Mobile: nothing is off-screen or unreachable on a phone.
 *
 * Geometry assertions, not screenshots: the visual corpus is desktop and
 * Linux-only, and these are the questions that matter on a 390 × 664 or a
 * 750 × 340 viewport.
 */

import { test, expect } from '../fixtures';
import { waitForLuxarReady } from '../helpers';
import { rectOf, tapRail } from './touch-helpers';

const DATASET = 'http://localhost:9000/datasets/examples/layers_test_example.luxar.zarr';

test.describe('mobile layout', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug&dpr=1`);
    await waitForLuxarReady(page);
    await page.waitForTimeout(500);
  });

  test('the control rail fits the viewport and its buttons keep their size', async ({ page }) => {
    const rail = await rectOf(page, '.luxar-control-rail');
    expect(rail).not.toBeNull();
    expect(rail!.top).toBeGreaterThanOrEqual(-0.5);
    expect(rail!.bottom).toBeLessThanOrEqual(rail!.vh + 0.5);
    const btn = await rectOf(page, '.luxar-control-rail__btn');
    expect(btn!.height).toBeGreaterThanOrEqual(36);
    // When the rail is taller than the viewport, its items box scrolls and the
    // last button is reachable by scrolling — never clipped.
    const items = await rectOf(page, '.luxar-control-rail__items');
    if (items && items.scrollable) {
      const reachable = await page.evaluate(() => {
        const box = document.querySelector('.luxar-control-rail__items')!;
        box.scrollTop = box.scrollHeight;
        const btns = box.querySelectorAll('.luxar-control-rail__btn');
        const r = btns[btns.length - 1].getBoundingClientRect();
        return r.bottom <= window.innerHeight + 0.5 && r.top >= -0.5;
      });
      expect(reachable).toBe(true);
    }
  });

  test('the help overlay opens fully on-screen', async ({ page }) => {
    await tapRail(page, 'help');
    const help = await rectOf(page, '.luxar-help-overlay');
    expect(help).not.toBeNull();
    expect(help!.left).toBeGreaterThanOrEqual(-0.5);
    expect(help!.right).toBeLessThanOrEqual(help!.vw + 0.5);
    expect(help!.bottom).toBeLessThanOrEqual(help!.vh + 0.5);
  });

  test('the data monitor, expanded, stays inside the viewport', async ({ page }) => {
    await tapRail(page, 'monitor');
    await page.locator('.luxar-data-monitor__expand-btn').tap();
    const monitor = await rectOf(page, '.luxar-data-monitor');
    expect(monitor).not.toBeNull();
    expect(monitor!.left).toBeGreaterThanOrEqual(-0.5);
    expect(monitor!.right).toBeLessThanOrEqual(monitor!.vw + 0.5);
  });

  test('the layers panel keeps its controls reachable', async ({ page }) => {
    await tapRail(page, 'layers');
    const panel = await rectOf(page, '.luxar-layers-panel');
    expect(panel).not.toBeNull();
    expect(panel!.bottom).toBeLessThanOrEqual(panel!.vh + 0.5);
    const controls = await rectOf(page, '.luxar-layers-panel__controls');
    if (controls) {
      // Either it fits, or it scrolls — never spills off the bottom unscrollably.
      expect(controls.bottom <= controls.vh + 0.5 || controls.scrollable).toBe(true);
    }
  });
});
