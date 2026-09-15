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
    expect(btn).not.toBeNull();
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

  test('left-authored text overlays clear the rail and keep a readable measure', async ({
    page,
  }) => {
    await page.evaluate(() => {
      const manager = (
        window as unknown as {
          __luxarDebug: {
            getOverlayManager: () => {
              loadOverlays: (configs: unknown[], baseUrl: string) => Promise<void>;
            } | null;
          };
        }
      ).__luxarDebug.getOverlayManager();
      if (!manager) throw new Error('OverlayManager not available on __luxarDebug');
      return manager.loadOverlays(
        [
          {
            name: '__rail_clearance_probe',
            type: 'overlay_text',
            position: [0.02, 0.5],
            opacity: 1,
            anchor: 'top-left',
            transition: 'none',
            transition_duration: 0,
            interactive: false,
            z_index: 5,
            hover: false,
            text: 'Readable touch caption',
            width: 0.1,
          },
          {
            name: '__viewport_cap_probe',
            type: 'overlay_text',
            position: [0.8, 0.6],
            opacity: 1,
            anchor: 'top-left',
            transition: 'none',
            transition_duration: 0,
            interactive: false,
            z_index: 5,
            hover: false,
            text: 'Viewport-capped caption',
            width: 0.5,
          },
          {
            name: '__right_edge_probe',
            type: 'overlay_text',
            position: [0.98, 0.65],
            opacity: 1,
            anchor: 'top-left',
            transition: 'none',
            transition_duration: 0,
            interactive: false,
            z_index: 5,
            hover: false,
            text: 'Right-edge caption',
            width: 0.3,
          },
          {
            name: '__center_anchor_probe',
            type: 'overlay_text',
            position: [0.19, 0.7],
            opacity: 1,
            anchor: 'top-center',
            transition: 'none',
            transition_duration: 0,
            interactive: false,
            z_index: 5,
            hover: false,
            text: 'Centered caption',
            width: 0.26,
          },
        ],
        ''
      );
    });

    const rail = await rectOf(page, '.luxar-control-rail');
    const overlay = await rectOf(page, '[data-overlay-name="__rail_clearance_probe"]');
    const readableFloor = await page
      .locator('[data-overlay-name="__rail_clearance_probe"]')
      .evaluate((element) => {
        const computed = getComputedStyle(element);
        const probe = document.createElement('div');
        probe.style.position = 'fixed';
        probe.style.visibility = 'hidden';
        probe.style.font = computed.font;
        probe.style.width = '18ch';
        document.body.appendChild(probe);
        const width = probe.getBoundingClientRect().width;
        probe.remove();
        return width;
      });
    expect(rail).not.toBeNull();
    expect(overlay).not.toBeNull();
    expect(overlay!.left).toBeGreaterThanOrEqual(rail!.right + 8);
    expect(overlay!.width).toBeGreaterThanOrEqual(readableFloor - 0.5);
    expect(overlay!.right).toBeLessThanOrEqual(overlay!.vw - 11.5);

    const capped = await rectOf(page, '[data-overlay-name="__viewport_cap_probe"]');
    expect(capped).not.toBeNull();
    expect(capped!.width).toBeGreaterThanOrEqual(readableFloor - 0.5);
    expect(capped!.width).toBeLessThan(capped!.vw * 0.5 - 0.5);
    expect(capped!.right).toBeLessThanOrEqual(capped!.vw - 11.5);

    const rightEdge = await rectOf(page, '[data-overlay-name="__right_edge_probe"]');
    expect(rightEdge).not.toBeNull();
    expect(rightEdge!.width).toBeGreaterThanOrEqual(readableFloor - 0.5);
    expect(rightEdge!.right).toBeLessThanOrEqual(rightEdge!.vw - 11.5);

    const centeredLeft = await page
      .locator('[data-overlay-name="__center_anchor_probe"]')
      .evaluate((element) => parseFloat(getComputedStyle(element).left));
    expect(centeredLeft).toBeCloseTo(overlay!.vw * 0.19, 0);
  });

  test('the rail gutter remains defined without the presence marker', async ({ page }) => {
    const metrics = await page.evaluate(() => {
      const debugConsole = document.querySelector<HTMLElement>('.luxar-debug-console');
      if (!debugConsole) throw new Error('Debug console not found');
      const hadMarker = document.body.classList.contains('luxar-has-control-rail');
      const previousDisplay = debugConsole.style.display;
      const previousVisibility = debugConsole.style.visibility;
      document.body.classList.remove('luxar-has-control-rail');
      debugConsole.style.display = 'block';
      debugConsole.style.visibility = 'hidden';
      const gutter = getComputedStyle(document.documentElement)
        .getPropertyValue('--luxar-rail-gutter')
        .trim();
      const width = debugConsole.getBoundingClientRect().width;
      debugConsole.style.display = previousDisplay;
      debugConsole.style.visibility = previousVisibility;
      if (hadMarker) document.body.classList.add('luxar-has-control-rail');
      return { gutter, width };
    });

    expect(metrics.gutter).toBe('79px');
    expect(metrics.width).toBeGreaterThan(100);
    expect(metrics.width).toBeLessThanOrEqual(600);
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
      expect(
        controls.bottom <= controls.vh + 0.5 || controls.scrollable,
        'layers controls should fit the viewport or provide scrolling'
      ).toBe(true);
    }
  });
});
