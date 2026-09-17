/**
 * Controls Interaction Tests for Luxar Viewer
 *
 * These tests verify:
 * - Keyboard shortcuts work correctly
 * - Camera controls respond to input
 * - Control mode switching works
 * - Help overlay appears
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  getLuxarState,
  waitForNextRender,
  dismissDatasetBrowser,
  isFocusOnTypingSurface,
} from './helpers';

test.describe('Luxar Controls & Keyboard Shortcuts', () => {
  test.fixme('fullscreen is blocked in headless Chromium', async ({ page }) => {
    // Fullscreen API requires user gesture and is blocked by security policy
    // in headless Chromium. This test cannot meaningfully verify fullscreen toggling.
    // `&noOpfs` on every load: this spec never asserts the L2 OPFS tier, and
    // automated Chromium's OPFS stalls systemically (10s per op — issue #1645),
    // starving scene readiness past the test budget. The circuit breaker only
    // helps un-flagged real sessions (it still pays ~3 timeouts per fresh page).
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);
    await page.keyboard.press('Space');
  });

  test('should show help overlay with H key', async ({ page }) => {
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);

    // Dismiss the dataset browser modal so keyboard events reach the app
    await dismissDatasetBrowser(page);

    // Press H to show help
    await page.keyboard.press('h');
    await waitForNextRender(page);

    // Verify the help overlay appeared using its actual DOM id and class
    const helpVisible = await page.evaluate(() => {
      const overlay = document.getElementById('luxar-help-overlay');
      if (!overlay) return false;
      return overlay.classList.contains('luxar-help-overlay');
    });
    expect(helpVisible).toBe(true);

    // Verify the overlay has the expected title
    const title = await page.evaluate(() => {
      const el = document.getElementById('luxar-help-overlay-title');
      return el?.textContent ?? '';
    });
    expect(title).toContain('Luxar Controls');

    // Issue #1922: the overlay used to autofocus its filter field, which
    // trips `InputHandler`'s typing guard — the second `H` was swallowed as
    // typing and the "toggle" only ever opened. Focus is now on the overlay
    // container, so the round-trip below works. Asserting the predicate
    // directly pins WHY, not just that the close happened to work.
    expect(await isFocusOnTypingSurface(page)).toBe(false);

    // Press H again to dismiss
    await page.keyboard.press('h');
    await waitForNextRender(page);

    const helpGone = await page.evaluate(() => {
      return document.getElementById('luxar-help-overlay') === null;
    });
    expect(helpGone).toBe(true);
  });

  test('typing filters the help overlay from the first keystroke', async ({ page }) => {
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);
    await dismissDatasetBrowser(page);

    await page.keyboard.press('h');
    await waitForNextRender(page);
    await expect(page.locator('#luxar-help-overlay')).toBeVisible();

    // The other half of #1922: dropping the autofocus must not cost
    // type-to-filter. The container-level forwarder hands the first printable
    // key to the filter, so no keystroke is lost.
    await page.keyboard.press('f');

    const filter = page.locator('#luxar-help-overlay .luxar-panel-filter__input');
    await expect(filter).toBeFocused();
    await expect(filter).toHaveValue('f');

    // And it really filtered: every visible row matches the query.
    const rowsAllMatch = await page.evaluate(() => {
      const rows = Array.from(
        document.querySelectorAll<HTMLElement>('#luxar-help-overlay .luxar-help-overlay__row')
      ).filter((row) => row.style.display !== 'none');
      return (
        rows.length > 0 && rows.every((row) => (row.textContent ?? '').toLowerCase().includes('f'))
      );
    });
    expect(rowsAllMatch).toBe(true);
  });

  test('should track camera position changes via mouse drag', async ({ page }) => {
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);

    // Dismiss the dataset browser modal so mouse events reach the canvas
    await dismissDatasetBrowser(page);

    // Get initial camera position
    const initialPosition = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        x: debug.camera.position.x,
        y: debug.camera.position.y,
        z: debug.camera.position.z,
      };
    });

    // Perform a mouse drag to rotate the camera via orbit controls.
    // Orbit controls have damping (factor 0.25), so we need a large drag
    // and must wait long enough for the damping to apply.
    const viewport = page.viewportSize()!;
    const centerX = viewport.width / 2;
    const centerY = viewport.height / 2;

    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX + 200, centerY + 100, { steps: 10 });
    await page.mouse.up();

    // Force a render and wait for damping to apply (orbit controls update on render)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      // Trigger multiple renders to let damping settle
      for (let i = 0; i < 5; i++) {
        debug?.renderOnce?.();
      }
    });
    // Damping settle past the synchronous renderOnce loop — same shape
    // as mouse-interactions.spec.ts and ortho-mode.spec.ts.
    await page.waitForTimeout(500);

    // Get new camera position
    const newPosition = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        x: debug.camera.position.x,
        y: debug.camera.position.y,
        z: debug.camera.position.z,
      };
    });

    // Camera position should have changed from the orbit drag
    const dx = Math.abs(newPosition.x - initialPosition.x);
    const dy = Math.abs(newPosition.y - initialPosition.y);
    const dz = Math.abs(newPosition.z - initialPosition.z);
    const totalDelta = dx + dy + dz;
    expect(totalDelta).toBeGreaterThan(0.01);
  });

  test('should switch control modes', async ({ page }) => {
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);

    // Dismiss the dataset browser modal so keyboard events reach the app
    await dismissDatasetBrowser(page);

    // Get initial control type
    const initialType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });

    // Should start in orbit mode
    expect(initialType).toBe('orbit');

    // Press V to cycle through control modes (orbit → fly → ortho)
    await page.keyboard.press('v');
    await waitForNextRender(page);

    const afterFirst = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(afterFirst).toBe('fly');

    // Press V again: fly → ortho
    await page.keyboard.press('v');
    await waitForNextRender(page);

    const afterSecond = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(afterSecond).toBe('ortho');

    // Press V again: ortho → orbit (full cycle)
    await page.keyboard.press('v');
    await waitForNextRender(page);

    const afterThird = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });
    expect(afterThird).toBe('orbit');
  });

  test('should access console interceptor', async ({ page }) => {
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);

    // Verify console interceptor is available
    const interceptorInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        hasInterceptor: !!debug.consoleInterceptor,
        // Check for the messages array/property instead of getMessages()
        hasMessages:
          debug.consoleInterceptor &&
          (Array.isArray(debug.consoleInterceptor.messages) ||
            typeof debug.consoleInterceptor.getMessages === 'function'),
        interceptorType: debug.consoleInterceptor ? typeof debug.consoleInterceptor : 'undefined',
      };
    });

    // Console interceptor should exist (basic check only)
    expect(interceptorInfo.hasInterceptor).toBe(true);
    expect(interceptorInfo.interceptorType).toBe('object');

    // Note: Specific API methods (getMessages, messages array) are optional
    // Tests pass as long as the interceptor object exists
  });

  test('should verify FOV within valid range', async ({ page }) => {
    await page.goto('/?debug&noOpfs');
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);

    // FOV should be within reasonable bounds (10-170 degrees, must be <180°)
    expect(state.cameraFov).toBeGreaterThan(10);
    expect(state.cameraFov).toBeLessThan(170);
  });
});
