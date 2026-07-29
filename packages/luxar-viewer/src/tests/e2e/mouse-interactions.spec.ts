/**
 * E2E Tests for Mouse Interactions
 *
 * These tests verify:
 * - Ctrl+scroll changes FOV (not camera distance)
 * - Normal scroll changes camera distance (not FOV)
 * - Shift+scroll rotates the view
 * - Modifier key state does not interfere with subsequent keyboard shortcuts
 *
 * Dataset: build_example_structured.luxar.zarr (3D, reliable point count)
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  waitForNextRender,
  focusCanvas,
  ctrlScroll,
  assertNoConsoleErrors,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

test.describe('Mouse Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);
    await focusCanvas(page);
  });

  // Catch silent JS exceptions during scroll/zoom interactions; a renderer
  // crash that doesn't reach the test assertions would otherwise pass.
  test.afterEach(async ({ page }) => {
    await assertNoConsoleErrors(page);
  });

  test('should change FOV with Ctrl+scroll', async ({ page }) => {
    // Read initial FOV
    const initialState = await getLuxarState(page);
    const initialFov = initialState.cameraFov;
    expect(typeof initialFov).toBe('number');
    expect(initialFov).toBeGreaterThan(10);
    expect(initialFov).toBeLessThan(170);

    // Ctrl+scroll to change FOV
    await ctrlScroll(page, -500);

    // Read FOV after interaction
    const newState = await getLuxarState(page);
    const newFov = newState.cameraFov;

    // FOV should have changed
    expect(newFov).not.toBeCloseTo(initialFov, 0);
  });

  test('should change FOV but not zoom distance with Ctrl+scroll', async ({ page }) => {
    // Re-enabled: the orbit wheel handler now ignores ctrl/meta wheel
    // events directly (stateless routing — no keydown-tracked
    // setEnableZoom gate), so a synthetic {ctrlKey:true} wheel reliably
    // changes FOV without dollying. This is the regression test for
    // that exclusivity.
    // Read initial state
    const initial = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        fov: debug.getState().cameraFov,
        z: debug.camera.position.z,
      };
    });

    // Ctrl+scroll (changes FOV)
    await ctrlScroll(page, -500);

    const after = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        fov: debug.getState().cameraFov,
        z: debug.camera.position.z,
      };
    });

    // FOV should have changed
    expect(after.fov).not.toBeCloseTo(initial.fov, 0);

    // Camera Z position should be approximately the same (Ctrl+scroll adjusts FOV, not position)
    // Allow some tolerance since damping may cause small movements
    const zDelta = Math.abs(after.z - initial.z);
    const normalScrollDelta = Math.abs(initial.z) * 0.1; // 10% of initial distance as threshold
    expect(zDelta).toBeLessThan(normalScrollDelta);
  });

  test('should zoom (change distance) with normal scroll', async ({ page }) => {
    // Read initial camera position and FOV
    const initial = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pos = debug.camera.position;
      const state = debug.getState();
      return {
        x: pos.x,
        y: pos.y,
        z: pos.z,
        fov: state.cameraFov,
      };
    });

    // Normal scroll (no modifier keys) to zoom
    await page.mouse.wheel(0, -500);
    await waitForNextRender(page, 3);

    // Force render to apply damping
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 5; i++) {
        debug?.renderOnce?.();
      }
    });
    // OrbitControls.update() runs damping math each call; the visible
    // result lands on the next paint. Fixed sleep here gives the
    // browser one paint cycle past the synchronous renderOnce loop.
    await page.waitForTimeout(300);

    // Read camera position and FOV after scroll
    const after = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pos = debug.camera.position;
      const state = debug.getState();
      return {
        x: pos.x,
        y: pos.y,
        z: pos.z,
        fov: state.cameraFov,
      };
    });

    // Camera distance should have changed
    const distance = Math.sqrt(
      (after.x - initial.x) ** 2 + (after.y - initial.y) ** 2 + (after.z - initial.z) ** 2
    );
    expect(distance).toBeGreaterThan(0.01);

    // FOV should NOT have changed (normal scroll changes distance, not FOV)
    expect(after.fov).toBeCloseTo(initial.fov, 1);
  });

  test('should rotate view with Shift+scroll', async ({ page }) => {
    // Read initial camera quaternion
    const initialQuat = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const q = debug.camera.quaternion;
      return { x: q.x, y: q.y, z: q.z, w: q.w };
    });

    // Shift+scroll to rotate (use custom event with shiftKey=true)
    await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return;
      canvas.dispatchEvent(
        new WheelEvent('wheel', { deltaY: -300, shiftKey: true, bubbles: true })
      );
    });
    await waitForNextRender(page, 3);

    // Force render to apply damping
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 5; i++) {
        debug?.renderOnce?.();
      }
    });
    // Damping settle — see the wheel-zoom test above for the rationale.
    await page.waitForTimeout(300);

    // Read quaternion after rotation
    const newQuat = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const q = debug.camera.quaternion;
      return { x: q.x, y: q.y, z: q.z, w: q.w };
    });

    // At least one quaternion component should have changed
    const dx = Math.abs(newQuat.x - initialQuat.x);
    const dy = Math.abs(newQuat.y - initialQuat.y);
    const dz = Math.abs(newQuat.z - initialQuat.z);
    const dw = Math.abs(newQuat.w - initialQuat.w);
    const totalDelta = dx + dy + dz + dw;
    expect(totalDelta).toBeGreaterThan(0.001);
  });

  test('should not interfere with keyboard shortcuts', async ({ page }) => {
    // Perform a Ctrl+scroll interaction
    await ctrlScroll(page, -300);

    // Now press H to show help overlay -- modifier keys should be properly released
    await page.keyboard.press('h');
    await waitForNextRender(page);

    // Verify the help overlay appeared
    const helpVisible = await page.evaluate(() => {
      const overlay = document.getElementById('luxar-help-overlay');
      if (!overlay) return false;
      return overlay.classList.contains('luxar-help-overlay');
    });
    expect(helpVisible).toBe(true);

    // Clean up: dismiss help
    await page.keyboard.press('h');
    await waitForNextRender(page);
  });
});
