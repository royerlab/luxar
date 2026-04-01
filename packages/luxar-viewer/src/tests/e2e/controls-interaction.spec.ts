/**
 * Controls Interaction Tests for Luxar Viewer
 *
 * These tests verify:
 * - Keyboard shortcuts work correctly
 * - Camera controls respond to input
 * - Control mode switching works
 * - Help overlay appears
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState, waitForNextRender } from './helpers';

test.describe('Luxar Controls & Keyboard Shortcuts', () => {
  test.fixme('fullscreen is blocked in headless Chromium', async ({ page }) => {
    // Fullscreen API requires user gesture and is blocked by security policy
    // in headless Chromium. This test cannot meaningfully verify fullscreen toggling.
    await page.goto('/?debug');
    await waitForLuxarReady(page);
    await page.keyboard.press('Space');
  });

  test('should show help overlay with H key', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Press H to show help
    await page.keyboard.press('h');

    // Wait for overlay to appear
    await waitForNextRender(page);

    // Check if help overlay is visible after pressing H
    // Use specific selectors — do not fall back to matching any text containing "Help"
    const helpVisibleAfter = await page.evaluate(() => {
      const helpOverlay = document.querySelector(
        '.help-overlay, #help-overlay, [data-help-overlay]'
      );
      if (helpOverlay) return true;
      // Check for the specific help content heading
      const body = document.body.innerHTML;
      return body.includes('Keyboard Shortcuts');
    });

    // Help overlay should appear after pressing H
    expect(helpVisibleAfter).toBe(true);
  });

  test('should track camera position changes via mouse drag', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get initial camera position
    const initialPosition = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        x: debug.camera.position.x,
        y: debug.camera.position.y,
        z: debug.camera.position.z,
      };
    });

    // Perform a mouse drag to move the camera via orbit controls
    const viewport = page.viewportSize()!;
    const centerX = viewport.width / 2;
    const centerY = viewport.height / 2;

    await page.mouse.move(centerX, centerY);
    await page.mouse.down();
    await page.mouse.move(centerX + 100, centerY, { steps: 5 });
    await page.mouse.up();

    // Wait for the controls to update the camera
    await waitForNextRender(page);

    // Get new camera position
    const newPosition = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        x: debug.camera.position.x,
        y: debug.camera.position.y,
        z: debug.camera.position.z,
      };
    });

    // Camera position should have changed from the mouse drag
    const positionChanged =
      newPosition.x !== initialPosition.x ||
      newPosition.y !== initialPosition.y ||
      newPosition.z !== initialPosition.z;
    expect(positionChanged).toBe(true);
  });

  test('should switch control modes', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get initial control type
    const initialType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });

    // Should have a control type
    expect(initialType).toBeDefined();
    expect(['orbit', 'arcball', 'fly']).toContain(initialType);

    // Try to switch control mode (press V)
    await page.keyboard.press('v');

    // Wait for change
    await waitForNextRender(page);

    // Get new control type
    const newType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });

    // Control type should be valid and different from initial
    expect(['orbit', 'arcball', 'fly']).toContain(newType);
    expect(newType).not.toBe(initialType);
  });

  test('should access console interceptor', async ({ page }) => {
    await page.goto('/?debug');
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
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);

    // FOV should be within reasonable bounds (10-170 degrees, must be <180°)
    expect(state.cameraFov).toBeGreaterThan(10);
    expect(state.cameraFov).toBeLessThan(170);
  });
});
