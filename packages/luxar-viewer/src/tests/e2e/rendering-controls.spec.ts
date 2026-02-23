/**
 * E2E Tests for Rendering Controls
 *
 * These tests verify:
 * - Rendering controls panel toggle with 'R' key
 * - Basic rendering controls functionality
 * - Camera and navigation settings
 *
 * Note: Complex bug fix scenarios are covered by unit tests in:
 * - src/tests/unit/rendering/rendering-controls.test.ts (14 tests)
 * - src/tests/unit/rendering/material-manager.test.ts (34 tests)
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState, waitForNextRender } from './helpers';

test.describe('Rendering Controls Panel', () => {
  test('should toggle rendering controls panel with R key', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Click on body to dismiss any modal and ensure no input has focus
    await page.click('body', { position: { x: 10, y: 10 } });
    await waitForNextRender(page, 1);

    // Initially panel should be hidden
    const initiallyVisible = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui');
      return panel && (panel as HTMLElement).style.display !== 'none';
    });
    expect(initiallyVisible).toBe(false);

    // Press R to show panel
    await page.keyboard.press('r');
    await waitForNextRender(page);

    // Panel should now be visible
    const visibleAfterR = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui');
      return panel && (panel as HTMLElement).style.display !== 'none';
    });
    expect(visibleAfterR).toBe(true);

    // Press R again to hide
    await page.keyboard.press('r');
    await waitForNextRender(page);

    // Panel should be hidden again
    const hiddenAfterSecondR = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-gui');
      return panel && (panel as HTMLElement).style.display === 'none';
    });
    expect(hiddenAfterSecondR).toBe(true);
  });

  test('should have rendering controls panel with expected title', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Click on body to dismiss any modal and ensure no input has focus
    await page.click('body', { position: { x: 10, y: 10 } });
    await waitForNextRender(page, 1);

    // Show panel
    await page.keyboard.press('r');
    await waitForNextRender(page);

    // Check for panel title
    const hasRenderingControlsTitle = await page.evaluate(() => {
      const titles = document.querySelectorAll('.luxar-gui .luxar-gui__title');
      return Array.from(titles).some((el) => el.textContent?.includes('Rendering'));
    });
    expect(hasRenderingControlsTitle).toBe(true);
  });
});

test.describe('Camera and Navigation Settings', () => {
  test('should have correct initial camera FOV', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);

    // Default FOV should be around 47 degrees (50mm Normal)
    expect(state.cameraFov).toBeGreaterThan(40);
    expect(state.cameraFov).toBeLessThan(55);
  });

  test('should apply FOV changes', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get initial FOV
    const initialState = await getLuxarState(page);
    const initialFov = initialState.cameraFov;

    // Change FOV via scene manager
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.app.components.sceneManager.updateFOV(5); // Increase FOV
    });

    await waitForNextRender(page, 1);

    // Get new FOV
    const newState = await getLuxarState(page);

    // FOV should have changed
    expect(newState.cameraFov).not.toBe(initialFov);
  });

  test('should have valid control type', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const controlType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });

    // Should be one of the valid control types
    expect(['orbit', 'arcball', 'fly']).toContain(controlType);
  });

  test('should be able to switch control types', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Switch to fly controls
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.app.components.sceneManager.setControlType('fly');
    });

    await waitForNextRender(page, 1);

    // Verify fly controls are active
    const controlType = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.controls?.getControlType?.();
    });

    expect(controlType).toBe('fly');
  });
});

test.describe('Post-Processing Settings', () => {
  test('should have post-processing manager accessible', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Check if post-processing manager is available
    const hasPostProcessing = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return !!debug.postProcessing;
    });

    expect(hasPostProcessing).toBe(true);
  });
});

test.describe('Dynamic Clipping', () => {
  test('should have dynamic clipping state accessible', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const clippingState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.app.components.sceneManager?.getDynamicClippingState?.();
    });

    expect(clippingState).toBeDefined();
    expect(typeof clippingState.enabled).toBe('boolean');
    expect(typeof clippingState.adaptSpeed).toBe('number');
  });

  test('should be able to toggle dynamic clipping', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get initial state
    const initialState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.app.components.sceneManager?.getDynamicClippingState?.().enabled;
    });

    // Toggle dynamic clipping
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const currentState = debug.app.components.sceneManager.getDynamicClippingState().enabled;
      debug.app.components.sceneManager.setDynamicClipping(!currentState, 0.1);
    });

    await waitForNextRender(page, 1);

    // Get new state
    const newState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.app.components.sceneManager?.getDynamicClippingState?.().enabled;
    });

    // State should have toggled
    expect(newState).toBe(!initialState);
  });
});

test.describe('Rendering Controls API', () => {
  test('should have rendering controls accessible', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const hasRenderingControls = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return !!debug.renderingControls;
    });

    expect(hasRenderingControls).toBe(true);
  });

  test('should have settings object with expected properties', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const settingsProps = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const settings = debug.renderingControls?.settings;
      if (!settings) return null;

      return {
        hasHdrMultiplier: 'hdrMultiplier' in settings,
        hasBloomStrength: 'bloomStrength' in settings,
        hasBloomThreshold: 'bloomThreshold' in settings,
        hasFov: 'fov' in settings,
        hasControlType: 'controlType' in settings,
        hasAoEnabled: 'aoEnabled' in settings,
      };
    });

    expect(settingsProps).not.toBeNull();
    expect(settingsProps?.hasHdrMultiplier).toBe(true);
    expect(settingsProps?.hasBloomStrength).toBe(true);
    expect(settingsProps?.hasBloomThreshold).toBe(true);
    expect(settingsProps?.hasFov).toBe(true);
    expect(settingsProps?.hasControlType).toBe(true);
    expect(settingsProps?.hasAoEnabled).toBe(true);
  });

  test('should have HDR multiplier with valid default value', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const hdrMultiplier = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.renderingControls?.settings?.hdrMultiplier;
    });

    // Default HDR multiplier is 1.0 (neutral)
    expect(hdrMultiplier).toBe(1.0);
  });
});
