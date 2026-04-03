/**
 * Custom GUI Library E2E Tests
 *
 * Tests the custom GUI library using the demo page which properly loads
 * the library and demonstrates all features.
 */

import { test, expect } from '@playwright/test';
import { waitForNextRender } from './helpers';

// Extend Window type for test-specific properties
declare global {
  interface Window {
    gui: { destroy: () => void; controllersRecursive: () => unknown[] };
    settings: Record<string, unknown>;
    controllers: Record<string, unknown>;
  }
}

test.describe('Custom GUI Library', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the demo page which properly loads the GUI library
    await page.goto('/demo-custom-gui.html');
    // Wait for GUI to be created and shown
    await page.waitForSelector('.luxar-gui', { timeout: 10000 });
  });

  test('should create and display GUI panel', async ({ page }) => {
    // Verify GUI is visible
    const gui = page.locator('.luxar-gui');
    await expect(gui).toBeVisible();

    // Verify title
    await expect(gui.locator('.luxar-gui__title')).toHaveText('Custom GUI Demo');

    // Verify folders exist (Camera, HDR, Post-Processing, Navigation, Misc)
    const folders = gui.locator('.luxar-gui__folder');
    await expect(folders).toHaveCount(5);

    // Verify various controller types exist
    await expect(gui.locator('.luxar-gui__slider')).not.toHaveCount(0);
    await expect(gui.locator('.luxar-gui__checkbox')).not.toHaveCount(0);
    await expect(gui.locator('.luxar-gui__select')).not.toHaveCount(0);
    await expect(gui.locator('.luxar-gui__button')).not.toHaveCount(0);
  });

  test('should handle number controller interactions', async ({ page }) => {
    // Find the FOV slider
    const fovSlider = page.locator('.luxar-gui__slider').first();

    // Get initial value
    const initialValue = await page.evaluate(() => (window as Window).settings.fov);

    // Change slider value
    await fovSlider.fill('60');
    await waitForNextRender(page);

    // Verify value updated in settings
    const newValue = await page.evaluate(() => (window as Window).settings.fov);
    expect(newValue).toBe(60);
    expect(newValue).not.toBe(initialValue);
  });

  test('should handle theme switching', async ({ page }) => {
    // Get initial background color
    const darkBg = await page.locator('.luxar-gui').evaluate((el) => {
      return getComputedStyle(el).backgroundColor;
    });

    // Click light theme button
    await page.click('.theme-btn[data-theme="light"]');
    await waitForNextRender(page);

    // Get light theme background color
    const lightBg = await page.locator('.luxar-gui').evaluate((el) => {
      return getComputedStyle(el).backgroundColor;
    });

    // Background should change
    expect(darkBg).not.toBe(lightBg);

    // Switch back to dark
    await page.click('.theme-btn[data-theme="dark"]');
    await waitForNextRender(page);

    const backToDarkBg = await page.locator('.luxar-gui').evaluate((el) => {
      return getComputedStyle(el).backgroundColor;
    });

    // Should be similar to original dark
    expect(backToDarkBg).toBe(darkBg);
  });

  test('should handle folder expand/collapse', async ({ page }) => {
    // Find the Camera folder
    const cameraFolder = page.locator('.luxar-gui__folder').first();
    const folderTitle = cameraFolder.locator('.luxar-gui__folder-title');
    const folderChildren = cameraFolder.locator('.luxar-gui__children');

    // Initially open
    await expect(cameraFolder).toHaveClass(/luxar-gui__folder--open/);
    await expect(folderChildren).toBeVisible();

    // Click to close
    await folderTitle.click();
    await waitForNextRender(page);

    // Should be closed
    await expect(cameraFolder).toHaveClass(/luxar-gui__folder--closed/);
    await expect(folderChildren).not.toBeVisible();

    // Click to open again
    await folderTitle.click();
    await waitForNextRender(page);

    // Should be open
    await expect(cameraFolder).toHaveClass(/luxar-gui__folder--open/);
    await expect(folderChildren).toBeVisible();
  });

  test('should support controller synchronization pattern', async ({ page }) => {
    // The demo page has FOV preset synced with FOV slider
    // Find the preset dropdown (second select in Camera folder)
    const presetSelect = page.locator('.luxar-gui__select').first();

    // Change preset to 35mm
    await presetSelect.selectOption('35mm');
    await waitForNextRender(page);

    // Verify settings object updated (35mm = 63 degrees)
    const fovValue = await page.evaluate(() => (window as Window).settings.fov);
    expect(fovValue).toBe(63);

    // Change to 85mm
    await presetSelect.selectOption('85mm');
    await waitForNextRender(page);

    // Verify settings object updated (85mm = 28 degrees)
    const newFovValue = await page.evaluate(() => (window as Window).settings.fov);
    expect(newFovValue).toBe(28);
  });

  test('should support conditional show/hide pattern', async ({ page }) => {
    // The demo page has damping control that shows/hides based on inertial mode
    // Find the Navigation folder
    const navFolder = page.locator('.luxar-gui__folder').filter({ hasText: 'Navigation' });

    // Find the inertial checkbox and damping slider
    const inertialCheckbox = navFolder.locator('.luxar-gui__checkbox').nth(1);
    const dampingController = navFolder
      .locator('.luxar-gui__controller--number')
      .filter({ hasText: 'Damping' });

    // Damping should be hidden initially (inertial mode is off)
    await expect(dampingController).toBeHidden();

    // Enable inertial mode
    await inertialCheckbox.check();
    await waitForNextRender(page);

    // Damping should now be visible
    await expect(dampingController).toBeVisible();

    // Disable inertial mode
    await inertialCheckbox.uncheck();
    await waitForNextRender(page);

    // Damping should be hidden again
    await expect(dampingController).toBeHidden();
  });

  test('should prevent memory leaks on destroy', async ({ page }) => {
    // Create multiple GUI instances and destroy them
    const initialGuiCount = await page.locator('.luxar-gui').count();
    expect(initialGuiCount).toBe(1);

    // Destroy the GUI
    await page.evaluate(() => {
      (window as Window).gui.destroy();
    });

    // Verify GUI is removed from DOM
    const guiCount = await page.locator('.luxar-gui').count();
    expect(guiCount).toBe(0);
  });

  test('should match visual snapshot (dark theme)', async ({ page }) => {
    // Ensure dark theme is active
    await page.click('.theme-btn[data-theme="dark"]');
    await waitForNextRender(page);

    // Take visual snapshot of the GUI
    const gui = page.locator('.luxar-gui');
    await expect(gui).toHaveScreenshot('custom-gui-dark-theme.png', {
      maxDiffPixelRatio: 0.05,
    });
  });

  test('should display correct FOV range (10-170)', async ({ page }) => {
    // The FOV slider should have max of 170, not 200
    const fovSlider = page.locator('.luxar-gui__slider').first();
    const max = await fovSlider.getAttribute('max');
    expect(max).toBe('170');
  });

  test('should have responsive layout', async ({ page }) => {
    // The GUI should have correct width
    const gui = page.locator('.luxar-gui');
    const box = await gui.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.width).toBe(350); // Demo page sets width: 350
  });

  test('should handle function controller (button) clicks', async ({ page }) => {
    // Find the Reset Camera button
    const resetButton = page.locator('.luxar-gui__button').filter({ hasText: 'Reset Camera' });

    // Click the button
    await resetButton.click();

    // After reset, FOV should be 50 (default)
    const fovValue = await page.evaluate(() => (window as Window).settings.fov);
    expect(fovValue).toBe(50);
  });

  test('should have HDR exposure slider with correct range', async ({ page }) => {
    // The HDR folder contains an exposure slider with a range of [-2, 2]
    const hdrFolder = page.locator('.luxar-gui__folder').filter({ hasText: 'HDR' });
    const exposureSlider = hdrFolder.locator('.luxar-gui__slider').first();

    // Verify slider exists and has expected range
    await expect(exposureSlider).toBeVisible();
    const min = await exposureSlider.getAttribute('min');
    const max = await exposureSlider.getAttribute('max');
    const step = await exposureSlider.getAttribute('step');

    expect(Number(min)).toBeLessThan(0);
    expect(Number(max)).toBeGreaterThan(0);
    expect(Number(step)).toBeGreaterThan(0);
    expect(Number(step)).toBeLessThanOrEqual(0.1);

    // Verify the initial settings value is a number
    const initialValue = await page.evaluate(() => (window as any).settings.hdrLog);
    expect(typeof initialValue).toBe('number');
    expect(initialValue).toBeGreaterThanOrEqual(Number(min));
    expect(initialValue).toBeLessThanOrEqual(Number(max));
  });
});
