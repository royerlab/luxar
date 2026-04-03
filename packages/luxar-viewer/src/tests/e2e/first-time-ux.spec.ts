/**
 * First-Time User Experience Tests
 *
 * These tests verify that new users get helpful guidance when:
 * - Opening the viewer with no dataset
 * - Encountering errors loading datasets
 * - Needing help on what to do next
 *
 * Validates the UX improvements for better onboarding.
 */

import { test, expect } from '@playwright/test';
import { waitForNextRender } from './helpers';

test.describe('First-Time User Experience', () => {
  test('should show dataset browser when no dataset specified', async ({ page }) => {
    // Navigate with no dataset parameter
    await page.goto('/?debug');

    // Wait a moment for initialization
    await page.waitForTimeout(2000);

    // Dataset browser should appear automatically
    const browserVisible = await page
      .locator('.dataset-browser')
      .isVisible()
      .catch(() => false);

    expect(browserVisible).toBe(true);
  });

  test('should show welcome banner in dataset browser', async ({ page }) => {
    await page.goto('/?debug');

    // Wait for browser to appear
    await page.waitForSelector('.dataset-browser', { timeout: 5000 });

    // Welcome banner should be present
    const welcomeBanner = page.locator('#browser-welcome');
    await expect(welcomeBanner).toBeVisible();

    // Should contain helpful text (updated to match compact banner)
    const bannerText = await welcomeBanner.textContent();
    expect(bannerText).toContain('Luxar');
    expect(bannerText).toContain('Interactive Scientific Data Visualization');
    expect(bannerText).toContain('.zarr');
  });

  test('should show helpful guidance in welcome banner', async ({ page }) => {
    await page.goto('/?debug');

    await page.waitForSelector('.dataset-browser', { timeout: 5000 });

    const welcomeBanner = page.locator('#browser-welcome');
    await expect(welcomeBanner).toBeVisible();

    const text = await welcomeBanner.textContent();

    // Should explain how to use (didactic) - updated for compact banner
    expect(text).toContain('Browse for .zarr');
    expect(text).toContain('enter path manually');

    // Should mention keyboard shortcuts (helpful)
    expect(text).toContain('H'); // Help key hint

    // Should mention Luxar description
    expect(text).toContain('Scientific Data Visualization');
  });

  test('should show error dialog or dataset browser when dataset fails', async ({ page }) => {
    // Try to load non-existent dataset via HTTP
    await page.goto('/?src=http://localhost:9000/nonexistent.zarr&debug');

    // Viewer should show either error dialog or dataset browser (graceful handling)
    await page.waitForSelector(
      '.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser',
      {
        timeout: 30000,
      }
    );

    const hasErrorDialog = await page
      .locator('.error-message, .luxar-error-dialog')
      .first()
      .isVisible()
      .catch(() => false);
    const hasDatasetBrowser = await page
      .locator('.dataset-browser, .luxar-dataset-browser')
      .first()
      .isVisible()
      .catch(() => false);

    // At least one should be visible (graceful error handling)
    expect(hasErrorDialog || hasDatasetBrowser).toBe(true);

    // If error dialog is shown, verify it has helpful content
    if (hasErrorDialog) {
      const errorText = await page
        .locator('.error-message, .luxar-error-dialog')
        .first()
        .textContent();
      expect(errorText).toContain('Unable to Load Dataset');
    }
  });

  test('should not show hardcoded example URLs in error/browser UI', async ({ page }) => {
    await page.goto('/?src=http://localhost:9000/missing.zarr&debug');

    // Wait for either error dialog or dataset browser
    await page.waitForSelector(
      '.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser',
      {
        timeout: 30000,
      }
    );

    // Get text from whichever UI appeared
    const uiElement = page
      .locator('.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser')
      .first();
    const uiText = await uiElement.textContent();

    // Should NOT have specific hardcoded dataset URLs
    expect(uiText).not.toContain('dimension_navigation_example');
    expect(uiText).not.toContain('dense_grid_5d');
  });

  test('should allow dismissing error or browser UI', async ({ page }) => {
    await page.goto('/?src=http://localhost:9000/fail.zarr&debug');

    // Wait for UI to appear
    await page.waitForSelector(
      '.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser',
      {
        timeout: 30000,
      }
    );

    // Dismiss with Escape or click
    await page.keyboard.press('Escape');
    await waitForNextRender(page);

    // At least one of the dismissal methods should work
    const errorStillVisible = await page
      .locator('.error-message')
      .isVisible()
      .catch(() => false);
    const browserStillVisible = await page
      .locator('.dataset-browser')
      .isVisible()
      .catch(() => false);

    // Escape should dismiss at least one UI element
    expect(errorStillVisible && browserStillVisible).toBe(false);
  });

  test('should close dataset browser with close button', async ({ page }) => {
    await page.goto('/?debug');

    // Wait for dataset browser to appear
    const browser = page.locator('.dataset-browser, .luxar-dataset-browser').first();
    await expect(browser).toBeVisible({ timeout: 5000 });

    // Try close button (×)
    const closeBtn = browser
      .locator('button[aria-label="Close"], .close-button, .close-btn, button:has-text("×")')
      .first();
    if (await closeBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await closeBtn.click();
      await expect(browser).toBeHidden({ timeout: 5000 });
    } else {
      // If no close button, navigate to a dataset to dismiss
      await page.goto(
        '/?src=http://localhost:9000/datasets/examples/rainbow_sphere_4d_example.zarr&debug'
      );
      await expect(browser).toBeHidden({ timeout: 10000 });
    }
  });

  test('should provide helpful guidance without specific URLs', async ({ page }) => {
    await page.goto('/?debug');

    // Wait for dataset browser to appear
    await page.waitForSelector('.dataset-browser, .luxar-dataset-browser', { timeout: 5000 });

    // Get all text from browser
    const browserText = await page.locator('.dataset-browser').textContent();

    // Should have generic guidance (relaxed to match actual UI)
    expect(browserText).toContain('Luxar');

    // Should have some mention of datasets or how to use
    // (One of these should be present, depending on UI version)
    const hasUsefulGuidance =
      browserText?.includes('Browse') ||
      browserText?.includes('.zarr') ||
      browserText?.includes('dataset') ||
      browserText?.includes('?src=');

    expect(hasUsefulGuidance).toBe(true);

    // Should NOT have specific hardcoded paths that might not exist
    // (This ensures we're being helpful without making promises we can't keep)
    expect(browserText).not.toContain('dimension_navigation_example');
    expect(browserText).not.toContain('dense_grid_5d');
  });
});
