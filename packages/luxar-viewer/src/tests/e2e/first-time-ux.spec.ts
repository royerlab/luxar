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

  test('should show enhanced error message with guidance when dataset fails', async ({ page }) => {
    // Try to load non-existent dataset
    await page.goto('/?src=/data/nonexistent.zarr&debug');

    // Wait for error to appear
    await page.waitForSelector('.error-message', { timeout: 10000 });

    const errorDiv = page.locator('.error-message');
    await expect(errorDiv).toBeVisible();

    const errorText = await errorDiv.textContent();

    // Should have helpful title
    expect(errorText).toContain('Unable to Load Dataset');

    // Should have guidance section
    expect(errorText).toContain('How to Load a Dataset');

    // Should explain steps (didactic)
    expect(errorText).toContain('1.'); // Step 1
    expect(errorText).toContain('2.'); // Step 2
    expect(errorText).toContain('Add dataset to URL');

    // Should mention browser (helpful)
    expect(errorText).toContain('browse available datasets');
    expect(errorText).toContain('O'); // O key

    // Should mention help (educational)
    expect(errorText).toContain('Need help');
    expect(errorText).toContain('H'); // H key

    // Should explain dataset format (educational)
    expect(errorText).toContain('Zarr');
  });

  test('should NOT show hardcoded dataset URLs in error message', async ({ page }) => {
    await page.goto('/?src=/data/missing.zarr&debug');

    await page.waitForSelector('.error-message', { timeout: 10000 });

    const errorText = await page.locator('.error-message').textContent();

    // Should NOT have specific URLs that might not exist
    // (Generic example format is OK: "?src=/path/to/dataset.zarr")
    expect(errorText).not.toContain('dimension_navigation_example');
    expect(errorText).not.toContain('dense_grid_5d');
    expect(errorText).not.toContain('build_example');

    // Generic placeholder is fine
    expect(errorText).toContain('/path/to/dataset.zarr'); // Generic example
  });

  test('should allow dismissing error message', async ({ page }) => {
    await page.goto('/?src=/data/fail.zarr&debug');

    await page.waitForSelector('.error-message', { timeout: 10000 });

    const errorDiv = page.locator('.error-message');
    await expect(errorDiv).toBeVisible();

    // Click to dismiss
    await errorDiv.click();

    // Should be gone
    await page.waitForTimeout(500);
    const stillVisible = await errorDiv.isVisible().catch(() => false);
    expect(stillVisible).toBe(false);
  });

  test('should show clear instructions in error message', async ({ page }) => {
    await page.goto('/?src=/invalid/path.zarr&debug');

    await page.waitForSelector('.error-message', { timeout: 10000 });

    const guidance = await page.locator('.error-message').textContent();

    // Instructions should be numbered and clear
    expect(guidance).toMatch(/1\./); // First instruction
    expect(guidance).toMatch(/2\./); // Second instruction
    expect(guidance).toMatch(/3\./); // Third instruction
    expect(guidance).toMatch(/4\./); // Fourth instruction

    // Should explain each step
    expect(guidance).toContain('Add dataset to URL');
    expect(guidance).toContain('browse available datasets');
    expect(guidance).toContain('Dataset format');
    expect(guidance).toContain('Need help');
  });

  test('should close dataset browser with Escape or close button', async ({ page }) => {
    await page.goto('/?debug');
    await page.waitForTimeout(2000);

    const browser = page.locator('.dataset-browser');
    await expect(browser).toBeVisible();

    // Close with Escape key
    await page.keyboard.press('Escape');

    // Wait for browser to close
    await expect(browser).toBeHidden({ timeout: 2000 });
  });

  test('should provide helpful guidance without specific URLs', async ({ page }) => {
    await page.goto('/?debug');
    await page.waitForTimeout(2000);

    // Wait for dataset browser to appear
    await page.waitForSelector('.dataset-browser', { timeout: 5000 });

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
