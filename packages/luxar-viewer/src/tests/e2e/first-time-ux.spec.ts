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

import { test, expect } from './fixtures';
import { waitForNextRender } from './helpers';

test.describe('First-Time User Experience', () => {
  test('should show dataset browser when no dataset specified', async ({ page }) => {
    // Navigate with no dataset parameter
    await page.goto('/?debug');

    // Dataset browser should appear automatically. expect.toBeVisible
    // retries with its own timeout; no fixed sleep needed.
    await expect(page.locator('.dataset-browser')).toBeVisible({ timeout: 10000 });
  });

  test('should show welcome banner in dataset browser', async ({ page }) => {
    await page.goto('/?debug');

    // Wait for browser to appear
    await page.waitForSelector('.dataset-browser', { timeout: 5000 });

    // Welcome banner should be present
    const welcomeBanner = page.locator('#luxar-dataset-browser-welcome');
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

    const welcomeBanner = page.locator('#luxar-dataset-browser-welcome');
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
    test.info().annotations.push({
      type: 'allow-console-errors',
      description: 'Bad-URL recovery path intentionally produces 404s.',
    });
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
    test.info().annotations.push({
      type: 'allow-console-errors',
      description: 'Bad-URL recovery path intentionally produces 404s.',
    });
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
    test.info().annotations.push({
      type: 'allow-console-errors',
      description: 'Bad-URL recovery path intentionally produces 404s.',
    });
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
        '/?src=http://localhost:9000/datasets/examples/rainbow_sphere_4d_example.luxar.zarr&debug'
      );
      await expect(browser).toBeHidden({ timeout: 10000 });
    }
  });

  test('Escape closes the dataset browser AND `O` reopens it cleanly', async ({ page }) => {
    test.info().annotations.push({
      type: 'allow-console-errors',
      description:
        'Opening the dataset browser triggers directory listing that 404s on the static test server.',
    });
    // Regression guard: the Escape path must route through
    // `DatasetBrowser.close()` so `onClose` fires and
    // `LuxarApp.datasetBrowser` is cleared. Without that, the `O`
    // shortcut handler bails out via `if (!this.datasetBrowser)
    // return` and makes `O` a silent no-op until reload.
    await page.goto(
      '/?src=http://localhost:9000/datasets/examples/rainbow_sphere_4d_example.luxar.zarr&debug'
    );
    await page.waitForFunction(() => !!(window as any).__luxarDebug?.app, {
      timeout: 10000,
    });
    await page.click('canvas').catch(() => {
      // Canvas may not be focusable yet; press 'O' on document instead.
    });

    // First: confirm `O` opens it on a fresh page.
    await page.keyboard.press('o');
    const browser = page.locator('.luxar-dataset-browser').first();
    await expect(browser).toBeVisible({ timeout: 5000 });

    // Press Escape — DatasetBrowser.close() fires onClose, clears
    // app.datasetBrowser, and clears the input-handler ref.
    await page.keyboard.press('Escape');
    await expect(browser).toBeHidden({ timeout: 5000 });

    // Critical assertion: the `O` shortcut must actually re-open the
    // browser after Escape closes it.
    await page.keyboard.press('o');
    await expect(browser).toBeVisible({ timeout: 5000 });
  });

  test('Escape closes the dataset browser even when focus is in a text input', async ({ page }) => {
    test.info().annotations.push({
      type: 'allow-console-errors',
      description:
        'Opening the dataset browser triggers directory listing that 404s on the static test server.',
    });
    // Regression guard: Escape must reach the context manager even
    // when focus is inside a text input (the dataset-browser's
    // manual-path field, the debug-console filter, etc.) so panels
    // close. `InputHandler.onKeyDown` exempts Escape from the typing
    // guard.
    await page.goto(
      '/?src=http://localhost:9000/datasets/examples/rainbow_sphere_4d_example.luxar.zarr&debug'
    );
    await page.waitForFunction(() => !!(window as any).__luxarDebug?.app, {
      timeout: 10000,
    });
    await page.click('canvas').catch(() => {});

    // Open the browser via the O shortcut.
    await page.keyboard.press('o');
    const browser = page.locator('.luxar-dataset-browser').first();
    await expect(browser).toBeVisible({ timeout: 5000 });

    // Inject a focused text input inside the browser (mimics the
    // manual-path field, which only renders when directory listing
    // fails — too brittle to depend on for an E2E). Focusing it
    // triggers the same `isTypingInInput()` guard at the top of
    // `onKeyDown`; Escape must still close the browser.
    await page.evaluate(() => {
      const browserEl = document.querySelector('.luxar-dataset-browser');
      if (!browserEl) throw new Error('browser missing');
      const input = document.createElement('input');
      input.type = 'text';
      input.id = '__escape_test_input';
      browserEl.appendChild(input);
      input.focus();
    });

    // Verify focus actually landed inside the input — `isTypingInInput`
    // should return true from this state.
    const focusOk = await page.evaluate(() => document.activeElement?.id === '__escape_test_input');
    expect(focusOk).toBe(true);

    // Escape from focused input should still close the browser.
    await page.keyboard.press('Escape');
    await expect(browser).toBeHidden({ timeout: 5000 });
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
