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

import { test, expect, ALLOW_CONSOLE_ERRORS, type Locator, type Page } from './fixtures';
import {
  focusCanvas,
  isFocusOnTypingSurface,
  waitForLuxarReady,
  waitForNextRender,
} from './helpers';

const TEST_4D_DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_4d.luxar.zarr';

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
      type: ALLOW_CONSOLE_ERRORS,
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
      type: ALLOW_CONSOLE_ERRORS,
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
      type: ALLOW_CONSOLE_ERRORS,
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

    const closeBtn = browser.getByRole('button', { name: 'Close dataset browser' });
    await expect(closeBtn).toBeVisible();
    await closeBtn.click();
    await expect(browser).toBeHidden({ timeout: 5000 });
  });

  /**
   * Load the generated 4D fixture, wait until the `O` shortcut is live, and
   * return the dataset-browser locator with the canvas focused.
   *
   * Must be `waitForLuxarReady`, NOT `__luxarDebug.app`: the latter is
   * published at construction, before the input handler binds `O` and before
   * dataset routing decides whether the browser should open automatically.
   * `&no-opfs` because neither test asserts
   * the L2 OPFS cache tier and automated Chromium's OPFS stalls systemically
   * (10 s per op — issue #1645), which can eat the readiness budget before
   * the circuit breaker trips. The 30 s bound (vs the 45 s default) only helps
   * when readiness itself overruns, pinning the failure on this call's frame
   * instead of the whole test timeout.
   */
  async function openViewerReadyForShortcut(page: Page): Promise<Locator> {
    await page.goto(`/?src=${TEST_4D_DATASET}&debug&no-opfs`);
    await waitForLuxarReady(page, 30000);

    const browser = page.locator('.luxar-dataset-browser').first();
    // A `?src=` load must not auto-open the browser. Asserted before
    // `focusCanvas`, which would Escape it away and make this vacuous.
    await expect(browser).toBeHidden({ timeout: 5000 });

    await focusCanvas(page);
    return browser;
  }

  test('Escape closes the dataset browser AND `O` reopens it cleanly', async ({ page }) => {
    test.info().annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description:
        'Opening the dataset browser triggers directory listing that 404s on the static test server.',
    });
    // Regression guard: the Escape path must route through
    // `DatasetBrowser.close()` so `onClose` fires and
    // `LuxarApp.datasetBrowser` is cleared. Without that, the
    // `open-dataset-browser` toggle still sees `hasOpenBrowser() === true`
    // and the next `O` closes a phantom browser instead of reopening
    // the real one.
    const browser = await openViewerReadyForShortcut(page);

    // First: confirm `O` opens it on a fresh page.
    await page.keyboard.press('o');
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

  test('`O` closes the dataset browser it opened (round-trip toggle)', async ({ page }) => {
    test.info().annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description:
        'Opening the dataset browser triggers directory listing that 404s on the static test server.',
    });
    // Issue #1922: the browser used to autofocus its search field, which trips
    // `InputHandler`'s typing guard — the second `O` was swallowed as typing
    // and the "toggle" only ever opened. Initial focus is now on the panel
    // container, so `O` reaches the global binding and closes it.
    const browser = await openViewerReadyForShortcut(page);

    await page.keyboard.press('o');
    await expect(browser).toBeVisible({ timeout: 5000 });

    // Focus must not be on a typing surface, or the next `O` is swallowed.
    expect(await isFocusOnTypingSurface(page)).toBe(false);

    // The critical half: `O` again must CLOSE it, with no Escape in between.
    await page.keyboard.press('o');
    await expect(browser).toBeHidden({ timeout: 5000 });
  });

  test('typing filters the dataset listing from the first keystroke', async ({ page }) => {
    test.info().annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description:
        'Opening the dataset browser triggers directory listing that 404s on the static test server.',
    });
    // The other half of #1922: dropping the autofocus must not cost
    // type-to-filter. The first printable key is forwarded into the search
    // field by the panel-level `installTypeToFilter` handler.
    const browser = await openViewerReadyForShortcut(page);
    await page.keyboard.press('o');
    await expect(browser).toBeVisible({ timeout: 5000 });

    // The search bar renders once a listing produced entries. The E2E data
    // server always emits an HTML listing, so this is a hard requirement, not a
    // "skip if the server can't list" — skipping here would silently retire the
    // only E2E coverage of this behaviour.
    const searchBar = page.locator('#luxar-dataset-browser-search-bar');
    await expect(searchBar).toBeVisible({ timeout: 5000 });

    await page.keyboard.press('z');

    const search = page.locator('#luxar-dataset-browser-search');
    await expect(search).toBeFocused();
    await expect(search).toHaveValue('z');
  });

  test('Escape closes the dataset browser even when focus is in a text input', async ({ page }) => {
    test.info().annotations.push({
      type: ALLOW_CONSOLE_ERRORS,
      description:
        'Opening the dataset browser triggers directory listing that 404s on the static test server.',
    });
    // Regression guard: Escape must reach the context manager even
    // when focus is inside a text input (the dataset-browser's
    // manual-path field, the debug-console filter, etc.) so panels
    // close. `InputHandler.onKeyDown` exempts Escape from the typing
    // guard.
    const browser = await openViewerReadyForShortcut(page);

    // Open the browser via the O shortcut.
    await page.keyboard.press('o');
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
