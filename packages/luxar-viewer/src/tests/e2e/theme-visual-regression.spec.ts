/**
 * Theme Visual Regression Tests
 *
 * Comprehensive visual regression testing for all UI components across all themes.
 * Ensures consistent appearance and proper theme switching functionality.
 *
 * Tests each component in:
 * - Dark theme (default)
 * - Light theme
 * - High contrast theme
 */

import { test, expect, type Page } from '@playwright/test';

// Themes to test
const THEMES = ['dark', 'light', 'frosted-glass'] as const;

// Helper to wait for theme application
async function waitForTheme(page: Page, themeId: string): Promise<void> {
  // Wait for data-theme attribute to be set
  await page.waitForFunction(
    (theme) => document.documentElement.getAttribute('data-theme') === theme,
    themeId, // pass theme as argument
    { timeout: 2000 }
  );

  // Wait a bit for CSS to apply
  await page.waitForTimeout(200);
}

/**
 * Test error dialog in all themes
 */
for (const theme of THEMES) {
  test(`error dialog - ${theme} theme`, async ({ page }) => {
    // Navigate with theme and invalid dataset to trigger error
    await page.goto(`/?src=invalid-dataset.zarr&theme=${theme}`);

    // Wait for theme to be applied
    await waitForTheme(page, theme);

    // Wait for error dialog to appear
    const errorDialog = page.locator('.luxar-error-dialog');
    await expect(errorDialog).toBeVisible({ timeout: 5000 });

    // Take screenshot for visual regression
    await expect(errorDialog).toHaveScreenshot(`error-dialog-${theme}.png`);
  });
}

/**
 * Test help overlay in all themes
 */
for (const theme of THEMES) {
  test(`help overlay - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug`);
    await waitForTheme(page, theme);

    // Trigger help overlay with H key
    await page.keyboard.press('h');

    const helpOverlay = page.locator('.luxar-help-overlay');
    await expect(helpOverlay).toBeVisible({ timeout: 2000 });

    // Take screenshot
    await expect(helpOverlay).toHaveScreenshot(`help-overlay-${theme}.png`);
  });
}

/**
 * Test dimension sliders in all themes
 * Note: Use 5D dataset - sliders appear automatically for nD datasets
 */
for (const theme of THEMES) {
  test(`dimension sliders - ${theme} theme`, async ({ page }) => {
    // Use a 5D dataset - dimension sliders appear automatically for nD data
    const testDataUrl =
      'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug`);
    await waitForTheme(page, theme);

    // Wait for app to be fully initialized and data to load
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug && debug.getState && debug.getState().initialized;
      },
      null, // no arguments
      { timeout: 45000 }
    );

    // For 5D datasets, dimension sliders appear automatically (no need to press 'n')
    // The 'n' key toggles visibility - so pressing it would HIDE the sliders
    const dimensionSliders = page.locator('.luxar-dimension-sliders');
    await expect(dimensionSliders).toBeVisible({ timeout: 5000 });

    // Take screenshot
    await expect(dimensionSliders).toHaveScreenshot(`dimension-sliders-${theme}.png`);
  });
}

/**
 * Test data loading monitor (compact view) in all themes
 */
for (const theme of THEMES) {
  test(`data monitor compact - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug`);
    await waitForTheme(page, theme);

    // Wait for app to be fully initialized
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug && debug.getState && debug.getState().initialized;
      },
      null, // no arguments
      { timeout: 45000 }
    );

    // Press M to cycle to compact view (data monitor starts hidden, first M shows compact)
    await page.keyboard.press('m');

    // Wait for compact monitor to appear
    await page.waitForSelector('.luxar-data-monitor--compact', { timeout: 5000 });

    const dataMonitor = page.locator('.luxar-data-monitor--compact');
    await expect(dataMonitor).toBeVisible();

    // Take screenshot
    await expect(dataMonitor).toHaveScreenshot(`data-monitor-compact-${theme}.png`);
  });
}

/**
 * Test data loading monitor (expanded view) in all themes
 */
for (const theme of THEMES) {
  test(`data monitor expanded - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug`);
    await waitForTheme(page, theme);

    // Wait for app to be fully initialized
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug && debug.getState && debug.getState().initialized;
      },
      null, // no arguments
      { timeout: 45000 }
    );

    // Press M twice to cycle to expanded view
    await page.keyboard.press('m');
    await page.waitForTimeout(300);
    await page.keyboard.press('m');

    // Wait for expanded monitor to appear
    await page.waitForSelector('.luxar-data-monitor--expanded', { timeout: 2000 });

    const dataMonitor = page.locator('.luxar-data-monitor--expanded');
    await expect(dataMonitor).toBeVisible();

    // Take screenshot
    await expect(dataMonitor).toHaveScreenshot(`data-monitor-expanded-${theme}.png`);
  });
}

/**
 * Test debug console in all themes
 */
for (const theme of THEMES) {
  test(`debug console - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug`);
    await waitForTheme(page, theme);

    // Trigger debug console with Ctrl+L
    await page.keyboard.press('Control+l');

    const debugConsole = page.locator('.debug-console-panel');
    await expect(debugConsole).toBeVisible({ timeout: 2000 });

    // Take screenshot
    await expect(debugConsole).toHaveScreenshot(`debug-console-${theme}.png`);
  });
}

/**
 * Test dataset browser in all themes
 */
for (const theme of THEMES) {
  test(`dataset browser - ${theme} theme`, async ({ page }) => {
    await page.goto(`/?theme=${theme}&debug`);
    await waitForTheme(page, theme);

    // Trigger dataset browser with O key
    await page.keyboard.press('o');

    const datasetBrowser = page.locator('.luxar-dataset-browser');
    await expect(datasetBrowser).toBeVisible({ timeout: 2000 });

    // Take screenshot
    await expect(datasetBrowser).toHaveScreenshot(`dataset-browser-${theme}.png`);
  });
}

/**
 * Test theme switching behavior
 */
test('theme switching updates all CSS variables', async ({ page }) => {
  const testDataUrl =
    'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
  await page.goto(`/?src=${testDataUrl}&debug`);

  // Check dark theme variables
  await page.evaluate(() => {
    // Access ThemeManager via window (it's imported in main.ts)
    const themeManager =
      (window as any).ThemeManager?.getInstance?.() || (window as any).__luxarDebug?.app;
    if (themeManager) themeManager.setTheme?.('dark');
  });
  await waitForTheme(page, 'dark');

  const darkBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  expect(darkBg.trim()).toBe('#111111');

  // Switch to light theme
  await page.evaluate(() => {
    const themeManager = (window as any).ThemeManager?.getInstance?.();
    if (themeManager) themeManager.setTheme('light');
  });
  await waitForTheme(page, 'light');

  const lightBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  expect(lightBg.trim()).toBe('#ffffff');

  // Switch to frosted-glass theme
  await page.evaluate(() => {
    const themeManager = (window as any).ThemeManager?.getInstance?.();
    if (themeManager) themeManager.setTheme('frosted-glass');
  });
  await waitForTheme(page, 'frosted-glass');

  const frostedBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  // Frosted glass uses rgba(255, 255, 255, 0.15) for subtle glass tint
  expect(frostedBg.trim()).toBe('rgba(255, 255, 255, 0.15)');

  // Switch to liquid-glass theme
  await page.evaluate(() => {
    const themeManager = (window as any).ThemeManager?.getInstance?.();
    if (themeManager) themeManager.setTheme('liquid-glass');
  });
  await waitForTheme(page, 'liquid-glass');

  const liquidBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  // Liquid glass uses rgba(255, 255, 255, 0.18) for subtle glass tint
  expect(liquidBg.trim()).toBe('rgba(255, 255, 255, 0.18)');
});

/**
 * Test theme persistence across page reloads
 */
test('theme persists across page reloads', async ({ page }) => {
  // Set light theme (use dataset to avoid browser dialog)
  const testDataUrl =
    'http://localhost:9000/packages/luxar/examples/dimension_sliders_5d_example.zarr';
  await page.goto(`/?src=${testDataUrl}&debug`);

  await page.evaluate(() => {
    const themeManager = (window as any).ThemeManager?.getInstance?.();
    if (themeManager) themeManager.setTheme('light');
  });
  await waitForTheme(page, 'light');

  // Reload page
  await page.reload();

  // Wait for theme to be restored after reload
  await waitForTheme(page, 'light');

  // Check that light theme is still active via data-theme attribute
  const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  expect(theme).toBe('light');
});

/**
 * Test URL parameter theme override
 */
test('URL parameter sets initial theme', async ({ page }) => {
  await page.goto('/?theme=frosted-glass&debug');

  // Wait for theme to be applied
  await waitForTheme(page, 'frosted-glass');

  // Verify theme is set via data-theme attribute
  const theme = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
  expect(theme).toBe('frosted-glass');

  // Verify CSS variable
  const bgColor = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  expect(bgColor.trim()).toBe('rgba(255, 255, 255, 0.15)');
});
