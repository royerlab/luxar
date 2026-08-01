/**
 * Theme Visual Regression Tests
 *
 * Comprehensive visual regression testing for all UI components across all themes.
 * Ensures consistent appearance and proper theme switching functionality.
 *
 * Tests each component in:
 * - Dark theme
 * - Light theme
 * - Frosted Glass theme (default)
 */

import { test, expect, type Page } from './fixtures';
import { waitForNextRender } from './helpers';

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

  // Intentional fixed sleep: the data-theme attribute is set above, but
  // CSS variables that drive ::after / backdrop-filter / transitions need
  // a moment to apply on next paint. Frosted-glass and animation-driven
  // themes especially benefit from this small settle window before any
  // screenshot is taken.
  await page.waitForTimeout(200);
}

/**
 * Test error dialog in all themes.
 *
 * The dialog is driven directly through the debug-only
 * `__luxarDebug.showError(message)` hook (registered in
 * `bootstrap.ts`) rather than via a known-bad `?src=`. URL routing has
 * its own behaviours — relative-path failures fall back to the dataset
 * browser since `app.ts:shouldShowBrowser`'s zarr-metadata HEAD probes —
 * and coupling the dialog's visual regression to those routing semantics
 * was the original cause of false test failures when the routing
 * improved. Driving the dialog directly tests exactly what this spec
 * cares about: the dialog's appearance per theme.
 */
for (const theme of THEMES) {
  test(`@visual error dialog - ${theme} theme`, async ({ page }) => {
    // Navigate with debug enabled so __luxarDebug.showError is exposed.
    await page.goto(`/?theme=${theme}&debug&dpr=1`);

    // Wait for the theme to be applied AND the debug interface to be
    // populated. The bootstrap seeds __luxarDebug before init() runs, so
    // showError is available before the first paint.
    await waitForTheme(page, theme);
    await page.waitForFunction(
      () => typeof (window as any).__luxarDebug?.showError === 'function',
      null,
      { timeout: 10_000 }
    );

    // Trigger the same message bootstrap.ts uses on init failure so the
    // dialog content matches the production error path.
    await page.evaluate(() => {
      (window as any).__luxarDebug?.showError?.(
        'Failed to start the application. Please check the console for details.'
      );
    });

    // Wait for the dialog to render. luxar-error-dialog is the styling
    // class; error-message is kept as an E2E hook for compatibility.
    const errorDialog = page.locator('.luxar-error-dialog');
    await expect(errorDialog).toBeVisible({ timeout: 5000 });

    // Frosted-glass uses backdrop-filter blur + an entrance animation; let
    // it settle so the screenshot is stable.
    if (theme === 'frosted-glass') {
      await page.waitForTimeout(500);
    }

    await expect(errorDialog).toHaveScreenshot(`error-dialog-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
      // The example URL embeds window.location.origin, so the pixels vary
      // with the dev-server port (5173 vs worktree/CI ports). Mask it.
      mask: [page.locator('.luxar-error-dialog__guidance-code')],
    });
  });
}

/**
 * Test help overlay in all themes
 */
for (const theme of THEMES) {
  test(`@visual help overlay - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    // Trigger help overlay with H key
    await page.keyboard.press('h');

    const helpOverlay = page.locator('.luxar-help-overlay');
    await expect(helpOverlay).toBeVisible({ timeout: 2000 });

    // Take screenshot
    await expect(helpOverlay).toHaveScreenshot(`help-overlay-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test dimension sliders in all themes
 * Note: Use 4D dataset to ensure sliders are shown
 */
for (const theme of THEMES) {
  test(`@visual dimension sliders - ${theme} theme`, async ({ page }) => {
    // Use a 5D dataset to ensure dimension sliders appear (use local for speed)
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
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

    // Wait for dimension sliders to be created (they're shown by default for nD datasets)
    const dimensionSliders = page.locator('.luxar-dimension-sliders');
    await expect(dimensionSliders).toBeAttached({ timeout: 5000 });

    // For nD datasets, sliders are shown by default - ensure they're visible
    // (pressing 'n' would toggle them off)
    const isVisible = await page.evaluate(() => {
      const el = document.querySelector('.luxar-dimension-sliders') as HTMLElement;
      return el && el.style.display !== 'none';
    });

    // If hidden (shouldn't happen for 5D), press 'n' to show
    if (!isVisible) {
      await page.keyboard.press('n');
    }

    await expect(dimensionSliders).toBeVisible({ timeout: 2000 });

    // Take screenshot
    await expect(dimensionSliders).toHaveScreenshot(`dimension-sliders-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test data loading monitor (mini view) in all themes
 */
for (const theme of THEMES) {
  test(`@visual data monitor mini - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
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

    // Press M to cycle to mini/compact view (hidden → compact)
    await page.keyboard.press('m');

    // Wait for compact monitor to appear (class is --compact, not --mini)
    await page.waitForSelector('.luxar-data-monitor--compact', { timeout: 2000 });

    const dataMonitor = page.locator('.luxar-data-monitor--compact');
    await expect(dataMonitor).toBeVisible();

    // Take screenshot
    await expect(dataMonitor).toHaveScreenshot(`data-monitor-compact-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test data loading monitor (expanded view) in all themes
 */
for (const theme of THEMES) {
  test(`@visual data monitor expanded - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
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
    await waitForNextRender(page);
    await page.keyboard.press('m');

    // Wait for expanded monitor to appear
    await page.waitForSelector('.luxar-data-monitor--expanded', { timeout: 2000 });

    const dataMonitor = page.locator('.luxar-data-monitor--expanded');
    await expect(dataMonitor).toBeVisible();

    // Take screenshot
    await expect(dataMonitor).toHaveScreenshot(`data-monitor-expanded-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test debug console in all themes
 */
for (const theme of THEMES) {
  test(`@visual debug console - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    // Wait for app to be initialized before keyboard input
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug && debug.getState && debug.getState().initialized;
      },
      null,
      { timeout: 45000 }
    );

    // Trigger debug console with Ctrl+L (correct class is luxar-debug-console)
    await page.keyboard.press('Control+l');

    const debugConsole = page.locator('.luxar-debug-console');
    await expect(debugConsole).toBeVisible({ timeout: 2000 });

    // Take screenshot
    await expect(debugConsole).toHaveScreenshot(`debug-console-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test dataset browser in all themes
 */
for (const theme of THEMES) {
  test(`@visual dataset browser - ${theme} theme`, async ({ page }) => {
    await page.goto(`/?theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    // The dataset browser auto-opens when no dataset is loaded (welcome UX).
    // The 'o' key TOGGLES it, so only press it if it hasn't already opened —
    // pressing 'o' on an already-open browser would toggle it closed. By the
    // time waitForTheme resolves, the auto-show has fired, so this is
    // deterministic while still opening the browser if auto-show is disabled.
    const datasetBrowser = page.locator('.luxar-dataset-browser');
    if (!(await datasetBrowser.isVisible())) {
      await page.keyboard.press('o');
    }
    await expect(datasetBrowser).toBeVisible({ timeout: 2000 });

    // Extra wait for frosted-glass theme which has animation/blur effects
    if (theme === 'frosted-glass') {
      await page.waitForTimeout(500);
    }

    // Take screenshot
    await expect(datasetBrowser).toHaveScreenshot(`dataset-browser-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test the control rail in all themes.
 *
 * The rail is the always-visible left-edge activity bar (PR #452 made
 * Navigation/Settings/Performance rail-native). It idle-dims after ~2.6s,
 * so wake it with a pointer move just before the screenshot to pin the
 * awake (full-opacity) state deterministically.
 */
for (const theme of THEMES) {
  test(`@visual control rail - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    const rail = page.locator('.luxar-control-rail');
    await expect(rail).toBeVisible({ timeout: 5000 });

    // Let the scene settle (the rail is a glass surface — canvas content
    // bleeds through its backdrop blur, so the background must be stable).
    await page.waitForTimeout(1000);

    // Wake the rail (pointer movement over the canvas brightens it) without
    // hovering the rail itself, which would show a button tooltip.
    await page.mouse.move(640, 360);
    await waitForNextRender(page);

    if (theme === 'frosted-glass') {
      await page.waitForTimeout(500);
    }

    await expect(rail).toHaveScreenshot(`control-rail-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test the rendering-controls panel in all themes.
 *
 * After PR #452 the panel is purely image/appearance: Camera, HDR,
 * Anti-Aliasing, Post-Processing (Navigation/Theme/Performance moved to
 * rail popovers) — this baseline pins that slimmed layout per theme.
 */
for (const theme of THEMES) {
  test(`@visual rendering controls panel - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    // Let the scene settle so the panel's near/far readouts are stable.
    await page.waitForTimeout(1000);

    await page.keyboard.press('r');

    // Several .luxar-gui elements exist (e.g. the hidden recording panel);
    // target the one titled "Rendering Controls".
    const panel = page.locator('.luxar-gui', {
      has: page.locator('.luxar-gui__title', { hasText: 'Rendering Controls' }),
    });
    await expect(panel).toBeVisible({ timeout: 2000 });

    if (theme === 'frosted-glass') {
      await page.waitForTimeout(500);
    }

    await expect(panel).toHaveScreenshot(`rendering-controls-panel-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test the Navigation rail popover in all themes.
 *
 * Right-clicking the Navigation rail button opens a panel popover with the
 * segmented mode selector (Orbit/Fly/Ortho) + the current mode's parameters
 * (PR #452). Dispatched programmatically — Playwright's right-click does not
 * reliably fire `contextmenu` for this delegated handler.
 */
for (const theme of THEMES) {
  test(`@visual navigation popover - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    // Let the scene settle (glass popover blurs the canvas behind it).
    await page.waitForTimeout(1000);

    await page.evaluate(() => {
      document
        .querySelector('[data-rail-id="nav"]')
        ?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });

    const popover = page.locator('.luxar-control-rail__popover');
    await expect(popover).toBeVisible({ timeout: 2000 });

    if (theme === 'frosted-glass') {
      await page.waitForTimeout(500);
    }

    await expect(popover).toHaveScreenshot(`navigation-popover-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test the Home rail popover in all themes.
 *
 * Right-clicking the Home rail button opens the reset-actions popover (fit
 * scene / center on origin / reset dimensions / reset rendering) — the one
 * rail popover with plain action rows instead of a nested GUI. Dispatched
 * programmatically for the same delegated-handler reason as above.
 */
for (const theme of THEMES) {
  test(`@visual home popover - ${theme} theme`, async ({ page }) => {
    const testDataUrl =
      'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.luxar.zarr';
    await page.goto(`/?src=${testDataUrl}&theme=${theme}&debug&dpr=1`);
    await waitForTheme(page, theme);

    // Let the scene settle (glass popover blurs the canvas behind it).
    await page.waitForTimeout(1000);

    await page.evaluate(() => {
      document
        .querySelector('[data-rail-id="home"]')
        ?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });

    const popover = page.locator('.luxar-control-rail__popover');
    await expect(popover).toBeVisible({ timeout: 2000 });

    if (theme === 'frosted-glass') {
      await page.waitForTimeout(500);
    }

    await expect(popover).toHaveScreenshot(`home-popover-${theme}.png`, {
      maxDiffPixelRatio: 0.1,
      threshold: 0.3,
    });
  });
}

/**
 * Test theme switching behavior via URL parameter
 * Note: ThemeManager is not exposed to window, so we test via URL params
 */
test('theme switching updates all CSS variables', async ({ page }) => {
  // Test dark theme
  await page.goto('/?theme=dark&debug&dpr=1');
  await waitForTheme(page, 'dark');

  const darkBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  expect(darkBg.trim()).toBe('#111111');

  // Test light theme
  await page.goto('/?theme=light&debug&dpr=1');
  await waitForTheme(page, 'light');

  const lightBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  expect(lightBg.trim()).toBe('#ffffff');

  // Test frosted-glass theme
  await page.goto('/?theme=frosted-glass&debug&dpr=1');
  await waitForTheme(page, 'frosted-glass');

  const frostedBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  // Frosted glass uses rgba(255, 255, 255, 0.15) for subtle glass tint
  expect(frostedBg.trim()).toBe('rgba(255, 255, 255, 0.15)');

  // Test liquid-glass theme
  await page.goto('/?theme=liquid-glass&debug&dpr=1');
  await waitForTheme(page, 'liquid-glass');

  const liquidBg = await page.evaluate(() =>
    getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary')
  );
  // Liquid glass uses rgba(255, 255, 255, 0.18) for subtle glass tint
  expect(liquidBg.trim()).toBe('rgba(255, 255, 255, 0.18)');
});

/**
 * Test theme persistence across page reloads
 * Uses URL parameter to set theme, which persists to localStorage
 */
test('theme persists across page reloads', async ({ page }) => {
  // Navigate with ?theme=light to set the theme (this persists to localStorage)
  await page.goto('/?theme=light&debug&dpr=1');
  await waitForTheme(page, 'light');

  // Verify light theme is active
  const themeBeforeReload = await page.evaluate(() =>
    document.documentElement.getAttribute('data-theme')
  );
  expect(themeBeforeReload).toBe('light');

  // Reload page WITHOUT theme parameter - should restore from localStorage
  await page.goto('/?debug&dpr=1');

  // Wait for theme to be restored after reload
  await waitForTheme(page, 'light');

  // Check that light theme is still active via data-theme attribute
  const themeAfterReload = await page.evaluate(() =>
    document.documentElement.getAttribute('data-theme')
  );
  expect(themeAfterReload).toBe('light');
});

/**
 * Test URL parameter theme override
 */
test('URL parameter sets initial theme', async ({ page }) => {
  await page.goto('/?theme=frosted-glass&debug&dpr=1');

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
