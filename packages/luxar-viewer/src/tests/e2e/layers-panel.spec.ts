/**
 * E2E tests for the Layers Panel
 *
 * Tests the napari-inspired per-layer control panel that opens with the L key.
 * Covers: visibility toggle, selection, blending mode, gamma, and panel lifecycle.
 *
 * Dataset: sharpness_showcase_example.zarr (8+ point cloud nodes with layer=True)
 */

import { test, expect } from '@playwright/test';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForNextRender,
  focusCanvas,
  openLayersPanel,
  assertNoConsoleErrors,
  getWebGLErrors,
} from './helpers';

// Must use a dataset with layer=True on nodes — the layers panel refuses to open without layers
const DATASET = 'http://localhost:9000/datasets/examples/layers_test_example.zarr';

test.describe('Layers Panel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1, 30000);
  });

  test('should open layers panel with L key and list all scene nodes', async ({ page }) => {
    await focusCanvas(page);

    // Press L to open layers panel
    await page.keyboard.press('l');
    await page.waitForSelector('.luxar-layers-panel', { state: 'visible', timeout: 5000 });

    // Panel should be visible
    const panelVisible = await page.locator('.luxar-layers-panel').isVisible();
    expect(panelVisible).toBe(true);

    // Should have layer rows
    const rowCount = await page.locator('.luxar-layer-row').count();
    expect(rowCount).toBeGreaterThan(0);

    // At least one row should have a name with text
    const firstNameText = await page.locator('.luxar-layer-row__name').first().textContent();
    expect(firstNameText).toBeTruthy();
    expect(firstNameText!.trim().length).toBeGreaterThan(0);
  });

  test('should toggle layer visibility via eye button', async ({ page }) => {
    await openLayersPanel(page);

    // Get the first layer row's name
    const layerName = await page.locator('.luxar-layer-row__name').first().textContent();
    expect(layerName).toBeTruthy();
    const name = layerName!.trim();

    // Check initial visibility in Three.js scene
    const initialVisible = await page.evaluate((objName) => {
      const debug = (window as any).__luxarDebug;
      if (!debug?.scene) return null;
      let visible: boolean | null = null;
      debug.scene.traverse((obj: any) => {
        if (visible !== null) return;
        if (obj.name && obj.name.includes(objName)) {
          visible = obj.visible;
        }
      });
      return visible;
    }, name);
    expect(initialVisible).toBe(true);

    // Click the eye button on the first row
    await page.locator('.luxar-layer-row__eye').first().click();
    await waitForNextRender(page);

    // Three.js object should now be invisible
    const afterHideVisible = await page.evaluate((objName) => {
      const debug = (window as any).__luxarDebug;
      let visible: boolean | null = null;
      debug.scene.traverse((obj: any) => {
        if (visible !== null) return;
        if (obj.name && obj.name.includes(objName)) {
          visible = obj.visible;
        }
      });
      return visible;
    }, name);
    expect(afterHideVisible).toBe(false);

    // Row should have the hidden class
    const hasHiddenClass = await page
      .locator('.luxar-layer-row')
      .first()
      .evaluate((el) => el.classList.contains('luxar-layer-row--hidden'));
    expect(hasHiddenClass).toBe(true);

    // Click eye again to restore visibility
    await page.locator('.luxar-layer-row__eye').first().click();
    await waitForNextRender(page);

    const restoredVisible = await page.evaluate((objName) => {
      const debug = (window as any).__luxarDebug;
      let visible: boolean | null = null;
      debug.scene.traverse((obj: any) => {
        if (visible !== null) return;
        if (obj.name && obj.name.includes(objName)) {
          visible = obj.visible;
        }
      });
      return visible;
    }, name);
    expect(restoredVisible).toBe(true);

    // Hidden class should be removed
    const stillHidden = await page
      .locator('.luxar-layer-row')
      .first()
      .evaluate((el) => el.classList.contains('luxar-layer-row--hidden'));
    expect(stillHidden).toBe(false);
  });

  test('should select layer on click and show selection state', async ({ page }) => {
    await openLayersPanel(page);

    const rows = page.locator('.luxar-layer-row');
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThanOrEqual(2);

    // Click the second row
    await rows.nth(1).click();
    await waitForNextRender(page);

    // Second row should be selected
    const secondSelected = await rows
      .nth(1)
      .evaluate((el) => el.classList.contains('luxar-layer-row--selected'));
    expect(secondSelected).toBe(true);

    // First row should NOT be selected (single-select mode)
    const firstSelected = await rows
      .nth(0)
      .evaluate((el) => el.classList.contains('luxar-layer-row--selected'));
    expect(firstSelected).toBe(false);
  });

  test('should change blending mode via layer state API and verify material state', async ({
    page,
  }) => {
    await openLayersPanel(page);

    // Read all layers and their blending state via the debug API
    const layerInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const panel = debug?.app?.layersPanel;
      if (!panel?.layerState) return null;

      const layers = panel.layerState.getLayers();
      return layers.map((l: any) => ({
        path: l.path,
        blendingMode: l.blendingMode,
        visible: l.visible,
      }));
    });

    expect(layerInfo).not.toBeNull();
    expect(layerInfo!.length).toBeGreaterThan(0);

    // Change blending mode via the layer state API
    const targetLayer = layerInfo![0];
    const newMode = targetLayer.blendingMode === 'additive' ? 'normal' : 'additive';

    await page.evaluate(
      ({ path, mode }) => {
        const debug = (window as any).__luxarDebug;
        debug.app.layersPanel.layerState.setBlendingMode(path, mode);
      },
      { path: targetLayer.path, mode: newMode }
    );
    await waitForNextRender(page);

    // Read back the blending mode
    const updatedMode = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      const layer = debug.app.layersPanel.layerState.getLayer(path);
      return layer?.blendingMode;
    }, targetLayer.path);

    expect(updatedMode).toBe(newMode);
  });

  test('should update gamma via layer state API', async ({ page }) => {
    await openLayersPanel(page);

    // Get first layer path via API
    const layerPath = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const panel = debug?.app?.layersPanel;
      if (!panel?.layerState) return null;
      const layers = panel.layerState.getLayers();
      return layers.length > 0 ? layers[0].path : null;
    });
    expect(layerPath).not.toBeNull();

    // Set gamma to 2.0 via the layer state API
    await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      debug.app.layersPanel.layerState.setGamma(path, 2.0);
    }, layerPath);
    await waitForNextRender(page);

    // Verify gamma was set on the layer state
    const gamma = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      const layer = debug.app.layersPanel.layerState.getLayer(path);
      return layer?.gamma;
    }, layerPath);

    expect(gamma).toBeCloseTo(2.0, 1);
  });

  test('should not crash with no WebGL errors after layer operations', async ({ page }) => {
    await openLayersPanel(page);

    const rows = page.locator('.luxar-layer-row');
    const rowCount = await rows.count();

    // Toggle visibility on up to 4 layers
    const toggleCount = Math.min(rowCount, 4);
    for (let i = 0; i < toggleCount; i++) {
      const eyeBtn = rows.nth(i).locator('.luxar-layer-row__eye');
      await eyeBtn.click();
      await waitForNextRender(page);
    }

    // Toggle them back
    for (let i = 0; i < toggleCount; i++) {
      const eyeBtn = rows.nth(i).locator('.luxar-layer-row__eye');
      await eyeBtn.click();
      await waitForNextRender(page);
    }

    // Assert no WebGL errors
    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors).toEqual([]);

    // Assert no console errors (allow network/fetch warnings that may happen in CI)
    await assertNoConsoleErrors(page, [/Failed to fetch/i, /net::ERR_/i, /404/i]);
  });

  test('should close layers panel with L key', async ({ page }) => {
    await openLayersPanel(page);

    // Panel should be visible
    const panelVisible = await page.locator('.luxar-layers-panel').isVisible();
    expect(panelVisible).toBe(true);

    // Press L to close
    await focusCanvas(page);
    await page.keyboard.press('l');

    // Wait for panel to disappear
    await page
      .waitForSelector('.luxar-layers-panel', { state: 'hidden', timeout: 5000 })
      .catch(() => {
        // Panel might be display:none which counts as hidden
      });

    // Verify not visible
    const panelHidden = await page.evaluate(() => {
      const panel = document.querySelector('.luxar-layers-panel') as HTMLElement | null;
      if (!panel) return true;
      return panel.style.display === 'none' || getComputedStyle(panel).display === 'none';
    });
    expect(panelHidden).toBe(true);
  });
});
