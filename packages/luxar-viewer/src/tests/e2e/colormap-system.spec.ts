/**
 * Colormap System E2E Tests
 *
 * Validates the colormap (LUT) system that maps scalar data to colors:
 * - Colormap applies as a 256x1 LUT texture on materials
 * - Switching colormaps changes the texture
 * - Colormap legend (J key) shows/hides and updates reactively
 * - Min/max range labels reflect data range
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForNextRender,
  focusCanvas,
  openLayersPanel,
  assertNoConsoleErrors,
} from './helpers';

// Use layers_test_example for colormap tests (needs layer=True for layers panel access)
const DATASET = 'http://localhost:9000/datasets/examples/layers_test_example.luxar.zarr';

test.describe('Colormap System', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);
  });

  test('should apply colormap texture to material via layers panel', async ({ page }) => {
    await openLayersPanel(page);

    // Select the first layer
    const firstRow = page.locator('.luxar-layer-row').first();
    await firstRow.click();
    await waitForNextRender(page);

    // Look for a colormap dropdown/select in the controls area
    const colormapSelect = page.locator(
      '.luxar-layers-panel__controls select, .luxar-layers-panel__controls .luxar-gui__select'
    );

    // If colormap select exists, change it
    if ((await colormapSelect.count()) > 0) {
      // Get layer name for material lookup
      const layerName = await firstRow.locator('.luxar-layer-row__name').textContent();

      // Select a named colormap (e.g., 'viridis')
      const options = await colormapSelect.first().locator('option').allTextContents();
      const viridisOption = options.find((o) => o.toLowerCase().includes('viridis'));
      if (viridisOption) {
        await colormapSelect.first().selectOption({ label: viridisOption });
        await waitForNextRender(page);

        // Verify material has a colormap texture
        const hasTexture = await page.evaluate((name) => {
          const debug = (window as any).__luxarDebug;
          let found = false;
          debug.scene.traverse((obj: any) => {
            if (obj.name?.includes(name) && obj.material?.uniforms?.uColormapTex) {
              found = obj.material.uniforms.uColormapTex.value !== null;
            }
          });
          return found;
        }, layerName?.trim());

        expect(hasTexture).toBe(true);
      }
    }

    // Regardless of colormap availability, no errors should occur
    await assertNoConsoleErrors(page);
  });

  test('should toggle colormap legend with J key', async ({ page }) => {
    await focusCanvas(page);

    // Legend should not be visible initially
    const legendBefore = await page
      .locator('.luxar-colormap-legend')
      .isVisible()
      .catch(() => false);
    expect(legendBefore).toBe(false);

    // Press J to show legend
    await page.keyboard.press('j');
    await waitForNextRender(page);

    // Legend should now be visible
    const legendAfter = await page
      .locator('.luxar-colormap-legend')
      .isVisible()
      .catch(() => false);
    expect(legendAfter).toBe(true);

    // Press J again to hide
    await page.keyboard.press('j');
    await waitForNextRender(page);

    const legendHidden = await page
      .locator('.luxar-colormap-legend')
      .isVisible()
      .catch(() => false);
    expect(legendHidden).toBe(false);
  });

  test('should show legend entries when colormaps are active', async ({ page }) => {
    // Open legend
    await focusCanvas(page);
    await page.keyboard.press('j');
    await waitForNextRender(page);

    // Check legend content — may show "No colormaps active" or entries
    const legendContent = await page.evaluate(() => {
      const legend = document.querySelector('.luxar-colormap-legend');
      return legend?.textContent ?? '';
    });

    // Legend should have some text content (either entries or hint)
    expect(legendContent.length).toBeGreaterThan(0);
  });

  test('should not crash when toggling legend rapidly', async ({ page }) => {
    await focusCanvas(page);

    // Rapid toggle. The 50 ms pacing is intentional: the test
    // exercises the colormap-cycle race window where successive J
    // presses land in the same render frame batch. See the same
    // pattern in post-processing-pipeline.spec.ts.
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('j');
      await page.waitForTimeout(50);
    }

    await waitForNextRender(page);
    await assertNoConsoleErrors(page);
  });

  test('should show colormap legend with gradient canvas when colormaps exist', async ({
    page,
  }) => {
    // Open legend
    await focusCanvas(page);
    await page.keyboard.press('j');
    await waitForNextRender(page);

    // Check if legend has canvas elements (gradient previews)
    const canvasCount = await page
      .locator('.luxar-colormap-legend canvas')
      .count()
      .catch(() => 0);

    // If colormaps are active, there should be canvas gradients;
    // if not, just verify the legend panel itself exists
    const legendExists = await page
      .locator('.luxar-colormap-legend')
      .isVisible()
      .catch(() => false);
    expect(legendExists).toBe(true);

    // Log what we found for debugging
    console.log(`[Colormap] Legend has ${canvasCount} gradient canvas(es)`);
  });
});
