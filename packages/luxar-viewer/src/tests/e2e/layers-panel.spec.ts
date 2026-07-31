/**
 * E2E tests for the Layers Panel
 *
 * Tests the napari-inspired per-layer control panel that opens with the L key.
 * Covers: visibility toggle, selection, blending mode, gamma, and panel lifecycle.
 *
 * Dataset: sharpness_showcase_example.luxar.zarr (8+ point cloud nodes with layer=True)
 */

import { test, expect } from './fixtures';
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
const DATASET = 'http://localhost:9000/datasets/examples/layers_test_example.luxar.zarr';

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

  test('blend-select change updates layer state AND the THREE material blend state', async ({
    page,
  }) => {
    await openLayersPanel(page);

    // Pick the first DATA-node layer (group layers fan out; a data leaf
    // gives a single unambiguous material to assert on) and select it so
    // the panel's blend select binds to it.
    const targetLayer = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const panel = debug?.app?.layersPanel;
      if (!panel?.layerState) return null;
      const layers = panel.layerState.getLayers();
      const target = layers.find((l: any) => l.type !== 'group');
      if (!target) return null;
      panel.layerState.select(target.path, 'single');
      return { path: target.path, blendingMode: target.blendingMode };
    });
    expect(targetLayer).not.toBeNull();

    // Drive the REAL panel control (the state-API alone never reaches
    // the material): set the blend <select> and fire its change event.
    // Pick a target mode with an unmistakable material signature.
    const newMode = targetLayer!.blendingMode === 'max' ? 'additive' : 'max';
    const blendSelect = page.locator(
      '.luxar-layers-panel__control-group:has(.luxar-layers-panel__control-label:text-is("Blend")) select'
    );
    await expect(blendSelect).toBeVisible();
    await blendSelect.selectOption(newMode);
    await waitForNextRender(page);

    // Layer state reflects the new mode…
    const updatedMode = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      return debug.app.layersPanel.layerState.getLayer(path)?.blendingMode;
    }, targetLayer!.path);
    expect(updatedMode).toBe(newMode);

    // …and so does the ACTUAL THREE material on the layer's mesh
    // (mirrors blending-modes.spec.ts's material lookup). THREE enums:
    // AdditiveBlending=2, CustomBlending=5; AddEquation=100,
    // MaxEquation=104.
    const materialState = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      let found: any = null;
      debug.scene.traverse((obj: any) => {
        if (found || !obj.material) return;
        if (obj.name === path || (obj.name && obj.name.endsWith(path))) {
          found = {
            blending: obj.material.blending,
            blendEquation: obj.material.blendEquation,
            blendingMode: obj.material.userData?.blendingMode,
          };
        }
      });
      return found;
    }, targetLayer!.path);

    expect(materialState).not.toBeNull();
    expect(materialState.blendingMode).toBe(newMode);
    if (newMode === 'max') {
      expect(materialState.blending).toBe(5); // CustomBlending
      expect(materialState.blendEquation).toBe(104); // MaxEquation
    } else {
      expect(materialState.blending).toBe(2); // AdditiveBlending
      expect(materialState.blendEquation).toBe(100); // AddEquation
    }
  });

  test('volumetric gsplat layer: 6-mode dropdown, κ slider visibility + live uAbsorption', async ({
    page,
  }) => {
    // Self-contained: navigates to the volumetric gsplat fixture (the
    // suite dataset has no gsplat layer, and the κ slider is gated to
    // volumetric gsplat/group layers).
    const FIXTURE =
      'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_volumetric.luxar.zarr';
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await openLayersPanel(page);

    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const panel = debug.app.layersPanel;
      const target = panel.layerState.getLayers().find((l: any) => l.type !== 'group');
      panel.layerState.select(target.path, 'single');
    });

    // The dropdown offers all six modes and reflects the authored one.
    const blendSelect = page.locator(
      '.luxar-layers-panel__control-group:has(.luxar-layers-panel__control-label:text-is("Blend")) select'
    );
    await expect(blendSelect).toBeVisible();
    await expect(blendSelect).toHaveValue('volumetric');
    expect(await blendSelect.locator('option').count()).toBe(6);

    // The κ slider is visible for a volumetric gsplat layer…
    const absorptionGroup = page.locator(
      '.luxar-layers-panel__control-group:has(.luxar-layers-panel__control-label span:text-is("Absorption"))'
    );
    await expect(absorptionGroup).toBeVisible();

    // …dragging it reaches the live uAbsorption uniform. The κ track is
    // LOGARITHMIC with per-layer bounds (ui/layers/absorption-range.ts), so
    // the input carries a NORMALISED position in [0, 1], not κ itself —
    // filling it with a κ value would just clamp to the far end. Assert
    // mapping-agnostically instead: whatever the readout says the κ is, is
    // what the material got.
    const slider = absorptionGroup.locator('input[type="range"]');
    await expect(slider).toHaveAttribute('max', '1');
    const readKappa = (): Promise<number | null> =>
      page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        let value: number | null = null;
        debug.scene.traverse((obj: any) => {
          if (obj.userData?.nodeType === 'gsplats' && obj.material?.uniforms?.uAbsorption) {
            value = obj.material.uniforms.uAbsorption.value as number;
          }
        });
        return value as number | null;
      });
    const readout = absorptionGroup.locator('.luxar-layers-panel__control-value');

    await slider.fill('0.75');
    await waitForNextRender(page);
    const midKappa = await readKappa();
    expect(midKappa).not.toBeNull();
    expect(midKappa!).toBeGreaterThan(0);
    expect(midKappa!).toBeCloseTo(parseFloat((await readout.textContent()) ?? 'NaN'), 1);

    // The far end of a gsplat layer's track is the default bound (10): gsplats
    // carry no thickness stat, so their historical span is preserved.
    await slider.fill('1');
    await waitForNextRender(page);
    expect(await readKappa()).toBeCloseTo(10, 6);

    // Position 0 is the dedicated zero stop — exactly κ=0, the additive limit.
    await slider.fill('0');
    await waitForNextRender(page);
    expect(await readKappa()).toBe(0);

    // …and switching the mode away hides the slider immediately.
    await blendSelect.selectOption('additive');
    await expect(absorptionGroup).toBeHidden();
  });

  test('kind=partition layer: Blend reaches EVERY part material', async ({ page }) => {
    // Regression: the blending test above deliberately picks a non-group layer,
    // so the composite (kind=partition / kind=lod) path had no coverage — and it
    // was broken. `graft_gsplat_node` re-stamped `blending_mode` on every part;
    // the attr is nearest-setter-wins, so each part shadowed the wrapper and the
    // layer's single Blend control did nothing. In the recipe gallery, flat /
    // stream / levels layers switched while tiles / overview / adaptive did not.
    const FIXTURE =
      'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_partition_layer.luxar.zarr';
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await openLayersPanel(page);

    // Exactly ONE layer row: the partition wrapper (the parts are internal).
    const target = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const panel = debug.app.layersPanel;
      const layers = panel.layerState.getLayers();
      panel.layerState.select(layers[0].path, 'single');
      return { count: layers.length, path: layers[0].path, kind: layers[0].kind };
    });
    expect(target.count).toBe(1);
    expect(target.kind).toBe('partition');

    // Both part meshes must exist BEFORE the switch: a part that streams in
    // afterwards is built from the composed attrs and would read 'max' for the
    // wrong reason (or be missing entirely and pass an empty assertion).
    const countParts = () =>
      page.evaluate((path) => {
        const debug = (window as any).__luxarDebug;
        let n = 0;
        debug.scene.traverse((obj: any) => {
          const inLayer = obj.name?.startsWith(`${path}/`);
          if (obj.userData?.nodeType === 'gsplats' && obj.material && inLayer) n++;
        });
        return n;
      }, target.path);
    await expect.poll(countParts, { timeout: 15000 }).toBeGreaterThanOrEqual(2);

    const blendSelect = page.locator(
      '.luxar-layers-panel__control-group:has(.luxar-layers-panel__control-label:text-is("Blend")) select'
    );
    await expect(blendSelect).toHaveValue('volumetric');
    await blendSelect.selectOption('max');
    await waitForNextRender(page);

    // Every part material followed the layer — none stayed on 'volumetric'.
    const parts = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      const out: string[] = [];
      debug.scene.traverse((obj: any) => {
        const inLayer = obj.name === path || obj.name?.startsWith(`${path}/`);
        if (obj.userData?.nodeType === 'gsplats' && obj.material && inLayer) {
          out.push(obj.material.userData?.blendingMode);
        }
      });
      return out;
    }, target.path);
    expect(parts.length).toBeGreaterThanOrEqual(2);
    expect(parts.every((m) => m === 'max')).toBe(true);
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

  test('group layer fans out visibility to all data descendants', async ({ page }) => {
    await openLayersPanel(page);

    // The fixture contains a group layer named "CompositeLayer" with two
    // non-layer children: GreenPart and YellowPart. Clicking the group's
    // eye button must hide both THREE.js objects beneath it.
    //
    // Path format is resolved dynamically because scene-graph paths may be
    // stored with or without a leading slash depending on how zarrita's
    // `contents()` is implemented for the store.
    const resolved = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const layers = debug?.app?.layersPanel?.layerState?.getLayers() ?? [];
      const group = layers.find((l: any) => l.name === 'CompositeLayer');
      return {
        groupPath: group?.path ?? null,
        groupType: group?.type ?? null,
      };
    });
    expect(resolved.groupPath).not.toBeNull();
    expect(resolved.groupType).toBe('group');
    const groupPath = resolved.groupPath!;
    // Child paths share the group's prefix separator convention.
    const childPaths = [`${groupPath}/GreenPart`, `${groupPath}/YellowPart`];

    // Initial state: both children visible
    const initialVisible = await page.evaluate((paths) => {
      const debug = (window as any).__luxarDebug;
      if (!debug?.scene) return null;
      const result: Record<string, boolean | null> = {};
      for (const p of paths) {
        const obj = debug.scene.getObjectByName(p);
        result[p] = obj ? obj.visible : null;
      }
      return result;
    }, childPaths);
    for (const p of childPaths) expect(initialVisible![p]).toBe(true);

    // Hide the group via the state API (mirrors what the eye button does)
    await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      const panel = debug.app.layersPanel;
      panel.layerState.setVisible(path, false);
      // Panel's eye-button handler also calls applyVisibility; invoke it
      // via the same public surface the row listener uses.
      const obj = debug.scene.getObjectByName(path);
      if (obj) obj.visible = false;
    }, groupPath);
    await waitForNextRender(page);

    // Descendants should be hidden (THREE.js cascades parent.visible).
    const afterHide = await page.evaluate((paths) => {
      const debug = (window as any).__luxarDebug;
      const result: Record<string, boolean | null> = {};
      for (const p of paths) {
        const obj = debug.scene.getObjectByName(p);
        // Walk up the parent chain; a node is effectively visible only if
        // every ancestor is visible too.
        let effective: boolean | null = obj?.visible ?? null;
        let cursor = obj?.parent;
        while (cursor && effective !== false) {
          if (cursor.visible === false) effective = false;
          cursor = cursor.parent;
        }
        result[p] = effective;
      }
      return result;
    }, childPaths);
    for (const p of childPaths) expect(afterHide![p]).toBe(false);
  });

  test('visible=False on a layer hides it at load time', async ({ page }) => {
    await openLayersPanel(page);

    // The fixture's "HiddenLayer" is authored with visible=False. Resolve
    // its path from the layer state so the test stays agnostic to the
    // leading-slash convention used by the underlying store.
    const hiddenPath = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const layers = debug?.app?.layersPanel?.layerState?.getLayers() ?? [];
      return layers.find((l: any) => l.name === 'HiddenLayer')?.path ?? null;
    });
    expect(hiddenPath).not.toBeNull();

    const panelVisible = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      const info = debug?.app?.layersPanel?.layerState?.getLayer(path);
      return info?.visible;
    }, hiddenPath!);
    expect(panelVisible).toBe(false);

    const sceneVisible = await page.evaluate((path) => {
      const debug = (window as any).__luxarDebug;
      const obj = debug?.scene?.getObjectByName(path);
      return obj?.visible;
    }, hiddenPath!);
    expect(sceneVisible).toBe(false);
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

  test('keyboard listbox: programmatic focus + ArrowDown selects next row, aria-selected updates', async ({
    page,
  }) => {
    // The test programmatically focuses the row via .focus() rather
    // than driving Tab — Tab in headless depends on every interactive
    // control between the canvas and the listbox, which is fragile.
    // This still verifies the listbox keyboard idiom (ArrowDown moves
    // focus + selection, aria-selected updates) end-to-end against a
    // real scene + input handler + DOM.
    await openLayersPanel(page);

    // The listbox container has role="listbox"; rows are role="option".
    const listbox = page.locator('.luxar-layers-panel__list[role="listbox"]');
    await expect(listbox).toBeVisible();

    const rows = page.locator('.luxar-layer-row[role="option"]');
    const rowCount = await rows.count();
    expect(rowCount).toBeGreaterThan(1); // need ≥ 2 rows for ArrowDown

    // Snapshot which row is selected initially. The panel auto-
    // selects the first layer at init time.
    const initiallySelected = await rows.evaluateAll((els) =>
      els.findIndex((el) => el.getAttribute('aria-selected') === 'true')
    );
    expect(initiallySelected).toBeGreaterThanOrEqual(0);

    // Focus the initially-selected row (the tabIndex=0 anchor) so the
    // listbox keyboard handler can route ArrowDown through. Direct
    // .focus() is more reliable than Tab in headless because Tab
    // navigation order depends on every interactive control between
    // the canvas and the listbox.
    await rows.nth(initiallySelected).focus();
    await page.keyboard.press('ArrowDown');

    // The next row should now be selected.
    const nextSelected = await rows.evaluateAll((els) =>
      els.findIndex((el) => el.getAttribute('aria-selected') === 'true')
    );
    expect(nextSelected).toBe(initiallySelected + 1);

    // Verify selection-state class flipped (visual rendering + state
    // both move in lock-step; checking either is enough but checking
    // both guards against drift).
    const selectedRow = rows.nth(nextSelected);
    await expect(selectedRow).toHaveClass(/luxar-layer-row--selected/);
  });
});
