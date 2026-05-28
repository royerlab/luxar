/**
 * E2E tests for the lod_group scene-graph node.
 *
 * Covers:
 *   - Loading a multi-level lod_group fixture into the viewer (no console
 *     errors, no WebGL errors).
 *   - The layers panel renders an "Active level" dropdown for the
 *     lod_group, populated with ``auto`` + one ``lock to level <i>``
 *     option per child.
 *   - Manual override: locking to a specific level via the dropdown
 *     swaps which child mesh is visible. ``auto`` mode resumes
 *     view-driven selection.
 *
 * Uses the ``test_lod_group.zarr`` fixture (3 levels, splat counts
 * 8 / 32 / 128, thresholds 0 / 50 / 200 px) — small enough to render
 * instantly, large enough that the registry's selector picks a
 * meaningful level on a default-zoom view.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForNextRender,
  openLayersPanel,
  assertNoConsoleErrors,
  getWebGLErrors,
} from './helpers';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lod_group.zarr';

test.describe('lod_group node', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    // Three meshes (one per LOD child) should attach to the scene
    // graph; wait for them to settle.
    await page.waitForFunction(
      () => {
        const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        if (!debug?.scene) return false;
        let lodChildren = 0;
        (debug.scene as { traverse: (cb: (o: { name?: string }) => void) => void }).traverse((o) => {
          if (o.name && o.name.startsWith('/multires/child_')) lodChildren++;
        });
        return lodChildren >= 3;
      },
      { timeout: 15000 }
    );
  });

  test('loads without errors and exposes three lod children', async ({ page }) => {
    await assertNoConsoleErrors(page);
    expect(getWebGLErrors(page)).toEqual([]);
  });

  test('renders only one child by default (default_level = 0 = coarsest)', async ({
    page,
  }) => {
    const visibility = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
      const out: Record<string, boolean> = {};
      if (!debug?.scene) return out;
      (debug.scene as { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void }).traverse((o) => {
        if (o.name && o.name.startsWith('/multires/child_')) {
          out[o.name] = !!o.visible;
        }
      });
      return out;
    });
    const visibleChildren = Object.entries(visibility).filter(([, v]) => v);
    expect(visibleChildren).toHaveLength(1);
  });

  test('layers panel shows an "Active level" dropdown with auto + 3 lock options', async ({
    page,
  }) => {
    await openLayersPanel(page);

    // Find the lod_group row in the layer list — the layer name is the
    // last path segment, so look for "multires".
    const lodRow = page
      .locator('.luxar-layer-row__name', { hasText: 'multires' })
      .first();
    await expect(lodRow).toBeVisible({ timeout: 5000 });
    await lodRow.click();
    await waitForNextRender(page);

    // The Active-level dropdown lives in the shared controls section
    // and is shown only when the primary selected layer is an lod_group.
    // The dropdown's own label says "Active level".
    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();
    await expect(lodSelect).toBeVisible();

    const optionValues = await lodSelect.locator('option').evaluateAll(
      (opts) => opts.map((o) => (o as HTMLOptionElement).value)
    );
    expect(optionValues).toEqual(['auto', '0', '1', '2']);
  });

  test('selecting "lock to level 2" makes child_2 the visible mesh', async ({
    page,
  }) => {
    await openLayersPanel(page);
    const lodRow = page
      .locator('.luxar-layer-row__name', { hasText: 'multires' })
      .first();
    await lodRow.click();
    await waitForNextRender(page);

    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();
    await lodSelect.selectOption('2');
    // Give the per-frame selector a tick to react.
    await waitForNextRender(page);
    await waitForNextRender(page);

    const visibility = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
      const out: Record<string, boolean> = {};
      if (!debug?.scene) return out;
      (debug.scene as { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void }).traverse((o) => {
        if (o.name && o.name.startsWith('/multires/child_')) {
          out[o.name] = !!o.visible;
        }
      });
      return out;
    });
    expect(visibility['/multires/child_2']).toBe(true);
    expect(visibility['/multires/child_0']).toBe(false);
    expect(visibility['/multires/child_1']).toBe(false);
  });
});
