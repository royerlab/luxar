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
 * Uses the ``test_lod_group.luxar.zarr`` fixture (3 levels, splat counts
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
  focusCanvas,
} from './helpers';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lod_group.luxar.zarr';

test.describe('lod_group node', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    // Three meshes (one per LOD child) should attach to the scene
    // graph; wait for them to settle.
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        let lodChildren = 0;
        (debug.scene as { traverse: (cb: (o: { name?: string }) => void) => void }).traverse(
          (o) => {
            if (o.name && o.name.startsWith('/multires/child_')) lodChildren++;
          }
        );
        return lodChildren >= 3;
      },
      { timeout: 15000 }
    );
  });

  test('loads without errors and exposes three lod children', async ({ page }) => {
    await assertNoConsoleErrors(page);
    expect(await getWebGLErrors(page)).toEqual([]);
  });

  test('renders only one child by default (default_level = 0 = coarsest)', async ({ page }) => {
    const visibility = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } })
        .__luxarDebug;
      const out: Record<string, boolean> = {};
      if (!debug?.scene) return out;
      (
        debug.scene as { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void }
      ).traverse((o) => {
        if (o.name && o.name.startsWith('/multires/child_')) {
          out[o.name] = !!o.visible;
        }
      });
      return out;
    });
    const visibleChildren = Object.entries(visibility).filter(([, v]) => v);
    expect(visibleChildren).toHaveLength(1);
  });

  test('debug snapshot reports the lod_group with its active level', async ({ page }) => {
    // The monitor's LOD-awareness is fed from the same scene markers the
    // debug snapshot reads. Assert getState().lodGroups surfaces the
    // multires group, that exactly one level is active, and that the
    // reported activeLevel matches the visible child index.
    type LodGroupInfo = { name: string; levelCount: number; activeLevel: number };
    const { lodGroups, visibleIndex } = (await page.evaluate(() => {
      const debug = (
        window as unknown as {
          __luxarDebug?: {
            getState?: () => {
              lodGroups: { name: string; levelCount: number; activeLevel: number }[];
            };
            scene?: { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void };
          };
        }
      ).__luxarDebug;
      const state = debug?.getState?.();
      let visibleIndex = -1;
      debug?.scene?.traverse((o) => {
        const m = o.name?.match(/\/multires\/child_(\d+)$/);
        if (m && o.visible) visibleIndex = Number(m[1]);
      });
      return { lodGroups: state?.lodGroups ?? [], visibleIndex };
    })) as { lodGroups: LodGroupInfo[]; visibleIndex: number };

    const multires = lodGroups.find((g) => g.name === '/multires');
    expect(multires).toBeDefined();
    expect(multires!.levelCount).toBe(3);
    expect(multires!.activeLevel).toBe(visibleIndex);
    expect(visibleIndex).toBeGreaterThanOrEqual(0);
  });

  test('layers panel shows an "Active level" dropdown with auto + 3 lock options', async ({
    page,
  }) => {
    await openLayersPanel(page);

    // Find the lod_group row in the layer list — the layer name is the
    // last path segment, so look for "multires".
    const lodRow = page.locator('.luxar-layer-row__name', { hasText: 'multires' }).first();
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

    const optionValues = await lodSelect
      .locator('option')
      .evaluateAll((opts) => opts.map((o) => (o as HTMLOptionElement).value));
    expect(optionValues).toEqual(['auto', '0', '1', '2']);
  });

  test('selecting "lock to level 2" makes child_2 the visible mesh', async ({ page }) => {
    await openLayersPanel(page);
    const lodRow = page.locator('.luxar-layer-row__name', { hasText: 'multires' }).first();
    await lodRow.click();
    await waitForNextRender(page);

    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();
    await lodSelect.selectOption('2');
    // child_2's geometry is loaded lazily on first activation, so the
    // swap takes a frame to fire ``ensureLoaded`` plus the async fetch.
    // Poll until child_2 is the sole visible mesh (auto-retries absorb
    // the deferred load).
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        let child2Visible = false;
        let otherVisible = false;
        (
          debug.scene as {
            traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void;
          }
        ).traverse((o) => {
          if (!o.name || !o.name.startsWith('/multires/child_')) return;
          if (o.name === '/multires/child_2') child2Visible = !!o.visible;
          else if (o.visible) otherVisible = true;
        });
        return child2Visible && !otherVisible;
      },
      { timeout: 5000 }
    );

    const visibility = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } })
        .__luxarDebug;
      const out: Record<string, boolean> = {};
      if (!debug?.scene) return out;
      (
        debug.scene as { traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void }
      ).traverse((o) => {
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

  test('monitor visible-splat count tracks the active LOD level (not the coarsest)', async ({
    page,
  }) => {
    // Regression for the "always shows the coarsest count" bug: a
    // substitutive-LOD swap happens per-frame (camera/lock) with no data
    // reload, so the monitor's visible tally must be refreshed on the
    // switch — otherwise it stays pinned to the default/coarsest level.
    // Reads the value the MONITOR actually displays (not a scene recompute)
    // so it guards the full wiring. Fixture levels: 8 / 32 / 128.
    const visibleSplats = page.locator('[data-field="visible-splats"]').first();

    // Expand the monitor (hidden → mini → expanded) so the Overview metric
    // cards render and the polling loop patches them.
    await focusCanvas(page);
    await page.keyboard.press('m');
    await page.keyboard.press('m');
    await expect(visibleSplats).toBeVisible({ timeout: 5000 });

    // Lock to the finest level (128) via the layers dropdown.
    await openLayersPanel(page);
    const lodRow = page.locator('.luxar-layer-row__name', { hasText: 'multires' }).first();
    await lodRow.click();
    await waitForNextRender(page);
    const lodSelect = page
      .locator('.luxar-layers-panel__control-group', { hasText: 'Active level' })
      .locator('select')
      .first();

    await lodSelect.selectOption('2');
    // toHaveText auto-retries, absorbing the per-frame swap + ~100ms poll.
    await expect(visibleSplats).toHaveText('128', { timeout: 5000 });

    // Lock to the coarsest level (8) — the monitor count must drop, proving
    // it tracks the active level rather than staying high or summing levels.
    await lodSelect.selectOption('0');
    await expect(visibleSplats).toHaveText('8', { timeout: 5000 });
  });
});
