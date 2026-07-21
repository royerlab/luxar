/**
 * E2E tests for the lod_group scene-graph node.
 *
 * Covers:
 *   - Loading a multi-level lod_group fixture into the viewer (no console
 *     errors, no WebGL errors).
 *   - The layers panel renders an "Active level" dropdown for the
 *     lod_group, populated with ``auto`` + one ``lock to level <n>``
 *     option per child (label is 1-based; the option value stays 0-based
 *     to match the registry's lockLevel API).
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

  test('auto mode shows the active level, plus at most its cross-fade partner', async ({
    page,
  }) => {
    // HISTORY: this test used to assert "exactly one visible child" and
    // FLAKED under parallel runs — a different assertion failing each run.
    // That premise predates the coverage CROSS-FADE (lod-blend.ts): in the
    // blend band the registry deliberately shows TWO ADJACENT levels with
    // blended opacities, and this fixture's default view sits in the
    // level-0/1 band, settling on {child_0, child_1} a few frames after
    // load. The old test only passed by sampling before the fade engaged;
    // one-shot reads under CPU load landed after it. The real invariant:
    // the visible set is {active} or {active, active+1} — never a
    // non-adjacent pair, never all three, never empty.
    await page.waitForFunction(
      () => {
        const debug = (
          window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } }
        ).__luxarDebug;
        if (!debug?.scene) return false;
        const vis: Record<string, boolean> = {};
        (
          debug.scene as {
            traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void;
          }
        ).traverse((o) => {
          const m = o.name?.match(/\/multires\/child_(\d+)$/);
          if (m) vis[m[1]] = !!o.visible;
        });
        const idxs = Object.entries(vis)
          .filter(([, v]) => v)
          .map(([k]) => Number(k))
          .sort((a, b) => a - b);
        // Settled: ≥1 visible, ≤2 visible, and if 2 they are adjacent.
        return (
          Object.keys(vis).length >= 3 &&
          idxs.length >= 1 &&
          idxs.length <= 2 &&
          (idxs.length === 1 || idxs[1] === idxs[0] + 1)
        );
      },
      { timeout: 10000 }
    );
    // Pin the settled shape: the coarsest level participates at default zoom.
    const visibleIdxs = await page.evaluate(() => {
      const debug = (window as Window & typeof globalThis & { __luxarDebug?: { scene?: unknown } })
        .__luxarDebug;
      const vis: Record<string, boolean> = {};
      (
        debug!.scene as {
          traverse: (cb: (o: { name?: string; visible?: boolean }) => void) => void;
        }
      ).traverse((o) => {
        const m = o.name?.match(/\/multires\/child_(\d+)$/);
        if (m) vis[m[1]] = !!o.visible;
      });
      return Object.entries(vis)
        .filter(([, v]) => v)
        .map(([k]) => Number(k))
        .sort((a, b) => a - b);
    });
    expect(visibleIdxs.length).toBeGreaterThanOrEqual(1);
    expect(visibleIdxs.length).toBeLessThanOrEqual(2);
    if (visibleIdxs.length === 2) {
      expect(visibleIdxs[1]).toBe(visibleIdxs[0] + 1); // cross-fade partners are adjacent
    }
  });

  test('debug snapshot reports the lod_group with its active level', async ({ page }) => {
    // The monitor's LOD-awareness is fed from the same scene markers the
    // debug snapshot reads. Assert getState().lodGroups surfaces the
    // multires group, that exactly one level is active, and that the
    // reported activeLevel matches the visible child index.
    type LodGroupInfo = { name: string; levelCount: number; activeLevel: number };
    // HISTORY: this test used to demand a single visible child equal to
    // activeLevel and FLAKED under parallel runs — the coverage cross-fade
    // (lod-blend.ts) deliberately shows the active level's ADJACENT partner
    // in the blend band, so the real contract is: activeLevel is AMONG the
    // visible children and every visible child is activeLevel or its +1
    // partner. Poll until the snapshot and scene agree on that settled
    // shape before pinning details (one-shot reads catch mid-selection
    // states under CPU load).
    await page.waitForFunction(
      () => {
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
        const g = debug?.getState?.()?.lodGroups?.find((x) => x.name === '/multires');
        if (!g) return false;
        const vis: Record<string, boolean> = {};
        debug?.scene?.traverse((o) => {
          const m = o.name?.match(/\/multires\/child_(\d+)$/);
          if (m) vis[m[1]] = !!o.visible;
        });
        const idxs = Object.entries(vis)
          .filter(([, v]) => v)
          .map(([k]) => Number(k))
          .sort((a, b) => a - b);
        return (
          idxs.length >= 1 &&
          idxs.length <= 2 &&
          idxs[0] === g.activeLevel &&
          (idxs.length === 1 || idxs[1] === g.activeLevel + 1)
        );
      },
      { timeout: 10000 }
    );
    const { lodGroups, visibleIdxs } = (await page.evaluate(() => {
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
      const vis: Record<string, boolean> = {};
      debug?.scene?.traverse((o) => {
        const m = o.name?.match(/\/multires\/child_(\d+)$/);
        if (m) vis[m[1]] = !!o.visible;
      });
      return {
        lodGroups: state?.lodGroups ?? [],
        visibleIdxs: Object.entries(vis)
          .filter(([, v]) => v)
          .map(([k]) => Number(k))
          .sort((a, b) => a - b),
      };
    })) as { lodGroups: LodGroupInfo[]; visibleIdxs: number[] };

    const multires = lodGroups.find((g) => g.name === '/multires');
    expect(multires).toBeDefined();
    expect(multires!.levelCount).toBe(3);
    // The active level is the PRIMARY visible child; any second visible
    // child is its cross-fade partner (active + 1).
    expect(visibleIdxs[0]).toBe(multires!.activeLevel);
    expect(visibleIdxs.length).toBeLessThanOrEqual(2);
    if (visibleIdxs.length === 2) {
      expect(visibleIdxs[1]).toBe(multires!.activeLevel + 1);
    }
    expect(multires!.activeLevel).toBeGreaterThanOrEqual(0);
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

  // Selects by option VALUE '2' (0-based child index 2 = finest), whose
  // 1-based label reads "lock to level 3". Asserting the value, not the
  // label text, keeps this robust to label wording.
  test('locking the finest level (value 2) makes child_2 the visible mesh', async ({ page }) => {
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
