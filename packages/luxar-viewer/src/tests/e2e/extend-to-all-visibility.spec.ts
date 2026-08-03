/**
 * Regression E2E: a node whose `extend_to_all` covers ALL non-displayed
 * dimensions must be QUERIED — at the opening slice and at every slice
 * after it.
 *
 * Scene (`test_extend_to_all_4d.luxar.zarr`): a hidden discrete `time` dim
 * spanning 0..4, three fully-extended nodes (`ext_pts` with a 300/700/1200
 * additive ladder, `ext_lines`, `ext_gsplats`) authored at `time=2` via
 * `fill`, plus `sliced_pts` — a real `time` column at `time=3` — as the
 * sliced control.
 *
 * Pre-fix behaviour this pins against: `deriveNodeViewState` returned a bare
 * `{ skip: 'extend_to_all' }` for a fully-extended node BEFORE the tolerance
 * override that implements "ignore this dimension" ran. Nothing ever
 * performed the one query the skip assumed had already happened, so the node
 * fetched nothing at all (symptom 1), its additive ladder stayed frozen at
 * whatever the eager initial load committed (symptom 2), and gsplat levels
 * were filtered out during nD->3D projection because `extendToAllDims` is
 * derived from the tolerance sentinel that was never set (symptom 3).
 *
 * Two fixture properties keep this test honest, and both are asserted rather
 * than assumed:
 *
 * - The opening slice is `time=0`, NOT the `time=2` the extended nodes were
 *   authored at. A layer whose `fill` coincides with the opening slice
 *   renders correctly even when fully broken — that coincidence is exactly
 *   what hid this bug in the demos, so the spec fails loudly if a future
 *   fixture edit lines the two up.
 * - `sliced_pts` must be EMPTY wherever the extended nodes are full. Without
 *   that control, "everything is visible everywhere" would pass vacuously.
 *
 * All three geometry kinds are covered because they reach the extended query
 * differently: Points and GSplats take the tolerance override directly, Lines
 * opt out of the PARTIAL override (their segment bounds already encode the
 * extent) and so exercise the full-extend path on its own, and GSplats
 * additionally rebuild their `extendToAllDims` set from the `1e10` tolerance
 * sentinel while projecting.
 *
 * Measured against the pre-fix build, this spec fails on `ext_pts` (0 instead
 * of 1200) and `ext_gsplats` (0 instead of 40). `ext_lines` rendered its full
 * 199 segments even pre-fix, so the Lines assertion is forward-looking
 * coverage of the shared derivation rather than a reproduction of the
 * original symptom — do not read a passing Lines count as evidence that the
 * extended query ran.
 *
 * Committed counts alone are NOT sufficient to pin the per-slice requirement:
 * an implementation that queries once on load and then skips every later
 * extended-node query, keeping the geometry it already committed, holds the
 * counts full forever. That is not hypothetical — it is what the pre-fix skip
 * did, and it is why the ladder froze. The scrub test therefore also asserts
 * each extended node's `loadedViewVersion` ADVANCES on every sweep, which only
 * a node the sweep actually reached can do. Verified by mutation: with the
 * handlers patched to converge-then-skip, the count assertions still pass and
 * only the freshness assertion fails.
 */

import { test, expect } from './fixtures';
import type { Page } from './fixtures';
import { waitForLuxarReady, waitForDataLoaded } from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const DATASET = `${FIXTURES_BASE}/test_extend_to_all_4d.luxar.zarr`;

/** The hidden `time` dim index in the fixture (x,y,z displayed; time=3). */
const TIME_DIM = 3;

/** Every step of the hidden dim, in scrub order (range 0..4, step 1). */
const TIME_VALUES = [0, 1, 2, 3, 4] as const;

/** The slice the extended nodes were authored at (`fill={'time': 2}`). */
const FILL_TIME = 2;

/** The only slice at which the sliced control carries data. */
const CONTROL_TIME = 3;

/**
 * Full committed size of each node. `ext_pts` is the ladder check: its
 * rungs are 300/700/1200, so a node stuck on its first rung — a frozen
 * ladder — reads 300 here, not 1200.
 */
const FULL = { ext_pts: 1200, ext_lines: 199, ext_gsplats: 40 } as const;

/** The fully-extended nodes, one per geometry kind. */
const EXTENDED_NODES = ['ext_pts', 'ext_lines', 'ext_gsplats'] as const;

/** Committed counts per node, read from the live scene graph. */
async function getCommittedCounts(page: Page): Promise<{
  ext_pts: number;
  ext_lines: number;
  ext_gsplats: number;
  sliced_pts: number;
}> {
  return page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
    const counts: Record<string, number> = {
      ext_pts: 0,
      ext_lines: 0,
      ext_gsplats: 0,
      sliced_pts: 0,
    };
    debug.scene.traverse((obj: any) => {
      const name = obj?.name as string | undefined;
      if (!name || !obj.geometry) return;
      // Match the node and any sub-LOD/part meshes beneath it.
      for (const key of Object.keys(counts)) {
        if (name === `/${key}` || name.startsWith(`/${key}/`)) {
          counts[key] += obj.geometry.instanceCount ?? 0;
        }
      }
    });
    return counts as {
      ext_pts: number;
      ext_lines: number;
      ext_gsplats: number;
      sliced_pts: number;
    };
  });
}

/**
 * Per-node freshness stamps (`userData.loadedViewVersion`), which the loader
 * writes on every commit — including the stamp-only no-op commit a
 * same-view re-query produces. A node that is genuinely re-queried each
 * sweep therefore advances its stamp; one that is skipped keeps the stamp it
 * was left with, however full its retained geometry looks.
 */
async function getLoadedViewVersions(page: Page): Promise<Record<string, number>> {
  return page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
    const stamps: Record<string, number> = {};
    debug.scene.traverse((obj: any) => {
      const name = obj?.name as string | undefined;
      if (!name || typeof obj.userData?.loadedViewVersion !== 'number') return;
      for (const key of ['ext_pts', 'ext_lines', 'ext_gsplats']) {
        if (name === `/${key}` || name.startsWith(`/${key}/`)) {
          // Lowest stamp across a node's meshes: every one of them must move.
          stamps[key] = Math.min(stamps[key] ?? Infinity, obj.userData.loadedViewVersion);
        }
      }
    });
    return stamps;
  });
}

/** Read the viewer's current step along the hidden `time` dim. */
async function getTimeStep(page: Page): Promise<number> {
  return page.evaluate((dim) => {
    const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
    return debug.getState().dimensions.currentStep[dim] as number;
  }, TIME_DIM);
}

/** Scrub the hidden dim and await the resulting data update. */
async function scrubTime(page: Page, value: number): Promise<void> {
  await page.evaluate(
    async ([dim, v]) => {
      const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
      await debug.app.setDimensionValue(dim, v);
      await debug.app.awaitDimensionUpdate();
    },
    [TIME_DIM, value]
  );
}

/**
 * Poll until all three extended nodes are FULLY committed and the sliced
 * control matches the slice. Both halves matter: the extended half is the
 * regression, the control half is what stops it passing vacuously.
 */
async function expectExtendedFullAt(page: Page, time: number, label: string): Promise<void> {
  await expect
    .poll(async () => getCommittedCounts(page), {
      message:
        `${label}: every fully-extended node must be committed in full at time=${time} ` +
        '(ext_pts=300 would mean a ladder frozen on its first rung), and the sliced ' +
        'control must follow its own slice',
      timeout: 20000,
    })
    .toEqual({ ...FULL, sliced_pts: time === CONTROL_TIME ? 400 : 0 });
}

test.describe('extend_to_all full extension', () => {
  test('fully-extended nodes render at a slice their fill does not match', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Guard the fixture property this whole spec rests on: if the opening
    // slice ever equals the extended nodes' `fill`, the assertions below
    // would pass on the pre-fix code too.
    const openingTime = await getTimeStep(page);
    expect(
      openingTime,
      "the opening slice must differ from the extended nodes' fill, or a broken " +
        'extend_to_all would render correctly by coincidence'
    ).not.toBe(FILL_TIME);

    await expectExtendedFullAt(page, openingTime, `initial load (time=${openingTime})`);
  });

  test('extended nodes stay full at every slice while the control tracks its own', async ({
    page,
  }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Sweep the whole hidden axis, including the fill slice (2), the
    // control's slice (3), and back down again — a fully-extended node's
    // query is slice-invariant, so its committed count must never move,
    // and re-querying must not reset the ladder it already converged.
    for (const time of [...TIME_VALUES, ...[...TIME_VALUES].reverse()]) {
      const before = await getLoadedViewVersions(page);
      await scrubTime(page, time);
      await expectExtendedFullAt(page, time, `after scrub to time=${time}`);

      // Counts alone cannot tell "re-queried and re-committed the same data"
      // apart from "skipped the query and kept the old geometry" — and the
      // second is exactly the pre-fix design that froze the ladder. The
      // freshness stamp is what separates them: only a node the sweep
      // actually reached gets re-stamped.
      await expect
        .poll(
          async () => {
            const now = await getLoadedViewVersions(page);
            return Object.fromEntries(EXTENDED_NODES.map((key) => [key, now[key] > before[key]]));
          },
          {
            message:
              `after scrub to time=${time}: every extended node must be re-queried, ` +
              'not merely left holding its previous geometry — a node reported ' +
              `false here kept the stamp it had before the scrub (${JSON.stringify(before)})`,
            timeout: 20000,
          }
        )
        .toEqual({ ext_pts: true, ext_lines: true, ext_gsplats: true });
    }
  });
});
