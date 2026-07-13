/**
 * Regression E2E: Lines must be re-culled when scrubbing a non-displayed
 * dimension, exactly like Points.
 *
 * Scene (`test_lines_categorical.luxar.zarr`): a hidden categorical dim
 * `sel` (A/B) with one Lines node + one Points node at sel=0 (`line_a`,
 * `pts_a`) and a second pair at sel=1 (`line_b`, `pts_b`). Scrubbing
 * `sel` must SWAP each pair (A xor B).
 *
 * Pre-fix behaviour this pins against: the Lines projection dispatcher
 * rejected the canonical empty payload (hardcoded numItems=1 in
 * validateProjectionInputs), the throw was swallowed as a failed loader
 * update, and the previous slot's Lines mesh was never cleared — Lines
 * accumulated (A ∪ B) across scrubs while Points swapped correctly. The
 * Points pair doubles as the known-good control in the same frames, and
 * the suite-wide no-console-errors fixture would catch the validation
 * throw on its own.
 */

import { test, expect } from './fixtures';
import type { Page } from './fixtures';
import { waitForLuxarReady, waitForDataLoaded } from './helpers';

const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const DATASET = `${FIXTURES_BASE}/test_lines_categorical.luxar.zarr`;

/** The hidden categorical dim index in the fixture (x,y,z displayed; sel=3). */
const SEL_DIM = 3;

/** Committed instance counts per node, read from the live scene graph. */
async function getCommittedCounts(page: Page): Promise<{
  line_a: number;
  line_b: number;
  pts_a: number;
  pts_b: number;
}> {
  return page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
    const counts: Record<string, number> = {};
    debug.scene.traverse((obj: any) => {
      const name = obj?.name as string | undefined;
      if (name && /^\/(line|pts)_[ab]$/.test(name)) {
        counts[name.slice(1)] = obj.geometry?.instanceCount ?? 0;
      }
    });
    return counts as { line_a: number; line_b: number; pts_a: number; pts_b: number };
  });
}

/** Scrub the hidden dim and await the resulting data update. */
async function scrubSel(page: Page, value: number): Promise<void> {
  await page.evaluate(
    async ([dim, v]) => {
      const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
      await debug.app.setDimensionValue(dim, v);
      await debug.app.awaitDimensionUpdate();
    },
    [SEL_DIM, value]
  );
}

/**
 * Poll until the four nodes' committed counts match the expected slot:
 * the "on" pair non-empty, the "off" pair EMPTY. The off-pair-empty
 * check is the regression: pre-fix, an off-slot Lines node kept its
 * previous count forever.
 */
async function expectSlot(page: Page, slot: 'a' | 'b', label: string): Promise<void> {
  const on = slot;
  const off = slot === 'a' ? 'b' : 'a';
  await expect
    .poll(async () => getCommittedCounts(page), {
      message: `${label}: expected only the '${on}' pair committed (lines must swap like points)`,
      timeout: 15000,
    })
    .toEqual({
      [`line_${on}`]: 399,
      [`line_${off}`]: 0,
      [`pts_${on}`]: 400,
      [`pts_${off}`]: 0,
    });
}

test.describe('Lines non-displayed-dimension re-cull', () => {
  test('scrubbing a hidden categorical dim swaps Lines exactly like Points', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Initial slice sel=0: only the A pair renders.
    await expectSlot(page, 'a', 'initial load (sel=0)');

    // Scrub to sel=1: the A pair must CLEAR (the bug left line_a committed).
    await scrubSel(page, 1);
    await expectSlot(page, 'b', 'after scrub to sel=1');

    // Scrub back to sel=0: the B pair must clear and A must return
    // (exercises the S-cache revisit path as well as the fresh load).
    await scrubSel(page, 0);
    await expectSlot(page, 'a', 'after scrub back to sel=0');
  });
});
