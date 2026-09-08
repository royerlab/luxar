/**
 * Partition frustum resync must not re-stage the rest of the scene.
 *
 * A `kind=partition` whose parts are `kind=lod` groups (the hosted 44-part
 * h2afva time-lapse, every `--recipe adaptive` store). Off-screen parts are
 * frustum-culled and skip slice updates, so a part RE-ENTERING the view asks
 * the scene loader to resync it. That resync used to be a plain
 * `updateView({})`, which bumped the global view version although the view had
 * not changed; lazy LOD levels live outside the update sweep and are never
 * re-stamped by it, so EVERY group in the scene read stale and fell to its
 * coarsest level for hundreds of milliseconds, then re-streamed — on the
 * deployed viewer a 4 s drag fired five such reprocesses and collapsed 14/44
 * fine groups to coarse.
 *
 * This spec parks the camera INSIDE one part looking away from the other (so
 * the other sits entirely behind the camera and is culled), then turns around
 * so the far part comes back, and asserts:
 *   - the far part was really culled and really came back (the trigger fired),
 *   - the NEAR part's group never left its finest level while that happened,
 *   - no new `Updating view v…` pass ran — the resync is logged as
 *     `Resyncing view` under the unchanged version.
 *
 * The far part's own group is deliberately NOT asserted on: while culled it
 * is legitimately held at its coarsest ready level by the off-screen gate.
 *
 * Fixture: `test_partition_of_lod_points.luxar.zarr` (two clusters at x = ±8,
 * each a 3-level substitutive ladder, no additive rungs).
 */

import { test, expect } from './fixtures';
import {
  captureConsoleMessages,
  getLuxarState,
  placeCameraAt,
  waitForLuxarReady,
  waitForRenderStable,
} from './helpers';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_partition_of_lod_points.luxar.zarr';

/**
 * Camera park, INSIDE the near cluster's authored bounds (x ∈ [-12.7, -2.6],
 * centre -8): the selector's camera-inside rule pins that part at its finest
 * level in both poses below. Why inside-and-behind rather than "beside": the
 * frustum gate unions each part's rendered FOOTPRINT into its bounds, and the
 * far part's coarsest level is 25 merged Gaussians whose truncation extent
 * inflates that footprint to x ∈ [-6.2, 21.7] — so no lateral pose can push it
 * out of the padded frustum. A box entirely BEHIND the camera is culled no
 * matter how wide it is, as long as it starts past the camera plane: with the
 * camera at x = -11 looking toward −x, the inflated far box (min x = −6.2)
 * begins 4.8 units behind it.
 */
const PARK = { x: -11, y: 0, z: 0 };
/** Looking away from the far cluster (−x): the far part is behind the camera. */
const LOOK_AWAY = { x: -12, y: 0, z: 0 };
/** Turned around (+x): the far cluster is straight ahead — the rising edge. */
const LOOK_BACK = { x: -10, y: 0, z: 0 };

interface LodGroupInfo {
  name: string;
  levelCount: number;
  activeLevel: number;
}
interface PartitionInfo {
  name: string;
  partCount: number;
  visibleParts: number;
}

async function readGroups(page: import('@playwright/test').Page): Promise<{
  partition: PartitionInfo;
  near: LodGroupInfo;
}> {
  const state = await getLuxarState(page);
  const partition = (state.partitions as PartitionInfo[]).find((p) => p.name.endsWith('tiled'));
  const near = (state.lodGroups as LodGroupInfo[]).find((g) => g.name.endsWith('part_0'));
  expect(partition, 'the tiled partition is registered').toBeDefined();
  expect(near, 'part_0 is a kind=lod group').toBeDefined();
  return { partition: partition!, near: near! };
}

/**
 * Keep the on-demand render loop awake and poll `getState()` until `pred`
 * holds. The registry's frustum gate and level swaps run inside per-frame
 * callbacks, so a parked loop would never reflect a camera move.
 */
async function waitForGroups(
  page: import('@playwright/test').Page,
  pred: (g: { partition: PartitionInfo; near: LodGroupInfo }) => boolean,
  label: string,
  timeout = 20000
): Promise<void> {
  const deadline = Date.now() + timeout;
  let last: { partition: PartitionInfo; near: LodGroupInfo } | null = null;
  while (Date.now() < deadline) {
    await page.evaluate(() => {
      (window as any).__luxarDebug?.animationController?.startAnimation?.();
    });
    last = await readGroups(page);
    if (pred(last)) return;
    await page.waitForTimeout(100);
  }
  throw new Error(`${label}: timed out; last state ${JSON.stringify(last)}`);
}

test.describe('kind=partition of kind=lod parts — frustum re-entry resync', () => {
  test('a part re-entering the frustum does not drop the on-screen groups to coarse', async ({
    page,
  }) => {
    test.setTimeout(120000);
    const console = captureConsoleMessages(page);
    await page.goto(`/?src=${FIXTURE}&debug`);
    await waitForLuxarReady(page);
    await waitForRenderStable(page);

    // Both parts on screen at the opening framing.
    await waitForGroups(page, (g) => g.partition.visibleParts === 2, 'opening framing');

    // Park inside the near cluster looking away from the far one: the near
    // part is at its finest level (camera inside its bounds) and the far part,
    // footprint and all, sits entirely behind the camera — culled.
    const parked = await placeCameraAt(page, PARK, { target: LOOK_AWAY });
    expect(parked, 'camera placement is honoured').not.toBeNull();
    await waitForGroups(
      page,
      (g) => g.partition.visibleParts === 1 && g.near.activeLevel === g.near.levelCount - 1,
      `far part culled, near part at its finest level (placement ${JSON.stringify(parked)})`
    );
    const updatingBefore = console.logs.filter((l) => /Updating view v/.test(l)).length;
    const nearBefore = (await readGroups(page)).near;

    // Turn around in place: the far cluster is now straight ahead and re-enters
    // the frustum — the rising edge under test. The camera has not moved, so
    // the near part's selection is unchanged (still camera-inside ⇒ finest).
    const yawed = await placeCameraAt(page, PARK, { target: LOOK_BACK });
    expect(yawed, 'camera placement is honoured').not.toBeNull();

    // Sample across the window in which the regression showed: the stale hold
    // lasted 250 ms, then the group dropped to coarse and re-streamed for
    // seconds. The near part must stay on its finest level the whole time.
    const deadline = Date.now() + 3000;
    let sawFarBack = false;
    const nearLevels = new Set<number>();
    while (Date.now() < deadline) {
      await page.evaluate(() => {
        (window as any).__luxarDebug?.animationController?.startAnimation?.();
      });
      const g = await readGroups(page);
      if (g.partition.visibleParts === 2) sawFarBack = true;
      nearLevels.add(g.near.activeLevel);
      await page.waitForTimeout(50);
    }
    expect(sawFarBack, 'the far part re-entered the frustum (trigger fired)').toBe(true);
    expect(
      [...nearLevels],
      'the on-screen group never left its finest level during the resync'
    ).toEqual([nearBefore.levelCount - 1]);

    // The resync re-swept under the unchanged view version: no new
    // "Updating view vN" pass — only a bump-free "Resyncing view" one. This is
    // the assertion that FIRES on the pre-fix code (verified: 1 → 2 passes on
    // the same fixture). The level assertion above is belt-and-braces here: a
    // local two-cluster fixture re-streams its fine level inside the 250 ms
    // stale hold, so the visible collapse needs a hosted-scale scene to show.
    const updatingAfter = console.logs.filter((l) => /Updating view v/.test(l)).length;
    expect(updatingAfter, 'no view-version bump from a camera move').toBe(updatingBefore);
    expect(
      console.logs.some((l) => /Resyncing view v\d+ \(1 target path/.test(l)),
      'the re-entering part was resynced on its own'
    ).toBe(true);

    await waitForRenderStable(page);
    const settled = await getLuxarState(page);
    expect(settled.isLoading).toBe(false);
  });
});
