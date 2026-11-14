/**
 * The viewer depth-sorting on the GPU (#3162), end to end on native WebGPU.
 *
 * Loads the front-to-back-declared overlap fixture with `?gpuSort=1`, orbits
 * the camera through three large steps, and checks after each that
 *
 * - a re-sort completed, and it came from the GPU producer, not the worker;
 * - the ordering the node is DRAWN with (read back from its GPU buffer by
 *   `__luxarDebug.readDrawnOrdering`) is back to front under the new pose;
 * - the overlap composites front over back.
 *
 * `?gpuSort=0` runs the same orbit on the worker as the control arm. Runs in
 * the native-WebGPU project only (`pnpm test:e2e:webgpu`).
 *
 * @module tests/e2e/gpu-depth-sort-viewer.spec
 */

import type { Page } from '@playwright/test';

import { test, expect } from './fixtures';

import { waitForLuxarReady } from './helpers';
import { isWebGPUProject } from './webgpu-project';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap_reversed.luxar.zarr';

/** The first gsplats node's uuid and splat count. */
async function gsplatNode(page: Page): Promise<{ uuid: string; count: number } | null> {
  return page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
    let found: { uuid: string; count: number } | null = null;
    debug?.scene?.traverse((obj: any) => {
      if (found || obj.userData?.nodeType !== 'gsplats') return;
      const count = obj.userData?.visibleSplatCount ?? 0;
      if (count >= 2) found = { uuid: obj.uuid, count };
    });
    return found;
  });
}

/** Whether the node's DRAWN ordering is back to front under the current pose. */
async function drawnBackToFront(page: Page, uuid: string, count: number): Promise<boolean> {
  return page.evaluate(
    async ({ uuid, count }) => {
      const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
      const ordering: number[] | null = await debug.readDrawnOrdering(uuid, count);
      const mesh = debug.scene.getObjectByProperty('uuid', uuid);
      const texels = mesh?.geometry?.userData?.elementTexture?.image?.data;
      if (!ordering || !texels) return false;
      const mwi = debug.camera.matrixWorldInverse.elements;
      const mw = mesh.matrixWorld.elements;
      let previous = -Infinity;
      for (const i of ordering) {
        const [x, y, z] = [texels[i * 16], texels[i * 16 + 1], texels[i * 16 + 2]];
        const wx = mw[0] * x + mw[4] * y + mw[8] * z + mw[12];
        const wy = mw[1] * x + mw[5] * y + mw[9] * z + mw[13];
        const wz = mw[2] * x + mw[6] * y + mw[10] * z + mw[14];
        const viewZ = mwi[2] * wx + mwi[6] * wy + mwi[10] * wz + mwi[14];
        if (viewZ < previous - 1e-4) return false;
        previous = Math.max(previous, viewZ);
      }
      return true;
    },
    { uuid, count }
  );
}

/** The depth-sort completions so far: their count and the last one's producer. */
async function completions(page: Page): Promise<{ total: number; route?: string }> {
  return page.evaluate(() => {
    const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
    const { total, events } = debug.getSceneLoader().getProfiler().getDepthSortCompletions();
    return { total, route: events.at(-1)?.route };
  });
}

async function orbit(page: Page, degrees: number): Promise<void> {
  await page.evaluate((deg) => {
    const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
    const pose = debug.app.getCameraPose();
    const rad = (deg * Math.PI) / 180;
    const [px, py, pz] = pose.position;
    const [tx, , tz] = pose.target;
    const dx = px - tx;
    const dz = pz - tz;
    debug.app.setCameraPose({
      ...pose,
      position: [
        tx + dx * Math.cos(rad) + dz * Math.sin(rad),
        py,
        tz - dx * Math.sin(rad) + dz * Math.cos(rad),
      ],
    });
    debug.renderOnce();
  }, degrees);
}

for (const gpuSort of [1, 0] as const) {
  const route = gpuSort ? 'gpu' : 'worker';
  test(`orbit re-sorts back to front on the ${route} (gpuSort=${gpuSort})`, async ({
    page,
  }, testInfo) => {
    test.skip(!isWebGPUProject(testInfo), 'needs a native WebGPU device');
    await page.goto(`/?src=${FIXTURE}&debug&dpr=1&gpuSort=${gpuSort}`);
    await waitForLuxarReady(page);
    await expect.poll(() => gsplatNode(page), { timeout: 30_000 }).not.toBeNull();
    const node = (await gsplatNode(page))!;
    await expect
      .poll(() => drawnBackToFront(page, node.uuid, node.count), { timeout: 30_000 })
      .toBe(true);

    for (const step of [60, 75, 90]) {
      const before = (await completions(page)).total;
      await orbit(page, step);
      await expect
        .poll(async () => (await completions(page)).total, { timeout: 15_000 })
        .toBeGreaterThan(before);
      expect((await completions(page)).route).toBe(route);
      await expect
        .poll(() => drawnBackToFront(page, node.uuid, node.count), { timeout: 15_000 })
        .toBe(true);
    }
  });
}

test('a capture refresh flips the order buffer on the GPU, without a host upload', async ({
  page,
}, testInfo) => {
  test.skip(!isWebGPUProject(testInfo), 'needs a native WebGPU device');
  await page.goto(`/?src=${FIXTURE}&debug&dpr=1`);
  await waitForLuxarReady(page);
  await expect.poll(() => gsplatNode(page), { timeout: 30_000 }).not.toBeNull();
  const node = (await gsplatNode(page))!;
  // The commit's first sort is the worker's; wait for it to be drawn.
  await expect
    .poll(() => drawnBackToFront(page, node.uuid, node.count), { timeout: 30_000 })
    .toBe(true);
  const state = await page.evaluate(async (uuid) => {
    const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
    const geometry = debug.scene.getObjectByProperty('uuid', uuid).geometry;
    const snapshot = () => ({
      slot: geometry.userData.sortedIndexSlot ?? 0,
      versions: [
        geometry.getAttribute('aSortedIndex').version,
        geometry.getAttribute('aSortedIndexB').version,
      ],
    });
    const before = snapshot();
    await debug.resortDepthOrderingForCapture();
    return { before, after: snapshot() };
  }, node.uuid);
  expect(state.after.slot).toBe(1 - state.before.slot);
  expect(state.after.versions).toEqual(state.before.versions);
  // A completion is recorded once the new ordering is drawn, and the capture
  // drain never draws: render the frame it was refreshed for.
  const before = (await completions(page)).total;
  await page.evaluate(() =>
    (window as unknown as { __luxarDebug?: any }).__luxarDebug.renderOnce()
  );
  await expect
    .poll(async () => (await completions(page)).total, { timeout: 15_000 })
    .toBeGreaterThan(before);
  expect((await completions(page)).route).toBe('gpu');
  await expect
    .poll(() => drawnBackToFront(page, node.uuid, node.count), { timeout: 15_000 })
    .toBe(true);
});
