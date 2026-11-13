/** The viewer chooses the texture-based GPU sorter for a native WebGPU refresh. */
import { expect, test } from './fixtures';
import { isWebGPUProject } from './webgpu-project';
import { waitForLuxarReady, waitForRenderStable } from './helpers';
import type * as THREE from 'three';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap_reversed.luxar.zarr';

test('native GSplat refresh flips the order buffer without a worker upload', async ({ page }) => {
  test.setTimeout(120_000);
  test.skip(!isWebGPUProject(), 'requires a native WebGPU adapter');
  await page.goto(`/?src=${FIXTURE}&debug&dpr=1`);
  await waitForLuxarReady(page);
  await waitForRenderStable(page);
  await page.waitForFunction(async () => {
    const modulePath = '/src/rendering/depth-sort-coordinator/gpu-order-producer.ts';
    const { canGpuSort } = (await import(
      /* @vite-ignore */ modulePath
    )) as typeof import('../../rendering/depth-sort-coordinator/gpu-order-producer');
    let ready = false;
    window.__luxarDebug?.scene?.traverse((object) => {
      const mesh = object as unknown as {
        userData?: { nodeType?: string };
        geometry?: THREE.InstancedBufferGeometry;
      };
      if (mesh.userData?.nodeType !== 'gsplats' || !mesh.geometry) return;
      ready = canGpuSort(mesh.geometry, mesh.geometry.instanceCount);
    });
    return ready;
  });
  const state = await page.evaluate(async () => {
    const debug = window.__luxarDebug!;
    if (!debug.scene || !debug.resortDepthOrderingForCapture) {
      throw new Error('depth-sort debug interface missing');
    }
    let geometry: THREE.InstancedBufferGeometry | undefined;
    debug.scene.traverse((object) => {
      const mesh = object as unknown as {
        userData?: { nodeType?: string };
        geometry?: THREE.InstancedBufferGeometry;
      };
      if (mesh.userData?.nodeType === 'gsplats') geometry = mesh.geometry;
    });
    const g = geometry as THREE.InstancedBufferGeometry | undefined;
    if (!g) throw new Error('sorted GSplat geometry missing');
    const snapshot = () => ({
      slot: g.userData.sortedIndexSlot ?? 0,
      versions: [
        (g.getAttribute('aSortedIndex') as THREE.BufferAttribute).version,
        (g.getAttribute('aSortedIndexB') as THREE.BufferAttribute).version,
      ],
    });
    const before = snapshot();
    await debug.resortDepthOrderingForCapture();
    return { before, after: snapshot() };
  });
  expect(state.after.slot).toBe(1 - state.before.slot);
  expect(state.after.versions).toEqual(state.before.versions);
});
