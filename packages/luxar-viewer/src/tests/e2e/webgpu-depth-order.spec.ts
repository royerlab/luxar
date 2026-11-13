/** Native WebGPU compute ordering must match stable counting-sort output. */
import { expect, test } from './fixtures';
import { isWebGPUProject } from './webgpu-project';
import { waitForLuxarReady, waitForRenderStable } from './helpers';

const FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_normal_overlap_reversed.luxar.zarr';

test('compute-built depth order is a stable permutation across tile boundaries', async ({
  page,
}) => {
  test.skip(!isWebGPUProject(), 'requires a native WebGPU adapter');
  await page.goto('/');
  const results = await page.evaluate(async () => {
    const harnessPath = '/src/tests/e2e/harnesses/webgpu-order-harness.ts';
    const { probe } = (await import(/* @vite-ignore */ harnessPath)) as {
      probe(): Promise<Array<{ n: number; equal: boolean }>>;
    };
    return probe();
  });
  for (const result of results) {
    expect(result.equal, `n=${result.n}: ${JSON.stringify(result)}`).toBe(true);
  }
});

test('native GSplat draw consumes a compute-built order without an index upload', async ({
  page,
}) => {
  test.skip(!isWebGPUProject(), 'requires a native WebGPU adapter');
  await page.goto(`/?src=${FIXTURE}&debug&dpr=1`);
  await waitForLuxarReady(page);
  await waitForRenderStable(page);
  await page.waitForFunction(() => {
    let selected = false;
    window.__luxarDebug?.scene?.traverse((object) => {
      const mesh = object as unknown as {
        userData?: { nodeType?: string };
        geometry?: { userData?: { sortedIndexSlot?: number } };
      };
      if (mesh.userData?.nodeType === 'gsplats' && mesh.geometry?.userData?.sortedIndexSlot === 1)
        selected = true;
    });
    return selected;
  });
  const state = await page.evaluate(() => {
    let result: { version: number; storage: boolean; order: number[] } | null = null;
    window.__luxarDebug?.scene?.traverse((object) => {
      const mesh = object as unknown as {
        userData?: { nodeType?: string };
        geometry?: {
          userData?: { sortedIndexSlot?: number };
          instanceCount: number;
          getAttribute(name: string): {
            version: number;
            isStorageInstancedBufferAttribute?: boolean;
            array: Uint32Array;
          };
        };
      };
      if (mesh.userData?.nodeType !== 'gsplats') return;
      const attr = mesh.geometry?.getAttribute('aSortedIndexB');
      if (attr && mesh.geometry?.userData?.sortedIndexSlot === 1) {
        result = {
          version: attr.version,
          storage: attr.isStorageInstancedBufferAttribute === true,
          order: Array.from(attr.array.slice(0, mesh.geometry.instanceCount)),
        };
      }
    });
    return result as { version: number; storage: boolean; order: number[] } | null;
  });
  expect(state).not.toBeNull();
  if (!state) throw new Error('sorted GSplat node was not found');
  expect(state.storage).toBe(true);
  expect(state.version).toBe(0);
  expect(state.order.slice().sort((a, b) => a - b)).toEqual([0, 1, 2]);
  expect(state.order).not.toEqual([0, 1, 2]);
});
