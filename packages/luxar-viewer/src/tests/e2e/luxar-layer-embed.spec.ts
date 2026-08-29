import { expect, test } from './fixtures';
import { captureCanvasRGBA } from './helpers';
import type { Page } from '@playwright/test';

const EXAMPLE_URL = '/examples/layer';
const DATA_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const LOD_FIXTURE = `${DATA_BASE}/test_lod_group.luxar.zarr`;
const DIMENSION_FIXTURE = `${DATA_BASE}/test_layer_4d_gsplats.luxar.zarr`;
const CLEAR_COLOR_SUM = 0x05 + 0x08 + 0x12;

interface LayerExampleState {
  loaded: boolean;
  visibleSplatCount: number;
  visibleLodLevels: number[];
  dimensions: { names: string[]; currentStep: number[] } | null;
}

interface LayerExampleApi {
  dispose(): Promise<void>;
  getState(): LayerExampleState;
  setCameraDistance(distance: number): void;
  setDimensionValue(index: number, value: number): Promise<void>;
}

async function openLayerExample(page: Page, src: string) {
  await page.goto(`${EXAMPLE_URL}?src=${encodeURIComponent(src)}`);
  await page.waitForFunction(() => {
    const api = (window as Window & typeof globalThis & { __luxarLayerExample?: LayerExampleApi })
      .__luxarLayerExample;
    return api?.getState().loaded === true;
  });
}

async function getExampleState(page: Page): Promise<LayerExampleState> {
  return page.evaluate(() => {
    const api = (window as Window & typeof globalThis & { __luxarLayerExample?: LayerExampleApi })
      .__luxarLayerExample;
    if (!api) throw new Error('LuxarLayer example API is unavailable');
    return api.getState();
  });
}

test.describe('LuxarLayer host example', () => {
  test('renders real layer geometry into the host framebuffer', async ({ page }) => {
    await openLayerExample(page, LOD_FIXTURE);

    const state = await getExampleState(page);
    expect(state.visibleSplatCount).toBeGreaterThan(0);

    const frame = await captureCanvasRGBA(page, '#layer-canvas');
    let litPixels = 0;
    for (let index = 0; index < frame.rgba.length; index += 4) {
      if (frame.rgba[index] + frame.rgba[index + 1] + frame.rgba[index + 2] > CLEAR_COLOR_SUM + 8) {
        litPixels++;
      }
    }
    expect(litPixels).toBeGreaterThan(frame.width * frame.height * 0.01);
  });

  test('camera dolly changes the selected LOD geometry', async ({ page }) => {
    await openLayerExample(page, LOD_FIXTURE);
    const initial = await getExampleState(page);
    expect(initial.visibleLodLevels.length).toBeGreaterThan(0);

    const distances = [2, 4, 8, 16, 32, 64, 128];
    let distanceIndex = 0;
    await expect
      .poll(
        async () => {
          const state = await getExampleState(page);
          if (state.visibleSplatCount !== initial.visibleSplatCount) {
            return state.visibleSplatCount;
          }
          const distance = distances[Math.min(distanceIndex, distances.length - 1)];
          distanceIndex++;
          await page.evaluate((nextDistance) => {
            const api = (
              window as Window & typeof globalThis & { __luxarLayerExample?: LayerExampleApi }
            ).__luxarLayerExample;
            if (!api) throw new Error('LuxarLayer example API is unavailable');
            api.setCameraDistance(nextDistance);
          }, distance);
          return initial.visibleSplatCount;
        },
        {
          message: 'camera distance sweep should cross an LOD threshold',
          timeout: 5000,
          intervals: [0, 150, 150, 150, 150, 150, 150, 250, 500],
        }
      )
      .not.toBe(initial.visibleSplatCount);

    const changed = await getExampleState(page);
    expect(changed.visibleLodLevels).not.toEqual(initial.visibleLodLevels);
  });

  test('dispose removes the layer geometry', async ({ page }) => {
    await openLayerExample(page, LOD_FIXTURE);
    await page.evaluate(async () => {
      const api = (window as Window & typeof globalThis & { __luxarLayerExample?: LayerExampleApi })
        .__luxarLayerExample;
      if (!api) throw new Error('LuxarLayer example API is unavailable');
      await api.dispose();
    });

    await expect
      .poll(() => getExampleState(page))
      .toMatchObject({
        loaded: false,
        visibleSplatCount: 0,
        visibleLodLevels: [],
      });
  });

  test('setDimensionValue commits a different splat count', async ({ page }) => {
    await openLayerExample(page, DIMENSION_FIXTURE);
    const before = await getExampleState(page);
    const timeIndex = before.dimensions?.names.indexOf('time') ?? -1;
    expect(timeIndex).toBeGreaterThanOrEqual(0);
    expect(before.visibleSplatCount).toBe(24);

    await page.evaluate(
      async ({ index, value }) => {
        const api = (
          window as Window & typeof globalThis & { __luxarLayerExample?: LayerExampleApi }
        ).__luxarLayerExample;
        if (!api) throw new Error('LuxarLayer example API is unavailable');
        await api.setDimensionValue(index, value);
      },
      { index: timeIndex, value: 1 }
    );

    await expect.poll(() => getExampleState(page)).toMatchObject({ visibleSplatCount: 72 });
    const after = await getExampleState(page);
    expect(after.dimensions?.currentStep[timeIndex]).toBe(1);
  });
});
