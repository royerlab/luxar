import { test, expect } from './fixtures';
import { waitForDataLoaded, waitForLuxarReady } from './helpers';

const DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_2d.luxar.zarr';
const ROOT_METADATA = `${DATASET}/zarr.json`;

test('cinematic FOV preserves auto-framed screen occupancy', async ({ browser }) => {
  const loadAndMeasure = async (cinematic: boolean) => {
    const context = await browser.newContext({ baseURL: 'http://127.0.0.1:5173' });
    const page = await context.newPage();
    await page.route(ROOT_METADATA, async (route) => {
      if (route.request().method() !== 'GET') {
        await route.continue();
        return;
      }
      const response = await route.fetch();
      const metadata = (await response.json()) as {
        attributes: { viewer_config?: { cinematic_mode: boolean } };
      };
      if (cinematic) {
        metadata.attributes.viewer_config = { cinematic_mode: true };
      }
      await route.fulfill({ response, json: metadata });
    });

    await page.goto(`/?debug&src=${encodeURIComponent(DATASET)}`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
    const measurement = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const distance = debug.camera.position.distanceTo(debug.controls.getFocusTarget());
      return {
        fov: debug.camera.fov as number,
        halfFrameSpan: distance * Math.tan((debug.camera.fov * Math.PI) / 360),
      };
    });
    await context.close();
    return measurement;
  };

  const standard = await loadAndMeasure(false);
  const cinematicView = await loadAndMeasure(true);

  expect(standard.fov).toBe(47);
  expect(cinematicView.fov).toBe(63);
  expect(cinematicView.halfFrameSpan).toBeCloseTo(standard.halfFrameSpan, 5);
});
