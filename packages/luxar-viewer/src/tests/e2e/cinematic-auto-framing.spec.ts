import type { Browser } from '@playwright/test';
import { test, expect } from './fixtures';
import { waitForDataLoaded, waitForLuxarReady } from './helpers';
import { StorageKeys } from '../../utils/storage-keys';

const DATASET =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats_2d.luxar.zarr';
// This fixture is format 3; the explicit 63° assertion below catches a future format-2 mismatch.
const ROOT_METADATA = `${DATASET}/zarr.json`;
const SCENE_ID = DATASET.replace(/[^a-zA-Z0-9]/g, '_');

interface LoadOptions {
  cinematic: boolean;
  authoredPosition?: [number, number, number];
  storedFov?: number;
}

const loadAndMeasure = async (
  browser: Browser,
  { cinematic, authoredPosition, storedFov }: LoadOptions
) => {
  const context = await browser.newContext();
  if (storedFov !== undefined) {
    // The per-scene document is a `{ version, settings }` envelope
    // (settings-persistence.ts); a bare `{ fov }` would be discarded as
    // pre-envelope. `version: 1` = RENDERING_SETTINGS_VERSION, pinned by
    // rendering-controls-persistence.test.ts (not imported here: that module
    // pulls the viewer config into the Playwright Node context).
    await context.addInitScript(
      ({ key, fov }) =>
        localStorage.setItem(key, JSON.stringify({ version: 1, settings: { fov } })),
      { key: StorageKeys.rendering(SCENE_ID), fov: storedFov }
    );
  }
  const page = await context.newPage();
  await page.route(ROOT_METADATA, async (route) => {
    if (route.request().method() !== 'GET') {
      await route.continue();
      return;
    }
    const response = await route.fetch();
    const metadata = (await response.json()) as {
      attributes: { viewer_config?: Record<string, unknown> };
    };
    if (cinematic || authoredPosition) {
      const viewerConfig = metadata.attributes.viewer_config ?? {};
      metadata.attributes.viewer_config = {
        ...viewerConfig,
        ...(cinematic ? { cinematic_mode: true } : {}),
        ...(authoredPosition
          ? {
              camera: {
                ...((viewerConfig.camera as Record<string, unknown> | undefined) ?? {}),
                position: authoredPosition,
                target: [1.5, 1.5, 0],
              },
            }
          : {}),
      };
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
      position: debug.camera.position.toArray() as [number, number, number],
      // The fixture is planar, so nearestDepth is zero and this is the exact fitted span.
      halfFrameSpan: distance * Math.tan((debug.camera.fov * Math.PI) / 360),
    };
  });
  await context.close();
  return measurement;
};

test('cinematic FOV preserves auto-framed screen occupancy', async ({ browser }) => {
  const standard = await loadAndMeasure(browser, { cinematic: false });
  const cinematicView = await loadAndMeasure(browser, { cinematic: true });

  expect(standard.fov).toBe(47);
  expect(cinematicView.fov).toBe(63);
  expect(cinematicView.halfFrameSpan).toBeCloseTo(standard.halfFrameSpan, 5);
});

test('authored position carries its cinematic FOV past stored settings', async ({ browser }) => {
  // This auto-framed leg proves SCENE_ID still matches the viewer's derived
  // storage key; otherwise the authored-position assertion could pass without
  // exercising stored-settings precedence at all.
  const storedAutoFrame = await loadAndMeasure(browser, {
    cinematic: true,
    storedFov: 47,
  });
  const authoredPosition: [number, number, number] = [1.5, 1.5, 10];
  const authoredView = await loadAndMeasure(browser, {
    cinematic: true,
    authoredPosition,
    storedFov: 47,
  });

  expect(storedAutoFrame.fov).toBe(47);
  expect(authoredView.fov).toBe(63);
  expect(authoredView.position).toEqual(authoredPosition);
});
