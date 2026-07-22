/**
 * TEMPORARY verification probe (campaign 4, iteration 10) — DELETE ME.
 *
 * WebGPU-renderer-forced-to-WebGL battery: load three representative
 * datasets (points / lines / gsplats) under
 * `?renderer=webgpu&webgpu-force-webgl` (TSL NodeMaterial pipeline on
 * Three.js's internal WebGL2 backend) and assert:
 *   - capabilities.apiSurface === 'webgpu' (the diagnostic surface took)
 *   - expected element counts loaded (totalPoints/totalLines/totalGSplats)
 *   - render coverage > 0 in a UI-free center crop
 *   - zero page errors, zero console errors
 */

import { test, expect } from './fixtures';
import type { Page } from './fixtures';
import { waitForLuxarReady, waitForNextRender, assertNoConsoleErrors } from './helpers';

const CASES: Array<{
  name: string;
  url: string;
  countField: 'totalPoints' | 'totalLines' | 'totalGSplats';
}> = [
  {
    name: 'points',
    url: 'http://localhost:9000/datasets/examples/scene_dimensions_example.luxar.zarr',
    countField: 'totalPoints',
  },
  {
    name: 'lines',
    url: 'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lines.luxar.zarr',
    countField: 'totalLines',
  },
  {
    name: 'gsplats',
    url: 'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_gsplats.luxar.zarr',
    countField: 'totalGSplats',
  },
];

/** Crop that avoids the left rail/info panels, bottom nav panel and toasts. */
const CROP = { x0: 0.36, y0: 0.05, x1: 0.95, y1: 0.7 };

async function croppedCoverage(
  page: Page,
  savePath?: string
): Promise<{ frac: number; nonBlack: number }> {
  const buf = await page
    .locator('canvas')
    .first()
    .screenshot(savePath ? { path: savePath } : {});
  const dataUrl = `data:image/png;base64,${buf.toString('base64')}`;
  return await page.evaluate(
    async ({ url, crop }) => {
      const img = new Image();
      img.decoding = 'sync';
      const loaded = new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('croppedCoverage: decode failed'));
      });
      img.src = url;
      await loaded;
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      const off = document.createElement('canvas');
      off.width = w;
      off.height = h;
      const ctx = off.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      const sx = Math.floor(crop.x0 * w);
      const sy = Math.floor(crop.y0 * h);
      const sw = Math.floor((crop.x1 - crop.x0) * w);
      const sh = Math.floor((crop.y1 - crop.y0) * h);
      const pixels = ctx.getImageData(sx, sy, sw, sh).data;
      let nonBlack = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 10) nonBlack++;
      }
      return { frac: nonBlack / (sw * sh), nonBlack };
    },
    { url: dataUrl, crop: CROP }
  );
}

for (const c of CASES) {
  test(`iter10: ${c.name} renders under ?renderer=webgpu&webgpu-force-webgl`, async ({ page }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.goto(`/?src=${c.url}&renderer=webgpu&webgpu-force-webgl&debug`);
    await waitForLuxarReady(page);

    // The diagnostic surface took: WebGPURenderer API surface.
    const api = await page.evaluate(
      () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
    );
    expect(api).toBe('webgpu');

    // Wait for the expected geometry to load.
    await page.waitForFunction(
      (field: string) => {
        const state = (window as any).__luxarDebug?.getState?.();
        return state && state[field] > 0;
      },
      c.countField,
      { timeout: 30000 }
    );
    const state = await page.evaluate(() => (window as any).__luxarDebug.getState());
    await waitForNextRender(page, 4);

    const cover = await croppedCoverage(
      page,
      `delme/zz-probe4-evidence/iter10-${c.name}-forcewebgl.png`
    );
    console.log(
      `[probe4-iter10] ${c.name}: api=${api} ${c.countField}=${state[c.countField]} coverage=${cover.frac.toFixed(5)} (${cover.nonBlack}px) pageErrors=${pageErrors.length}`
    );

    expect(state[c.countField]).toBeGreaterThan(0);
    expect(cover.nonBlack).toBeGreaterThan(0);
    expect(pageErrors).toEqual([]);
    await assertNoConsoleErrors(page);
  });
}
