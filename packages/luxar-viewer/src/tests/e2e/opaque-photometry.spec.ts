import { test, expect, type Page } from './fixtures';
import { waitForLuxarReady, waitForNextRender } from './helpers';

const POINTS_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_blending_modes.luxar.zarr';
const LINES_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lines_blending_modes.luxar.zarr';

async function meanCanvasLinearLuminance(page: Page): Promise<number> {
  const canvas = page.locator('canvas').first();
  await canvas.waitFor({ state: 'visible' });
  const png = await canvas.screenshot({ animations: 'disabled' });
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;

  return page.evaluate(async (url) => {
    const img = new Image();
    img.decoding = 'sync';
    const loaded = new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('opaque-photometry: failed to decode screenshot'));
    });
    img.src = url;
    await loaded;

    const offscreen = document.createElement('canvas');
    offscreen.width = img.naturalWidth;
    offscreen.height = img.naturalHeight;
    const context = offscreen.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('opaque-photometry: 2D context unavailable');
    context.drawImage(img, 0, 0);

    const data = context.getImageData(0, 0, offscreen.width, offscreen.height).data;
    const toLinear = (value: number): number => {
      const channel = value / 255;
      return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
    };

    let sum = 0;
    for (let index = 0; index < data.length; index += 4) {
      sum +=
        0.2126 * toLinear(data[index]) +
        0.7152 * toLinear(data[index + 1]) +
        0.0722 * toLinear(data[index + 2]);
    }
    return sum / (data.length / 4);
  }, dataUrl);
}

async function isolateOpaqueGeometry(page: Page, nodeType: 'points' | 'lines'): Promise<number> {
  return page.evaluate((type) => {
    let opaqueCount = 0;
    const debug = (window as any).__luxarDebug;
    debug.scene.traverse((object: any) => {
      if (object.userData?.nodeType !== type || !object.material) return;
      const isOpaque = object.material.userData?.blendingMode === 'opaque';
      object.visible = isOpaque;
      if (isOpaque && (object.geometry?.instanceCount ?? 0) > 0) opaqueCount++;
    });
    debug.renderOnce?.();
    return opaqueCount;
  }, nodeType);
}

async function setOpaqueOpacity(page: Page, nodeType: 'points' | 'lines', opacity: number) {
  return page.evaluate(
    ({ type, value }) => {
      let updated = 0;
      const debug = (window as any).__luxarDebug;
      debug.scene.traverse((object: any) => {
        if (
          object.userData?.nodeType !== type ||
          object.material?.userData?.blendingMode !== 'opaque'
        )
          return;
        const uniform = object.material.uniforms?.uOpacity;
        if (!uniform) return;
        uniform.value = value;
        updated++;
      });
      debug.renderOnce?.();
      return updated;
    },
    { type: nodeType, value: opacity }
  );
}

for (const [nodeType, fixture] of [
  ['points', POINTS_FIXTURE],
  ['lines', LINES_FIXTURE],
] as const) {
  test(`opaque ${nodeType} preserve fragment photometry`, async ({ page }) => {
    await page.goto(`/?src=${fixture}&debug&dpr=1&no-opfs`);
    await waitForLuxarReady(page);
    await page.waitForFunction(
      (type) => {
        let committed = 0;
        (window as any).__luxarDebug.scene.traverse((object: any) => {
          if (object.userData?.nodeType === type && (object.geometry?.instanceCount ?? 0) > 0)
            committed++;
        });
        return committed >= 6;
      },
      nodeType,
      { timeout: 60000 }
    );

    expect(await isolateOpaqueGeometry(page, nodeType)).toBeGreaterThan(0);
    await waitForNextRender(page, 2);
    const full = await meanCanvasLinearLuminance(page);

    expect(await setOpaqueOpacity(page, nodeType, 0.1)).toBeGreaterThan(0);
    await waitForNextRender(page, 2);
    const dimmed = await meanCanvasLinearLuminance(page);

    expect(full, `${nodeType}: isolated opaque geometry must render`).toBeGreaterThan(1e-5);
    expect(
      dimmed / full,
      `${nodeType}: opaque framebuffer output must respond to fragment alpha`
    ).toBeLessThan(0.35);
  });
}
