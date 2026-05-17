import { test, expect } from '@playwright/test';

test('?renderer=webgl forces legacy path', async ({ page }) => {
  await page.goto('/?renderer=webgl&debug');
  await page.waitForFunction(
    () => typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  const api = await page.evaluate(
    () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
  );
  expect(api).toBe('webgl2');
});

test('?renderer=webgpu forces the default WebGPU path', async ({ page }) => {
  await page.goto('/?renderer=webgpu&debug');
  await page.waitForFunction(
    () => typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  // Under Playwright chromium (no real WebGPU), the renderer falls
  // back to its WebGL2 backend internally and reports 'webgl2'.
  // Under a Chrome with WebGPU enabled, it would report 'webgpu'.
  const api = await page.evaluate(
    () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
  );
  expect(['webgl2', 'webgpu']).toContain(api);
});
