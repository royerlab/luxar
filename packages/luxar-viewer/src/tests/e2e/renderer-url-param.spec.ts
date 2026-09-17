import { test, expect } from '@playwright/test';

test('default (no URL flag) uses the WebGL path', async ({ page }) => {
  await page.goto('/?debug');
  await page.waitForFunction(
    () =>
      typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  const api = await page.evaluate(
    () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
  );
  expect(api).toBe('webgl2');
});

test('?renderer=webgl pins the WebGL path', async ({ page }) => {
  await page.goto('/?renderer=webgl&debug');
  await page.waitForFunction(
    () =>
      typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  const api = await page.evaluate(
    () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
  );
  expect(api).toBe('webgl2');
});

test('?renderer=webgpu opts into the WebGPU path', async ({ page }) => {
  await page.goto('/?renderer=webgpu&debug');
  await page.waitForFunction(
    () =>
      typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  // Under Playwright chromium (no real WebGPU), WebGPURenderer falls
  // back to its WebGL2 backend internally. Either way `caps.apiSurface`
  // reports 'webgpu' (the renderer surface), not the underlying
  // backend.
  const api = await page.evaluate(
    () => (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface
  );
  expect(api).toBe('webgpu');
});

test('?renderer=webgl reports framebufferYDown=false (bottom-up FBO)', async ({ page }) => {
  await page.goto('/?renderer=webgl&debug');
  await page.waitForFunction(
    () =>
      typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  const framebufferYDown = await page.evaluate(
    () => (window as any).__luxarDebug.app.sceneManager.capabilities.framebufferYDown
  );
  expect(framebufferYDown).toBe(false);
});

test('?renderer=webgpu&webgpuForceWebgl reports framebufferYDown=true', async ({ page }) => {
  // The forceWebGL diagnostic path routes draws through Three.js's
  // WebGL2 backend underneath, but WebGPURenderer normalises Y so its
  // user-visible output matches real WebGPU — i.e. the effective
  // framebuffer is top-down. Don't confuse the backing API
  // (`backend.isWebGLBackend === true`) with the effective Y layout.
  await page.goto('/?renderer=webgpu&webgpuForceWebgl&debug');
  await page.waitForFunction(
    () =>
      typeof (window as any).__luxarDebug?.app?.sceneManager?.capabilities?.apiSurface === 'string',
    null,
    { timeout: 10000 }
  );
  const probe = await page.evaluate(() => ({
    api: (window as any).__luxarDebug.app.sceneManager.capabilities.apiSurface,
    framebufferYDown: (window as any).__luxarDebug.app.sceneManager.capabilities.framebufferYDown,
  }));
  expect(probe.api).toBe('webgpu');
  expect(probe.framebufferYDown).toBe(true);
});
