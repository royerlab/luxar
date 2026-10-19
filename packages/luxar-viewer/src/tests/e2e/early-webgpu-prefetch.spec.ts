import { test, expect } from '@playwright/test';

test('requests the WebGPU material chunk before the viewer entry runs', async ({ page }) => {
  let releaseEntry: (() => void) | undefined;
  const entryBlocked = new Promise<void>((resolve) => {
    releaseEntry = resolve;
  });
  const requested: string[] = [];
  page.on('request', (request) => requested.push(request.url()));
  await page.route('**/src/core/main.ts', async (route) => {
    await entryBlocked;
    await route.fulfill({ status: 200, contentType: 'text/javascript', body: 'export {}' });
  });

  try {
    await page.goto('/?renderer=webgpu', { waitUntil: 'commit' });
    await expect
      .poll(() => requested.some((url) => url.includes('/rendering/tsl/registry.ts')))
      .toBe(true);
  } finally {
    releaseEntry?.();
  }
});
