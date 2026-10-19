import { test, expect, type Page } from '@playwright/test';

async function blockViewerEntry(page: Page): Promise<{
  requested: string[];
  release: () => void;
}> {
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

  return { requested, release: () => releaseEntry?.() };
}

for (const renderer of ['webgpu', 'WEBGPU']) {
  test(`requests the WebGPU chunk before the viewer entry runs (${renderer})`, async ({ page }) => {
    const { requested, release } = await blockViewerEntry(page);
    try {
      await page.goto(`/?renderer=${renderer}`, { waitUntil: 'commit' });
      await expect
        .poll(() => requested.some((url) => url.includes('/rendering/tsl/registry.ts')))
        .toBe(true);
    } finally {
      release();
    }
  });
}

test('does not request the WebGPU chunk for WebGL', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      get: () => {
        (window as Window & { gpuReads?: number }).gpuReads =
          ((window as Window & { gpuReads?: number }).gpuReads ?? 0) + 1;
        return undefined;
      },
    });
  });
  const { requested, release } = await blockViewerEntry(page);
  try {
    await page.goto('/?renderer=webgl', { waitUntil: 'commit' });
    release();
    await page.waitForLoadState('domcontentloaded');
    expect(requested.some((url) => url.includes('/rendering/tsl/registry.ts'))).toBe(false);
    expect(
      await page.evaluate(() => (window as Window & { gpuReads?: number }).gpuReads ?? 0)
    ).toBe(0);
  } finally {
    release();
  }
});

test('uses stored renderer preference before the viewer starts', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      'luxar.settings',
      JSON.stringify({ version: 1, advanced: { renderer: 'webgpu' } })
    );
  });
  const { requested, release } = await blockViewerEntry(page);
  try {
    await page.goto('/', { waitUntil: 'commit' });
    await expect
      .poll(() => requested.some((url) => url.includes('/rendering/tsl/registry.ts')))
      .toBe(true);
  } finally {
    release();
  }
});

test('WebGL URL override beats a stored WebGPU preference', async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      'luxar.settings',
      JSON.stringify({ version: 1, advanced: { renderer: 'webgpu' } })
    );
  });
  const { requested, release } = await blockViewerEntry(page);
  try {
    await page.goto('/?renderer=WEBGL2', { waitUntil: 'commit' });
    release();
    await page.waitForLoadState('domcontentloaded');
    expect(requested.some((url) => url.includes('/rendering/tsl/registry.ts'))).toBe(false);
  } finally {
    release();
  }
});
