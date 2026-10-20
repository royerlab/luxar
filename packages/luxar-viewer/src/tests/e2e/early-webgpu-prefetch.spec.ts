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

test('requests the adapter before the viewer entry runs', async ({ page }) => {
  await page.addInitScript(() => {
    (window as Window & { adapterRequests?: number }).adapterRequests = 0;
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      value: {
        requestAdapter: async () => {
          (window as Window & { adapterRequests?: number }).adapterRequests! += 1;
          return null;
        },
      },
    });
  });
  const { release } = await blockViewerEntry(page);
  try {
    await page.goto('/?renderer=webgpu', { waitUntil: 'commit' });
    await expect
      .poll(() =>
        page.evaluate(() => (window as Window & { adapterRequests?: number }).adapterRequests)
      )
      .toBeGreaterThanOrEqual(1);
  } finally {
    release();
  }
});

test('requests the device with forwarded limits before the viewer entry runs', async ({ page }) => {
  await page.addInitScript(() => {
    (window as Window & { deviceDescriptors?: unknown[] }).deviceDescriptors = [];
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      value: {
        requestAdapter: async () => ({
          features: new Set(['timestamp-query']),
          limits: {
            maxVertexBuffers: 20,
            maxBufferSize: 4096,
            maxStorageBufferBindingSize: 2048,
            maxTextureDimension2D: 16384,
          },
          requestDevice: async (descriptor: unknown) => {
            (window as Window & { deviceDescriptors?: unknown[] }).deviceDescriptors!.push(
              descriptor
            );
            return {};
          },
        }),
      },
    });
  });
  const { release } = await blockViewerEntry(page);
  try {
    await page.goto('/?renderer=webgpu', { waitUntil: 'commit' });
    await expect
      .poll(() =>
        page.evaluate(
          () => (window as Window & { deviceDescriptors?: unknown[] }).deviceDescriptors
        )
      )
      .toEqual([
        {
          requiredFeatures: ['timestamp-query'],
          requiredLimits: {
            maxVertexBuffers: 16,
            maxBufferSize: 4096,
            maxStorageBufferBindingSize: 2048,
            maxTextureDimension2D: 16384,
          },
        },
      ]);
  } finally {
    release();
  }
});

test('keeps the diagnostic WebGL bridge off the native adapter path', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', {
      configurable: true,
      get: () => {
        (window as Window & { gpuReads?: number }).gpuReads =
          ((window as Window & { gpuReads?: number }).gpuReads ?? 0) + 1;
        return { requestAdapter: async () => null };
      },
    });
  });
  const { requested, release } = await blockViewerEntry(page);
  try {
    await page.goto('/?renderer=webgpu&webgpuForceWebgl', { waitUntil: 'commit' });
    await expect
      .poll(() => requested.some((url) => url.includes('/rendering/tsl/registry.ts')))
      .toBe(true);
    expect(
      await page.evaluate(() => (window as Window & { gpuReads?: number }).gpuReads ?? 0)
    ).toBe(0);
  } finally {
    release();
  }
});

for (const scenario of [
  { name: 'WebGL URL', path: '/?renderer=webgl', stored: null },
  { name: 'default WebGL', path: '/', stored: null },
  { name: 'stored WebGL', path: '/', stored: 'webgl' },
]) {
  test(`does not request the WebGPU chunk for ${scenario.name}`, async ({ page }) => {
    await page.addInitScript((stored) => {
      if (stored) {
        localStorage.setItem(
          'luxar.settings',
          JSON.stringify({ version: 1, advanced: { renderer: stored } })
        );
      } else {
        localStorage.removeItem('luxar.settings');
      }
      Object.defineProperty(navigator, 'gpu', {
        configurable: true,
        get: () => {
          (window as Window & { gpuReads?: number }).gpuReads =
            ((window as Window & { gpuReads?: number }).gpuReads ?? 0) + 1;
          return undefined;
        },
      });
    }, scenario.stored);
    const { requested, release } = await blockViewerEntry(page);
    try {
      await page.goto(scenario.path, { waitUntil: 'commit' });
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
}

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
