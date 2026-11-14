import { chromium } from '@playwright/test';
import { describe, expect, it, vi } from 'vitest';

import { browserKit, caseUrl, selectBackends } from './browser.mjs';

vi.mock('@playwright/test', () => ({ chromium: { launch: vi.fn() } }));

describe('render-gate browser launch', () => {
  async function launch(config, perf = false, extraArgs = []) {
    const browser = { isConnected: () => true, close: vi.fn() };
    vi.mocked(chromium.launch).mockResolvedValueOnce(browser);
    const session = browserKit({
      channel: 'chrome',
      headless: false,
      chromeArgs: [],
      log: vi.fn(),
      ...config,
    }).browserSession(perf, extraArgs);
    await session.run((opened) => opened);
    await session.close();
    return vi.mocked(chromium.launch).mock.lastCall[0];
  }

  it('keeps the forced GPU setup by default', async () => {
    const options = await launch({});
    expect(options.channel).toBe('chrome');
    expect(options.args).toContain('--ignore-gpu-blocklist');
    expect(options.args).toContain('--enable-unsafe-webgpu');
    if (process.platform === 'linux') expect(options.args).toContain('--use-angle=vulkan');
  });

  it('uses the requested executable and omits forced flags in stock mode', async () => {
    const options = await launch({
      stock: true,
      executable: '/tmp/chrome',
      chromeArgs: ['--ozone-platform=wayland'],
    });
    expect(options.executablePath).toBe('/tmp/chrome');
    expect(options.channel).toBeUndefined();
    expect(options.headless).toBe(false);
    expect(options.args).toEqual(['--ozone-platform=wayland']);
  });

  it('keeps perf unpaced unless --vsync is requested, including in stock mode', async () => {
    const unpaced = await launch({ stock: true }, true, ['--ignore-certificate-errors']);
    expect(unpaced.args).toEqual([
      '--disable-gpu-vsync',
      '--disable-frame-rate-limit',
      '--ignore-certificate-errors',
    ]);
    const paced = await launch({ stock: true, vsync: true }, true);
    expect(paced.args).toEqual([]);
  });
});

describe('render-gate backend URLs', () => {
  it('forces the WebGPURenderer WebGL2 fallback for the exact arm', () => {
    expect(caseUrl('http://localhost:4801', {}, 'webgpu-gl', 1, 'lodFinest')).toBe(
      'http://localhost:4801/?debug&dpr=1&renderer=webgpu&webgpuForceWebgl&lodFinest'
    );
  });
});

describe('render-gate backend selection', () => {
  it('keeps only perf backends when --suite all requests every exact arm', () => {
    expect(selectBackends(['webgl', 'webgpu', 'webgpu-gl'], ['webgl', 'webgpu'])).toEqual([
      'webgl',
      'webgpu',
    ]);
  });

  it('uses allowed defaults and respects a case-specific backend restriction', () => {
    expect(selectBackends(null, ['webgl', 'webgpu-gl'])).toEqual(['webgl', 'webgpu-gl']);
    expect(selectBackends(['webgl', 'webgpu-gl'], ['webgpu-gl'])).toEqual(['webgpu-gl']);
  });
});
