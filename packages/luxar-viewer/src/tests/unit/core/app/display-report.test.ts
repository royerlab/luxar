// @vitest-environment jsdom
/**
 * The display report (`core/app/display-report.ts`): what the page renders
 * with, posted once to `/display-report` for `serve.py --chrome`.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DISPLAY_REPORT_PATH,
  buildDisplayReport,
  sendDisplayReport,
} from '../../../../core/app/display-report';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

const native = { apiSurface: 'webgpu', backendKind: 'webgpu-native' } as RendererCapabilities;

function withGpu(adapter: unknown): void {
  vi.stubGlobal('navigator', {
    ...navigator,
    userAgent: 'TestAgent/1.0',
    gpu: { requestAdapter: vi.fn(async () => adapter) },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('buildDisplayReport', () => {
  it('reports the renderer, the high-performance adapter and the window', async () => {
    withGpu({
      info: {
        vendor: 'nvidia',
        architecture: 'ampere',
        device: '',
        description: 'NVIDIA RTX',
        isFallbackAdapter: false,
      },
    });
    const report = await buildDisplayReport(native, 'webgpu');
    expect(report).toMatchObject({
      api: 'webgpu',
      backend: 'webgpu-native',
      requested: 'webgpu',
      webgpuAvailable: true,
      adapter: { vendor: 'nvidia', architecture: 'ampere', isFallbackAdapter: false },
      userAgent: 'TestAgent/1.0',
    });
    expect(report.window.width).toBe(window.innerWidth);
    expect(report.screen.width).toBe(window.screen.width);
  });

  it('says when there is no WebGPU at all, or no adapter', async () => {
    vi.stubGlobal('navigator', { ...navigator, gpu: undefined });
    const none = await buildDisplayReport(
      { apiSurface: 'webgl2', backendKind: 'webgl' } as RendererCapabilities,
      'webgpu'
    );
    expect(none.webgpuAvailable).toBe(false);
    expect(none.adapter).toBeNull();
    withGpu(null);
    expect((await buildDisplayReport(native, null)).adapter).toBeNull();
  });

  it('reads a software adapter from either place the browser puts the flag', async () => {
    withGpu({ info: { vendor: 'google' }, isFallbackAdapter: true });
    expect((await buildDisplayReport(native, null)).adapter?.isFallbackAdapter).toBe(true);
  });
});

describe('sendDisplayReport', () => {
  it('posts the report as JSON to the same-origin path', async () => {
    withGpu(null);
    const post = vi.fn(async () => undefined);
    await sendDisplayReport(native, 'webgpu', post);
    expect(post).toHaveBeenCalledTimes(1);
    const [url, body] = post.mock.calls[0] as unknown as [string, string];
    expect(url).toBe(new URL(DISPLAY_REPORT_PATH, window.location.origin).href);
    expect(JSON.parse(body).backend).toBe('webgpu-native');
  });

  it('never throws: a server that does not listen changes nothing', async () => {
    withGpu(null);
    await expect(
      sendDisplayReport(native, null, async () => {
        throw new Error('404');
      })
    ).resolves.toBeUndefined();
  });
});
