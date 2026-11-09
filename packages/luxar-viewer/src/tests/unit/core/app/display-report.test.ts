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
    const report = await buildDisplayReport(native, { requested: 'webgpu', nonce: 'n0nce' });
    expect(report).toMatchObject({
      api: 'webgpu',
      backend: 'webgpu-native',
      requested: 'webgpu',
      nonce: 'n0nce',
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
      { requested: 'webgpu', nonce: '' }
    );
    expect(none.webgpuAvailable).toBe(false);
    expect(none.adapter).toBeNull();
    withGpu(null);
    expect((await buildDisplayReport(native, { requested: null, nonce: '' })).adapter).toBeNull();
  });

  it('reads a software adapter from either place the browser puts the flag', async () => {
    withGpu({ info: { vendor: 'google' }, isFallbackAdapter: true });
    expect(
      (await buildDisplayReport(native, { requested: null, nonce: '' })).adapter?.isFallbackAdapter
    ).toBe(true);
  });
});

describe('the adapter probe', () => {
  it('gives up after a few seconds: a hung GPU process cannot stop the report', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('navigator', {
      ...navigator,
      gpu: { requestAdapter: () => new Promise(() => {}) },
    });
    const pending = buildDisplayReport(native, { requested: null, nonce: '' });
    await vi.advanceTimersByTimeAsync(3000);
    expect((await pending).adapter).toBeNull();
    vi.useRealTimers();
  });
});

describe('a viewer that failed to start', () => {
  it('still reports, with the reason and no renderer', async () => {
    withGpu(null);
    const post = vi.fn(async () => undefined);
    await sendDisplayReport(
      undefined,
      { requested: 'webgpu', nonce: 'n', error: 'WebGL is not supported' },
      post
    );
    const [, body] = post.mock.calls[0] as unknown as [string, string];
    expect(JSON.parse(body)).toMatchObject({
      api: null,
      backend: null,
      error: 'WebGL is not supported',
    });
  });
});

describe('sendDisplayReport', () => {
  it('posts the report as JSON to the same-origin path', async () => {
    withGpu(null);
    const post = vi.fn(async () => undefined);
    await sendDisplayReport(
      native,
      { requested: 'webgpu', nonce: 'n', datasetError: '404 data' },
      post
    );
    expect(post).toHaveBeenCalledTimes(1);
    const [url, body] = post.mock.calls[0] as unknown as [string, string];
    expect(url).toBe(new URL(DISPLAY_REPORT_PATH, window.location.origin).href);
    expect(JSON.parse(body)).toMatchObject({
      backend: 'webgpu-native',
      nonce: 'n',
      datasetError: '404 data',
    });
  });

  it('never throws: a server that does not listen changes nothing', async () => {
    withGpu(null);
    await expect(
      sendDisplayReport(native, { requested: null, nonce: '' }, async () => {
        throw new Error('404');
      })
    ).resolves.toBeUndefined();
  });
});
