// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  selectBackend,
  shouldPrefetchTsl,
} from '../../../../../scene/scene-manager/render-pipeline/backend-selection';
import { STAGED_WEBGPU_PLATFORM_RULES } from '../../../../../scene/scene-manager/render-pipeline/backend-policy';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('early TSL prefetch decision', () => {
  function stubLinuxChromiumWithGpu(): void {
    vi.stubGlobal('navigator', {
      userAgentData: { platform: 'Linux', brands: [{ brand: 'Chromium' }] },
      gpu: { requestAdapter: vi.fn() },
    });
  }

  it('defaults to WebGL on Linux + Chromium while the WebGPU rows are held (#3172)', () => {
    stubLinuxChromiumWithGpu();
    vi.stubEnv('VITE_LUXAR_WEBGPU_STAGED_RULES', '');
    expect(selectBackend(undefined)).toEqual({
      backend: 'webgl',
      source: 'platform-policy',
      requireNativeAdapter: true,
    });
  });

  it('waits for adapter vendor qualification under the platform policy', () => {
    stubLinuxChromiumWithGpu();
    const selection = selectBackend(undefined, undefined, STAGED_WEBGPU_PLATFORM_RULES);
    expect(selection.backend).toBe('webgpu');
    expect(selection.platformRules?.map((rule) => rule.vendor)).toEqual(['nvidia']);
    expect(shouldPrefetchTsl(true, selection.requireNativeAdapter, selection.platformRules)).toBe(
      false
    );
  });

  it('prefetches for explicit and environment WebGPU selections', () => {
    stubLinuxChromiumWithGpu();
    const explicit = selectBackend('webgpu');
    expect(shouldPrefetchTsl(true, explicit.requireNativeAdapter, explicit.platformRules)).toBe(
      true
    );
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const fromEnv = selectBackend(undefined);
    expect(fromEnv.source).toBe('env-var');
    expect(shouldPrefetchTsl(true, fromEnv.requireNativeAdapter, fromEnv.platformRules)).toBe(true);
  });
});
