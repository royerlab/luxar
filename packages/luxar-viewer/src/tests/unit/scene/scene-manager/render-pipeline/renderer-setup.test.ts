/**
 * Unit tests for the renderer-setup helpers used by SceneManager.
 *
 * `createWebGLRenderer` is intentionally NOT unit-tested: mocking the
 * WebGL2 context surface (getContext / getExtension / getParameter /
 * getShaderPrecisionFormat) would test the mock, not the GL contract.
 * E2E coverage via `basic-rendering.spec.ts` is the safety net there.
 *
 * `selectBackend` is the pure-fn precedence ladder.
 * `createWebGPURenderer` has 5 logical branches worth pinning:
 *   - forceWebGL diagnostic mode bypasses navigator.gpu;
 *   - adapter sufficiency fallback when maxVertexBuffers < spec minimum;
 *   - rendererOverride='webgpu' forces through a degenerate adapter;
 *   - the requiredLimits descriptor handed to requestDevice;
 *   - requestDevice failure recovers silently (no `device` option).
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock `three/webgpu` so `WebGPURenderer` resolves to a
// constructible-but-inert stub. The stub captures its constructor
// args so we can assert on them per branch.
const webgpuConstructorArgs = vi.hoisted<Array<unknown>>(() => []);
vi.mock('three/webgpu', () => {
  class FakeWebGPURenderer {
    init = vi.fn(async () => {});
    constructor(opts: unknown) {
      webgpuConstructorArgs.push(opts);
    }
  }
  return { WebGPURenderer: FakeWebGPURenderer };
});

// Mock the capabilities + HDR helpers — they run after WebGPURenderer
// construction in createWebGPURenderer's epilogue.
vi.mock('../../../../../rendering/renderer-capabilities', () => ({
  createRendererCapabilities: vi.fn(() => ({
    apiSurface: 'webgpu',
    hdr: {},
    pointSizeRange: [1, 64],
  })),
}));
vi.mock('../../../../../utils/hdr/hdr-detection', () => ({
  configureHDRRenderer: vi.fn(),
  logHDRCapabilities: vi.fn(),
}));

// Imports MUST come after vi.mock to honour the mocks.
import {
  createWebGPURenderer,
  forwardedAdapterLimits,
  selectBackend,
} from '../../../../../scene/scene-manager/render-pipeline/renderer-setup';

describe('selectBackend', () => {
  const originalEnv = { ...import.meta.env };

  beforeEach(() => {
    // Clean env between cases.
    delete (import.meta.env as Record<string, string | undefined>).VITE_LUXAR_USE_WEBGPU;
    delete (import.meta.env as Record<string, string | undefined>).VITE_LUXAR_USE_WEBGPU_RENDERER;
    delete (import.meta.env as Record<string, string | undefined>).VITE_LUXAR_USE_LEGACY_WEBGL;
  });

  afterEach(() => {
    Object.assign(import.meta.env, originalEnv);
  });

  it('honours an explicit ?renderer=webgl URL override', () => {
    const r = selectBackend('webgl');
    expect(r).toEqual({ backend: 'webgl', source: 'url-param' });
  });

  it('honours an explicit ?renderer=webgpu URL override', () => {
    const r = selectBackend('webgpu');
    expect(r).toEqual({ backend: 'webgpu', source: 'url-param' });
  });

  it('respects VITE_LUXAR_USE_WEBGPU=1 env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgpu', source: 'env-var' });
    vi.unstubAllEnvs();
  });

  it('respects VITE_LUXAR_USE_WEBGPU_RENDERER=1 env var alias', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU_RENDERER', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgpu', source: 'env-var' });
    vi.unstubAllEnvs();
  });

  it('respects VITE_LUXAR_USE_LEGACY_WEBGL=1 env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_LEGACY_WEBGL', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgl', source: 'env-var' });
    vi.unstubAllEnvs();
  });

  it('URL override beats env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const r = selectBackend('webgl');
    expect(r).toEqual({ backend: 'webgl', source: 'url-param' });
    vi.unstubAllEnvs();
  });

  it('defaults to webgl when neither URL nor env vars are set', () => {
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgl', source: 'default' });
  });
});

describe('forwardedAdapterLimits', () => {
  it('forwards each advertised numeric limit into requiredLimits', () => {
    // The maxTextureDimension2D forwarding is the per-node element
    // ceiling: without it the device falls back to the 8192-row spec
    // default and a lines node silently clamps at 5,586,944 segments.
    expect(
      forwardedAdapterLimits({
        maxBufferSize: 1024,
        maxStorageBufferBindingSize: 512,
        maxTextureDimension2D: 16384,
        maxVertexBuffers: 30, // handled separately (clamped) — never forwarded here
      })
    ).toEqual({
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 512,
      maxTextureDimension2D: 16384,
    });
  });

  it('omits absent or non-numeric limits so the device keeps spec defaults', () => {
    expect(forwardedAdapterLimits(undefined)).toEqual({});
    expect(forwardedAdapterLimits({})).toEqual({});
    expect(
      forwardedAdapterLimits({
        maxBufferSize: undefined,
        maxTextureDimension2D: 'huge' as unknown as number,
      })
    ).toEqual({});
  });
});

describe('createWebGPURenderer', () => {
  // Each test stubs navigator.gpu separately. The shared helpers below
  // build adapters/canvases with the minimum surface the helper reads.
  let originalNavigator: typeof globalThis.navigator;

  beforeEach(() => {
    originalNavigator = globalThis.navigator;
    webgpuConstructorArgs.length = 0;
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
      writable: true,
    });
  });

  function stubNavigatorGpu(requestAdapter: () => Promise<unknown>): void {
    Object.defineProperty(globalThis, 'navigator', {
      value: { ...originalNavigator, gpu: { requestAdapter } },
      configurable: true,
      writable: true,
    });
  }

  function makeCanvas(): HTMLCanvasElement {
    return document.createElement('canvas');
  }

  function makeAdapter(opts: {
    maxVertexBuffers?: number;
    requestDevice?: ReturnType<typeof vi.fn>;
  }) {
    const requestDevice = opts.requestDevice ?? vi.fn().mockResolvedValue({});
    return {
      features: new Set<string>(),
      limits: {
        maxVertexBuffers: opts.maxVertexBuffers,
        maxBufferSize: 4 * 1024 * 1024 * 1024,
        maxStorageBufferBindingSize: 1 * 1024 * 1024 * 1024,
        maxTextureDimension2D: 16384,
      },
      requestDevice,
    };
  }

  it('forceWebGL diagnostic mode: skips navigator.gpu and sets forceWebGL on renderer', async () => {
    // Even if navigator.gpu exists, forceWebGL must bypass it.
    const requestAdapter = vi.fn();
    stubNavigatorGpu(requestAdapter);

    const result = await createWebGPURenderer(makeCanvas(), { webgpuForceWebGL: true });

    expect(requestAdapter).not.toHaveBeenCalled();
    expect(result).toMatchObject({ fallback: false });
    expect(webgpuConstructorArgs).toHaveLength(1);
    expect(webgpuConstructorArgs[0]).toMatchObject({ forceWebGL: true });
    // Mutual exclusivity: forceWebGL means no pre-built device override.
    expect((webgpuConstructorArgs[0] as { device?: unknown }).device).toBeUndefined();
  });

  it('returns { fallback: true } when adapter advertises maxVertexBuffers below the spec minimum', async () => {
    const adapter = makeAdapter({ maxVertexBuffers: 4 });
    stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));

    const result = await createWebGPURenderer(makeCanvas(), {});

    expect(result).toEqual({ fallback: true });
    // Helper short-circuits before WebGPURenderer construction.
    expect(webgpuConstructorArgs).toHaveLength(0);
    // And before requesting a device.
    expect(adapter.requestDevice).not.toHaveBeenCalled();
  });

  it("rendererOverride='webgpu' forces through a degenerate adapter (no fallback)", async () => {
    const adapter = makeAdapter({ maxVertexBuffers: 4 });
    stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));

    const result = await createWebGPURenderer(makeCanvas(), { rendererOverride: 'webgpu' });

    expect(result).toMatchObject({ fallback: false });
    expect(adapter.requestDevice).toHaveBeenCalledTimes(1);
    expect(webgpuConstructorArgs).toHaveLength(1);
  });

  it('requests the adapter limits on the device, with maxVertexBuffers clamped to 16', async () => {
    // The descriptor is the whole point of the helper: a device that
    // does not ask for maxTextureDimension2D keeps the 8192-row spec
    // default, which clamps a lines node to 5,586,944 segments.
    const adapter = makeAdapter({ maxVertexBuffers: 32 });
    stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));

    await createWebGPURenderer(makeCanvas(), {});

    expect(adapter.requestDevice).toHaveBeenCalledTimes(1);
    const descriptor = adapter.requestDevice.mock.calls[0][0] as {
      requiredLimits: Record<string, number>;
    };
    expect(descriptor.requiredLimits).toEqual({
      maxBufferSize: 4 * 1024 * 1024 * 1024,
      maxStorageBufferBindingSize: 1 * 1024 * 1024 * 1024,
      maxTextureDimension2D: 16384,
      maxVertexBuffers: 16,
    });
  });

  it('requestDevice failure recovers silently: renderer constructed without device option', async () => {
    const adapter = makeAdapter({
      maxVertexBuffers: 16,
      requestDevice: vi.fn().mockRejectedValue(new Error('device exhausted')),
    });
    stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));

    const result = await createWebGPURenderer(makeCanvas(), {});

    // Helper does NOT throw on device-creation failure; it falls
    // through to Three's internal compat-mode init by omitting the
    // `device` option from the renderer ctor.
    expect(result).toMatchObject({ fallback: false });
    expect(webgpuConstructorArgs).toHaveLength(1);
    expect((webgpuConstructorArgs[0] as { device?: unknown }).device).toBeUndefined();
    expect((webgpuConstructorArgs[0] as { forceWebGL?: boolean }).forceWebGL).toBeUndefined();
  });
});
