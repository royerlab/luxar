// @vitest-environment jsdom
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
// The backend three ends up on after init() (HDR output needs real WebGPU).
const fakeBackend = vi.hoisted(() => ({ isWebGPUBackend: true, device: {} }));
// What `(dynamic-range: high)` answers for the HDR-output gate.
const displayHdr = vi.hoisted(() => ({ hdr: false }));
const webgpuImport = vi.hoisted(() => ({ failures: 0, calls: 0 }));
vi.mock('three/webgpu', () => {
  webgpuImport.calls++;
  if (webgpuImport.failures-- > 0) throw new Error('chunk unavailable');
  class FakeWebGPURenderer {
    init = vi.fn(async () => {});
    backend = fakeBackend;
    constructor(opts: unknown) {
      webgpuConstructorArgs.push(opts);
    }
  }
  return { WebGPURenderer: FakeWebGPURenderer };
});
const loadTslMaterials = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
vi.mock('../../../../../rendering/tsl/load', () => ({ loadTslMaterials }));

// Mock the capabilities + HDR helpers — they run after WebGPURenderer
// construction in createWebGPURenderer's epilogue.
vi.mock('../../../../../rendering/renderer-capabilities', () => ({
  createRendererCapabilities: vi.fn(() => ({
    apiSurface: 'webgpu',
    backendKind: 'webgpu-native',
    hdr: {},
    pointSizeRange: [1, 64],
  })),
  probeUnmaskedGpuRenderer: vi.fn(() => null),
}));
vi.mock('../../../../../utils/hdr/hdr-detection', () => ({
  configureHDRRenderer: vi.fn(),
  logHDRCapabilities: vi.fn(),
  detectDisplayCapabilities: vi.fn(() => ({ hdr: displayHdr.hdr })),
  HDR_DISPLAY_QUERY: '(dynamic-range: high)',
}));

import * as THREE from 'three';
// Imports MUST come after vi.mock to honour the mocks.
import {
  createUncapturedGpuErrorHandler,
  createWebGPURenderer,
  GPU_ERROR_DISTINCT_LIMIT,
  GPU_ERROR_REPEAT_LIMIT,
  requestWebGPUDevice,
  selectBackend,
} from '../../../../../scene/scene-manager/render-pipeline/renderer-setup';
import { log, Modules } from '../../../../../utils/log';
import {
  forwardedAdapterLimits,
  preRequestWebGPUDevice,
  takePreRequestedWebGPUDevice,
} from '../../../../../scene/scene-manager/render-pipeline/early-device-request';

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
    expect(r).toEqual({ backend: 'webgl', source: 'url-param', requireNativeAdapter: true });
  });

  it('honours an explicit ?renderer=webgpu URL override', () => {
    const r = selectBackend('webgpu');
    expect(r).toEqual({ backend: 'webgpu', source: 'url-param', requireNativeAdapter: false });
  });

  it('respects VITE_LUXAR_USE_WEBGPU=1 env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgpu', source: 'env-var', requireNativeAdapter: true });
    vi.unstubAllEnvs();
  });

  it('respects VITE_LUXAR_USE_WEBGPU_RENDERER=1 env var alias', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU_RENDERER', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgpu', source: 'env-var', requireNativeAdapter: true });
    vi.unstubAllEnvs();
  });

  it('respects VITE_LUXAR_USE_LEGACY_WEBGL=1 env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_LEGACY_WEBGL', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgl', source: 'env-var', requireNativeAdapter: true });
    vi.unstubAllEnvs();
  });

  it('URL override beats env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const r = selectBackend('webgl');
    expect(r).toEqual({ backend: 'webgl', source: 'url-param', requireNativeAdapter: true });
    vi.unstubAllEnvs();
  });

  it('defaults to webgl when neither URL nor env vars are set', () => {
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgl', source: 'default', requireNativeAdapter: true });
  });

  it('a WebGPU default requires a native adapter; an explicit ?renderer=webgpu does not', () => {
    expect(selectBackend(undefined, 'webgpu')).toEqual({
      backend: 'webgpu',
      source: 'default',
      requireNativeAdapter: true,
    });
    expect(selectBackend('webgpu', 'webgpu').requireNativeAdapter).toBe(false);
    expect(selectBackend('webgl', 'webgpu').backend).toBe('webgl');
  });
});

describe('forwardedAdapterLimits', () => {
  it('forwards each advertised numeric limit into requiredLimits', () => {
    // The maxTextureDimension2D forwarding is the per-node element
    // ceiling: without it the device falls back to the 8192-row spec
    // default and a lines node silently clamps at 5,586,944 segments.
    // Via a variable: the adapter's real limits object carries many more
    // keys than the forwarded trio, and only object LITERALS get
    // excess-property checking — this mirrors the production call shape.
    const adapterLimits = {
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 512,
      maxTextureDimension2D: 16384,
      maxVertexBuffers: 30, // handled separately (clamped) — never forwarded here
    };
    expect(forwardedAdapterLimits(adapterLimits)).toEqual({
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
    vi.mocked(loadTslMaterials).mockClear();
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
      writable: true,
    });
  });

  /** `undefined` stubs a browser without `navigator.gpu` at all. */
  function stubNavigatorGpu(requestAdapter: (() => Promise<unknown>) | undefined): void {
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        ...originalNavigator,
        gpu: requestAdapter === undefined ? undefined : { requestAdapter },
      },
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

  it('starts both imports while requestDevice is pending and retries a failed renderer prefetch', async () => {
    let releaseDevice!: (device: object) => void;
    const requestDevice = vi.fn(
      () =>
        new Promise<object>((resolve) => {
          releaseDevice = resolve;
        })
    );
    const adapter = makeAdapter({ maxVertexBuffers: 16, requestDevice });
    stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));
    webgpuImport.failures = 1;
    loadTslMaterials.mockClear();

    const creating = createWebGPURenderer(makeCanvas(), {});
    await vi.waitFor(() => expect(requestDevice).toHaveBeenCalledOnce());
    expect(loadTslMaterials).toHaveBeenCalledOnce();
    await new Promise((resolve) => setTimeout(resolve, 0));
    releaseDevice({});
    expect(await creating).toMatchObject({ fallback: false });
    expect(webgpuImport.calls).toBe(2);
  });

  it('consumes an early device request once without renegotiating', async () => {
    const adapter = makeAdapter({ maxVertexBuffers: 16 });
    const requestAdapter = vi.fn().mockResolvedValue(adapter);
    stubNavigatorGpu(requestAdapter);

    preRequestWebGPUDevice('webgpu');
    expect(requestAdapter).toHaveBeenCalledOnce();
    const early = takePreRequestedWebGPUDevice();
    expect(early).toBeDefined();
    expect(takePreRequestedWebGPUDevice()).toBeUndefined();

    const result = await requestWebGPUDevice('webgpu', false, early);
    expect(result).toMatchObject({ fallback: false });
    expect(requestAdapter).toHaveBeenCalledOnce();
    expect(adapter.requestDevice).toHaveBeenCalledOnce();
  });

  it('keeps the low-limit adapter fallback before an early device request', async () => {
    const adapter = makeAdapter({ maxVertexBuffers: 4 });
    const requestAdapter = vi.fn().mockResolvedValue(adapter);
    stubNavigatorGpu(requestAdapter);

    preRequestWebGPUDevice();
    const result = await requestWebGPUDevice(undefined, true, takePreRequestedWebGPUDevice());
    expect(result).toEqual({ fallback: true });
    expect(requestAdapter).toHaveBeenCalledOnce();
    expect(adapter.requestDevice).not.toHaveBeenCalled();
  });

  it('falls back when an early device request rejects', async () => {
    const requestDevice = vi.fn().mockRejectedValue(new Error('device exhausted'));
    const adapter = makeAdapter({ maxVertexBuffers: 16, requestDevice });
    const requestAdapter = vi.fn().mockResolvedValue(adapter);
    stubNavigatorGpu(requestAdapter);

    preRequestWebGPUDevice();
    const result = await requestWebGPUDevice(undefined, true, takePreRequestedWebGPUDevice());
    expect(result).toEqual({ fallback: true });
    expect(requestAdapter).toHaveBeenCalledOnce();
    expect(requestDevice).toHaveBeenCalledOnce();
  });

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

  it('uses a pre-requested device once without negotiating a second adapter', async () => {
    const device = { label: 'early device' };
    let resolveDevice!: (device: unknown) => void;
    const requestDevice = vi.fn(() => new Promise<unknown>((resolve) => (resolveDevice = resolve)));
    const adapter = makeAdapter({ maxVertexBuffers: 16, requestDevice });
    const requestAdapter = vi.fn().mockResolvedValue(adapter);
    stubNavigatorGpu(requestAdapter);

    const early = requestWebGPUDevice('webgpu');
    await vi.waitFor(() => expect(requestDevice).toHaveBeenCalledTimes(1));
    expect(loadTslMaterials).toHaveBeenCalledTimes(1);
    const creating = createWebGPURenderer(makeCanvas(), { preRequestedWebGPUDevice: early });
    expect(webgpuConstructorArgs).toHaveLength(0);
    resolveDevice(device);
    expect(await creating).toMatchObject({ fallback: false });
    expect(webgpuConstructorArgs[0]).toMatchObject({ device });
    expect(requestAdapter).toHaveBeenCalledTimes(1);
    expect(requestDevice).toHaveBeenCalledTimes(1);
  });

  it('preserves early adapter fallback without constructing a WebGPU renderer', async () => {
    const adapter = makeAdapter({ maxVertexBuffers: 4 });
    let resolveAdapter!: (adapter: ReturnType<typeof makeAdapter>) => void;
    stubNavigatorGpu(vi.fn(() => new Promise((resolve) => (resolveAdapter = resolve))));

    const early = requestWebGPUDevice();
    expect(loadTslMaterials).toHaveBeenCalledTimes(1);
    resolveAdapter(adapter);
    const result = await createWebGPURenderer(makeCanvas(), { preRequestedWebGPUDevice: early });
    expect(result).toEqual({ fallback: true });
    expect(adapter.requestDevice).not.toHaveBeenCalled();
    expect(webgpuConstructorArgs).toHaveLength(0);
  });

  it('routes uncaptured GPU errors through log.error', async () => {
    const adapter = makeAdapter({ maxVertexBuffers: 16 });
    stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));
    const error = vi.spyOn(log, 'error').mockImplementation(() => {});

    const result = await createWebGPURenderer(makeCanvas(), {});
    if (result.fallback) throw new Error('expected a renderer');
    const onError = (result.renderer as unknown as { onError: (info: unknown) => void }).onError;
    expect(typeof onError).toBe('function');
    onError({ api: 'WebGPU', type: 'GPUValidationError', message: 'bad bind group' });
    expect(error).toHaveBeenCalledWith(
      Modules.RENDERER,
      'Uncaptured WebGPU GPUValidationError: bad bind group'
    );
    error.mockRestore();
  });

  it('requestDevice failure recovers silently: renderer constructed without device option', async () => {
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
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
    expect(warning).toHaveBeenCalledWith(
      Modules.RENDERER,
      'WebGPU requestDevice failed (Error: device exhausted).'
    );
    warning.mockRestore();
  });

  describe('requireNativeAdapter', () => {
    it('skips the TSL prefetch when native WebGPU is unavailable', async () => {
      stubNavigatorGpu(undefined);
      expect(await requestWebGPUDevice(undefined, true)).toEqual({ fallback: true });
      expect(loadTslMaterials).not.toHaveBeenCalled();
    });

    it('prefetches TSL for the explicit WebGPU bridge without navigator.gpu', async () => {
      stubNavigatorGpu(undefined);
      expect(await requestWebGPUDevice('webgpu', false)).toEqual({
        fallback: false,
        device: undefined,
      });
      expect(loadTslMaterials).toHaveBeenCalledOnce();
    });

    it('falls back to WebGL without navigator.gpu, before importing three/webgpu', async () => {
      stubNavigatorGpu(undefined);
      const result = await createWebGPURenderer(makeCanvas(), { requireNativeAdapter: true });
      expect(result).toEqual({ fallback: true });
      expect(webgpuConstructorArgs).toHaveLength(0);
    });

    it('falls back to WebGL when no adapter is returned', async () => {
      stubNavigatorGpu(vi.fn().mockResolvedValue(null));
      const result = await createWebGPURenderer(makeCanvas(), { requireNativeAdapter: true });
      expect(result).toEqual({ fallback: true });
      expect(webgpuConstructorArgs).toHaveLength(0);
    });

    it('falls back to WebGL when requestDevice fails', async () => {
      const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
      const info = vi.spyOn(log, 'info').mockImplementation(() => {});
      const adapter = makeAdapter({
        maxVertexBuffers: 16,
        requestDevice: vi.fn().mockRejectedValue(new Error('device exhausted')),
      });
      stubNavigatorGpu(vi.fn().mockResolvedValue(adapter));
      const result = await createWebGPURenderer(makeCanvas(), { requireNativeAdapter: true });
      expect(result).toEqual({ fallback: true });
      expect(webgpuConstructorArgs).toHaveLength(0);
      expect(warning).toHaveBeenCalledWith(
        Modules.RENDERER,
        'WebGPU requestDevice failed (Error: device exhausted).'
      );
      expect(info).toHaveBeenCalledWith(
        Modules.RENDERER,
        'No native WebGPU device (requestDevice failed); using WebGLRenderer.'
      );
      warning.mockRestore();
      info.mockRestore();
    });

    it('builds the renderer on the native device when one is available', async () => {
      stubNavigatorGpu(vi.fn().mockResolvedValue(makeAdapter({ maxVertexBuffers: 16 })));
      const result = await createWebGPURenderer(makeCanvas(), { requireNativeAdapter: true });
      expect(result).toMatchObject({ fallback: false });
      expect((webgpuConstructorArgs[0] as { device?: unknown }).device).toBeDefined();
    });

    it('is ignored under webgpuForceWebGL, which asks for the WebGL2 backend', async () => {
      stubNavigatorGpu(undefined);
      const result = await createWebGPURenderer(makeCanvas(), {
        requireNativeAdapter: true,
        webgpuForceWebGL: true,
      });
      expect(result).toMatchObject({ fallback: false });
      expect(webgpuConstructorArgs[0]).toMatchObject({ forceWebGL: true });
    });
  });

  describe('HDR output (?hdr)', () => {
    /**
     * Every canvas's WebGPU context (the support probe configures a throwaway
     * one) reports `toneMapping.mode` = `mode` after `configure`.
     */
    function hdrCanvas(mode: string | null): HTMLCanvasElement {
      vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
        configure: vi.fn(),
        unconfigure: vi.fn(),
        getConfiguration: () => (mode === null ? null : { toneMapping: { mode } }),
      } as never);
      return makeCanvas();
    }

    beforeEach(() => {
      displayHdr.hdr = true;
      fakeBackend.isWebGPUBackend = true;
      stubNavigatorGpu(vi.fn().mockResolvedValue(makeAdapter({ maxVertexBuffers: 16 })));
    });
    afterEach(() => {
      displayHdr.hdr = false;
      fakeBackend.isWebGPUBackend = true;
      vi.restoreAllMocks();
    });

    const outputType = (): unknown =>
      (webgpuConstructorArgs[0] as { outputType?: unknown }).outputType;

    it('presents an rgba16float extended canvas when true HDR is achievable', async () => {
      const result = await createWebGPURenderer(hdrCanvas('extended'), {
        hdr: true,
        hdrHeadroom: 4,
      });
      expect(outputType()).toBe(THREE.HalfFloatType);
      expect(result).toMatchObject({
        fallback: false,
        capabilities: { hdrOutput: { headroom: 4, pinned: true } },
      });
    });

    it('clamps a pinned headroom into the slider range', async () => {
      const result = await createWebGPURenderer(hdrCanvas('extended'), {
        hdr: true,
        hdrHeadroom: 99,
      });
      expect(result).toMatchObject({ capabilities: { hdrOutput: { headroom: 16 } } });
    });

    it('stays SDR (no float canvas) when HDR output is off', async () => {
      const result = await createWebGPURenderer(hdrCanvas('extended'), { hdr: false });
      expect(outputType()).toBeUndefined();
      expect(result).not.toHaveProperty('capabilities.hdrOutput');
    });

    it('falls back to an SDR canvas on a display that does not report HDR', async () => {
      displayHdr.hdr = false;
      const result = await createWebGPURenderer(hdrCanvas('extended'), { hdr: true });
      expect(outputType()).toBeUndefined();
      expect(result).not.toHaveProperty('capabilities.hdrOutput');
    });

    it('falls back to SDR without a native adapter (three would use WebGL2)', async () => {
      stubNavigatorGpu(vi.fn().mockResolvedValue(null));
      const result = await createWebGPURenderer(hdrCanvas('extended'), { hdr: true });
      expect(outputType()).toBeUndefined();
      expect(result).not.toHaveProperty('capabilities.hdrOutput');
    });

    it('falls back to SDR tone mapping when three ended up on its WebGL2 backend', async () => {
      fakeBackend.isWebGPUBackend = false;
      const result = await createWebGPURenderer(hdrCanvas('extended'), { hdr: true });
      expect(result).not.toHaveProperty('capabilities.hdrOutput');
    });

    it('keeps the canvas SDR when the browser cannot do extended tone mapping', async () => {
      // Probed before the renderer exists: no rgba16float canvas to pay for.
      for (const mode of ['standard', null]) {
        webgpuConstructorArgs.length = 0;
        const result = await createWebGPURenderer(hdrCanvas(mode), { hdr: true });
        expect(outputType()).toBeUndefined();
        expect(result).not.toHaveProperty('capabilities.hdrOutput');
      }
    });

    it('keeps the canvas SDR when the device request failed (three picks its own)', async () => {
      stubNavigatorGpu(
        vi.fn().mockResolvedValue(
          makeAdapter({
            maxVertexBuffers: 16,
            requestDevice: vi.fn().mockRejectedValue(new Error('device exhausted')),
          })
        )
      );
      const result = await createWebGPURenderer(hdrCanvas('extended'), { hdr: true });
      expect(outputType()).toBeUndefined();
      expect(result).not.toHaveProperty('capabilities.hdrOutput');
    });

    it('treats a headroom pinned to 1x as SDR output', async () => {
      const result = await createWebGPURenderer(hdrCanvas('extended'), {
        hdr: true,
        hdrHeadroom: 1,
      });
      expect(outputType()).toBeUndefined();
      expect(result).not.toHaveProperty('capabilities.hdrOutput');
    });
  });
});

describe('createUncapturedGpuErrorHandler', () => {
  const info = (message: string) => ({ api: 'WebGPU', type: 'GPUValidationError', message });

  function messages(spy: { mock: { calls: unknown[][] } }): string[] {
    return spy.mock.calls.map((call) => String(call[1]));
  }

  it('logs the first repeats of a message, one suppression note, then nothing', () => {
    const error = vi.spyOn(log, 'error').mockImplementation(() => {});
    const handle = createUncapturedGpuErrorHandler();
    for (let frame = 0; frame < 100; frame++) handle(info('per-frame failure'));
    const text = 'Uncaptured WebGPU GPUValidationError: per-frame failure';
    expect(messages(error)).toEqual([
      ...Array<string>(GPU_ERROR_REPEAT_LIMIT).fill(text),
      `Suppressing further repeats of: ${text}`,
    ]);
    expect(error.mock.calls.every((call) => call[0] === Modules.RENDERER)).toBe(true);
    error.mockRestore();
  });

  it('counts each distinct message separately', () => {
    const error = vi.spyOn(log, 'error').mockImplementation(() => {});
    const handle = createUncapturedGpuErrorHandler();
    handle(info('a'));
    handle(info('b'));
    handle(info('a'));
    expect(messages(error)).toHaveLength(3);
    error.mockRestore();
  });

  it('stops tracking new messages past the distinct limit with one note', () => {
    const error = vi.spyOn(log, 'error').mockImplementation(() => {});
    const handle = createUncapturedGpuErrorHandler();
    for (let i = 0; i < GPU_ERROR_DISTINCT_LIMIT + 10; i++) handle(info(`message ${i}`));
    const logged = messages(error);
    expect(logged).toHaveLength(GPU_ERROR_DISTINCT_LIMIT + 1);
    expect(logged.at(-1)).toContain('suppressing all further GPU errors');
    error.mockRestore();
  });

  it('accepts the plain-string shape @types/three declares', () => {
    const error = vi.spyOn(log, 'error').mockImplementation(() => {});
    createUncapturedGpuErrorHandler()('raw message');
    expect(error).toHaveBeenCalledWith(Modules.RENDERER, 'raw message');
    error.mockRestore();
  });
});
