/**
 * Tests for the page-load graphics-API probe.
 *
 * The probe has three branches:
 *   - `navigator.gpu.requestAdapter()` resolves to a truthy adapter → `'webgpu'`
 *   - WebGL2 context creates → `'webgl2'`
 *   - neither → `'unsupported'`
 *
 * Tests stub `navigator.gpu` and `HTMLCanvasElement.getContext` to
 * exercise each branch without needing real GPU support.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  getRendererAPI,
  getRendererAPISync,
  _resetCachedAPI,
} from '../../../utils/webgpu-availability';

const originalGetContext = HTMLCanvasElement.prototype.getContext;
let originalGpu: PropertyDescriptor | undefined;

function setGpu(value: unknown): void {
  originalGpu = Object.getOwnPropertyDescriptor(navigator, 'gpu');
  Object.defineProperty(navigator, 'gpu', {
    value,
    configurable: true,
    writable: true,
  });
}

function restoreGpu(): void {
  if (originalGpu) {
    Object.defineProperty(navigator, 'gpu', originalGpu);
  } else {
    delete (navigator as unknown as { gpu?: unknown }).gpu;
  }
  originalGpu = undefined;
}

function stubWebGL2(available: boolean): void {
  HTMLCanvasElement.prototype.getContext = vi.fn((type: string) => {
    if (type === 'webgl2') return available ? {} : null;
    return null;
  }) as unknown as typeof HTMLCanvasElement.prototype.getContext;
}

describe('getRendererAPI', () => {
  beforeEach(() => {
    _resetCachedAPI();
  });

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = originalGetContext;
    restoreGpu();
    _resetCachedAPI();
  });

  it('returns "webgpu" when navigator.gpu requestAdapter resolves to an adapter', async () => {
    setGpu({ requestAdapter: vi.fn().mockResolvedValue({ name: 'fake-adapter' }) });
    await expect(getRendererAPI()).resolves.toBe('webgpu');
  });

  it('falls back to "webgl2" when navigator.gpu is missing', async () => {
    restoreGpu();
    stubWebGL2(true);
    await expect(getRendererAPI()).resolves.toBe('webgl2');
  });

  it('falls back to "webgl2" when requestAdapter resolves to null', async () => {
    setGpu({ requestAdapter: vi.fn().mockResolvedValue(null) });
    stubWebGL2(true);
    await expect(getRendererAPI()).resolves.toBe('webgl2');
  });

  it('falls back to "webgl2" when requestAdapter throws', async () => {
    setGpu({
      requestAdapter: vi.fn().mockRejectedValue(new Error('no adapter')),
    });
    stubWebGL2(true);
    await expect(getRendererAPI()).resolves.toBe('webgl2');
  });

  it('returns "unsupported" when neither WebGPU nor WebGL2 is available', async () => {
    restoreGpu();
    stubWebGL2(false);
    await expect(getRendererAPI()).resolves.toBe('unsupported');
  });

  it('caches the probe result across calls', async () => {
    const requestAdapter = vi.fn().mockResolvedValue({ name: 'fake-adapter' });
    setGpu({ requestAdapter });
    await getRendererAPI();
    await getRendererAPI();
    await getRendererAPI();
    expect(requestAdapter).toHaveBeenCalledTimes(1);
  });
});

describe('getRendererAPISync', () => {
  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = originalGetContext;
    restoreGpu();
  });

  it('reports "webgpu" when navigator.gpu is present (no adapter probe)', () => {
    // utils.md W6 / Phase E62 strengthening: pin the no-adapter-probe
    // contract by asserting `requestAdapter` was never called. A
    // regression turning the sync path into async-and-await would still
    // return the correct string from the cached probe but would now be
    // observable here. Without this assertion the sync vs async paths
    // are indistinguishable from the caller's perspective.
    const requestAdapter = vi.fn();
    setGpu({ requestAdapter });
    expect(getRendererAPISync()).toBe('webgpu');
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it('reports "webgl2" when navigator.gpu is absent but webgl2 context creates', () => {
    restoreGpu();
    stubWebGL2(true);
    expect(getRendererAPISync()).toBe('webgl2');
  });

  it('reports "unsupported" when neither path is available', () => {
    restoreGpu();
    stubWebGL2(false);
    expect(getRendererAPISync()).toBe('unsupported');
  });
});
