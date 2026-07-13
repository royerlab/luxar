/**
 * Unit tests for data-worker/initialize.ts — specifically that the
 * embedder's `wasmPath` is honored INSIDE the worker module scope.
 *
 * The main-thread `setWasmJsUrl` override does not cross into the worker
 * (separate module instance), so the pool forwards `wasmPath` over the
 * Comlink `initialize()` RPC and `initialize()` must apply it via
 * `setWasmJsUrl` BEFORE calling `initWasm()`. Without this, a relocated
 * WASM binary fails to load in the worker and silently falls back to the
 * slow TypeScript path on the hot path.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const calls: string[] = [];

vi.mock('../../../../wasm', () => {
  const fakeModule = { __isWasm: true };
  return {
    initWasm: vi.fn(async () => {
      calls.push('initWasm');
      return fakeModule;
    }),
    isWasmFallback: vi.fn(() => false),
    getFallback: vi.fn(() => ({ __isFallback: true })),
    setWasmJsUrl: vi.fn((url: string) => {
      calls.push(`setWasmJsUrl:${url}`);
    }),
  };
});

vi.mock('../../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn(), success: vi.fn() },
  Modules: { WORKER_POOL: 'WorkerPool' },
}));

import { initialize } from '../../../../workers/data-worker/initialize';
import { setWasmJsUrl } from '../../../../wasm';
import type { WasmCtx } from '../../../../workers/data-worker/state';

function makeCtx(): WasmCtx {
  return { wasm: null, tsFallback: null };
}

describe('worker initialize() — wasmPath forwarding', () => {
  beforeEach(() => {
    calls.length = 0;
    vi.clearAllMocks();
  });

  it('applies wasmPath via setWasmJsUrl BEFORE initWasm()', async () => {
    const ctx = makeCtx();
    await initialize(ctx, '/static/luxar/wasm/luxar_wasm.js');
    expect(setWasmJsUrl).toHaveBeenCalledExactlyOnceWith('/static/luxar/wasm/luxar_wasm.js');
    // Ordering matters: the override must be set before the module loads.
    expect(calls).toEqual(['setWasmJsUrl:/static/luxar/wasm/luxar_wasm.js', 'initWasm']);
    expect(ctx.wasm).not.toBeNull();
  });

  it('does not call setWasmJsUrl when no wasmPath is supplied', async () => {
    const ctx = makeCtx();
    await initialize(ctx);
    expect(setWasmJsUrl).not.toHaveBeenCalled();
    expect(calls).toEqual(['initWasm']);
  });

  it('reports wasmFallback=false when compiled WASM loads (custom path honored)', async () => {
    const ctx = makeCtx();
    const result = await initialize(ctx, '/custom/wasm.js');
    expect(result.wasmFallback).toBe(false);
  });
});
