/**
 * `getSharedWasmModule()` — compile the WASM binary once on the main thread so
 * every data worker instantiates from the same `WebAssembly.Module` instead of
 * compiling its own (measured: 15 workers came ready ~160 ms apart, 2.2 s from
 * first to last).
 *
 * The contract that matters most here is the NEGATIVE one: this is an
 * optimization, so every failure path must resolve `null` — never reject, never
 * hang — and let each worker fall back to self-initializing. A rejection or a
 * stall would take the whole pool down with it.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getSharedWasmModule, resetSharedWasmModule } from '../../../wasm/shared-module';

/** Stand-in for a compiled module; identity is what the tests assert on. */
const FAKE_MODULE = { __fake: 'wasm-module' } as unknown as WebAssembly.Module;

/**
 * An http shim URL. Required in these tests because the default candidates
 * resolve to `file://` under vitest, which the loader skips by design (see the
 * non-http test below) — so without an override every case would trivially
 * resolve `null` and pass for the wrong reason.
 */
const HTTP_SHIM = 'https://example.test/wasm/luxar_wasm.js';

function stubWasm(overrides: {
  compileStreaming?: unknown;
  compile?: unknown;
  clone?: unknown;
  fetchImpl?: unknown;
}) {
  vi.stubGlobal('WebAssembly', {
    ...WebAssembly,
    compileStreaming: overrides.compileStreaming ?? vi.fn(async () => FAKE_MODULE),
    compile: overrides.compile ?? vi.fn(async () => FAKE_MODULE),
  });
  vi.stubGlobal('fetch', overrides.fetchImpl ?? vi.fn(async () => new Response(new Uint8Array())));
  vi.stubGlobal('structuredClone', overrides.clone ?? vi.fn((value: unknown) => value));
}

describe('getSharedWasmModule', () => {
  beforeEach(() => {
    resetSharedWasmModule();
    vi.unstubAllGlobals();
    void new Response(new Uint8Array());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    resetSharedWasmModule();
  });

  it('compiles once and hands the same module to every concurrent caller', async () => {
    const compileStreaming = vi.fn(async () => FAKE_MODULE);
    stubWasm({ compileStreaming });

    // 15 workers all reach for it at once, as they do in production.
    const results = await Promise.all(
      Array.from({ length: 15 }, () => getSharedWasmModule(HTTP_SHIM))
    );

    expect(compileStreaming).toHaveBeenCalledTimes(1);
    expect(new Set(results).size).toBe(1);
    expect(results[0]).toBe(FAKE_MODULE);
  });

  it('falls back to buffered compile when compileStreaming rejects on a bad MIME type', async () => {
    // Plenty of static hosts serve .wasm as application/octet-stream, which
    // `compileStreaming` refuses outright. Losing the optimization to a
    // response header would be a silly way to lose it.
    const compileStreaming = vi.fn(async () => {
      throw new TypeError('Incorrect response MIME type');
    });
    const compile = vi.fn(async () => FAKE_MODULE);
    stubWasm({ compileStreaming, compile });

    await expect(getSharedWasmModule(HTTP_SHIM)).resolves.toBe(FAKE_MODULE);
    expect(compile).toHaveBeenCalledTimes(1);
  });

  it('resolves null — never rejects — when every candidate fails', async () => {
    stubWasm({
      compileStreaming: vi.fn(async () => {
        throw new Error('nope');
      }),
      compile: vi.fn(async () => {
        throw new Error('nope either');
      }),
    });

    await expect(getSharedWasmModule(HTTP_SHIM)).resolves.toBeNull();
  });

  it('resolves null when the module is not structured-cloneable', async () => {
    // THE degradation that matters: without this probe, an un-cloneable module
    // becomes a DataCloneError inside Comlink's postMessage, which surfaces as
    // a synchronous throw in the init guard and fails EVERY worker identically
    // — strictly worse than never having tried.
    stubWasm({
      clone: vi.fn(() => {
        throw new DOMException('could not be cloned', 'DataCloneError');
      }),
    });

    await expect(getSharedWasmModule(HTTP_SHIM)).resolves.toBeNull();
  });

  it('resolves null instead of hanging when the fetch never settles', async () => {
    vi.useFakeTimers();
    stubWasm({
      compileStreaming: vi.fn(() => new Promise<never>(() => {})),
      compile: vi.fn(() => new Promise<never>(() => {})),
      fetchImpl: vi.fn(() => new Promise<never>(() => {})),
    });

    const pending = getSharedWasmModule(HTTP_SHIM);
    await vi.advanceTimersByTimeAsync(30_000);
    await expect(pending).resolves.toBeNull();
    vi.useRealTimers();
  });

  it('resolves null when the host has no WebAssembly.compile', async () => {
    vi.stubGlobal('WebAssembly', { ...WebAssembly, compile: undefined });
    vi.stubGlobal('fetch', vi.fn());

    await expect(getSharedWasmModule()).resolves.toBeNull();
  });

  it('never fetches a non-http candidate', async () => {
    // Under Node/SSR the shim candidates resolve to `file://`, which `fetch`
    // cannot serve — and depending on the runtime it stalls rather than
    // throwing, which would delay every worker's init by the full deadline.
    const seen: string[] = [];
    const fetchImpl = vi.fn(async (url: string) => {
      seen.push(String(url));
      return new Response(new Uint8Array());
    });
    stubWasm({ fetchImpl });

    await expect(getSharedWasmModule('file:///tmp/luxar_wasm.js')).resolves.toBeNull();
    expect(seen).toEqual([]);
  });

  it('prefers an explicit shim-URL override over the default candidates', async () => {
    const seen: string[] = [];
    const compileStreaming = vi.fn(async (input: Promise<Response>) => {
      await input;
      return FAKE_MODULE;
    });
    stubWasm({
      compileStreaming,
      fetchImpl: vi.fn(async (url: string) => {
        seen.push(String(url));
        return new Response(new Uint8Array());
      }),
    });

    await getSharedWasmModule('https://cdn.example.com/custom/luxar_wasm.js');

    // The binary is resolved as a SIBLING of the shim, which is exactly how the
    // generated shim resolves it for itself — so `setDataWorkerWasmPath` keeps
    // working without a second override.
    expect(seen[0]).toBe('https://cdn.example.com/custom/luxar_wasm_bg.wasm');
  });
});
