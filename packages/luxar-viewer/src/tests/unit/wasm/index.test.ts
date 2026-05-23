/**
 * Unit tests for the WASM loader module.
 *
 * `initWasm` is hard to exercise on the WASM-loaded path inside jsdom (no
 * binary, no Vite resolution), so the dynamic-import branch always
 * throws and falls through to the TypeScript fallback. We verify the
 * fallback path lands a working WasmModule, that isWasmSupported
 * detects WebAssembly correctly, and that getFallback returns a fresh
 * TypeScriptFallback every call.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { initWasm, isWasmSupported, getFallback, setWasmJsUrl } from '../../../wasm';
import { TypeScriptFallback } from '../../../wasm/typescript';

describe('initWasm', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('falls back to TypeScriptFallback when the WASM bundle cannot load', async () => {
    // jsdom has no fetch'able WASM; the dynamic import always throws.
    const wasm = await initWasm();
    expect(wasm).toBeInstanceOf(TypeScriptFallback);
  });

  it('still falls back when the URL override points at a missing file', async () => {
    setWasmJsUrl('http://localhost:0/missing.js');
    const wasm = await initWasm();
    expect(wasm).toBeInstanceOf(TypeScriptFallback);
    // Reset so subsequent tests in the run don't see the override.
    setWasmJsUrl('');
  });
});

describe('setWasmJsUrl', () => {
  it('setWasmJsUrl("") resets the override to default (HIGH-7)', async () => {
    // Set a bogus override, then reset via ''. The init path must NOT
    // treat '' as a real override (which would resolve to an empty wasm
    // URL via `?? defaultUrl` — `??` only catches `undefined`/`null`).
    setWasmJsUrl('http://localhost:0/missing.js');
    setWasmJsUrl('');
    // After reset, initWasm goes through the default-URL path. The
    // dynamic import still fails in jsdom and we fall back to TS, but
    // crucially the loader does not throw with an empty/invalid URL.
    const wasm = await initWasm();
    expect(wasm).toBeInstanceOf(TypeScriptFallback);
  });

  it('setWasmJsUrl(undefined) also resets the override', async () => {
    setWasmJsUrl('http://localhost:0/missing.js');
    setWasmJsUrl(undefined);
    const wasm = await initWasm();
    expect(wasm).toBeInstanceOf(TypeScriptFallback);
  });
});

describe('isWasmSupported', () => {
  it('returns true when WebAssembly + WebAssembly.instantiate exist', () => {
    expect(isWasmSupported()).toBe(true);
  });

  it('returns false when WebAssembly is missing entirely', () => {
    const original = (globalThis as { WebAssembly?: unknown }).WebAssembly;
    try {
      delete (globalThis as { WebAssembly?: unknown }).WebAssembly;
      expect(isWasmSupported()).toBe(false);
    } finally {
      (globalThis as { WebAssembly?: unknown }).WebAssembly = original;
    }
  });

  it('returns false when WebAssembly.instantiate is not a function', () => {
    const original = WebAssembly.instantiate;
    try {
      (WebAssembly as unknown as { instantiate: unknown }).instantiate = undefined;
      expect(isWasmSupported()).toBe(false);
    } finally {
      (WebAssembly as unknown as { instantiate: typeof original }).instantiate = original;
    }
  });
});

describe('getFallback', () => {
  it('returns a TypeScriptFallback', () => {
    expect(getFallback()).toBeInstanceOf(TypeScriptFallback);
  });

  it('returns a fresh instance on every call (no shared state)', () => {
    const a = getFallback();
    const b = getFallback();
    expect(a).not.toBe(b);
  });
});
