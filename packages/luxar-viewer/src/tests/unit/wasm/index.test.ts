/**
 * Unit tests for the WASM loader module.
 *
 * `initWasm` is hard to exercise on the WASM-loaded path inside jsdom (no
 * binary, no Vite resolution), so the dynamic-import branch always
 * throws and falls through to the TypeScript fallback. We verify the
 * fallback path lands a working WasmModule, that isWasmSupported
 * detects WebAssembly correctly, and that getFallback returns a fresh
 * TypeScriptFallback every call.
 *
 * `assertRequiredWasmExports` is covered directly here because the artifact
 * it rejects (a stale build missing a newer kernel) cannot be synthesized in
 * jsdom — the stub below stands in for one.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  initWasm,
  isWasmSupported,
  getFallback,
  setWasmJsUrl,
  assertRequiredWasmExports,
} from '../../../wasm';
import { TypeScriptFallback } from '../../../wasm/typescript';

/**
 * A module stub that answers EVERY property with a function, minus the
 * explicitly withheld/overridden ones. Built as a Proxy so the required-export
 * list stays single-sourced in `wasm/index.ts` instead of being copied here.
 */
function stubModule(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return new Proxy(overrides, {
    get: (target, prop) =>
      Object.prototype.hasOwnProperty.call(target, prop)
        ? (target as Record<string | symbol, unknown>)[prop]
        : () => undefined,
  }) as Record<string, unknown>;
}

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

describe('assertRequiredWasmExports', () => {
  it('accepts a module exposing every required kernel', () => {
    expect(() => assertRequiredWasmExports(stubModule())).not.toThrow();
  });

  it('names the missing export AND the remediation when a kernel is absent', () => {
    // `compute_joint_codes` is the export whose absence in a stale
    // gitignored build motivated the guard — pin the message format on it.
    // The build hint is part of the message on purpose: on the direct-import
    // path this throw is the only diagnosis the reader gets.
    expect(() => assertRequiredWasmExports(stubModule({ compute_joint_codes: undefined }))).toThrow(
      'Loaded WASM module is stale: missing required export "compute_joint_codes" — ' +
        'rebuild it with pnpm build:wasm (or make build-wasm)'
    );
  });

  // One negative case PER required export, with the names restated literally.
  // This is a deliberate tripwire against the production list silently
  // shrinking, which would make the check pass on a build it should reject:
  // iterating REQUIRED_WASM_EXPORTS here instead would be tautological —
  // deleting a name would delete its own test. Do not "simplify" these into a
  // loop over the exported list.
  it('names mesh_vertex_visibility_mask when only that kernel is missing', () => {
    expect(() =>
      assertRequiredWasmExports(stubModule({ mesh_vertex_visibility_mask: undefined }))
    ).toThrow(/missing required export "mesh_vertex_visibility_mask"/);
  });

  it('names compact_visible_faces when only that kernel is missing', () => {
    expect(() =>
      assertRequiredWasmExports(stubModule({ compact_visible_faces: undefined }))
    ).toThrow(/missing required export "compact_visible_faces"/);
  });

  it('rejects a required export that is present but not callable', () => {
    // wasm-bindgen shims can expose a non-function binding (e.g. a memory
    // view or a renamed constant); `typeof !== 'function'` must catch it,
    // otherwise the cast still hands out an unusable "kernel".
    expect(() => assertRequiredWasmExports(stubModule({ compute_joint_codes: 42 }))).toThrow(
      /missing required export "compute_joint_codes"/
    );
  });

  it('rejects an empty module', () => {
    expect(() => assertRequiredWasmExports({})).toThrow(/missing required export/);
  });

  it('rejects a current shim whose instantiated binary predates a kernel', () => {
    // The MIXED artifact: the JS shim was overwritten but the `.wasm` binary
    // beside it was not. The shim declares a static wrapper per kernel, so its
    // namespace looks complete, and instantiation succeeds regardless
    // (WebAssembly links imports, not exports) — only the exports object the
    // init call hands back shows the gap.
    expect(() =>
      assertRequiredWasmExports(stubModule(), stubModule({ compute_joint_codes: undefined }))
    ).toThrow(/missing required export "compute_joint_codes"/);
  });

  it('checks the namespace alone when no instance exports are supplied', () => {
    // A shim shape that hands back nothing usable must not be read as stale —
    // that would send every embedder down the TypeScript fallback.
    expect(() => assertRequiredWasmExports(stubModule())).not.toThrow();
    expect(() => assertRequiredWasmExports(stubModule(), undefined)).not.toThrow();
    expect(() => assertRequiredWasmExports(stubModule(), null)).not.toThrow();
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
