/**
 * Unit tests for the WASM loader module.
 *
 * `initWasm`'s success path cannot be exercised under vitest in ANY
 * environment: the loader reaches the shim through a
 * `new Function('url', 'return import(url)')` indirection, which vitest's VM
 * module runner does not service (`ERR_VM_DYNAMIC_IMPORT_CALLBACK_MISSING`).
 * So the dynamic import always throws and falls through to the TypeScript
 * fallback, and pinning a file to jsdom buys no compiled-kernel coverage —
 * anything that wants the real kernels loads the built artifact through
 * `src/tests/helpers/wasm-artifact.ts` instead. What we verify here is the
 * loader's own logic: that the fallback path lands a working WasmModule, that
 * URL resolution survives a host with no DOM globals (#1642), that
 * isWasmSupported detects WebAssembly correctly, and that getFallback returns a
 * fresh TypeScriptFallback every call.
 *
 * `assertRequiredWasmExports` is covered directly here because the artifact it
 * rejects (a stale build missing a newer kernel) cannot be loaded under vitest
 * either — the stub below stands in for one.
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
 * list stays single-sourced in `wasm/required-exports.ts` instead of being
 * copied here.
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
    // Under vitest the dynamic import always throws, whatever the environment.
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

describe('initWasm URL resolution without a DOM', () => {
  it('resolves through import.meta.url instead of dereferencing a browser global (#1642)', async () => {
    // Precondition. This file runs in vitest's default `node` environment, so
    // there genuinely is no DOM here. Asserting it keeps the test from turning
    // vacuous if the file is ever re-pinned to jsdom: under jsdom both globals
    // exist and the dev-origin branch — not the branch under test — would run.
    expect(typeof self).toBe('undefined');
    expect(typeof location).toBe('undefined');

    // `console.log` is spied purely to keep the two remediation `log.info` lines
    // out of the reporter. Both spies are installed before the `try` and
    // restored inline before the assertions (so a failure reads a live console)
    // AND in the `finally` (so a rejection from `initWasm` cannot leak them);
    // `mockRestore` is idempotent, so running it twice is harmless.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const info = vi.spyOn(console, 'log').mockImplementation(() => {});
    try {
      // The override is module-level state SHARED by every test in this file,
      // and other tests here set one; clear it so the default resolution path
      // runs.
      setWasmJsUrl('');

      const wasm = await initWasm();
      // Filter to the loader's OWN warning rather than indexing by position:
      // any unrelated `console.warn` inside the spy window would otherwise
      // shift the index and silently assert against the wrong message.
      const failures = warn.mock.calls.filter(([m]) =>
        String(m).includes('Failed to load WASM module')
      );
      warn.mockRestore();
      info.mockRestore();

      // Landing in the TypeScript fallback is EXPECTED here and is not what
      // this test is about: vitest's VM module runner cannot service the
      // loader's `new Function('url', 'return import(url)')` import in any
      // environment. What is pinned is that the loader got as far as computing a
      // URL rather than dying on a missing browser global.
      expect(wasm).toBeInstanceOf(TypeScriptFallback);

      expect(failures.length).toBeGreaterThan(0);
      const [message, error] = failures[0] as [string, unknown];
      // These tests are the pin on this message's shape: repo-wide, nothing
      // outside this file and its sibling keys on `Failed to load WASM module`.
      // The substring the E2E specs do depend on is `TypeScript fallback`
      // (`src/tests/e2e/worker-wasm-integration.spec.ts`), which the message
      // still ends with.
      expect(message).toContain('Failed to load WASM module');
      // Resolution completed via `import.meta.url` with no DOM present.
      expect(message).toMatch(/src\/wasm\/luxar_wasm\.js/);
      // Pre-fix the swallowed error was exactly `ReferenceError: self is not
      // defined`; assert on the error object itself, not on stringified text.
      expect(error).toBeInstanceOf(Error);
      expect(error).not.toBeInstanceOf(ReferenceError);
    } finally {
      warn.mockRestore();
      info.mockRestore();
    }
  });
});

describe('initWasm URL resolution on the dev server origin', () => {
  it('resolves /wasm/luxar_wasm.js against location.origin when a location exists', async () => {
    // `import.meta.env.DEV` is true under vitest, so with a `location` in scope
    // this is the dev-origin branch of `initWasm` — the assertion below is not
    // vacuous, and this is the only test that exercises the TRUE side of
    // `isDev && typeof location !== 'undefined' && location?.origin`.
    //
    // The stub must be undone in a `finally`: `unstubGlobals` is NOT enabled in
    // `vitest.config.ts`, so a leaked `location` would silently break the
    // `typeof location === 'undefined'` precondition of the #1642 test above.
    vi.stubGlobal('location', { origin: 'http://dev.test' });
    // `console.log` is spied purely to keep the two remediation `log.info`
    // lines out of the reporter. Both spies are installed before the `try` and
    // restored inline before the assertions AND in the `finally`, so neither a
    // failing assertion nor a rejection can leak one; `mockRestore` is
    // idempotent.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const info = vi.spyOn(console, 'log').mockImplementation(() => {});
    try {
      // The override is module-level state shared by every test in this file.
      setWasmJsUrl('');

      const wasm = await initWasm();
      // Filter to the loader's own warning rather than indexing by position:
      // an unrelated `console.warn` in the spy window would shift the index.
      const failures = warn.mock.calls.filter(([m]) =>
        String(m).includes('Failed to load WASM module')
      );
      warn.mockRestore();
      info.mockRestore();

      // Landing in the TypeScript fallback is EXPECTED and is not what this test
      // is about — vitest's VM module runner cannot service the loader's
      // `new Function('url', 'return import(url)')` import, whatever the URL.
      expect(wasm).toBeInstanceOf(TypeScriptFallback);

      expect(failures.length).toBeGreaterThan(0);
      const [message] = failures[0] as [string, unknown];
      // The whole point: the leading-slash, origin-relative resolution. A
      // relative `wasm/luxar_wasm.js` would resolve to /src/wasm/…, which
      // `scripts/build-wasm.sh` never writes, 404 in dev, and silently drop the
      // dev server onto the TypeScript backend with the suite still green.
      expect(message).toContain('http://dev.test/wasm/luxar_wasm.js');
    } finally {
      warn.mockRestore();
      info.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it('reports "resolution failed before the import" when the origin is not a usable URL base', async () => {
    // Where `'null'` comes from in practice: an opaque origin serialises to the
    // STRING "null" — a dev build opened from `file://`, inside a sandboxed
    // iframe, or in a `data:` document. It is truthy, so the guard passes, and
    // `new URL('/wasm/luxar_wasm.js', 'null')` then throws
    // `TypeError: Invalid URL` before `wasmJsUrl` is ever assigned. That is the
    // one path that reaches the placeholder arm of the catch's message, and the
    // placeholder is what a reader keys on to tell "the artifact isn't there"
    // apart from "the loader never computed a URL" — the exact confusion that
    // kept #1642 invisible.
    //
    // This is deliberately NOT a claim that falling back is the ideal outcome
    // for a `file://` dev build; only that the diagnostic says WHICH of the two
    // failure classes happened.
    vi.stubGlobal('location', { origin: 'null' });
    // `console.log` is spied purely to keep the two remediation `log.info`
    // lines out of the reporter. Both spies are installed before the `try` and
    // restored inline before the assertions AND in the `finally`, so neither a
    // failing assertion nor a rejection can leak one; `mockRestore` is
    // idempotent.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const info = vi.spyOn(console, 'log').mockImplementation(() => {});
    try {
      // The override is module-level state shared by every test in this file.
      setWasmJsUrl('');

      const wasm = await initWasm();
      // Filter to the loader's own warning rather than indexing by position:
      // an unrelated `console.warn` in the spy window would shift the index.
      const failures = warn.mock.calls.filter(([m]) =>
        String(m).includes('Failed to load WASM module')
      );
      warn.mockRestore();
      info.mockRestore();

      expect(wasm).toBeInstanceOf(TypeScriptFallback);

      expect(failures.length).toBeGreaterThan(0);
      const [message] = failures[0] as [string, unknown];
      expect(message).toContain('Failed to load WASM module');
      expect(message).toContain('<URL resolution failed before the import>');
    } finally {
      warn.mockRestore();
      info.mockRestore();
      vi.unstubAllGlobals();
    }
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
    // dynamic import still fails under vitest and we fall back to TS, but
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
