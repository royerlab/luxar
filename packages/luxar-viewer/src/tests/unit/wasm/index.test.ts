// @vitest-environment jsdom
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
 *
 * The shim URL resolution (`wasmShimCandidateUrls`) and the candidate walk
 * (`importFirstWasmShim`) are likewise covered directly rather than through
 * `initWasm`: `import.meta.env.DEV` is true under vitest, so EVERY in-suite
 * `initWasm()` takes the single-candidate dev-server branch and never reaches
 * the candidate list at all.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  initWasm,
  isWasmSupported,
  getFallback,
  setWasmJsUrl,
  assertRequiredWasmExports,
  wasmShimCandidateUrls,
  importFirstWasmShim,
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

describe('wasmShimCandidateUrls', () => {
  // The WASM artifact always lands in a `wasm/` directory at the OUTPUT ROOT,
  // but the chunk carrying the loader sits at two different depths depending
  // on the build, so a single bundle-relative literal cannot reach it from
  // both. These bases are the real measured layouts, not hypotheticals.
  const SHIM = 'wasm/luxar_wasm.js';

  it('reaches the artifact from the library ENTRY chunk at the output root', () => {
    // dist/lib/luxar-viewer.js — '../wasm/…' escapes dist/lib/ and 404s here,
    // which is exactly the embedder-visible bug (#1649): compiled kernels in
    // the workers, silent TypeScript fallback on the main thread.
    const candidates = wasmShimCandidateUrls('https://cdn.example/pkg/dist/lib/luxar-viewer.js');
    expect(candidates).toContain(`https://cdn.example/pkg/dist/lib/${SHIM}`);
  });

  it('reaches the artifact from a library worker chunk under assets/', () => {
    // dist/lib/assets/data-worker-*.js — one level deeper, so the SAME target
    // is reached by the other candidate. This depth already worked with the
    // single pre-fix '../wasm/…' literal: it pins pre-existing behaviour that
    // the multi-candidate list must not regress, not the bug itself.
    const candidates = wasmShimCandidateUrls(
      'https://cdn.example/pkg/dist/lib/assets/data-worker-abc123.js'
    );
    expect(candidates).toContain(`https://cdn.example/pkg/dist/lib/${SHIM}`);
  });

  it('tries the app-build depth FIRST', () => {
    // dist/assets/index-*.js — the hot path. Pin the ordering, not just
    // membership: '../wasm/…' resolving first is what keeps the app build (and
    // both worker chunks) from paying a failed import on every single load, so
    // a silent reorder must fail here rather than quietly cost a 404.
    const candidates = wasmShimCandidateUrls('https://example.com/app/assets/index-abc.js');
    expect(candidates[0]).toBe(`https://example.com/app/${SHIM}`);
    expect(candidates[1]).toBe(`https://example.com/app/assets/${SHIM}`);
    expect(candidates).toHaveLength(2);
  });
});

describe('importFirstWasmShim', () => {
  // The candidate walk itself, driven by a stub importer. `initWasm` cannot
  // reach it in this environment (see the file docblock), so without these
  // tests a revert to "import candidate 0 and hope" passes every gate.
  const A = 'https://cdn.example/pkg/dist/wasm/luxar_wasm.js';
  const B = 'https://cdn.example/pkg/dist/lib/wasm/luxar_wasm.js';

  /** A minimal wasm-bindgen-shaped namespace: the loader only needs `default`. */
  function shim(): { default: () => Promise<unknown> } {
    return { default: async () => ({}) };
  }

  it('returns the FIRST candidate that loads, without touching the rest', async () => {
    const tried: string[] = [];
    const first = shim();
    const mod = await importFirstWasmShim([A, B], async (url) => {
      tried.push(url);
      return first;
    });
    expect(mod).toBe(first);
    // Short-circuit, in order: the app build and both worker chunks resolve on
    // candidate 0, so a second request on the hot path would be pure waste.
    expect(tried).toEqual([A]);
  });

  it('falls through to the next candidate when the first import rejects', async () => {
    // The library ENTRY chunk (#1649): '../wasm/…' escapes dist/lib/ and 404s.
    const second = shim();
    const tried: string[] = [];
    const mod = await importFirstWasmShim([A, B], async (url) => {
      tried.push(url);
      if (url === A) throw new Error('404');
      return second;
    });
    expect(mod).toBe(second);
    expect(tried).toEqual([A, B]);
  });

  it('falls through when a candidate imports but has no callable default', async () => {
    // A host that answers the 404 with a JS-typed SPA fallback page: the import
    // SUCCEEDS and yields a module with no `default`. Ending the walk there
    // would blow up on `wasmModule.default()` and never try the real path.
    const second = shim();
    const tried: string[] = [];
    const mod = await importFirstWasmShim([A, B], async (url) => {
      tried.push(url);
      return url === A ? { notDefault: 1 } : second;
    });
    expect(mod).toBe(second);
    expect(tried).toEqual([A, B]);
  });

  it('rethrows the ORIGINAL error object when only one candidate was tried', async () => {
    // The override and dev-server branches pass a single URL; their failure log
    // must stay exactly what it has always been, not an AggregateError wrapper.
    const cause = new Error('blocked by CSP');
    await expect(
      importFirstWasmShim([A], async () => {
        throw cause;
      })
    ).rejects.toBe(cause);
  });

  it('aggregates every candidate error, naming each URL in order', async () => {
    // Reporting only the LAST error blames dist/assets/wasm/luxar_wasm.js — a
    // directory that exists in no layout — for a real failure on candidate 0.
    const errA = new Error('served as text/plain');
    const errB = new Error('404');
    const rejected = importFirstWasmShim([A, B], async (url) => {
      throw url === A ? errA : errB;
    });
    await expect(rejected).rejects.toBeInstanceOf(AggregateError);
    const error = (await rejected.catch((e: unknown) => e)) as AggregateError;
    expect(error.message).toContain(A);
    expect(error.message).toContain(B);
    expect(error.errors).toEqual([errA, errB]);
  });

  it('throws a real Error when the candidate list is empty', async () => {
    // `throw lastError` on an empty list would throw `undefined`, which the
    // fallback path logs as an unreadable warning.
    await expect(importFirstWasmShim([], async () => shim())).rejects.toThrow(
      /No WASM shim candidate URL/
    );
  });
});
