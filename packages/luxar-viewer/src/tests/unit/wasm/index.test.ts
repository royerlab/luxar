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
 *
 * The shim URL resolution (`wasmShimCandidateUrls`) and the candidate walk
 * (`importFirstWasmShim`) are likewise covered directly rather than through
 * `initWasm`: the walk's interesting outcomes (a hit, the callable-`default`
 * shape rule, the error attribution) all need an importer that can actually
 * resolve a module, which `initWasm`'s `new Function` import never is here —
 * every candidate it tries rejects, so a regression to "candidate 0 only" would
 * still look green.
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
  instantiateWasmShim,
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
      // Catches a regression to `wasmJsUrls = [wasmShimCandidateUrls(…)[0]]`:
      // candidate 0 alone satisfies the assertion above, but the whole point of
      // the DOM-less fall-through is that it reaches the LIST. Measured here, the
      // two candidates are `…/src/wasm/luxar_wasm.js` and
      // `…/src/wasm/wasm/luxar_wasm.js`, so the second has a distinct spelling.
      expect(message).toMatch(/src\/wasm\/wasm\/luxar_wasm\.js/);
      // Pre-fix the swallowed error was exactly `ReferenceError: self is not
      // defined`; assert on the error object itself, not on stringified text.
      expect(error).toBeInstanceOf(Error);
      // Cardinality pin, the error-type half: `importFirstWasmShim` rethrows a
      // bare single error only when exactly ONE URL was tried, so an
      // AggregateError proves >1 candidate was walked. Measured: AggregateError
      // carrying 2 sub-errors.
      expect(error).toBeInstanceOf(AggregateError);
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
      // Cardinality pin the other way: this branch is documented as a SINGLE
      // candidate ("the bundle-relative ones would only add guaranteed 404s"),
      // and the `toContain` above passes just as well if a regression APPENDED
      // them. The bundle-relative candidates resolve under `/src/wasm/` here, so
      // their absence is what pins the list at one entry.
      expect(message).not.toContain('/src/wasm/');
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
    // `TypeError: Invalid URL` before `wasmJsUrls` is ever assigned. That is the
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

describe('initWasm fallback warning', () => {
  it('doubles % in the reported URLs so the swallowed error is not eaten as a substitution', async () => {
    // The warning's message is `console.warn`'s FIRST argument, i.e. a format
    // string, and a percent-escape can genuinely reach it: `new URL()` only
    // uppercases escapes it ADDS, so one already present in the base survives
    // into the resolved candidate. An unescaped `%d` there consumes the trailing
    // `error` argument — `console.warn('a %d0', err)` prints `a NaN0` and drops
    // `err` entirely, i.e. the exception vanishes from the one diagnostic #1642
    // added it to. Doubling every `%` neutralises that; console collapses `%%`
    // back to a single `%` whenever an extra argument is present, and `error`
    // always is.
    const overrideUrl = 'http://localhost:0/a%d0%9f/luxar_wasm.js';
    setWasmJsUrl(overrideUrl);
    // `console.log` is spied purely to keep the two remediation `log.info` lines
    // out of the reporter. Both spies are restored inline before the assertions
    // AND in the `finally`; `mockRestore` is idempotent.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const info = vi.spyOn(console, 'log').mockImplementation(() => {});
    try {
      await initWasm();
      const failures = warn.mock.calls.filter(([m]) =>
        String(m).includes('Failed to load WASM module')
      );
      warn.mockRestore();
      info.mockRestore();

      expect(failures.length).toBeGreaterThan(0);
      const [message, error] = failures[0] as [string, unknown];
      expect(message).toContain(overrideUrl.replace(/%/g, '%%'));
      // The raw spelling must NOT appear: it is the one that would be read as a
      // substitution. Asserting its absence is what fails if the escape is
      // dropped — the doubled `toContain` above would then be the only pin, and
      // nothing else in the suite exercises a URL containing a `%` at all.
      expect(message).not.toContain(overrideUrl);
      // The argument the escape exists to protect.
      expect(error).toBeInstanceOf(Error);
    } finally {
      warn.mockRestore();
      info.mockRestore();
      setWasmJsUrl('');
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

  it('collapses to ONE candidate when the chunk sits at the URL root', () => {
    // dist/lib/* copied to a site root — the common unbundled deployment.
    // `new URL()` clamps at the root, so '../wasm/…' and './wasm/…' resolve to
    // the same href. Without the dedupe a genuine failure (CSP, wrong MIME)
    // would report `tried, in order: X, X` and send the reader looking for two
    // different layouts.
    const candidates = wasmShimCandidateUrls('https://cdn.example/luxar-viewer.js');
    expect(candidates).toEqual([`https://cdn.example/${SHIM}`]);
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

  it('falls through when a candidate imports but has no default at all', async () => {
    // A host answering the miss with a 200 carrying an empty body, or a JS stub
    // module standing in for the absent artifact: the import SUCCEEDS and
    // yields a namespace with no `default`. Ending the walk there would blow up
    // on `wasmModule.default()` and never try the real path. (An HTML error
    // page is NOT this case — it fails to parse as a module and is handled by
    // the import-rejection arm above.)
    const second = shim();
    const tried: string[] = [];
    const mod = await importFirstWasmShim([A, B], async (url) => {
      tried.push(url);
      return url === A ? { notDefault: 1 } : second;
    });
    expect(mod).toBe(second);
    expect(tried).toEqual([A, B]);
  });

  it('falls through when a candidate has a default that is not callable', async () => {
    // The "callable" half of the rule, which a `default !== undefined` check
    // would pass: a redirect/stub module can export a non-function default
    // (a string, a config object). It would be accepted as the winner and then
    // throw `default is not a function` on the very next line of initWasm.
    const second = shim();
    const tried: string[] = [];
    const mod = await importFirstWasmShim([A, B], async (url) => {
      tried.push(url);
      return url === A ? { default: 'not the shim' } : second;
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

  it('aggregates a SHAPE rejection together with an import rejection', async () => {
    // The mixed walk is what makes the shape rejection worth recording: with
    // only candidate 1 having pushed an error, the walk would fall to the
    // single-error branch and rethrow candidate 1's 404 bare, blaming the one
    // URL that was never the problem. Every failed candidate must appear.
    const errB = new Error('404');
    const rejected = importFirstWasmShim([A, B], async (url) => {
      if (url === A) return { notDefault: 1 };
      throw errB;
    });
    const error = (await rejected.catch((e: unknown) => e)) as AggregateError;
    expect(error).toBeInstanceOf(AggregateError);
    expect(error.errors).toHaveLength(2);
    expect((error.errors[0] as Error).message).toContain(A);
    expect((error.errors[0] as Error).message).toMatch(/not a wasm-bindgen shim/);
    expect(error.errors[1]).toBe(errB);
    expect(error.message).toContain(`${A}, ${B}`);
  });

  it('throws a real Error when the candidate list is empty', async () => {
    // `throw errors[0]` on an empty list would throw `undefined`, which the
    // fallback path logs as an unreadable warning. Assert on the VALUE, not via
    // `rejects.toThrow(/…/)` — that matcher passes on an `undefined` rejection,
    // i.e. it is vacuous against the exact regression this pins.
    const err = await importFirstWasmShim([], async () => shim()).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(Error);
    expect((err as Error).message).toMatch(/No WASM shim candidate URL/);
  });
});

describe('instantiateWasmShim', () => {
  /** A stand-in for the compiled module; only identity is asserted. */
  const precompiled = { __fake: 'module' } as unknown as WebAssembly.Module;

  // Exercised directly rather than through `initWasm`, which reaches the shim
  // via a `new Function('url','return import(url)')` indirection that vitest's
  // module runner never services — every candidate there rejects, so this
  // success path is unreachable from the outside.

  it('hands wasm-bindgen the precompiled module in its object form', async () => {
    const init = vi.fn(async () => ({ memory: {} }));
    const shim = { default: init } as unknown as Parameters<typeof instantiateWasmShim>[0];

    await instantiateWasmShim(shim, precompiled);

    // `{ module_or_path }` is the shape `__wbg_init` destructures, and passing
    // an existing Module makes `__wbg_load` skip compilation entirely.
    expect(init).toHaveBeenCalledTimes(1);
    expect(init).toHaveBeenCalledWith({ module_or_path: precompiled });
  });

  it('calls default() bare when there is no precompiled module', async () => {
    const init = vi.fn(async () => ({ memory: {} }));
    const shim = { default: init } as unknown as Parameters<typeof instantiateWasmShim>[0];

    await instantiateWasmShim(shim);

    // Byte-identical to the pre-change behaviour: no argument at all.
    expect(init).toHaveBeenCalledWith();
  });

  it('retries bare when the shim rejects the object form', async () => {
    // A relocated or older shim that cannot take a module must degrade to
    // self-initialization rather than failing the worker.
    const init = vi.fn(async (arg?: unknown) => {
      if (arg !== undefined) throw new TypeError('unsupported argument');
      return { memory: {} };
    });
    const shim = { default: init } as unknown as Parameters<typeof instantiateWasmShim>[0];

    await expect(instantiateWasmShim(shim, precompiled)).resolves.toEqual({ memory: {} });
    expect(init).toHaveBeenCalledTimes(2);
    expect(init).toHaveBeenLastCalledWith();
  });
});
