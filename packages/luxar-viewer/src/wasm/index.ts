/**
 * WASM module loader and main exports.
 *
 * This module handles loading the compiled WASM module with automatic
 * fallback to TypeScript implementation when WASM is unavailable.
 *
 * ## Usage
 *
 * ```typescript
 * import { initWasm } from '../wasm';
 *
 * const wasm = await initWasm();
 * const count = wasm.clip_segments_batch(...);
 * ```
 *
 * ## Architecture
 *
 * The WASM module can be used from:
 * - Main thread (for quick synchronous operations)
 * - Web Workers (for heavy background processing)
 *
 * The same interface is provided regardless of where it's used.
 */

import { REQUIRED_WASM_EXPORTS } from './required-exports';
import { TypeScriptFallback } from './typescript';
import type { WasmModule } from './types';
import { log, Modules } from '../utils/log';

// Re-export types for convenience
export type { WasmModule } from './types';

/**
 * Optional override for the WASM JS shim URL.
 *
 * The default resolution walks a short list of `import.meta.url`-relative
 * candidates (see {@link wasmShimCandidateUrls}) that covers both depths a
 * chunk carrying this module can sit at, and works for the standalone Vite
 * app and most consumer bundlers (Vite, Rollup, webpack 5). Bundlers that
 * don't support the `import.meta.url` pattern, or consumers that ship the
 * WASM files from a non-default location, can call {@link setWasmJsUrl} once
 * at startup with an explicit absolute URL.
 *
 * Set via `LuxarAppOptions.wasmPath` from `LuxarApp.init`.
 */
let wasmJsUrlOverride: string | undefined;

/**
 * Bundle-relative specifiers for the wasm-bindgen JS shim, in the order they
 * are tried. One literal cannot serve every build, because the WASM artifact
 * always lands in a `wasm/` directory at the OUTPUT ROOT while the chunk that
 * carries this module sits at one of two different depths:
 *
 *   • `dist/assets/index-*.js` (app build) and `dist/lib/assets/*-worker-*.js`
 *     (library build's worker chunks) are one level down, so `wasm/` is a
 *     sibling of `assets/` and `../wasm/…` is right;
 *   • `dist/lib/luxar-viewer.js` (the library build's ENTRY chunk) sits AT the
 *     output root, where `../wasm/…` escapes `dist/lib/` entirely and 404s —
 *     `./wasm/…` is right.
 *
 * `../` is listed first because every hot path (the app build, and the library
 * build's data/sort workers) resolves on it, so the common case still costs
 * exactly one request; only the library entry chunk pays a failed import
 * before finding its shim.
 */
const WASM_SHIM_RELATIVE_SPECIFIERS = ['../wasm/luxar_wasm.js', './wasm/luxar_wasm.js'] as const;

/**
 * Resolve the bundle-relative WASM shim candidates against a base URL,
 * yielding absolute hrefs in try order.
 *
 * `initWasm` calls this with `import.meta.url` — i.e. the URL of the chunk
 * this module was bundled into — and imports the candidates in order until
 * one loads. The order is part of the contract: the first entry is the one
 * that resolves for the app build and for the library build's worker chunks,
 * so reordering the list would make every ordinary load spend a 404 before
 * reaching the artifact.
 *
 * The result is deduplicated, so it is not always two entries: `new URL()`
 * clamps at the URL root, so a chunk served AT the root (`dist/lib/*` copied
 * to a site root, the common unbundled deployment) resolves both specifiers
 * to the same href. Keeping the duplicate would cost nothing on the success
 * path but would make a genuine failure report `tried, in order: X, X`.
 *
 * @param baseUrl Absolute URL of the chunk to resolve against.
 * @returns Absolute candidate hrefs, most-likely first, without duplicates.
 */
export function wasmShimCandidateUrls(baseUrl: string): string[] {
  return [
    ...new Set(WASM_SHIM_RELATIVE_SPECIFIERS.map((specifier) => new URL(specifier, baseUrl).href)),
  ];
}

/**
 * The wasm-bindgen JS shim's namespace, as far as the loader cares: a
 * `default()` that instantiates the binary, plus the per-kernel wrappers
 * {@link assertRequiredWasmExports} inspects by name. Exported because it is
 * {@link importFirstWasmShim}'s return type — a private alias in a public
 * signature trips the TypeDoc warning ratchet.
 */
export type WasmShimModule = {
  /**
   * wasm-bindgen's `__wbg_init`. Takes no argument in the common case; a
   * `{ module_or_path }` object (or a bare `WebAssembly.Module`) hands it an
   * already-compiled binary to instantiate — see {@link instantiateWasmShim}.
   */
  default: (moduleOrPath?: unknown) => Promise<unknown>;
} & Record<string, unknown>;

/**
 * Import the first URL that yields something shaped like the wasm-bindgen
 * shim, trying the candidates in order.
 *
 * Split out of {@link initWasm} and given an injectable importer because
 * `initWasm` cannot exercise a SUCCESSFUL walk under vitest: it reaches the shim
 * through a `new Function('url', 'return import(url)')` indirection that vitest's
 * VM module runner does not service, so every candidate it tries rejects and a
 * regression back to "use candidate 0 only" would still end in the TypeScript
 * fallback and pass every gate.
 *
 * Two rules, both load-bearing:
 * - A candidate must expose a callable `default` to count as a hit. A host that
 *   answers the entry chunk's miss with a 200 carrying an empty body, or a
 *   JavaScript stub/redirect module standing in for the absent artifact, yields
 *   a namespace whose `default` is missing or is not a function; ending the loop
 *   there would blow up on `wasmModule.default()` before the real candidate is
 *   ever tried — #1649's exact symptom, surviving on that host class. (An HTML
 *   error page is a different case and needs no help here: HTML does not parse
 *   as an ES module, so it REJECTS the import and the `catch` arm above already
 *   moves on.) The check only reads a property, so nothing is instantiated and
 *   the "commit to the winner" rule below is untouched.
 * - Only the IMPORT is retried. Once a candidate wins, the caller runs
 *   `default()` and the staleness check against that module alone: re-running
 *   them elsewhere could instantiate the binary twice, and would hide a
 *   genuinely stale artifact behind the next candidate's 404.
 *
 * @param urls Candidate hrefs in try order (see {@link wasmShimCandidateUrls}).
 * @param importModule Dynamic-import indirection. {@link initWasm} passes a
 *   `new Function`-built importer so neither TypeScript nor Vite resolves the
 *   specifier at build time.
 * @returns The winning candidate's module namespace.
 * @throws The single candidate's own error when only one URL was tried (so the
 *   override and dev-server paths log exactly what they always did), otherwise
 *   an `AggregateError` naming every URL in order — attributing a real failure
 *   (shim served as `text/plain`, blocked by CSP, corrupt) to the LAST
 *   candidate would point at a directory that exists in no layout.
 */
export async function importFirstWasmShim(
  urls: readonly string[],
  importModule: (url: string) => Promise<unknown>
): Promise<WasmShimModule> {
  const errors: unknown[] = [];
  for (const url of urls) {
    let candidate: unknown;
    try {
      candidate = await importModule(url);
    } catch (importError) {
      errors.push(importError);
      continue;
    }
    if (typeof (candidate as { default?: unknown } | undefined)?.default === 'function') {
      return candidate as WasmShimModule;
    }
    errors.push(
      new Error(`Module at ${url} is not a wasm-bindgen shim: no callable default export`)
    );
  }
  if (errors.length === 0) {
    // An empty candidate list is a caller bug, not a load failure; rethrowing
    // `undefined` here would surface as an unreadable fallback warning.
    throw new Error('No WASM shim candidate URL was resolved');
  }
  if (errors.length === 1) {
    throw errors[0];
  }
  throw new AggregateError(
    errors,
    `Failed to load the WASM shim from any candidate URL (tried, in order: ${urls.join(', ')})`
  );
}

/**
 * Reject a module that imported and initialised fine but predates one of the
 * `REQUIRED_WASM_EXPORTS` kernels. The list itself lives in
 * `./required-exports.ts` because the vitest global setup shares it; see the
 * comment there for why it is a separate module.
 *
 * `make build-wasm` is the fix for a genuinely stale build, but the two caller
 * classes react to a failed check very differently:
 * - {@link initWasm} catches it and enters the normal TypeScript fallback path,
 *   which is correct but slower;
 * - the test/benchmark loader (`src/tests/helpers/wasm-artifact.ts`) lets it
 *   throw — there is no fallback there, and failing loudly by name is the whole
 *   point.
 *
 * {@link initWasm} calls this before handing the module out, so the runtime
 * path is covered. Tests, benchmarks and tools that want the compiled kernels
 * load the built artifact themselves — `await import('.../public/wasm/
 * luxar_wasm.js')` + `initSync()`, bypassing `initWasm` — and get this check
 * from the ONE loader they all share, `src/tests/helpers/wasm-artifact.ts`.
 * That loader calls it before the `as unknown as WasmModule` cast (the double
 * cast promises the entire interface while the artifact may be missing half of
 * it, so a stale gitignored build otherwise surfaces as an opaque "x is not a
 * function" from whichever kernel assertion runs first) and OUTSIDE the catch
 * that downgrades a load failure to a skip (inside it, the named message would
 * be swallowed too). `direct-import-guard.test.ts` pins exactly one rule to
 * keep that true: no source outside that helper may load the artifact itself —
 * it fails on any other `.ts` under `src/`, `tools/` or `scripts/` that both
 * mentions `luxar_wasm.js` and calls `initSync(`.
 *
 * {@link initWasm} is the one deliberate exception to the placement rule: it
 * calls this INSIDE the try whose catch returns a {@link TypeScriptFallback},
 * because entering the documented fallback is the right response to a stale
 * build on the runtime path. The guard test exempts this module for that reason.
 *
 * @param module The wasm-bindgen JS shim's namespace.
 * @param instanceExports What `initSync()` / the shim's `default()` returned —
 *   the instantiated `.wasm` exports. Checking the namespace alone is not
 *   enough for a MIXED artifact (only one of `luxar_wasm.js` /
 *   `luxar_wasm_bg.wasm` overwritten): the shim declares a static wrapper per
 *   kernel, so its namespace reads as complete while the binary behind it
 *   predates the kernel, and instantiation still succeeds because WebAssembly
 *   only links imports. Omit it (or pass a non-object) to check the namespace
 *   alone.
 */
export function assertRequiredWasmExports(
  module: Record<string, unknown>,
  instanceExports?: unknown
): void {
  const compiled =
    typeof instanceExports === 'object' && instanceExports !== null
      ? (instanceExports as Record<string, unknown>)
      : undefined;
  for (const name of REQUIRED_WASM_EXPORTS) {
    if (typeof module[name] !== 'function' || (compiled && typeof compiled[name] !== 'function')) {
      // The remediation rides IN the message: on the hard direct-import path
      // this throw is all the reader gets (no logger, no fallback warning), and
      // a vitest failure line that names the missing kernel without saying how
      // to fix it is only half the diagnosis #1412 asked for.
      throw new Error(
        `Loaded WASM module is stale: missing required export "${name}" — ` +
          'rebuild it with pnpm build:wasm (or make build-wasm)'
      );
    }
  }
}

/**
 * Override the URL used to load the WASM JS shim. Pass an absolute URL
 * (e.g. `new URL('/static/luxar/wasm/luxar_wasm.js', location.origin).href`).
 * Call before {@link initWasm}.
 *
 * Pass `''` or `undefined` to RESET the override back to the default
 * `import.meta.url`-relative resolution. Treating `''` as a valid value
 * would otherwise survive a `??` check in {@link initWasm} and yield a
 * bogus empty wasm URL.
 */
export function setWasmJsUrl(url: string | undefined): void {
  wasmJsUrlOverride = url === '' ? undefined : url;
}

/**
 * Initialize WASM module.
 *
 * Attempts to load the compiled WASM module, falls back to TypeScript
 * implementation if WASM is unavailable.
 *
 * ## Load Order
 *
 * 1. Resolve the JS shim URL candidates, first match wins: an explicit
 *    {@link setWasmJsUrl} override; else, in a dev build with a `location`
 *    global, the single candidate `/wasm/luxar_wasm.js` on the dev-server
 *    origin; else the bundle-relative candidates from
 *    {@link wasmShimCandidateUrls}, which cover both depths a chunk carrying
 *    this module can sit at.
 * 2. Import the candidates in order until one loads as a shim
 *    ({@link importFirstWasmShim}), then load its `luxar_wasm_bg.wasm` binary.
 * 3. If all of that fails (not built, stale build, wrong layout, browser
 *    incompatibility, or a URL that could not be resolved at all), use the
 *    TypeScript fallback.
 *
 * @returns Promise resolving to WasmModule interface
 */
/**
 * The JS-shim candidate hrefs {@link initWasm} will try, in order.
 *
 * Extracted so the main-thread precompiler (`wasm/shared-module.ts`) resolves
 * exactly the same URLs, and honours the same {@link setWasmJsUrl} override,
 * instead of duplicating this branch and drifting from it.
 *
 * See {@link initWasm}'s own comment for why each build lands where it does.
 */
export function resolveWasmShimUrls(): string[] {
  const isDev = Boolean((import.meta as { env?: { DEV?: boolean } }).env?.DEV);
  if (wasmJsUrlOverride) {
    return [wasmJsUrlOverride];
  }
  if (isDev && typeof location !== 'undefined' && location?.origin) {
    // public/ is served at the server root in dev regardless of the
    // production-only relative `base`.
    return [new URL('/wasm/luxar_wasm.js', location.origin).href];
  }
  return wasmShimCandidateUrls(import.meta.url);
}

/**
 * Run a loaded shim's `default()` — handing it an already-compiled module when
 * one is available.
 *
 * wasm-bindgen's generated `__wbg_init` accepts either a raw value or
 * `{ module_or_path }`, and its `__wbg_load` skips `new WebAssembly.Module(...)`
 * when the value already IS a `WebAssembly.Module`. So passing one means the
 * binary is neither re-fetched nor re-compiled — which is the whole point:
 * 15 workers each compiling their own copy came ready ~160 ms apart, 2.2 s from
 * first to last.
 *
 * Exported (and given an injectable shim) for the same reason
 * `importFirstWasmShim` is: `initWasm` reaches the real shim through a
 * `new Function('url', 'return import(url)')` indirection that vitest's module
 * runner does not service, so this path cannot be exercised through
 * `initWasm` itself.
 *
 * Degrades: a relocated or older shim that rejects the object form is retried
 * once with no argument. `__wbg_init` only assigns its module-scoped `wasm`
 * inside `__wbg_finalize_init`, so a throw before that leaves the retry clean.
 */
export async function instantiateWasmShim(
  shim: WasmShimModule,
  precompiled?: WebAssembly.Module
): Promise<unknown> {
  if (!precompiled) return shim.default();
  try {
    return await shim.default({ module_or_path: precompiled });
  } catch (error) {
    log.warning(
      Modules.WASM,
      'Precompiled WASM module rejected by the shim; falling back to self-initialization',
      error
    );
    return shim.default();
  }
}

export async function initWasm(precompiled?: WebAssembly.Module): Promise<WasmModule> {
  // Declared outside the try so the catch can report the candidate list this
  // load RESOLVED. It is deliberately NOT phrased as "the candidates it tried":
  // the same catch also covers a post-import failure — `default()` throwing, or
  // `assertRequiredWasmExports` rejecting a stale artifact — and in those cases
  // one of the listed candidates DID load. What the list is always good for is
  // telling an absent artifact apart from a loader that never computed a URL at
  // all, which is exactly how #1642 (a bare `self` dereference in a DOM-less
  // host) stayed invisible.
  let wasmJsUrls: string[] | undefined;
  try {
    // Compute the WASM shim URL(s). The correct base differs by build:
    //
    //   • Production app / library build: this module is bundled into a chunk
    //     and the WASM files live in wasm/ at the output root — but the chunk
    //     itself sits at one of two depths (assets/*.js vs the library entry
    //     chunk at the root), so there is no single relative literal that
    //     works for both. Try the candidates in order instead; see
    //     WASM_SHIM_RELATIVE_SPECIFIERS for which build lands on which.
    //     Resolving through a variable also prevents Vite from trying to
    //     resolve the path as a source asset at build time.
    //
    //   • Vite dev server, WHEN a `location` global exists: this module is
    //     served from /src/wasm/index.ts, so any bundle-relative path would
    //     resolve under /src/wasm/ — but `make build-wasm` writes the compiled
    //     module to public/wasm/, which the dev server serves at /wasm/.
    //     Resolve against the origin in that case (a SINGLE candidate: dev
    //     serves public/ at the root, so the bundle-relative ones would only
    //     add guaranteed 404s) so dev picks up the built WASM instead of
    //     silently falling back to the (slower) TypeScript implementation.
    //     `location` (bare, not `self.location`) is present in both window and
    //     dedicated-worker scopes, and the read is reached only in a dev build
    //     AND only through `typeof`, so a production bundle in a non-browser
    //     host never touches `location` at all. A dev build in a host that
    //     merely LACKS the global falls through to the bundle-relative
    //     candidates below — in Node/SSR there is no dev-server origin to
    //     resolve against, so that is the only resolution that could mean
    //     anything there (#1642).
    //
    // Embedders whose bundlers don't support `import.meta.url` resolution
    // can override the URL via {@link setWasmJsUrl} (forwarded by
    // LuxarAppOptions.wasmPath); the override takes precedence over both.
    wasmJsUrls = resolveWasmShimUrls();

    // Use Function constructor to avoid TypeScript compile-time module resolution
    // This allows the code to compile even when WASM module doesn't exist yet
    const importWasm = new Function('url', 'return import(url)') as (
      url: string
    ) => Promise<unknown>;
    // Candidate walking, the shim shape check and error attribution all live in
    // importFirstWasmShim so they can be unit-tested with a stub importer —
    // under vitest every import attempted from here rejects, whatever it
    // resolved, so a hit can never be observed on this path.
    const wasmModule = await importFirstWasmShim(wasmJsUrls, importWasm);

    // Initialize WASM (loads the .wasm binary), then reject mixed/stale dev
    // artifacts before returning them as the WasmModule interface. This check
    // sits INSIDE the swallowing try ON PURPOSE — the inverse of what the
    // test-side loader does: throwing here enters the normal TypeScript
    // fallback path, which is the documented runtime behaviour for a missing or
    // stale build. Do not "fix" it by moving the call out of the try; that
    // would make every embedder crash instead.
    const instanceExports = await instantiateWasmShim(wasmModule, precompiled);
    assertRequiredWasmExports(wasmModule as Record<string, unknown>, instanceExports);

    log.info(Modules.WASM, 'Loaded compiled WASM module');

    // Return the WASM module (it already implements WasmModule interface)
    return wasmModule as unknown as WasmModule;
  } catch (error) {
    // WASM not available - use TypeScript fallback.
    //
    // The message states what the list IS (the resolved candidates, in order)
    // rather than claiming each was fetched and failed: only the exhausted-walk
    // case is a per-URL failure, while `default()` throwing and a stale-artifact
    // rejection both happen AFTER one candidate loaded fine.
    //
    // `%` is doubled because this string is console.warn's FIRST argument, i.e. a
    // format string: `new URL()` preserves percent-escapes present in the base,
    // so a path containing `%d`/`%s` would consume `error` as its substitution
    // and drop it from the log entirely. Console collapses `%%` back to a single
    // `%` whenever any extra argument is present, and `error` always is.
    const candidates = wasmJsUrls
      ? `candidates, in order: ${wasmJsUrls.join(', ').replace(/%/g, '%%')}`
      : '<URL resolution failed before the import>';
    log.warning(
      Modules.WASM,
      `Failed to load WASM module (${candidates}), using TypeScript fallback`,
      error
    );
    log.info(Modules.WASM, 'To build WASM module: pnpm build:wasm (or make build-wasm)');
    log.info(
      Modules.WASM,
      'See packages/luxar-viewer/src/wasm/rust/README.md for build instructions'
    );

    return new TypeScriptFallback();
  }
}

/**
 * Check if WASM is available.
 *
 * Useful for feature detection before attempting to load WASM.
 *
 * @returns true if WebAssembly is supported by the browser
 */
export function isWasmSupported(): boolean {
  return typeof WebAssembly !== 'undefined' && typeof WebAssembly.instantiate === 'function';
}

/**
 * Get the TypeScript fallback implementation directly.
 *
 * Useful for testing or when you want to skip WASM loading entirely.
 *
 * @returns TypeScript implementation of WasmModule
 */
export function getFallback(): WasmModule {
  return new TypeScriptFallback();
}

/**
 * Whether the given module is the TypeScript fallback rather than the
 * compiled WASM backend. {@link initWasm} returns a {@link TypeScriptFallback}
 * when the compiled module can't be loaded (e.g. not built in dev), so
 * callers can report the active backend accurately instead of always
 * claiming "WASM loaded".
 */
export function isWasmFallback(module: WasmModule): boolean {
  return module instanceof TypeScriptFallback;
}
