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

import { TypeScriptFallback } from './typescript';
import type { WasmModule } from './types';
import { log, Modules } from '../utils/log';

// Re-export types for convenience
export type { WasmModule } from './types';

/**
 * Optional override for the WASM JS shim URL.
 *
 * The default resolution (`new URL('../wasm/luxar_wasm.js', import.meta.url)`)
 * works for the standalone Vite app and most consumer bundlers (Vite,
 * Rollup, webpack 5). Bundlers that don't support the `import.meta.url`
 * pattern, or consumers that ship the WASM files from a non-default
 * location, can call {@link setWasmJsUrl} once at startup with an explicit
 * absolute URL.
 *
 * Set via `LuxarAppOptions.wasmPath` from `LuxarApp.init`.
 */
let wasmJsUrlOverride: string | undefined;

/**
 * Exports added after the initial kernel set. A stale gitignored `public/wasm/`
 * build can still import and initialise successfully while missing them,
 * otherwise failing later at first use — where the symptom is an opaque
 * "x is not a function" rather than "your WASM build is old".
 *
 * - `compute_joint_codes` — added with the line cap-suppression kernel.
 * - `mesh_vertex_visibility_mask` / `compact_visible_faces` — added with the
 *   mesh culling kernels.
 *
 * Add a name here when you add a kernel, so a stale build is diagnosed rather
 * than silently half-working — and REMOVE it when you rename or delete that
 * kernel, or every site below reports a freshly built artifact as "stale" and
 * sends the reader to rebuild it in a loop. Only free functions belong here:
 * the names are matched against the raw `.wasm` exports as well as the shim's,
 * and wasm-bindgen mangles anything else (a struct method exports as
 * `<struct>_<method>`).
 *
 * `make build-wasm` is the fix for a genuinely stale build, but the two caller
 * classes react to a failed check very differently:
 * - {@link initWasm} catches it and enters the normal TypeScript fallback path,
 *   which is correct but slower;
 * - the test/benchmark loader (`src/tests/helpers/wasm-artifact.ts`) lets it
 *   throw — there is no fallback there, and failing loudly by name is the whole
 *   point.
 */
const REQUIRED_WASM_EXPORTS = [
  'compute_joint_codes',
  'mesh_vertex_visibility_mask',
  'compact_visible_faces',
] as const satisfies readonly (keyof WasmModule)[];

/**
 * Reject a module that imported and initialised fine but predates one of the
 * `REQUIRED_WASM_EXPORTS` kernels.
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
 * 1. Try to load compiled WASM from wasm/luxar_wasm_bg.wasm (resolved relative to bundle)
 * 2. If fails (not built or browser incompatibility), use TypeScript fallback
 *
 * @returns Promise resolving to WasmModule interface
 */
export async function initWasm(): Promise<WasmModule> {
  try {
    // Compute WASM module URL. The correct base differs by build:
    //
    //   • Production / library build: this module is bundled into
    //     assets/index-*.js and the WASM files live at wasm/ (sibling of
    //     assets/), so the import.meta.url-relative '../wasm/luxar_wasm.js'
    //     resolves correctly. Using a variable prevents Vite from trying to
    //     resolve the path as a source asset at build time.
    //
    //   • Vite dev server: this module is served from /src/wasm/index.ts, so
    //     the same relative path would resolve to /src/wasm/luxar_wasm.js —
    //     but `make build-wasm` writes the compiled module to public/wasm/,
    //     which the dev server serves at /wasm/. Resolve against the origin
    //     in that case so dev picks up the built WASM instead of silently
    //     falling back to the (slower) TypeScript implementation.
    //
    // Embedders whose bundlers don't support `import.meta.url` resolution
    // can override the URL via {@link setWasmJsUrl} (forwarded by
    // LuxarAppOptions.wasmPath); the override takes precedence over both.
    const wasmRelativePath = '../wasm/luxar_wasm.js';
    const isDev = Boolean((import.meta as { env?: { DEV?: boolean } }).env?.DEV);
    let wasmJsUrl: string;
    if (wasmJsUrlOverride) {
      wasmJsUrl = wasmJsUrlOverride;
    } else if (isDev) {
      // public/ is served at the server root in dev regardless of the
      // production-only relative `base`.
      wasmJsUrl = new URL('/wasm/luxar_wasm.js', self.location.origin).href;
    } else {
      wasmJsUrl = new URL(wasmRelativePath, import.meta.url).href;
    }

    // Use Function constructor to avoid TypeScript compile-time module resolution
    // This allows the code to compile even when WASM module doesn't exist yet
    const importWasm = new Function('url', 'return import(url)');
    const wasmModule = await importWasm(wasmJsUrl);

    // Initialize WASM (loads the .wasm binary), then reject mixed/stale dev
    // artifacts before returning them as the WasmModule interface. This check
    // sits INSIDE the swallowing try ON PURPOSE — the inverse of what the
    // test-side loader does: throwing here enters the normal TypeScript
    // fallback path, which is the documented runtime behaviour for a missing or
    // stale build. Do not "fix" it by moving the call out of the try; that
    // would make every embedder crash instead.
    const instanceExports = await wasmModule.default();
    assertRequiredWasmExports(wasmModule as Record<string, unknown>, instanceExports);

    log.info(Modules.WASM, 'Loaded compiled WASM module');

    // Return the WASM module (it already implements WasmModule interface)
    return wasmModule as unknown as WasmModule;
  } catch (error) {
    // WASM not available - use TypeScript fallback
    log.warning(Modules.WASM, 'Failed to load WASM module, using TypeScript fallback', error);
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
