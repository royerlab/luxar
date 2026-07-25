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
export type { WasmModule, DepthSorterHandle } from './types';

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

    // Initialize WASM (loads the .wasm binary)
    await wasmModule.default();

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
