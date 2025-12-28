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
 * const count = wasm.query_chunks_for_view(...);
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

// Re-export types for convenience
export type { WasmModule } from './types';

/**
 * Initialize WASM module.
 *
 * Attempts to load the compiled WASM module, falls back to TypeScript
 * implementation if WASM is unavailable.
 *
 * ## Load Order
 *
 * 1. Try to load compiled WASM from /wasm/luxar_wasm_bg.wasm
 * 2. If fails (not built or browser incompatibility), use TypeScript fallback
 *
 * @returns Promise resolving to WasmModule interface
 */
export async function initWasm(): Promise<WasmModule> {
  try {
    // Use Function constructor to avoid TypeScript compile-time module resolution
    // This allows the code to compile even when WASM module doesn't exist yet
    const importWasm = new Function('return import("/wasm/luxar_wasm.js")');
    const wasmModule = await importWasm();

    // Initialize WASM (loads the .wasm binary)
    await wasmModule.default();

    console.log('[WASM] Loaded compiled WASM module');

    // Return the WASM module (it already implements WasmModule interface)
    return wasmModule as unknown as WasmModule;
  } catch (error) {
    // WASM not available - use TypeScript fallback
    console.warn('[WASM] Failed to load WASM module, using TypeScript fallback:', error);
    console.log('[WASM] To build WASM module: pnpm build:wasm (or make wasm-build)');
    console.log('[WASM] See packages/luxar-viewer/src/wasm/rust/README.md for build instructions');

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
