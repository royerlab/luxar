/**
 * Worker-local mutable state shared by every task entry point.
 *
 * - `wasm` holds the compiled WASM module (or the TypeScript fallback)
 *   loaded once at `initialize()` and read by every task.
 *
 * Tasks receive the `state` object as an argument; mutating `state.wasm`
 * inside `initialize` is visible to subsequent task calls because they
 * read the same shared object reference.
 */

import type { WasmModule } from '../../wasm';
import { MAX_SUPPORTED_DIMS } from '../../config/constants';

export interface WasmCtx {
  wasm: WasmModule | null;
  /**
   * Uncapped TypeScript reference backend, set at init. Used by
   * {@link pickBackend} to serve `ndim > MAX_SUPPORTED_DIMS` operations, which
   * the compiled WASM kernels cannot handle (fixed-size `[_; 16]` arrays). When
   * compiled WASM failed to load, this is the same instance as `wasm`. Optional
   * so test harnesses that only exercise the `ndim <= 16` WASM path need not set it.
   */
  tsFallback?: WasmModule | null;
}

export const state: WasmCtx = {
  wasm: null,
  tsFallback: null,
};

/**
 * Single source of truth for the "task called before initialize()"
 * error. The header comment in data-worker.ts noted that an extracted
 * narrowing helper would lose TS narrowing if `wasmModule` were a
 * module-let. Reading via `requireWasm(ctx)` and returning the
 * unwrapped value sidesteps that: each task body uses a `const wasm`
 * that's narrowed for the rest of the function.
 */
export const NOT_INITIALIZED_MSG = '[DataWorker] Not initialized - call initialize() first';

export function requireWasm(ctx: WasmCtx): WasmModule {
  if (!ctx.wasm) throw new Error(NOT_INITIALIZED_MSG);
  return ctx.wasm;
}

/**
 * Select the backend for a dimension-sensitive operation.
 *
 * The compiled WASM kernels use fixed-size `[_; MAX_SUPPORTED_DIMS]` arrays and
 * panic (the crate is `panic = "abort"`) for `ndim > MAX_SUPPORTED_DIMS`. The
 * TypeScript reference implementation is uncapped and handles arbitrary ndim, so
 * for `ndim > MAX_SUPPORTED_DIMS` we transparently route the operation to it.
 * This is the ">16D uses the TypeScript fallback automatically (slower but
 * works)" contract — higher-dimensional datasets are SUPPORTED, not rejected.
 *
 * For `ndim <= MAX_SUPPORTED_DIMS` the active backend (compiled WASM when
 * available, otherwise the same TS fallback) is used.
 */
export function pickBackend(ctx: WasmCtx, ndim: number): WasmModule {
  const wasm = requireWasm(ctx);
  if (ndim > MAX_SUPPORTED_DIMS) {
    if (!ctx.tsFallback) throw new Error(NOT_INITIALIZED_MSG);
    return ctx.tsFallback;
  }
  return wasm;
}
