/**
 * Worker-local mutable state shared by every task entry point.
 *
 * - `wasm` holds the compiled WASM module (or the TypeScript fallback)
 *   loaded once at `initialize()` and read by every task.
 * - `visibilityMaskBuffer` is a pooled scratch buffer reused across
 *   visibility calls so the worker doesn't allocate per task. The three
 *   `computeNDVisibility*` tasks grow it as needed.
 *
 * Tasks receive the `state` object as an argument; mutating `state.wasm`
 * inside `initialize` is visible to subsequent task calls because they
 * read the same shared object reference.
 */

import type { WasmModule } from '../../wasm';

export interface WasmCtx {
  wasm: WasmModule | null;
  visibilityMaskBuffer: Uint8Array | null;
}

export const state: WasmCtx = {
  wasm: null,
  visibilityMaskBuffer: null,
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
