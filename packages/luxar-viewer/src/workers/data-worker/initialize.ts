/**
 * Worker bootstrap. Loads the compiled WASM module via initWasm()
 * (with TypeScript fallback when compiled WASM is missing) and
 * pre-allocates the pooled visibility scratch buffer.
 *
 * Throws if WASM initialization fails catastrophically — the viewer
 * requires a functioning worker.
 */

import { initWasm, isWasmFallback, getFallback, setWasmJsUrl } from '../../wasm';
import { log, Modules } from '../../utils/log';
import type { WasmCtx } from './state';

/** Outcome of a worker's one-time init, returned to the pool. */
export interface WorkerInitResult {
  /** True when this worker is running the TypeScript fallback (compiled WASM not loaded). */
  wasmFallback: boolean;
}

export async function initialize(ctx: WasmCtx, wasmPath?: string): Promise<WorkerInitResult> {
  log.info(Modules.WORKER_POOL, 'DataWorker initializing...');

  // Honor an embedder-supplied WASM location INSIDE the worker scope. The
  // main-thread setWasmJsUrl override does not propagate here (separate module
  // instance), so without this a relocated WASM binary would fail to load in
  // the worker and silently fall back to the slower TS path. Must run before
  // initWasm(). Empty string resets to the default resolution.
  if (wasmPath) setWasmJsUrl(wasmPath);

  // Load WASM module (falls back to TypeScript implementation if compiled WASM missing).
  // The active backend (WASM vs TypeScript fallback) is reported once by the
  // pool on the main thread as a single summary line rather than per-worker
  // here, to keep the console quiet.
  try {
    ctx.wasm = await initWasm();
    // The uncapped TypeScript reference serves >16D operations (the WASM kernels
    // cap at MAX_SUPPORTED_DIMS). Reuse the same instance when WASM itself fell
    // back to TS; otherwise keep a dedicated (stateless) fallback alongside WASM.
    ctx.tsFallback = isWasmFallback(ctx.wasm) ? ctx.wasm : getFallback();
  } catch (error) {
    log.error(Modules.WORKER_POOL, 'DataWorker WASM initialization failed', error);
    throw new Error(
      'WASM unavailable. Luxar requires WebAssembly support. ' +
        'Please use a modern browser (Chrome 57+, Firefox 52+, Safari 11+).',
      { cause: error }
    );
  }

  log.info(Modules.WORKER_POOL, 'DataWorker ready');
  return { wasmFallback: isWasmFallback(ctx.wasm) };
}
