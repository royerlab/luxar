/**
 * Worker bootstrap. Loads the compiled WASM module via initWasm()
 * (with TypeScript fallback when compiled WASM is missing) and
 * pre-allocates the pooled visibility scratch buffer.
 *
 * Throws if WASM initialization fails catastrophically — the viewer
 * requires a functioning worker.
 */

import { initWasm, isWasmFallback } from '../../wasm';
import { log, Modules } from '../../utils/log';
import type { WasmCtx } from './state';

/** Outcome of a worker's one-time init, returned to the pool. */
export interface WorkerInitResult {
  /** True when this worker is running the TypeScript fallback (compiled WASM not loaded). */
  wasmFallback: boolean;
}

export async function initialize(ctx: WasmCtx): Promise<WorkerInitResult> {
  log.info(Modules.WORKER_POOL, 'DataWorker initializing...');

  // Load WASM module (falls back to TypeScript implementation if compiled WASM missing).
  // The active backend (WASM vs TypeScript fallback) is reported once by the
  // pool on the main thread as a single summary line rather than per-worker
  // here, to keep the console quiet.
  try {
    ctx.wasm = await initWasm();
  } catch (error) {
    log.error(Modules.WORKER_POOL, 'DataWorker WASM initialization failed', error);
    throw new Error(
      'WASM unavailable. Luxar requires WebAssembly support. ' +
        'Please use a modern browser (Chrome 57+, Firefox 52+, Safari 11+).'
    );
  }

  // Pre-allocate visibility buffer (will grow as needed)
  ctx.visibilityMaskBuffer = new Uint8Array(100000); // 100K elements max

  log.info(Modules.WORKER_POOL, 'DataWorker ready');
  return { wasmFallback: isWasmFallback(ctx.wasm) };
}
