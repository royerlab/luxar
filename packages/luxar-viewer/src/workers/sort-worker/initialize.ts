/**
 * SortWorker bootstrap — mirrors the data-worker's `initialize` exactly:
 * honor an embedder-relocated WASM shim URL inside THIS worker's module
 * scope (the main-thread `setWasmJsUrl` override does not cross the
 * worker boundary), then load compiled WASM with automatic TypeScript
 * fallback.
 */

import { initWasm, isWasmFallback, setWasmJsUrl } from '../../wasm';
import { log, Modules } from '../../utils/log';
import type { SortWorkerCtx } from './state';

/** Outcome of the worker's one-time init, returned to the main thread. */
export interface SortWorkerInitResult {
  /** True when running the TypeScript fallback (compiled WASM not loaded). */
  wasmFallback: boolean;
}

export async function initialize(
  ctx: SortWorkerCtx,
  wasmPath?: string
): Promise<SortWorkerInitResult> {
  log.info(Modules.WORKER_POOL, 'SortWorker initializing...');

  // Must run before initWasm(); empty string resets to default resolution.
  if (wasmPath) setWasmJsUrl(wasmPath);

  try {
    ctx.wasm = await initWasm();
  } catch (error) {
    log.error(Modules.WORKER_POOL, 'SortWorker WASM initialization failed', error);
    throw new Error(
      'WASM unavailable. Luxar requires WebAssembly support. ' +
        'Please use a modern browser (Chrome 57+, Firefox 52+, Safari 11+).',
      { cause: error }
    );
  }

  log.info(Modules.WORKER_POOL, 'SortWorker ready');
  return { wasmFallback: isWasmFallback(ctx.wasm) };
}
