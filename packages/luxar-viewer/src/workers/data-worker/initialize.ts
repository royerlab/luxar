/**
 * Worker bootstrap. Loads the compiled WASM module via initWasm()
 * (with TypeScript fallback when compiled WASM is missing) and
 * pre-allocates the pooled visibility scratch buffer.
 *
 * Throws if WASM initialization fails catastrophically — the viewer
 * requires a functioning worker.
 */

import { initWasm } from '../../wasm';
import { log, Modules } from '../../utils/log';
import type { WasmCtx } from './state';

export async function initialize(ctx: WasmCtx): Promise<void> {
  log.info(Modules.WORKER_POOL, 'DataWorker initializing...');

  // Load WASM module (falls back to TypeScript implementation if compiled WASM missing)
  try {
    ctx.wasm = await initWasm();
    log.info(Modules.WORKER_POOL, 'DataWorker WASM module loaded successfully');
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
}
