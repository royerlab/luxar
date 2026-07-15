import { setWasmJsUrl } from '../../../wasm';
import { setDataWorkerUrl, setDataWorkerWasmPath } from '../../../workers/worker-pool';
import { setSortWorkerWasmPath } from '../../../rendering/depth-sort-coordinator';
import type { LuxarAppOptions } from '../options';

/**
 * Forward asset-URL overrides to the WASM and worker modules. Skipped
 * when the option is undefined so the modules use their default
 * `import.meta.url`-based resolution.
 *
 * `wasmPath` is applied TWICE on purpose: once on the main thread
 * ({@link setWasmJsUrl}, for the main-thread WASM consumers) and once for the
 * worker pool ({@link setDataWorkerWasmPath}, forwarded into each worker's
 * `initialize()` RPC). The worker runs in a separate module instance, so the
 * main-thread override does not reach it — without the second call a relocated
 * WASM binary would silently fall back to the slow TS path in the worker hot
 * path.
 *
 * NOTE: the override is module-level and sticks across init() calls —
 * once set, a subsequent init() without the option does not reset to
 * the default. In practice we only support one LuxarApp per page in
 * v1, so this is fine.
 */
export function applyModuleOverrides(
  options: Pick<LuxarAppOptions, 'wasmPath' | 'workerPath'>
): void {
  if (options.wasmPath) {
    setWasmJsUrl(options.wasmPath);
    setDataWorkerWasmPath(options.wasmPath);
    // Third consumer: the depth-sort worker (its own module instance,
    // same shim URL).
    setSortWorkerWasmPath(options.wasmPath);
  }
  // NOTE: `workerPath` is deliberately NOT forwarded to the sort worker —
  // it names the DATA worker bundle, which is a different chunk. Embedders
  // relocating the sort worker call `setSortWorkerUrl` directly.
  if (options.workerPath) setDataWorkerUrl(options.workerPath);
}
