import { setWasmJsUrl } from '../../../wasm';
import { setDataWorkerUrl } from '../../../workers/worker-pool';
import type { LuxarAppOptions } from '../options';

/**
 * Forward asset-URL overrides to the WASM and worker modules. Skipped
 * when the option is undefined so the modules use their default
 * `import.meta.url`-based resolution.
 *
 * NOTE: the override is module-level and sticks across init() calls —
 * once set, a subsequent init() without the option does not reset to
 * the default. In practice we only support one LuxarApp per page in
 * v1, so this is fine.
 */
export function applyModuleOverrides(options: Pick<LuxarAppOptions, 'wasmPath' | 'workerPath'>): void {
  if (options.wasmPath) setWasmJsUrl(options.wasmPath);
  if (options.workerPath) setDataWorkerUrl(options.workerPath);
}
