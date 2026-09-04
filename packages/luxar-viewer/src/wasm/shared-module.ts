/**
 * One compiled `WebAssembly.Module`, shared by every data worker.
 *
 * Each worker used to call `initWasm()` independently, which fetches and
 * COMPILES its own copy of the same binary. Measured on a hosted demo with 15
 * workers: they became ready on a ~160 ms cadence, 2.2 s from the first to the
 * last, and the first LOD's decode sat waiting for all of them.
 *
 * A `WebAssembly.Module` is structured-cloneable, so the main thread can
 * compile once and postMessage the result into each worker, which then only
 * has to instantiate. See `instantiateWasmShim` in `./index.ts` for the
 * receiving half.
 *
 * DEGRADES, ALWAYS. Every failure path resolves `null` rather than rejecting,
 * and `null` simply means each worker takes today's self-init path. That is
 * deliberate: this is an optimization, and a broken optimization must not be
 * able to take the viewer down with it.
 *
 * @module wasm/shared-module
 */

import { config } from '../config';
import { log, Modules } from '../utils/log';
import { resolveWasmShimUrls } from './index';

/** Fallback deadline when the configured worker-init timeout is disabled. */
const DEFAULT_COMPILE_DEADLINE_MS = 10_000;
/** Shared compilation is an optimization; never spend a full init budget on it. */
const MAX_COMPILE_DEADLINE_MS = 3_000;

let sharedPromise: Promise<WebAssembly.Module | null> | null = null;

/**
 * Compile the WASM binary once and hand the same module to every caller.
 *
 * Never rejects and never hangs.
 *
 * @param shimUrlOverride Preferred shim href — the pool passes its
 *   `setDataWorkerWasmPath` value. Read on the FIRST call only, which matches
 *   the existing "set the overrides before the first worker is created"
 *   contract of `setWasmJsUrl` / `setDataWorkerUrl`.
 */
export function getSharedWasmModule(shimUrlOverride?: string): Promise<WebAssembly.Module | null> {
  sharedPromise ??= compileOnce(shimUrlOverride);
  return sharedPromise;
}

/** Drop the memoized module. Test-only. */
export function resetSharedWasmModule(): void {
  sharedPromise = null;
}

/** The deadline for the whole compile, mirroring the init guard's convention. */
function compileDeadlineMs(): number {
  // `<= 0 || !isFinite` disables the guard for the worker-init timeout
  // (see `init-with-guard.ts`); here "disabled" must still mean bounded,
  // because this await sits BEFORE that guard's timer starts and a stalled
  // fetch would otherwise hang every worker's init indefinitely.
  const configured = config.dataLoading.performance.workerInitTimeoutMs;
  return configured > 0 && Number.isFinite(configured)
    ? Math.min(configured, MAX_COMPILE_DEADLINE_MS)
    : DEFAULT_COMPILE_DEADLINE_MS;
}

/**
 * The binary sits next to its JS shim (`luxar_wasm_bg.wasm`), which is exactly
 * how the generated shim resolves it for itself — so a relocated `wasmPath`
 * keeps working without a second override.
 */
function binaryUrlsFor(shimUrlOverride?: string): string[] {
  const shimUrls = shimUrlOverride
    ? [shimUrlOverride, ...resolveWasmShimUrls()]
    : resolveWasmShimUrls();
  const seen = new Set<string>();
  const out: string[] = [];
  for (const shimUrl of shimUrls) {
    try {
      const url = new URL('luxar_wasm_bg.wasm', shimUrl);
      // http(s) ONLY. In Node/SSR (and the unit suite) the candidates resolve
      // to `file://`, which `fetch` cannot serve — depending on the runtime it
      // either throws or stalls until the deadline, and stalling here would
      // delay EVERY worker's init by the full budget. Skipping such candidates
      // makes this a fast no-op wherever it could not have worked anyway.
      if (url.protocol !== 'http:' && url.protocol !== 'https:') continue;
      if (!seen.has(url.href)) {
        seen.add(url.href);
        out.push(url.href);
      }
    } catch {
      // A shim href this environment cannot resolve against is simply not a
      // candidate; the next one may still work.
    }
  }
  return out;
}

async function compileCandidate(url: string): Promise<WebAssembly.Module> {
  try {
    return await WebAssembly.compileStreaming(fetch(url));
  } catch {
    // `compileStreaming` REQUIRES an `application/wasm` content type, which
    // plenty of static hosts (and file-backed dev setups) do not send. Buffer
    // it instead rather than losing the optimization to a MIME header.
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return WebAssembly.compile(await response.arrayBuffer());
  }
}

async function compileOnce(shimUrlOverride?: string): Promise<WebAssembly.Module | null> {
  const compile = async (): Promise<WebAssembly.Module | null> => {
    if (
      typeof WebAssembly === 'undefined' ||
      typeof WebAssembly.compile !== 'function' ||
      typeof fetch !== 'function'
    ) {
      return null;
    }

    for (const url of binaryUrlsFor(shimUrlOverride)) {
      let module: WebAssembly.Module;
      try {
        module = await compileCandidate(url);
      } catch {
        continue; // Try the next candidate; the walk mirrors initWasm's.
      }

      // Cloneability probe. Without it, a host where `WebAssembly.Module` is
      // not postMessage-able turns a DataCloneError inside Comlink into a
      // synchronous throw in the init guard — failing ALL workers identically
      // instead of degrading. That would make this optimization strictly worse
      // than not having it.
      try {
        structuredClone(module);
      } catch {
        log.info(
          Modules.WASM,
          'WASM modules are not structured-cloneable here; each worker will compile its own'
        );
        return null;
      }
      return module;
    }
    return null;
  };

  try {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<null>((resolve) => {
      timer = setTimeout(() => resolve(null), compileDeadlineMs());
    });
    const result = await Promise.race([compile(), deadline]);
    if (timer !== undefined) clearTimeout(timer);
    return result;
  } catch {
    // Belt and braces: `compile()` already swallows per-candidate failures.
    return null;
  }
}
