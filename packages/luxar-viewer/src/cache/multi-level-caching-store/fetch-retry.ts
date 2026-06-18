import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import { withFetchGate } from '../../utils/fetch-concurrency';

const INITIAL_RETRY_DELAY_MS = 50;
const MAX_RETRY_DELAY_MS = 500;

/**
 * Merge two AbortSignals into one that fires when either source aborts.
 *
 * Uses native `AbortSignal.any` when available (Node 22+, modern browsers);
 * falls back to a hand-rolled relay otherwise.
 */
export function mergeAbortSignals(primary: AbortSignal, caller?: AbortSignal): AbortSignal {
  if (!caller) return primary;
  type StaticAny = { any?: (signals: AbortSignal[]) => AbortSignal };
  const anyImpl = (AbortSignal as unknown as StaticAny).any;
  if (typeof anyImpl === 'function') {
    return anyImpl([primary, caller]);
  }
  // Fallback: relay aborts onto a fresh controller.
  const relay = new AbortController();
  const onAbort = (): void => relay.abort();
  if (primary.aborted || caller.aborted) {
    relay.abort();
  } else {
    primary.addEventListener('abort', onAbort, { once: true });
    caller.addEventListener('abort', onAbort, { once: true });
  }
  return relay.signal;
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

/**
 * Build a clean URL by joining `baseUrl` and `key`, stripping any
 * trailing slashes on the base and leading slashes on the key. Prevents
 * the triple-slash bug observed when baseUrl includes a trailing `/`
 * and the key starts with `/`.
 */
export function buildUrl(baseUrl: string, key: string): string {
  const cleanBase = baseUrl.replace(/\/+$/, '');
  const cleanKey = key.replace(/^\/+/, '');
  return `${cleanBase}/${cleanKey}`;
}

/**
 * Generate a unique, collision-resistant 64-bit hash of `url` as a
 * `zarr-cache-<16 hex>` string. Used as the OPFS bucket-root identifier
 * for a dataset.
 */
export async function hashUrl(url: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(url);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  return `zarr-cache-${hashHex.slice(0, 16)}`;
}

/**
 * Fetch with retries for transient failures.
 *
 * 4xx responses are returned immediately because retrying cannot fix a
 * missing zarr key. Network errors, timeouts, 429, and 5xx responses are
 * retried using the configured retry budget. The configured timeout is
 * treated as a total budget across attempts so retries do not multiply
 * worst-case load time.
 *
 * @param url - URL to fetch.
 * @param options - Optional `timeoutMsOverride` (e.g. for cache-validation
 *   probes that want a shorter budget than the data-fetch timeout) and a
 *   caller `signal` for dispose-cancel propagation. The caller signal is
 *   merged with the per-attempt timeout signal so either abort source
 *   wins immediately. A caller-aborted call exits without consuming
 *   retry budget.
 */
export async function fetchWithRetry(
  url: string,
  options?: { timeoutMsOverride?: number; signal?: AbortSignal }
): Promise<Response | undefined> {
  const maxAttempts = Math.max(1, config.dataLoading.network.retryAttempts + 1);
  const totalTimeoutMs = options?.timeoutMsOverride ?? config.dataLoading.network.timeoutMs;
  const timeoutPerAttemptMs = Math.max(1, Math.ceil(totalTimeoutMs / maxAttempts));
  let lastError: unknown;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Caller-aborted requests must not be retried; bail out before the
    // next attempt.
    if (options?.signal?.aborted) {
      return undefined;
    }

    const timeoutController = new AbortController();
    const timeoutId = setTimeout(() => timeoutController.abort(), timeoutPerAttemptMs);
    const signal = mergeAbortSignals(timeoutController.signal, options?.signal);

    try {
      // Bounded-concurrency gate: zarrita fans out one fetch per chunk, so a
      // large LOD selection would otherwise fire thousands at once and exhaust
      // the browser (ERR_INSUFFICIENT_RESOURCES). The slot is held only for the
      // request itself (headers); the small chunk body is read by the caller.
      const response = await withFetchGate(() => fetch(url, { signal }));
      if (response.ok || (response.status < 500 && response.status !== 429)) {
        return response;
      }
      lastError = new Error(`HTTP ${response.status} for ${url}`);
    } catch (error) {
      lastError = error;
      // Caller-aborted: exit immediately rather than retrying.
      if (options?.signal?.aborted) {
        return undefined;
      }
    } finally {
      clearTimeout(timeoutId);
    }

    if (attempt < maxAttempts - 1) {
      // Exponential backoff with +/- 25% jitter so two stores that started
      // a retry simultaneously (e.g. two browser tabs sharing a CDN)
      // do not synchronize their next attempts. Jitter is bounded by
      // MAX_RETRY_DELAY_MS so the worst-case wait stays predictable.
      const base = INITIAL_RETRY_DELAY_MS * 2 ** attempt;
      const jitter = (Math.random() - 0.5) * 0.5 * base; // [-25%, +25%]
      const delayMs = Math.min(Math.max(0, base + jitter), MAX_RETRY_DELAY_MS);
      await sleep(delayMs);
    }
  }

  if (lastError) {
    const message = lastError instanceof Error ? lastError.message : String(lastError);
    log.warning(Modules.CACHE, `Fetch failed after ${maxAttempts} attempt(s): ${message}`);
  }
  return undefined;
}
