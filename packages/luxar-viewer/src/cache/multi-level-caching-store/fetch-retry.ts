import { log, Modules } from '../../utils/log';
import { config } from '../../config';
import {
  getActiveFetchCount,
  getFetchProgressEpoch,
  noteFetchProgress,
  type FetchLane,
  withFetchGate,
} from '../../utils/fetch-concurrency';
import { getErrorMessage } from '../../utils/format-error';
import { sha256Hex } from './sha256';

const INITIAL_RETRY_DELAY_MS = 50;
const MAX_RETRY_DELAY_MS = 500;
const MIN_BODY_THROUGHPUT_BYTES_PER_SECOND = 16 * 1024;
const BODY_DEADLINE_FALLBACK_MULTIPLIER = 8;
const MAX_BODY_DEADLINE_FALLBACK_CONCURRENCY_SCALE = 4;
const MAX_BODY_DEADLINE_CONCURRENCY_SCALE = 8;
const MAX_TIMER_DELAY_MS = 2_147_483_647;
const NOOP_DISPOSE = (): void => {};

/** A merged abort signal plus the cleanup for fallback source listeners. */
export interface AbortSignalScope {
  signal: AbortSignal;
  dispose: () => void;
}

/** A response whose body must be consumed inside the fetch-gate lease. */
export interface FetchAttempt {
  response: Response;
  readBody: () => Promise<Uint8Array<ArrayBuffer>>;
}

export interface FetchRetryOptions {
  timeoutMsOverride?: number;
  signal?: AbortSignal;
  headers?: HeadersInit;
  onExhausted?: (error: unknown) => void;
  lane?: FetchLane;
}

class FetchConsumerError {
  constructor(readonly cause: unknown) {}
}

class FetchBodyError extends Error {
  constructor(cause: unknown) {
    super(getErrorMessage(cause));
    this.name = 'FetchBodyError';
    this.cause = cause;
  }
}

interface BodyReadContext {
  timeoutController: AbortController;
  signal: AbortSignal;
  stallTimeoutMs: number;
  lane: FetchLane;
}

/**
 * Cancel a response body the caller chose not to read (retryable 429/5xx,
 * terminal non-OK, or a post-fetch abort). Without the cancel the server can
 * keep streaming the ignored body, consuming bandwidth and holding the
 * connection until GC. Consumed or locked bodies are left alone.
 */
function releaseUnconsumedBody(response: Response): void {
  const body = response.body;
  if (!body || response.bodyUsed || body.locked) return;
  void body.cancel().catch(() => {});
}

export function bodyAbsoluteDeadlineMs(
  response: Response,
  stallTimeoutMs: number,
  concurrentFetches: number
): number {
  const concurrencyScale = Math.min(
    MAX_BODY_DEADLINE_CONCURRENCY_SCALE,
    Math.max(1, concurrentFetches)
  );
  const fallbackConcurrencyScale = Math.min(
    MAX_BODY_DEADLINE_FALLBACK_CONCURRENCY_SCALE,
    concurrencyScale
  );
  const fallbackMs = stallTimeoutMs * BODY_DEADLINE_FALLBACK_MULTIPLIER * fallbackConcurrencyScale;
  const contentLengthHeader = response.headers?.get('content-length');
  if (contentLengthHeader === null || contentLengthHeader === undefined) {
    return Math.min(MAX_TIMER_DELAY_MS, fallbackMs);
  }

  const contentLength = Number(contentLengthHeader);
  if (!Number.isFinite(contentLength) || contentLength < 0) {
    return Math.min(MAX_TIMER_DELAY_MS, fallbackMs);
  }

  return Math.min(
    MAX_TIMER_DELAY_MS,
    Math.max(
      fallbackMs,
      Math.ceil((contentLength * 1_000 * concurrencyScale) / MIN_BODY_THROUGHPUT_BYTES_PER_SECOND)
    )
  );
}

function joinBodyChunks(chunks: Uint8Array[], byteLength: number): Uint8Array<ArrayBuffer> {
  const data = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return data;
}

async function readBodyWithStallWatchdog(
  response: Response,
  context: BodyReadContext
): Promise<Uint8Array<ArrayBuffer>> {
  const { timeoutController, signal, stallTimeoutMs, lane } = context;
  let stallTimeoutId: ReturnType<typeof setTimeout> | undefined;
  let absoluteTimeoutId: ReturnType<typeof setTimeout> | undefined;
  const startAbsoluteDeadline = (): void => {
    if (absoluteTimeoutId !== undefined) return;
    const absoluteDeadlineMs = bodyAbsoluteDeadlineMs(
      response,
      stallTimeoutMs,
      getActiveFetchCount(lane)
    );
    absoluteTimeoutId = setTimeout(
      () =>
        timeoutController.abort(
          new Error(`response body exceeded absolute deadline of ${absoluteDeadlineMs} ms`)
        ),
      absoluteDeadlineMs
    );
  };
  const armWatchdog = (): void => {
    if (stallTimeoutId !== undefined) clearTimeout(stallTimeoutId);
    const observedProgressEpoch = getFetchProgressEpoch();
    stallTimeoutId = setTimeout(() => {
      if (getFetchProgressEpoch() !== observedProgressEpoch) {
        armWatchdog();
        return;
      }
      timeoutController.abort(
        new Error(`response bodies made no aggregate progress for ${stallTimeoutMs} ms`)
      );
    }, stallTimeoutMs);
  };
  const abortPromise = new Promise<never>((_, reject) => {
    const rejectAbort = (): void => reject(signal.reason);
    if (signal.aborted) rejectAbort();
    else signal.addEventListener('abort', rejectAbort, { once: true });
  });

  try {
    startAbsoluteDeadline();
    const reader = response.body?.getReader();
    if (!reader) {
      armWatchdog();
      return new Uint8Array(
        await Promise.race([response.arrayBuffer(), abortPromise])
      ) as Uint8Array<ArrayBuffer>;
    }

    const chunks: Uint8Array[] = [];
    let byteLength = 0;
    armWatchdog();
    try {
      for (;;) {
        const { done, value } = await Promise.race([reader.read(), abortPromise]);
        if (done) break;
        if (value.byteLength === 0) continue;
        chunks.push(value);
        byteLength += value.byteLength;
        noteFetchProgress();
        armWatchdog();
      }
    } catch (error) {
      void reader.cancel(error).catch(() => {});
      throw error;
    } finally {
      reader.releaseLock();
    }

    return joinBodyChunks(chunks, byteLength);
  } catch (error) {
    throw new FetchBodyError(error);
  } finally {
    if (stallTimeoutId !== undefined) clearTimeout(stallTimeoutId);
    if (absoluteTimeoutId !== undefined) clearTimeout(absoluteTimeoutId);
  }
}

/**
 * Merge two AbortSignals into one that fires when either source aborts.
 *
 * Uses native `AbortSignal.any` when available (Node 22+, modern browsers).
 * The fallback registers listeners on both sources, so callers must invoke
 * `dispose()` when the operation using `signal` settles. An abort disposes
 * both source listeners immediately before relaying the cancellation.
 */
export function mergeAbortSignals(primary: AbortSignal, caller?: AbortSignal): AbortSignalScope {
  if (!caller) return { signal: primary, dispose: NOOP_DISPOSE };
  type StaticAny = { any?: (signals: AbortSignal[]) => AbortSignal };
  const anyImpl = (AbortSignal as unknown as StaticAny).any;
  if (typeof anyImpl === 'function') {
    return { signal: anyImpl([primary, caller]), dispose: NOOP_DISPOSE };
  }

  const relay = new AbortController();
  let listening = false;
  const dispose = (): void => {
    if (!listening) return;
    listening = false;
    primary.removeEventListener('abort', onAbort);
    caller.removeEventListener('abort', onAbort);
  };
  const onAbort = (): void => {
    dispose();
    relay.abort(primary.aborted ? primary.reason : caller.reason);
  };

  if (primary.aborted || caller.aborted) {
    relay.abort(primary.aborted ? primary.reason : caller.reason);
  } else {
    listening = true;
    primary.addEventListener('abort', onAbort, { once: true });
    caller.addEventListener('abort', onAbort, { once: true });
  }
  return { signal: relay.signal, dispose };
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

function retryDelayMs(attempt: number): number {
  const base = INITIAL_RETRY_DELAY_MS * 2 ** attempt;
  const jitter = (Math.random() - 0.5) * 0.5 * base;
  return Math.min(Math.max(0, base + jitter), MAX_RETRY_DELAY_MS);
}

function callerAborted(options: FetchRetryOptions | undefined): boolean {
  return options?.signal?.aborted === true;
}

function reportRetryExhaustion(
  options: FetchRetryOptions | undefined,
  lastError: unknown,
  maxAttempts: number
): void {
  if (!lastError) return;
  options?.onExhausted?.(lastError);
  log.warning(
    Modules.CACHE,
    `Fetch failed after ${maxAttempts} attempt(s): ${getErrorMessage(lastError)}`
  );
}

function isRetryableResponse(response: Response): boolean {
  return !response.ok && (response.status >= 500 || response.status === 429);
}

async function consumeResponse<T>(
  response: Response,
  context: BodyReadContext,
  consume: (attempt: FetchAttempt) => Promise<T>
): Promise<T> {
  let bodyRead = false;
  try {
    return await consume({
      response,
      readBody: () => {
        if (bodyRead) throw new Error('response body can only be read once');
        bodyRead = true;
        return readBodyWithStallWatchdog(response, context);
      },
    });
  } catch (error) {
    if (error instanceof FetchBodyError) throw error;
    throw new FetchConsumerError(error);
  }
}

async function runFetchAttempt<T>(
  url: string,
  options: FetchRetryOptions | undefined,
  timeoutPerAttemptMs: number,
  consume: (attempt: FetchAttempt) => Promise<T>
): Promise<T> {
  const lane = options?.lane ?? 'data';
  return withFetchGate(async () => {
    // Start the per-attempt timer only after acquiring the gate. Queue wait is
    // controlled by the caller signal and must not consume the request budget.
    const timeoutController = new AbortController();
    const timeoutId = setTimeout(
      () =>
        timeoutController.abort(
          new Error(`response headers timed out after ${timeoutPerAttemptMs} ms`)
        ),
      timeoutPerAttemptMs
    );
    const abortScope = mergeAbortSignals(timeoutController.signal, options?.signal);
    let response: Response | undefined;
    try {
      response = await fetch(url, {
        signal: abortScope.signal,
        ...(options?.headers ? { headers: options.headers } : {}),
      });
      clearTimeout(timeoutId);
      if (isRetryableResponse(response)) throw new Error(`HTTP ${response.status} for ${url}`);
      return await consumeResponse(
        response,
        {
          timeoutController,
          signal: abortScope.signal,
          stallTimeoutMs: timeoutPerAttemptMs,
          lane,
        },
        consume
      );
    } finally {
      clearTimeout(timeoutId);
      if (response) releaseUnconsumedBody(response);
      abortScope.dispose();
    }
  }, lane);
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
  const data = new TextEncoder().encode(url);
  const hashHex = await sha256Hex(data);
  return `zarr-cache-${hashHex.slice(0, 16)}`;
}

/**
 * Fetch with retries for transient failures.
 *
 * 4xx responses are returned immediately because retrying cannot fix a
 * missing zarr key. Network errors, timeouts, 429, and 5xx responses are
 * retried using the configured retry budget. The configured timeout is split
 * across attempts and applied separately to time-to-headers and aggregate
 * body-progress stalls. A quiet HTTP/2 stream remains valid while any live
 * fetch is receiving bytes. Once body reading begins, an absolute deadline —
 * the longer of eight stall windows or `Content-Length` divided by a 16 KiB/s
 * aggregate floor, scaled by at most eight concurrent leases — bounds both a
 * zero-byte response and a true trickle.
 *
 * @param url - URL to fetch.
 * @param options - Optional `timeoutMsOverride` (e.g. for cache-validation
 *   probes that want shorter windows than data fetches) and a
 *   caller `signal` for cancellation propagation. The caller signal is
 *   merged with the per-attempt timeout signal so either abort source
 *   wins immediately. `lane` selects the bounded data or metadata pool.
 *   `onExhausted` receives the final retryable error. A caller-aborted call
 *   exits without consuming retry budget.
 * @param consume - Runs inside the global fetch-gate lease. Call `readBody()`
 *   to consume a response with a no-progress watchdog; returning without
 *   reading cancels the body before the lease is released.
 * @returns The consumer result, or `undefined` after abort or retry exhaustion.
 */
export async function fetchWithRetry<T>(
  url: string,
  options: FetchRetryOptions | undefined,
  consume: (attempt: FetchAttempt) => Promise<T>
): Promise<T | undefined> {
  const maxAttempts = Math.max(1, config.dataLoading.network.retryAttempts + 1);
  const totalTimeoutMs = options?.timeoutMsOverride ?? config.dataLoading.network.timeoutMs;
  const timeoutPerAttemptMs = Math.max(1, Math.ceil(totalTimeoutMs / maxAttempts));
  let lastError: unknown;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Caller-aborted requests must not be retried; bail out before the
    // next attempt.
    if (callerAborted(options)) {
      return undefined;
    }

    try {
      return await runFetchAttempt(url, options, timeoutPerAttemptMs, consume);
    } catch (error) {
      if (error instanceof FetchConsumerError) throw error.cause;
      lastError = error;
      // Caller-aborted: exit immediately rather than retrying.
      if (callerAborted(options)) {
        return undefined;
      }
    }

    if (attempt < maxAttempts - 1) {
      // Exponential backoff with +/- 25% jitter so two stores that started
      // a retry simultaneously (e.g. two browser tabs sharing a CDN)
      // do not synchronize their next attempts. Jitter is bounded by
      // MAX_RETRY_DELAY_MS so the worst-case wait stays predictable.
      await sleep(retryDelayMs(attempt));
    }
  }

  reportRetryExhaustion(options, lastError, maxAttempts);
  return undefined;
}
