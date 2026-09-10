import { getEventListeners } from 'node:events';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  buildUrl,
  fetchWithRetry as fetchWithRetryScoped,
  hashUrl,
  mergeAbortSignals,
} from '../../../cache/multi-level-caching-store/fetch-retry';
import { MAX_CONCURRENT_CHUNK_FETCHES, withFetchGate } from '../../../utils/fetch-concurrency';
import { log } from '../../../utils/log';

function mockResponse(status: number, body: ArrayBuffer | string = ''): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    async arrayBuffer() {
      return typeof body === 'string' ? new TextEncoder().encode(body).buffer : body;
    },
  } as unknown as Response;
}

/** Response-like with a live ReadableStream body whose cancellation is
 *  observable — models a server still streaming an error/ignored body. */
function mockStreamingResponse(status: number): {
  response: Response;
  wasCancelled: () => boolean;
} {
  let cancelled = false;
  let used = false;
  const body = new ReadableStream<Uint8Array>({
    cancel() {
      cancelled = true;
    },
  });
  const response = {
    ok: status >= 200 && status < 300,
    status,
    body,
    get bodyUsed() {
      return used;
    },
    async arrayBuffer() {
      used = true;
      return new ArrayBuffer(0);
    },
  } as unknown as Response;
  return { response, wasCancelled: () => cancelled };
}

async function fetchWithRetry(
  url: string,
  options?: { timeoutMsOverride?: number; signal?: AbortSignal }
): Promise<Response | undefined> {
  return fetchWithRetryScoped(url, options, async ({ response }) => response);
}

function forceAbortSignalAnyFallback(): () => void {
  const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
  Object.defineProperty(AbortSignal, 'any', { configurable: true, value: undefined });
  return () => {
    if (descriptor) Object.defineProperty(AbortSignal, 'any', descriptor);
    else delete (AbortSignal as unknown as { any?: unknown }).any;
  };
}

describe('mergeAbortSignals', () => {
  it('returns the primary signal in a no-op scope when no caller is provided', () => {
    const primary = new AbortController().signal;
    const merged = mergeAbortSignals(primary);
    expect(merged.signal).toBe(primary);
    expect(() => merged.dispose()).not.toThrow();
  });

  it('keeps the native AbortSignal.any path unchanged', () => {
    const descriptor = Object.getOwnPropertyDescriptor(AbortSignal, 'any');
    const nativeSignal = new AbortController().signal;
    const any = vi.fn(() => nativeSignal);
    Object.defineProperty(AbortSignal, 'any', { configurable: true, value: any });
    try {
      const primary = new AbortController().signal;
      const caller = new AbortController().signal;
      const merged = mergeAbortSignals(primary, caller);
      expect(any).toHaveBeenCalledWith([primary, caller]);
      expect(merged.signal).toBe(nativeSignal);
      expect(() => merged.dispose()).not.toThrow();
    } finally {
      if (descriptor) Object.defineProperty(AbortSignal, 'any', descriptor);
      else delete (AbortSignal as unknown as { any?: unknown }).any;
    }
  });

  it('fallback abort from primary relays immediately and removes both listeners', () => {
    const restore = forceAbortSignalAnyFallback();
    try {
      const primary = new AbortController();
      const caller = new AbortController();
      const merged = mergeAbortSignals(primary.signal, caller.signal);
      expect(getEventListeners(primary.signal, 'abort')).toHaveLength(1);
      expect(getEventListeners(caller.signal, 'abort')).toHaveLength(1);

      const reason = new Error('primary timeout');
      primary.abort(reason);

      expect(merged.signal.aborted).toBe(true);
      expect(merged.signal.reason).toBe(reason);
      expect(getEventListeners(primary.signal, 'abort')).toHaveLength(0);
      expect(getEventListeners(caller.signal, 'abort')).toHaveLength(0);
    } finally {
      restore();
    }
  });

  it('fallback abort from caller relays immediately and removes both listeners', () => {
    const restore = forceAbortSignalAnyFallback();
    try {
      const primary = new AbortController();
      const caller = new AbortController();
      const merged = mergeAbortSignals(primary.signal, caller.signal);

      const reason = new Error('caller cancelled');
      caller.abort(reason);

      expect(merged.signal.aborted).toBe(true);
      expect(merged.signal.reason).toBe(reason);
      expect(getEventListeners(primary.signal, 'abort')).toHaveLength(0);
      expect(getEventListeners(caller.signal, 'abort')).toHaveLength(0);
    } finally {
      restore();
    }
  });

  it('fallback dispose is idempotent and prevents completed merges from accumulating', () => {
    const restore = forceAbortSignalAnyFallback();
    try {
      const primary = new AbortController();
      for (let i = 0; i < 100; i++) {
        const caller = new AbortController();
        const merged = mergeAbortSignals(primary.signal, caller.signal);
        merged.dispose();
        merged.dispose();
      }
      expect(getEventListeners(primary.signal, 'abort')).toHaveLength(0);
    } finally {
      restore();
    }
  });

  it('fallback returns an already-aborted scope without registering listeners', () => {
    const restore = forceAbortSignalAnyFallback();
    try {
      const primary = new AbortController();
      const reason = new Error('already timed out');
      primary.abort(reason);
      const caller = new AbortController();
      const merged = mergeAbortSignals(primary.signal, caller.signal);
      expect(merged.signal.aborted).toBe(true);
      expect(merged.signal.reason).toBe(reason);
      expect(getEventListeners(primary.signal, 'abort')).toHaveLength(0);
      expect(getEventListeners(caller.signal, 'abort')).toHaveLength(0);
    } finally {
      restore();
    }
  });
});

describe('buildUrl', () => {
  it('joins a clean base + key with a single slash', () => {
    expect(buildUrl('https://example.com/data.zarr', 'positions/0.0.0')).toBe(
      'https://example.com/data.zarr/positions/0.0.0'
    );
  });

  it('strips trailing slashes on the base', () => {
    expect(buildUrl('https://example.com/data.zarr/', 'k')).toBe('https://example.com/data.zarr/k');
    expect(buildUrl('https://example.com/data.zarr///', 'k')).toBe(
      'https://example.com/data.zarr/k'
    );
  });

  it('strips leading slashes on the key', () => {
    expect(buildUrl('https://example.com/d.zarr', '/k')).toBe('https://example.com/d.zarr/k');
    expect(buildUrl('https://example.com/d.zarr', '///k')).toBe('https://example.com/d.zarr/k');
  });

  it('prevents the triple-slash bug (base trailing + key leading)', () => {
    const url = buildUrl('https://example.com/d.zarr/', '/positions/0.0.0');
    expect(url).toBe('https://example.com/d.zarr/positions/0.0.0');
    // The only protocol-level "//" appears once; no // elsewhere.
    expect(url.replace('https://', '').includes('//')).toBe(false);
  });
});

describe('hashUrl', () => {
  it('matches the documented `zarr-cache-<16 hex>` format', async () => {
    expect(await hashUrl('https://example.com/d.zarr')).toMatch(/^zarr-cache-[0-9a-f]{16}$/);
  });

  it('is deterministic for the same URL', async () => {
    const url = 'https://example.com/d.zarr';
    expect(await hashUrl(url)).toBe(await hashUrl(url));
  });

  it('distinguishes different URLs', async () => {
    const a = await hashUrl('https://example.com/a.zarr');
    const b = await hashUrl('https://example.com/b.zarr');
    expect(a).not.toBe(b);
  });
});

describe('fetchWithRetry', () => {
  let originalFetch: typeof fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('returns the response on first success', async () => {
    const fetchMock = vi.fn(async () => mockResponse(200, 'hello'));
    global.fetch = fetchMock as unknown as typeof fetch;
    const response = await fetchWithRetry('https://example.com/x');
    expect(response?.ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('keeps fallback cancellation live through the body, then releases it', async () => {
    const restore = forceAbortSignalAnyFallback();
    const caller = new AbortController();
    try {
      global.fetch = vi.fn(async () => mockResponse(200, 'hello')) as unknown as typeof fetch;
      await fetchWithRetryScoped(
        'https://example.com/x',
        { signal: caller.signal },
        async ({ readBody }) => {
          expect(getEventListeners(caller.signal, 'abort')).toHaveLength(1);
          await readBody();
        }
      );

      expect(getEventListeners(caller.signal, 'abort')).toHaveLength(0);
    } finally {
      restore();
    }
  });

  it('returns a 4xx response immediately without retrying', async () => {
    const fetchMock = vi.fn(async () => mockResponse(404));
    global.fetch = fetchMock as unknown as typeof fetch;
    const response = await fetchWithRetry('https://example.com/x');
    expect(response?.status).toBe(404);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('retries on 5xx until the budget is exhausted', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => mockResponse(500));
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 10_000 });
    // Default retryAttempts: 3 → maxAttempts: 4. Backoffs: 50, 100, 200 ms
    // (jittered). Advance enough to flush every backoff.
    await vi.advanceTimersByTimeAsync(5_000);
    const response = await promise;

    // 5xx is retried; after the budget, returns undefined.
    expect(response).toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it('fallback retry exhaustion releases every per-attempt caller listener', async () => {
    vi.useFakeTimers();
    const restore = forceAbortSignalAnyFallback();
    const caller = new AbortController();
    try {
      global.fetch = vi.fn(async () => mockResponse(500)) as unknown as typeof fetch;

      const promise = fetchWithRetryScoped(
        'https://example.com/x',
        {
          timeoutMsOverride: 10_000,
          signal: caller.signal,
        },
        async ({ response }) => response
      );
      await vi.advanceTimersByTimeAsync(5_000);

      expect(await promise).toBeUndefined();
      expect(getEventListeners(caller.signal, 'abort')).toHaveLength(0);
    } finally {
      restore();
    }
  });

  it('cancels the unread body of every retried 5xx response', async () => {
    vi.useFakeTimers();
    const cancelFlags: Array<() => boolean> = [];
    const fetchMock = vi.fn(async () => {
      const { response, wasCancelled } = mockStreamingResponse(503);
      cancelFlags.push(wasCancelled);
      return response;
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetryScoped(
      'https://example.com/x',
      { timeoutMsOverride: 10_000 },
      async ({ response }) => response
    );
    await vi.advanceTimersByTimeAsync(5_000);

    expect(await promise).toBeUndefined();
    // Every retried response's still-streaming body was cancelled so the
    // server stops sending; otherwise up to maxAttempts bodies stream on.
    expect(cancelFlags).toHaveLength(4);
    for (const wasCancelled of cancelFlags) expect(wasCancelled()).toBe(true);
  });

  it('cancels a terminal response body the caller does not read', async () => {
    const { response, wasCancelled } = mockStreamingResponse(404);
    global.fetch = vi.fn(async () => response) as unknown as typeof fetch;

    const fetched = await fetchWithRetryScoped(
      'https://example.com/x',
      undefined,
      async ({ response: fetchedResponse }) => {
        expect(wasCancelled()).toBe(false);
        return fetchedResponse;
      }
    );
    expect(fetched?.status).toBe(404);
    expect(wasCancelled()).toBe(true);
  });

  it('leaves a consumed body alone', async () => {
    let cancelled = false;
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2, 3]));
          controller.close();
        },
        cancel() {
          cancelled = true;
        },
      })
    );
    global.fetch = vi.fn(async () => response) as unknown as typeof fetch;

    const data = await fetchWithRetryScoped(
      'https://example.com/x',
      undefined,
      async ({ readBody }) => readBody()
    );
    expect(Array.from(data ?? [])).toEqual([1, 2, 3]);
    expect(cancelled).toBe(false);
  });

  it('retries on 429 (rate-limited)', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => mockResponse(429));
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 10_000 });
    await vi.advanceTimersByTimeAsync(5_000);
    await promise;

    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it('retries on network errors (fetch throws)', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => {
      throw new Error('ENETUNREACH');
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 10_000 });
    await vi.advanceTimersByTimeAsync(5_000);
    const response = await promise;

    expect(response).toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it('preserves details from a non-Error object thrown by fetch', async () => {
    vi.useFakeTimers();
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
    global.fetch = vi.fn(async () => {
      throw { code: 'ENETUNREACH', retryable: true };
    }) as unknown as typeof fetch;

    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 10_000 });
    await vi.advanceTimersByTimeAsync(5_000);
    await promise;

    expect(warning).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringContaining('{"code":"ENETUNREACH","retryable":true}')
    );
    warning.mockRestore();
  });

  it('does not retry when the caller signal is already aborted at entry', async () => {
    const ac = new AbortController();
    ac.abort();
    const fetchMock = vi.fn(async () => mockResponse(200));
    global.fetch = fetchMock as unknown as typeof fetch;

    const response = await fetchWithRetry('https://example.com/x', { signal: ac.signal });
    expect(response).toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('bails mid-retry when the caller aborts', async () => {
    vi.useFakeTimers();
    const ac = new AbortController();
    const fetchMock = vi.fn(async () => {
      // Abort the caller signal after the first failure.
      if (fetchMock.mock.calls.length === 1) ac.abort();
      throw new Error('network');
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetry('https://example.com/x', {
      timeoutMsOverride: 10_000,
      signal: ac.signal,
    });
    await vi.advanceTimersByTimeAsync(5_000);
    const response = await promise;

    expect(response).toBeUndefined();
    // Either 1 (caught via signal check inside catch) or 2 (signal seen
    // by the loop-top check). Both honour the "no retries after caller
    // abort" contract; we just assert "did not exhaust the budget".
    expect(fetchMock.mock.calls.length).toBeLessThan(4);
  });

  it('honours the per-attempt timeout (= ceil(total / maxAttempts))', async () => {
    vi.useFakeTimers();
    // fetch resolves only when the timeout-controller aborts it.
    let abortCount = 0;
    const fetchMock = vi.fn(async (_url: string, opts?: { signal?: AbortSignal }) => {
      return new Promise<Response>((_, reject) => {
        const onAbort = () => {
          abortCount++;
          reject(new Error('aborted'));
        };
        if (opts?.signal?.aborted) {
          onAbort();
        } else {
          opts?.signal?.addEventListener('abort', onAbort, { once: true });
        }
      });
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    // Total budget 1000ms → per-attempt ceil(1000/4)=250ms.
    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 1000 });
    await vi.advanceTimersByTimeAsync(10_000);
    const response = await promise;

    expect(response).toBeUndefined();
    // Each attempt should have aborted via the per-attempt timeout.
    expect(abortCount).toBeGreaterThan(0);
  });

  // [cache.md/G10][P5] The "5xx then success on retry" branch was never
  // exercised directly against `fetchWithRetry` — only via the multi-level
  // store. This is the typical transient-outage flow.
  it('5xx → 200 on retry returns the eventual success response', async () => {
    vi.useFakeTimers();
    let calls = 0;
    const fetchMock = vi.fn(async () => {
      calls++;
      return calls < 3 ? mockResponse(503) : mockResponse(200, 'hello');
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 10_000 });
    await vi.advanceTimersByTimeAsync(5_000);
    const response = await promise;

    // Eventually-successful response is surfaced exactly (status, body).
    expect(response?.status).toBe(200);
    expect(response?.ok).toBe(true);
    // Exactly 3 calls: two 503s + one 200. NOT a 4th retry after success.
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('does not charge the per-attempt timeout for time spent in the concurrency queue', async () => {
    vi.useFakeTimers();
    // Saturate the global fetch gate so the next request must wait in the queue.
    const release: Array<() => void> = [];
    const held = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () =>
      withFetchGate(() => new Promise<void>((resolve) => release.push(resolve)))
    );
    try {
      let sawAbortedSignal: boolean | undefined;
      const fetchMock = vi.fn(async (_url: string, opts?: { signal?: AbortSignal }) => {
        sawAbortedSignal = opts?.signal?.aborted ?? false;
        return mockResponse(200, 'ok');
      });
      global.fetch = fetchMock as unknown as typeof fetch;

      // Total budget 1000ms → per-attempt ceil(1000/4)=250ms.
      const promise = fetchWithRetry('https://example.com/queued', { timeoutMsOverride: 1000 });

      // While the request is stuck behind a saturated gate, advance far past the
      // per-attempt budget. If the timeout started at enqueue (the pre-fix bug),
      // the request's signal would already be aborted by the time it runs.
      await vi.advanceTimersByTimeAsync(2000);
      expect(fetchMock).not.toHaveBeenCalled();

      // Free the gate; the queued request acquires a slot and starts its timer
      // fresh, so it fetches with a live (non-aborted) signal and succeeds.
      release.forEach((r) => r());
      await vi.advanceTimersByTimeAsync(0);
      const response = await promise;

      expect(response?.ok).toBe(true);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      // The discriminator: pre-fix this is true (queue wait aborted the signal).
      expect(sawAbortedSignal).toBe(false);
    } finally {
      // Always drain the gate so the module-global counter resets for later tests.
      release.forEach((r) => r());
      await Promise.all(held);
    }
  });

  it('holds each fetch-gate slot until its response body is consumed', async () => {
    const bodyControllers: ReadableStreamDefaultController<Uint8Array>[] = [];
    const fetchMock = vi.fn(
      async () =>
        new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              bodyControllers.push(controller);
            },
          })
        )
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const requests = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES + 1 }, (_, index) =>
      fetchWithRetryScoped(
        `https://example.com/chunk-${index}`,
        { timeoutMsOverride: 10_000 },
        async ({ readBody }) => readBody()
      )
    );
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(MAX_CONCURRENT_CHUNK_FETCHES));

    for (const controller of bodyControllers) controller.close();
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledTimes(MAX_CONCURRENT_CHUNK_FETCHES + 1)
    );
    bodyControllers.at(-1)?.close();
    await Promise.all(requests);
  });

  it('allows a body to exceed the deadline while bytes keep arriving', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      async () =>
        new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              for (let index = 1; index <= 4; index++) {
                setTimeout(() => controller.enqueue(new Uint8Array([index])), index * 200);
              }
              setTimeout(() => controller.close(), 850);
            },
          })
        )
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetryScoped(
      'https://example.com/slow-progress',
      { timeoutMsOverride: 1_000 },
      async ({ readBody }) => readBody()
    );
    await vi.advanceTimersByTimeAsync(1_000);

    expect(Array.from((await promise) ?? [])).toEqual([1, 2, 3, 4]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('derives the absolute deadline from Content-Length at the throughput floor', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(
      async () =>
        new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              for (let index = 1; index <= 15; index++) {
                setTimeout(() => controller.enqueue(new Uint8Array([index])), index * 80);
              }
              setTimeout(() => controller.close(), 1_250);
            },
          }),
          { headers: { 'Content-Length': String(32 * 1024) } }
        )
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetryScoped(
      'https://example.com/known-length',
      { timeoutMsOverride: 400 },
      async ({ readBody }) => readBody()
    );
    await vi.advanceTimersByTimeAsync(2_000);

    expect((await promise)?.byteLength).toBe(15);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('allows a known-length body to use the fallback absolute deadline', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => {
      let cancelled = false;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            for (let index = 1; index <= 64; index++) {
              setTimeout(() => {
                if (!cancelled) controller.enqueue(new Uint8Array(1024));
              }, index * 200);
            }
            setTimeout(() => {
              if (!cancelled) controller.close();
            }, 12_900);
          },
          cancel() {
            cancelled = true;
          },
        }),
        { headers: { 'Content-Length': String(64 * 1024) } }
      );
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetryScoped(
      'https://example.com/slow-known-length',
      { timeoutMsOverride: 30_000 },
      async ({ readBody }) => readBody()
    );
    await vi.runAllTimersAsync();

    expect((await promise)?.byteLength).toBe(64 * 1024);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('bounds a continuously trickling body by an absolute deadline', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn(async () => {
      let cancelled = false;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            for (let index = 1; index <= 20; index++) {
              setTimeout(() => {
                if (!cancelled) controller.enqueue(new Uint8Array([1]));
              }, index * 200);
            }
          },
          cancel() {
            cancelled = true;
          },
        })
      );
    });
    global.fetch = fetchMock as unknown as typeof fetch;
    let exhaustedCause: unknown;

    const promise = fetchWithRetryScoped(
      'https://example.com/endless-trickle',
      { timeoutMsOverride: 1_000, onExhausted: (error) => (exhaustedCause = error) },
      async ({ readBody }) => readBody()
    );
    await vi.runAllTimersAsync();

    expect(await promise).toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(exhaustedCause).toMatchObject({
      cause: expect.objectContaining({ message: expect.stringContaining('absolute deadline') }),
    });
  });

  it('rejects a second read of the same response body', async () => {
    global.fetch = vi.fn(
      async () => new Response(new Uint8Array([1, 2, 3]))
    ) as unknown as typeof fetch;

    await expect(
      fetchWithRetryScoped('https://example.com/double-read', undefined, async ({ readBody }) => {
        await readBody();
        return readBody();
      })
    ).rejects.toThrow('response body can only be read once');
  });

  it('aborts a stalled body and retries it', async () => {
    vi.useFakeTimers();
    let attempt = 0;
    const fetchMock = vi.fn(async () => {
      attempt += 1;
      if (attempt === 1) {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(new Uint8Array([1]));
            },
          })
        );
      }
      return new Response(new Uint8Array([2, 3]));
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetryScoped(
      'https://example.com/stall',
      { timeoutMsOverride: 1_000 },
      async ({ readBody }) => readBody()
    );
    await vi.advanceTimersByTimeAsync(2_000);

    expect(Array.from((await promise) ?? [])).toEqual([2, 3]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('does not abort a quiet multiplexed body while another fetch is making progress', async () => {
    vi.useFakeTimers();
    const calls = new Map<string, number>();
    const fetchMock = vi.fn(async (url: string) => {
      calls.set(url, (calls.get(url) ?? 0) + 1);
      if (url.endsWith('/active')) {
        return new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              for (let index = 1; index <= 12; index++) {
                setTimeout(() => controller.enqueue(new Uint8Array([index])), index * 75);
              }
              setTimeout(() => controller.close(), 950);
            },
          }),
          { headers: { 'Content-Length': String(32 * 1024) } }
        );
      }
      let timer: ReturnType<typeof setTimeout>;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            setTimeout(() => controller.enqueue(new Uint8Array([8])), 50);
            timer = setTimeout(() => {
              controller.enqueue(new Uint8Array([9]));
              controller.close();
            }, 900);
          },
          cancel() {
            clearTimeout(timer);
          },
        })
      );
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const active = fetchWithRetryScoped(
      'https://example.com/active',
      { timeoutMsOverride: 400 },
      async ({ readBody }) => readBody()
    );
    const quiet = fetchWithRetryScoped(
      'https://example.com/quiet',
      { timeoutMsOverride: 400 },
      async ({ readBody }) => readBody()
    );

    await vi.advanceTimersByTimeAsync(1_000);

    expect((await active)?.byteLength).toBe(12);
    expect(Array.from((await quiet) ?? [])).toEqual([8, 9]);
    expect(calls.get('https://example.com/quiet')).toBe(1);
  });

  it('stops a body read immediately when the caller aborts', async () => {
    const caller = new AbortController();
    const fetchMock = vi.fn(
      async () =>
        new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(new Uint8Array([1]));
            },
          })
        )
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const promise = fetchWithRetryScoped(
      'https://example.com/caller-abort',
      { signal: caller.signal },
      async ({ readBody }) => readBody()
    );
    caller.abort();

    expect(await promise).toBeUndefined();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('caps backoff at MAX_RETRY_DELAY_MS (~500ms)', async () => {
    vi.useFakeTimers();
    const callTimes: number[] = [];
    const fetchMock = vi.fn(async () => {
      callTimes.push(Date.now());
      return mockResponse(500);
    });
    global.fetch = fetchMock as unknown as typeof fetch;

    const start = Date.now();
    const promise = fetchWithRetry('https://example.com/x', { timeoutMsOverride: 10_000 });
    await vi.advanceTimersByTimeAsync(10_000);
    await promise;

    // 4 attempts; backoffs between them: 50 (jittered ±25%), 100, 200
    // (capped at 500). Total worst-case ≈ 50*1.25 + 100*1.25 + 200*1.25 ≈
    // 437ms; cap ensures it cannot exceed 500*3 = 1500ms.
    const totalElapsed = callTimes[callTimes.length - 1] - start;
    expect(totalElapsed).toBeLessThan(1500);
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });
});
