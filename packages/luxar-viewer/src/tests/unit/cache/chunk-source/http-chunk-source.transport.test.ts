import { afterEach, describe, expect, it, vi } from 'vitest';
import { HttpChunkSource } from '../../../../cache/chunk-source/http-chunk-source';

const REQUESTS = 96;
const BODY_BYTES = 4 * 1024 * 1024;
const BYTES_PER_SECOND = 100 * 1024;
const BODY_DURATION_MS = (BODY_BYTES / BYTES_PER_SECOND) * 1_000;
const CONGESTION_STARTS_AT = 80;

interface TransportMetrics {
  completed: number;
  retries: number;
  peakBodies: number;
  peakBytes: number;
  totalMs: number;
}

async function runCongestedTransport(): Promise<TransportMetrics> {
  let activeBodies = 0;
  let peakBodies = 0;
  let fetches = 0;
  const startedAt = Date.now();

  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init?: RequestInit) => {
      fetches += 1;
      if (activeBodies >= CONGESTION_STARTS_AT) {
        return new Promise<Response>((_, reject) => {
          init?.signal?.addEventListener(
            'abort',
            () => reject(init.signal?.reason ?? new DOMException('aborted', 'AbortError')),
            { once: true }
          );
        });
      }

      activeBodies += 1;
      peakBodies = Math.max(peakBodies, activeBodies);
      let settled = false;
      let progressId: ReturnType<typeof setInterval> | undefined;
      let completionId: ReturnType<typeof setTimeout> | undefined;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            progressId = setInterval(() => controller.enqueue(new Uint8Array([1])), 1_000);
            completionId = setTimeout(() => {
              if (settled) return;
              settled = true;
              if (progressId !== undefined) clearInterval(progressId);
              controller.close();
              activeBodies -= 1;
            }, BODY_DURATION_MS);
          },
          cancel() {
            if (settled) return;
            settled = true;
            if (progressId !== undefined) clearInterval(progressId);
            if (completionId !== undefined) clearTimeout(completionId);
            activeBodies -= 1;
          },
        })
      );
    })
  );

  const outcomes = await Promise.all(
    Array.from({ length: REQUESTS }, (_, index) =>
      new HttpChunkSource('https://example.com/pinned-store').get(`chunk-${index}`)
    )
  );
  return {
    completed: outcomes.filter((outcome) => outcome.kind === 'ok').length,
    retries: fetches - REQUESTS,
    peakBodies,
    peakBytes: peakBodies * BODY_BYTES,
    totalMs: Date.now() - startedAt,
  };
}

describe('HttpChunkSource — throttled transport', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('completes a pinned 100 KB/s workload without retry amplification', async () => {
    vi.useFakeTimers();
    const metricsPromise = runCongestedTransport();
    await vi.runAllTimersAsync();
    const metrics = await metricsPromise;

    expect(metrics).toEqual({
      completed: REQUESTS,
      retries: 0,
      peakBodies: 64,
      peakBytes: 64 * BODY_BYTES,
      totalMs: 2 * BODY_DURATION_MS,
    });
  });
});
