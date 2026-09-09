import { afterEach, describe, expect, it, vi } from 'vitest';
import { HttpChunkSource } from '../../../../cache/chunk-source/http-chunk-source';
import { MAX_CONCURRENT_CHUNK_FETCHES } from '../../../../utils/fetch-concurrency';

const REQUESTS = MAX_CONCURRENT_CHUNK_FETCHES + 32;
const BODY_DURATION_MS = 1_000;

interface TransportMetrics {
  completed: number;
  peakBodies: number;
}

async function runModeledTransport(): Promise<TransportMetrics> {
  let activeBodies = 0;
  let peakBodies = 0;

  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      activeBodies += 1;
      peakBodies = Math.max(peakBodies, activeBodies);
      let settled = false;
      let completionId: ReturnType<typeof setTimeout> | undefined;
      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            completionId = setTimeout(() => {
              if (settled) return;
              settled = true;
              controller.enqueue(new Uint8Array([1]));
              controller.close();
              activeBodies -= 1;
            }, BODY_DURATION_MS);
          },
          cancel() {
            if (settled) return;
            settled = true;
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
    peakBodies,
  };
}

describe('HttpChunkSource — modeled transport', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('caps simultaneous response bodies at the shared fetch limit', async () => {
    vi.useFakeTimers();
    const metricsPromise = runModeledTransport();
    await vi.runAllTimersAsync();
    const metrics = await metricsPromise;

    expect(metrics).toEqual({
      completed: REQUESTS,
      peakBodies: MAX_CONCURRENT_CHUNK_FETCHES,
    });
  });
});
