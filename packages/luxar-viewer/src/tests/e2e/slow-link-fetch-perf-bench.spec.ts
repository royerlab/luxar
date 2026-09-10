/**
 * Real-network regression for #2678.
 *
 * Reads the exact pinned Visible Human chunks from the report through the
 * production fetch gate/retry module with a cold browser cache and CDP
 * throttling. This intentionally lives in test:perf:e2e: it depends on the
 * hosted store and is too slow for unit/E2E CI.
 */

import { expect, test } from '@playwright/test';
import { applyNetworkProfile, type NetworkProfile } from './perf-audit-helpers';

const STORE =
  'https://data.luxarviewer.dev/data/2026-09-02/gsplats_3d_visible_human_head.luxar.zarr';
const CHUNKS = [
  'visible_human_head/colors/c/3/0',
  'visible_human_head/centers/c/5/0',
  'visible_human_head/cholesky_factors_offdiag/c/19/0',
  'visible_human_head/amplitudes/c/1',
] as const;
const REQUESTS = 24;
const DEFAULT_PROFILES: NetworkProfile[] = ['slow100k', 'slow1m', 'slow3m'];
const profiles = (process.env.LUXAR_SLOW_LINK_PROFILES?.split(',').filter(Boolean) ??
  DEFAULT_PROFILES) as NetworkProfile[];

interface RetryModule {
  fetchWithRetry<T>(
    url: string,
    options: { timeoutMsOverride: number; onExhausted: () => void },
    consume: (attempt: { readBody: () => Promise<Uint8Array> }) => Promise<T>
  ): Promise<T | undefined>;
}

for (const profile of profiles) {
  test(`hosted multiplexed chunk wave completes without retries at ${profile}`, async ({
    page,
  }) => {
    const attempts = new Map<string, number>();
    page.on('request', (request) => {
      const url = request.url();
      if (url.startsWith(`${STORE}/`)) attempts.set(url, (attempts.get(url) ?? 0) + 1);
    });

    await page.goto('/');
    await page.evaluate(async () => {
      const loadModule = new Function(
        'return import("/src/cache/multi-level-caching-store/fetch-retry.ts")'
      ) as () => Promise<RetryModule>;
      const { fetchWithRetry } = await loadModule();
      (
        window as unknown as {
          __slowLinkFetch: RetryModule['fetchWithRetry'];
        }
      ).__slowLinkFetch = fetchWithRetry;
    });
    const cdp = await applyNetworkProfile(page, profile);
    await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
    const result = await page.evaluate(
      async ({ chunks, requestCount, store }) => {
        const fetchWithRetry = (
          window as unknown as {
            __slowLinkFetch: RetryModule['fetchWithRetry'];
          }
        ).__slowLinkFetch;
        let exhausted = 0;
        const startedAt = performance.now();
        const bodies = await Promise.all(
          Array.from({ length: requestCount }, (_, index) => {
            const chunk = chunks[index % chunks.length];
            return fetchWithRetry(
              `${store}/${chunk}?slow-link-audit=${index}`,
              { timeoutMsOverride: 30_000, onExhausted: () => (exhausted += 1) },
              async ({ readBody }) => readBody()
            );
          })
        );
        return {
          completed: bodies.filter((body) => body !== undefined).length,
          bytes: bodies.reduce((sum, body) => sum + (body?.byteLength ?? 0), 0),
          exhausted,
          elapsedMs: performance.now() - startedAt,
        };
      },
      { chunks: CHUNKS, requestCount: REQUESTS, store: STORE }
    );
    const retriedRequests = [...attempts.values()].filter((count) => count > 1).length;

    test.info().annotations.push({
      type: 'slow-link-metrics',
      description: JSON.stringify({ profile, retriedRequests, ...result }),
    });

    expect(result.completed).toBe(REQUESTS);
    expect(result.bytes).toBeGreaterThan(0);
    expect(result.exhausted).toBe(0);
    expect(retriedRequests).toBe(0);
  });
}
