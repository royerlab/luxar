/** Slow-link fetch regressions for #2678. */

import { expect, test, type Page } from '@playwright/test';
import { applyNetworkProfile, type NetworkProfile } from './perf-audit-helpers';
import { PERF_SLOW_DATA_BASE } from './perf-data-base';

const HOSTED_STORE =
  process.env.LUXAR_SLOW_LINK_STORE ??
  'https://data.luxarviewer.dev/data/2026-09-02/gsplats_3d_visible_human_head.luxar.zarr';
const HOSTED_CHUNKS = [
  'visible_human_head/colors/c/3/0',
  'visible_human_head/centers/c/5/0',
  'visible_human_head/cholesky_factors_offdiag/c/19/0',
  'visible_human_head/amplitudes/c/1',
] as const;
const LOCAL_TARGET = `${PERF_SLOW_DATA_BASE}/__luxar_slow_wave__`;
const HOSTED_REQUESTS = 24;
const LOCAL_REQUESTS = 6;
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

interface WaveResult {
  completed: number;
  bytes: number;
  exhausted: number;
  elapsedMs: number;
  retriedRequests: number;
}

test.skip(
  process.env.LUXAR_PERF_PREVIEW === '1',
  'the source-module fetch harness requires the Vite development server'
);

async function targetReachable(page: Page, url: string): Promise<boolean> {
  try {
    const response = await page.request.get(url, { timeout: 15_000 });
    const reachable = response.ok();
    await response.dispose();
    return reachable;
  } catch {
    return false;
  }
}

async function installRetryModule(page: Page): Promise<void> {
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
}

async function runWave(
  page: Page,
  profile: NetworkProfile,
  targets: readonly string[],
  requestCount: number
): Promise<WaveResult> {
  const attempts = new Map<string, number>();
  page.on('request', (request) => {
    const url = request.url();
    if (targets.some((target) => url.startsWith(target))) {
      attempts.set(url, (attempts.get(url) ?? 0) + 1);
    }
  });

  await installRetryModule(page);
  const cdp = await applyNetworkProfile(page, profile);
  await cdp.send('Network.setCacheDisabled', { cacheDisabled: true });
  try {
    const result = await page.evaluate(
      async ({ requestCount, urls }) => {
        const fetchWithRetry = (
          window as unknown as {
            __slowLinkFetch: RetryModule['fetchWithRetry'];
          }
        ).__slowLinkFetch;
        let exhausted = 0;
        const startedAt = performance.now();
        const waveId = crypto.randomUUID();
        const bodies = await Promise.all(
          Array.from({ length: requestCount }, (_, index) =>
            fetchWithRetry(
              `${urls[index % urls.length]}?slow-link-wave=${waveId}&slow-link-audit=${index}`,
              { timeoutMsOverride: 30_000, onExhausted: () => (exhausted += 1) },
              async ({ readBody }) => readBody()
            )
          )
        );
        return {
          completed: bodies.filter((body) => body !== undefined).length,
          bytes: bodies.reduce((sum, body) => sum + (body?.byteLength ?? 0), 0),
          exhausted,
          elapsedMs: performance.now() - startedAt,
        };
      },
      { requestCount, urls: targets }
    );
    return {
      ...result,
      retriedRequests: [...attempts.values()].filter((count) => count > 1).length,
    };
  } finally {
    await cdp.detach();
  }
}

function expectCompleteWave(result: WaveResult, expectedRequests: number): void {
  expect(result.completed).toBe(expectedRequests);
  expect(result.bytes).toBeGreaterThan(0);
  expect(result.exhausted).toBe(0);
}

test('hermetic HTTP/1.1 chunk wave completes at slow100k', async ({ page }) => {
  const result = await runWave(page, 'slow100k', [LOCAL_TARGET], LOCAL_REQUESTS);
  test.info().annotations.push({
    type: 'slow-link-metrics',
    description: JSON.stringify({ profile: 'slow100k', target: 'local', ...result }),
  });

  expectCompleteWave(result, LOCAL_REQUESTS);
  expect(result.retriedRequests).toBe(0);
});

for (const profile of profiles) {
  test(`hosted multiplexed chunk wave completes at ${profile}`, async ({ page }) => {
    const targets = HOSTED_CHUNKS.map((chunk) => `${HOSTED_STORE}/${chunk}`);
    test.skip(!(await targetReachable(page, targets[0])), `dataset not reachable: ${HOSTED_STORE}`);

    const result = await runWave(page, profile, targets, HOSTED_REQUESTS);
    test.info().annotations.push({
      type: 'slow-link-metrics',
      description: JSON.stringify({ profile, target: 'hosted', ...result }),
    });

    expectCompleteWave(result, HOSTED_REQUESTS);
    expect(result.retriedRequests).toBeLessThanOrEqual(profile === 'slow100k' ? 1 : 0);
  });
}
