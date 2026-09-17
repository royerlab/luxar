/**
 * Zipped-store benchmark: archive vs the directory store it was built from.
 *
 * This measures the zipped archive path against the directory store it was
 * built from, both cold and on a chunk-cache revisit. It reports five numbers
 * per variant:
 *
 *   1. time to first render (viewer `initialized`)
 *   2. request count
 *   3. bytes over the wire
 *   4. MAIN-THREAD LONG-TASK TOTAL
 *   5. central-directory preamble (zip only)
 *
 * (4) is the one that is easy to omit and decisive to have. `unzipit` ships
 * `useWorkers: false` and `ZipFileStore` never calls `setOptions`, so every
 * DEFLATE member is inflated ON THE MAIN THREAD. That cost does not
 * necessarily show up in wall-clock — it shows up as jank — so a benchmark
 * measuring only time/requests/bytes could bless a change that makes the
 * viewer stutter. The shipped demo archives are 100% DEFLATE.
 *
 * FAIRNESS NOTE: the default run gives every variant `?noCache`. That makes
 * it an uncached-vs-uncached comparison, which is the right A/B for the store
 * layer. Revisit mode instead leaves L1/L2 enabled for every variant and
 * measures the second load in the same browser context.
 */

import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { benchDataOrigin, benchFixtureDir } from './zip-bench-base';
import { waitForLuxarReady } from '../e2e/helpers';

/** How many times each variant is sampled; the median is reported. */
const REPEATS = Number(process.env.LUXAR_BENCH_REPEATS ?? 3);
/** Ready-wait budget. Software rendering in headless is slow; be generous. */
const READY_TIMEOUT_MS = Number(process.env.LUXAR_BENCH_READY_TIMEOUT_MS ?? 240_000);

/**
 * `LUXAR_BENCH_REVISIT=1` measures the SECOND load in the same browser context,
 * with the chunk cache ON.
 *
 * The default run is deliberately `?noCache`, which is the right A/B for the
 * store layer but says nothing about caching. And simply dropping `?noCache`
 * would say almost as little: every sample gets a fresh context, so L1 is empty
 * and L2/OPFS starts cold — a "cached" first visit is an uncached visit plus the
 * cost of populating the cache. The question caching actually answers is what a
 * RETURN visit costs, so measure that.
 */
const REVISIT = process.env.LUXAR_BENCH_REVISIT === '1';

interface Variant {
  readonly label: string;
  readonly file: string;
}

const ALL_VARIANTS: readonly Variant[] = [
  { label: 'directory', file: 'bench.luxar.zarr' },
  { label: 'zip (STORED)', file: 'bench-stored.luxar.zarr.zip' },
  { label: 'zip (DEFLATE)', file: 'bench-deflate.luxar.zarr.zip' },
];

/** `LUXAR_BENCH_VARIANTS=directory,zip (STORED)` narrows the run while iterating. */
const selected = process.env.LUXAR_BENCH_VARIANTS?.split(',').map((s) => s.trim());
const VARIANTS = selected?.length
  ? ALL_VARIANTS.filter((v) => selected.includes(v.label))
  : ALL_VARIANTS;

interface Sample {
  readyMs: number;
  requests: number;
  bytes: number;
  longTaskMs: number;
  /** What actually got loaded — see the equivalence guard in the test body. */
  totalPoints: number;
  pointClouds: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Bytes of central directory in a zip — the fixed preamble a reader must
 * download before it can resolve any chunk. Computed from the archive rather
 * than observed, so it is exact: 46 bytes of fixed record header per member
 * plus the variable-length name/extra/comment fields.
 */
function centralDirectoryBytes(archive: string): number {
  const buffer = fs.readFileSync(archive);
  // Locate the End of Central Directory record by scanning back for its magic.
  for (let i = buffer.length - 22; i >= 0; i--) {
    if (buffer.readUInt32LE(i) === 0x06054b50) {
      return buffer.readUInt32LE(i + 12); // size of the central directory
    }
  }
  return 0;
}

async function sample(page: import('@playwright/test').Page, datasetUrl: string): Promise<Sample> {
  let requests = 0;
  let bytes = 0;
  // Kept so a failed ready-wait can say WHY instead of just timing out.
  const problems: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') problems.push(`console: ${message.text()}`);
  });
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message}`));

  page.on('requestfinished', (request) => {
    const url = request.url();
    // Count DATA traffic only — the viewer bundle is identical across variants
    // and would swamp the signal we care about.
    if (!url.startsWith(benchDataOrigin)) return;
    requests += 1;
    void request
      .sizes()
      .then((sizes) => {
        bytes += sizes.responseBodySize + sizes.responseHeadersSize;
      })
      .catch(() => {
        /* request torn down with the page; its bytes are lost, not fatal */
      });
  });

  // Long tasks must be observed from before the first script runs.
  await page.addInitScript(() => {
    (window as unknown as { __longTaskMs: number }).__longTaskMs = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        (window as unknown as { __longTaskMs: number }).__longTaskMs += entry.duration;
      }
    }).observe({ entryTypes: ['longtask'] });
  });

  const query = REVISIT ? 'debug' : 'debug&noCache';

  if (REVISIT) {
    // First visit: populate the cache, then discard its counters entirely.
    await page.goto(`/?src=${encodeURIComponent(datasetUrl)}&${query}`);
    await waitForLuxarReady(page, READY_TIMEOUT_MS);
    requests = 0;
    bytes = 0;
    await page.evaluate(() => {
      (window as unknown as { __longTaskMs: number }).__longTaskMs = 0;
    });
  }

  const started = Date.now();
  await page.goto(`/?src=${encodeURIComponent(datasetUrl)}&${query}`);
  try {
    await waitForLuxarReady(page, READY_TIMEOUT_MS);
  } catch (error) {
    throw new Error(
      `${datasetUrl} never reached initialized.\n` +
        (problems.length ? problems.slice(0, 20).join('\n') : '(no console errors captured)'),
      { cause: error }
    );
  }
  const readyMs = Date.now() - started;

  // Let any trailing sizes() promises and long-task entries land.
  await page.waitForTimeout(500);
  const longTaskMs = await page.evaluate(
    () => (window as unknown as { __longTaskMs: number }).__longTaskMs ?? 0
  );

  const content = await page.evaluate(() => {
    const state = (
      window as unknown as {
        __luxarDebug?: { getState?: () => { totalPoints?: number; pointClouds?: unknown[] } };
      }
    ).__luxarDebug?.getState?.();
    return {
      totalPoints: state?.totalPoints ?? 0,
      pointClouds: state?.pointClouds?.length ?? 0,
    };
  });

  if (process.env.LUXAR_BENCH_DEBUG && problems.length) {
    console.log(`[bench] ${datasetUrl} console errors:\n  ${problems.slice(0, 10).join('\n  ')}`);
  }

  return { readyMs, requests, bytes, longTaskMs, ...content };
}

test('zipped vs directory cold open', async ({ browser }) => {
  const rows: string[] = [];
  const results: Record<string, Sample & { preambleBytes: number }> = {};

  for (const variant of VARIANTS) {
    const artifact = path.join(benchFixtureDir, variant.file);
    expect(
      fs.existsSync(artifact),
      `missing fixture ${artifact} — run pnpm bench:zip:fixtures`
    ).toBe(true);

    const samples: Sample[] = [];
    for (let run = 0; run < REPEATS; run++) {
      // A fresh context per run: no HTTP cache, no OPFS carry-over.
      const context = await browser.newContext();
      const page = await context.newPage();
      try {
        samples.push(await sample(page, `${benchDataOrigin}/${variant.file}`));
      } finally {
        await context.close();
      }
    }

    const preambleBytes = variant.file.endsWith('.zip') ? centralDirectoryBytes(artifact) : 0;
    const merged = {
      readyMs: median(samples.map((s) => s.readyMs)),
      requests: median(samples.map((s) => s.requests)),
      bytes: median(samples.map((s) => s.bytes)),
      longTaskMs: median(samples.map((s) => s.longTaskMs)),
      totalPoints: samples[0].totalPoints,
      pointClouds: samples[0].pointClouds,
      preambleBytes,
    };
    results[variant.label] = merged;

    rows.push(
      `| ${variant.label} | ${merged.readyMs.toFixed(0)} | ${merged.requests} | ` +
        `${(merged.bytes / 1024).toFixed(1)} | ${merged.longTaskMs.toFixed(0)} | ` +
        `${preambleBytes ? (preambleBytes / 1024).toFixed(1) : '—'} |`
    );
  }

  const table = [
    '',
    REVISIT
      ? `### Zipped-store REVISIT (median of ${REPEATS}, chunk cache ON, second load in one context)`
      : `### Zipped-store cold open (median of ${REPEATS}, uncached, Range-capable server)`,
    '',
    '| variant | ready (ms) | requests | bytes (kB) | long tasks (ms) | central dir (kB) |',
    '|---|---:|---:|---:|---:|---:|',
    ...rows,
    '',
  ].join('\n');
  console.log(table);

  const outFile = process.env.LUXAR_BENCH_OUT;
  if (outFile) {
    fs.mkdirSync(path.dirname(outFile), { recursive: true });
    fs.writeFileSync(outFile, JSON.stringify({ repeats: REPEATS, results }, null, 2));
  }

  // EQUIVALENCE GUARD. Without this the bench happily reports a spectacular
  // "win" for a store that loaded NOTHING: an empty scene still reaches
  // `initialized`, in a few hundred ms and a handful of requests. Every variant
  // is the same scene, so every variant must end up with the same geometry —
  // anything else means the run measured a failure, not a speed-up.
  const baseline = results[VARIANTS[0].label];
  for (const variant of VARIANTS) {
    const row = results[variant.label];
    expect(row.requests, `${variant.label}: no requests observed`).toBeGreaterThan(0);
    expect(row.totalPoints, `${variant.label}: loaded an EMPTY scene`).toBeGreaterThan(0);
    expect(
      row.totalPoints,
      `${variant.label} loaded ${row.totalPoints} points but ` +
        `${VARIANTS[0].label} loaded ${baseline.totalPoints} — the variants are not ` +
        'the same scene, so their timings are not comparable'
    ).toBe(baseline.totalPoints);
    expect(row.pointClouds, `${variant.label}: node count differs`).toBe(baseline.pointClouds);
  }
});
