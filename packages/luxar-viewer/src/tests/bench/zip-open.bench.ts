/**
 * Cold-open benchmark: zipped store vs the directory store it was built from.
 *
 * This is the instrument that decides whether reading `.zarr.zip` through the
 * chunk cache is worth building (royerlab/luxar#1716). It reports the five
 * numbers that issue names, per variant:
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
 * FAIRNESS NOTE: every variant runs with `?no-cache`, because a zipped store
 * currently bypasses L1/L2 regardless. That makes this an uncached-vs-uncached
 * comparison, which is the right A/B for the store layer but is NOT the cold
 * open a user experiences on a directory store with the cache on. Read the
 * directory row as "the same store, same conditions", not as today's baseline.
 */

import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { benchDataOrigin, benchFixtureDir } from './zip-bench-base';
import { waitForLuxarReady } from '../e2e/helpers';

/** How many times each variant is sampled; the median is reported. */
const REPEATS = Number(process.env.LUXAR_BENCH_REPEATS ?? 3);

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

  const started = Date.now();
  await page.goto(`/?src=${encodeURIComponent(datasetUrl)}&debug&no-cache`);
  try {
    await waitForLuxarReady(page, 120_000);
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

  return { readyMs, requests, bytes, longTaskMs };
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
    `### Zipped-store cold open (median of ${REPEATS}, uncached, Range-capable server)`,
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

  // The benchmark's job is to REPORT, not to gate — a regression here is a
  // decision for #1716, not a red build. Assert only that it measured something.
  for (const variant of VARIANTS) {
    expect(results[variant.label].requests).toBeGreaterThan(0);
  }
});
