/**
 * Line-rendering performance benchmark.
 *
 * Captures per-frame timing across a fixed set of line-heavy
 * scenarios under both the WebGL (GLSL) and WebGPU (TSL) backends.
 * Designed to be run under `playwright.perf.config.ts` against the
 * system-installed Chrome with WebGPU developer features enabled, so
 * the measurements reflect real GPU behaviour rather than software
 * fallback.
 *
 * Output: a single JSON file at
 * `perf-results/{commit-sha}/results.json`. The companion script
 * `scripts/perf-diff.mjs` compares two such files and prints a
 * Markdown delta table that goes into commit bodies.
 *
 * Per scenario we record:
 *   - JS frame time stats (median, p95, p99, mean) over a fixed
 *     sample window
 *   - First-render cost (one-shot, measured separately)
 *   - Active backend (`apiSurface`) so a silent fallback can't
 *     pollute the comparison
 *   - GPU pass time stats when `timestamp-query` is supported (best
 *     effort — many drivers don't expose it)
 *
 * @module tests/e2e/line-perf-bench.spec
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { test, type Page } from '@playwright/test';
import { waitForLuxarReady } from './helpers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = path.resolve(__dirname, '../../..');

function currentCommitSha(): string {
  try {
    return execSync('git rev-parse --short HEAD', { cwd: VIEWER_ROOT })
      .toString()
      .trim();
  } catch {
    return 'unknown';
  }
}

/**
 * Each scenario points at an existing test dataset that produces a
 * meaningful number of visible line segments. `lines_basic_example`
 * is always bundled; the zebrahub streamlines sets are present when
 * `make demo` has been run (we skip them gracefully if missing).
 */
const SCENARIOS = [
  {
    id: 'lines-basic',
    label: 'lines_basic_example.zarr (small, always present)',
    url: 'http://localhost:9000/datasets/examples/lines_basic_example.zarr',
  },
  {
    id: 'lines-zebrahub-hifi',
    label: 'zebrahub_velocity_streamlines_hifi.zarr (large)',
    url: 'http://localhost:9000/datasets/demos/zebrahub_velocity_streamlines_hifi.zarr',
  },
] as const;

const BACKENDS = ['webgl', 'webgpu'] as const;
type Backend = (typeof BACKENDS)[number];

const SAMPLE_WINDOW_MS = 3_000;
// Warmup is small + time-capped so very slow scenes (millions of
// segments at sub-30fps) don't exhaust the watchdog before we start
// collecting samples.
const WARMUP_FRAMES = 5;
const WARMUP_MAX_MS = 1_000;
const MIN_FRAMES = 30;

interface FrameStats {
  count: number;
  median: number;
  p95: number;
  p99: number;
  mean: number;
  min: number;
  max: number;
}

interface GpuStats {
  supported: boolean;
  count: number;
  medianMs: number | null;
  p95Ms: number | null;
}

interface ScenarioResult {
  scenarioId: string;
  scenarioLabel: string;
  backend: Backend;
  actualApi: string | null;
  visibleSegments: number;
  frameMs: FrameStats | null;
  firstRenderMs: number | null;
  gpu: GpuStats;
  notes: string[];
  skipped: boolean;
  skipReason?: string;
}

interface PerfRunResult {
  capturedAt: string;
  commit: string;
  sampleWindowMs: number;
  warmupFrames: number;
  scenarios: ScenarioResult[];
}

function statsOf(samples: number[]): FrameStats | null {
  if (samples.length === 0) return null;
  const sorted = samples.slice().sort((a, b) => a - b);
  const pick = (q: number): number => {
    const idx = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
    return sorted[idx];
  };
  return {
    count: sorted.length,
    median: sorted[Math.floor(sorted.length / 2)],
    p95: pick(0.95),
    p99: pick(0.99),
    mean: sorted.reduce((s, v) => s + v, 0) / sorted.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

async function urlExists(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    return r.ok;
  } catch {
    return false;
  }
}

/**
 * Measure one scenario under one backend. Captures the active API
 * surface, segment count, JS frame-time distribution, first-render
 * cost, and optional GPU timestamps.
 */
async function measureScenario(
  page: Page,
  scenarioId: string,
  scenarioLabel: string,
  url: string,
  backend: Backend
): Promise<ScenarioResult> {
  const notes: string[] = [];
  await page.goto(`/?src=${url}&renderer=${backend}&debug`);
  await waitForLuxarReady(page);

  // Probe the active backend and visible segment count up front so a
  // silent fallback (WebGPU → WebGL on unsupported hardware) or an
  // empty scene (nD nav needed) shows up in the output rather than
  // silently corrupting numbers.
  const probe = await page.evaluate(() => {
    const dbg = (
      window as unknown as {
        __luxarDebug?: {
          app?: {
            sceneManager?: {
              capabilities?: { apiSurface?: string };
              scene?: unknown;
            };
          };
        };
      }
    ).__luxarDebug;
    const api = dbg?.app?.sceneManager?.capabilities?.apiSurface ?? null;
    let visibleSegments = 0;
    const scene = dbg?.app?.sceneManager?.scene as
      | { traverse?: (cb: (o: unknown) => void) => void }
      | undefined;
    scene?.traverse?.((obj: unknown) => {
      const o = obj as {
        userData?: { nodeType?: string };
        geometry?: { instanceCount?: number };
      };
      if (
        o.userData?.nodeType === 'lines' &&
        typeof o.geometry?.instanceCount === 'number'
      ) {
        visibleSegments += o.geometry.instanceCount;
      }
    });
    return { api, visibleSegments };
  });

  if (probe.visibleSegments === 0) {
    notes.push(
      'visibleSegments=0 — dataset may need nD navigation to a populated slice'
    );
    return {
      scenarioId,
      scenarioLabel,
      backend,
      actualApi: probe.api,
      visibleSegments: 0,
      frameMs: null,
      firstRenderMs: null,
      gpu: { supported: false, count: 0, medianMs: null, p95Ms: null },
      notes,
      skipped: true,
      skipReason: 'no visible segments',
    };
  }

  // One-shot first-render measurement — useful for material-build /
  // pipeline-compile cost, separate from the steady-state frame loop.
  const firstRenderMs = await page.evaluate(async () => {
    const debug = (window as unknown as { __luxarDebug: { renderOnce: () => void } })
      .__luxarDebug;
    const t0 = performance.now();
    debug.renderOnce();
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
    return performance.now() - t0;
  });

  // Steady-state frame timing loop. Warmup is the lesser of N frames
  // and a wall-clock cap so slow scenes still collect samples before
  // the watchdog. Watchdog generously sized so a slow first-frame
  // doesn't kill the run.
  const timing = await page.evaluate(
    async (cfg: {
      windowMs: number;
      warmupFrames: number;
      warmupMaxMs: number;
      minFrames: number;
    }) => {
      const debug = (window as unknown as { __luxarDebug: { renderOnce: () => void } })
        .__luxarDebug;
      const dts: number[] = [];
      let frames = 0;
      let lastTime = performance.now();
      const start = lastTime;
      let collectingStart: number | null = null;

      return new Promise<{ frameDtMs: number[]; totalMs: number }>((resolve) => {
        const watchdog = window.setTimeout(
          () => resolve({ frameDtMs: dts, totalMs: performance.now() - start }),
          // Cap at 30s even for very slow scenes; better to bail than
          // hang the run indefinitely.
          Math.max(cfg.windowMs * 3, 30_000)
        );
        const tick = () => {
          const now = performance.now();
          const elapsedSinceStart = now - start;
          const warmupDone =
            frames > cfg.warmupFrames || elapsedSinceStart > cfg.warmupMaxMs;
          if (warmupDone) {
            if (collectingStart === null) collectingStart = now;
            dts.push(now - lastTime);
          }
          lastTime = now;
          frames++;
          const collectingElapsed = collectingStart === null ? 0 : now - collectingStart;
          if (collectingElapsed < cfg.windowMs || dts.length < cfg.minFrames) {
            debug.renderOnce();
            requestAnimationFrame(tick);
          } else {
            window.clearTimeout(watchdog);
            resolve({ frameDtMs: dts, totalMs: elapsedSinceStart });
          }
        };
        debug.renderOnce();
        requestAnimationFrame(tick);
      });
    },
    {
      windowMs: SAMPLE_WINDOW_MS,
      warmupFrames: WARMUP_FRAMES,
      warmupMaxMs: WARMUP_MAX_MS,
      minFrames: MIN_FRAMES,
    }
  );

  const frameMs = statsOf(timing.frameDtMs);

  // GPU timestamp-query support is best-effort and varies by driver.
  // We probe for the feature; if the device wasn't requested with it,
  // we can't measure GPU time from this side without renderer-level
  // hooks. Mark unsupported for now — wired in later if needed.
  const gpu: GpuStats = {
    supported: false,
    count: 0,
    medianMs: null,
    p95Ms: null,
  };
  notes.push(
    'GPU timestamp-query not wired through renderer; JS frame time only'
  );

  return {
    scenarioId,
    scenarioLabel,
    backend,
    actualApi: probe.api,
    visibleSegments: probe.visibleSegments,
    frameMs,
    firstRenderMs,
    gpu,
    notes,
    skipped: false,
  };
}

test('line perf bench — JS frame timing across backends', async ({ page }) => {
  test.setTimeout(900_000);

  const sha = currentCommitSha();
  const outDir = path.join(VIEWER_ROOT, 'perf-results', sha);
  fs.mkdirSync(outDir, { recursive: true });

  const scenarios: ScenarioResult[] = [];
  for (const scn of SCENARIOS) {
    const reachable = await urlExists(scn.url);
    if (!reachable) {
      for (const backend of BACKENDS) {
        scenarios.push({
          scenarioId: scn.id,
          scenarioLabel: scn.label,
          backend,
          actualApi: null,
          visibleSegments: 0,
          frameMs: null,
          firstRenderMs: null,
          gpu: { supported: false, count: 0, medianMs: null, p95Ms: null },
          notes: [`dataset URL not reachable: ${scn.url}`],
          skipped: true,
          skipReason: 'dataset not reachable',
        });
      }
      continue;
    }
    for (const backend of BACKENDS) {
      const result = await measureScenario(page, scn.id, scn.label, scn.url, backend);
      scenarios.push(result);

      const fm = result.frameMs;
      const summary = result.skipped
        ? `SKIP (${result.skipReason})`
        : fm
          ? `median=${fm.median.toFixed(2)}ms p95=${fm.p95.toFixed(2)}ms p99=${fm.p99.toFixed(2)}ms mean=${fm.mean.toFixed(2)}ms count=${fm.count}`
          : 'no samples';
      // eslint-disable-next-line no-console
      console.log(
        `  [${scn.id}/${backend} → ${result.actualApi ?? '?'}] segs=${result.visibleSegments} ${summary}`
      );
    }
  }

  const output: PerfRunResult = {
    capturedAt: new Date().toISOString(),
    commit: sha,
    sampleWindowMs: SAMPLE_WINDOW_MS,
    warmupFrames: WARMUP_FRAMES,
    scenarios,
  };

  const outPath = path.join(outDir, 'results.json');
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));

  // eslint-disable-next-line no-console
  console.log(`\n📊 perf-bench results written to ${outPath}`);
});
