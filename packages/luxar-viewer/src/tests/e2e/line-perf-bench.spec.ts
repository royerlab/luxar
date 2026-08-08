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
 *   - A warmed post-settle one-shot frame interval (`postSettleFrameMs`,
 *     measured separately) — NOT a cold first render; see the field's
 *     doc comment
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
// Dataset-server origin — shared with the gsplat and perf-tracking benches
// so every spec honors the port-parameterized perf config (see
// playwright.perf.config.ts: foreign servers squatting :9000 would
// otherwise skip every scenario as "dataset not reachable", including
// synthetic ones gated on their bootstrap URL).
import { PERF_DATA_BASE as DATA_BASE } from './perf-data-base';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = path.resolve(__dirname, '../../..');

function currentCommitSha(): string {
  try {
    return execSync('git rev-parse --short HEAD', { cwd: VIEWER_ROOT }).toString().trim();
  } catch {
    return 'unknown';
  }
}

/**
 * Each scenario is either a zarr URL (real-data path) or a synthetic
 * spec (`type: 'synthetic-lines'`) generated entirely in JS. The
 * synthetic path uses `__luxarDebug.injectSyntheticScene(...)` to
 * push a 10 M-segment random-walk into the live scene — exercises
 * the bandwidth-bound regime the production bench (lines_basic
 * 305 segs, zebrahub-hifi 4.4 M segs) can't reach on its own.
 */
type ScenarioSpec =
  | {
      type: 'zarr';
      id: string;
      label: string;
      url: string;
    }
  | {
      type: 'synthetic-lines';
      id: string;
      label: string;
      /** Lightweight zarr URL used to bootstrap the viewer before injection. */
      bootstrapUrl: string;
      count: number;
      /** Per-endpoint width for every segment (default 1.0). */
      width?: number;
      /** Random-walk step as a fraction of bounds (default 0.01). */
      stepScale?: number;
      /** Max per-step turning angle (radians) — smooth-walk variant. */
      turnAngle?: number;
    };

const SCENARIOS: ScenarioSpec[] = [
  {
    type: 'zarr',
    id: 'lines-basic',
    label: 'lines_basic_example.luxar.zarr (small, always present)',
    url: `${DATA_BASE}/datasets/examples/lines_basic_example.luxar.zarr`,
  },
  {
    type: 'zarr',
    id: 'lines-zebrahub-hifi',
    label: 'zebrahub_velocity_streamlines_hifi.luxar.zarr (large)',
    url: `${DATA_BASE}/datasets/demos/zebrahub_velocity_streamlines_hifi.luxar.zarr`,
  },
  {
    type: 'synthetic-lines',
    id: 'synthetic-lines-10M',
    label: 'synthetic random-walk lines, 10 M segments (bandwidth bound)',
    bootstrapUrl: `${DATA_BASE}/datasets/examples/lines_basic_example.luxar.zarr`,
    count: 10_000_000,
  },
  // The two #1352 fill-regime scenarios: segments much WIDER than they
  // are long (drawn half-width = 2 × width texel = 6 world units vs
  // ~1-unit steps), which is the worst case for footprint-area cost.
  // The SMOOTH variant (gentle ≤ ~14°/step turns) is the realistic
  // thick-streamline proxy; the NOISE variant turns ~90° at every
  // vertex — adversarial for any join-aware renderer, and measured 4-5×
  // more expensive than smooth on the G0 spike's volumetric primitive.
  {
    type: 'synthetic-lines',
    id: 'synthetic-lines-thick-smooth',
    label: 'synthetic thick smooth curves, 2 M segments (fill bound)',
    bootstrapUrl: `${DATA_BASE}/datasets/examples/lines_basic_example.luxar.zarr`,
    count: 2_000_000,
    width: 3.0,
    turnAngle: 0.25,
  },
  {
    type: 'synthetic-lines',
    id: 'synthetic-lines-thick-noise',
    label: 'synthetic thick 90°-noise walk, 2 M segments (fill bound, adversarial joins)',
    bootstrapUrl: `${DATA_BASE}/datasets/examples/lines_basic_example.luxar.zarr`,
    count: 2_000_000,
    width: 3.0,
  },
];

const BACKENDS = ['webgl', 'webgpu'] as const;
type Backend = (typeof BACKENDS)[number];

/**
 * Line-primitive axis (#1352): comma-separated list of `?linePrimitive=`
 * values to cross with every scenario × backend, e.g.
 * `LUXAR_PERF_LINE_PRIMITIVES=default,volumetric` for the quad-vs-
 * volumetric A/B. The sentinel `default` omits the URL parameter
 * entirely (today's shipping primitive), so the axis is a no-op until a
 * toggle exists — and stays harmless if one never does. Non-default
 * primitives are baked into the scenarioId (`<id>-<primitive>`) so
 * `perf-diff.mjs` keys both arms separately.
 */
const LINE_PRIMITIVES = (process.env.LUXAR_PERF_LINE_PRIMITIVES ?? 'default')
  .split(',')
  .map((p) => p.trim())
  .filter((p) => p.length > 0);

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
  /**
   * The renderer *surface* in use. 'webgl2' = `THREE.WebGLRenderer`;
   * 'webgpu' = `WebGPURenderer` (regardless of which backend it
   * dispatches through). See {@link isWebGLBackend} to distinguish
   * native WebGPU from WebGPURenderer's WebGL2 fallback.
   */
  actualApi: string | null;
  /**
   * True when `apiSurface === 'webgpu'` but WebGPURenderer is
   * dispatching through its internal WebGL2 backend — either because
   * the host has no real WebGPU adapter or because
   * `?webgpu-force-webgl` is set. Consumers benchmarking native
   * WebGPU specifically should discount runs where this is true.
   */
  isWebGLBackend: boolean;
  visibleSegments: number;
  frameMs: FrameStats | null;
  /**
   * A WARMED, post-settle one-shot frame interval (ms): `renderOnce()`
   * to the next animation frame, sampled AFTER navigation/injection has
   * already rendered and settled (same contract as the gsplat bench).
   * NOT a cold first render — it does not enclose upload, material/
   * pipeline compilation, or the initial draw. Null when the scenario
   * is skipped.
   */
  postSettleFrameMs: number | null;
  /** Renderer frame counter sampled just before the post-settle
   *  measurement — > 0 proves the metric is a warmed post-settle
   *  frame, not a cold first render (same probe as the gsplat bench).
   *  Null when the counter is unavailable: `WebGPURenderer` keeps a
   *  top-level `info.frame` that counts internal animation ticks, not
   *  completed draws, so only the WebGL surface's monotonic
   *  `info.render.frame` is honest evidence here. */
  renderedFramesBefore?: number | null;
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
 * surface, segment count, JS frame-time distribution, a warmed
 * post-settle one-shot frame interval, and optional GPU timestamps.
 */
async function measureScenario(
  page: Page,
  scn: ScenarioSpec,
  backend: Backend,
  primitive = 'default'
): Promise<ScenarioResult> {
  const scenarioId = primitive === 'default' ? scn.id : `${scn.id}-${primitive}`;
  const scenarioLabel =
    primitive === 'default' ? scn.label : `${scn.label} [linePrimitive=${primitive}]`;
  const primitiveParam = primitive === 'default' ? '' : `&linePrimitive=${primitive}`;
  const notes: string[] = [];
  // `?perf-timestamp` opts WebGPURenderer into `{trackTimestamp: true}`
  // so we can read per-frame GPU duration via
  // `renderer.resolveTimestampsAsync('render')` below. On WebGL or on a
  // WebGPU adapter without the `timestamp-query` feature, the renderer
  // silently returns the last cached value (0 for the first frame),
  // and the bench falls back to JS-only timing.
  //
  // Heavy datasets (millions of segments) can exceed the default 120s
  // navigation timeout when re-loading after a prior scenario. Raise
  // it generously — a real failure will still surface as the
  // measureScenario try/catch upstream.
  const navUrl =
    scn.type === 'zarr'
      ? `/?src=${scn.url}&renderer=${backend}&debug&perf-timestamp${primitiveParam}`
      : `/?src=${scn.bootstrapUrl}&renderer=${backend}&debug&perf-timestamp${primitiveParam}`;
  await page.goto(navUrl, { timeout: 300_000 });
  await waitForLuxarReady(page, 120_000);

  // Synthetic scenarios: after the bootstrap zarr finishes loading,
  // inject a giant random-walk mesh via the debug API and wait for
  // the first render to absorb it before sampling.
  //
  // Before injection, hide the bootstrap zarr's line nodes so they
  // don't contribute to the rendered workload — the segment-count
  // probe (further down) already filters by `userData.synthetic`,
  // but without this `visible = false` pass the bootstrap geometry
  // is still drawn each frame and skews the timing numbers. For the
  // 10M scenario the bootstrap is small enough to be in the noise,
  // but cheap to fix and makes smaller synthetic counts meaningful.
  if (scn.type === 'synthetic-lines') {
    await page.evaluate(
      async (spec: { count: number; width?: number; stepScale?: number; turnAngle?: number }) => {
        const dbg = (
          window as unknown as {
            __luxarDebug?: {
              injectSyntheticScene?: (spec: {
                type: 'lines';
                count: number;
                width?: number;
                stepScale?: number;
                turnAngle?: number;
              }) => Promise<unknown>;
              app?: {
                sceneManager?: {
                  scene?: {
                    traverse?: (cb: (o: unknown) => void) => void;
                  };
                };
              };
            };
          }
        ).__luxarDebug;
        if (!dbg?.injectSyntheticScene) {
          throw new Error(
            'synthetic scenario requires __luxarDebug.injectSyntheticScene (added in F2)'
          );
        }
        // Hide every existing line node BEFORE injection so the
        // synthetic mesh is the only line geometry rendered. Hiding
        // (vs `scene.remove`) keeps the loader's bookkeeping intact —
        // the bootstrap dataset still owns its uploaded buffers, just
        // doesn't render.
        dbg?.app?.sceneManager?.scene?.traverse?.((obj: unknown) => {
          const o = obj as {
            userData?: { nodeType?: string; synthetic?: boolean };
            visible?: boolean;
          };
          if (o.userData?.nodeType === 'lines' && o.userData?.synthetic !== true) {
            o.visible = false;
          }
        });
        await dbg.injectSyntheticScene({ type: 'lines', ...spec });
        // Yield one rAF so the renderer has a chance to upload the
        // attribute buffers before the bench's first measurement frame.
        await new Promise<void>((r) => requestAnimationFrame(() => r()));
      },
      { count: scn.count, width: scn.width, stepScale: scn.stepScale, turnAngle: scn.turnAngle }
    );
  }

  // Probe the active backend and visible segment count up front so a
  // silent fallback (WebGPU → WebGL on unsupported hardware) or an
  // empty scene (nD nav needed) shows up in the output rather than
  // silently corrupting numbers.
  //
  // `apiSurface` distinguishes the renderer *class* (WebGLRenderer vs
  // WebGPURenderer), but a WebGPURenderer can be running its internal
  // WebGL2 fallback backend (real WebGPU adapter unavailable, or
  // `?webgpu-force-webgl`). For benchmarking native-WebGPU performance
  // specifically, we also surface `isWebGLBackend` so a consumer can
  // discount fallback runs.
  const probe = await page.evaluate((onlySynthetic: boolean) => {
    const dbg = (
      window as unknown as {
        __luxarDebug?: {
          app?: {
            sceneManager?: {
              capabilities?: { apiSurface?: string };
              scene?: unknown;
            };
          };
          renderer?: { backend?: { isWebGLBackend?: boolean } };
        };
      }
    ).__luxarDebug;
    const api = dbg?.app?.sceneManager?.capabilities?.apiSurface ?? null;
    const isWebGLBackend = dbg?.renderer?.backend?.isWebGLBackend === true;
    let visibleSegments = 0;
    const scene = dbg?.app?.sceneManager?.scene as
      { traverse?: (cb: (o: unknown) => void) => void } | undefined;
    scene?.traverse?.((obj: unknown) => {
      const o = obj as {
        userData?: { nodeType?: string; synthetic?: boolean };
        geometry?: { instanceCount?: number };
      };
      if (
        o.userData?.nodeType === 'lines' &&
        typeof o.geometry?.instanceCount === 'number' &&
        // For synthetic scenarios, only count the injected mesh
        // (`userData.synthetic === true`) — the bootstrap zarr also
        // contributes line nodes, and conflating them would
        // overstate the workload size in the JSON.
        (!onlySynthetic || o.userData.synthetic === true)
      ) {
        visibleSegments += o.geometry.instanceCount;
      }
    });
    return { api, isWebGLBackend, visibleSegments };
  }, scn.type === 'synthetic-lines');

  if (probe.visibleSegments === 0) {
    notes.push('visibleSegments=0 — dataset may need nD navigation to a populated slice');
    return {
      scenarioId,
      scenarioLabel,
      backend,
      actualApi: probe.api,
      isWebGLBackend: probe.isWebGLBackend,
      visibleSegments: 0,
      frameMs: null,
      postSettleFrameMs: null,
      gpu: { supported: false, count: 0, medianMs: null, p95Ms: null },
      notes,
      skipped: true,
      skipReason: 'no visible segments',
    };
  }

  // Warmed post-settle one-shot frame measurement, separate from the
  // steady-state frame loop. NOT a cold first render: navigation (and,
  // for synthetic scenarios, injection) has already uploaded buffers,
  // compiled the material/pipeline, and drawn — so this does NOT
  // capture material-build / pipeline-compile cost.
  const postSettle = await page.evaluate(async () => {
    const debug = (
      window as unknown as {
        __luxarDebug: {
          renderOnce: () => void;
          renderer?: { info?: { render?: { frame?: number } } };
        };
      }
    ).__luxarDebug;
    // Frame-counter evidence (issue #706): sample the renderer's frame
    // counter before the timer starts. Only the WebGL surface exposes
    // the monotonic `info.render.frame`; on WebGPURenderer this reads
    // undefined and the probe records null (see the field doc).
    const framesBefore = debug.renderer?.info?.render?.frame;
    const renderedFramesBefore = typeof framesBefore === 'number' ? framesBefore : null;
    const t0 = performance.now();
    debug.renderOnce();
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
    return { ms: performance.now() - t0, renderedFramesBefore };
  });

  // Steady-state frame timing loop. Warmup is the lesser of N frames
  // and a wall-clock cap so slow scenes still collect samples before
  // the watchdog. Watchdog generously sized so a slow first-frame
  // doesn't kill the run.
  //
  // GPU timestamps: every N frames we call
  // `renderer.resolveTimestampsAsync('render')`. Three's
  // `*TimestampQueryPool.processQueries()` (`three/src/renderers/
  // webgpu/utils/WebGPUTimestampQueryPool.js` and the WebGL2 fallback)
  // returns the duration of the *most recent* frame in the pool's
  // current batch, NOT the sum across `frames` since the last
  // resolve — see the inline `// Return the total duration of the
  // last frame` comment at line 211 of that file. So each resolve
  // contributes a single per-frame sample, regardless of how many
  // frames elapsed in between. WebGL renderers and WebGPU drivers
  // without `timestamp-query` return 0 / lastValue; the bench
  // treats those as unsupported and falls back to JS-only timing.
  const timing = await page.evaluate(
    async (cfg: {
      windowMs: number;
      warmupFrames: number;
      warmupMaxMs: number;
      minFrames: number;
    }) => {
      const debug = (
        window as unknown as {
          __luxarDebug: {
            renderOnce: () => void;
            renderer?: {
              resolveTimestampsAsync?: (type: string) => Promise<number>;
              backend?: { trackTimestamp?: boolean };
            };
          };
        }
      ).__luxarDebug;
      const dts: number[] = [];
      const gpuPerFrameMs: number[] = [];
      let frames = 0;
      let lastTime = performance.now();
      const start = lastTime;
      let collectingStart: number | null = null;
      let framesSinceResolve = 0;

      // Resolve cadence: 16 frames per resolve. Frequent enough for
      // good per-batch averages, infrequent enough that the
      // resolveAsync overhead doesn't dominate the loop.
      const GPU_RESOLVE_EVERY = 16;
      const supportsTimestamp =
        typeof debug.renderer?.resolveTimestampsAsync === 'function' &&
        debug.renderer?.backend?.trackTimestamp === true;

      return new Promise<{
        frameDtMs: number[];
        gpuPerFrameMs: number[];
        totalMs: number;
        supportsTimestamp: boolean;
      }>((resolve) => {
        const watchdog = window.setTimeout(
          () =>
            resolve({
              frameDtMs: dts,
              gpuPerFrameMs,
              totalMs: performance.now() - start,
              supportsTimestamp,
            }),
          // Cap at 30s even for very slow scenes; better to bail than
          // hang the run indefinitely.
          Math.max(cfg.windowMs * 3, 30_000)
        );
        const tick = async () => {
          const now = performance.now();
          const elapsedSinceStart = now - start;
          // Frames start at 0; `>=` so `warmupFrames=5` skips exactly 5 warmup
          // frames (frames 0..4) and the 6th frame is the first sampled.
          const warmupDone = frames >= cfg.warmupFrames || elapsedSinceStart > cfg.warmupMaxMs;
          if (warmupDone) {
            if (collectingStart === null) collectingStart = now;
            dts.push(now - lastTime);

            // Resolve GPU timestamps every N frames. The pool returns
            // the last-frame duration only (see header comment), so
            // each resolve contributes exactly one per-frame sample —
            // do NOT divide by the batch size. The 16-frame cadence is
            // a cost/quality tradeoff: it keeps `resolveAsync` overhead
            // out of the inner loop without thinning the GPU sample
            // count below useful percentile resolution.
            framesSinceResolve++;
            if (supportsTimestamp && framesSinceResolve >= GPU_RESOLVE_EVERY) {
              try {
                const gpuFrameMs = await debug.renderer!.resolveTimestampsAsync!('render');
                if (
                  typeof gpuFrameMs === 'number' &&
                  Number.isFinite(gpuFrameMs) &&
                  gpuFrameMs > 0
                ) {
                  gpuPerFrameMs.push(gpuFrameMs);
                }
              } catch {
                // Drop the sample on transient failure; resolveAsync can
                // race with frame submission. Falls through to the next
                // batch.
              }
              framesSinceResolve = 0;
            }
          }
          lastTime = now;
          frames++;
          const collectingElapsed = collectingStart === null ? 0 : now - collectingStart;
          if (collectingElapsed < cfg.windowMs || dts.length < cfg.minFrames) {
            debug.renderOnce();
            requestAnimationFrame(() => void tick());
          } else {
            window.clearTimeout(watchdog);
            resolve({
              frameDtMs: dts,
              gpuPerFrameMs,
              totalMs: elapsedSinceStart,
              supportsTimestamp,
            });
          }
        };
        debug.renderOnce();
        requestAnimationFrame(() => void tick());
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

  // GPU timestamp-query support is best-effort: the renderer was
  // constructed with `trackTimestamp: true` (via `?perf-timestamp`),
  // but the feature only fires when the WebGPU adapter exposes
  // `timestamp-query`. On WebGL2 / WebGL-backed WebGPURenderer / older
  // GPUs the supportsTimestamp probe is false and we fall through to
  // JS frame timing.
  const gpuStatsInner = timing.supportsTimestamp ? statsOf(timing.gpuPerFrameMs) : null;
  const gpu: GpuStats = gpuStatsInner
    ? {
        supported: true,
        count: gpuStatsInner.count,
        medianMs: gpuStatsInner.median,
        p95Ms: gpuStatsInner.p95,
      }
    : {
        supported: false,
        count: 0,
        medianMs: null,
        p95Ms: null,
      };
  if (!timing.supportsTimestamp) {
    notes.push('GPU timestamp-query unavailable (WebGL backend or missing feature)');
  } else if (timing.gpuPerFrameMs.length === 0) {
    notes.push(
      'GPU timestamp-query enabled but no samples collected — driver may have rejected the queries'
    );
  }

  return {
    scenarioId,
    scenarioLabel,
    backend,
    actualApi: probe.api,
    isWebGLBackend: probe.isWebGLBackend,
    visibleSegments: probe.visibleSegments,
    frameMs,
    postSettleFrameMs: postSettle.ms,
    renderedFramesBefore: postSettle.renderedFramesBefore,
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
    // Synthetic scenarios still need the bootstrap zarr to be
    // reachable so the viewer can initialise before injection.
    const probeUrl = scn.type === 'zarr' ? scn.url : scn.bootstrapUrl;
    const reachable = await urlExists(probeUrl);
    if (!reachable) {
      for (const backend of BACKENDS) {
        scenarios.push({
          scenarioId: scn.id,
          scenarioLabel: scn.label,
          backend,
          actualApi: null,
          isWebGLBackend: false,
          visibleSegments: 0,
          frameMs: null,
          postSettleFrameMs: null,
          gpu: { supported: false, count: 0, medianMs: null, p95Ms: null },
          notes: [`dataset URL not reachable: ${probeUrl}`],
          skipped: true,
          skipReason: 'dataset not reachable',
        });
      }
      continue;
    }
    for (const backend of BACKENDS) {
      for (const primitive of LINE_PRIMITIVES) {
        // Resilience: huge scenes can lose the WebGPU device or trip
        // nav timeouts. A single failing scenario should not block the
        // rest of the bench or, more importantly, the JSON write at the
        // end. Wrap each measurement in try/catch and synthesize a
        // skipped result on failure.
        let result: ScenarioResult;
        try {
          result = await measureScenario(page, scn, backend, primitive);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          // Truncate the stack trace — long Playwright error messages
          // make the JSON unwieldy.
          const shortMsg = msg.split('\n').slice(0, 3).join(' | ');
          result = {
            scenarioId: primitive === 'default' ? scn.id : `${scn.id}-${primitive}`,
            scenarioLabel: scn.label,
            backend,
            actualApi: null,
            isWebGLBackend: false,
            visibleSegments: 0,
            frameMs: null,
            postSettleFrameMs: null,
            gpu: { supported: false, count: 0, medianMs: null, p95Ms: null },
            notes: [`measurement threw: ${shortMsg}`],
            skipped: true,
            skipReason: 'measurement error',
          };
        }
        scenarios.push(result);

        const fm = result.frameMs;
        const summary = result.skipped
          ? `SKIP (${result.skipReason})`
          : fm
            ? `median=${fm.median.toFixed(2)}ms p95=${fm.p95.toFixed(2)}ms p99=${fm.p99.toFixed(2)}ms mean=${fm.mean.toFixed(2)}ms count=${fm.count}`
            : 'no samples';
        const gpuSummary =
          result.gpu.supported && result.gpu.medianMs !== null
            ? ` gpu_median=${result.gpu.medianMs.toFixed(2)}ms (n=${result.gpu.count})`
            : '';
        console.log(
          `  [${result.scenarioId}/${backend} → ${result.actualApi ?? '?'}${
            result.isWebGLBackend ? ' (webgl-bk)' : ''
          }] segs=${result.visibleSegments} ${summary}${gpuSummary}`
        );
      }
    }
  }

  const outPath = path.join(outDir, 'results.json');
  // Merge-write: the per-SHA results.json is shared with the other
  // *-perf-bench specs (the gsplat bench merges `scenarioId`-keyed rows
  // into the same file) — a wholesale write here would clobber their
  // rows when this spec runs later in the same `pnpm test:perf:e2e`
  // invocation. Keep every existing row this run did not re-measure.
  const rowKey = (r: { id?: string; scenarioId?: string; backend?: string }): string =>
    `${r.scenarioId ?? r.id}/${r.backend}`;
  let keptRows: unknown[] = [];
  if (fs.existsSync(outPath)) {
    try {
      const prev = JSON.parse(fs.readFileSync(outPath, 'utf8')) as PerfRunResult;
      const ours = new Set(scenarios.map((s) => rowKey(s)));
      keptRows = (prev.scenarios ?? []).filter(
        (s) => !ours.has(rowKey(s as { id?: string; scenarioId?: string; backend?: string }))
      );
    } catch {
      // Corrupt/foreign file — fall back to writing just this run's rows.
    }
  }
  const output: PerfRunResult = {
    capturedAt: new Date().toISOString(),
    commit: sha,
    sampleWindowMs: SAMPLE_WINDOW_MS,
    warmupFrames: WARMUP_FRAMES,
    scenarios: [...keptRows, ...scenarios] as PerfRunResult['scenarios'],
  };
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));

  console.log(`\n📊 perf-bench results written to ${outPath}`);

  // Guard against silent total failure: per-scenario `measureScenario`
  // catches errors and records them as skipped rows so the JSON
  // preserves whatever partial work succeeded — but that also means a
  // shader-compile failure, GPU device loss, OOM, or a broken
  // synthetic injector could produce a "passing" test with zero
  // useful rows for a critical scenario.
  //
  // Pass criteria: every *reachable* scenario must have AT LEAST one
  // successful backend row. "Reachable" excludes scenarios whose
  // dataset URL was unreachable up front (urlExists() failed) — those
  // are environment skips, not failures. The stricter per-scenario
  // gate (vs the older global-zero check) is what makes the dedicated
  // perf suite meaningful: if `synthetic-lines-10M` is the
  // bandwidth-bound scenario the suite exists to measure, the test
  // must not pass when it fails on every backend.
  const isUnreachable = (s: ScenarioResult): boolean =>
    s.skipped && s.skipReason === 'dataset not reachable';
  // A scenario row's id carries a `-<primitive>` suffix on the non-default
  // arms of the LINE_PRIMITIVES axis, so ownership is prefix-based.
  const rowBelongsTo = (rowId: string, scnId: string): boolean =>
    rowId === scnId || LINE_PRIMITIVES.some((p) => p !== 'default' && rowId === `${scnId}-${p}`);
  const successfulIds = new Set(
    scenarios.filter((s) => !s.skipped && s.frameMs !== null).map((s) => s.scenarioId)
  );
  const failedScenarios = SCENARIOS.filter((scn) => {
    if (LINE_PRIMITIVES.some((p) => successfulIds.has(p === 'default' ? scn.id : `${scn.id}-${p}`)))
      return false;
    // Every row for this scenario was unreachable → environment skip,
    // not a failure.
    return !scenarios.filter((s) => rowBelongsTo(s.scenarioId, scn.id)).every(isUnreachable);
  });
  if (failedScenarios.length > 0) {
    const skipNotes = scenarios
      .filter((s) => failedScenarios.some((scn) => rowBelongsTo(s.scenarioId, scn.id)))
      .map((s) => `  - ${s.scenarioId}/${s.backend}: ${s.skipReason ?? 'unknown'}`)
      .join('\n');
    throw new Error(
      `perf-bench: ${failedScenarios.length} scenario(s) produced no successful timing on any backend ` +
        `(${failedScenarios.map((s) => s.id).join(', ')}). ` +
        `JSON still written to ${outPath} for inspection. Per-row reasons:\n${skipNotes}`
    );
  }

  // Probe (issue #706): `postSettleFrameMs` is a warmed post-settle
  // metric — where the frame counter is available (WebGL surface),
  // prove the renderer had already drawn before we sampled it, so the
  // number can never be silently mislabeled as a cold first render.
  // Checked after the JSON write so a violation still leaves its
  // diagnostic row on disk (same ordering as the gsplat bench).
  const coldProbeRows = scenarios.filter(
    (s) => !s.skipped && typeof s.renderedFramesBefore === 'number' && s.renderedFramesBefore <= 0
  );
  if (coldProbeRows.length > 0) {
    throw new Error(
      'perf-bench: post-settle frame measured before any render on ' +
        coldProbeRows.map((s) => `${s.scenarioId}/${s.backend}`).join(', ') +
        ' (renderedFramesBefore=0) — the warmed postSettleFrameMs contract is violated. ' +
        `JSON still written to ${outPath} for inspection.`
    );
  }
});
