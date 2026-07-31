/**
 * GSplat / points rendering performance benchmark — the gsplat sibling
 * of `line-perf-bench.spec.ts`.
 *
 * Captures per-frame timing for depth-sorted ('normal' blending)
 * gsplat and points scenes under the WebGL backend while the camera
 * ORBITS continuously (~30°/s), so depth-sort re-sorts fire throughout
 * the sampling window and the numbers reflect the sorted steady state
 * rather than a static-camera idle loop.
 *
 * Designed to be run under `playwright.perf.config.ts` (picked up via
 * its `testMatch: /.*perf-bench\.spec\.ts$/`).
 *
 * Output: MERGED into the same JSON file the line bench writes,
 * `perf-results/{commit-sha}/results.json` (rows keyed by
 * `scenarioId/backend`; existing rows from other benches are
 * preserved). `scripts/perf-diff.mjs` compares two such files.
 *
 * Per scenario we record:
 *   - JS frame time stats (median, p95, p99, mean, min, max) over a
 *     fixed sample window with a continuous orbit running
 *   - First-render cost (one-shot, measured separately)
 *   - `elementCount` — the capacity-CLAMPED drawn count reported by
 *     the injector / summed from `visibleSplatCount` (never the
 *     requested count; maxTextureSize caps a node at ≈16.8M splats)
 *   - The UNMASKED GPU renderer string (`WEBGL_debug_renderer_info`)
 *     — the SwiftShader guard for remote/CI runs — plus a derived
 *     `softwareRenderer` boolean so a software-rasterized run is
 *     obvious in the JSON instead of looking like a 1000x regression
 *   - Depth-sort dispatch stats sampled per rAF from
 *     `getSceneLoader().getProfiler().getDepthSortTimings()`: final
 *     sort count plus the series of `lastMs` values observed when the
 *     count increments (→ sortCount, sort-latency median/p95).
 *     Metadata MAY also carry numeric `kernelMs`/`boundaryMs`/
 *     `queueMs` (being added by a parallel change) — recorded when
 *     present, absence tolerated.
 *   - L8 GATE PROBE (10M scenario only): every sampled frame is
 *     classified 'sorting-adjacent' (an ordering apply landed within
 *     ±1 frame, detected via a depth-sort count increment) vs
 *     'idle-orbit'; the per-class p99s are reported separately as
 *     `sortAdjacentP99Ms` / `idleOrbitP99Ms`. This gate decides a
 *     later optimization lever.
 *
 * No hard assertions on timings (record-only, like the line bench) —
 * only structural sanity: reachable scenarios produced samples, at
 * least MIN_FRAMES frames, and a non-empty GPU renderer string.
 *
 * @module tests/e2e/gsplat-perf-bench.spec
 */

import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { test, expect, type Page } from '@playwright/test';
import { waitForLuxarReady } from './helpers';

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
 * Scenario shapes:
 *  - 'synthetic': bootstrap a tiny zarr, then push a clustered
 *    points/gsplats cloud into the live scene via
 *    `__luxarDebug.injectSyntheticScene(...)` and orbit-sample.
 *  - 'zarr-orbit': load a real dataset and orbit-sample it.
 *  - 'zarr-ladder': load a real progressive (additive-ladder) dataset
 *    and measure (a) wall ms until the ladder stops growing
 *    (`visibleSplatCount` plateau) and (b) frame stats DURING the
 *    progressive-load window.
 */
type ScenarioSpec =
  | {
      kind: 'synthetic';
      id: string;
      label: string;
      /** Lightweight zarr URL used to bootstrap the viewer before injection. */
      bootstrapUrl: string;
      geometry: 'points' | 'gsplats';
      count: number;
      seed: number;
      blending: string;
      /** Attach the L8 gate probe fields to this scenario's row. */
      l8Probe?: boolean;
    }
  | {
      kind: 'zarr-orbit';
      id: string;
      label: string;
      url: string;
    }
  | {
      kind: 'zarr-ladder';
      id: string;
      label: string;
      url: string;
    };

/**
 * Origin serving the repo root (the perf config boots
 * `python3 -m http.server 9000` there). Overridable via
 * `LUXAR_PERF_DATA_BASE` for the case where port 9000 is already held
 * by a foreign document root — Playwright's `reuseExistingServer`
 * would otherwise silently reuse it and 404 every dataset.
 */
const DATA_BASE =
  process.env.LUXAR_PERF_DATA_BASE ??
  `http://localhost:${process.env.LUXAR_PERF_DATA_PORT ?? 9000}`;

const BOOTSTRAP_URL = `${DATA_BASE}/datasets/examples/lines_basic_example.luxar.zarr`;

const SCENARIOS: ScenarioSpec[] = [
  {
    kind: 'synthetic',
    id: 'synthetic-gsplats-1M-orbit',
    label: 'synthetic clustered gsplats, 1 M splats, normal blending, 30°/s orbit',
    bootstrapUrl: BOOTSTRAP_URL,
    geometry: 'gsplats',
    count: 1_000_000,
    seed: 42,
    blending: 'normal',
  },
  {
    kind: 'synthetic',
    id: 'synthetic-gsplats-5M-orbit',
    label: 'synthetic clustered gsplats, 5 M splats, normal blending, 30°/s orbit',
    bootstrapUrl: BOOTSTRAP_URL,
    geometry: 'gsplats',
    count: 5_000_000,
    seed: 42,
    blending: 'normal',
  },
  {
    kind: 'synthetic',
    id: 'synthetic-gsplats-10M-orbit',
    label: 'synthetic clustered gsplats, 10 M splats, normal blending, 30°/s orbit (L8 gate probe)',
    bootstrapUrl: BOOTSTRAP_URL,
    geometry: 'gsplats',
    count: 10_000_000,
    seed: 42,
    blending: 'normal',
    l8Probe: true,
  },
  {
    kind: 'synthetic',
    id: 'synthetic-points-5M-orbit',
    label: 'synthetic clustered points, 5 M points, normal blending, 30°/s orbit (symmetry check)',
    bootstrapUrl: BOOTSTRAP_URL,
    geometry: 'points',
    count: 5_000_000,
    seed: 42,
    blending: 'normal',
  },
  {
    kind: 'zarr-ladder',
    id: 'visible-human-ladder-load',
    label: 'gsplats_3d_visible_human_head.luxar.zarr (1.91 M, single node, volumetric ladder)',
    url: `${DATA_BASE}/datasets/demos/gsplats_3d_visible_human_head.luxar.zarr`,
  },
  {
    kind: 'zarr-orbit',
    id: 'matrixcity-orbit',
    label: 'gsplats_interop_sog_matrixcity.luxar.zarr (168-leaf tiles), 30°/s orbit',
    url: `${DATA_BASE}/datasets/demos/gsplats_interop_sog_matrixcity.luxar.zarr`,
  },
];

/**
 * WebGL only. The gsplat depth-sort subsystem is what this bench
 * measures and its production path is the WebGL renderer; the WebGPU
 * axis is covered by the line bench and can be added here once the
 * gsplat TSL path is on the perf agenda.
 */
const BACKEND = 'webgl' as const;

const SAMPLE_WINDOW_MS = 3_000;
const WARMUP_FRAMES = 5;
const WARMUP_MAX_MS = 1_000;
const MIN_FRAMES = 30;
/** Continuous orbit rate during sampling — slow, ~30°/s. */
const ORBIT_DEG_PER_SEC = 30;
/**
 * Post-injection settle before sampling: injection resolves with
 * identity ordering and the first back-to-front sort lands async
 * ~1–2 frames later (worker cold spawn is slower on the first
 * injection) — give it a beat so warmup doesn't eat the sort landing.
 */
const INJECTION_SETTLE_MS = 250;
/** Ladder scenario: growth is "complete" after this long with no commit. */
const LADDER_STABLE_MS = 2_000;
/** Ladder scenario: hard cap on the polling loop. */
const LADDER_MAX_MS = 180_000;

interface FrameStats {
  count: number;
  median: number;
  p95: number;
  p99: number;
  mean: number;
  min: number;
  max: number;
}

interface DepthSortStats {
  /** Final profiler depth-sort count at the end of the window. */
  finalCount: number;
  /** Number of count increments observed during the window. */
  sortCount: number;
  /**
   * END-TO-END sort latency (the profiler's `lastMs`): dispatch →
   * ordering applied, so it INCLUDES the worker queue wait and
   * therefore at least one frame of main-thread occupancy. On a slow
   * frame budget it tracks the frame time, not the sort cost — use
   * {@link kernelMsMedian} for the kernel itself.
   */
  sortLatencyMedianMs: number | null;
  sortLatencyP95Ms: number | null;
  /**
   * Optional worker-stage breakdown medians from the profiler's timing
   * metadata (added by a parallel change; absence tolerated).
   * `kernelMs` = the sort kernel; `queueMs` = wait before it ran;
   * `boundaryMs` = the transfer/apply boundary.
   */
  kernelMsMedian?: number;
  boundaryMsMedian?: number;
  queueMsMedian?: number;
}

interface LadderStats {
  /** ms from navigation start to the last observed element-count commit. */
  wallMsToLadderComplete: number;
  /** False when the ladder finished before our polling loop started —
   *  wallMs is then a lower bound, not a measurement. */
  observedGrowth: boolean;
  /** Frame stats DURING the progressive-load window (start → last commit). */
  loadWindowFrameMs: FrameStats | null;
}

interface ScenarioResult {
  scenarioId: string;
  scenarioLabel: string;
  backend: typeof BACKEND;
  actualApi: string | null;
  isWebGLBackend: boolean;
  /**
   * Kept for `perf-diff.mjs` column compatibility with the line bench
   * — mirrors {@link elementCount}.
   */
  visibleSegments: number;
  /** Capacity-clamped drawn element count (splats / points). */
  elementCount: number;
  /** UNMASKED_RENDERER_WEBGL string — SwiftShader guard for remote runs. */
  gpuRenderer: string;
  /**
   * True when {@link gpuRenderer} names a software rasterizer
   * (SwiftShader / llvmpipe / Mesa softpipe / "Software"). Such a run's
   * absolute timings are NOT comparable to a GPU run — headless Chrome
   * with `--use-gl=egl` frequently lands here. Consumers must discount
   * these rows; the structural frame-count gate is relaxed for them
   * (a 1 M-splat frame can take seconds in software).
   */
  softwareRenderer: boolean;
  frameMs: FrameStats | null;
  firstRenderMs: number | null;
  depthSort: DepthSortStats | null;
  /** L8 gate probe (10M scenario only): p99 of frames within ±1 frame
   *  of an ordering apply. */
  sortAdjacentP99Ms?: number | null;
  /** L8 gate probe (10M scenario only): p99 of the remaining frames. */
  idleOrbitP99Ms?: number | null;
  /** Ladder-load metrics (visible-human scenario only). */
  ladder?: LadderStats;
  notes: string[];
  skipped: boolean;
  skipReason?: string;
}

interface PerfRunResult {
  capturedAt: string;
  commit: string;
  sampleWindowMs: number;
  warmupFrames: number;
  scenarios: unknown[];
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

function p99Of(samples: number[]): number | null {
  const s = statsOf(samples);
  return s ? s.p99 : null;
}

async function urlExists(url: string): Promise<boolean> {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    return r.ok;
  } catch {
    return false;
  }
}

function makeSkippedResult(
  scn: ScenarioSpec,
  reason: string,
  notes: string[] = []
): ScenarioResult {
  return {
    scenarioId: scn.id,
    scenarioLabel: scn.label,
    backend: BACKEND,
    actualApi: null,
    isWebGLBackend: false,
    visibleSegments: 0,
    elementCount: 0,
    gpuRenderer: '',
    softwareRenderer: false,
    frameMs: null,
    firstRenderMs: null,
    depthSort: null,
    notes,
    skipped: true,
    skipReason: reason,
  };
}

/**
 * Software-rasterizer detection on the renderer string. Headless
 * Chrome with `--use-gl=egl` routinely reports ANGLE-over-SwiftShader;
 * absolute timings from such a run are 100-1000x off a real GPU.
 */
function isSoftwareRenderer(renderer: string): boolean {
  return /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer);
}

/**
 * Probe the UNMASKED GPU renderer string via a throwaway canvas
 * (`WEBGL_debug_renderer_info`). Empty string when unavailable.
 * A SwiftShader / llvmpipe value here means the run measured software
 * rasterization — consumers must discount it.
 */
async function probeGpuRenderer(page: Page): Promise<string> {
  return page.evaluate(() => {
    try {
      const canvas = document.createElement('canvas');
      const gl =
        canvas.getContext('webgl2') ?? (canvas.getContext('webgl') as WebGLRenderingContext | null);
      if (!gl) return '';
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const val = ext
        ? (gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) as unknown)
        : (gl.getParameter(gl.RENDERER) as unknown);
      return typeof val === 'string' ? val : '';
    } catch {
      return '';
    }
  });
}

/** Probe the renderer API surface + WebGL-backend flag (line-bench parity). */
async function probeApiSurface(
  page: Page
): Promise<{ api: string | null; isWebGLBackend: boolean }> {
  return page.evaluate(() => {
    const dbg = (
      window as unknown as {
        __luxarDebug?: {
          app?: { sceneManager?: { capabilities?: { apiSurface?: string } } };
          renderer?: { backend?: { isWebGLBackend?: boolean } };
        };
      }
    ).__luxarDebug;
    return {
      api: dbg?.app?.sceneManager?.capabilities?.apiSurface ?? null,
      isWebGLBackend: dbg?.renderer?.backend?.isWebGLBackend === true,
    };
  });
}

/** Sum the clamped drawn element counts of gsplat/points nodes. */
async function probeElementCount(page: Page, onlySynthetic: boolean): Promise<number> {
  return page.evaluate((synthOnly: boolean) => {
    const dbg = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
    const scene = dbg?.scene as { traverse?: (cb: (o: unknown) => void) => void } | undefined;
    let total = 0;
    scene?.traverse?.((obj: unknown) => {
      const o = obj as {
        visible?: boolean;
        userData?: {
          nodeType?: string;
          synthetic?: boolean;
          visibleSplatCount?: number;
          visiblePointCount?: number;
        };
      };
      const ud = o.userData;
      if (!ud) return;
      if (synthOnly && ud.synthetic !== true) return;
      if (o.visible === false) return;
      if (ud.nodeType === 'gsplats' && typeof ud.visibleSplatCount === 'number') {
        total += ud.visibleSplatCount;
      } else if (ud.nodeType === 'points' && typeof ud.visiblePointCount === 'number') {
        total += ud.visiblePointCount;
      }
    });
    return total;
  }, onlySynthetic);
}

/** One-shot first-render measurement (line-bench parity). */
async function measureFirstRender(page: Page): Promise<number> {
  return page.evaluate(async () => {
    const debug = (window as unknown as { __luxarDebug: { renderOnce: () => void } }).__luxarDebug;
    const t0 = performance.now();
    debug.renderOnce();
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
    return performance.now() - t0;
  });
}

/** Raw per-frame series returned by the in-page orbit sampling loop. */
interface OrbitSamplingRaw {
  frameDtMs: number[];
  /** Per sampled frame: did the profiler depth-sort count increment? */
  sortInc: boolean[];
  /** One entry per observed count increment. */
  sortEvents: Array<{
    lastMs: number;
    kernelMs: number | null;
    boundaryMs: number | null;
    queueMs: number | null;
  }>;
  finalSortCount: number;
  totalMs: number;
}

/**
 * Steady-state frame loop with a continuous orbit and per-rAF
 * depth-sort polling. Drives frames via `renderOnce()` like the line
 * bench; the camera rotates about the current target around world +Y
 * at `orbitDegPerSec`, so the depth-sort scheduler keeps firing
 * re-sorts throughout the window.
 */
async function runOrbitSamplingLoop(page: Page): Promise<OrbitSamplingRaw> {
  return page.evaluate(
    (cfg: {
      windowMs: number;
      warmupFrames: number;
      warmupMaxMs: number;
      minFrames: number;
      orbitDegPerSec: number;
    }) => {
      const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
      // Cast-to-any chain per the debug-API contract: profiler shape is
      // internal and the metadata stage fields are optional/in-flight.
      const profiler = debug.getSceneLoader?.()?.getProfiler?.();
      const readSort = (): {
        count: number;
        lastMs: number;
        kernelMs: number | null;
        boundaryMs: number | null;
        queueMs: number | null;
      } | null => {
        try {
          const t = profiler?.getDepthSortTimings?.();
          if (!t || typeof t.count !== 'number') return null;
          const md = t.metadata ?? {};
          const num = (v: unknown): number | null => (typeof v === 'number' ? v : null);
          return {
            count: t.count,
            lastMs: typeof t.lastMs === 'number' ? t.lastMs : 0,
            kernelMs: num(md.kernelMs),
            boundaryMs: num(md.boundaryMs),
            queueMs: num(md.queueMs),
          };
        } catch {
          return null;
        }
      };

      const orbit = (dtMs: number): void => {
        try {
          const pose = debug.app.getCameraPose();
          const rad = (cfg.orbitDegPerSec * (dtMs / 1000) * Math.PI) / 180;
          const [px, py, pz] = pose.position as [number, number, number];
          const [tx, , tz] = pose.target as [number, number, number];
          const dx = px - tx;
          const dz = pz - tz;
          const nx = tx + dx * Math.cos(rad) + dz * Math.sin(rad);
          const nz = tz - dx * Math.sin(rad) + dz * Math.cos(rad);
          debug.app.setCameraPose({ ...pose, position: [nx, py, nz] });
        } catch {
          // Camera API unavailable — sample without motion rather than die.
        }
      };

      const dts: number[] = [];
      const sortInc: boolean[] = [];
      const sortEvents: OrbitSamplingRaw['sortEvents'] = [];
      let prevSortCount = readSort()?.count ?? 0;
      let frames = 0;
      let lastTime = performance.now();
      const start = lastTime;
      let collectingStart: number | null = null;

      return new Promise<OrbitSamplingRaw>((resolve) => {
        const finish = (): void => {
          resolve({
            frameDtMs: dts,
            sortInc,
            sortEvents,
            finalSortCount: readSort()?.count ?? prevSortCount,
            totalMs: performance.now() - start,
          });
        };
        const watchdog = window.setTimeout(finish, Math.max(cfg.windowMs * 3, 30_000));
        const tick = (): void => {
          const now = performance.now();
          const dt = now - lastTime;
          const elapsedSinceStart = now - start;
          const warmupDone = frames >= cfg.warmupFrames || elapsedSinceStart > cfg.warmupMaxMs;

          // Keep the orbit running through warmup too, so the sort
          // scheduler is already in its steady rhythm when sampling
          // starts.
          orbit(dt);

          // Read the sort counter on EVERY tick, warmup included:
          // sorts dispatched during warmup must advance the baseline,
          // otherwise the first sampled frame inherits their increment
          // and is misclassified as sorting-adjacent (and contributes a
          // spurious latency sample).
          const s = readSort();
          const inc = s !== null && s.count > prevSortCount;

          if (warmupDone) {
            if (collectingStart === null) collectingStart = now;
            dts.push(dt);
            sortInc.push(inc);
            if (inc && s) {
              sortEvents.push({
                lastMs: s.lastMs,
                kernelMs: s.kernelMs,
                boundaryMs: s.boundaryMs,
                queueMs: s.queueMs,
              });
            }
          }
          if (s) prevSortCount = s.count;
          lastTime = now;
          frames++;
          const collectingElapsed = collectingStart === null ? 0 : now - collectingStart;
          if (collectingElapsed < cfg.windowMs || dts.length < cfg.minFrames) {
            debug.renderOnce();
            requestAnimationFrame(tick);
          } else {
            window.clearTimeout(watchdog);
            finish();
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
      orbitDegPerSec: ORBIT_DEG_PER_SEC,
    }
  );
}

function depthSortStatsOf(raw: OrbitSamplingRaw): DepthSortStats {
  const latencies = raw.sortEvents.map((e) => e.lastMs).filter((v) => Number.isFinite(v) && v > 0);
  const latencyStats = statsOf(latencies);
  const stageMedian = (key: 'kernelMs' | 'boundaryMs' | 'queueMs'): number | undefined => {
    const vals = raw.sortEvents
      .map((e) => e[key])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const s = statsOf(vals);
    return s ? s.median : undefined;
  };
  const stats: DepthSortStats = {
    finalCount: raw.finalSortCount,
    sortCount: raw.sortEvents.length,
    sortLatencyMedianMs: latencyStats ? latencyStats.median : null,
    sortLatencyP95Ms: latencyStats ? latencyStats.p95 : null,
  };
  const kernel = stageMedian('kernelMs');
  const boundary = stageMedian('boundaryMs');
  const queue = stageMedian('queueMs');
  if (kernel !== undefined) stats.kernelMsMedian = kernel;
  if (boundary !== undefined) stats.boundaryMsMedian = boundary;
  if (queue !== undefined) stats.queueMsMedian = queue;
  return stats;
}

/**
 * L8 gate probe: split frames into 'sorting-adjacent' (a depth-sort
 * count increment within ±1 frame) vs 'idle-orbit' and return each
 * class's p99. Decides whether ordering applies are what spikes the
 * tail, or the orbit itself.
 */
function l8Classify(raw: OrbitSamplingRaw): {
  sortAdjacentP99Ms: number | null;
  idleOrbitP99Ms: number | null;
} {
  const adjacent: number[] = [];
  const idle: number[] = [];
  const inc = raw.sortInc;
  for (let i = 0; i < raw.frameDtMs.length; i++) {
    const near =
      (i > 0 && inc[i - 1] === true) ||
      inc[i] === true ||
      (i + 1 < inc.length && inc[i + 1] === true);
    (near ? adjacent : idle).push(raw.frameDtMs[i]);
  }
  return { sortAdjacentP99Ms: p99Of(adjacent), idleOrbitP99Ms: p99Of(idle) };
}

/**
 * Synthetic scenario: bootstrap a tiny zarr, hide its geometry, inject
 * the synthetic cloud, settle, place the camera on a deterministic
 * orbit shell, then orbit-sample.
 */
async function measureSyntheticScenario(
  page: Page,
  scn: Extract<ScenarioSpec, { kind: 'synthetic' }>
): Promise<ScenarioResult> {
  const notes: string[] = [];
  await page.goto(`/?src=${scn.bootstrapUrl}&renderer=${BACKEND}&debug&dpr=1`, {
    timeout: 300_000,
  });
  await waitForLuxarReady(page, 120_000);

  // Hide every bootstrap geometry node BEFORE injection so the
  // synthetic cloud is the only rendered workload (line-bench parity:
  // hiding, not removing, keeps loader bookkeeping intact).
  const injected = await page.evaluate(
    async (spec: { type: 'points' | 'gsplats'; count: number; seed: number; blending: string }) => {
      const dbg = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
      if (!dbg?.injectSyntheticScene) {
        throw new Error('synthetic scenario requires __luxarDebug.injectSyntheticScene');
      }
      dbg.scene?.traverse?.((obj: any) => {
        const nodeType = obj?.userData?.nodeType;
        if (
          (nodeType === 'lines' || nodeType === 'points' || nodeType === 'gsplats') &&
          obj.userData?.synthetic !== true
        ) {
          obj.visible = false;
        }
      });
      const result = await dbg.injectSyntheticScene(spec);
      // Yield one rAF so the renderer uploads the textures before the
      // bench's first measurement frame.
      await new Promise<void>((r) => requestAnimationFrame(() => r()));
      return { elementCount: result.elementCount as number };
    },
    { type: scn.geometry, count: scn.count, seed: scn.seed, blending: scn.blending }
  );

  if (injected.elementCount < scn.count) {
    notes.push(
      `elementCount clamped: requested ${scn.count}, drawn ${injected.elementCount} (texture capacity)`
    );
  }

  // Deterministic orbit shell: the synthetic cloud is clustered inside
  // ~[-100, 100]^3 around the origin, but the bootstrap zarr framed the
  // camera for ITS bounds. Re-aim at the origin from a fixed distance
  // so every run (and every count) starts the orbit from the same pose.
  await page.evaluate(() => {
    const dbg = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
    const pose = dbg.app.getCameraPose();
    dbg.app.setCameraPose({
      ...pose,
      position: [200, 140, 200],
      target: [0, 0, 0],
      up: [0, 1, 0],
    });
    dbg.renderOnce();
  });

  // Settle: the first back-to-front sort lands async ~1–2 frames after
  // injection (cold worker spawn on the first injection of the run).
  await page.waitForTimeout(INJECTION_SETTLE_MS);

  const apiProbe = await probeApiSurface(page);
  const gpuRenderer = await probeGpuRenderer(page);
  const firstRenderMs = await measureFirstRender(page);
  const raw = await runOrbitSamplingLoop(page);

  const result: ScenarioResult = {
    scenarioId: scn.id,
    scenarioLabel: scn.label,
    backend: BACKEND,
    actualApi: apiProbe.api,
    isWebGLBackend: apiProbe.isWebGLBackend,
    visibleSegments: injected.elementCount,
    elementCount: injected.elementCount,
    gpuRenderer,
    softwareRenderer: isSoftwareRenderer(gpuRenderer),
    frameMs: statsOf(raw.frameDtMs),
    firstRenderMs,
    depthSort: depthSortStatsOf(raw),
    notes,
    skipped: false,
  };
  if (scn.l8Probe) {
    const { sortAdjacentP99Ms, idleOrbitP99Ms } = l8Classify(raw);
    result.sortAdjacentP99Ms = sortAdjacentP99Ms;
    result.idleOrbitP99Ms = idleOrbitP99Ms;
  }
  return result;
}

/** Real-dataset orbit scenario (matrixcity tiles). */
async function measureZarrOrbitScenario(
  page: Page,
  scn: Extract<ScenarioSpec, { kind: 'zarr-orbit' }>
): Promise<ScenarioResult> {
  const notes: string[] = [];
  await page.goto(`/?src=${scn.url}&renderer=${BACKEND}&debug&dpr=1`, { timeout: 300_000 });
  await waitForLuxarReady(page, 120_000);

  // Let the first tiles commit so there is something to sort/draw;
  // don't wait for full ladder depth — steady-state orbit over a
  // partially-refined tile set is exactly the production regime.
  try {
    await page.waitForFunction(
      () => {
        const dbg = (window as unknown as { __luxarDebug?: { scene?: unknown } }).__luxarDebug;
        const scene = dbg?.scene as { traverse?: (cb: (o: unknown) => void) => void } | undefined;
        let total = 0;
        scene?.traverse?.((obj: unknown) => {
          const o = obj as { userData?: { nodeType?: string; visibleSplatCount?: number } };
          if (
            o.userData?.nodeType === 'gsplats' &&
            typeof o.userData.visibleSplatCount === 'number'
          ) {
            total += o.userData.visibleSplatCount;
          }
        });
        return total > 0;
      },
      undefined,
      { timeout: 60_000 }
    );
  } catch {
    notes.push('no splats committed within 60s — sampling anyway');
  }
  // Brief settle so the initial commit/sort burst doesn't dominate warmup.
  await page.waitForTimeout(1_000);

  const apiProbe = await probeApiSurface(page);
  const gpuRenderer = await probeGpuRenderer(page);
  const firstRenderMs = await measureFirstRender(page);
  const raw = await runOrbitSamplingLoop(page);
  const elementCount = await probeElementCount(page, false);

  if (elementCount === 0) {
    notes.push('elementCount=0 — dataset may need nD navigation to a populated slice');
    return { ...makeSkippedResult(scn, 'no visible elements', notes), gpuRenderer };
  }

  return {
    scenarioId: scn.id,
    scenarioLabel: scn.label,
    backend: BACKEND,
    actualApi: apiProbe.api,
    isWebGLBackend: apiProbe.isWebGLBackend,
    visibleSegments: elementCount,
    elementCount,
    gpuRenderer,
    softwareRenderer: isSoftwareRenderer(gpuRenderer),
    frameMs: statsOf(raw.frameDtMs),
    firstRenderMs,
    depthSort: depthSortStatsOf(raw),
    notes,
    skipped: false,
  };
}

/**
 * Progressive-ladder load scenario: poll the summed `visibleSplatCount`
 * each driven frame; the ladder is complete when the sum stops growing
 * for {@link LADDER_STABLE_MS}. Records the wall ms from navigation
 * start to the LAST observed commit plus the frame-time distribution
 * DURING the growth window (`performance.now()` in-page is relative to
 * navigation start, so commit timestamps ARE wall-ms-from-nav).
 */
async function measureZarrLadderScenario(
  page: Page,
  scn: Extract<ScenarioSpec, { kind: 'zarr-ladder' }>
): Promise<ScenarioResult> {
  const notes: string[] = [];
  // Start recording BEFORE navigation: waitForLuxarReady blocks until the
  // app is ready, by which time a fast ladder (visible-human loads in a
  // few seconds) has already finished growing — the post-ready polling
  // loop then sees zero growth and can only report a lower bound. The
  // init-script rAF loop ticks from document start, recording frame
  // deltas and the summed visibleSplatCount so the growth window is
  // captured from t=0 (performance.now() is relative to navigation
  // start, so timestamps are wall-ms-from-nav).
  await page.addInitScript(() => {
    const rec = {
      dts: [] as number[],
      lastT: null as number | null,
      lastSum: 0,
      lastChangeAt: 0,
      dtsUpToLastChange: 0,
      observedGrowth: false,
      initialCount: 0,
    };
    (window as unknown as { __ladderRec: typeof rec }).__ladderRec = rec;
    const sumSplats = (): number => {
      const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
      let total = 0;
      debug?.scene?.traverse?.((obj: any) => {
        if (
          obj?.userData?.nodeType === 'gsplats' &&
          typeof obj.userData.visibleSplatCount === 'number'
        ) {
          total += obj.userData.visibleSplatCount;
        }
      });
      return total;
    };
    const tick = (now: number): void => {
      if (rec.lastT !== null) {
        rec.dts.push(now - rec.lastT);
        const sum = sumSplats();
        if (sum !== rec.lastSum) {
          if (rec.dts.length > 0 && sum > 0) rec.observedGrowth = true;
          rec.lastSum = sum;
          rec.lastChangeAt = now;
          rec.dtsUpToLastChange = rec.dts.length;
        }
      }
      rec.lastT = now;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  // `clear-cache` pins the ladder to a COLD load every run: OPFS/L2 cache
  // state otherwise swings the load pacing several-fold between runs
  // (measured 2.7s vs 14.4s wall on the same box), drowning any
  // before/after comparison of load-window frame stats.
  await page.goto(`/?src=${scn.url}&renderer=${BACKEND}&debug&dpr=1&clear-cache`, {
    timeout: 300_000,
  });
  await waitForLuxarReady(page, 120_000);

  const ladderRaw = await page.evaluate(
    (cfg: { stableMs: number; maxMs: number }) => {
      const debug = (window as unknown as { __luxarDebug: any }).__luxarDebug;
      // The init-script recorder (installed before navigation) holds the
      // full growth history from t=0; this loop only drives frames until
      // the ladder is stable, then reads the recorder out.
      const rec = (
        window as unknown as {
          __ladderRec: {
            dts: number[];
            lastSum: number;
            lastChangeAt: number;
            dtsUpToLastChange: number;
            observedGrowth: boolean;
            initialCount: number;
          };
        }
      ).__ladderRec;

      const loopStart = performance.now();

      return new Promise<{
        wallMsToLadderComplete: number;
        observedGrowth: boolean;
        loadWindowDtMs: number[];
        finalCount: number;
        initialCount: number;
        timedOut: boolean;
      }>((resolve) => {
        const finish = (timedOut: boolean): void => {
          resolve({
            // performance.now() is relative to navigation start, and the
            // recorder ran from navigation start — this IS wall ms.
            wallMsToLadderComplete: rec.lastChangeAt,
            observedGrowth: rec.observedGrowth,
            loadWindowDtMs: rec.dts.slice(0, Math.max(rec.dtsUpToLastChange, 1)),
            finalCount: rec.lastSum,
            initialCount: rec.initialCount,
            timedOut,
          });
        };
        const tick = (): void => {
          const now = performance.now();
          if (now - loopStart > cfg.maxMs) {
            finish(true);
            return;
          }
          if (rec.lastSum > 0 && now - rec.lastChangeAt > cfg.stableMs) {
            finish(false);
            return;
          }
          debug.renderOnce();
          requestAnimationFrame(tick);
        };
        debug.renderOnce();
        requestAnimationFrame(tick);
      });
    },
    { stableMs: LADDER_STABLE_MS, maxMs: LADDER_MAX_MS }
  );

  if (ladderRaw.timedOut) {
    notes.push(`ladder polling hit the ${LADDER_MAX_MS}ms cap before stabilizing`);
  }
  if (!ladderRaw.observedGrowth) {
    notes.push(
      'ladder completed before polling began — wallMsToLadderComplete is a lower bound only'
    );
  } else {
    notes.push(
      `ladder grew ${ladderRaw.initialCount} → ${ladderRaw.finalCount} splats during polling`
    );
  }

  const apiProbe = await probeApiSurface(page);
  const gpuRenderer = await probeGpuRenderer(page);
  const firstRenderMs = await measureFirstRender(page);
  const elementCount = await probeElementCount(page, false);

  if (elementCount === 0) {
    notes.push('elementCount=0 after ladder polling');
    return { ...makeSkippedResult(scn, 'no visible elements', notes), gpuRenderer };
  }

  return {
    scenarioId: scn.id,
    scenarioLabel: scn.label,
    backend: BACKEND,
    actualApi: apiProbe.api,
    isWebGLBackend: apiProbe.isWebGLBackend,
    visibleSegments: elementCount,
    elementCount,
    gpuRenderer,
    softwareRenderer: isSoftwareRenderer(gpuRenderer),
    // The scenario's headline frame stats ARE the during-load window —
    // that's what this scenario exists to measure.
    frameMs: statsOf(ladderRaw.loadWindowDtMs),
    firstRenderMs,
    depthSort: null,
    ladder: {
      wallMsToLadderComplete: ladderRaw.wallMsToLadderComplete,
      observedGrowth: ladderRaw.observedGrowth,
      loadWindowFrameMs: statsOf(ladderRaw.loadWindowDtMs),
    },
    notes,
    skipped: false,
  };
}

async function measureScenario(page: Page, scn: ScenarioSpec): Promise<ScenarioResult> {
  switch (scn.kind) {
    case 'synthetic':
      return measureSyntheticScenario(page, scn);
    case 'zarr-orbit':
      return measureZarrOrbitScenario(page, scn);
    case 'zarr-ladder':
      return measureZarrLadderScenario(page, scn);
  }
}

/**
 * Merge one scenario row into `perf-results/{sha}/results.json` — the
 * SAME file the line bench writes. Rows are keyed `scenarioId/backend`;
 * a re-run replaces its own row and leaves every other bench's rows
 * intact, so `perf-diff.mjs` sees the union of all benches for a commit.
 *
 * Merge (rather than overwrite) is what lets this spec be a per-scenario
 * test: each scenario appends its row as it finishes, so a `-g`-filtered
 * partial run still produces a valid, additive file.
 *
 * NOTE: the line bench still writes its file wholesale, so running the
 * line bench AFTER this one for the same SHA drops these rows. Run this
 * spec last (or `-g`-filter per bench) when you want a combined file.
 */
function mergeScenarioRow(outPath: string, row: ScenarioResult): void {
  const key = `${row.scenarioId}/${row.backend}`;
  let kept: unknown[] = [];
  let prior: Partial<PerfRunResult> = {};
  if (fs.existsSync(outPath)) {
    try {
      const prev = JSON.parse(fs.readFileSync(outPath, 'utf8')) as PerfRunResult;
      prior = prev;
      kept = (prev.scenarios ?? []).filter((s) => {
        const r = s as { scenarioId?: string; backend?: string };
        return `${r.scenarioId}/${r.backend}` !== key;
      });
    } catch {
      // Corrupt/foreign file — start over with just our row.
    }
  }
  const output: PerfRunResult = {
    capturedAt: new Date().toISOString(),
    commit: currentCommitSha(),
    sampleWindowMs: prior.sampleWindowMs ?? SAMPLE_WINDOW_MS,
    warmupFrames: prior.warmupFrames ?? WARMUP_FRAMES,
    scenarios: [...kept, row],
  };
  fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
}

function logScenario(result: ScenarioResult): void {
  const fm = result.frameMs;
  const summary = result.skipped
    ? `SKIP (${result.skipReason})`
    : fm
      ? `median=${fm.median.toFixed(2)}ms p95=${fm.p95.toFixed(2)}ms p99=${fm.p99.toFixed(2)}ms max=${fm.max.toFixed(2)}ms count=${fm.count}`
      : 'no samples';
  const sortSummary = result.depthSort
    ? ` sorts=${result.depthSort.sortCount}` +
      (result.depthSort.sortLatencyMedianMs !== null
        ? ` sort_median=${result.depthSort.sortLatencyMedianMs.toFixed(2)}ms`
        : '') +
      (result.depthSort.kernelMsMedian !== undefined
        ? ` kernel_median=${result.depthSort.kernelMsMedian.toFixed(2)}ms`
        : '')
    : '';
  const ladderSummary = result.ladder
    ? ` ladder=${result.ladder.wallMsToLadderComplete.toFixed(0)}ms${
        result.ladder.observedGrowth ? '' : ' (lower bound)'
      }`
    : '';
  const l8Summary =
    result.sortAdjacentP99Ms !== undefined
      ? ` L8[sortAdjP99=${result.sortAdjacentP99Ms?.toFixed(2) ?? '—'}ms idleP99=${
          result.idleOrbitP99Ms?.toFixed(2) ?? '—'
        }ms]`
      : '';
  console.log(
    `  [${result.scenarioId}/${result.backend} → ${result.actualApi ?? '?'}] ` +
      `gpu="${result.gpuRenderer}" elements=${result.elementCount} ` +
      `${summary}${sortSummary}${ladderSummary}${l8Summary}`
  );
  for (const n of result.notes) console.log(`      note: ${n}`);
}

/**
 * One test per scenario so `-g <scenario-id>` runs exactly one (a
 * multi-million-splat scenario is minutes of GPU time; a single-test
 * bench would force all-or-nothing). Each test merges its own row, and
 * Playwright's `workers: 1` in the perf config keeps the writes serial.
 */
for (const scn of SCENARIOS) {
  test(scn.id, async ({ page }) => {
    test.setTimeout(900_000);
    console.log(`\n▶ ${scn.id} — ${scn.label}`);

    const outDir = path.join(VIEWER_ROOT, 'perf-results', currentCommitSha());
    fs.mkdirSync(outDir, { recursive: true });
    const outPath = path.join(outDir, 'results.json');

    // Skip-if-404: an absent demo dataset is an environment condition,
    // not a regression. Recorded in the JSON as a skipped row so the
    // diff shows the gap instead of silently omitting the scenario.
    const probeUrl = scn.kind === 'synthetic' ? scn.bootstrapUrl : scn.url;
    if (!(await urlExists(probeUrl))) {
      const row = makeSkippedResult(scn, 'dataset not reachable', [
        `dataset URL not reachable: ${probeUrl}`,
      ]);
      mergeScenarioRow(outPath, row);
      logScenario(row);
      test.skip(true, `dataset not reachable: ${probeUrl}`);
      return;
    }

    let result: ScenarioResult;
    try {
      result = await measureScenario(page, scn);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const shortMsg = msg.split('\n').slice(0, 3).join(' | ');
      result = makeSkippedResult(scn, 'measurement error', [`measurement threw: ${shortMsg}`]);
    }

    if (result.softwareRenderer) {
      result.notes.push(
        `SOFTWARE RASTERIZER ("${result.gpuRenderer}") — absolute timings are NOT comparable to a GPU run`
      );
    }
    if (result.frameMs !== null && result.frameMs.count < MIN_FRAMES) {
      result.notes.push(
        `low sample count: ${result.frameMs.count} frames (< ${MIN_FRAMES}) — percentiles are coarse`
      );
    }
    if (result.depthSort !== null && result.depthSort.sortCount === 0) {
      result.notes.push(
        'no depth-sort dispatch observed during the window — orbit may not have crossed the re-sort threshold'
      );
    }

    // Write BEFORE asserting so a partial/failed scenario still leaves
    // its diagnostic row on disk.
    mergeScenarioRow(outPath, result);
    logScenario(result);
    console.log(`📊 row merged into ${outPath}`);

    // Structural sanity only — no timing assertions (record-only bench).
    expect(result.skipped, `${scn.id}: ${result.skipReason} — ${result.notes.join('; ')}`).toBe(
      false
    );
    expect(result.frameMs, `${scn.id}: frame stats missing`).not.toBeNull();
    // Frame-count floor. Relaxed to >= 1 for two legitimate cases:
    //  - the ladder scenario, whose window is the (short) progressive-load
    //    period and can honestly be a handful of frames;
    //  - a software rasterizer, where one multi-million-element frame can
    //    take seconds so the 3 s window cannot hold MIN_FRAMES frames.
    // Both are recorded as notes rather than failures — a record-only
    // bench must not fail because the host has no GPU.
    const frameFloor = result.ladder || result.softwareRenderer ? 1 : MIN_FRAMES;
    expect(
      result.frameMs!.count,
      `${scn.id}: fewer than ${frameFloor} sampled frames`
    ).toBeGreaterThanOrEqual(frameFloor);
    expect(result.gpuRenderer, `${scn.id}: empty GPU renderer string`).not.toBe('');
    expect(result.elementCount, `${scn.id}: zero drawn elements`).toBeGreaterThan(0);
  });
}
