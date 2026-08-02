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
 *   - Depth-sort completion stats sampled per rAF from
 *     `getSceneLoader().getProfiler().getDepthSortCompletions()`: the
 *     monotonic completion total plus the drained per-completion `lastMs`
 *     series (→ sortCount, sort-latency median/p95). This dedicated stream
 *     (issue #711) records one event per applied ordering — unlike the
 *     seq-merged 'Depth Sort' root's `count`, it does not undercount
 *     multi-completion frames or drop late resolves. Each event MAY also
 *     carry numeric `kernelMs`/`boundaryMs`/`queueMs` — recorded when
 *     present, absence tolerated.
 *   - L8 GATE PROBE (10M scenario only): every sampled frame is
 *     classified 'sorting-adjacent' (an ordering apply landed within
 *     ±1 frame, detected via a depth-sort completion this frame) vs
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
 *  - 'synthetic': boot the viewer with NO dataset, then push a
 *    clustered points/gsplats cloud into the live scene via
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
      geometry: 'points' | 'gsplats';
      count: number;
      seed: number;
      blending: string;
      /** Attach the L8 gate probe fields to this scenario's row. */
      l8Probe?: boolean;
      /**
       * Gate a hard assertion that at least one depth sort completed during
       * the orbit window (relaxed on a software rasterizer).
       */
      requiresDepthSort?: boolean;
    }
  | {
      kind: 'zarr-orbit';
      id: string;
      label: string;
      url: string;
      /**
       * Gate a hard assertion that at least one depth sort completed during
       * the orbit window (relaxed on a software rasterizer).
       */
      requiresDepthSort?: boolean;
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

const SCENARIOS: ScenarioSpec[] = [
  {
    kind: 'synthetic',
    id: 'synthetic-gsplats-1M-orbit',
    label: 'synthetic clustered gsplats, 1 M splats, normal blending, 30°/s orbit',
    geometry: 'gsplats',
    count: 1_000_000,
    seed: 42,
    blending: 'normal',
    requiresDepthSort: true,
  },
  {
    kind: 'synthetic',
    id: 'synthetic-gsplats-5M-orbit',
    label: 'synthetic clustered gsplats, 5 M splats, normal blending, 30°/s orbit',
    geometry: 'gsplats',
    count: 5_000_000,
    seed: 42,
    blending: 'normal',
    requiresDepthSort: true,
  },
  {
    kind: 'synthetic',
    id: 'synthetic-gsplats-10M-orbit',
    label: 'synthetic clustered gsplats, 10 M splats, normal blending, 30°/s orbit (L8 gate probe)',
    geometry: 'gsplats',
    count: 10_000_000,
    seed: 42,
    blending: 'normal',
    l8Probe: true,
    requiresDepthSort: true,
  },
  {
    kind: 'synthetic',
    id: 'synthetic-points-5M-orbit',
    label: 'synthetic clustered points, 5 M points, normal blending, 30°/s orbit (symmetry check)',
    geometry: 'points',
    count: 5_000_000,
    seed: 42,
    blending: 'normal',
    requiresDepthSort: true,
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
    requiresDepthSort: true,
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
  /**
   * Present (> 0) when completions aged out of the profiler's bounded
   * ring between two polls, so they are counted in {@link finalCount}
   * but missing from {@link sortCount} and the latency percentiles —
   * treat the latency stats of such a window as under-sampled.
   */
  droppedCompletions?: number;
}

interface LadderStats {
  /** ms from navigation start to ladder completion. Exact
   *  stamp-confirmed completion time (the first tick of the last
   *  continuous committedLadderComplete window) UNLESS {@link
   *  usedFallback} (a count-plateau estimate) or {@link timedOut} (the
   *  ladder never completed within the cap, so this is a last-commit
   *  lower bound). */
  wallMsToLadderComplete: number;
  /** True when the recorder observed the summed splat count grow during
   *  the run. */
  observedGrowth: boolean;
  /** True when NO committed gsplat leaf was ever observed (the defensive
   *  path), so completion was estimated from the count plateau rather
   *  than confirmed by committed-ladder stamps. Always false on a
   *  timed-out run — {@link timedOut} takes precedence there, since no
   *  plateau estimate was established either. */
  usedFallback: boolean;
  /** True when the poll hit LADDER_MAX_MS before completion —
   *  wallMsToLadderComplete is then a last-commit lower bound, not a
   *  confirmed completion. */
  timedOut: boolean;
  /** Frame stats DURING the progressive-load window (start → completion:
   *  the confirmed stamp window on stamped runs, else the last count
   *  change). */
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
  /** Per sampled frame: did any depth-sort completion land this frame? */
  sortInc: boolean[];
  /** One entry per drained depth-sort completion (issue #711 stream). */
  sortEvents: Array<{
    lastMs: number;
    kernelMs: number | null;
    boundaryMs: number | null;
    queueMs: number | null;
  }>;
  /**
   * Completions that aged out of the profiler's bounded (512-event) ring
   * before this loop could drain them — i.e. a single poll gap saw more
   * completions than the ring retains. `finalSortCount` stays exact (it
   * reads the monotonic total), but these events contribute no latency
   * sample, so median/p95 under-sample when this is > 0.
   */
  droppedCompletions: number;
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
      // Drain the profiler's dedicated MONOTONIC depth-sort completion stream
      // (issue #711) instead of inferring per-sort events from the seq-merged
      // 'Depth Sort' profiler root's `count`. That root undercounts frames
      // where several leaves finish at once (an increase of 1 and of 20 both
      // added exactly one event) and can DROP late/out-of-order resolves; the
      // completion stream records one event per applied ordering, in order.
      const readCompletions = (): {
        total: number;
        events: Array<{
          seq: number;
          lastMs: number;
          kernelMs: number | null;
          boundaryMs: number | null;
          queueMs: number | null;
        }>;
      } | null => {
        try {
          const c = profiler?.getDepthSortCompletions?.();
          if (!c || typeof c.total !== 'number' || !Array.isArray(c.events)) return null;
          return c;
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
      let droppedCompletions = 0;
      let prevCompletionSeq = readCompletions()?.total ?? 0;
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
            droppedCompletions,
            finalSortCount: readCompletions()?.total ?? prevCompletionSeq,
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

          // Drain the completion stream on EVERY tick, warmup included:
          // sorts that complete during warmup must advance the baseline,
          // otherwise the first sampled frame inherits their events and is
          // misclassified as sorting-adjacent (and contributes spurious
          // latency samples). `inc` = did any completion land this frame;
          // every completion since the previous poll is drained (a frame
          // with N completions pushes N events, not one — the multi-
          // completion undercount issue #711 fixed).
          const c = readCompletions();
          const inc = c !== null && c.total > prevCompletionSeq;

          if (warmupDone) {
            if (collectingStart === null) collectingStart = now;
            dts.push(dt);
            sortInc.push(inc);
            // Drain EVERY completion since the previous poll — one sortEvents
            // entry per completion (a frame with N completions pushes N).
            if (c) {
              let drained = 0;
              for (const e of c.events) {
                if (e.seq > prevCompletionSeq) {
                  drained++;
                  sortEvents.push({
                    lastMs: e.lastMs,
                    kernelMs: e.kernelMs,
                    boundaryMs: e.boundaryMs,
                    queueMs: e.queueMs,
                  });
                }
              }
              // Ring overflow: more completions landed since the previous
              // poll than the profiler's bounded ring retains. `total`
              // stays exact, but the aged-out events can never contribute
              // a latency sample — count them so the stats disclose the
              // gap instead of silently under-sampling.
              droppedCompletions += Math.max(0, c.total - prevCompletionSeq - drained);
            }
          }
          if (c) prevCompletionSeq = c.total;
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
  if (raw.droppedCompletions > 0) stats.droppedCompletions = raw.droppedCompletions;
  return stats;
}

/**
 * L8 gate probe: split frames into 'sorting-adjacent' (a depth-sort
 * completion within ±1 frame) vs 'idle-orbit' and return each
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
 * Synthetic scenario: boot the viewer with NO dataset, inject the
 * synthetic cloud directly via the debug ports (which exist at init
 * regardless of any loaded dataset), settle, place the camera on a
 * deterministic orbit shell, then orbit-sample. Self-contained — needs
 * no external zarr, so it never silently skips on a fresh checkout.
 */
async function measureSyntheticScenario(
  page: Page,
  scn: Extract<ScenarioSpec, { kind: 'synthetic' }>
): Promise<ScenarioResult> {
  const notes: string[] = [];
  await page.goto(`/?renderer=${BACKEND}&debug&dpr=1`, {
    timeout: 300_000,
  });
  await waitForLuxarReady(page, 120_000);

  // The no-dataset boot routes an empty `src` to "must-browse" and opens
  // the DatasetBrowser modal; dismiss it via its own close button so its
  // backdrop-blurred overlay doesn't composite over the canvas during
  // sampling and inflate the synthetic frame times.
  await page.evaluate(() => {
    document.querySelector<HTMLElement>('.luxar-dataset-browser__close-btn')?.click();
  });
  // Verify the panel is actually gone (DatasetBrowser.close() removes it
  // from the DOM). A silently-failed dismissal — say the button's class
  // changes — would corrupt every synthetic measurement without any
  // signal, which is the exact failure class this bench must not have;
  // the throw surfaces as a measurement error and fails the scenario.
  await page.waitForFunction(() => !document.getElementById('luxar-dataset-browser'), undefined, {
    timeout: 5_000,
  });

  // Hide any pre-existing geometry nodes BEFORE injection so the
  // synthetic cloud is the only rendered workload. With a no-dataset
  // boot there is normally nothing to hide; kept as a harmless,
  // future-proof guard (line-bench parity: hiding, not removing, keeps
  // loader bookkeeping intact).
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
  // ~[-100, 100]^3 around the origin, but the no-dataset boot leaves the
  // camera at its default empty-scene framing. Re-aim at the origin from
  // a fixed distance so every run (and every count) starts the orbit
  // from the same pose.
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
 * Progressive-ladder load scenario: poll each visible gsplat leaf's
 * `userData.committedLadderComplete` stamp every driven frame; the
 * ladder is complete once every visible leaf reports a committed FULL
 * ladder (no leaf still stamped `false`). The viewer writes that stamp
 * whenever a gsplat leaf commits, so the stamped path is the normal
 * one; only if NO committed leaf is ever observed does the scenario
 * fall back to the count plateau — the ladder is treated complete when
 * the summed `visibleSplatCount` stops growing for
 * {@link LADDER_STABLE_MS} — and the result is flagged `usedFallback`.
 * Records the wall ms from navigation start to completion plus the
 * frame-time distribution DURING the growth window (`performance.now()`
 * in-page is relative to navigation start, so commit timestamps ARE
 * wall-ms-from-nav).
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
      // Stamp-based completion (the authoritative signal): sticky flag
      // that any leaf ever carried a committedLadderComplete stamp, plus
      // the start timestamp of the current continuous "all stamped
      // complete" window (0 while any visible leaf is still incomplete).
      sawStamps: false,
      ladderCompleteAt: 0,
      // Frame count at the tick the current completion window opened —
      // the stamped-run load-window length (dtsUpToLastChange only covers
      // the last COUNT change, which a stamp-only final commit can trail).
      dtsUpToComplete: 0,
    };
    (window as unknown as { __ladderRec: typeof rec }).__ladderRec = rec;
    // Per-tick scan: sum visible splats (drives the count-plateau
    // fallback) AND fold each visible leaf's committedLadderComplete
    // stamp into the completion signal.
    const scanScene = (): {
      sum: number;
      nodeCount: number;
      stampedCount: number;
      anyStamped: boolean;
      anyIncomplete: boolean;
    } => {
      const debug = (window as unknown as { __luxarDebug?: any }).__luxarDebug;
      let sum = 0;
      let nodeCount = 0;
      let stampedCount = 0;
      let anyStamped = false;
      let anyIncomplete = false;
      // Use traverseVisible so hidden subtrees — e.g. an abandoned hidden
      // LOD level stamped `false` — are pruned, matching production's
      // foldProgress.
      debug?.scene?.traverseVisible?.((obj: any) => {
        if (
          obj?.userData?.nodeType === 'gsplats' &&
          typeof obj.userData.visibleSplatCount === 'number'
        ) {
          sum += obj.userData.visibleSplatCount;
          nodeCount += 1;
          const stamp = obj.userData.committedLadderComplete;
          if (typeof stamp === 'boolean') {
            anyStamped = true;
            stampedCount += 1;
            if (stamp === false) anyIncomplete = true;
          }
        }
      });
      return { sum, nodeCount, stampedCount, anyStamped, anyIncomplete };
    };
    const tick = (now: number): void => {
      if (rec.lastT !== null) {
        rec.dts.push(now - rec.lastT);
        const { sum, nodeCount, stampedCount, anyStamped, anyIncomplete } = scanScene();
        if (sum !== rec.lastSum) {
          if (rec.dts.length > 0 && sum > 0) rec.observedGrowth = true;
          rec.lastSum = sum;
          rec.lastChangeAt = now;
          rec.dtsUpToLastChange = rec.dts.length;
        }
        if (anyStamped) rec.sawStamps = true;
        // Complete only when EVERY counted visible leaf carries a stamp
        // AND none is still incomplete — a created-but-uncommitted leaf
        // (numeric count, no stamp yet) keeps this false.
        const stampedComplete = nodeCount > 0 && stampedCount === nodeCount && !anyIncomplete;
        if (stampedComplete) {
          if (rec.ladderCompleteAt === 0) {
            rec.ladderCompleteAt = now;
            rec.dtsUpToComplete = rec.dts.length;
          }
        } else {
          rec.ladderCompleteAt = 0;
          rec.dtsUpToComplete = 0;
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
            sawStamps: boolean;
            ladderCompleteAt: number;
            dtsUpToComplete: number;
          };
        }
      ).__ladderRec;

      const loopStart = performance.now();

      return new Promise<{
        wallMsToLadderComplete: number;
        observedGrowth: boolean;
        usedFallback: boolean;
        loadWindowDtMs: number[];
        finalCount: number;
        initialCount: number;
        timedOut: boolean;
      }>((resolve) => {
        const finish = (timedOut: boolean): void => {
          // A timed-out run established NO completion of either kind
          // (the loop checks completion before the cap), so neither
          // completion label applies: wallMs is the last-commit lower
          // bound and timedOut takes precedence over usedFallback.
          const usedFallback = !timedOut && !rec.sawStamps;
          const stampConfirmed = !timedOut && rec.ladderCompleteAt > 0;
          resolve({
            // performance.now() is relative to navigation start, and the
            // recorder ran from navigation start — this IS wall ms. A
            // stamp-confirmed run reports the confirmed completion
            // timestamp; fallback/timed-out runs report the last count
            // change (exact plateau estimate / lower bound respectively).
            wallMsToLadderComplete: stampConfirmed ? rec.ladderCompleteAt : rec.lastChangeAt,
            observedGrowth: rec.observedGrowth,
            usedFallback,
            // Stamped runs window start → confirmed completion; otherwise
            // start → last count change (a stamp-only final commit can
            // trail the last count change).
            loadWindowDtMs: rec.dts.slice(
              0,
              Math.max(stampConfirmed ? rec.dtsUpToComplete : rec.dtsUpToLastChange, 1)
            ),
            finalCount: rec.lastSum,
            initialCount: rec.initialCount,
            timedOut,
          });
        };
        const tick = (): void => {
          const now = performance.now();
          // Completion first, THEN the cap — a completion established
          // just before LADDER_MAX_MS must not be misreported as a
          // timeout by a poll tick that fires just after it.
          if (rec.sawStamps) {
            // Authoritative: finish once every visible leaf has committed
            // its FULL ladder (ladderCompleteAt is the start of that window).
            if (rec.ladderCompleteAt > 0) {
              finish(false);
              return;
            }
          } else if (rec.lastSum > 0 && now - rec.lastChangeAt > cfg.stableMs) {
            // Defensive fallback: no committed leaf was ever observed, so
            // infer completion from the count plateau (approximate — see
            // usedFallback).
            finish(false);
            return;
          }
          if (now - loopStart > cfg.maxMs) {
            finish(true);
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
  if (ladderRaw.usedFallback) {
    notes.push(
      `no committed gsplat leaf observed — completion estimated from a ${LADDER_STABLE_MS}ms count plateau (approximate)`
    );
  } else if (!ladderRaw.timedOut) {
    notes.push('ladder completion confirmed via committed-ladder stamps');
  }
  if (ladderRaw.observedGrowth) {
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
      usedFallback: ladderRaw.usedFallback,
      timedOut: ladderRaw.timedOut,
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
        result.ladder.usedFallback ? ' (fallback est.)' : ''
      }${result.ladder.timedOut ? ' (lower bound)' : ''}`
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
    // ONLY zarr scenarios have an external dataset — synthetic scenarios
    // build their geometry entirely in JS via the debug ports and must
    // NEVER silently skip on a fresh checkout (that was the whole point
    // of issue #705), so they bypass this guard.
    if (scn.kind !== 'synthetic' && !(await urlExists(scn.url))) {
      const row = makeSkippedResult(scn, 'dataset not reachable', [
        `dataset URL not reachable: ${scn.url}`,
      ]);
      mergeScenarioRow(outPath, row);
      logScenario(row);
      test.skip(true, `dataset not reachable: ${scn.url}`);
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
    if (result.depthSort?.droppedCompletions) {
      result.notes.push(
        `${result.depthSort.droppedCompletions} depth-sort completions aged out of the bounded ` +
          'ring between polls — sortCount and the latency percentiles under-sample ' +
          '(finalCount is still exact)'
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

    // A scenario that requires depth sorting must complete at least one
    // sort round-trip during the orbit window. The profiler's sort count
    // stays 0 when the scheduler never dispatches or the SortWorker never
    // returns within the window — the "green depth-sort benchmark that
    // actually measured UNSORTED rendering" failure this guards against.
    // Relaxed on a software rasterizer for the same reason as the
    // frame-count floor: one multi-million-element frame can take seconds
    // there, so the orbit may not cross the re-sort threshold within the
    // sample window (recorded as a note).
    const requiresDepthSort = 'requiresDepthSort' in scn && scn.requiresDepthSort === true;
    if (requiresDepthSort && !result.skipped && !result.softwareRenderer) {
      expect(
        result.depthSort?.sortCount ?? 0,
        `${scn.id}: requires depth sorting but no sort completed during the orbit window`
      ).toBeGreaterThan(0);
    }
  });
}
