/**
 * Helpers for `viewer-audit-perf-bench.spec.ts` — the measurement recipe the
 * 2026-09 viewer performance audit used, ported into the perf-bench family so
 * every optimisation is validated against the same numbers.
 *
 * What is deliberately NOT here: WebGL timer queries. On ANGLE/Metal
 * `EXT_disjoint_timer_query_webgl2` reported 40–56 ms "GPU frames" for scenes
 * that render at a solid 120 fps. Frame cost is measured as the rAF cadence
 * under forced continuous rendering instead (rAF throttles to the compositor
 * when the GPU is behind, which is what the user sees).
 */

import type { CDPSession, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { placeCameraAt } from './helpers';

/** Network profiles for CDP emulation (bytes per second). */
export const NETWORK_PROFILES = {
  /** A typical hosted demo: 25 Mbps down, 30 ms RTT. */
  hosted: { latency: 30, downloadThroughput: 25e6 / 8, uploadThroughput: 5e6 / 8 },
} as const;

export type NetworkProfile = keyof typeof NETWORK_PROFILES;

/** In-page observers installed before navigation via `addInitScript`. */
interface AuditWindow {
  __luxarAudit: {
    longTaskCount: number;
    longTaskMs: number;
    longTaskMax: number;
    frames: number[];
  };
}

/** Subset of `__luxarDebug.getPerf()` the bench reads (see perf-snapshot.ts). */
export interface PerfSnapshotLite {
  perfReady: boolean;
  runtimeReady: boolean;
  isSettled: boolean | null;
  timeline: {
    milestones: Record<string, number | undefined>;
    firstCommit: Record<string, number | undefined>;
    refinement: { passes: number; rungs: number; rungsFromCache: number; complete: boolean };
    measures: {
      ttfpMs: number | null;
      metadataReadyMs: number | null;
      poolReadyMs: number | null;
      sceneLoadedMs: number | null;
      initUpdateDoneMs: number | null;
      refinementCompleteMs: number | null;
    };
  };
  rendererInfo: { calls: number; triangles: number } | null;
  adaptiveDpr: {
    currentDPR: number;
    scaleDowns: number;
    scaleUps: number;
    verdicts: Array<{ kind: string; fpsRatio: number | null }>;
  } | null;
}

/**
 * Install the long-task observer and a rAF timestamp ring. Must run before
 * `page.goto` so the observer sees the load.
 */
export async function installAuditObservers(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const w = window as unknown as AuditWindow;
    w.__luxarAudit = { longTaskCount: 0, longTaskMs: 0, longTaskMax: 0, frames: [] };
    try {
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          w.__luxarAudit.longTaskCount += 1;
          w.__luxarAudit.longTaskMs += entry.duration;
          w.__luxarAudit.longTaskMax = Math.max(w.__luxarAudit.longTaskMax, entry.duration);
        }
      }).observe({ type: 'longtask', buffered: true });
    } catch {
      /* longtask unsupported — counters stay 0 */
    }
  });
}

/** Apply (or clear with `null`) a CDP network throttle to the page. */
export async function applyNetworkProfile(
  page: Page,
  profile: NetworkProfile | null
): Promise<CDPSession> {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Network.enable');
  if (profile) {
    await cdp.send('Network.emulateNetworkConditions', {
      offline: false,
      ...NETWORK_PROFILES[profile],
    });
  }
  return cdp;
}

/** Wait until `__luxarDebug.getPerf` exists (seeded at bootstrap, before init). */
export async function waitForPerfReady(page: Page, timeout = 60_000): Promise<void> {
  await page.waitForFunction(
    () =>
      typeof (window as unknown as { __luxarDebug?: { getPerf?: unknown } }).__luxarDebug
        ?.getPerf === 'function',
    null,
    { timeout }
  );
}

/**
 * Wait for `getPerf().isSettled === true`: no update pass, no load pass, no
 * lazy LOD level, post-load refinement complete. Returns false on timeout so
 * the caller can record a partial row instead of failing the run.
 */
export async function waitForPerfSettled(page: Page, timeout: number): Promise<boolean> {
  try {
    await page.waitForFunction(
      () =>
        (
          window as unknown as { __luxarDebug: { getPerf: () => { isSettled: boolean | null } } }
        ).__luxarDebug.getPerf().isSettled === true,
      null,
      { timeout, polling: 100 }
    );
    return true;
  } catch {
    return false;
  }
}

export async function readPerf(page: Page): Promise<PerfSnapshotLite> {
  return page.evaluate(
    () =>
      (
        window as unknown as { __luxarDebug: { getPerf: () => PerfSnapshotLite } }
      ).__luxarDebug.getPerf() as PerfSnapshotLite
  );
}

export async function readLongTasks(
  page: Page
): Promise<{ count: number; ms: number; max: number }> {
  return page.evaluate(() => {
    const a = (window as unknown as AuditWindow).__luxarAudit;
    return { count: a.longTaskCount, ms: Math.round(a.longTaskMs), max: Math.round(a.longTaskMax) };
  });
}

export async function resetLongTasks(page: Page): Promise<void> {
  await page.evaluate(() => {
    const a = (window as unknown as AuditWindow).__luxarAudit;
    a.longTaskCount = 0;
    a.longTaskMs = 0;
    a.longTaskMax = 0;
  });
}

/** `__luxarDebug.cache.getStats().l2WriteQueue.dropped`, or null when unavailable. */
export async function readOpfsDropped(page: Page): Promise<number | null> {
  return page.evaluate(() => {
    try {
      const d = (
        window as unknown as {
          __luxarDebug: { cache?: { getStats: () => { l2WriteQueue?: { dropped?: number } } } };
        }
      ).__luxarDebug;
      return d.cache?.getStats().l2WriteQueue?.dropped ?? null;
    } catch {
      return null;
    }
  });
}

/** Timestamp (ms since navigation) at which `getState` became available, or null. */
export async function waitForStateReady(page: Page, timeout: number): Promise<number | null> {
  try {
    await page.waitForFunction(
      () =>
        typeof (window as unknown as { __luxarDebug?: { getState?: unknown } }).__luxarDebug
          ?.getState === 'function',
      null,
      { timeout, polling: 50 }
    );
    return page.evaluate(() => performance.now());
  } catch {
    return null;
  }
}

export interface FrameCadence {
  frames: number;
  p50: number;
  p95: number;
  max: number;
  rafHz: number;
}

/**
 * Force a render on every animation frame for `seconds` and report the rAF
 * interval distribution. Under GPU back-pressure the compositor throttles
 * rAF to the presented frame rate, so p50 is the frame cost the user sees.
 */
export async function measureContinuousRender(page: Page, seconds: number): Promise<FrameCadence> {
  return page.evaluate(
    (secs) =>
      new Promise<FrameCadence>((resolve) => {
        const d = (window as unknown as { __luxarDebug: { renderOnce: () => void } }).__luxarDebug;
        d.renderOnce();
        setTimeout(() => {
          const t0 = performance.now();
          const ts: number[] = [];
          const tick = (t: number) => {
            d.renderOnce();
            ts.push(t);
            if (t - t0 < secs * 1000) {
              requestAnimationFrame(tick);
              return;
            }
            const dd: number[] = [];
            for (let i = 1; i < ts.length; i++) dd.push(ts[i] - ts[i - 1]);
            dd.sort((a, b) => a - b);
            const pick = (q: number) => dd[Math.min(dd.length - 1, Math.floor(q * dd.length))] ?? 0;
            resolve({
              frames: ts.length,
              p50: +pick(0.5).toFixed(2),
              p95: +pick(0.95).toFixed(2),
              max: +(dd[dd.length - 1] ?? 0).toFixed(2),
              rafHz: +(ts.length / ((t - t0) / 1000)).toFixed(1),
            });
          };
          requestAnimationFrame(tick);
        }, 300);
      }),
    seconds
  );
}

/** Pin the render DPR through the adaptive-DPR manager's manual setter. */
export async function setManualDpr(page: Page, dpr: number): Promise<void> {
  await page.evaluate((v) => {
    const app = (
      window as unknown as {
        __luxarDebug: { app: { adaptiveDPRManager?: { setManualDPR?: (d: number) => void } } };
      }
    ).__luxarDebug.app;
    app.adaptiveDPRManager?.setManualDPR?.(v);
  }, dpr);
  await page.waitForTimeout(400);
}

/**
 * Dolly the camera along its current view axis so the distance to the orbit
 * target becomes `factor × current` (0.25 = four times closer). Uses the
 * supported `__luxarE2ECamera.place()` path (a raw `camera.position` write is
 * undone by the controls' per-frame distance clamp).
 */
export async function dollyCamera(page: Page, factor: number): Promise<void> {
  const pose = await page.evaluate(() => {
    const sm = (
      window as unknown as {
        __luxarDebug: {
          app: {
            sceneManager: {
              camera: { position: { x: number; y: number; z: number } };
              controls: { currentControls?: { target?: { x: number; y: number; z: number } } };
            };
          };
        };
      }
    ).__luxarDebug.app.sceneManager;
    const p = sm.camera.position;
    const t = sm.controls.currentControls?.target ?? { x: 0, y: 0, z: 0 };
    return { p: { x: p.x, y: p.y, z: p.z }, t: { x: t.x, y: t.y, z: t.z } };
  });
  const position = {
    x: pose.t.x + (pose.p.x - pose.t.x) * factor,
    y: pose.t.y + (pose.p.y - pose.t.y) * factor,
    z: pose.t.z + (pose.p.z - pose.t.z) * factor,
  };
  await placeCameraAt(page, position, { target: pose.t });
}

export interface DimensionStep {
  /** Time to the first commit (`awaitDimensionUpdate` resolving). */
  firstMs: number;
  /** Time until `getPerf().isSettled` (full ladder), or null on timeout. */
  fullMs: number | null;
  elements: number;
}

/**
 * Step a non-displayed dimension to `value` and time the first commit and the
 * settled state. `settleTimeoutMs` bounds the second wait.
 */
export async function stepDimension(
  page: Page,
  dim: number,
  value: number,
  settleTimeoutMs = 20_000
): Promise<DimensionStep> {
  return page.evaluate(
    async ({ dim, value, settleTimeoutMs }) => {
      const d = (
        window as unknown as {
          __luxarDebug: {
            app: {
              setDimensionValue: (i: number, v: number) => void;
              awaitDimensionUpdate: () => Promise<void>;
            };
            getPerf: () => { isSettled: boolean | null };
            getState: () => { totalElements: number };
          };
        }
      ).__luxarDebug;
      const t0 = performance.now();
      d.app.setDimensionValue(dim, value);
      await d.app.awaitDimensionUpdate();
      const firstMs = performance.now() - t0;
      let fullMs: number | null = null;
      const deadline = t0 + settleTimeoutMs;
      while (performance.now() < deadline) {
        if (d.getPerf().isSettled === true) {
          fullMs = performance.now() - t0;
          break;
        }
        await new Promise((r) => setTimeout(r, 10));
      }
      return { firstMs, fullMs, elements: d.getState().totalElements };
    },
    { dim, value, settleTimeoutMs }
  );
}

/** First non-displayed dimension index and its integer range, or null for a 3-D scene. */
export async function findSliceDimension(
  page: Page
): Promise<{ dim: number; lo: number; hi: number } | null> {
  return page.evaluate(() => {
    const dims = (
      window as unknown as {
        __luxarDebug: {
          app: {
            getDimensions: () => {
              ndim: number;
              displayed: number[];
              ranges?: Array<[number, number]>;
            } | null;
          };
        };
      }
    ).__luxarDebug.app.getDimensions();
    if (!dims) return null;
    for (let i = 0; i < dims.ndim; i++) {
      if (dims.displayed.includes(i)) continue;
      const r = dims.ranges?.[i] ?? [0, 0];
      return { dim: i, lo: Math.ceil(r[0]), hi: Math.floor(r[1]) };
    }
    return null;
  });
}

export function median(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (v.length === 0) return null;
  return v[Math.floor(v.length / 2)];
}

/** (max − min) / median, as a fraction; null when fewer than two values. */
export function spread(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x));
  if (v.length < 2) return null;
  const m = median(v);
  if (!m) return null;
  return +((Math.max(...v) - Math.min(...v)) / m).toFixed(3);
}

/** Round to one decimal for the results file. */
export function r1(value: number | null): number | null {
  return value === null ? null : Math.round(value * 10) / 10;
}

/**
 * Merge one scenario row into `perf-results/<sha>/results.json`, keyed by
 * `scenarioId/backend` like the line and gsplat benches, so `perf-diff.mjs`
 * sees the union of all benches for a commit.
 */
export function mergeResultRow(
  outPath: string,
  commit: string,
  row: { scenarioId: string; backend: string } & Record<string, unknown>
): void {
  const key = `${row.scenarioId}/${row.backend}`;
  let kept: unknown[] = [];
  let prior: { sampleWindowMs?: number; warmupFrames?: number } = {};
  if (fs.existsSync(outPath)) {
    try {
      const prev = JSON.parse(fs.readFileSync(outPath, 'utf8')) as {
        scenarios?: unknown[];
        sampleWindowMs?: number;
        warmupFrames?: number;
      };
      prior = prev;
      kept = (prev.scenarios ?? []).filter((s) => {
        const r = s as { scenarioId?: string; backend?: string };
        return `${r.scenarioId}/${r.backend}` !== key;
      });
    } catch {
      // Corrupt/foreign file — start over with just our row.
    }
  }
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(
    outPath,
    JSON.stringify(
      {
        capturedAt: new Date().toISOString(),
        commit,
        sampleWindowMs: prior.sampleWindowMs ?? 3000,
        warmupFrames: prior.warmupFrames ?? 0,
        scenarios: [...kept, row],
      },
      null,
      2
    )
  );
}
