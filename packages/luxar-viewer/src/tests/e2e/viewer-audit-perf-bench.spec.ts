/**
 * Viewer audit bench — the end-to-end numbers the 2026-09 performance audit
 * was built on, reproducible per commit so each optimisation can be
 * validated (or rejected) against the same baseline.
 *
 * Per scene, per repetition, in a FRESH browser context (empty OPFS = cold):
 *   load     ttfpMs, sceneLoadedMs, initUpdateDoneMs, refinementCompleteMs,
 *            stateReadyMs (when `getState` appears = blend warm-up settled),
 *            requests, bytes, longTaskMs during the load, opfsDropped (also
 *            re-read at end-of-run and ASSERTED to be 0 whenever the resolved
 *            write-queue allowance covers the bytes streamed — see #2561),
 *            opfsWriteQueueMaxBytes, opfsWrites, refinement passes/rungs
 *   frames   rAF-cadence p50 under forced continuous render at DPR 1 and 0.5
 *            in the default framing, and at DPR 1 dollied 4x closer
 *   playback (4-D scenes) first-commit and settled time per timepoint step,
 *            cold then warm
 * plus one warm re-load in the last repetition's context, and a separate
 * adaptive-DPR steady-state row (no `dpr=` pin) for the dense point scene.
 *
 * Medians across repetitions land in `perf-results/<sha>/results.json` under
 * `audit`, keyed `audit-<scene>[-hosted]/<backend>`; `spread` holds
 * (max−min)/median per metric so a noisy host is visible in the row itself.
 *
 * Env: `LUXAR_PERF_AUDIT_REPEATS` (default 3), `LUXAR_PERF_AUDIT_SCENES`
 * (comma-separated scene ids), `LUXAR_PERF_AUDIT_NET=hosted` (CDP throttle
 * 25 Mbps / 30 ms; adds `-hosted` to the scenario id), `LUXAR_PERF_BACKEND`
 * (row backend label, default `webgl`). Run with `LUXAR_PERF_PREVIEW=1` so
 * the production bundle is measured — dev-mode ESM inflates TTFP and requests.
 */

import { expect, test, type Browser, type BrowserContext, type Page } from '@playwright/test';
import { execSync } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { resolvePerfDataBase } from './perf-data-base';
import {
  applyNetworkProfile,
  dollyCamera,
  findSliceDimension,
  installAuditObservers,
  measureContinuousRender,
  median,
  mergeResultRow,
  r1,
  readLongTasks,
  readOpfsDropped,
  readOpfsWriteQueueMaxBytes,
  readOpfsWrites,
  readPerf,
  setManualDpr,
  spread,
  stepDimension,
  waitForPerfReady,
  waitForPerfSettled,
  waitForStateReady,
  type NetworkProfile,
} from './perf-audit-helpers';

const VIEWER_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const DATA_BASE = resolvePerfDataBase();
const REPEATS = Math.max(1, Number(process.env.LUXAR_PERF_AUDIT_REPEATS ?? 3));
const BACKEND = process.env.LUXAR_PERF_BACKEND ?? 'webgl';
const NET = (process.env.LUXAR_PERF_AUDIT_NET as NetworkProfile | undefined) ?? null;
const FRAME_SECONDS = 3;
const PLAYBACK_STEPS = 8;

interface AuditScene {
  id: string;
  /** Repo-relative store path served by the perf data server. */
  path: string;
  /** Has a non-displayed dimension to scrub. */
  playback?: boolean;
  /** Bound on waiting for `isSettled` after load. */
  settleTimeoutMs?: number;
}

const SCENES: AuditScene[] = [
  { id: 'dense-points', path: 'datasets/examples/dense_cubic_gradient_example.luxar.zarr' },
  { id: 'bench-100-nodes', path: 'datasets/examples/performance_benchmark_example.luxar.zarr' },
  { id: 'ct-gsplats', path: 'datasets/demos/gsplats_3d_ct_totalsegmentator.luxar.zarr' },
  {
    id: 'neuromast-4d',
    path: 'datasets/demos/gsplats_4d_neuromast_2ch.luxar.zarr',
    playback: true,
  },
  {
    id: 'celegans-4d',
    path: 'datasets/demos/gsplats_4d_celegans_tracking.luxar.zarr',
    playback: true,
  },
  {
    id: 'cmu1-2d',
    path: 'datasets/demos/gsplats_2d_cmu1_pathology.luxar.zarr',
    settleTimeoutMs: 240_000,
  },
];

const filter = (process.env.LUXAR_PERF_AUDIT_SCENES ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
const activeScenes = filter.length ? SCENES.filter((s) => filter.includes(s.id)) : SCENES;

function currentCommitSha(): string {
  try {
    return execSync('git rev-parse --short HEAD', { cwd: VIEWER_ROOT }).toString().trim();
  } catch {
    return 'unknown';
  }
}

function sceneUrl(scene: AuditScene, extra = '&dpr=1'): string {
  return `/?src=${DATA_BASE}/${scene.path}&debug${extra}`;
}

/** Skip cleanly when the store is not served (datasets live in the main checkout). */
async function storeAvailable(scene: AuditScene): Promise<boolean> {
  for (const doc of ['zarr.json', '.zattrs']) {
    try {
      const res = await fetch(`${DATA_BASE}/${scene.path}/${doc}`, { method: 'GET' });
      if (res.ok) return true;
    } catch {
      /* try the next document */
    }
  }
  return false;
}

interface LoadMetrics {
  ttfpMs: number | null;
  sceneLoadedMs: number | null;
  initUpdateDoneMs: number | null;
  refinementCompleteMs: number | null;
  stateReadyMs: number | null;
  settled: boolean;
  requests: number;
  bytes: number;
  longTaskMs: number;
  longTaskMax: number;
  opfsDropped: number | null;
  /** Bytes pending L2 writes may retain this session (the #2561 allowance). */
  opfsWriteQueueMaxBytes: number | null;
  /** Cumulative L2 writes that landed — liveness for the drop counters. */
  opfsWrites: number | null;
  refinementPasses: number;
  refinementRungs: number;
}

interface RunMetrics extends LoadMetrics {
  /**
   * `opfsDropped` re-read at the END of the run. The load-phase sample is
   * taken before the dolly + settle and the playback steps, whose late chunk
   * traffic is the largest burst on some scenes; the counter is cumulative and
   * monotonic, so only a final read can see those drops.
   */
  opfsDroppedFinal: number | null;
  frameP50Ms_dpr1: number | null;
  frameP95Ms_dpr1: number | null;
  frameP50Ms_dpr05: number | null;
  frameP50Ms_zoom4x: number | null;
  drawCalls: number | null;
  playbackFirstMs_cold: number | null;
  playbackFullMs_cold: number | null;
  playbackFirstMs_warm: number | null;
  playbackFullMs_warm: number | null;
}

async function newAuditContext(browser: Browser): Promise<BrowserContext> {
  return browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
}

/** Navigate and measure the load phase; the page stays open for the caller. */
async function measureLoad(page: Page, scene: AuditScene, url: string): Promise<LoadMetrics> {
  let requests = 0;
  let bytes = 0;
  page.on('requestfinished', (req) => {
    requests += 1;
    void req
      .sizes()
      .then((s) => {
        bytes += s.responseBodySize;
      })
      .catch(() => undefined);
  });
  await installAuditObservers(page);
  await applyNetworkProfile(page, NET);
  await page.goto(url);
  await waitForPerfReady(page);
  const stateReadyMs = await waitForStateReady(page, 180_000);
  const settled = await waitForPerfSettled(page, scene.settleTimeoutMs ?? 120_000);
  // Let trailing request bookkeeping settle before reading the counters.
  await page.waitForTimeout(500);
  const perf = await readPerf(page);
  const lt = await readLongTasks(page);
  const opfsDropped = await readOpfsDropped(page);
  const opfsWriteQueueMaxBytes = await readOpfsWriteQueueMaxBytes(page);
  const opfsWrites = await readOpfsWrites(page);
  const m = perf.timeline.measures;
  const start = perf.timeline.milestones.loadStart;
  return {
    ttfpMs: r1(m.ttfpMs),
    sceneLoadedMs: r1(m.sceneLoadedMs),
    initUpdateDoneMs: r1(m.initUpdateDoneMs),
    refinementCompleteMs: r1(m.refinementCompleteMs),
    stateReadyMs: stateReadyMs !== null && start !== undefined ? r1(stateReadyMs - start) : null,
    settled,
    requests,
    bytes,
    longTaskMs: lt.ms,
    longTaskMax: lt.max,
    opfsDropped,
    opfsWriteQueueMaxBytes,
    opfsWrites,
    refinementPasses: perf.timeline.refinement.passes,
    refinementRungs: perf.timeline.refinement.rungs,
  };
}

async function measureFrames(page: Page): Promise<Partial<RunMetrics>> {
  await setManualDpr(page, 1);
  const dpr1 = await measureContinuousRender(page, FRAME_SECONDS);
  await setManualDpr(page, 0.5);
  const dpr05 = await measureContinuousRender(page, FRAME_SECONDS);
  await setManualDpr(page, 1);
  const perf = await readPerf(page);
  await dollyCamera(page, 0.25);
  // A closer pose may promote lazy LOD levels; let them land before sampling.
  await waitForPerfSettled(page, 30_000);
  const zoom = await measureContinuousRender(page, FRAME_SECONDS);
  return {
    frameP50Ms_dpr1: dpr1.p50,
    frameP95Ms_dpr1: dpr1.p95,
    frameP50Ms_dpr05: dpr05.p50,
    frameP50Ms_zoom4x: zoom.p50,
    drawCalls: perf.rendererInfo?.calls ?? null,
  };
}

async function measurePlayback(page: Page): Promise<Partial<RunMetrics>> {
  const slice = await findSliceDimension(page);
  if (!slice) return {};
  const values: number[] = [];
  for (let i = 1; i <= PLAYBACK_STEPS && slice.lo + i <= slice.hi; i++) values.push(slice.lo + i);
  if (values.length === 0) return {};
  const cold = { first: [] as number[], full: [] as number[] };
  const warm = { first: [] as number[], full: [] as number[] };
  for (const v of values) {
    const s = await stepDimension(page, slice.dim, v);
    cold.first.push(s.firstMs);
    if (s.fullMs !== null) cold.full.push(s.fullMs);
  }
  await stepDimension(page, slice.dim, slice.lo);
  for (const v of values) {
    const s = await stepDimension(page, slice.dim, v);
    warm.first.push(s.firstMs);
    if (s.fullMs !== null) warm.full.push(s.fullMs);
  }
  return {
    playbackFirstMs_cold: r1(median(cold.first)),
    playbackFullMs_cold: r1(median(cold.full)),
    playbackFirstMs_warm: r1(median(warm.first)),
    playbackFullMs_warm: r1(median(warm.full)),
  };
}

async function runOnce(context: BrowserContext, scene: AuditScene): Promise<RunMetrics> {
  const page = await context.newPage();
  try {
    const load = await measureLoad(page, scene, sceneUrl(scene));
    const frames = await measureFrames(page);
    const playback = scene.playback ? await measurePlayback(page) : {};
    // Last read before the page closes: catches drops from the dolly/settle
    // and the playback steps, which the load-phase sample cannot see.
    const opfsDroppedFinal = await readOpfsDropped(page);
    return {
      ...load,
      opfsDroppedFinal,
      frameP50Ms_dpr1: null,
      frameP95Ms_dpr1: null,
      frameP50Ms_dpr05: null,
      frameP50Ms_zoom4x: null,
      drawCalls: null,
      playbackFirstMs_cold: null,
      playbackFullMs_cold: null,
      playbackFirstMs_warm: null,
      playbackFullMs_warm: null,
      ...frames,
      ...playback,
    };
  } finally {
    await page.close();
  }
}

function aggregate(runs: RunMetrics[]): {
  medians: Record<string, number | null | boolean>;
  spreads: Record<string, number | null>;
} {
  const medians: Record<string, number | null | boolean> = {};
  const spreads: Record<string, number | null> = {};
  const keys = Object.keys(runs[0]) as Array<keyof RunMetrics>;
  for (const key of keys) {
    const vals = runs.map((r) => r[key]);
    if (typeof vals[0] === 'boolean') {
      medians[key] = vals.every((v) => v === true);
      continue;
    }
    const nums = vals.filter((v): v is number => typeof v === 'number');
    medians[key] = nums.length ? r1(median(nums)) : null;
    spreads[key] = spread(nums);
  }
  return { medians, spreads };
}

test.describe('viewer audit bench', () => {
  for (const scene of activeScenes) {
    test(`audit ${scene.id}${NET ? ` (${NET})` : ''}`, async ({ browser }) => {
      test.setTimeout(900_000);
      test.skip(!(await storeAvailable(scene)), `${scene.path} is not served at ${DATA_BASE}`);

      const runs: RunMetrics[] = [];
      let warm: LoadMetrics | null = null;
      for (let i = 0; i < REPEATS; i++) {
        const context = await newAuditContext(browser);
        try {
          runs.push(await runOnce(context, scene));
          if (i === REPEATS - 1) {
            // Same context = populated OPFS: the warm second visit.
            const page = await context.newPage();
            try {
              warm = await measureLoad(page, scene, sceneUrl(scene));
            } finally {
              await page.close();
            }
          }
        } finally {
          await context.close();
        }
      }

      const { medians, spreads } = aggregate(runs);
      const scenarioId = `audit-${scene.id}${NET ? `-${NET}` : ''}`;
      const row = {
        scenarioId,
        backend: BACKEND,
        repeats: REPEATS,
        frameMs: {
          median: medians.frameP50Ms_dpr1,
          p95: medians.frameP95Ms_dpr1,
          count: runs.length,
        },
        audit: {
          ...medians,
          warmSceneLoadedMs: warm?.sceneLoadedMs ?? null,
          warmRequests: warm?.requests ?? null,
          spread: spreads,
        },
      };
      const outPath = path.join(VIEWER_ROOT, 'perf-results', currentCommitSha(), 'results.json');
      mergeResultRow(outPath, currentCommitSha(), row);
      console.log(`[audit] ${scenarioId}: ${JSON.stringify(row.audit)}`);

      // Guard for #2561: a retained-byte cap that binds below what a scene
      // streams silently turns L2 into a no-op, and the only symptom is a
      // slower warm revisit — so assert the drop counters here rather than
      // trusting a reader to notice the row. Asserted AFTER the row is written
      // so a regression still leaves the measurement behind, and soft so one
      // scene does not hide the others' numbers.
      const numbers = (vals: Array<number | null>): number[] =>
        vals.filter((v): v is number => v !== null);
      const dropSamples = numbers([
        ...runs.map((r) => r.opfsDropped),
        ...runs.map((r) => r.opfsDroppedFinal),
      ]);
      const writeSamples = numbers(runs.map((r) => r.opfsWrites));
      const capSamples = numbers(runs.map((r) => r.opfsWriteQueueMaxBytes));
      // Total response bytes, so an OVER-estimate of the chunk bytes the queue
      // actually sees (bundle + wasm included). It is compared against the cap
      // below, and over-estimating there only ever skips the assertion rather
      // than failing it — the safe direction for a guard.
      const streamedBytes = Math.max(...runs.map((r) => r.bytes));

      // (1) The counters must be READABLE on every repetition, load and final.
      // A helper that regresses to always-null would otherwise turn every
      // assertion below into a silent no-op.
      expect
        .soft(dropSamples.length, `${scene.id}: opfsDropped unreadable on some repetition`)
        .toBe(runs.length * 2);
      // (2) LIVENESS: zero drops proves nothing if the tier never wrote. An
      // absent L2 store or a tripped OPFS circuit breaker (seen under
      // automated Chromium) reports zero drops too.
      expect
        .soft(
          writeSamples.length > 0 ? Math.max(...writeSamples) : 0,
          `${scene.id}: L2 wrote nothing — dead tier, not a clean run`
        )
        .toBeGreaterThan(0);
      // (3) The drop assertion, but only where the allowance actually covers
      // what the scene streamed. Below that, dropping is the DESIGNED
      // heap-relative behaviour (a small heap must not retain bytes it does
      // not have), so demanding 0 there would fail for the right reason with
      // the wrong diagnosis.
      const resolvedCap = capSamples.length > 0 ? Math.min(...capSamples) : null;
      const capCovers = resolvedCap !== null && resolvedCap >= streamedBytes;
      const dropMessage =
        `${scene.id}: L2 write-queue drops must stay at 0 (#2561) — resolved cap ` +
        `${resolvedCap ?? 'unknown'} B vs ${streamedBytes} B streamed`;
      if (capCovers) {
        expect.soft(Math.max(...dropSamples, 0), dropMessage).toBe(0);
        const warmDrops = warm?.opfsDropped ?? null;
        if (warmDrops !== null) expect.soft(warmDrops, `${dropMessage} — warm revisit`).toBe(0);
      } else {
        console.log(
          `[audit] ${scenarioId}: drop assertion skipped — allowance ${resolvedCap ?? 'unknown'} B ` +
            `does not cover the ${streamedBytes} B streamed (expected on a small heap), ` +
            `drops ${dropSamples.length > 0 ? Math.max(...dropSamples) : 'unknown'}`
        );
      }
    });
  }

  test('audit dense-points adaptive-dpr steady state', async ({ browser }) => {
    test.setTimeout(600_000);
    const scene = SCENES[0];
    test.skip(!(await storeAvailable(scene)), `${scene.path} is not served at ${DATA_BASE}`);
    const context = await newAuditContext(browser);
    try {
      const page = await context.newPage();
      await installAuditObservers(page);
      await page.goto(sceneUrl(scene, ''));
      await waitForPerfReady(page);
      await waitForPerfSettled(page, 120_000);
      // 30 s of continuous rendering: enough for the controller to walk,
      // probe and settle (probe window 1.5 s, floor TTL 30 s).
      const cadence = await measureContinuousRender(page, 30);
      const perf = await readPerf(page);
      const row = {
        scenarioId: `audit-${scene.id}-adaptive`,
        backend: BACKEND,
        audit: {
          steadyDpr: perf.adaptiveDpr?.currentDPR ?? null,
          scaleDowns: perf.adaptiveDpr?.scaleDowns ?? null,
          scaleUps: perf.adaptiveDpr?.scaleUps ?? null,
          verdicts: perf.adaptiveDpr?.verdicts.length ?? null,
          verdictKinds: (perf.adaptiveDpr?.verdicts ?? []).map((v) => v.kind).join(','),
          frameP50Ms_steady: cadence.p50,
        },
      };
      const outPath = path.join(VIEWER_ROOT, 'perf-results', currentCommitSha(), 'results.json');
      mergeResultRow(outPath, currentCommitSha(), row);
      console.log(`[audit] ${row.scenarioId}: ${JSON.stringify(row.audit)}`);
    } finally {
      await context.close();
    }
  });
});
