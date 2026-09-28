/**
 * Generic runner for the render gate's MANIFEST suites (`counters`,
 * `playback`, `scrub`, `hosted`, `trees`, `cache`, ... — any key of the
 * manifest's `suites` block).
 *
 * A suite case names a scene (`store` or `synthetic`), a `workload` (what to
 * do to the viewer: idle, drag, play a timeline, scrub it, orbit, zoom, a cold
 * load) and the `metrics` it is judged on. Each metric is a field of the
 * workload's result, a viewer perf counter by name (`render.count`), a
 * build-independent ext counter (`ext.gpuUploadBytes`, see
 * `ext-counters.mjs`) or a server-side figure of a hosted run
 * (`requestsToFirstFrame`). Arms run in the same rotating base / cand / base2
 * order as the perf suite; the base/base2 pair is the A/A control.
 *
 * @module scripts/render-gate/suites
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { installExtCounters } from './ext-counters.mjs';
import * as ops from './page-ops.mjs';
import {
  anyMissing,
  judgeCounter,
  judgePerf,
  median,
  noiseFloor,
  ROTATIONS,
} from './perf-stats.mjs';

/** Expectation vocabulary for `--expect`. */
export const EXPECTATIONS = ['win', 'zero', 'same', 'pass'];

/**
 * Load and validate an `--expect` file: `{ "<caseId>": { "<metric>": "win" |
 * "zero" | "same" | "pass" } }`. A case or metric the manifest does not
 * declare is a hard error (a typo would otherwise expect nothing).
 *
 * @param {string} path JSON file.
 * @param {Record<string, object>} manifestSuites The manifest's `suites`.
 * @returns {Record<string, Record<string, string>>}
 */
export function loadExpectations(path, manifestSuites) {
  const raw = JSON.parse(readFileSync(path, 'utf8'));
  return validateExpectations(raw, manifestSuites);
}

/** Validate a parsed expectation object against the manifest (see `loadExpectations`). */
export function validateExpectations(raw, manifestSuites) {
  const cases = new Map();
  for (const suite of Object.values(manifestSuites ?? {})) {
    for (const c of suite.cases ?? []) cases.set(c.id, c);
  }
  for (const [caseId, metrics] of Object.entries(raw)) {
    const c = cases.get(caseId);
    if (!c) throw new Error(`--expect: unknown case '${caseId}'`);
    for (const [metric, want] of Object.entries(metrics)) {
      if (!(c.metrics ?? []).some((m) => m.name === metric)) {
        throw new Error(`--expect: case '${caseId}' declares no metric '${metric}'`);
      }
      if (!EXPECTATIONS.includes(want)) {
        throw new Error(
          `--expect: ${caseId}.${metric}: '${want}' is not one of ${EXPECTATIONS.join('|')}`
        );
      }
    }
  }
  return raw;
}

/**
 * Whether a judged metric meets its expectation. `n/a` never does.
 *
 * - `win`: the verdict is a win;
 * - `zero`: the candidate median is exactly 0;
 * - `same`: the verdict is pass and, for a counter, the medians are equal;
 * - `pass`: no regression (pass or win).
 */
export function meetsExpectation(want, judged, kind) {
  if (judged.verdict === 'n/a' || judged.verdict === undefined) return false;
  if (want === 'win') return judged.verdict === 'win';
  if (want === 'zero') return judged.candMedian === 0;
  if (want === 'same') {
    return (
      judged.verdict === 'pass' && (kind !== 'counter' || judged.baseMedian === judged.candMedian)
    );
  }
  return judged.verdict === 'pass' || judged.verdict === 'win';
}

/**
 * Judge one declared metric over the three arms.
 *
 * @param {{ name: string, better?: 'lower'|'higher', kind?: 'counter'|'timing', tol?: number, relTol?: number }} m
 * @param {{ base: object[], base2: object[], cand: object[] }} samples
 */
export function judgeMetric(m, samples) {
  const better = m.better ?? 'lower';
  const kind = m.kind ?? 'counter';
  const pick = (arm) => samples[arm].map((s) => s[m.name]);
  const base = pick('base');
  const base2 = pick('base2');
  const cand = pick('cand');
  const common = { kind, better };
  if (kind === 'counter')
    return { ...common, ...judgeCounter(base, cand, base2, { better, tol: m.tol ?? 0, relTol: m.relTol ?? 0 }) };
  if (anyMissing(base, cand, base2)) return { ...common, verdict: 'n/a' };
  return {
    ...common,
    baseMedian: median(base),
    candMedian: median(cand),
    ...judgePerf(base, cand, noiseFloor(base, base2), m.tol ?? 0, { better }),
  };
}

/**
 * Server-side figures of one arm from the server's request log.
 *
 * `navT` is the server clock when navigation began and `firstFrameMs` the
 * page's first commit in ms since navigation, so `navT + firstFrameMs`
 * approximates the first frame on the server's clock. Dataset requests only
 * (`/datasets/`) count toward the first-frame figures: the viewer's own
 * bundle is not what the data path changes.
 *
 * `serialDepth` approximates the longest dependent request chain before the
 * first frame as the number of start WAVES: sorted by start time, a request
 * starting at least `max(latencyMs, 20)` ms after the previous one opens a new
 * wave (a dependent request cannot start before its parent's response, which
 * takes at least the link latency).
 */
export function serverMetrics(log, { navT, firstFrameMs, latencyMs = 0, maxInflight }) {
  const data = log.filter((e) => e.url.startsWith('/datasets/'));
  const out = {
    serverRequests: data.length,
    serverBytes: data.reduce((s, e) => s + e.bytes, 0),
    maxInflight,
  };
  if (typeof firstFrameMs !== 'number' || !Number.isFinite(firstFrameMs)) return out;
  const cutoff = navT + firstFrameMs;
  const early = data.filter((e) => e.startMs <= cutoff).sort((a, b) => a.startMs - b.startMs);
  const gap = Math.max(latencyMs, 20);
  let waves = early.length > 0 ? 1 : 0;
  for (let i = 1; i < early.length; i++) {
    if (early[i].startMs - early[i - 1].startMs >= gap) waves++;
  }
  return {
    ...out,
    requestsToFirstFrame: early.length,
    bytesToFirstFrame: early.reduce((s, e) => s + e.bytes, 0),
    serialDepth: waves,
  };
}

// ---------------------------------------------------------------------------
// Workloads (node side; the in-page halves live in page-ops.mjs)
// ---------------------------------------------------------------------------

/**
 * A drag on the canvas with REAL input events (Playwright's mouse, i.e. CDP
 * `Input.dispatchMouseEvent`): an in-page synthetic `PointerEvent` has no
 * active pointer, so the controls' `setPointerCapture` throws on it. Then a
 * `tailMs` idle. `renders` covers drag + tail, `tailRenders` the tail alone.
 */
async function dragWorkload(page, { px = 4, tailMs = 3000, moves = 8 }) {
  const box = await page.evaluate(() => {
    const r = window.__luxarDebug.app.sceneManager.renderer.domElement.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  });
  await page.evaluate(ops.beginRenderCount);
  let tail;
  try {
    await page.mouse.move(box.x, box.y);
    await page.mouse.down();
    await page.mouse.move(box.x + px, box.y, { steps: Math.max(1, moves) });
    await page.mouse.up();
    tail = await page.evaluate(ops.idle, { ms: tailMs });
  } finally {
    const renders = await page.evaluate(ops.endRenderCount);
    tail = { ...(tail ?? {}), total: renders };
  }
  return { renders: tail.total, tailRenders: tail.renders, tailFrames: tail.frames };
}

const PAGE_WORKLOADS = {
  idle: ops.idle,
  playback: ops.playback,
  scrub: ops.scrub,
  orbit: ops.orbit,
  zoom: ops.zoom,
  coldLoad: ops.coldLoad,
};

async function runWorkload(page, workload, ctx) {
  const { kind, ...params } = workload;
  if (kind === 'drag') return dragWorkload(page, params);
  const fn = PAGE_WORKLOADS[kind];
  if (!fn) throw new Error(`unknown workload kind '${kind}'`);
  if (kind === 'coldLoad') return page.evaluate(fn, { pose: ctx.pose, ...params });
  return page.evaluate(fn, params);
}

/**
 * Screenshot the page WITHOUT forcing a render, then render once and
 * screenshot again. A viewer that stopped its loop on a frame older than its
 * state (a stale frame) shows different pixels after the forced render.
 *
 * @returns {Promise<boolean>} True when both PNGs are byte-identical.
 */
export async function staleFrameCheck(page) {
  const before = await page.screenshot();
  await page.evaluate(async () => {
    window.__luxarDebug.renderOnce();
    for (let i = 0; i < 2; i++) await new Promise((r) => requestAnimationFrame(() => r()));
  });
  const after = await page.screenshot();
  return before.equals(after);
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

const prefixed = (obj, prefix) =>
  Object.fromEntries(Object.entries(obj ?? {}).map(([k, v]) => [`${prefix}${k}`, v]));

async function measureArm(browser, server, c, backend, { defaults, profile, openCase }) {
  const cold = c.workload.kind === 'coldLoad';
  const pose = c.pose ?? defaults.pose;
  let navT = 0;
  const { context, page } = await openCase(browser, server.origin, c, {
    backend,
    dsf: c.dsf ?? defaults.dsf ?? 1,
    viewport: c.viewport ?? defaults.viewport ?? { width: 960, height: 540 },
    urlParams: defaults.urlParams,
    contextOptions: { ignoreHTTPSErrors: true },
    initScripts: [installExtCounters],
    beforeGoto: () => {
      server.reset();
      navT = server.now();
    },
  });
  try {
    if (!cold) {
      if (pose) {
        await page.evaluate(ops.applyView, {
          projection: c.projection ?? defaults.projection ?? 'perspective',
          fov: pose.fov,
          pose,
        });
      }
      const st = await page.evaluate(ops.settle, {
        minFrames: 30,
        requireLoaderSettled: !c.synthetic,
      });
      if (!st.settled) throw new Error('did not settle');
      await page.evaluate(ops.resetCounters);
    }
    const result = await runWorkload(page, c.workload, { pose });
    const counters = await page.evaluate(ops.readCounters);
    const ext = await page.evaluate(ops.readExtCounters);
    const metrics = { ...counters, ...prefixed(ext, 'ext.'), ...result };
    Object.assign(
      metrics,
      serverMetrics(server.log, {
        navT,
        firstFrameMs: result.firstFrameMs,
        latencyMs: profile.latencyMs ?? 0,
        maxInflight: server.stats().maxInflight,
      })
    );
    if (c.staleFrameCheck) metrics.staleFrame = (await staleFrameCheck(page)) ? 0 : 1;
    return metrics;
  } finally {
    await context.close();
  }
}

function applyExpectations(row, c, expected) {
  if (!expected) return;
  for (const [name, want] of Object.entries(expected)) {
    const judged = row.metrics[name];
    const m = c.metrics.find((x) => x.name === name);
    const met = !!judged && meetsExpectation(want, judged, m?.kind ?? 'counter');
    if (judged) Object.assign(judged, { expected: want, met });
    if (!met) {
      row.status = 'fail';
      row.reasons.push('expectation');
      row.failures.push(
        `expectation: ${name} expected ${want}, got ${judged?.verdict ?? 'nothing'}`
      );
    }
  }
}

/**
 * Run one manifest suite.
 *
 * @param {object} p
 * @param {string} p.name Suite name.
 * @param {{ defaults?: object, cases: object[] }} p.suite Manifest suite.
 * @param {object} p.profile Resolved server profile (`name` + server options).
 * @param {object} p.session `browserSession` of this suite.
 * @param {Function} p.openCase `browserKit().openCase`.
 * @param {{ base: object, cand: object }} p.servers This suite's servers.
 * @param {string} p.dataRoot Datasets root (a missing store is an error row).
 * @param {object} p.opts Run options (`only`, `backends`, `rounds`, `heavy`).
 * @param {object|null} p.expectations Validated `--expect` content.
 * @param {(m: string) => void} p.log
 * @returns {Promise<object[]>} Rows.
 */
export async function runSuite({
  name,
  suite,
  profile,
  session,
  openCase,
  servers,
  dataRoot,
  opts,
  expectations,
  log,
}) {
  const defaults = suite.defaults ?? {};
  const rounds = opts.rounds ?? defaults.rounds ?? 5;
  const rows = [];
  const cases = (suite.cases ?? []).filter((c) => !opts.only || opts.only.includes(c.id));
  for (const c of cases) {
    const expected = expectations?.[c.id];
    const allowed = c.backends ?? defaults.backends ?? ['webgl', 'webgpu'];
    const backends = (opts.backends ?? allowed).filter((b) => allowed.includes(b));
    if (c.heavy && !opts.heavy) {
      const row = {
        suite: name,
        case: c.id,
        backend: '-',
        status: 'skipped',
        metrics: {},
        failures: ['heavy (pass --heavy)'],
        reasons: [],
      };
      if (expected) {
        row.status = 'fail';
        row.reasons.push('expectation');
        row.failures.push('expectation: case skipped as heavy; pass --heavy');
      }
      rows.push(row);
      continue;
    }
    for (const backend of backends) {
      log(`${name}: ${c.id} ${backend} (${rounds} rounds, server ${profile.name})`);
      const row = {
        suite: name,
        case: c.id,
        backend,
        status: 'pass',
        metrics: {},
        failures: [],
        reasons: [],
      };
      if (c.store && !existsSync(join(dataRoot, c.store))) {
        rows.push({ ...row, status: 'error', failures: [`store missing: datasets/${c.store}`] });
        continue;
      }
      const samples = { base: [], base2: [], cand: [] };
      try {
        for (let r = 0; r < rounds; r++) {
          for (const arm of ROTATIONS[r % ROTATIONS.length]) {
            const server = arm === 'cand' ? servers.cand : servers.base;
            samples[arm].push(
              await session.run((browser) =>
                measureArm(browser, server, c, backend, { defaults, profile, openCase })
              )
            );
          }
        }
      } catch (e) {
        rows.push({ ...row, status: 'error', failures: [String(e.message ?? e)] });
        continue;
      }
      for (const m of c.metrics ?? []) row.metrics[m.name] = judgeMetric(m, samples);
      if (Object.values(row.metrics).some((v) => v.verdict === 'fail')) {
        row.status = 'fail';
        row.reasons.push('regression');
        row.failures.push('a metric regressed beyond its floor');
      }
      if (c.staleFrameCheck) {
        const stale = (arm) => samples[arm].filter((s) => s.staleFrame === 1).length;
        row.staleFrames = { base: stale('base') + stale('base2'), cand: stale('cand') };
        if (row.staleFrames.cand > 0) {
          row.status = 'fail';
          row.reasons.push('stale-frame');
          row.failures.push(
            `stale-frame: ${row.staleFrames.cand}/${rounds} candidate arms changed on a forced render`
          );
        }
      }
      applyExpectations(row, c, expected);
      rows.push(row);
    }
  }
  return rows;
}

/** A number for the report, or '-'. */
function fmt(x) {
  if (typeof x !== 'number' || !Number.isFinite(x)) return '-';
  return Number.isInteger(x) ? String(x) : x.toFixed(3);
}

/**
 * The report table of one suite (lines, for joining).
 *
 * @param {string} name Suite name.
 * @param {object[]} rows `runSuite` rows.
 * @returns {string[]}
 */
export function suiteMarkdown(name, rows) {
  const lines = [
    `## Suite: ${name}`,
    '',
    '| case | backend | metric | base median | cand median | ratio | verdict |',
    '|---|---|---|---|---|---|---|',
  ];
  for (const r of rows) {
    const note = r.failures?.length ? ` (${r.failures.join('; ')})` : '';
    const entries = Object.entries(r.metrics ?? {});
    if (r.status === 'error' || r.status === 'skipped' || entries.length === 0) {
      lines.push(`| ${r.case} | ${r.backend} | - | | | | ${r.status}${note} |`);
      continue;
    }
    for (const [m, v] of entries) {
      const exp = v.expected ? ` [expect ${v.expected}: ${v.met ? 'met' : 'UNMET'}]` : '';
      lines.push(
        `| ${r.case} | ${r.backend} | ${m} | ${fmt(v.baseMedian)} | ${fmt(v.candMedian)} | ${fmt(v.ratio)} | ${v.verdict}${exp} |`
      );
    }
    if (r.status !== 'pass')
      lines.push(`| ${r.case} | ${r.backend} | (row) | | | | ${r.status}${note} |`);
  }
  lines.push('');
  return lines;
}
