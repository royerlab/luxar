#!/usr/bin/env node
/**
 * Render gate: measure a candidate viewer build against a baseline build for
 * render EXACTNESS and render PERFORMANCE, on this machine, and exit non-zero
 * when the candidate fails its declared verdict class.
 *
 *   node scripts/render-gate/run-gate.mjs --base origin/main --cand HEAD \
 *     [--suite exact|perf|counters|playback|scrub|hosted|trees|cache|all] \
 *     [--class IDENTICAL|ULP] [--intended id,id] [--only id,id] \
 *     [--backends webgl,webgpu,webgpu-gl] [--rounds 7] [--heavy] [--expect <file.json>] \
 *     [--server-profile local|hosted] [--out <dir>]
 *   node scripts/render-gate/run-gate.mjs --from-json <dir>/report.json
 *     (rewrite <dir>/report.md from a saved report, measuring nothing)
 *
 * `exact` and `perf` are built in; every other suite is declared in the
 * manifest's `suites` block and run by `suites.mjs` (workload + declared
 * metrics, judged against an A/A control).
 *
 * Both builds are served from their own `dist/` (cached per commit SHA, see
 * `builds.mjs`) with the checkout's `datasets/` mounted at `/datasets/`, and
 * driven through debug surfaces present on main (`page-ops.mjs`), so the
 * baseline may predate the gate.
 *
 * Every measurement carries its own control arm: exactness captures the
 * baseline TWICE (two page loads) and a case whose A/A pair is not bit-identical
 * is excluded as nondeterministic rather than blamed on the candidate; perf
 * interleaves base, candidate and a second base arm in rotating order, and the
 * base/base spread sets that metric's noise floor. Method, classes and
 * thresholds: docs/guides/developer/RENDER_GATE.md.
 */

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { browserKit } from './browser.mjs';
import { ensureBuild } from './builds.mjs';
import { judge, scoreBlocks, scoreFloatBuffers, scorePickBuffers } from './exactness.mjs';
import { writeUlpHeatmap } from './heatmap.mjs';
import * as ops from './page-ops.mjs';
import { judgePerf, median, noiseFloor, ROTATIONS } from './perf-stats.mjs';
import { startServer } from './server.mjs';
import { loadExpectations, runSuite, suiteMarkdown } from './suites.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const viewerRoot = resolve(here, '../..');

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return fallback;
  const next = process.argv[i + 1];
  return next === undefined || next.startsWith('--') ? true : next;
}
const list = (v) => (typeof v === 'string' && v.length ? v.split(',').map((s) => s.trim()) : null);

const opts = {
  base: arg('base', 'origin/main'),
  cand: arg('cand', 'HEAD'),
  suite: arg('suite', 'exact'),
  cls: arg('class', 'IDENTICAL'),
  intended: list(arg('intended', '')) ?? [],
  only: list(arg('only', '')),
  backends: list(arg('backends', '')),
  dsf: list(arg('dsf', ''))?.map(Number) ?? null,
  rounds: Number(arg('rounds', 0)) || null,
  basePort: Number(arg('base-port', 4801)),
  channel: arg('channel', 'chrome'),
  headless: arg('headed', false) !== true,
  chromeArgs: list(arg('chrome-args', '')) ?? [],
  label: arg('label', null),
  scenes: arg('scenes', join(here, 'gate-scenes.json')),
  // Prebuilt dist/ for an arm (calibration: a baseline patched by hand).
  baseDist: arg('base-dist', null),
  candDist: arg('cand-dist', null),
  // Re-render report.md from a saved report.json, measuring nothing.
  fromJson: arg('from-json', null),
  // Manifest suites: include `heavy: true` cases, check declared expectations,
  // and override the suites' server profile.
  heavy: arg('heavy', false) === true,
  expect: arg('expect', null),
  serverProfile: arg('server-profile', null),
};
/** A bad invocation is a harness error (exit 2), never a FAIL verdict (exit 1). */
function configError(message) {
  console.error(`[render-gate] ${message}`);
  process.exit(2);
}
if (!['IDENTICAL', 'ULP'].includes(opts.cls)) configError(`bad --class ${opts.cls}`);

const repoRoot = execFileSync('git', ['rev-parse', '--show-toplevel'], {
  cwd: viewerRoot,
  encoding: 'utf8',
}).trim();
const manifest = JSON.parse(readFileSync(opts.scenes, 'utf8'));
const log = (m) => console.log(`[render-gate] ${m}`);
const manifestSuites = manifest.suites ?? {};
const BUILT_IN = ['exact', 'perf'];
if (opts.suite !== 'all' && !BUILT_IN.includes(opts.suite) && !manifestSuites[opts.suite]) {
  configError(
    `bad --suite ${opts.suite} (known: ${[...BUILT_IN, ...Object.keys(manifestSuites), 'all'].join(', ')})`
  );
}
/** The manifest suites this run executes, in manifest order. */
const selectedSuites =
  opts.suite === 'all'
    ? Object.keys(manifestSuites)
    : manifestSuites[opts.suite]
      ? [opts.suite]
      : [];
const runsBuiltIn = (name) => opts.suite === 'all' || opts.suite === name;
if (opts.serverProfile && !manifest.serverProfiles?.[opts.serverProfile]) {
  configError(`unknown --server-profile ${opts.serverProfile}`);
}
let expectations = null;
if (opts.expect) {
  try {
    expectations = loadExpectations(opts.expect, manifestSuites);
  } catch (e) {
    configError(e.message);
  }
}
const kit = browserKit({
  channel: opts.channel,
  headless: opts.headless,
  chromeArgs: opts.chromeArgs,
  log,
});
const { seenGpus, openCase } = kit;
const browserSession = kit.browserSession;

// ---------------------------------------------------------------------------
// Decoding captures
// ---------------------------------------------------------------------------

const HALF_TO_FLOAT = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h >>> 10) & 0x1f;
    const m = h & 0x3ff;
    t[h] =
      e === 0
        ? s * m * 2 ** -24
        : e === 0x1f
          ? m
            ? NaN
            : s * Infinity
          : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

function decode(packed) {
  const bytes = Buffer.from(packed.data, 'base64');
  if (packed.format === 'f16') {
    const half = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const out = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) out[i] = HALF_TO_FLOAT[half[i]];
    return out;
  }
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
  );
}

// ---------------------------------------------------------------------------
// Exactness
// ---------------------------------------------------------------------------

function exactVariants(c) {
  const d = manifest.defaults;
  const backends = (opts.backends ?? c.backends ?? d.backends).filter((b) =>
    (c.backends ?? d.backends).includes(b)
  );
  const dsfs = opts.dsf ?? c.dsf ?? d.dsf;
  const out = [];
  for (const backend of backends)
    for (const dsf of dsfs) out.push({ backend, dsf, viewport: c.viewport ?? d.viewport });
  return out;
}

async function captureArm(browser, origin, c, variant) {
  const d = manifest.defaults;
  const { context, page, errors } = await openCase(browser, origin, c, {
    ...variant,
    urlParams: d.urlParams,
  });
  const shots = {};
  try {
    for (const projection of c.projections ?? d.projections) {
      const poses = c.poses ?? manifest.syntheticPoses;
      for (let i = 0; i < poses.length; i++) {
        const { fov, ...pose } = poses[i];
        await page.evaluate(ops.applyView, { projection, fov, pose });
        const st = await page.evaluate(ops.settle, {
          waitEnvironment: !!c.waitEnvironment,
          requireLoaderSettled: !c.synthetic,
        });
        const counts = await page.evaluate(ops.elementCounts);
        const cap = await page.evaluate(ops.captureFrame, { pick: !!c.pick });
        shots[`${projection}#${i}`] = {
          settled: st.settled,
          counts,
          width: cap.width,
          height: cap.height,
          stable: cap.hdrStable && cap.ldrStable,
          hdr: decode(cap.hdr),
          ldr: decode(cap.ldr),
          pick: cap.pick ? decode({ format: 'f32', data: cap.pick.data }) : null,
        };
      }
    }
  } finally {
    await context.close();
  }
  return { shots, errors };
}

function sameCounts(a, b) {
  const ka = Object.keys(a).sort();
  const kb = Object.keys(b).sort();
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && a[k] === b[k]);
}

/** A score without its per-pixel map, for the JSON report. */
function strip(score) {
  const rest = { ...score };
  delete rest.perPixelUlp;
  return rest;
}

async function runExact(session, servers, outDir) {
  const results = [];
  const cases = manifest.exact.filter((c) => !opts.only || opts.only.includes(c.id));
  for (const c of cases) {
    for (const variant of exactVariants(c)) {
      const tag = `${c.id} ${variant.backend} dsf${variant.dsf}`;
      log(`exact: ${tag}`);
      let arms;
      try {
        // Warm-up pass per build, discarded. On ANGLE/Metal the FIRST load of a
        // given scene's programs in a browser renders a few ULP16 differently
        // from every later load (measured: loads 2..N agree bit-for-bit, load 1
        // differs; an unrelated warm-up scene or --disable-gpu-program-cache
        // does not help). Each build's shaders may differ, so each build warms
        // its own programs through the same views before its measured arms.
        arms = await session.run(async (browser) => {
          await captureArm(browser, servers.base.origin, c, variant);
          const base = await captureArm(browser, servers.base.origin, c, variant);
          const base2 = await captureArm(browser, servers.base.origin, c, variant);
          await captureArm(browser, servers.cand.origin, c, variant);
          const cand = await captureArm(browser, servers.cand.origin, c, variant);
          return { base, base2, cand };
        });
      } catch (e) {
        results.push({
          case: c.id,
          ...variant,
          view: '*',
          status: 'error',
          failures: [String(e.message ?? e)],
        });
        continue;
      }
      const cls = opts.intended.includes(c.id) ? 'INTENDED' : opts.cls;
      for (const view of Object.keys(arms.base.shots)) {
        const a = arms.base.shots[view];
        const a2 = arms.base2.shots[view];
        const b = arms.cand.shots[view];
        const row = { case: c.id, backend: variant.backend, dsf: variant.dsf, view, cls };
        const aa = scoreFloatBuffers(a.hdr, a2.hdr);
        const aaLdr = scoreFloatBuffers(a.ldr, a2.ldr);
        const hdr = scoreFloatBuffers(a.hdr, b.hdr);
        const ldr = scoreFloatBuffers(a.ldr, b.ldr);
        const pick =
          a.pick && b.pick && a.pick.length === b.pick.length
            ? scorePickBuffers(a.pick, b.pick)
            : null;
        const blocks = scoreBlocks(a.hdr, b.hdr, a.width, a.height);
        Object.assign(row, { hdr: strip(hdr), ldr: strip(ldr), pick, blocks });
        if (!a.settled || !b.settled || !a2.settled) {
          row.status = 'error';
          row.failures = ['did not settle'];
        } else if (!(hdr.peak > 0)) {
          // An empty frame matches an empty frame and certifies nothing.
          row.status = 'error';
          row.failures = ['baseline frame is empty (nothing on screen)'];
        } else if (c.pick && (!pick || pick.hits === 0)) {
          row.status = 'error';
          row.failures = [pick ? 'pick buffer carries no ids' : 'pick buffer not captured'];
        } else if (!a.stable || !a2.stable || aa.differing > 0 || aaLdr.differing > 0) {
          row.status = 'excluded';
          row.failures = [`nondeterministic baseline (A/A differs on ${aa.differing} HDR px)`];
        } else if (!sameCounts(a.counts, b.counts)) {
          row.status = cls === 'INTENDED' ? 'changed' : 'fail';
          row.failures = ['drawn element counts differ between builds'];
          row.counts = { base: a.counts, cand: b.counts };
        } else if (cls === 'INTENDED') {
          row.status = hdr.differing > 0 || ldr.differing > 0 ? 'changed' : 'unchanged';
        } else {
          const verdict = judge(cls, hdr, ldr, pick, blocks);
          row.status = verdict.pass ? 'pass' : 'fail';
          row.failures = verdict.failures;
        }
        if (hdr.differing > 0) {
          const file = `${c.id}-${variant.backend}-dsf${variant.dsf}-${view.replace('#', '')}.png`;
          writeUlpHeatmap(join(outDir, 'heatmaps', file), hdr.perPixelUlp, a.width, a.height);
          row.heatmap = `heatmaps/${file}`;
        }
        results.push(row);
      }
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

async function measureArm(browser, origin, c, backend) {
  const d = manifest.perfDefaults;
  const variant = {
    backend,
    dsf: c.dsf ?? d.dsf,
    viewport: c.viewport ?? d.viewport,
    urlParams: d.urlParams,
  };
  const { context, page } = await openCase(browser, origin, c, variant);
  try {
    const projection = c.projection ?? d.projection;
    await page.evaluate(ops.applyView, { projection, pose: c.pose ?? d.pose });
    const st = await page.evaluate(ops.settle, {
      minFrames: 30,
      requireLoaderSettled: !c.synthetic,
    });
    if (!st.settled) throw new Error('did not settle');
    const cdp = await context.newCDPSession(page);
    await cdp.send('Performance.enable');
    const script = async () =>
      (await cdp.send('Performance.getMetrics')).metrics.find((m) => m.name === 'ScriptDuration')
        .value;
    // Static GPU cost first (settled pose), then the orbit.
    const gpu = await page.evaluate(ops.gpuCost, {});
    const s0 = await script();
    const mo = await page.evaluate(ops.motion, { pose: c.pose ?? d.pose });
    const s1 = await script();
    const wk = await page.evaluate(ops.wake, {});
    return {
      gpuMs: gpu.minMs,
      frameMs: mo.frameMs,
      frameP95Ms: mo.p95Ms,
      // CDP ScriptDuration does not count the time motion's GPU sync blocks in
      // readPixels (measured: 2.3 ms of script in a 55 ms synced frame), so this
      // stays CPU cost only.
      cpuMs: ((s1 - s0) * 1000) / Math.max(1, mo.frames),
      rendersPerFrame: mo.rendersPerFrame,
      wakeRenders: wk.renders,
      wakeBlockMs: wk.blockMs,
    };
  } finally {
    await context.close();
  }
}

const PERF_METRICS = [
  'gpuMs',
  'frameMs',
  'frameP95Ms',
  'cpuMs',
  'rendersPerFrame',
  'wakeRenders',
  'wakeBlockMs',
];
/**
 * Absolute tolerance per metric, in the metric's own unit, for metrics that
 * sit near the timer's resolution. Chrome coarsens `performance.now()` to
 * 100 µs without cross-origin isolation, so a wake that blocks for ~0 ms reads
 * 0 or 0.1 ms: two quanta are never judged.
 */
const PERF_ABS_TOLERANCE = { wakeBlockMs: 0.2 };

async function runPerf(session, servers) {
  const d = manifest.perfDefaults;
  const rounds = opts.rounds ?? d.rounds;
  const results = [];
  const cases = manifest.perf.filter((c) => !opts.only || opts.only.includes(c.id));
  for (const c of cases) {
    for (const backend of opts.backends ?? d.backends) {
      log(`perf: ${c.id} ${backend} (${rounds} rounds)`);
      const samples = { base: [], base2: [], cand: [] };
      try {
        for (let r = 0; r < rounds; r++) {
          for (const arm of ROTATIONS[r % ROTATIONS.length]) {
            const origin = arm === 'cand' ? servers.cand.origin : servers.base.origin;
            samples[arm].push(
              await session.run((browser) => measureArm(browser, origin, c, backend))
            );
          }
        }
      } catch (e) {
        results.push({ case: c.id, backend, status: 'error', failures: [String(e.message ?? e)] });
        continue;
      }
      const row = { case: c.id, backend, metrics: {} };
      for (const m of PERF_METRICS) {
        const base = samples.base.map((s) => s[m]);
        const base2 = samples.base2.map((s) => s[m]);
        const cand = samples.cand.map((s) => s[m]);
        row.metrics[m] = {
          baseMedian: median(base),
          candMedian: median(cand),
          ...judgePerf(base, cand, noiseFloor(base, base2), PERF_ABS_TOLERANCE[m] ?? 0),
        };
      }
      row.status = Object.values(row.metrics).some((v) => v.verdict === 'fail') ? 'fail' : 'pass';
      results.push(row);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/** A perf statistic, or '-' when it is undefined (a ratio or CI over a zero base median). */
function fmtNum(x, digits) {
  return typeof x === 'number' && Number.isFinite(x) ? x.toFixed(digits) : '-';
}

function fmtScore(s) {
  if (!s) return '';
  return `${s.differing} px, drift ${s.maxDriftUlp.toFixed(1)}, p99.99 ${s.p9999Ulp.toFixed(1)}, flips ${s.flips}`;
}

function markdown(meta, exact, perf, suites) {
  const lines = [
    `# Render gate: ${meta.label}`,
    '',
    `- base: \`${meta.base.ref}\` (${meta.base.sha.slice(0, 10)})`,
  ];
  lines.push(
    `- cand: \`${meta.cand.ref}\` (${meta.cand.sha.slice(0, 10)})`,
    `- class: ${opts.cls}; intended: ${opts.intended.join(', ') || '-'}`
  );
  lines.push(`- host: ${meta.host}; GPU: ${meta.gpu ?? '?'}`);
  if (meta.browserRelaunches > 0) {
    lines.push(
      `- browser relaunched ${meta.browserRelaunches}x after disconnecting (cases retried once)`
    );
  }
  lines.push(`- verdict: **${meta.verdict}**`, '');
  if (exact) {
    lines.push(
      '## Exactness',
      '',
      '| case | backend | dsf | view | class | status | HDR | LDR | pick |',
      '|---|---|---|---|---|---|---|---|---|'
    );
    for (const r of exact) {
      const pick = r.pick ? `node ${r.pick.nodeMismatches} / element ${r.pick.mismatches}` : '';
      lines.push(
        `| ${r.case} | ${r.backend} | ${r.dsf} | ${r.view} | ${r.cls ?? ''} | ${r.status}${r.failures?.length ? ` (${r.failures.join('; ')})` : ''} | ${fmtScore(r.hdr)} | ${fmtScore(r.ldr)} | ${pick} |`
      );
    }
    lines.push('');
  }
  if (perf) {
    lines.push(
      '## Performance (ratio cand/base, 95% CI, A/A floor)',
      '',
      '`frameMs` / `frameP95Ms` INCLUDE GPU completion: each orbit frame ends with a GPU sync, so a ' +
        'frame costs its CPU plus GPU work, serialized. `gpuMs` is GPU cost alone (settled pose); ' +
        '`cpuMs` is script time alone.',
      '',
      '| case | backend | metric | base | cand | ratio | CI | floor | verdict |',
      '|---|---|---|---|---|---|---|---|---|'
    );
    for (const r of perf) {
      if (r.status === 'error') {
        lines.push(`| ${r.case} | ${r.backend} | - | | | | | | error: ${r.failures.join('; ')} |`);
        continue;
      }
      for (const [m, v] of Object.entries(r.metrics)) {
        lines.push(
          `| ${r.case} | ${r.backend} | ${m} | ${fmtNum(v.baseMedian, 3)} | ${fmtNum(v.candMedian, 3)} | ${fmtNum(v.ratio, 3)} | ${fmtNum(v.lo, 3)}–${fmtNum(v.hi, 3)} | ${typeof v.floor === 'number' ? `${fmtNum(v.floor * 100, 1)}%` : '-'} | ${v.verdict} |`
        );
      }
    }
    lines.push('');
  }
  for (const [name, rows] of Object.entries(suites ?? {})) lines.push(...suiteMarkdown(name, rows));
  return `${lines.join('\n')}\n`;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const cacheRoot = join(repoRoot, 'delme', 'gate-builds');
  const prebuilt = (dir, label) => ({ ref: label, sha: `dist:${dir}`, distDir: resolve(dir) });
  const base = opts.baseDist
    ? prebuilt(opts.baseDist, 'base-dist')
    : ensureBuild({ repoRoot, cacheRoot, ref: opts.base, log });
  const cand = opts.candDist
    ? prebuilt(opts.candDist, 'cand-dist')
    : ensureBuild({ repoRoot, cacheRoot, ref: opts.cand, log });
  const label = opts.label ?? `${base.sha.slice(0, 8)}-vs-${cand.sha.slice(0, 8)}-${opts.suite}`;
  const outDir = resolve(arg('out', join(repoRoot, 'delme', 'gate', label)));
  mkdirSync(outDir, { recursive: true });

  const dataRoot = join(repoRoot, 'datasets');
  let exact = null;
  let perf = null;
  const suites = {};
  let browserRelaunches = 0;

  if (runsBuiltIn('exact') || runsBuiltIn('perf')) {
    const servers = {
      base: await startServer({ distDir: base.distDir, dataRoot, port: opts.basePort }),
      cand: await startServer({ distDir: cand.distDir, dataRoot, port: opts.basePort + 1 }),
    };
    try {
      if (runsBuiltIn('exact')) {
        const session = browserSession(false);
        try {
          exact = await runExact(session, servers, outDir);
        } finally {
          browserRelaunches += session.relaunches();
          await session.close();
        }
      }
      if (runsBuiltIn('perf')) {
        const session = browserSession(true);
        try {
          perf = await runPerf(session, servers);
        } finally {
          browserRelaunches += session.relaunches();
          await session.close();
        }
      }
    } finally {
      await servers.base.close();
      await servers.cand.close();
    }
  }

  // Manifest suites: each gets its own server pair on the same two ports,
  // configured by its server profile (a hosted suite simulates the link).
  for (const name of selectedSuites) {
    const suite = manifestSuites[name];
    const profileName = opts.serverProfile ?? suite.defaults?.serverProfile ?? 'local';
    const profile = manifest.serverProfiles?.[profileName];
    if (!profile) throw new Error(`suite ${name}: unknown server profile ${profileName}`);
    const certDir = join(repoRoot, 'delme', 'gate-certs');
    const servers = {
      base: await startServer({
        distDir: base.distDir,
        dataRoot,
        port: opts.basePort,
        certDir,
        ...profile,
      }),
      cand: await startServer({
        distDir: cand.distDir,
        dataRoot,
        port: opts.basePort + 1,
        certDir,
        ...profile,
      }),
    };
    const session = browserSession(false, profile.h2 ? ['--ignore-certificate-errors'] : []);
    try {
      suites[name] = await runSuite({
        name,
        suite,
        profile: { name: profileName, ...profile },
        session,
        openCase,
        servers,
        dataRoot,
        opts,
        expectations,
        log,
      });
    } finally {
      browserRelaunches += session.relaunches();
      await session.close();
      await servers.base.close();
      await servers.cand.close();
    }
  }

  const rows = [...(exact ?? []), ...(perf ?? []), ...Object.values(suites).flat()];
  const failed = rows.some((r) => r.status === 'fail' || r.status === 'error');
  // An excluded row is a view the gate could not see (nondeterministic
  // baseline). It must never read as a pass: the gate is INCOMPLETE until
  // the nondeterminism is fixed or the case is dropped from the manifest.
  const blind = rows.some((r) => r.status === 'excluded');
  const verdict = failed ? 'FAIL' : blind ? 'INCOMPLETE' : 'PASS';
  const meta = {
    label,
    base: { ref: base.ref, sha: base.sha },
    cand: { ref: cand.ref, sha: cand.sha },
    host: `${process.platform}/${process.arch}`,
    gpu: [...seenGpus].join(' | '),
    browserRelaunches,
    verdict,
    options: opts,
  };
  writeFileSync(
    join(outDir, 'report.json'),
    JSON.stringify({ meta, exact, perf, suites }, null, 2)
  );
  writeFileSync(join(outDir, 'report.md'), markdown(meta, exact, perf, suites));
  log(`${verdict}: ${join(outDir, 'report.md')}`);
  process.exit(verdict === 'PASS' ? 0 : verdict === 'FAIL' ? 1 : 3);
}

function rerender(jsonPath) {
  const { meta, exact, perf, suites } = JSON.parse(readFileSync(jsonPath, 'utf8'));
  const mdPath = join(dirname(resolve(jsonPath)), 'report.md');
  writeFileSync(mdPath, markdown(meta, exact, perf, suites));
  log(`${meta.verdict}: ${mdPath}`);
}

(opts.fromJson ? Promise.resolve().then(() => rerender(opts.fromJson)) : main()).catch((e) => {
  console.error(e);
  process.exit(2);
});
