#!/usr/bin/env node
/**
 * Compare two perf-bench result files and print a Markdown delta table.
 *
 * Usage:
 *   node scripts/perf-diff.mjs <baseline-results.json> <new-results.json>
 *
 * Reads JSON of the shape produced by
 * `src/tests/e2e/line-perf-bench.spec.ts` or
 * `src/tests/e2e/gsplat-perf-bench.spec.ts` and writes a Markdown table
 * to stdout suitable for pasting into a commit body. Scenarios that
 * don't appear in BOTH inputs are reported but not deltaed. A row whose
 * two sides disagree on the presence of `excludedResolveIntervals`
 * straddles the line bench's GPU-timestamp exclusion change and is
 * called out as not comparable.
 *
 * The markdown-building logic lives in the exported pure function
 * {@link buildPerfDiff} (so it can be unit-tested); all CLI behaviour
 * (arg parsing, file reads, stdout/stderr) runs inside {@link main},
 * which only executes when the script is run directly.
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const fmt = (n) => (typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '—');

/**
 * Compute "% change" of newVal vs baseVal where lower is better.
 * Returns a string with sign and percent, e.g. "-7.3% ✅".
 * Returns "—" for any non-numeric side (missing/null) and for a zero
 * baseline — never a phantom "0.0%".
 */
function delta(baseVal, newVal) {
  if (typeof baseVal !== 'number' || typeof newVal !== 'number') return '—';
  if (baseVal === 0) return '—';
  const pct = ((newVal - baseVal) / baseVal) * 100;
  const sign = pct > 0 ? '+' : '';
  let mark = '';
  if (pct <= -5) mark = ' 🟢';
  else if (pct >= 5) mark = ' 🔴';
  return `${sign}${pct.toFixed(1)}%${mark}`;
}

function keyOf(scn) {
  return `${scn.scenarioId}/${scn.backend}`;
}

/**
 * API-column text for a scenario pair. Appends ` (webgl-bk)` when a
 * WebGPURenderer run fell back to its internal WebGL2 backend, and
 * ` (sw)` when EITHER side ran on a software rasterizer
 * (`softwareRenderer`, gsplat bench only) — the bench states such
 * rows' absolute timings are not comparable to a GPU run, so a diff
 * involving one must be visibly discountable.
 */
function apiOf(apiSurface, isWebGLBackend, b, n) {
  const sw = b?.softwareRenderer === true || n?.softwareRenderer === true;
  return (isWebGLBackend ? `${apiSurface} (webgl-bk)` : apiSurface) + (sw ? ' (sw)' : '');
}

/**
 * Build a generic "metric" section: one row per scenario, and for each
 * field that AT LEAST ONE scenario carries (on either side) three
 * columns (base / new / Δ). Fields absent everywhere are dropped; a
 * field missing on one side of a present column renders `—` (via
 * {@link fmt} / {@link delta}), never a phantom 0.0%. Rows whose
 * scenario carries none of the present fields are skipped. Lower is
 * better for every field. Returns '' when nothing is present.
 *
 * @param {string} title      section heading
 * @param {string} unitsNote  one-line note under the heading (units etc.)
 * @param {string[]} sortedKeys
 * @param {Map} baseByKey
 * @param {Map} nextByKey
 * @param {(scn: any) => any} getObj  extracts the field-holder from a scenario
 * @param {{label: string, key: string}[]} fields
 */
function metricSection(title, unitsNote, sortedKeys, baseByKey, nextByKey, getObj, fields) {
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const present = fields.filter(({ key }) =>
    sortedKeys.some((k) => {
      const b = getObj(baseByKey.get(k));
      const n = getObj(nextByKey.get(k));
      return isNum(b?.[key]) || isNum(n?.[key]);
    })
  );
  if (present.length === 0) return '';

  let header = '| Scenario / backend |';
  let sep = '|---|';
  for (const { label } of present) {
    header += ` base ${label} | new ${label} | Δ ${label} |`;
    sep += '---:|---:|---|';
  }

  let md = `\n## ${title}\n\n`;
  md += `${unitsNote}\n\n`;
  md += `${header}\n`;
  md += `${sep}\n`;

  for (const k of sortedKeys) {
    const b = getObj(baseByKey.get(k));
    const n = getObj(nextByKey.get(k));
    // Skip scenarios that carry none of the present fields on either
    // side (e.g. non-10M rows in the L8 sort-tail section).
    const hasAny = present.some(({ key }) => isNum(b?.[key]) || isNum(n?.[key]));
    if (!hasAny) continue;

    let row = `| ${k} |`;
    for (const { key } of present) {
      row += ` ${fmt(b?.[key])} | ${fmt(n?.[key])} | ${delta(b?.[key], n?.[key])} |`;
    }
    md += `${row}\n`;
  }
  return md;
}

/**
 * Turn two parsed perf-bench result objects into the Markdown delta
 * report. Pure: no I/O, no process access.
 *
 * @param {any} base parsed baseline PerfRunResult
 * @param {any} next parsed new PerfRunResult
 * @returns {string} markdown
 */
export function buildPerfDiff(base, next) {
  const baseByKey = new Map((base.scenarios ?? []).map((s) => [keyOf(s), s]));
  const nextByKey = new Map((next.scenarios ?? []).map((s) => [keyOf(s), s]));

  const allKeys = new Set([...baseByKey.keys(), ...nextByKey.keys()]);
  const sortedKeys = [...allKeys].sort();

  let md = '';
  md += `# Perf diff: \`${base.commit}\` → \`${next.commit}\`\n\n`;
  md += `Baseline captured: ${base.capturedAt}\n`;
  md += `New captured: ${next.capturedAt}\n\n`;
  md += `Sample window: ${next.sampleWindowMs ?? '?'} ms, warmup: ${next.warmupFrames ?? '?'} frames.\n\n`;

  md += `## JS frame timing\n\n`;
  md += `| Scenario / backend | API | Segs | base median (ms) | new median (ms) | Δ median | base p95 | new p95 | Δ p95 |\n`;
  md += `|---|---|---:|---:|---:|---|---:|---:|---|\n`;

  for (const k of sortedKeys) {
    const b = baseByKey.get(k);
    const n = nextByKey.get(k);
    const ref = n ?? b;
    if (!ref) continue;

    // Surface the physical-backend flag alongside `apiSurface`. A
    // WebGPURenderer run that fell back to its internal WebGL2 backend
    // would otherwise look like a clean "webgpu" row, masking the
    // distinction the perf-bench JSON deliberately captures.
    const apiSurface = n?.actualApi ?? b?.actualApi ?? '?';
    const isWebGLBackend = n?.isWebGLBackend ?? b?.isWebGLBackend ?? false;
    const api = apiOf(apiSurface, isWebGLBackend, b, n);
    const segs = n?.visibleSegments ?? b?.visibleSegments ?? 0;

    if (n?.skipped && b?.skipped) {
      md += `| ${k} | ${api} | ${segs} | SKIP | SKIP | — | — | — | — |\n`;
      continue;
    }
    if (!b) {
      const nm = n.frameMs?.median;
      const np = n.frameMs?.p95;
      md += `| ${k} | ${api} | ${segs} | — | ${fmt(nm)} | NEW | — | ${fmt(np)} | NEW |\n`;
      continue;
    }
    if (!n) {
      const bm = b.frameMs?.median;
      const bp = b.frameMs?.p95;
      md += `| ${k} | ${api} | ${segs} | ${fmt(bm)} | — | DROPPED | ${fmt(bp)} | — | DROPPED |\n`;
      continue;
    }

    const bm = b.frameMs?.median;
    const nm = n.frameMs?.median;
    const bp = b.frameMs?.p95;
    const np = n.frameMs?.p95;

    md += `| ${k} | ${api} | ${segs} | ${fmt(bm)} | ${fmt(nm)} | ${delta(bm, nm)} | ${fmt(bp)} | ${fmt(np)} | ${delta(bp, np)} |\n`;
  }

  // The line bench excludes the frame interval that follows each GPU
  // timestamp resolve (the readback latency lands there, not in the
  // scene's work) and records the drop count as
  // `excludedResolveIntervals`. That field did not exist before the
  // exclusion did, so a row whose two sides disagree on its PRESENCE
  // straddles the change: the older side's frame stats still include
  // readback latency, and the deltas above are instrument drift, not a
  // rendering change. Warn rather than let a −86% p95 read as a win.
  // Only measured rows are considered — a skipped row has no timing to
  // compare and omits the field on purpose.
  //
  // The test is presence-only, and deliberately conservative: whether the
  // OLDER side resolved timestamps at all is not recoverable from its
  // JSON (a run with the feature enabled but no usable samples reports
  // `gpu.supported: false` all the same), so an arm that never resolved —
  // every WebGL row — gets flagged too. The message says so rather than
  // asserting readback latency in stats that cannot contain it.
  const straddling = sortedKeys.filter((k) => {
    const b = baseByKey.get(k);
    const n = nextByKey.get(k);
    if (!b || !n || b.skipped || n.skipped) return false;
    return 'excludedResolveIntervals' in b !== 'excludedResolveIntervals' in n;
  });
  if (straddling.length > 0) {
    md += `\n⚠️ JS frame timing not comparable for ${straddling.join(', ')}: one side predates the bench's exclusion of GPU-timestamp resolve intervals, so its frame stats still include readback latency wherever that run resolved timestamps (p95/p99 worst; an arm that never resolved — any WebGL row — is listed conservatively and is in fact comparable). Re-measure that side before reading these deltas.\n`;
  }

  // GPU-time section. Only emitted when at least one row has a real
  // `gpu.medianMs` on either side — JS-only runs still show the JS
  // table above.
  const anyGpu = [...allKeys].some((k) => {
    const b = baseByKey.get(k);
    const n = nextByKey.get(k);
    return b?.gpu?.supported || n?.gpu?.supported;
  });

  if (anyGpu) {
    md += `\n## GPU pass time (timestamp-query)\n\n`;
    md += `| Scenario / backend | API | Segs | base median (ms) | new median (ms) | Δ median | base p95 | new p95 | Δ p95 |\n`;
    md += `|---|---|---:|---:|---:|---|---:|---:|---|\n`;

    for (const k of sortedKeys) {
      const b = baseByKey.get(k);
      const n = nextByKey.get(k);
      const ref = n ?? b;
      if (!ref) continue;
      const apiSurface = n?.actualApi ?? b?.actualApi ?? '?';
      const isWebGLBackend = n?.isWebGLBackend ?? b?.isWebGLBackend ?? false;
      const api = apiOf(apiSurface, isWebGLBackend, b, n);
      const segs = n?.visibleSegments ?? b?.visibleSegments ?? 0;

      const bSup = b?.gpu?.supported === true;
      const nSup = n?.gpu?.supported === true;
      if (!bSup && !nSup) {
        md += `| ${k} | ${api} | ${segs} | n/a | n/a | — | n/a | n/a | — |\n`;
        continue;
      }
      const bm = bSup ? b.gpu.medianMs : null;
      const nm = nSup ? n.gpu.medianMs : null;
      const bp = bSup ? b.gpu.p95Ms : null;
      const np = nSup ? n.gpu.p95Ms : null;
      md += `| ${k} | ${api} | ${segs} | ${fmt(bm)} | ${fmt(nm)} | ${delta(bm, nm)} | ${fmt(bp)} | ${fmt(np)} | ${delta(bp, np)} |\n`;
    }
  }

  // Depth-sort worker-stage medians + end-to-end sort latency. Only the
  // columns some scenario actually carries are shown (worker-stage
  // breakdown is optional; sort-latency may be null). Lower is better.
  md += metricSection(
    'Viewer audit (load + frames)',
    'Rows from viewer-audit-perf-bench.spec.ts: load milestones in ms from loadStart (lower is better), request/byte counts, main-thread long-task ms during the load, forced-continuous-render frame p50 (ms) at DPR 1 / 0.5 / dollied 4x, playback step ms (first commit / settled), OPFS write drops.',
    sortedKeys,
    baseByKey,
    nextByKey,
    (scn) => scn?.audit,
    [
      { label: 'ttfp', key: 'ttfpMs' },
      { label: 'sceneLoaded', key: 'sceneLoadedMs' },
      { label: 'initUpdate', key: 'initUpdateDoneMs' },
      { label: 'refined', key: 'refinementCompleteMs' },
      { label: 'stateReady', key: 'stateReadyMs' },
      { label: 'warmLoaded', key: 'warmSceneLoadedMs' },
      { label: 'requests', key: 'requests' },
      { label: 'bytes', key: 'bytes' },
      { label: 'longTask', key: 'longTaskMs' },
      { label: 'frame@1', key: 'frameP50Ms_dpr1' },
      { label: 'frame@0.5', key: 'frameP50Ms_dpr05' },
      { label: 'frame@zoom', key: 'frameP50Ms_zoom4x' },
      { label: 'playFirst', key: 'playbackFirstMs_warm' },
      { label: 'playFull', key: 'playbackFullMs_warm' },
      { label: 'opfsDrop', key: 'opfsDropped' },
      { label: 'steadyDPR', key: 'steadyDpr' },
    ]
  );

  md += metricSection(
    'Depth-sort stages',
    'Per-scenario depth-sort timing (ms, lower is better). Worker-stage medians are optional; sort-latency may be absent.',
    sortedKeys,
    baseByKey,
    nextByKey,
    (scn) => scn?.depthSort,
    [
      { label: 'kernel', key: 'kernelMsMedian' },
      { label: 'queue', key: 'queueMsMedian' },
      { label: 'boundary', key: 'boundaryMsMedian' },
      { label: 'sortLat med', key: 'sortLatencyMedianMs' },
      { label: 'sortLat p95', key: 'sortLatencyP95Ms' },
    ]
  );

  // L8 sort-tail p99s (10M scenario only). Top-level scenario fields.
  md += metricSection(
    'L8 sort-tail (p99)',
    'Sort-tail p99 latencies (ms, lower is better) — present on the 10M scenario only.',
    sortedKeys,
    baseByKey,
    nextByKey,
    (scn) => scn,
    [
      { label: 'sortAdjacent p99', key: 'sortAdjacentP99Ms' },
      { label: 'idleOrbit p99', key: 'idleOrbitP99Ms' },
    ]
  );

  // Ladder-load section (visible-human ladder scenario only). Bespoke
  // because `observedGrowth: false` makes the wall time a lower bound,
  // not a measurement — those cells get a `(lb)` marker and no delta.
  const anyLadder = sortedKeys.some((k) => baseByKey.get(k)?.ladder || nextByKey.get(k)?.ladder);
  if (anyLadder) {
    md += `\n## Ladder load\n\n`;
    md += `Wall time from navigation to ladder-complete (ms, lower is better). \`(lb)\` = lower bound: the ladder finished before polling began (\`observedGrowth: false\`), so no delta is computed.\n\n`;
    md += `| Scenario / backend | base wallMs | new wallMs | Δ wallMs |\n`;
    md += `|---|---:|---:|---|\n`;
    for (const k of sortedKeys) {
      const b = baseByKey.get(k)?.ladder;
      const n = nextByKey.get(k)?.ladder;
      if (!b && !n) continue;
      const bVal = b?.wallMsToLadderComplete;
      const nVal = n?.wallMsToLadderComplete;
      const bGrew = b?.observedGrowth === true;
      const nGrew = n?.observedGrowth === true;
      const bCell =
        typeof bVal === 'number' && Number.isFinite(bVal)
          ? `${fmt(bVal)}${bGrew ? '' : ' (lb)'}`
          : '—';
      const nCell =
        typeof nVal === 'number' && Number.isFinite(nVal)
          ? `${fmt(nVal)}${nGrew ? '' : ' (lb)'}`
          : '—';
      // Skip the delta whenever either present side is a lower bound —
      // comparing against a lower bound would mislead.
      const d = b && n && (!bGrew || !nGrew) ? '—' : delta(bVal, nVal);
      md += `| ${k} | ${bCell} | ${nCell} | ${d} |\n`;
    }
  }

  md += '\n';
  md += `🟢 = ≥5% faster on this metric · 🔴 = ≥5% slower\n`;

  return md;
}

function usage() {
  // eslint-disable-next-line no-console
  console.error('Usage: perf-diff.mjs <baseline.json> <new.json>');
  process.exit(2);
}

function main() {
  if (process.argv.length < 4) usage();

  const [, , basePath, newPath] = process.argv;
  if (!fs.existsSync(basePath)) {
    // eslint-disable-next-line no-console
    console.error(`Baseline not found: ${basePath}`);
    process.exit(2);
  }
  if (!fs.existsSync(newPath)) {
    // eslint-disable-next-line no-console
    console.error(`New results not found: ${newPath}`);
    process.exit(2);
  }

  const base = JSON.parse(fs.readFileSync(basePath, 'utf8'));
  const next = JSON.parse(fs.readFileSync(newPath, 'utf8'));

  const md = buildPerfDiff(base, next);

  // eslint-disable-next-line no-console
  console.log(md);

  // Resolve any relative paths for the user.
  const baseAbs = path.resolve(basePath);
  const newAbs = path.resolve(newPath);
  // eslint-disable-next-line no-console
  console.error(`baseline: ${baseAbs}\nnew:      ${newAbs}`);
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) main();
