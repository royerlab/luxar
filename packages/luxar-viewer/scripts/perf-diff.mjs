#!/usr/bin/env node
/**
 * Compare two perf-bench result files and print a Markdown delta table.
 *
 * Usage:
 *   node scripts/perf-diff.mjs <baseline-results.json> <new-results.json>
 *
 * Reads JSON of the shape produced by
 * `src/tests/e2e/line-perf-bench.spec.ts` and writes a Markdown table
 * to stdout suitable for pasting into a commit body. Scenarios that
 * don't appear in BOTH inputs are reported but not deltaed.
 */

import * as fs from 'fs';
import * as path from 'path';

function usage() {
  // eslint-disable-next-line no-console
  console.error('Usage: perf-diff.mjs <baseline.json> <new.json>');
  process.exit(2);
}

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

const fmt = (n) => (typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '—');

/**
 * Compute "% change" of newVal vs baseVal where lower is better.
 * Returns a string with sign and percent, e.g. "-7.3% ✅".
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
  const api = isWebGLBackend ? `${apiSurface} (webgl-bk)` : apiSurface;
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
    const api = isWebGLBackend ? `${apiSurface} (webgl-bk)` : apiSurface;
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

md += '\n';
md += `🟢 = ≥5% faster on this metric · 🔴 = ≥5% slower\n`;

// eslint-disable-next-line no-console
console.log(md);

// Resolve any relative paths for the user.
const baseAbs = path.resolve(basePath);
const newAbs = path.resolve(newPath);
// eslint-disable-next-line no-console
console.error(`baseline: ${baseAbs}\nnew:      ${newAbs}`);
