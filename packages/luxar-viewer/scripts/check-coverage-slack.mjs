#!/usr/bin/env node
/**
 * Coverage-floor slack guard.
 *
 * A coverage threshold only gates anything while it sits just under the value
 * it measures. Left alone it decays: tests accumulate, the measured number
 * climbs, the floor stays put, and eventually the gate would pass with a third
 * of the suite deleted. That is precisely what had happened here — the viewer
 * floors sat 17 points under measured before this guard existed.
 *
 * So: fail when a floor falls more than `MAX_SLACK_POINTS` below its measured
 * value, and print the block that would fix it.
 *
 * It also closes a fail-open hole that vitest itself does not: a glob key
 * matching zero files yields pct `"Unknown"`, and `"Unknown" < 86` is `false`,
 * so a renamed directory silently turns its gate into one that inspects
 * nothing. Verified empirically against vitest 4.1.10. Every glob key is
 * therefore asserted to match at least one file.
 *
 * Usage: node scripts/check-coverage-slack.mjs [--max-slack N] [--summary PATH]
 * Reads the `json-summary` reporter output that `pnpm test:coverage` emits.
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import { COVERAGE_THRESHOLDS, MAX_SLACK_POINTS, METRICS } from '../coverage-thresholds.mjs';

const DEFAULT_SUMMARY = 'coverage/coverage-summary.json';

/** True when a threshold entry is a per-glob sub-gate rather than a global metric. */
export function isGlobKey(key) {
  return !METRICS.includes(key);
}

/**
 * Match a coverage-summary path against a vitest threshold glob.
 *
 * vitest resolves glob keys with picomatch against paths relative to the
 * config root. Only the `dir/**` shape is used in coverage-thresholds.mjs, so
 * a prefix test is exact here and avoids taking on a matcher dependency.
 */
export function matchesGlob(relativePath, glob) {
  if (!glob.endsWith('/**')) {
    return relativePath === glob;
  }
  return relativePath.startsWith(glob.slice(0, -2));
}

/** Reduce absolute covered/total counts for a set of summary entries. */
export function aggregate(entries, metric) {
  let covered = 0;
  let total = 0;
  for (const entry of entries) {
    covered += entry[metric].covered;
    total += entry[metric].total;
  }
  return { covered, total, pct: total === 0 ? null : (100 * covered) / total };
}

/** Normalize summary keys to config-root-relative POSIX paths. */
export function relativizeSummary(summary, viewerRoot) {
  const root = path.resolve(viewerRoot).replaceAll('\\', '/').replace(/\/$/, '');
  const out = new Map();
  for (const [key, value] of Object.entries(summary)) {
    if (key === 'total') continue;
    const normalized = key.replaceAll('\\', '/');
    out.set(normalized.startsWith(root) ? normalized.slice(root.length + 1) : normalized, value);
  }
  return out;
}

/**
 * Compare every floor against the measurement.
 *
 * Returns `{ failures, rows }`. A failure is either excessive slack or — the
 * hole vitest leaves open — a glob that matched nothing.
 */
export function evaluate(summary, thresholds, viewerRoot, maxSlack) {
  const files = relativizeSummary(summary, viewerRoot);
  const all = [...files.values()];
  const failures = [];
  const rows = [];

  for (const [key, value] of Object.entries(thresholds)) {
    const glob = isGlobKey(key);
    const entries = glob
      ? [...files.entries()].filter(([p]) => matchesGlob(p, key)).map(([, v]) => v)
      : all;

    if (glob && entries.length === 0) {
      failures.push(
        `${key}: matched 0 files. vitest reports pct "Unknown" for an empty ` +
          `group and passes it silently, so this gate currently inspects ` +
          `nothing. Fix the glob or delete the entry.`
      );
      continue;
    }

    const constraints = glob ? value : { [key]: value };
    for (const [metric, floor] of Object.entries(constraints)) {
      const { pct } = aggregate(entries, metric);
      if (pct === null) continue;
      const slack = pct - floor;
      rows.push({ key, metric, floor, measured: pct, slack, files: entries.length });
      if (slack > maxSlack) {
        failures.push(
          `${key} ${metric}: floor ${floor} is ${slack.toFixed(1)} pts under ` +
            `measured ${pct.toFixed(2)} (budget ${maxSlack}). Raise it to ` +
            `${Math.floor(pct - 1)}.`
        );
      }
      if (slack < 0) {
        failures.push(
          `${key} ${metric}: measured ${pct.toFixed(2)} is BELOW floor ${floor}. ` +
            `vitest should already have failed; check the reporter output.`
        );
      }
    }
  }
  return { failures, rows };
}

export function parseArgs(argv) {
  const args = { maxSlack: MAX_SLACK_POINTS, summary: DEFAULT_SUMMARY };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--max-slack') {
      const value = Number(argv[(i += 1)]);
      if (!Number.isFinite(value)) {
        throw new Error('--max-slack requires a finite number');
      }
      args.maxSlack = value;
    } else if (argv[i] === '--summary') args.summary = argv[(i += 1)];
  }
  return args;
}

async function main() {
  const { maxSlack, summary: summaryPath } = parseArgs(process.argv.slice(2));
  const viewerRoot = path.resolve(import.meta.dirname, '..');
  const resolved = path.resolve(viewerRoot, summaryPath);

  let summary;
  try {
    summary = JSON.parse(await readFile(resolved, 'utf8'));
  } catch (error) {
    // Fail closed. An unreadable summary means the guard inspected nothing,
    // and a gate that passes having inspected nothing is the bug class this
    // script exists to close.
    console.error(
      `check-coverage-slack: cannot read ${resolved}.\n` +
        `Run \`pnpm test:coverage\` first (it emits the json-summary reporter).\n` +
        `${error.message}`
    );
    process.exit(1);
  }

  const { failures, rows } = evaluate(summary, COVERAGE_THRESHOLDS, viewerRoot, maxSlack);

  const widest = Math.max(...rows.map((r) => r.key.length));
  for (const r of rows.sort((a, b) => b.slack - a.slack)) {
    const flag = r.slack > maxSlack ? ' <-- stale' : '';
    console.log(
      `  ${r.key.padEnd(widest)}  ${r.metric.padEnd(10)} floor ${String(r.floor).padStart(3)}` +
        `  measured ${r.measured.toFixed(2).padStart(6)}  slack ${r.slack.toFixed(1).padStart(5)}${flag}`
    );
  }

  if (failures.length > 0) {
    console.error(`\ncheck-coverage-slack: ${failures.length} problem(s)\n`);
    for (const f of failures) console.error(`  - ${f}`);
    console.error(
      `\nCoverage floors must stay within ${maxSlack} pts of measured, or they ` +
        `stop gating anything. Update packages/luxar-viewer/coverage-thresholds.mjs.`
    );
    process.exit(1);
  }
  console.log(
    `\ncheck-coverage-slack: OK — ${rows.length} floors all within ${maxSlack} pts of measured.`
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
