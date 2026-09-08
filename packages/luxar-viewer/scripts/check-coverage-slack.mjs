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
 * Separately, warn when measured coverage falls more than
 * `MAX_EROSION_POINTS` below the value recorded when the floor was set. That
 * catches downward drift before the floor is crossed without blocking a
 * legitimate refactor.
 *
 * It also closes a fail-open hole that vitest itself does not: a glob key
 * matching zero files yields pct `"Unknown"`, and `"Unknown" < 86` is `false`,
 * so a renamed directory silently turns its gate into one that inspects
 * nothing. Verified empirically against vitest 4.1.10. Every glob key is
 * therefore asserted to match at least one file and each configured metric is
 * asserted to have at least one countable item.
 *
 * Usage: node scripts/check-coverage-slack.mjs [--max-slack N] [--summary PATH]
 * Reads the `json-summary` reporter output that `pnpm test:coverage` emits.
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

import {
  COVERAGE_RECORDED,
  COVERAGE_THRESHOLDS,
  MAX_EROSION_POINTS,
  MAX_SLACK_POINTS,
  METRICS,
} from '../coverage-thresholds.mjs';

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

function compareKeys(expected, actual, label) {
  const missing = expected.filter((key) => !actual.includes(key));
  const extra = actual.filter((key) => !expected.includes(key));
  if (missing.length === 0 && extra.length === 0) return null;
  const details = [
    missing.length > 0 ? `missing ${missing.join(', ')}` : '',
    extra.length > 0 ? `extra ${extra.join(', ')}` : '',
  ].filter(Boolean);
  return `${label}: ${details.join('; ')}`;
}

/** Assert that every floor has exactly one structured recorded baseline. */
export function validateRecorded(thresholds, recorded) {
  const failures = [];
  const topLevel = compareKeys(
    Object.keys(thresholds),
    Object.keys(recorded),
    'coverage recorded baselines must have exactly the threshold keys'
  );
  if (topLevel) failures.push(topLevel);

  for (const [key, value] of Object.entries(thresholds)) {
    if (!(key in recorded)) continue;
    if (!isGlobKey(key)) {
      if (!Number.isFinite(recorded[key])) {
        failures.push(`${key} recorded baseline must be a finite number`);
      }
      continue;
    }
    if (typeof recorded[key] !== 'object' || recorded[key] === null) {
      failures.push(`${key} recorded baseline must be a metric map`);
      continue;
    }
    const metrics = compareKeys(
      Object.keys(value),
      Object.keys(recorded[key]),
      `${key} recorded baselines must have exactly the threshold metrics`
    );
    if (metrics) failures.push(metrics);
    for (const metric of Object.keys(value)) {
      if (metric in recorded[key] && !Number.isFinite(recorded[key][metric])) {
        failures.push(`${key} ${metric} recorded baseline must be a finite number`);
      }
    }
  }
  return failures;
}

/**
 * Compare every floor against the measurement.
 *
 * Returns `{ failures, rows }`. A failure is either excessive slack or — the
 * holes vitest leaves open — a glob that matched nothing or a configured glob
 * metric with no countable items.
 */
export function evaluate(summary, thresholds, recorded, viewerRoot, maxSlack, maxErosion) {
  const files = relativizeSummary(summary, viewerRoot);
  const all = [...files.values()];
  const failures = validateRecorded(thresholds, recorded);
  const warnings = [];
  const rows = [];

  for (const [key, value] of Object.entries(thresholds)) {
    const glob = isGlobKey(key);
    const entries = glob
      ? [...files.entries()].filter(([p]) => matchesGlob(p, key)).map(([, v]) => v)
      : all;

    if (glob && entries.length === 0) {
      failures.push(
        `${key}: matched 0 files. vitest reports pct "Unknown" for an empty ` +
          'group and passes it silently, so this gate currently inspects ' +
          'nothing. Fix the glob or delete the entry.'
      );
      continue;
    }

    const constraints = glob ? value : { [key]: value };
    for (const [metric, floor] of Object.entries(constraints)) {
      const { pct } = aggregate(entries, metric);
      if (pct === null) {
        if (glob) {
          failures.push(
            `${key}: ${metric} has 0 countable items across ${entries.length} ` +
              'matched file(s), so this floor currently inspects nothing. ' +
              'Fix the glob or delete the metric floor.'
          );
        }
        continue;
      }
      const slack = pct - floor;
      const baseline = glob ? recorded[key]?.[metric] : recorded[key];
      const erosion = typeof baseline === 'number' ? baseline - pct : null;
      rows.push({ key, metric, floor, recorded: baseline, measured: pct, slack, erosion });
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
            'vitest should already have failed; check the reporter output.'
        );
      }
      if (erosion !== null && erosion > maxErosion) {
        const label = glob ? `${key} ${metric}` : key;
        warnings.push(
          `${label}: measured ${pct.toFixed(2)} is ${erosion.toFixed(2)} pts below ` +
            `recorded ${baseline.toFixed(2)} (budget ${maxErosion}).`
        );
      }
    }
  }
  return { failures, warnings, rows };
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
        'Run `pnpm test:coverage` first (it emits the json-summary reporter).\n' +
        `${error.message}`
    );
    process.exit(1);
  }

  const { failures, warnings, rows } = evaluate(
    summary,
    COVERAGE_THRESHOLDS,
    COVERAGE_RECORDED,
    viewerRoot,
    maxSlack,
    MAX_EROSION_POINTS
  );

  const widest = Math.max(...rows.map((r) => r.key.length));
  for (const r of rows.sort((a, b) => b.slack - a.slack)) {
    const flags = [
      r.slack > maxSlack ? 'stale floor' : '',
      r.erosion !== null && r.erosion > MAX_EROSION_POINTS ? 'eroded' : '',
    ].filter(Boolean);
    const flag = flags.length > 0 ? ` <-- ${flags.join(', ')}` : '';
    console.log(
      `  ${r.key.padEnd(widest)}  ${r.metric.padEnd(10)} floor ${String(r.floor).padStart(3)}` +
        `  measured ${r.measured.toFixed(2).padStart(6)}  slack ${r.slack.toFixed(1).padStart(5)}${flag}`
    );
  }

  if (warnings.length > 0) {
    console.warn(`\ncheck-coverage-slack: ${warnings.length} erosion warning(s)\n`);
    for (const warning of warnings) console.warn(`  - ${warning}`);
  }

  if (failures.length > 0) {
    console.error(`\ncheck-coverage-slack: ${failures.length} problem(s)\n`);
    for (const f of failures) console.error(`  - ${f}`);
    console.error(
      `\nCoverage floors must stay within ${maxSlack} pts of measured, or they ` +
        'stop gating anything. Update packages/luxar-viewer/coverage-thresholds.mjs.'
    );
    process.exit(1);
  }
  console.log(
    `\ncheck-coverage-slack: OK — ${rows.length} floors all within ${maxSlack} pts of measured` +
      (warnings.length > 0 ? `; ${warnings.length} erosion warning(s).` : '.')
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
