#!/usr/bin/env node
/**
 * Ratchet knip's unused-export backlog against `knip-baseline.json`.
 *
 * `check:knip:ci` gates unused FILES and dependencies only; an export, type or
 * enum member nothing imports is reported by the full `check:knip` but never
 * fails anything, so scaffolding left behind by optimisation work accumulates.
 * This check fails on the PR that adds the next one, while tolerating the
 * recorded backlog. Semantics mirror `scripts/ruff_ratchet.py` +
 * `complexity_baseline.json` at the repository root:
 *
 * - a NEW entry fails;
 * - a paid-down or MOVED entry also fails until `--update-baseline` tightens the
 *   baseline, so the recorded backlog never over-declares;
 * - a changed `knip.json`, knip version or issue-type set fails CLOSED (the
 *   fingerprint no longer matches), because any of them can change what knip
 *   reports and the old keys could no longer be trusted;
 * - a run RESTRICTED to path prefixes is advisory for vanished entries (keys
 *   outside the scope only look vanished) and refuses `--update-baseline`.
 *
 * Usage:
 *   node scripts/check-knip-ratchet.mjs                    # full gate
 *   node scripts/check-knip-ratchet.mjs src/ui src/data    # restricted (advisory)
 *   node scripts/check-knip-ratchet.mjs --update-baseline  # record the current tree
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const BASELINE_PATH = join(PACKAGE_ROOT, 'knip-baseline.json');
const KNIP_CONFIG_PATH = join(PACKAGE_ROOT, 'knip.json');
const KNIP_PACKAGE_PATH = join(PACKAGE_ROOT, 'node_modules/knip/package.json');
const UPDATE_COMMAND = 'pnpm run check:knip:ratchet --update-baseline';

/** knip 6's export-level issue types (its `--exports` shortcut set). */
export const ISSUE_TYPES = Object.freeze([
  'exports',
  'nsExports',
  'types',
  'nsTypes',
  'enumMembers',
  'namespaceMembers',
  'duplicates',
]);

const BASELINE_COMMENT =
  'Unused-export backlog for scripts/check-knip-ratchet.mjs (`pnpm run check:knip:ratchet`). ' +
  "Each entry is '<issue type>:<package-relative path>::<symbol>' (a namespaced member is " +
  "'<parent>.<member>', a duplicate-export group joins its names with '|'). A finding not listed " +
  'here fails; a listed entry knip no longer reports (paid down, or moved to another file) also ' +
  `fails until the baseline is refreshed with \`${UPDATE_COMMAND}\`. 'fingerprint' hashes ` +
  'knip.json, the knip version and the issue types: a change to any fails closed until the ' +
  'baseline is regenerated. Prefer deleting dead code (or tagging a deliberate export ' +
  '@internal) to adding entries.';

function entryName(item) {
  if (Array.isArray(item)) {
    return item
      .map((member) => member.name)
      .sort()
      .join('|');
  }
  return item.namespace ? `${item.namespace}.${item.name}` : item.name;
}

/** Normalise a knip `--reporter json` report to sorted, unique entry keys. */
export function issueKeys(report, issueTypes = ISSUE_TYPES) {
  if (!report || !Array.isArray(report.issues)) {
    throw new Error('knip JSON report has no `issues` array; refusing to trust it');
  }
  const keys = new Set();
  for (const row of report.issues) {
    for (const type of issueTypes) {
      for (const item of row[type] ?? []) keys.add(`${type}:${row.file}::${entryName(item)}`);
    }
  }
  return [...keys].sort();
}

/** Hash everything that can change what knip reports for the same tree. */
export function configFingerprint({ knipConfig, knipVersion, issueTypes = ISSUE_TYPES }) {
  const payload = JSON.stringify({ knipConfig, knipVersion, issueTypes: [...issueTypes] });
  return createHash('sha256').update(payload).digest('hex');
}

function inScope(key, scope) {
  if (!scope.length) return true;
  const file = key.slice(key.indexOf(':') + 1, key.lastIndexOf('::'));
  return scope.some(
    (prefix) => file === prefix || file.startsWith(`${prefix.replace(/\/$/, '')}/`)
  );
}

function symbolOf(key) {
  return `${key.slice(0, key.indexOf(':'))}:${key.slice(key.lastIndexOf('::') + 2)}`;
}

/**
 * Diff the current keys against the baseline. `moved` pairs a vanished key with
 * a new one of the same type and symbol in a different file; such a pair is
 * reported once, as a move, rather than as an unrelated addition and removal.
 */
export function compareToBaseline(baselineKeys, currentKeys, scope = []) {
  const baseline = new Set(baselineKeys.filter((key) => inScope(key, scope)));
  const current = new Set(currentKeys.filter((key) => inScope(key, scope)));
  let added = [...current].filter((key) => !baseline.has(key));
  let removed = [...baseline].filter((key) => !current.has(key));
  const moved = [];
  for (const from of [...removed]) {
    const to = added.find((key) => symbolOf(key) === symbolOf(from));
    if (!to) continue;
    moved.push({ from, to });
    added = added.filter((key) => key !== to);
    removed = removed.filter((key) => key !== from);
  }
  return { added, removed, moved };
}

/** Parse CLI arguments: optional path prefixes plus `--update-baseline`. */
export function parseArgs(argv) {
  const options = { updateBaseline: false, scope: [] };
  for (const arg of argv) {
    if (arg === '--update-baseline') options.updateBaseline = true;
    else if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`);
    else options.scope.push(arg.replace(/^\.\//, ''));
  }
  if (options.updateBaseline && options.scope.length) {
    throw new Error(
      '--update-baseline with path arguments would drop every baselined entry outside those ' +
        'paths. Run it without paths.'
    );
  }
  return options;
}

/** Run knip once over the whole project; fail closed on anything but a clean JSON report. */
export function runKnip(issueTypes = ISSUE_TYPES, spawn = spawnSync) {
  const args = ['exec', 'knip', '--no-progress', '--no-config-hints', '--no-exit-code'];
  args.push('--reporter', 'json', '--include', issueTypes.join(','));
  const proc = spawn('pnpm', args, { cwd: PACKAGE_ROOT, encoding: 'utf8', maxBuffer: 64 << 20 });
  if (proc.error) throw new Error(`Could not run knip: ${proc.error.message}`);
  if (proc.status !== 0) {
    throw new Error(`knip failed (exit ${proc.status}):\n${(proc.stderr ?? '').trim()}`);
  }
  try {
    return JSON.parse(proc.stdout);
  } catch (error) {
    throw new Error(`knip output is not JSON (${error.message}); refusing to trust it`, {
      cause: error,
    });
  }
}

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function currentFingerprint() {
  return configFingerprint({
    knipConfig: readJson(KNIP_CONFIG_PATH),
    knipVersion: readJson(KNIP_PACKAGE_PATH).version,
  });
}

function loadBaseline() {
  if (!existsSync(BASELINE_PATH)) {
    throw new Error(`Missing knip-baseline.json; create it with \`${UPDATE_COMMAND}\`.`);
  }
  const data = readJson(BASELINE_PATH);
  if (!Array.isArray(data.entries) || typeof data.fingerprint !== 'string') {
    throw new Error('knip-baseline.json is malformed (needs `fingerprint` and `entries`).');
  }
  return data;
}

function writeBaseline(keys, fingerprint) {
  const previous = existsSync(BASELINE_PATH) ? (readJson(BASELINE_PATH).entries ?? []) : [];
  const retired = previous.filter((key) => !keys.includes(key));
  const data = { _comment: BASELINE_COMMENT, fingerprint, entries: keys };
  writeFileSync(BASELINE_PATH, `${JSON.stringify(data, null, 2)}\n`);
  console.log(`Wrote ${keys.length} entries to knip-baseline.json.`);
  if (retired.length) console.log(`Retired ${retired.length}:\n  ${retired.join('\n  ')}`);
}

function list(title, items) {
  if (items.length) console.error(`\n${title}\n  ${items.join('\n  ')}`);
}

function report({ added, removed, moved }, restricted) {
  list(
    `${added.length} NEW unused export(s) — delete the dead code, or tag a deliberate export @internal:`,
    added
  );
  list(
    `${moved.length} baselined entr${moved.length === 1 ? 'y' : 'ies'} MOVED (refresh the baseline):`,
    moved.map(({ from, to }) => `${from} -> ${to}`)
  );
  if (restricted) {
    list(
      `${removed.length} baselined entries no longer reported (advisory: restricted run):`,
      removed
    );
    return added.length > 0;
  }
  list(`${removed.length} baselined entries PAID DOWN (refresh the baseline):`, removed);
  return added.length + moved.length + removed.length > 0;
}

/** Run the ratchet; returns true when it passes. */
export function runCheck(options, knipReport = runKnip()) {
  const keys = issueKeys(knipReport);
  const fingerprint = currentFingerprint();
  if (options.updateBaseline) {
    writeBaseline(keys, fingerprint);
    return true;
  }
  const baseline = loadBaseline();
  if (baseline.fingerprint !== fingerprint) {
    console.error(
      'knip.json, the knip version or the issue-type set changed since knip-baseline.json was ' +
        `recorded, so its entries cannot be trusted. Review the new report, then run \`${UPDATE_COMMAND}\`.`
    );
    return false;
  }
  const restricted = options.scope.length > 0;
  const failed = report(compareToBaseline(baseline.entries, keys, options.scope), restricted);
  if (failed) {
    console.error(`\nAfter an intended change, refresh with \`${UPDATE_COMMAND}\`.`);
    return false;
  }
  const scope = restricted ? ` (restricted to ${options.scope.join(', ')})` : '';
  console.log(`knip ratchet: no new unused exports${scope}; ${keys.length} baselined.`);
  return true;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    if (!runCheck(parseArgs(process.argv.slice(2)))) process.exitCode = 1;
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  }
}
