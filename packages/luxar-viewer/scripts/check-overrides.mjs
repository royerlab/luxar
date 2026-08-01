#!/usr/bin/env node
/**
 * Guard the single source of truth for pnpm `overrides` (security-advisory pins).
 *
 * Background: the pins may live in `pnpm-workspace.yaml` OR in package.json's
 * `pnpm.overrides`. When BOTH exist they can disagree silently, because the two
 * consumers read different files:
 *
 *   - pnpm >=10.6 (every version that can read pnpm-workspace.yaml at all, and
 *     so everything CI and `make setup-dev` run) prefers package.json.
 *   - Dependabot's updater reads pnpm-workspace.yaml.
 *
 * That split-brain is invisible on main and only surfaces on Dependabot PRs,
 * which land a lockfile carrying the *other* override set and then fail every
 * job with `ERR_PNPM_LOCKFILE_CONFIG_MISMATCH`. It also means one of the two pin
 * sets is not actually applied — a silent security-pin regression.
 *
 * This script asserts:
 *   1. package.json has no `pnpm.overrides` block (pnpm-workspace.yaml owns it).
 *   2. pnpm-lock.yaml's `overrides:` block matches pnpm-workspace.yaml's, so a
 *      stale lockfile is reported with a message that names the real cause.
 *   3. pnpm-workspace.yaml declares at least one pin, so the whole block can't
 *      vanish unnoticed (1 and 2 both hold trivially at zero).
 *
 * Run via `pnpm run check:overrides` (part of `check:ci`).
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = resolve(HERE, '..');

const failures = [];

/**
 * Extract the top-level `overrides:` mapping from a pnpm YAML file.
 *
 * Deliberately hand-rolled rather than pulling in a YAML dependency: both files
 * emit this block as a flat `key: value` mapping, and the guard must stay
 * runnable before `pnpm install` has resolved anything.
 *
 * @param {string} text - Full file contents.
 * @returns {Map<string, string>} Override selector → replacement range.
 */
function parseOverridesBlock(text) {
  const entries = new Map();
  const lines = text.split('\n');
  let inBlock = false;

  for (const line of lines) {
    if (!inBlock) {
      if (line.trimEnd() === 'overrides:') inBlock = true;
      continue;
    }
    // The block ends at the first non-indented, non-blank line.
    if (line.trim() === '') continue;
    if (!/^\s/.test(line)) break;

    const stripped = line.trim();
    if (stripped.startsWith('#')) continue;

    // Selectors contain `@` and may contain spaces ("pkg@>=1.0.0 <2.0.0"), so
    // split on the LAST colon-space, which separates selector from replacement.
    const sep = stripped.lastIndexOf(': ');
    if (sep === -1) continue;
    const key = stripped.slice(0, sep).replace(/^['"]|['"]$/g, '');
    const value = stripped
      .slice(sep + 2)
      .trim()
      .replace(/^['"]|['"]$/g, '');
    entries.set(key, value);
  }
  return entries;
}

// 1. package.json must not carry a competing copy.
const pkg = JSON.parse(readFileSync(resolve(PKG_ROOT, 'package.json'), 'utf8'));
if (pkg.pnpm?.overrides) {
  failures.push(
    'package.json declares `pnpm.overrides`. Security pins live ONLY in ' +
      'pnpm-workspace.yaml — pnpm prefers package.json but Dependabot reads ' +
      'pnpm-workspace.yaml, so two copies drift apart and break every ' +
      'Dependabot PR with ERR_PNPM_LOCKFILE_CONFIG_MISMATCH. Move the entries ' +
      'into pnpm-workspace.yaml and delete the package.json block.'
  );
}

// 2. The lockfile must agree with pnpm-workspace.yaml.
const workspace = parseOverridesBlock(
  readFileSync(resolve(PKG_ROOT, 'pnpm-workspace.yaml'), 'utf8')
);
const lock = parseOverridesBlock(readFileSync(resolve(PKG_ROOT, 'pnpm-lock.yaml'), 'utf8'));

const format = (m) =>
  m.size === 0 ? '    (none)' : [...m].map(([k, v]) => `    ${k}: ${v}`).join('\n');

// 3. The pins must not silently vanish. `pnpm audit` is continue-on-error in
// ci.yml, so nothing else in CI notices if the whole block disappears and the
// lockfile is regenerated to match — checks 1 and 2 would both pass on 0 == 0.
if (workspace.size === 0) {
  failures.push(
    'pnpm-workspace.yaml declares no `overrides`. The security-advisory pins ' +
      'are gone, and no other CI gate catches that (`pnpm audit` is ' +
      'continue-on-error). If every advisory really is resolved and you mean ' +
      'to drop the pins, delete this check along with them.'
  );
}

const sameSize = workspace.size === lock.size;
const sameEntries = [...workspace].every(([k, v]) => lock.get(k) === v);
if (!sameSize || !sameEntries) {
  failures.push(
    'pnpm-lock.yaml `overrides` does not match pnpm-workspace.yaml.\n' +
      '  pnpm-workspace.yaml declares:\n' +
      `${format(workspace)}\n` +
      '  pnpm-lock.yaml records:\n' +
      `${format(lock)}\n` +
      '  Regenerate the lockfile: `pnpm install --lockfile-only` in ' +
      'packages/luxar-viewer/. On a Dependabot PR, comment `@dependabot recreate`.'
  );
}

if (failures.length > 0) {
  console.error('❌ pnpm overrides check failed:\n');
  for (const f of failures) console.error(`  • ${f}\n`);
  process.exit(1);
}

console.log(`✅ pnpm overrides single-sourced in pnpm-workspace.yaml (${workspace.size} pins)`);
