#!/usr/bin/env node
/**
 * Keep jsdom on the 30.0 patch line until Vitest has a working Blob bridge.
 *
 * Vitest currently reads `_buffer` from jsdom's private Blob implementation.
 * jsdom 30.1 removes the wrapper-to-implementation symbol, so that access
 * throws. jsdom 30.0 is only a silent compatibility stopgap: `_buffer` is
 * already undefined there, so Blob bytes do not round-trip correctly. Unpin
 * only when Vitest uses a public path such as `blob.bytes()` or
 * `blob.arrayBuffer()`; a Vitest version bump alone is insufficient because
 * 5.0.1 ships the same `createCompatUtils` implementation.
 *
 * Dependabot's minor/major ignore also suppresses advisory PRs that require a
 * newer jsdom line. When unpinning, re-check that risk and the contributor Node
 * 22.22 floor derived from jsdom 30's undici 8 dependency. The package
 * `engines.node` contract deliberately tracks Vite's supported range instead,
 * so it should move only if that range changes.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(VIEWER_ROOT, '../..');
const REQUIRED_UPDATE_TYPES = ['version-update:semver-minor', 'version-update:semver-major'];

function viewerUpdater(contents) {
  const lines = contents.split('\n');
  const directoryIndex = lines.findIndex((line) =>
    /^\s+directory:\s*["']\/packages\/luxar-viewer["']\s*$/.test(line)
  );
  if (directoryIndex === -1) return null;

  let start = directoryIndex;
  while (start > 0 && !/^ {2}- package-ecosystem:/.test(lines[start])) start -= 1;
  let end = directoryIndex + 1;
  while (end < lines.length && !/^ {2}- package-ecosystem:/.test(lines[end])) end += 1;
  return lines.slice(start, end);
}

function dependencyIgnoreEntry(lines, dependencyName) {
  const start = lines.findIndex((line) => {
    const match = line.match(/^(\s*)- dependency-name:\s*["']([^"']+)["']\s*$/);
    return match?.[2] === dependencyName;
  });
  if (start === -1) return null;

  const indent = lines[start].match(/^\s*/)?.[0] ?? '';
  let end = start + 1;
  while (end < lines.length && !lines[end].startsWith(`${indent}- dependency-name:`)) end += 1;
  return lines.slice(start, end).join('\n');
}

export function checkJsdomVersion(viewerRoot = VIEWER_ROOT, repoRoot = REPO_ROOT) {
  const pkg = JSON.parse(readFileSync(resolve(viewerRoot, 'package.json'), 'utf8'));
  const declared = pkg.devDependencies?.jsdom;
  if (typeof declared !== 'string') {
    return ['package.json must declare devDependencies.jsdom'];
  }

  const problems = [];
  if (!/^~30\.0\.\d+$/.test(declared)) {
    problems.push(
      `Expected jsdom to use a plain ~30.0.PATCH range, found ${JSON.stringify(declared)}`
    );
  }

  const dependabot = readFileSync(resolve(repoRoot, '.github', 'dependabot.yml'), 'utf8');
  const updater = viewerUpdater(dependabot);
  const ignoreEntry = updater ? dependencyIgnoreEntry(updater, 'jsdom') : null;
  if (ignoreEntry === null) {
    problems.push('Dependabot must ignore jsdom updates for /packages/luxar-viewer');
    return problems;
  }
  for (const updateType of REQUIRED_UPDATE_TYPES) {
    if (!new RegExp(`-\\s*["']${updateType}["']`).test(ignoreEntry)) {
      problems.push(`Dependabot jsdom ignore must include ${updateType}`);
    }
  }
  return problems;
}

function main() {
  const problems = checkJsdomVersion();
  if (problems.length > 0) {
    for (const problem of problems) console.error(`❌ ${problem}`);
    process.exit(1);
  }
  console.log('✅ jsdom pin and Dependabot ignore match the Blob bridge contract');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
