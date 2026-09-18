#!/usr/bin/env node
/**
 * Keep the three runtime peer and its declarations on the same minor.
 *
 * #2799 showed that a type-only minor bump can expose APIs absent from the
 * installed runtime. Three upgrades follow THREE_VERSION_NOTES.md and move
 * both tilde ranges together; Dependabot remains free to take patch updates.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = resolve(HERE, '..');

function threeMinor(value) {
  const match = value.match(/^~0\.(\d+)\.\d+$/);
  return match ? Number(match[1]) : null;
}

export function checkThreeTypesVersion(viewerRoot = VIEWER_ROOT) {
  const pkg = JSON.parse(readFileSync(resolve(viewerRoot, 'package.json'), 'utf8'));
  const runtime = pkg.peerDependencies?.three;
  if (typeof runtime !== 'string') {
    return ['package.json must declare peerDependencies.three'];
  }

  const types = pkg.devDependencies?.['@types/three'];
  if (typeof types !== 'string') {
    return ['package.json must declare devDependencies["@types/three"]'];
  }

  const runtimeMinor = threeMinor(runtime);
  const typesMinor = threeMinor(types);
  if (runtimeMinor === null || typesMinor === null) {
    return [
      `Expected plain ~0.MINOR.PATCH ranges, found three=${JSON.stringify(runtime)} ` +
        `and @types/three=${JSON.stringify(types)}`,
    ];
  }
  return runtimeMinor === typesMinor
    ? []
    : [`package.json three targets r${runtimeMinor}, but @types/three targets r${typesMinor}`];
}

function main() {
  const problems = checkThreeTypesVersion();
  if (problems.length > 0) {
    for (const problem of problems) console.error(`❌ ${problem}`);
    process.exit(1);
  }
  console.log('✅ @types/three minor matches the three peer range');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
