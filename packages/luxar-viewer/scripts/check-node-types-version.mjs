#!/usr/bin/env node
/**
 * Keep `@types/node` aligned with the viewer's Node runtime contract.
 *
 * #2519 showed that newer declarations can make APIs such as `node:vfs`
 * typecheck even though the runtime selected by `.nvmrc` cannot provide them.
 * Treat `.nvmrc` as the single source of truth for the typings major. The
 * Dependabot semver-major ignore prevents automated drift, while
 * `src/tests/type/node-runtime-contract.ts` provides a type-level tripwire.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(VIEWER_ROOT, '../..');

function versionMajor(value) {
  const match = value.trim().match(/^[v~^]?(\d+)(?:\.\d+){0,2}$/);
  return match ? Number(match[1]) : null;
}

export function checkNodeTypesVersion(viewerRoot = VIEWER_ROOT, repoRoot = REPO_ROOT) {
  const nodeVersion = readFileSync(resolve(repoRoot, '.nvmrc'), 'utf8');
  const nodeMajor = versionMajor(nodeVersion);
  const pkg = JSON.parse(readFileSync(resolve(viewerRoot, 'package.json'), 'utf8'));
  const declared = pkg.devDependencies?.['@types/node'];
  if (typeof declared !== 'string') {
    return ['package.json must declare devDependencies["@types/node"]'];
  }

  const declaredMajor = versionMajor(declared);
  if (nodeMajor === null || declaredMajor === null) {
    return [
      `Expected single-major versions, found .nvmrc=${JSON.stringify(nodeVersion.trim())} ` +
        `and @types/node=${JSON.stringify(declared)}`,
    ];
  }
  return declaredMajor === nodeMajor
    ? []
    : [
        `package.json @types/node targets Node ${declaredMajor}, but .nvmrc targets Node ${nodeMajor}`,
      ];
}

function main() {
  const problems = checkNodeTypesVersion();
  if (problems.length > 0) {
    for (const problem of problems) console.error(`❌ ${problem}`);
    process.exit(1);
  }
  console.log('✅ @types/node major matches .nvmrc');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
