#!/usr/bin/env node
/**
 * Keep the three runtime peer, declarations, and embed pins on the same minor.
 *
 * #2799 showed that a type-only minor bump can expose APIs absent from the
 * installed runtime. Three upgrades follow THREE_VERSION_NOTES.md and move
 * these declarations together. This deliberately checks declared ranges and
 * importmaps, not lockfile resolutions: patch skew remains allowed. The 0.x
 * parser is intentional because Three's minor is its revision; a 1.0 release
 * requires revisiting this contract rather than silently accepting it.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const VIEWER_ROOT = resolve(HERE, '..');
const EMBED_IMPORTMAPS = ['examples/embed/index.html', 'examples/embed/README.md'];

function threeMinor(value) {
  const match = value.match(/^~0\.(\d+)\.\d+$/);
  return match ? Number(match[1]) : null;
}

function embedThreeMinor(contents) {
  const match = contents.match(
    /"three"\s*:\s*"https:\/\/cdn\.jsdelivr\.net\/npm\/three@0\.(\d+)\.\d+\/build\/three\.module\.js"/
  );
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
      'Expected plain ~0.MINOR.PATCH ranges under the pre-1.0 Three revision scheme, ' +
        `found three=${JSON.stringify(runtime)} ` +
        `and @types/three=${JSON.stringify(types)}`,
    ];
  }

  const problems = [];
  if (runtimeMinor !== typesMinor) {
    problems.push(
      `package.json three targets r${runtimeMinor}, but @types/three targets r${typesMinor}`
    );
  }
  for (const relativePath of EMBED_IMPORTMAPS) {
    const embedMinor = embedThreeMinor(readFileSync(resolve(viewerRoot, relativePath), 'utf8'));
    if (embedMinor === null) {
      problems.push(`${relativePath} must pin three@0.MINOR.PATCH in its jsDelivr importmap`);
    } else if (embedMinor !== runtimeMinor) {
      problems.push(
        `${relativePath} targets r${embedMinor}, but package.json three targets r${runtimeMinor}`
      );
    }
  }
  return problems;
}

function main() {
  const problems = checkThreeTypesVersion();
  if (problems.length > 0) {
    for (const problem of problems) console.error(`❌ ${problem}`);
    process.exit(1);
  }
  console.log('✅ @types/three and embed pins match the three peer minor');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
