#!/usr/bin/env node
/**
 * Sanity-check the built library bundle.
 *
 * Asserts that:
 *   1. The expected output files exist (JS, CSS, types).
 *   2. The JS bundle exports the public symbols (LuxarApp,
 *      bootstrapStandalone, readUrlParams, StorageKeys).
 *   3. Importing the bundle does NOT monkey-patch the host console — the
 *      embedability contract from src/index.ts.
 *   4. `three` is externalized (not bundled). A bundled `three` would be a
 *      multi-MB regression and break peer-dep semantics.
 *
 * Run via `pnpm build:lib:check` after `pnpm build:lib`.
 */

import { existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = resolve(HERE, '..');
const LIB_DIR = resolve(PKG_ROOT, 'dist/lib');

const failures = [];
function fail(msg) {
  failures.push(msg);
}

// 1. Output files exist.
const expected = [
  'luxar-viewer.js',
  'luxar-viewer.css',
  'types/index.d.ts',
];
for (const file of expected) {
  const abs = resolve(LIB_DIR, file);
  if (!existsSync(abs)) {
    fail(`Missing expected output: dist/lib/${file}`);
  }
}

// 2. Exports check via dynamic import.
const jsPath = resolve(LIB_DIR, 'luxar-viewer.js');
if (existsSync(jsPath)) {
  // 3. Console snapshot before import.
  const consoleBefore = console.log;

  try {
    const mod = await import(pathToFileURL(jsPath).href);
    for (const sym of [
      'LuxarApp',
      'bootstrapStandalone',
      'readUrlParams',
      'StorageKeys',
    ]) {
      if (!(sym in mod)) {
        fail(`Bundle does not export "${sym}"`);
      }
    }

    if (console.log !== consoleBefore) {
      fail(
        'Importing the library patched console.log. The barrel must be ' +
          'side-effect-free; move console patching into LuxarApp.init() or ' +
          'an explicit factory.'
      );
    }
  } catch (err) {
    fail(`Failed to dynamically import the bundle: ${err.message}`);
  }

  // 4. `three` should be external. Heuristic: the JS bundle should not
  // contain the THREE.WebGLRenderer source (a large, recognizable string).
  // A real check would parse imports, but this is a fast sanity check.
  const bundleText = readFileSync(jsPath, 'utf8');
  const bundleSizeKB = statSync(jsPath).size / 1024;
  if (bundleText.includes('class WebGLRenderer')) {
    fail(
      `Bundle appears to contain THREE source (size: ${bundleSizeKB.toFixed(0)}KB). ` +
        'three must be externalized as a peer dependency.'
    );
  }

  // Friendly status line.
  console.log(
    `Library bundle: ${bundleSizeKB.toFixed(0)}KB (excluding three peer dep)`
  );
}

if (failures.length > 0) {
  console.error('\ncheck-lib-exports FAILED:');
  for (const f of failures) console.error('  -', f);
  process.exit(1);
}

console.log('check-lib-exports OK');
