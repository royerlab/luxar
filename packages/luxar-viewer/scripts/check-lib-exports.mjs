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

  // 4. `three` (and all `three/*` subpaths) should be external. Heuristic: the
  // JS bundle should not contain THREE source. `class WebGLRenderer` catches an
  // inlined `three.module.js`, but the real failure mode is a duplicated THREE
  // *core* (`three.core.js`), inlined when the `three/webgpu` / `three/tsl`
  // subpaths the TSL materials import are not externalized — and core carries
  // no `WebGLRenderer`. So we also scan for markers unique to THREE's core
  // source that the viewer's own code never produces. A real check would parse
  // imports, but this is a fast sanity check.
  const bundleText = readFileSync(jsPath, 'utf8');
  const bundleSizeKB = statSync(jsPath).size / 1024;
  // Markers that appear only if THREE source was inlined into the bundle.
  // The lib build runs unminified through rolldown (Vite 8), which rewrites
  // `class Foo {` → `var Foo = class {` and `const X = '…'` → `var X = "…"`,
  // so match the class/assignment forms (covering classic Rollup too) rather
  // than the verbatim source text. `\b` before each name avoids matching a
  // longer identifier that merely ends in the name (e.g. MockEventDispatcher):
  //  - EventDispatcher — THREE's root base class (three.core.js), dragged in
  //    by ANY inlined core, including via three/webgpu and three/tsl
  //  - WebGLRenderer   — the full renderer (three.module.js)
  //  - REVISION        — the version constant (three.core.js), version-agnostic
  const threeSourceMarkers = [
    /\bclass EventDispatcher\b|\bEventDispatcher\s*=\s*class\b/,
    /\bclass WebGLRenderer\b|\bWebGLRenderer\s*=\s*class\b/,
    /\bREVISION\s*=\s*['"]/,
  ];
  const foundMarker = threeSourceMarkers.find((re) => re.test(bundleText));
  if (foundMarker) {
    fail(
      `Bundle appears to contain an inlined THREE core / duplicate runtime ` +
        `(matched ${foundMarker}, size: ${bundleSizeKB.toFixed(0)}KB). ` +
        'three is a peer dependency: `three` and ALL `three/*` subpaths ' +
        '(three/webgpu, three/tsl, …) must be externalized so the host page ' +
        'supplies a single THREE runtime.'
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
