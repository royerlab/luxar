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
 *   5. Every emitted chunk that references the WASM JS shim can actually
 *      reach it from its own directory. The entry chunk and the worker
 *      chunks sit at different depths, so a single relative specifier is
 *      not enough (#1649).
 *
 * Run via `pnpm build:lib:check` after `pnpm build:lib`.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, sep } from 'node:path';

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

// 5. The WASM shim is reachable from every chunk that names it.
//
// The loader resolves the wasm-bindgen shim relative to `import.meta.url`, but
// the chunks it lands in sit at two depths: the entry chunk at dist/lib/ and
// the worker chunks at dist/lib/assets/. `wasm/` is emitted at the output root
// (publicDir copy), so a chunk under assets/ needs '../wasm/…' while the entry
// chunk needs './wasm/…'. Getting this wrong is invisible at runtime — the
// import 404s and the viewer silently drops to the TypeScript fallback — and
// `build:lib` is not run by PR CI, so this script is the only gate on it.
//
// This is a TEXT-level layout check: it sees the specifiers a chunk literally
// spells out, so it cannot see a URL assembled at runtime (a `wasmPath`
// override, or a specifier built by string concatenation), and it does not
// prove the shim actually imports. It only proves that a chunk which names the
// shim names at least one path that lands on a real file.
const wasmArtifact = resolve(LIB_DIR, 'wasm/luxar_wasm.js');
if (!existsSync(wasmArtifact)) {
  // Without the publicDir copy every chunk is broken and the per-chunk scan
  // below would have nothing to resolve against — i.e. it would pass vacuously.
  fail('Missing WASM shim: dist/lib/wasm/luxar_wasm.js (publicDir copy not emitted?)');
}

/** Recursively collect every `.js` file under `dir`. */
function collectJsFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const abs = resolve(dir, entry.name);
    if (entry.isDirectory()) out.push(...collectJsFiles(abs));
    else if (entry.isFile() && entry.name.endsWith('.js')) out.push(abs);
  }
  return out;
}

if (existsSync(LIB_DIR)) {
  // The lib build sets `minify: false`, so JSDoc that merely MENTIONS
  // '../wasm/luxar_wasm.js' ships verbatim inside the chunk. Strip block
  // comments first and accept only quoted string literals, or the check would
  // pass on prose alone.
  const shimSpecifier = /["'`](\.{1,2}\/(?:\.\.\/)*wasm\/luxar_wasm\.js)["'`]/g;
  for (const file of collectJsFiles(LIB_DIR)) {
    const code = readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    const specifiers = [...code.matchAll(shimSpecifier)].map((m) => m[1]);
    if (specifiers.length === 0) continue;
    const reachable = specifiers.some((spec) => {
      const target = resolve(dirname(file), spec);
      // Existence alone is not enough: only `dist/lib/**` is published, so a
      // specifier that escapes LIB_DIR is unreachable for a consumer even
      // when it happens to land on something locally — a sibling app build
      // leaves a `dist/wasm/` behind, which is exactly what the broken
      // '../wasm/…' from the entry chunk resolves onto in a dev tree.
      return target.startsWith(LIB_DIR + sep) && existsSync(target);
    });
    if (!reachable) {
      const rel = file.slice(LIB_DIR.length + 1);
      fail(
        `dist/lib/${rel} references the WASM shim but none of its specifiers ` +
          `resolve to an existing file (tried: ${specifiers.join(', ')}). ` +
          'The shim lives at dist/lib/wasm/luxar_wasm.js; a chunk at the ' +
          "output root needs './wasm/…' while one under assets/ needs " +
          "'../wasm/…'."
      );
    }
  }
}

if (failures.length > 0) {
  console.error('\ncheck-lib-exports FAILED:');
  for (const f of failures) console.error('  -', f);
  process.exit(1);
}

console.log('check-lib-exports OK');
