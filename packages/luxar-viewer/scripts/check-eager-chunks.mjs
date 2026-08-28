#!/usr/bin/env node
/**
 * Assert that the WebGPU/TSL cone stays OFF the app's critical path.
 *
 * The production default backend is WebGL (`selectBackend()` in
 * renderer-setup.ts), so the ~182 kB gzipped `three-webgpu` chunk must be
 * fetched only when a session actually opts into WebGPU. Issue #1679: it used
 * to be a static dependency of the entry chunk AND `modulepreload`ed from
 * `index.html`, so every WebGL user downloaded and parsed a renderer that never
 * ran — about a quarter of the initial JS payload.
 *
 * Why this exists ALONGSIDE the ESLint `no-restricted-imports` rule: that rule
 * governs *source* imports, and the last mile of #1679 was not a source
 * problem at all. `three.core.js` is shared by `three.module.js` and
 * `three.webgpu.js`, and with no chunk group of its own it landed such that the
 * plain `three` chunk imported `three-webgpu` — pinning the whole node system
 * into the eager graph while every source file was already clean. Only an
 * assertion on the built output can see that class of regression.
 *
 * Run after `vite build`, against `dist/`.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const DIST = join(import.meta.dirname, '..', 'dist');
const ASSETS = join(DIST, 'assets');

/** Chunk-name stems that must never be reachable eagerly from the entry. */
const LAZY_ONLY = ['three-webgpu', 'three-ktx2'];
const KTX2_ZSTD_DECODER_MARKER = 'emscripten_notify_memory_growth';

const failures = [];
const notes = [];

function fail(message) {
  failures.push(message);
}

// ── 0. The extractor must actually extract ─────────────────────────────────
// A regex that stops matching does not fail this script, it makes every check
// below vacuous and prints a ✅ — which is exactly what the first version of
// this file did. So pin the two minified shapes `staticImportsOf` (below) calls
// load-bearing, plus the dynamic form that must NOT count, against a fixture.
// This asserts on the script itself rather than on whatever the bundler
// happened to emit today, so a build that legitimately has no chunk splits
// cannot make it red.
{
  const fixture = 'import{a as b}from"./eager.js";export*from"../reexport.js";import("./lazy.js");';
  const found = staticImportsOf(fixture);
  const expected = ['./eager.js', '../reexport.js'];
  const missing = expected.filter((s) => !found.has(s));
  if (missing.length > 0 || found.has('./lazy.js')) {
    fail(
      `staticImportsOf() no longer parses built-chunk imports: expected ` +
        `${expected.join(', ')} and not './lazy.js', got ${[...found].join(', ') || '(nothing)'}. ` +
        `Every assertion below reads the same specifiers, so they would all pass vacuously. ` +
        `Fix the regex before trusting this check.`
    );
  }
}

// ── 1. index.html must not preload or script-tag a lazy-only chunk ─────────
const html = readFileSync(join(DIST, 'index.html'), 'utf8');
for (const stem of LAZY_ONLY) {
  const preloaded = [...html.matchAll(/<link[^>]+rel="modulepreload"[^>]+href="([^"]+)"/g)]
    .map((m) => m[1])
    .filter((href) => href.includes(stem));
  if (preloaded.length > 0) {
    fail(
      `dist/index.html modulepreloads ${preloaded.join(', ')}. A lazily-imported ` +
        `chunk is never preloaded, so this means something in the eager graph ` +
        `still imports '${stem}' statically.`
    );
  }
  const scripted = [...html.matchAll(/<script[^>]+src="([^"]+)"/g)]
    .map((m) => m[1])
    .filter((src) => src.includes(stem));
  if (scripted.length > 0) {
    fail(`dist/index.html loads ${scripted.join(', ')} with a <script> tag.`);
  }
}

// ── 2. The entry chunk must not STATICALLY import a lazy-only chunk ────────
const entryHref = html.match(/<script[^>]+type="module"[^>]+src="([^"]+)"/)?.[1];
if (!entryHref) {
  fail('Could not find the entry <script type="module"> in dist/index.html.');
}

const assetFiles = readdirSync(ASSETS);

for (const suffix of ['.js', '.wasm']) {
  if (!assetFiles.some((file) => file.startsWith('basis_transcoder-') && file.endsWith(suffix))) {
    fail(`No Basis transcoder ${suffix} asset was emitted into dist/assets.`);
  }
}

/**
 * Static (not dynamic) import specifiers of a built ESM chunk.
 *
 * Matches `import ... from "x"`, bare `import "x"`, and `export ... from "x"`
 * at a statement boundary. Two details are load-bearing against MINIFIED
 * output, and both were wrong in the first version of this script (which
 * silently matched nothing and made the checks below vacuous):
 *
 *  - no `\s` may be required before `from`: rolldown emits `}from"./x.js"`.
 *  - `(` must be excluded from the run before the quote, so a dynamic
 *    `import("./x.js")` is NOT counted as a static edge — counting it would
 *    flag the very lazy boundary this script exists to protect.
 */
function staticImportsOf(source) {
  const specifiers = new Set();
  const re = /(?:^|[;}\n])\s*(?:import|export)\b[^;'"(]*?["']([^"']+)["']/g;
  for (const m of source.matchAll(re)) specifiers.add(m[1]);
  return specifiers;
}

if (entryHref) {
  const entryName = entryHref.split('/').pop();
  const entrySource = readFileSync(join(ASSETS, entryName), 'utf8');
  const statics = staticImportsOf(entrySource);
  for (const stem of LAZY_ONLY) {
    const offenders = [...statics].filter((s) => s.includes(stem));
    if (offenders.length > 0) {
      fail(
        `The entry chunk (${entryName}) statically imports ${offenders.join(', ')}. ` +
          `It must only appear behind a dynamic import().`
      );
    }
  }

  // ── 3. …and no chunk the entry reaches statically may import one either ──
  // This is the `three.core.js` case: the offending edge was `three` →
  // `three-webgpu`, one hop from the entry, invisible to step 2 alone.
  const visited = new Set();
  const queue = [entryName];
  while (queue.length > 0) {
    const name = queue.shift();
    if (visited.has(name)) continue;
    visited.add(name);
    let source;
    try {
      source = readFileSync(join(ASSETS, name), 'utf8');
    } catch {
      continue; // not an emitted asset (e.g. an external specifier)
    }
    if (source.includes(KTX2_ZSTD_DECODER_MARKER)) {
      fail(
        `Chunk '${name}' contains Three's KTX2 zstd decoder and is reachable from the entry ` +
          `chunk. Keep zstddec.module.js in the lazy 'three-ktx2' chunk.`
      );
    }
    for (const spec of staticImportsOf(source)) {
      const dep = spec.split('/').pop();
      // Reported before the relative-path filter below: an absolute or bare
      // specifier is one we cannot follow, but naming the lazy chunk is a
      // failure regardless of how the bundler spelled the path.
      for (const stem of LAZY_ONLY) {
        if (dep.includes(stem)) {
          fail(
            `Chunk '${name}' statically imports '${dep}', and '${name}' is reachable ` +
              `from the entry chunk. The whole '${stem}' chunk is therefore eager. ` +
              `Keep '${stem}' dynamically imported. For 'three' → 'three-webgpu', ` +
              `the shared three.core.js needs its own codeSplitting group in ` +
              `vite.config.ts (see #1679).`
          );
        }
      }
      if (!spec.startsWith('./') && !spec.startsWith('../')) continue;
      queue.push(dep);
    }
  }
  notes.push(`eagerly reachable chunks: ${visited.size}`);
}

// ── 4. The lazy chunk must still EXIST ────────────────────────────────────
// Guards the degenerate "fix": dropping a lazy feature would also make every
// assertion above pass.
for (const stem of LAZY_ONLY) {
  if (!assetFiles.some((f) => f.includes(stem) && f.endsWith('.js'))) {
    fail(
      `No '${stem}' chunk was emitted at all. The '${stem}' path must still be built ` +
        `— it is only supposed to be LAZY, not absent.`
    );
  }
}

if (failures.length > 0) {
  console.error('❌ Eager-chunk check failed:\n');
  for (const f of failures) console.error(`  • ${f}\n`);
  process.exit(1);
}

console.log(`✅ lazy-only chunks remain deferred (${notes.join('; ')})`);
