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

/**
 * Chunk-name stems that must still be BUILT, only lazily.
 *
 * Separate from the per-entry rules below because the two ask different
 * questions. This one guards the degenerate "fix" of deleting a lazy feature
 * outright, which would otherwise make every reachability assertion pass.
 */
const LAZY_ONLY = ['three-webgpu', 'three-ktx2'];

/**
 * What each HTML entry may not reach eagerly.
 *
 * The axis differs per page, which is why this is a table rather than one
 * global list:
 *
 * - `index.html` is the viewer. It legitimately ships `three`; the rule is only
 *   that the WebGPU/KTX2 cone stays behind a dynamic import (#1679).
 * - `control.html` is the kiosk touch panel. It renders no 3D at all and loads
 *   no data, so **nothing** three-shaped or codec-shaped may be reachable from
 *   it. That is not a source-level property: the page imports the viewer's
 *   `config` and `ui` modules, and a shared chunk could drag a renderer in
 *   without a single offending import in the page's own tree. Only an
 *   assertion on the built graph can see it.
 */
const ENTRIES = [
  { html: 'index.html', forbid: LAZY_ONLY },
  {
    html: 'control.html',
    forbid: ['three', 'blosc', 'zstd', 'basis_transcoder', 'data-worker'],
  },
];

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
      'staticImportsOf() no longer parses built-chunk imports: expected ' +
        `${expected.join(', ')} and not './lazy.js', got ${[...found].join(', ') || '(nothing)'}. ` +
        'Every assertion below reads the same specifiers, so they would all pass vacuously. ' +
        'Fix the regex before trusting this check.'
    );
  }
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

/**
 * Audit one HTML entry: its preloads, its entry chunk, and everything that
 * chunk reaches statically.
 *
 * Steps 1–3 of the original single-entry check, parameterised by page. The
 * reachability walk is the part with teeth — the `three.core.js` regression was
 * an edge one hop from the entry (`three` → `three-webgpu`), invisible to a
 * direct-imports check and to every source-level lint.
 */
function auditEntry({ html: htmlName, forbid }) {
  let html;
  try {
    html = readFileSync(join(DIST, htmlName), 'utf8');
  } catch {
    // Degenerate-fix guard: a page that stopped being built would otherwise
    // make all of its assertions pass by having nothing to assert about.
    fail(
      `dist/${htmlName} was not emitted. Both HTML entries must be declared in ` +
        "vite.config.ts's `rolldownOptions.input` — setting `input` at all " +
        'removes the implicit index.html default.'
    );
    return;
  }

  // ── 1. The page must not preload or script-tag a forbidden chunk ─────────
  for (const stem of forbid) {
    const preloaded = [...html.matchAll(/<link[^>]+rel="modulepreload"[^>]+href="([^"]+)"/g)]
      .map((m) => m[1])
      .filter((href) => href.includes(stem));
    if (preloaded.length > 0) {
      fail(
        `dist/${htmlName} modulepreloads ${preloaded.join(', ')}. A lazily-imported ` +
          'chunk is never preloaded, so this means something in the eager graph ' +
          `still imports '${stem}' statically.`
      );
    }
    const scripted = [...html.matchAll(/<script[^>]+src="([^"]+)"/g)]
      .map((m) => m[1])
      .filter((src) => src.includes(stem));
    if (scripted.length > 0) {
      fail(`dist/${htmlName} loads ${scripted.join(', ')} with a <script> tag.`);
    }
  }

  // ── 2. The entry chunk must not STATICALLY import a forbidden chunk ──────
  const entryHref = html.match(/<script[^>]+type="module"[^>]+src="([^"]+)"/)?.[1];
  if (!entryHref) {
    fail(`Could not find the entry <script type="module"> in dist/${htmlName}.`);
    return;
  }
  const entryName = entryHref.split('/').pop();

  // ── 3. …and nothing it reaches statically may import one either ──────────
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
        `Chunk '${name}' contains Three's KTX2 zstd decoder and is reachable from ` +
          `dist/${htmlName}. Keep zstddec.module.js in the lazy 'three-ktx2' chunk.`
      );
    }
    for (const spec of staticImportsOf(source)) {
      const dep = spec.split('/').pop();
      // Reported before the relative-path filter below: an absolute or bare
      // specifier is one we cannot follow, but naming a forbidden chunk is a
      // failure regardless of how the bundler spelled the path.
      for (const stem of forbid) {
        if (dep.includes(stem)) {
          fail(
            `Chunk '${name}' statically imports '${dep}', and '${name}' is reachable ` +
              `from dist/${htmlName}. The whole '${stem}' chunk is therefore eager ` +
              `for that page. For index.html, keep '${stem}' behind a dynamic ` +
              "import() — 'three' → 'three-webgpu' needs three.core.js in its own " +
              'codeSplitting group (see #1679). For control.html, the panel renders ' +
              'no 3D and loads no data, so it must not reach a renderer or a codec ' +
              'at all; check what its config/ui imports pulled in.'
          );
        }
      }
      if (!spec.startsWith('./') && !spec.startsWith('../')) continue;
      queue.push(dep);
    }
  }
  notes.push(`${htmlName}: ${visited.size} eager chunks`);
}

for (const entry of ENTRIES) auditEntry(entry);

// ── 4. The lazy chunk must still EXIST ────────────────────────────────────
// Guards the degenerate "fix": dropping a lazy feature would also make every
// assertion above pass.
for (const stem of LAZY_ONLY) {
  if (!assetFiles.some((f) => f.includes(stem) && f.endsWith('.js'))) {
    fail(
      `No '${stem}' chunk was emitted at all. The '${stem}' path must still be built ` +
        '— it is only supposed to be LAZY, not absent.'
    );
  }
}

if (failures.length > 0) {
  console.error('❌ Eager-chunk check failed:\n');
  for (const f of failures) console.error(`  • ${f}\n`);
  process.exit(1);
}

console.log(`✅ lazy-only chunks remain deferred (${notes.join('; ')})`);
