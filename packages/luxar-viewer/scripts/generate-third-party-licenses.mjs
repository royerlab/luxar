#!/usr/bin/env node
/**
 * Emit THIRD_PARTY_LICENSES.txt next to a build output.
 *
 * The minified viewer bundle is redistributed four ways -- the npm package, the
 * wheel's `luxar/_viewer_dist`, `luxar export` folders, and the native launcher
 * bundles -- and MIT / Apache-2.0 / MPL-2.0 / BSD all require their notices to
 * accompany a BINARY redistribution, not merely to exist in some source tree.
 * Nothing generated one, because the production Vite build runs zero plugins.
 *
 * Everything here is DERIVED from what the build actually consumes: the pnpm
 * production closure for JavaScript, `cargo metadata` for the crates compiled
 * into the WASM module, and the colormap generator's own tables for the baked
 * LUTs. There is no hand-maintained package list, so a new dependency cannot be
 * shipped without its notice.
 *
 * It FAILS rather than emit an incomplete file. A notices file that silently
 * omits a dependency is worse than none: it looks like diligence.
 *
 *   node scripts/generate-third-party-licenses.mjs dist [dist/lib ...]
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const VIEWER_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RUST_ROOT = join(VIEWER_ROOT, 'src/wasm/rust');
const REPO_ROOT = resolve(VIEWER_ROOT, '../..');
const require = createRequire(join(VIEWER_ROOT, 'noop.js'));

const LICENSE_FILE = /^(LICEN[CS]E|COPYING|NOTICE)([-.].*)?$/i;
const RULE = '='.repeat(78);

/**
 * Every package in the pnpm PRODUCTION closure, deduplicated by name.
 *
 * `problems` is optional so callers that just want the list can omit it, but
 * when supplied a tooling failure is RECORDED rather than thrown -- matching
 * `rustSections`, so main() can print one deliberate refusal instead of a
 * stack trace from whichever half happened to run first.
 */
export function productionPackages(problems) {
  let raw;
  try {
    raw = execFileSync('pnpm', ['list', '--prod', '--depth', 'Infinity', '--json'], {
      cwd: VIEWER_ROOT,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (err) {
    const message = `pnpm list failed (${err.message.split('\n')[0]}); cannot determine which packages are redistributed`;
    if (!problems) throw new Error(message);
    problems.push(message);
    return new Map();
  }
  const found = new Map();
  const walk = (deps) => {
    for (const [name, info] of Object.entries(deps ?? {})) {
      if (found.has(name)) continue;
      found.set(name, info);
      walk(info.dependencies);
    }
  };
  for (const entry of JSON.parse(raw)) {
    walk(entry.dependencies);
    walk(entry.unsavedDependencies);
  }
  return found;
}

function packageDir(name, info) {
  if (info?.path && existsSync(info.path)) return info.path;
  try {
    return dirname(require.resolve(`${name}/package.json`));
  } catch {
    return null;
  }
}

function licenseTexts(dir) {
  return readdirSync(dir)
    .filter((f) => LICENSE_FILE.test(f))
    .sort()
    .map((f) => ({ file: f, text: readFileSync(join(dir, f), 'utf8').trimEnd() }));
}

function declaredLicense(pkg) {
  if (typeof pkg.license === 'string') return pkg.license;
  if (Array.isArray(pkg.licenses)) return pkg.licenses.map((l) => l.type).join(' OR ');
  return null;
}

export function javascriptSections(problems) {
  const out = [];
  const closure = [...productionPackages(problems)].sort(([a], [b]) => a.localeCompare(b));
  for (const [name, info] of closure) {
    const dir = packageDir(name, info);
    if (!dir) {
      problems.push(`${name}: could not resolve an install directory`);
      continue;
    }
    const pkg = JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8'));
    const spdx = declaredLicense(pkg);
    const texts = licenseTexts(dir);
    if (!spdx) problems.push(`${name}: package.json declares no license`);
    if (texts.length === 0) problems.push(`${name}: no LICENSE file found in ${dir}`);
    const home =
      pkg.homepage ??
      (typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url) ??
      '';
    out.push(
      [
        RULE,
        `${name} ${pkg.version ?? ''}`.trim(),
        `SPDX: ${spdx ?? 'UNDECLARED'}`,
        home ? `Home: ${home}` : null,
        RULE,
        '',
        texts.map((t) => t.text).join('\n\n'),
        '',
      ]
        .filter((l) => l !== null)
        .join('\n')
    );
  }
  return out;
}

/** Crates in the WASM build graph, with their license text from the registry. */
export function rustSections(problems) {
  let meta;
  try {
    meta = JSON.parse(
      execFileSync('cargo', ['metadata', '--format-version', '1'], {
        cwd: RUST_ROOT,
        encoding: 'utf8',
        maxBuffer: 64 * 1024 * 1024,
      })
    );
  } catch (err) {
    problems.push(
      `cargo metadata failed (${err.message.split('\n')[0]}). The production ` +
        `build already requires the Rust toolchain for build:wasm, so this is ` +
        `a broken toolchain rather than an optional step.`
    );
    return [];
  }
  const out = [];
  for (const pkg of [...meta.packages].sort((a, b) => a.name.localeCompare(b.name))) {
    if (pkg.name === 'luxar-wasm') continue; // Luxar's own; covered by the root LICENSE.
    const dir = pkg.manifest_path ? dirname(pkg.manifest_path) : null;
    const spdx = pkg.license ?? null;
    const texts = dir && existsSync(dir) ? licenseTexts(dir) : [];
    if (!spdx) problems.push(`crate ${pkg.name}: no license field`);
    if (texts.length === 0) problems.push(`crate ${pkg.name}: no license file under ${dir}`);
    out.push(
      [
        RULE,
        `${pkg.name} ${pkg.version} (Rust crate)`,
        `SPDX: ${spdx ?? 'UNDECLARED'}`,
        pkg.repository ? `Home: ${pkg.repository}` : null,
        RULE,
        '',
        texts.map((t) => `--- ${t.file} ---\n${t.text}`).join('\n\n'),
        '',
      ]
        .filter((l) => l !== null)
        .join('\n')
    );
  }
  return out;
}

/**
 * Colormap LUTs baked into both the Python package and the viewer bundle.
 *
 * Read from the generator's own tables rather than restated, so the split
 * between Luxar-computed ramps and matplotlib-sampled maps cannot drift from
 * the code that produces them.
 */
export function colormapSection(problems) {
  const gen = join(REPO_ROOT, 'scripts/generate_builtin_colormaps.py');
  if (!existsSync(gen)) {
    problems.push(`colormap generator not found at ${gen}`);
    return '';
  }
  const src = readFileSync(gen, 'utf8');
  const ramps = [
    ...(src.match(/^LINEAR_RAMPS = \{([\s\S]*?)^\}/m)?.[1] ?? '').matchAll(/"([a-z_]+)":/g),
  ].map((m) => m[1]);
  const mpl = [
    ...(src.match(/matplotlib_maps = \[([\s\S]*?)\]/)?.[1] ?? '').matchAll(/"([A-Za-z_]+)"/g),
  ].map((m) => m[1]);
  if (ramps.length === 0 || mpl.length === 0) {
    problems.push(
      `could not read the colormap tables out of ${gen} ` +
        `(ramps=${ramps.length}, matplotlib=${mpl.length})`
    );
    return '';
  }
  return [
    RULE,
    'Baked colormap lookup tables',
    RULE,
    '',
    `Computed by Luxar and covered by the root LICENSE (black-to-colour linear`,
    `ramps): ${ramps.join(', ')}.`,
    `Also Luxar's own: fire, ice, phase.`,
    '',
    `Sampled from matplotlib's colormaps at build time and baked as 256x3 uint8`,
    `LUTs: ${mpl.join(', ')}.`,
    'These originate upstream of matplotlib and matplotlib credits them as:',
    '',
    '  viridis, inferno, plasma  Nathaniel J. Smith, Stefan van der Walt and',
    '                            Eric Firing, contributed to matplotlib.',
    '  turbo                     Anton Mikhailov, Google.',
    '  RdBu                      ColorBrewer, developed by Cynthia Brewer',
    '                            (https://colorbrewer2.org/); matplotlib includes',
    '                            it under an Apache-style licence -- see',
    '                            LICENSE_COLORBREWER in the matplotlib source',
    '                            distribution.',
    '  coolwarm                  Kenneth Moreland, "Diverging Color Maps for',
    '                            Scientific Visualization"',
    '                            (http://www.kennethmoreland.com/color-maps/).',
    '',
    'For the exact terms, see the `LICENSE` directory of the matplotlib source',
    'distribution, which is the authority for each of these.',
    '',
  ].join('\n');
}

export function main() {
  const outDirs = process.argv.slice(2);
  if (outDirs.length === 0) {
    console.error('usage: generate-third-party-licenses.mjs <outdir> [<outdir> ...]');
    process.exit(2);
  }
  const problems = [];
  const js = javascriptSections(problems);
  const rust = rustSections(problems);
  const colormaps = colormapSection(problems);

  if (problems.length > 0) {
    console.error(
      `Refusing to write an incomplete THIRD_PARTY_LICENSES.txt ` +
        `(${problems.length} problem(s)):`
    );
    for (const p of problems) console.error(`  - ${p}`);
    console.error('\nA notices file that silently omits a dependency is worse than none.');
    process.exit(1);
  }

  // Embedded rather than referenced: `luxar export` produces a folder that is
  // handed around on its own, so "see the accompanying LICENSE file" would be a
  // dangling pointer exactly where the notices matter most.
  const ownLicense = readFileSync(join(REPO_ROOT, 'LICENSE'), 'utf8').trimEnd();

  const body = [
    'THIRD-PARTY SOFTWARE NOTICES',
    '',
    'This document lists the third-party components redistributed WITH the',
    'compiled viewer bundle, and reproduces their licences as those licences',
    'require. Luxar itself is BSD-3-Clause, reproduced immediately below.',
    '',
    RULE,
    'Luxar',
    RULE,
    '',
    ownLicense,
    '',
    'It is generated at build time from the pnpm production dependency closure,',
    'the cargo build graph of the WASM module, and the colormap generator, by',
    'packages/luxar-viewer/scripts/generate-third-party-licenses.mjs. Do not edit',
    'it by hand.',
    '',
    `JavaScript packages: ${js.length}. Rust crates: ${rust.length}.`,
    '',
    'Note: the Rust list is the full cargo build graph, which over-includes',
    'proc-macro crates that execute at compile time and contribute no code to the',
    'shipped .wasm. Over-attribution is deliberate; under-attribution is the risk.',
    '',
    colormaps,
    '',
    RULE,
    'JavaScript',
    RULE,
    '',
    js.join('\n'),
    RULE,
    'Rust / WebAssembly',
    RULE,
    '',
    rust.join('\n'),
  ].join('\n');

  for (const dir of outDirs) {
    const target = resolve(VIEWER_ROOT, dir);
    mkdirSync(target, { recursive: true });
    writeFileSync(join(target, 'THIRD_PARTY_LICENSES.txt'), body, 'utf8');
    console.log(
      `wrote ${join(dir, 'THIRD_PARTY_LICENSES.txt')} ` +
        `(${js.length} JS packages, ${rust.length} Rust crates)`
    );
  }
}

// Only run when INVOKED, so the sibling test can exercise the pieces without
// requiring a Rust toolchain (the TypeScript CI job has none).
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main();
}
