/**
 * Every source file in this package must be reached by a linter.
 *
 * This is the anti-decay guard for audit A15-09. 5,774 LOC across 37 build,
 * gate and config files had drifted outside eslint, prettier and tsc — not by
 * decision, but because `eslint src` and `prettier src` name one directory and
 * everything added elsewhere lands outside them silently. Widening the scope
 * once fixes today; only a check that reads the tree keeps it fixed.
 *
 * The selection was adverse, which is why it matters: the escaped files are the
 * Playwright configs and the `.mjs` scripts that enforce every OTHER gate. One
 * of them had five `eslint-disable no-console` directives that had never
 * suppressed anything, because nothing had ever linted the file.
 *
 * Derived on both sides. The file list comes from walking the tree, the
 * coverage answer from asking ESLint itself. Neither is a list written here
 * that could go stale — the failure this guards against IS a stale list.
 */

import { describe, it, expect } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'fs';
import { dirname, join, relative, resolve } from 'path';
import { fileURLToPath } from 'url';
import { ESLint } from 'eslint';

const PKG = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Extensions that carry executable code and therefore must be linted. */
const CODE = ['.ts', '.tsx', '.js', '.mjs', '.cjs'];

/**
 * Directories holding generated or downloaded output, never authored here.
 *
 * Deliberately short and each entry justified. A long list is how a scope gate
 * turns back into decoration.
 */
const NOT_AUTHORED = new Set([
  'node_modules',
  'dist', // vite output
  'coverage', // vitest output
  'public', // wasm-pack output (public/wasm)
  'test-results', // playwright run artifacts
  'test-results-perf',
  'playwright-report',
  '.turbo',
]);

/** Every authored code file in the package, relative to it. */
function authoredFiles(dir = PKG) {
  const out = [];
  const walk = (d) => {
    for (const name of readdirSync(d)) {
      if (name.startsWith('.') || NOT_AUTHORED.has(name)) continue;
      const full = join(d, name);
      if (statSync(full).isDirectory()) {
        // Rust's target/ lives under src/wasm/rust and is build output.
        if (name === 'target') continue;
        walk(full);
      } else if (CODE.some((ext) => name.endsWith(ext)) && !name.endsWith('.d.ts')) {
        out.push(relative(PKG, full));
      }
    }
  };
  walk(dir);
  return out;
}

const FILES = authoredFiles();

describe('lint scope', () => {
  it('found a tree to check at all', () => {
    // Fail closed. A walk that returns nothing — wrong cwd, a rename, a broken
    // exclusion — would make every assertion below vacuously true, which is the
    // precise shape of the bug this file exists to catch.
    expect(FILES.length).toBeGreaterThan(300);
    expect(FILES.filter((f) => f.startsWith('scripts/')).length).toBeGreaterThan(5);
    expect(FILES.filter((f) => f.startsWith('tools/')).length).toBeGreaterThan(5);
    expect(FILES.filter((f) => !f.startsWith('src/')).length).toBeGreaterThan(20);
  });

  it('reaches every authored file with an eslint configuration', async () => {
    const eslint = new ESLint({ cwd: PKG });

    // `@typescript-eslint/no-unused-vars` is the marker: both the `src` block
    // and the tooling block set it, and nothing else does. A file matched by
    // NEITHER still resolves a config (js.configs.recommended has no `files`,
    // so it applies universally) — which is why "does it have a config" is not
    // the question to ask. "Is it inside a block someone wrote for it" is.
    const MARKER = '@typescript-eslint/no-unused-vars';

    const unreached = [];
    for (const file of FILES) {
      const abs = join(PKG, file);
      if (await eslint.isPathIgnored(abs)) {
        unreached.push(`${file} (ignored)`);
        continue;
      }
      const config = await eslint.calculateConfigForFile(abs);
      if (!config?.rules?.[MARKER]) unreached.push(`${file} (matched by no files: block)`);
    }

    expect(
      unreached,
      `add these to a \`files:\` block in eslint.config.js:\n${unreached.join('\n')}`
    ).toEqual([]);
  });
});

describe('prettier scope', () => {
  /** The paths the format scripts actually pass to prettier. */
  function formatTargets() {
    const pkg = JSON.parse(readFileSync(join(PKG, 'package.json'), 'utf8'));
    // Both scripts must agree, or `format` would fix files `check:format`
    // never inspects (or worse, the reverse — a permanently red gate).
    const write = pkg.scripts.format.replace('--write', '');
    const check = pkg.scripts['check:format'].replace('--check', '');
    expect(write).toBe(check);
    return check;
  }

  it('formats every top-level directory that holds authored code', () => {
    const targets = formatTargets();
    const dirs = new Set(
      FILES.map((f) => f.split('/')[0]).filter((d) => d.includes('.') === false)
    );
    for (const dir of dirs) {
      expect(targets, `prettier does not cover ${dir}/`).toContain(dir);
    }
  });

  it('formats the root-level config files', () => {
    // These are the five Playwright configs, both Vite configs and both Vitest
    // configs — the files the audit found outside every tool.
    expect(FILES.some((f) => !f.includes('/') && f.endsWith('.config.ts'))).toBe(true);
    expect(formatTargets()).toMatch(/\*\.\{[^}]*ts[^}]*\}/);
  });
});
