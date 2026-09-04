import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  colormapSection,
  javascriptSections,
  productionPackages,
  rustSections,
} from './generate-third-party-licenses.mjs';

/**
 * The notices file is a legal artifact, and its failure mode is silence: a
 * dependency that is bundled but unlisted looks exactly like a dependency that
 * was checked. So every assertion here is about the generator REFUSING to be
 * quietly incomplete, not about the happy path producing something.
 */
describe('third-party licence notices', () => {
  it('lists every production package the bundle actually pulls in', () => {
    const problems = [];
    const sections = javascriptSections(problems);
    const packages = [...productionPackages().keys()];

    // Derived on both sides: the expectation is the pnpm closure itself, so a
    // new dependency cannot be added without appearing here.
    expect(packages.length).toBeGreaterThan(5);
    expect(sections).toHaveLength(packages.length);
    for (const name of packages) {
      expect(sections.join('\n')).toContain(name);
    }
    expect(problems).toEqual([]);
  });

  it('reproduces the licence TEXT, not just the SPDX identifier', () => {
    // Naming a licence is not the same as reproducing it, and reproducing it is
    // the actual obligation for MIT / BSD / Apache binary redistribution.
    const body = javascriptSections([]).join('\n');
    expect(body).toContain('Permission is hereby granted, free of charge');
    // mediabunny is MPL-2.0, the one weak-copyleft dependency in the tree.
    expect(body).toContain('Mozilla Public License');
  });

  it('records a problem rather than emitting an empty Rust section', () => {
    // Holds in BOTH environments: with a Rust toolchain the crates are found,
    // without one `cargo metadata` fails and a problem is recorded. What must
    // never happen is an empty section with an empty problem list, which is
    // what a `try { } catch { return [] }` would give.
    const problems = [];
    const sections = rustSections(problems);
    expect(sections.length > 0 || problems.length > 0).toBe(true);
    if (sections.length > 0) {
      expect(problems).toEqual([]);
      expect(sections.join('\n')).toContain('wasm-bindgen');
    }
  });

  it("attributes the colormap LUTs that are not Luxar's own", () => {
    const problems = [];
    const section = colormapSection(problems);
    expect(problems).toEqual([]);
    // Read out of the colormap generator's own tables, so this cannot drift
    // from the code that bakes the LUTs.
    for (const name of ['viridis', 'inferno', 'plasma', 'turbo', 'RdBu', 'coolwarm']) {
      expect(section).toContain(name);
    }
    // The upstream originators matplotlib itself credits.
    expect(section).toContain('Cynthia Brewer');
    expect(section).toContain('Kenneth Moreland');
    expect(section).toContain('Anton Mikhailov');
    // And the ramps Luxar computes, including the four that were once orphaned
    // from the generator.
    for (const name of ['bop_blue', 'bop_orange', 'bop_purple', 'orange']) {
      expect(section).toContain(name);
    }
  });

  it('REFUSES to write a file it cannot complete', () => {
    // The guard the whole script exists for, exercised end to end rather than
    // asserted about. Running it with an empty PATH makes `cargo metadata`
    // unrunnable, which is the realistic shape of "one section could not be
    // built". It must exit non-zero AND leave no file behind -- a partial
    // notices file is worse than none, because it looks like diligence.
    const out = mkdtempSync(join(tmpdir(), 'luxar-licences-'));
    const result = spawnSync(
      process.execPath,
      [fileURLToPath(new URL('./generate-third-party-licenses.mjs', import.meta.url)), out],
      { encoding: 'utf8', env: { PATH: '' } }
    );

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain('Refusing to write an incomplete');
    expect(existsSync(join(out, 'THIRD_PARTY_LICENSES.txt'))).toBe(false);
  });
});
