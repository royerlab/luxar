import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

/**
 * A browser-support table may only name engines the suite can actually run.
 *
 * The READMEs asserted four browsers "fully supported", with version floors, at
 * a time when `playwright.config.ts` declared exactly ONE project -- firefox and
 * webkit were commented out, so the claim had never been executed even once.
 * Commented-out config is not a plan; it is a claim with nothing behind it.
 *
 * Both sides are read from the tree: the engines from the Playwright config, the
 * claims from the markdown. Nothing here is a list to keep in step by hand.
 */
const VIEWER_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO_ROOT = resolve(VIEWER_ROOT, '../..');

const DOCS = [join(REPO_ROOT, 'README.md'), join(VIEWER_ROOT, 'README.md')];

/** Project names declared in playwright.config.ts, opt-in ones included. */
function declaredProjects() {
  const config = readFileSync(join(VIEWER_ROOT, 'playwright.config.ts'), 'utf8');
  // Ignore commented-out lines: a project that cannot run does not count, which
  // is the whole point.
  const live = config
    .split('\n')
    .filter((line) => !line.trim().startsWith('//'))
    .join('\n');
  return new Set([...live.matchAll(/name:\s*'([a-z]+)'/g)].map((m) => m[1]));
}

/** Engine names claimed by a "Browser Compatibility" table's rows. */
function claimedEngines(markdown) {
  const section = markdown.split('### Browser Compatibility')[1];
  if (section === undefined) return null;
  const table = section.split('\n\n').find((b) => b.trimStart().startsWith('|'));
  if (!table) return new Set();
  return new Set(
    table
      .split('\n')
      .slice(2) // header + separator
      .map((row) => row.split('|')[1]?.trim().toLowerCase())
      .filter((cell) => cell && !/^-+$/.test(cell))
  );
}

describe('browser support matrix', () => {
  it('declares the engines it claims to have tested', () => {
    const projects = declaredProjects();
    // Fail closed: a config we cannot parse would make every claim pass.
    expect(projects.size).toBeGreaterThanOrEqual(1);
    expect(projects).toContain('chromium');

    for (const doc of DOCS) {
      const claimed = claimedEngines(readFileSync(doc, 'utf8'));
      expect(claimed, `${doc} has no "### Browser Compatibility" section`).not.toBeNull();
      expect(claimed.size, `${doc} claims no engines at all`).toBeGreaterThan(0);
      for (const engine of claimed) {
        expect(
          projects,
          `${doc} claims "${engine}" was tested, but playwright.config.ts declares ` +
            `no such project (live projects: ${[...projects].join(', ')}). Either ` +
            `add the project or stop claiming the engine.`
        ).toContain(engine);
      }
    }
  });

  it('keeps firefox and webkit runnable rather than commented out', () => {
    // They are opt-in behind LUXAR_E2E_BROWSERS=all so the default suite stays
    // at one engine, but they must be REAL projects: the previous state made
    // the documented matrix impossible to check at all.
    const config = readFileSync(join(VIEWER_ROOT, 'playwright.config.ts'), 'utf8');
    expect(config).toContain("LUXAR_E2E_BROWSERS === 'all'");
    for (const engine of ['firefox', 'webkit']) {
      expect(config).not.toMatch(new RegExp(`//\\s*name:\\s*'${engine}'`));
    }
  });
});
