/**
 * Structural tripwire: all direct loading of the BUILT WASM artifact goes
 * through `src/tests/helpers/wasm-artifact.ts`.
 *
 * That helper calls `assertRequiredWasmExports` on the freshly initialised
 * namespace before casting it to `WasmModule`, so a stale gitignored build fails
 * as "missing required export `<kernel>`" instead of an opaque "x is not a
 * function" from whichever kernel assertion runs first (#1412). A NEW loader
 * written from scratch would silently lose that, which no behavioural test on
 * the existing callers can see — hence a source-level check.
 *
 * It keys on what a Node-side loader has to DO: name the shim and `initSync(`
 * it. It deliberately does NOT inspect how the path is composed, how the module
 * is cast, or where the guard call sits — the previous version did, and each was
 * dodgeable by writing the same bug slightly differently. Someone determined to
 * hide a loader still can (assemble the filename from fragments, or await the
 * shim's async `default()` init); the case worth catching is the accidental one.
 */

import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

/** This file's absolute path — the scanner quotes both markers below. */
const THIS_FILE = resolve(fileURLToPath(import.meta.url));
/** `packages/luxar-viewer` — this file lives at src/tests/unit/wasm/. */
const PACKAGE_ROOT = resolve(THIS_FILE, '../../../../..');
/**
 * TypeScript roots of the viewer package. `src/` is not enough: `tools/` and
 * `scripts/` hold executable TS covered by `tsconfig.tooling.json`, so a loader
 * added in either would be invisible. A missing root is skipped.
 */
const SCAN_ROOTS = ['src', 'tools', 'scripts'];

/** The sanctioned loader — the one file allowed to match both markers. */
const HELPER = 'src/tests/helpers/wasm-artifact.ts';
/**
 * The production loader, exempt because it legitimately loads the artifact for
 * the app (and calls the guard inside the `try` whose `catch` returns the
 * documented TypeScript fallback). One path is enough now that it is the only
 * exemption; if this file moves, the failure below names it and says so.
 */
const PRODUCTION_LOADER = 'src/wasm/index.ts';

const ARTIFACT_MARKER = 'luxar_wasm.js';
const INIT_MARKER = 'initSync(';

/** Package-relative, POSIX-separated path — stable across platforms. */
function label(file: string): string {
  return relative(PACKAGE_ROOT, file).split(sep).join('/');
}

/**
 * Package-relative labels of every `.ts`/`.tsx` under the scan roots that
 * mentions the artifact AND initialises it. Reads tolerate ENOENT: other agents
 * edit this tree concurrently (CLAUDE.md rule 7), so a file vanishing between
 * the walk and the read must not fail the scan.
 */
function artifactLoaders(): string[] {
  const found: string[] = [];
  for (const root of SCAN_ROOTS) {
    const dir = resolve(PACKAGE_ROOT, root);
    if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
    for (const entry of readdirSync(dir, { recursive: true }) as string[]) {
      if (!entry.endsWith('.ts') && !entry.endsWith('.tsx')) continue;
      const file = resolve(dir, entry);
      if (file === THIS_FILE) continue;
      let text: string;
      try {
        text = readFileSync(file, 'utf8');
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue;
        throw error;
      }
      if (text.includes(ARTIFACT_MARKER) && text.includes(INIT_MARKER)) found.push(label(file));
    }
  }
  return found.sort();
}

const loaders = artifactLoaders();

describe('direct WASM artifact loading', () => {
  it('finds the shared helper (markers still match reality)', () => {
    expect(
      loaders,
      `${HELPER} no longer matches "${ARTIFACT_MARKER}" + "${INIT_MARKER}", so ` +
        'this scan proves nothing. Update the markers to whatever a loader ' +
        'must now contain, or fix SCAN_ROOTS.'
    ).toContain(HELPER);
  });

  it('has no loader outside the shared helper', () => {
    const rogue = loaders.filter((file) => file !== HELPER && file !== PRODUCTION_LOADER);
    expect(
      rogue,
      `These files load the built WASM artifact themselves: ${rogue.join(', ')}. ` +
        `Import loadWasmArtifact / tryLoadWasmArtifact from ${HELPER} instead — it ` +
        'calls assertRequiredWasmExports before the WasmModule cast and outside the ' +
        'catch that downgrades a load failure to a skip, so a stale gitignored build ' +
        'names the missing kernel instead of failing as an opaque "x is not a ' +
        `function". (If ${PRODUCTION_LOADER} moved, update PRODUCTION_LOADER here.)`
    ).toEqual([]);
  });
});
