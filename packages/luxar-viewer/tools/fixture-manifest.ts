/**
 * The generated-zarr-fixture manifest, shared by the vitest and Playwright global setups.
 *
 * `tests/fixtures/generate_test_data.py` declares `FIXTURE_NAMES` and asserts at the end of
 * `main()` that every name in it was actually produced. That constant is the single source of
 * truth for which fixtures exist; this module is the single place that reads it.
 *
 * It lives in `tools/` rather than `src/` for two reasons: `src/` is the shipped library (a
 * test-only helper there is dead weight in the bundle and fails `knip --include files`), and
 * `tools/**\/*.ts` is already a knip entry point alongside `e2e-server-identity.ts`, the other
 * helper both harnesses share.
 *
 * @module tools/fixture-manifest
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

/**
 * Where the fixtures live, relative to the repository root.
 *
 * Load-bearing as BOTH a filesystem suffix and a URL path: the E2E data server is a plain
 * `http.server` rooted at the repository, so the 19 specs that read fixtures spell them
 * `http://localhost:9000/packages/luxar-viewer/tests/fixtures/<name>`. Keeping one constant
 * means a directory move cannot leave the HTTP probe checking a path nothing serves.
 */
export const FIXTURES_REPO_RELATIVE_PATH = 'packages/luxar-viewer/tests/fixtures';

/**
 * Parse fixture names from generate_test_data.py — the single source of truth.
 *
 * Reads the declarative `FIXTURE_NAMES: list[str] = [...]` constant at the top of the Python
 * generator (audit C1 viewer-integration-fixtures fix). Previously this regex matched scattered
 * `FIXTURES_DIR / "..."` usages, which silently broke if a generator function switched to single
 * quotes, f-strings, or path concatenation. Targeting a single canonical declaration is robust to
 * those variations.
 *
 * The Python script asserts at the end of main() that every name in FIXTURE_NAMES was actually
 * produced — keeping the manifest and the generators in sync.
 *
 * @param generatorPath Absolute path to `tests/fixtures/generate_test_data.py`.
 * @returns The fixture directory names, sorted and de-duplicated.
 */
export function parseGeneratedFixtureNames(generatorPath: string): string[] {
  const source = readFileSync(generatorPath, 'utf-8');
  // Match the FIXTURE_NAMES list declaration. The body captures
  // everything between the [ and ] including newlines; we then pull out
  // each "..." or '...' literal ending in .zarr.
  const listMatch = /FIXTURE_NAMES\s*(?::[^=]*)?=\s*\[([^\]]+)\]/.exec(source);
  if (!listMatch) {
    throw new Error(
      `[test-setup] FIXTURE_NAMES declaration not found in ${generatorPath}. ` +
        'Expected a top-level `FIXTURE_NAMES: list[str] = [...]` block. ' +
        'Update generate_test_data.py to declare the manifest, or update this parser.'
    );
  }
  const body = listMatch[1];
  const re = /['"]([^'"]+\.zarr)['"]/g;
  const names = new Set<string>();
  let m: RegExpExecArray | null;
  while ((m = re.exec(body)) !== null) {
    names.add(m[1]);
  }
  if (names.size === 0) {
    throw new Error(
      `[test-setup] FIXTURE_NAMES list in ${generatorPath} is empty or unparseable. ` +
        'Expected `.zarr`-terminated string literals.'
    );
  }
  return [...names].sort();
}

/**
 * Whether a fixture directory is COMPLETE, not merely present.
 *
 * An existence check alone accepts a directory the generator was interrupted while
 * writing — Ctrl-C on the last fixture leaves every other one whole, so a plain
 * `existsSync` sweep reports the whole set ready and the spec that reads the stump dies
 * on its own 45 s content-wait, which is exactly the opaque failure the preflight exists
 * to replace.
 *
 * `.zmetadata` is the completeness signal because `LuxarZarrCompiler.finalize()` writes it
 * last, via `zarr.consolidate_metadata()`, after every array and attribute is in place
 * (`packages/luxar/src/luxar/cli/gsplat_ops/batch/validation.py` reads it the same way for
 * the same reason). Cheap enough to run per fixture: one `stat` each, no HTTP.
 *
 * @param fixturePath Absolute path to a `*.zarr` fixture directory.
 */
export function isGeneratedFixtureComplete(fixturePath: string): boolean {
  return existsSync(join(fixturePath, '.zmetadata'));
}
