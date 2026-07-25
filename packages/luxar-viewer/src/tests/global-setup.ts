/**
 * Vitest global setup — runs once before the entire test suite.
 *
 * Ensures Python-generated zarr test fixtures exist, and that the compiled
 * WASM module is built so the WASM-vs-TS parity tests actually run (rather
 * than silently skipping and reporting green having verified nothing about
 * the compiled backend).
 *
 * If any fixtures are missing, runs the generator script automatically.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'fs';
import { execSync } from 'child_process';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';

// Use import.meta.url for reliable path resolution in vitest global setup
const THIS_DIR = resolve(fileURLToPath(import.meta.url), '..');
const VIEWER_ROOT = resolve(THIS_DIR, '../..');
const FIXTURES_DIR = resolve(VIEWER_ROOT, 'tests/fixtures');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');
const WASM_JS_PATH = resolve(VIEWER_ROOT, 'public/wasm/luxar_wasm.js');
const WASM_BIN_PATH = resolve(VIEWER_ROOT, 'public/wasm/luxar_wasm_bg.wasm');
const GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_test_data.py');
const EXPECTATIONS_GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_expectations.py');
const EXPECTATIONS_PATH = resolve(FIXTURES_DIR, 'roundtrip_expectations.json');
/**
 * The Python packages the fixtures are generated THROUGH. A change in any of
 * these alters what the generator writes without touching the generator
 * script itself, so fixture staleness must be measured against them too:
 *   - `encoding/` — array encodings (e.g. #448's uint16 per-axis fixed-point
 *     for COORDINATE positions, the miss that motivated this check);
 *   - `io/` — the LuxarZarrCompiler machinery (`io/_compiler` chunking,
 *     spatial ordering, gsplat assembly/tree) every fixture byte flows through.
 * Deliberately NOT the whole `luxar/` package: fitting/CLI/demo code does not
 * affect compiled-fixture bytes, and over-widening would regenerate the
 * ~minute-long fixture set on every unrelated Python edit.
 */
const FIXTURE_INPUT_SOURCE_DIRS = [
  resolve(PROJECT_ROOT, 'packages/luxar/src/luxar/encoding'),
  resolve(PROJECT_ROOT, 'packages/luxar/src/luxar/io'),
];

/**
 * Parse fixture names from generate_test_data.py — the single source of truth.
 *
 * Reads the declarative `FIXTURE_NAMES: list[str] = [...]` constant at the
 * top of the Python generator (audit C1 viewer-integration-fixtures fix).
 * Previously this regex matched scattered `FIXTURES_DIR / "..."` usages,
 * which silently broke if a generator function switched to single quotes,
 * f-strings, or path concatenation. Targeting a single canonical
 * declaration is robust to those variations.
 *
 * The Python script asserts at the end of main() that every name in
 * FIXTURE_NAMES was actually produced — keeping the manifest and the
 * generators in sync.
 */
function parseGeneratedFixtureNames(): string[] {
  const source = readFileSync(GENERATOR_PATH, 'utf-8');
  // Match the FIXTURE_NAMES list declaration. The body captures
  // everything between the [ and ] including newlines; we then pull out
  // each "..." or '...' literal ending in .zarr.
  const listMatch = /FIXTURE_NAMES\s*(?::[^=]*)?=\s*\[([^\]]+)\]/.exec(source);
  if (!listMatch) {
    throw new Error(
      `[test-setup] FIXTURE_NAMES declaration not found in ${GENERATOR_PATH}. ` +
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
      `[test-setup] FIXTURE_NAMES list in ${GENERATOR_PATH} is empty or unparseable. ` +
        'Expected `.zarr`-terminated string literals.'
    );
  }
  return [...names].sort();
}

const EXPECTED_FIXTURES = parseGeneratedFixtureNames();

function runPythonGenerator(command: string, label: string): void {
  try {
    execSync(command, {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      timeout: 120_000,
    });
    console.log(`[test-setup] ${label} generated successfully.`);
  } catch (err: unknown) {
    // execSync errors include stderr/stdout as Buffers
    const stderr =
      err && typeof err === 'object' && 'stderr' in err && err.stderr ? String(err.stderr) : '';
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[test-setup] Failed to generate ${label}: ${message}`);
    if (stderr) console.error(stderr);
    console.error(`[test-setup] Run manually: ${command}`);
    throw new Error(`${label} generation failed. See above for details.`);
  }
}

/** Newest mtime of any `.py` source under `dir` (non-recursive dirs skipped safely). */
function newestPySourceMtime(dir: string): number {
  if (!existsSync(dir)) return 0;
  let newest = 0;
  for (const entry of readdirSync(dir, { recursive: true }) as string[]) {
    if (!entry.endsWith('.py')) continue;
    const full = join(dir, entry);
    if (!existsSync(full)) continue;
    const mtime = statSync(full).mtimeMs;
    if (mtime > newest) newest = mtime;
  }
  return newest;
}

/**
 * Whether any EXISTING zarr fixture predates its generation inputs — the
 * generator script itself or the Python encoder/compiler sources it writes
 * through (see FIXTURE_INPUT_SOURCE_DIRS).
 *
 * The generate-if-MISSING gate alone let #448 slip through: the fixtures all
 * existed (git-ignored, generated locally in June) but still carried the old
 * `float32` positions encoding, so five array-decoder tests failed while
 * setup regenerated nothing. Missing fixtures are handled separately by the
 * caller; this only compares mtimes of the ones present.
 */
function areFixturesStale(fixtureNames: string[]): boolean {
  const inputsMtime = Math.max(
    statSync(GENERATOR_PATH).mtimeMs,
    ...FIXTURE_INPUT_SOURCE_DIRS.map(newestPySourceMtime)
  );
  return fixtureNames.some((name) => {
    const path = resolve(FIXTURES_DIR, name);
    return existsSync(path) && statSync(path).mtimeMs < inputsMtime;
  });
}

function isExpectationsStale(fixtureNames: string[]): boolean {
  if (!existsSync(EXPECTATIONS_PATH)) return true;

  const expectationsMtime = statSync(EXPECTATIONS_PATH).mtimeMs;
  const dependencyPaths = [
    GENERATOR_PATH,
    EXPECTATIONS_GENERATOR_PATH,
    ...fixtureNames.map((name) => resolve(FIXTURES_DIR, name)),
  ];

  return dependencyPaths.some(
    (dependencyPath) =>
      existsSync(dependencyPath) && statSync(dependencyPath).mtimeMs > expectationsMtime
  );
}

/**
 * Ensure the compiled WASM module exists so the WASM-vs-TS parity tests run.
 *
 * The parity harness (`wasm-vs-typescript.test.ts`) skips every case via
 * `it.skipIf(!wasmFilesExist)` when the artifacts are absent — which means a
 * dev box that never ran `make build-wasm` gets a fully-green suite that has
 * verified NOTHING about the compiled backend (the only guard against Rust↔TS
 * drift). To avoid that silent-coverage trap:
 *   - if the artifacts are missing and a Rust/wasm-pack toolchain is present,
 *     build them automatically (mirrors the auto fixture-generation above);
 *   - if no toolchain is available, emit a LOUD warning so the skip is visible;
 *   - if `LUXAR_REQUIRE_WASM_TESTS=1` (CI), a missing/unbuildable module is a
 *     hard failure.
 */
export function ensureWasmBuilt(): void {
  if (existsSync(WASM_JS_PATH) && existsSync(WASM_BIN_PATH)) return;

  const require = process.env.LUXAR_REQUIRE_WASM_TESTS === '1';
  const hasToolchain = ((): boolean => {
    try {
      execSync('command -v wasm-pack', { stdio: 'ignore', shell: '/bin/bash' });
      return true;
    } catch {
      return false;
    }
  })();

  if (!hasToolchain) {
    const msg =
      '[test-setup] Compiled WASM not found and wasm-pack is not installed.\n' +
      '             WASM-vs-TypeScript parity tests will be SKIPPED — the compiled\n' +
      '             backend is NOT being verified. Install Rust + wasm-pack and run\n' +
      '             `make build-wasm` (or `pnpm build:wasm`) for full coverage.';
    if (require) {
      throw new Error(`${msg}\n(LUXAR_REQUIRE_WASM_TESTS=1 — refusing to run without WASM.)`);
    }
    console.warn(`\n⚠️  ${msg}\n`);
    return;
  }

  console.log('[test-setup] Compiled WASM missing — building via `pnpm build:wasm`...');
  try {
    execSync('pnpm run build:wasm', { cwd: VIEWER_ROOT, stdio: 'inherit' });
  } catch (err) {
    const msg = `[test-setup] WASM build failed: ${err instanceof Error ? err.message : String(err)}`;
    if (require) throw new Error(msg);
    console.warn(`\n⚠️  ${msg}\n   Parity tests will be skipped.\n`);
    return;
  }
  if (!existsSync(WASM_JS_PATH) || !existsSync(WASM_BIN_PATH)) {
    const msg = '[test-setup] WASM build ran but artifacts are still missing.';
    if (require) throw new Error(msg);
    console.warn(`\n⚠️  ${msg}\n`);
  }
}

export async function setup(): Promise<void> {
  ensureWasmBuilt();

  const missing = EXPECTED_FIXTURES.filter((name) => !existsSync(resolve(FIXTURES_DIR, name)));
  const stale = areFixturesStale(EXPECTED_FIXTURES);

  if (missing.length > 0 || stale) {
    console.log(
      missing.length > 0
        ? `\n[test-setup] ${missing.length} zarr fixture(s) missing — generating...`
        : '\n[test-setup] zarr fixtures predate the generator/encoder sources — regenerating...'
    );
    runPythonGenerator(
      'hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py',
      'fixtures'
    );
  }

  // Verify fixture generation succeeded before generating expectations from them.
  const stillMissing = EXPECTED_FIXTURES.filter((name) => !existsSync(resolve(FIXTURES_DIR, name)));
  if (stillMissing.length > 0) {
    throw new Error(
      `Fixture generation ran but these are still missing: ${stillMissing.join(', ')}`
    );
  }

  if (isExpectationsStale(EXPECTED_FIXTURES)) {
    console.log('[test-setup] Round-trip expectations missing/stale — generating...');
    runPythonGenerator(
      'hatch run python packages/luxar-viewer/tests/fixtures/generate_expectations.py',
      'round-trip expectations'
    );
  }
}
