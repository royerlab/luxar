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

import { existsSync, readFileSync } from 'fs';
import { execSync } from 'child_process';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import {
  areFixturesStale,
  ensureGeneratedFixtures,
  FIXTURE_GENERATOR_TIMEOUT_MS,
} from '../../tools/fixture-freshness';
import {
  isGeneratedFixtureComplete,
  parseGeneratedFixtureNames,
} from '../../tools/fixture-manifest';
import { REQUIRED_WASM_EXPORTS } from '../wasm/required-exports';

// Use import.meta.url for reliable path resolution in vitest global setup
const THIS_DIR = resolve(fileURLToPath(import.meta.url), '..');
const VIEWER_ROOT = resolve(THIS_DIR, '../..');
const FIXTURES_DIR = resolve(VIEWER_ROOT, 'tests/fixtures');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');
const WASM_JS_PATH = resolve(VIEWER_ROOT, 'public/wasm/luxar_wasm.js');
const WASM_BIN_PATH = resolve(VIEWER_ROOT, 'public/wasm/luxar_wasm_bg.wasm');
const GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_test_data.py');
const EXPECTED_FIXTURES = parseGeneratedFixtureNames(GENERATOR_PATH);

/**
 * Wall-clock budget for one generator run.
 *
 * Measured: `generate_test_data.py` takes ~215 s on an M-series laptop, so the
 * previous 120 s could not finish it — every regeneration was SIGTERM'd
 * mid-write, which leaves incomplete stores AND relands on the same wall the
 * next run, because the stamp is only written on success. The failure reads as
 * `spawnSync ETIMEDOUT`, which looks like a hung shell rather than a budget
 * that was never survivable.
 *
 * The first run also creates the separate ~1.2 GB `fixtures` Hatch environment.
 * The 1,200 s default preserves the previous 600 s generation budget plus the
 * same allowance for that one-time download/install. `LUXAR_FIXTURE_GEN_TIMEOUT_MS`
 * overrides it rather than requiring a source edit on a machine that needs more.
 */
function runPythonGenerator(command: string, label: string): void {
  try {
    execSync(command, {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      timeout: FIXTURE_GENERATOR_TIMEOUT_MS,
    });
    console.log(`[test-setup] ${label} generated successfully.`);
  } catch (err: unknown) {
    // execSync errors include stderr/stdout as Buffers
    const stderr =
      err && typeof err === 'object' && 'stderr' in err && err.stderr ? String(err.stderr) : '';
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[test-setup] Failed to generate ${label}: ${message}`);
    if (stderr) console.error(stderr);
    // A timeout reads as `spawnSync ETIMEDOUT`, which looks like a hung shell.
    // Say which budget was exceeded and how to raise it, so the next person is
    // not left comparing a generator that works by hand against one that does
    // not work here.
    if (err && typeof err === 'object' && (err as { code?: string }).code === 'ETIMEDOUT') {
      console.error(
        `[test-setup] ...that was the ${FIXTURE_GENERATOR_TIMEOUT_MS} ms budget, not a hang. ` +
          'Raise it with LUXAR_FIXTURE_GEN_TIMEOUT_MS if this machine is slower.'
      );
    }
    console.error(`[test-setup] Run manually: ${command}`);
    throw new Error(`${label} generation failed. See above for details.`, { cause: err });
  }
}

/**
 * Required exports that `wrapperSource` — the text of a built `luxar_wasm.js` —
 * does not declare. Empty when the build is current.
 *
 * wasm-pack emits one `export function <name>(` per kernel, which is what the
 * pattern anchors on. Matching the bare name would not do: the wrapper also
 * contains an internal `wasm.<name>(...)` call for every kernel it forwards, so
 * a substring scan reports a stale build as current.
 *
 * Exported so `wasm-export-scan.test.ts` can pin both directions — a detector
 * that never detects is worse than none.
 */
export function missingExportsIn(wrapperSource: string): readonly string[] {
  return REQUIRED_WASM_EXPORTS.filter(
    (name) => !new RegExp(`export function ${name}\\b`).test(wrapperSource)
  );
}

/**
 * Required exports that the built `luxar_wasm.js` does not declare — i.e. the
 * evidence that a PRESENT build predates a kernel. Empty when it is current.
 *
 * A text scan of the wrapper, not an import: this runs in plain Node before any
 * browser environment exists, and the wrapper's own `import` would initialise
 * the module.
 *
 * Deliberately NOT an mtime comparison against `src/wasm/rust/`. A `git
 * checkout` or a fresh worktree can leave an artifact NEWER than the source it
 * does not match, so mtime reports fresh exactly when it is most wrong. Export
 * membership is the signal that actually goes stale.
 */
function missingWasmExports(): readonly string[] {
  let wrapper: string;
  try {
    wrapper = readFileSync(WASM_JS_PATH, 'utf8');
  } catch {
    // Unreadable but PRESENT — `existsSync` has already passed, so this is not
    // the absence check firing. It is a real artifact we cannot vouch for, and
    // returning "nothing missing" here would trust it and skip the rebuild:
    // the exact fail-open this function exists to close. Report everything
    // missing so the caller rebuilds.
    return REQUIRED_WASM_EXPORTS;
  }
  return missingExportsIn(wrapper);
}

/**
 * Ensure a CURRENT compiled WASM module exists so the WASM-vs-TS parity tests
 * run against the real backend.
 *
 * The parity harness (`wasm-vs-typescript.test.ts`) skips every case via
 * `it.skipIf(!wasmFilesExist)` when the artifacts are absent — which means a
 * dev box that never ran `make build-wasm` gets a fully-green suite that has
 * verified NOTHING about the compiled backend (the only guard against Rust↔TS
 * drift). A build that is merely STALE is worse still: it does not skip, it is
 * loaded — and the shared loader (`src/tests/helpers/wasm-artifact.ts`) aborts
 * every suite that loads it with "missing required export `<kernel>`". To
 * avoid both traps:
 *   - if the artifacts are missing OR out of date (see `missingWasmExports`
 *     below) and a Rust/wasm-pack toolchain is present, build them
 *     automatically (mirrors the auto fixture-generation above);
 *   - if no toolchain is available, emit a LOUD warning naming what will
 *     actually happen — a missing build skips the parity harness, a stale one
 *     runs and fails;
 *   - if `LUXAR_REQUIRE_WASM_TESTS=1` (CI), a missing/stale/unbuildable module
 *     is a hard failure.
 */
export function ensureWasmBuilt(): void {
  const present = existsSync(WASM_JS_PATH) && existsSync(WASM_BIN_PATH);
  // A build missing a kernel is as useless as no build: it imports and
  // initialises fine, then aborts every suite that loads it in `beforeAll` with
  // "missing required export `<kernel>`". Same branch, same fix.
  const stale = present ? missingWasmExports() : [];
  if (present && stale.length === 0) return;

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
    // The two cases have OPPOSITE consequences, so they must not share a
    // sentence. `it.skipIf(!wasmFilesExist)` keys on PRESENCE: a missing build
    // skips the parity harness (silent under-coverage), while a stale one is
    // loaded — and the shared loader aborts the file in `beforeAll` on the
    // first kernel it does not export. Only deleting the artifact turns the
    // second case into the first.
    const msg = stale.length
      ? `[test-setup] Compiled WASM is STALE (missing ${stale.join(', ')}) and wasm-pack\n` +
        '             is not installed. A stale build is NOT skipped — the WASM-vs-\n' +
        '             TypeScript parity tests load it and ABORT in `beforeAll` with\n' +
        '             "missing required export <kernel>". Install Rust + wasm-pack and\n' +
        '             run `make build-wasm` (or `pnpm build:wasm`), or delete\n' +
        '             packages/luxar-viewer/public/wasm/ to skip those tests instead.'
      : '[test-setup] Compiled WASM not found and wasm-pack is not installed.\n' +
        '             WASM-vs-TypeScript parity tests will be SKIPPED — the compiled\n' +
        '             backend is NOT being verified. Install Rust + wasm-pack and run\n' +
        '             `make build-wasm` (or `pnpm build:wasm`) for full coverage.';
    if (require) {
      throw new Error(
        `${msg}\n(LUXAR_REQUIRE_WASM_TESTS=1 — refusing to run without a current WASM build.)`
      );
    }
    console.warn(`\n⚠️  ${msg}\n`);
    return;
  }

  console.log(
    stale.length
      ? `[test-setup] Compiled WASM is STALE (missing ${stale.join(', ')}) — rebuilding via \`pnpm build:wasm\`...`
      : '[test-setup] Compiled WASM missing — building via `pnpm build:wasm`...'
  );
  try {
    execSync('pnpm run build:wasm', { cwd: VIEWER_ROOT, stdio: 'inherit' });
  } catch (err) {
    const msg = `[test-setup] WASM build failed: ${err instanceof Error ? err.message : String(err)}`;
    if (require) throw new Error(msg, { cause: err });
    // Same asymmetry as the no-toolchain branch: the stale artifact is still on
    // disk, so the parity harness runs against it rather than skipping.
    const consequence = stale.length
      ? `The stale artifact is still in place, so parity tests will FAIL (missing ${stale.join(', ')}), not skip.`
      : 'Parity tests will be skipped.';
    console.warn(`\n⚠️  ${msg}\n   ${consequence}\n`);
    return;
  }
  // Re-check on the SAME terms as the entry condition. A build that ran but did
  // not produce the kernel is the stale case all over again, and reporting only
  // on absence here would let it through after appearing to fix itself.
  if (!existsSync(WASM_JS_PATH) || !existsSync(WASM_BIN_PATH)) {
    const msg = '[test-setup] WASM build ran but artifacts are still missing.';
    if (require) throw new Error(msg);
    console.warn(`\n⚠️  ${msg}\n`);
    return;
  }
  const stillMissing = missingWasmExports();
  if (stillMissing.length) {
    const msg =
      '[test-setup] WASM build ran but the artifacts still do not export ' +
      `${stillMissing.join(', ')}. Either the Rust source does not define ` +
      `${stillMissing.length > 1 ? 'these kernels' : 'this kernel'}, or the name in ` +
      'src/wasm/required-exports.ts is wrong.';
    if (require) throw new Error(msg);
    console.warn(`\n⚠️  ${msg}\n`);
  }
}

export async function setup(): Promise<void> {
  ensureWasmBuilt();

  // Incomplete counts as missing: a generator killed mid-write leaves a directory that
  // `existsSync` accepts but no decoder can read (see `isGeneratedFixtureComplete`).
  // Regenerating is the fix for both, so both take the same branch.
  const missing = EXPECTED_FIXTURES.filter(
    (name) => !isGeneratedFixtureComplete(resolve(FIXTURES_DIR, name))
  );
  const stale = areFixturesStale(PROJECT_ROOT, FIXTURES_DIR);

  if (missing.length > 0 || stale) {
    console.log(
      missing.length > 0
        ? `\n[test-setup] ${missing.length} zarr fixture(s) missing or incomplete — generating...`
        : "\n[test-setup] fixture producer's imports changed — regenerating..."
    );
  }

  ensureGeneratedFixtures(PROJECT_ROOT, FIXTURES_DIR, (scriptPath) => {
    const expectations = scriptPath.endsWith('generate_expectations.py');
    if (expectations) {
      console.log('[test-setup] Round-trip expectations missing/stale — generating...');
    }
    runPythonGenerator(
      expectations
        ? 'hatch run fixtures:python packages/luxar-viewer/tests/fixtures/generate_expectations.py'
        : 'hatch run fixtures:python packages/luxar-viewer/tests/fixtures/generate_test_data.py',
      expectations ? 'round-trip expectations' : 'fixtures'
    );
  });
}
