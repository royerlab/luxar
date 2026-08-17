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

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'fs';
import { createHash } from 'crypto';
import { execSync } from 'child_process';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';
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
const EXPECTATIONS_GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_expectations.py');
const EXPECTATIONS_PATH = resolve(FIXTURES_DIR, 'roundtrip_expectations.json');
/**
 * Sidecars recording WHICH inputs produced the current generated artifacts.
 * Both live beside the (git-ignored) fixtures they describe.
 */
const FIXTURE_STAMP_PATH = resolve(FIXTURES_DIR, '.fixture-inputs.sha256');
const EXPECTATIONS_STAMP_PATH = resolve(FIXTURES_DIR, '.expectations-inputs.sha256');
/**
 * The Python packages the fixtures are generated THROUGH. A change in any of
 * these alters what the generator writes without touching the generator
 * script itself, so fixture staleness must be measured against them too:
 *   - `encoding/` — array encodings (e.g. #448's uint16 per-axis fixed-point
 *     for COORDINATE positions, the miss that motivated this check);
 *   - `io/` — the LuxarZarrCompiler machinery (`io/_compiler` chunking,
 *     spatial ordering, gsplat assembly/tree) every fixture byte flows through;
 *   - `typing_utils/` — the constants those two READ, so a one-line edit there
 *     silently changes the bytes (`TARGET_CHUNK_BYTES` sets every chunk shape,
 *     `DEFAULT_POINT_RADIUS` the pad on a no-radii chunk's stored bounds).
 * Deliberately NOT the whole `luxar/` package: fitting/CLI/demo code does not
 * affect compiled-fixture bytes, and over-widening would regenerate the
 * ~minute-long fixture set on every unrelated Python edit.
 */
const FIXTURE_INPUT_SOURCE_DIRS = [
  resolve(PROJECT_ROOT, 'packages/luxar/src/luxar/encoding'),
  resolve(PROJECT_ROOT, 'packages/luxar/src/luxar/io'),
  resolve(PROJECT_ROOT, 'packages/luxar/src/luxar/typing_utils'),
];

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
 * 600 s is ~2.8x the measured time, headroom for a slower or loaded machine
 * (CI runners are not faster than a laptop here). `LUXAR_FIXTURE_GEN_TIMEOUT_MS`
 * overrides it rather than requiring a source edit on a machine that needs more.
 */
const GENERATOR_TIMEOUT_MS = Number(process.env.LUXAR_FIXTURE_GEN_TIMEOUT_MS) || 600_000;

function runPythonGenerator(command: string, label: string): void {
  try {
    execSync(command, {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      timeout: GENERATOR_TIMEOUT_MS,
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
        `[test-setup] ...that was the ${GENERATOR_TIMEOUT_MS} ms budget, not a hang. ` +
          'Raise it with LUXAR_FIXTURE_GEN_TIMEOUT_MS if this machine is slower.'
      );
    }
    console.error(`[test-setup] Run manually: ${command}`);
    throw new Error(`${label} generation failed. See above for details.`, { cause: err });
  }
}

/**
 * Every Python file whose CONTENT can change what the generator writes: the
 * generator itself plus the non-test sources under FIXTURE_INPUT_SOURCE_DIRS.
 *
 * `**\/tests\/**` and `conftest.py` are excluded deliberately. A Python unit
 * test cannot change a fixture byte, but 48 of them live under `encoding/` and
 * `io/`, so including them meant that editing an unrelated Python test — or
 * rebasing, or switching worktrees — invalidated all 50 fixtures and paid a
 * ~minute-long regeneration before a single TypeScript test ran.
 */
function fixtureInputFiles(): string[] {
  const files = [GENERATOR_PATH];
  for (const dir of FIXTURE_INPUT_SOURCE_DIRS) {
    if (!existsSync(dir)) continue;
    for (const entry of readdirSync(dir, { recursive: true }) as string[]) {
      if (!entry.endsWith('.py')) continue;
      const parts = entry.split(/[\\/]/);
      if (parts.includes('tests') || parts[parts.length - 1] === 'conftest.py') continue;
      const full = join(dir, entry);
      if (existsSync(full)) files.push(full);
    }
  }
  return files.sort();
}

/** Content digest of `paths` — path names included so a rename counts as a change. */
function hashFiles(paths: string[]): string {
  const digest = createHash('sha256');
  for (const path of paths) {
    digest.update(path);
    digest.update(readFileSync(path));
  }
  return digest.digest('hex');
}

function readStamp(path: string): string | null {
  return existsSync(path) ? readFileSync(path, 'utf-8').trim() : null;
}

/**
 * Whether the generated fixtures were produced by the CURRENT inputs — the
 * generator script and the Python encoder/compiler sources it writes through
 * (see FIXTURE_INPUT_SOURCE_DIRS), compared by CONTENT.
 *
 * The generate-if-MISSING gate alone let #448 slip through: the fixtures all
 * existed (git-ignored, generated locally in June) but still carried the old
 * `float32` positions encoding, so five array-decoder tests failed while setup
 * regenerated nothing. So staleness must be checked — but NOT by mtime.
 *
 * mtime is wrong here in both directions. It fires spuriously, because a `git
 * checkout` or a fresh worktree rewrites source mtimes without changing a byte;
 * and content is what actually determines the output. `ensureWasmBuilt` below
 * already rejected mtime for the same reason and inspects the artifact instead.
 * This records a digest of the inputs next to the fixtures and compares against
 * it, so regeneration happens exactly when the inputs really changed.
 */
function areFixturesStale(): boolean {
  return readStamp(FIXTURE_STAMP_PATH) !== hashFiles(fixtureInputFiles());
}

/**
 * The expectations describe the fixtures, so they are stale when the fixture
 * inputs changed or their own generator did — no need to stat 50 zarr trees,
 * and no mtime churn.
 *
 * NOT sufficient on its own: see the `regeneratedFixtures` argument below.
 */
function expectationsFingerprint(): string {
  return hashFiles([...fixtureInputFiles(), EXPECTATIONS_GENERATOR_PATH]);
}

/**
 * @param regeneratedFixtures whether this run rebuilt the zarr fixtures.
 *
 * That flag is load-bearing, not belt-and-braces. `generate_test_data.py` is
 * NOT byte-reproducible: rebuilding the fixtures from unchanged inputs still
 * produces stores whose encoded values differ from the ones the committed
 * expectations were computed against, and 23 round-trip tests fail. The old
 * mtime comparison coupled the two implicitly (any fixture rebuild bumped the
 * fixtures past the expectations file); keying purely on the input digest
 * decoupled them and broke that invariant. So: fixtures rebuilt => expectations
 * rebuilt, always.
 */
function isExpectationsStale(regeneratedFixtures: boolean): boolean {
  if (regeneratedFixtures) return true;
  if (!existsSync(EXPECTATIONS_PATH)) return true;
  return readStamp(EXPECTATIONS_STAMP_PATH) !== expectationsFingerprint();
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
  const stale = areFixturesStale();
  const regeneratedFixtures = missing.length > 0 || stale;

  if (regeneratedFixtures) {
    console.log(
      missing.length > 0
        ? `\n[test-setup] ${missing.length} zarr fixture(s) missing or incomplete — generating...`
        : '\n[test-setup] fixture generator/encoder sources changed — regenerating...'
    );
    // All-or-nothing: generate_test_data.py takes no arguments, so there is no
    // per-fixture regeneration to reach for. That is affordable now only
    // because the content-digest check above stops this firing spuriously.
    runPythonGenerator(
      'hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py',
      'fixtures'
    );
  }

  // Verify fixture generation succeeded before generating expectations from them.
  //
  // This stays a HARD failure. Letting the run continue would report a green
  // suite whose round-trip coverage silently did not execute, which is the one
  // outcome worse than an obvious abort.
  const stillMissing = EXPECTED_FIXTURES.filter(
    (name) => !isGeneratedFixtureComplete(resolve(FIXTURES_DIR, name))
  );
  if (stillMissing.length > 0) {
    throw new Error(
      `Fixture generation ran but these are still missing or incomplete: ${stillMissing.join(', ')}`
    );
  }

  // Stamp only after the fixtures are verified complete, so a failed or
  // interrupted generation is retried on the next run rather than recorded as
  // current.
  writeFileSync(FIXTURE_STAMP_PATH, `${hashFiles(fixtureInputFiles())}\n`);

  if (isExpectationsStale(regeneratedFixtures)) {
    console.log('[test-setup] Round-trip expectations missing/stale — generating...');
    runPythonGenerator(
      'hatch run python packages/luxar-viewer/tests/fixtures/generate_expectations.py',
      'round-trip expectations'
    );
    writeFileSync(EXPECTATIONS_STAMP_PATH, `${expectationsFingerprint()}\n`);
  }
}
