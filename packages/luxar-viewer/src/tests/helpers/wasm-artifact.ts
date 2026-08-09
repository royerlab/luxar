/**
 * Shared loader for the BUILT WASM artifact in `public/wasm/`.
 *
 * Tests, benchmarks and tools that want to exercise the compiled kernels can't
 * go through `initWasm()`: it resolves a browser URL and, when that fails,
 * silently hands back a `TypeScriptFallback` — which would make a WASM-vs-TS
 * comparison compare TS with TS. They instead read the `.wasm` bytes
 * and `initSync` the wasm-bindgen shim directly. This module is the ONE place
 * that does that; `src/tests/unit/wasm/direct-import-guard.test.ts` keeps it
 * that way, and every caller gets the staleness check for free instead of
 * hand-copying it.
 *
 * ## Why the staleness assertion is never inside the load `catch`
 *
 * `public/wasm/` is gitignored build output, so a checkout can hold a build that
 * imports and initialises fine while predating a newer kernel. A vitest run
 * mostly does not get that far — `global-setup.ts` scans the built shim for the
 * same required exports and REBUILDS a stale artifact before any test loads it —
 * so this check is the diagnosis for what setup cannot see: a MIXED build whose
 * shim declares every name while the `.wasm` behind it does not, and any caller
 * that runs outside that setup (benchmarks, tools).
 * {@link assertRequiredWasmExports} names those cases; it must be able to THROW.
 * Inside the catch that downgrades a load failure to a soft skip, its "missing
 * required export" message would be flattened into the generic "failed to load"
 * console line, the module would stay `null`, and every case below would then
 * fail on a null module — strictly worse than the opaque `x is not a function`
 * the check exists to replace. So {@link tryLoadWasmArtifact} catches ONLY the
 * import/`initSync` step and asserts after it, and {@link loadWasmArtifact}
 * catches nothing at all.
 *
 * Pure Node module — no DOM, no vitest globals.
 *
 * @module tests/helpers/wasm-artifact
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { assertRequiredWasmExports } from '../../wasm';
import type { WasmModule } from '../../wasm/types';

/** `packages/luxar-viewer` — this module lives at `src/tests/helpers/`. */
const PACKAGE_ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../..');

/** wasm-bindgen JS shim written by `pnpm build:wasm`. */
export const wasmJsPath = join(PACKAGE_ROOT, 'public/wasm/luxar_wasm.js');
/** Compiled WASM binary written by `pnpm build:wasm`. */
const wasmBinaryPath = join(PACKAGE_ROOT, 'public/wasm/luxar_wasm_bg.wasm');

/** Remediation line shared by every "no usable WASM build" diagnostic. */
export const WASM_BUILD_HINT = 'Build WASM with: pnpm build:wasm (or make build-wasm)';

/**
 * Whether both halves of the build output are present.
 *
 * Synchronous so callers can gate `describe.skipIf(...)` / `it.runIf(...)` at
 * module scope, and a function rather than a const so a caller that checks
 * later in the process sees the current state of the directory.
 */
export function wasmArtifactExists(): boolean {
  return existsSync(wasmJsPath) && existsSync(wasmBinaryPath);
}

/** The shim's namespace plus the instantiated `.wasm` exports behind it. */
interface LoadedArtifact {
  namespace: Record<string, unknown>;
  /** What `initSync` returned — see {@link assertRequiredWasmExports}. */
  instanceExports: unknown;
}

/**
 * Read the binary, import the shim and initialise it — the step that fails when
 * the artifact is absent or incompatible with this runtime. `initSync` (rather
 * than the async default export) works in Node without `fetch`, and passing the
 * bytes explicitly bypasses any bundler path resolution vitest/jsdom would
 * mangle. Its return value is kept because the two halves of the build can be
 * mismatched independently.
 */
async function importWasmArtifact(): Promise<LoadedArtifact> {
  const wasmBinary = readFileSync(wasmBinaryPath);
  const namespace = await import(wasmJsPath);
  const instanceExports = namespace.initSync({ module: wasmBinary });
  return { namespace: namespace as Record<string, unknown>, instanceExports };
}

/**
 * Load the artifact, failing HARD on anything: an absent/incompatible build
 * throws from the import, a stale one throws by name from the staleness check.
 *
 * For callers with no fallback path, where a silent skip would report success
 * having measured or compared nothing.
 */
export async function loadWasmArtifact(): Promise<WasmModule> {
  const { namespace, instanceExports } = await importWasmArtifact();
  assertRequiredWasmExports(namespace, instanceExports);
  return namespace as unknown as WasmModule;
}

/** Default note for {@link tryLoadWasmArtifact} when the artifact won't load. */
function reportLoadFailure(error: unknown): void {
  console.warn('Failed to load the built WASM artifact:', error);
  console.warn(WASM_BUILD_HINT);
}

/**
 * Load the artifact, downgrading a LOAD failure to `null` after logging a note
 * — but still throwing by name for a stale build (see the module comment).
 *
 * For callers that treat a missing compiled backend as "skip these cases".
 *
 * @param onLoadFailure Note emitted for a load failure; defaults to a warning
 *   carrying {@link WASM_BUILD_HINT}. Pass one to match a caller's existing
 *   console formatting.
 */
export async function tryLoadWasmArtifact(
  onLoadFailure: (error: unknown) => void = reportLoadFailure
): Promise<WasmModule | null> {
  let loaded: LoadedArtifact;
  try {
    loaded = await importWasmArtifact();
  } catch (error) {
    onLoadFailure(error);
    return null;
  }
  assertRequiredWasmExports(loaded.namespace, loaded.instanceExports);
  return loaded.namespace as unknown as WasmModule;
}
