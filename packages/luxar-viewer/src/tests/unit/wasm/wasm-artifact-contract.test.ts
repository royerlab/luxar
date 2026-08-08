/**
 * Behavioural counterpart to `direct-import-guard.test.ts`: the split between
 * the two loaders in `src/tests/helpers/wasm-artifact.ts`. That helper's module
 * comment explains WHY the staleness assertion sits outside the load `catch`
 * (#1412); this file pins that it still does.
 *
 * Both failure modes are simulated rather than staged on disk: the
 * `../../../wasm` mock decides whether the build reads as stale, and the shim
 * mock decides whether it loads at all. What is NOT simulated is the helper's
 * `readFileSync` of `luxar_wasm_bg.wasm` — under this vitest setup a Node
 * builtin cannot be mocked for a module the test merely imports (`node:fs`
 * stays external, so neither `vi.mock('node:fs')` nor `vi.spyOn(fs, …)` reaches
 * the helper). These cases therefore need the built artifact to be readable and
 * SKIP rather than turn red without it: CI always has it (`typescript-tests`
 * builds WASM and runs with `LUXAR_REQUIRE_WASM_TESTS=1`), and locally
 * `global-setup.ts` builds it whenever wasm-pack is installed.
 */

import { describe, it, expect, afterEach, vi } from 'vitest';
import { wasmArtifactExists, wasmJsPath } from '../../helpers/wasm-artifact';

/** The helper's real byte read needs the artifact — see the module comment. */
const artifactPresent = wasmArtifactExists();

/** Message a stale build's `assertRequiredWasmExports` throws with. */
const STALE_MESSAGE = 'missing required export "compute_joint_codes"';
/** Message the simulated import/`initSync` step fails with. */
const LOAD_MESSAGE = 'simulated unloadable WASM artifact';

/**
 * Re-import the helper with both of its failure modes under our control.
 *
 * `stale` decides whether the staleness assertion throws. `loadable` decides
 * whether the import/`initSync` step succeeds: the mocked shim's `initSync`
 * throws, which stands in for the whole step (an incompatible build, or a shim
 * that won't import, both surface here). The real bytes are read and handed to
 * that mock, which ignores them.
 */
async function loadHelperWith({ stale, loadable }: { stale: boolean; loadable: boolean }) {
  vi.resetModules();
  vi.doMock('../../../wasm', () => ({
    assertRequiredWasmExports: () => {
      if (stale) throw new Error(STALE_MESSAGE);
    },
  }));
  vi.doMock(wasmJsPath, () => ({
    initSync: () => {
      if (!loadable) throw new Error(LOAD_MESSAGE);
    },
  }));
  return import('../../helpers/wasm-artifact');
}

afterEach(() => {
  vi.doUnmock('../../../wasm');
  vi.doUnmock(wasmJsPath);
  vi.resetModules();
});

describe.skipIf(!artifactPresent)('tryLoadWasmArtifact', () => {
  it('THROWS for a stale build, without reporting a load failure', async () => {
    // The assertion that pins "the staleness check is outside the load catch":
    // were it inside, this would resolve to null via onLoadFailure and every
    // caller would then fail on a null module.
    const { tryLoadWasmArtifact } = await loadHelperWith({ stale: true, loadable: true });
    const onLoadFailure = vi.fn();
    await expect(tryLoadWasmArtifact(onLoadFailure)).rejects.toThrow(STALE_MESSAGE);
    expect(onLoadFailure).not.toHaveBeenCalled();
  });

  it('returns null and reports when the artifact will not load', async () => {
    const { tryLoadWasmArtifact } = await loadHelperWith({ stale: false, loadable: false });
    const onLoadFailure = vi.fn();
    await expect(tryLoadWasmArtifact(onLoadFailure)).resolves.toBeNull();
    expect(onLoadFailure).toHaveBeenCalledOnce();
    expect((onLoadFailure.mock.calls[0][0] as Error).message).toBe(LOAD_MESSAGE);
  });

  it('resolves to the module when the build loads and is current', async () => {
    const { tryLoadWasmArtifact } = await loadHelperWith({ stale: false, loadable: true });
    const onLoadFailure = vi.fn();
    await expect(tryLoadWasmArtifact(onLoadFailure)).resolves.not.toBeNull();
    expect(onLoadFailure).not.toHaveBeenCalled();
  });
});

describe.skipIf(!artifactPresent)('loadWasmArtifact', () => {
  it('throws for a stale build (catches nothing)', async () => {
    const { loadWasmArtifact } = await loadHelperWith({ stale: true, loadable: true });
    await expect(loadWasmArtifact()).rejects.toThrow(STALE_MESSAGE);
  });

  it('throws when the artifact will not load (catches nothing)', async () => {
    const { loadWasmArtifact } = await loadHelperWith({ stale: false, loadable: false });
    await expect(loadWasmArtifact()).rejects.toThrow(LOAD_MESSAGE);
  });
});
