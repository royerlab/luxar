/**
 * Unit tests for core/app/init/module-overrides.ts (G3).
 *
 * `applyModuleOverrides` is a tiny but load-bearing helper: it forwards
 * `wasmPath` / `workerPath` overrides to the WASM and worker modules.
 *
 * Invariants exercised:
 *   - Calls the corresponding setter exactly once per supplied option.
 *   - Skips the setter when the option is undefined (so the module
 *     keeps its default `import.meta.url`-based resolution).
 *   - Empty string is treated as falsy and SKIPS the override — same
 *     as undefined. The setter docstring above explicitly opts into the
 *     truthy check via `if (options.wasmPath)`, so an empty string
 *     would otherwise smash the URL to "".
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../../../../../wasm', () => ({
  setWasmJsUrl: vi.fn(),
}));

vi.mock('../../../../../workers/worker-pool', () => ({
  setDataWorkerUrl: vi.fn(),
  setDataWorkerWasmPath: vi.fn(),
}));

import { applyModuleOverrides } from '../../../../../core/app/init/module-overrides';
import { setWasmJsUrl } from '../../../../../wasm';
import { setDataWorkerUrl, setDataWorkerWasmPath } from '../../../../../workers/worker-pool';

describe('applyModuleOverrides', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('forwards wasmPath to BOTH the main thread and the worker pool', () => {
    applyModuleOverrides({ wasmPath: '/custom/wasm.js' });
    // Main-thread consumers...
    expect(setWasmJsUrl).toHaveBeenCalledExactlyOnceWith('/custom/wasm.js');
    // ...and the worker pool (the override does not cross the worker boundary
    // on its own, so it must be forwarded explicitly).
    expect(setDataWorkerWasmPath).toHaveBeenCalledExactlyOnceWith('/custom/wasm.js');
    expect(setDataWorkerUrl).not.toHaveBeenCalled();
  });

  it('forwards workerPath when set', () => {
    applyModuleOverrides({ workerPath: '/custom/worker.js' });
    expect(setDataWorkerUrl).toHaveBeenCalledExactlyOnceWith('/custom/worker.js');
    expect(setWasmJsUrl).not.toHaveBeenCalled();
  });

  it('forwards both when both are set', () => {
    applyModuleOverrides({
      wasmPath: '/a.js',
      workerPath: '/b.js',
    });
    expect(setWasmJsUrl).toHaveBeenCalledExactlyOnceWith('/a.js');
    expect(setDataWorkerUrl).toHaveBeenCalledExactlyOnceWith('/b.js');
  });

  it('is a no-op when both options are undefined', () => {
    applyModuleOverrides({});
    expect(setWasmJsUrl).not.toHaveBeenCalled();
    expect(setDataWorkerUrl).not.toHaveBeenCalled();
  });

  it('skips the setter when the wasmPath is an empty string (treats falsy as unset)', () => {
    // Boundary: `if (options.wasmPath)` rejects ''. If the implementation
    // ever switched to `!== undefined`, an empty string would smash the
    // module URL to '' and break asset loading — this test catches that
    // regression.
    applyModuleOverrides({ wasmPath: '', workerPath: '' });
    expect(setWasmJsUrl).not.toHaveBeenCalled();
    expect(setDataWorkerUrl).not.toHaveBeenCalled();
  });

  it('skips wasmPath but forwards workerPath when only one is set', () => {
    applyModuleOverrides({ workerPath: '/only-worker.js' });
    expect(setWasmJsUrl).not.toHaveBeenCalled();
    expect(setDataWorkerUrl).toHaveBeenCalledExactlyOnceWith('/only-worker.js');
  });
});
