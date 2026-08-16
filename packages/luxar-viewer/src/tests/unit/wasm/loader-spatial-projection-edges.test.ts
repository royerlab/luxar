/**
 * Edge-case tests for WASM loader (index.ts) and TypeScript-fallback
 * projection helpers. Closes wasm.md gap cluster:
 *   - [wasm.md G22][P5] isWasmSupported with WebAssembly.instantiate set to a
 *                       TRUTHY non-function (the documented threat at L118).
 *   - [wasm.md G23][P5] initWasm: a shim URL that is not a loadable WASM module
 *                       → falls back gracefully instead of propagating.
 *   - [wasm.md G24][P5] setWasmJsUrl: a non-empty override is read at call
 *                       time, not memoised at module init.
 *   - [wasm.md G31][P5] extract_3d_positions displayDims[j] >= ndim → OOB
 *                       read → undefined → Float32Array stores NaN.
 *
 * Pure math / pure module API. The only test doubles are `console` spies, used
 * to read back the URL the loader resolved from its own fallback warning.
 *
 * Note on G23/G24: these were written to discriminate WHERE the load failed
 * (import resolved but `default()` missing, vs. the URL never being honoured),
 * but no vitest environment can actually resolve `initWasm`'s dynamic import —
 * the `new Function('url', 'return import(url)')` indirection is not serviceable
 * by vitest's VM module runner (`ERR_VM_DYNAMIC_IMPORT_CALLBACK_MISSING`), so
 * even a `data:` URL throws before `default()` is reached. The URL the loader
 * resolved is therefore observable only through the message its `catch` logs,
 * which is what the G23/G24 tests below assert on: the override reaches the
 * resolution step, a second override is read at call time rather than memoised
 * at module init, and every one of these URLs lands in the documented
 * TypeScript fallback. Loading the compiled kernels for real goes through
 * `src/tests/helpers/wasm-artifact.ts`.
 */

import { describe, it, expect, vi } from 'vitest';
import { initWasm, isWasmSupported, setWasmJsUrl } from '../../../wasm';
import { TypeScriptFallback } from '../../../wasm/typescript';
import { extract_3d_positions } from '../../../wasm/typescript/projection';

describe('isWasmSupported — truthy non-function instantiate [wasm.md G22]', () => {
  it('[G22] WebAssembly.instantiate set to a TRUTHY non-function (e.g. {}) returns false', () => {
    // The function uses `typeof === "function"`, which correctly rejects
    // {}, [], 'string', 42, etc. Pin this so a regression to a simple
    // truthy check (`!!WebAssembly.instantiate`) would fail.
    const original = WebAssembly.instantiate;
    try {
      (WebAssembly as unknown as { instantiate: unknown }).instantiate = {};
      expect(isWasmSupported()).toBe(false);
    } finally {
      (WebAssembly as unknown as { instantiate: typeof original }).instantiate = original;
    }
  });

  it('[G22] WebAssembly.instantiate set to a string also returns false', () => {
    // String is truthy but `typeof "fake" === "string" !== "function"`.
    const original = WebAssembly.instantiate;
    try {
      (WebAssembly as unknown as { instantiate: unknown }).instantiate = 'fake';
      expect(isWasmSupported()).toBe(false);
    } finally {
      (WebAssembly as unknown as { instantiate: typeof original }).instantiate = original;
    }
  });
});

describe('initWasm + setWasmJsUrl — URL pass-through discriminator [wasm.md G23, G24]', () => {
  it('[G23][G24] override URL pointing at a partial shim (no `default`) still falls back', async () => {
    // Originally written as a discriminator: a data: URL that imports fine
    // (defining `foo` but not `default`) would fail at `wasmModule.default()`,
    // proving the override reached the import. Under vitest the dynamic import
    // itself is unserviceable (see the docblock), so the failure happens one
    // step earlier and `default()` is never reached. Two things are pinned
    // instead: the fallback warning names the `data:` URL, which is what proves
    // the override reached the resolution step at all (the pass-through this
    // describe block is named for); and an override naming something that is
    // not a loadable WASM shim lands in the TypeScript fallback rather than
    // propagating.
    const overrideUrl = 'data:text/javascript,export const foo = 1';
    setWasmJsUrl(overrideUrl);
    try {
      // `console.log` is spied purely to keep the two remediation `log.info`
      // lines out of the reporter. Restore both BEFORE any expect so a failing
      // assertion cannot leak a spy.
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const info = vi.spyOn(console, 'log').mockImplementation(() => {});
      const wasm = await initWasm();
      const calls = warn.mock.calls;
      warn.mockRestore();
      info.mockRestore();

      expect(wasm).toBeInstanceOf(TypeScriptFallback);
      expect(calls.length).toBeGreaterThan(0);
      const [message] = calls[0] as [string, unknown];
      expect(message).toContain(overrideUrl);
    } finally {
      setWasmJsUrl('');
    }
  });

  it('[G23] shim URL missing a required kernel falls back to a complete module', async () => {
    // Written to mimic a stale gitignored public/wasm build: a shim whose
    // default initializer succeeds but whose newer kernel export is absent,
    // which `assertRequiredWasmExports` must reject rather than hand back a
    // partial module that throws TypeError during a later projection. That
    // rejection is covered directly in `index.test.ts` — here the import never
    // resolves under vitest, so what is asserted is the end state: whatever the
    // reason, the caller receives a module with every kernel present.
    setWasmJsUrl('data:text/javascript,export default async function init() {}');
    try {
      const wasm = await initWasm();
      expect(wasm).toBeInstanceOf(TypeScriptFallback);
      expect(typeof wasm.compute_joint_codes).toBe('function');
    } finally {
      setWasmJsUrl('');
    }
  });

  it('[G24] non-empty override is read at call time, not memoised on first use', async () => {
    // Pin that each initWasm re-reads the override. The fallback warning names
    // the URL the loader resolved, so the two calls must name missing-1.js and
    // missing-2.js in that order — a regression that memoised the URL (at
    // module init, or on the first initWasm) would name missing-1.js twice and
    // fail here. `instanceof TypeScriptFallback` alone cannot discriminate:
    // every URL lands in the fallback under vitest (see the docblock).
    setWasmJsUrl('http://localhost:0/missing-1.js');
    try {
      // `console.log` is spied purely to keep the two remediation `log.info`
      // lines out of the reporter. Restore both BEFORE any expect so a failing
      // assertion cannot leak a spy.
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const info = vi.spyOn(console, 'log').mockImplementation(() => {});
      const w1 = await initWasm();
      setWasmJsUrl('http://localhost:0/missing-2.js');
      const w2 = await initWasm();
      const calls = warn.mock.calls;
      warn.mockRestore();
      info.mockRestore();

      expect(w1).toBeInstanceOf(TypeScriptFallback);
      expect(w2).toBeInstanceOf(TypeScriptFallback);
      expect(calls.length).toBeGreaterThanOrEqual(2);
      const [first] = calls[0] as [string, unknown];
      const [second] = calls[1] as [string, unknown];
      expect(first).toContain('missing-1.js');
      expect(second).toContain('missing-2.js');
    } finally {
      setWasmJsUrl('');
    }
  });
});

describe('extract_3d_positions — OOB displayDims [wasm.md G31]', () => {
  it('[G31] displayDims[j] >= ndim: OOB read → undefined → Float32 stores NaN', () => {
    // ndim=2 but displayDims includes index 5. The OOB read returns
    // undefined; coerced to Float32 = NaN. Pin contract; future hardening
    // (throw / clamp) surfaces as intentional change.
    const positionsNd = new Float32Array([10, 20]); // 1 point × 2 dims
    const displayDims = new Uint32Array([0, 5, 1]); // index 5 OOB
    const output = new Float32Array(3);
    extract_3d_positions(positionsNd, displayDims, 2, 1, output);
    expect(output[0]).toBe(10); // dim 0 = first slot
    expect(Number.isNaN(output[1])).toBe(true); // OOB
    expect(output[2]).toBe(20); // dim 1 = second slot
  });

  it('[G31] displayDims.length === 0: every output column is zero-filled', () => {
    // numDisplayDims = min(0, 3) = 0; first inner loop never runs;
    // pad-zeros loop fills all three columns.
    const positionsNd = new Float32Array([1, 2, 3, 4]); // 2 pts × 2 dims
    const displayDims = new Uint32Array(0);
    const output = new Float32Array(6).fill(99);
    extract_3d_positions(positionsNd, displayDims, 2, 2, output);
    expect(Array.from(output)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('[G31] numPoints === 0 boundary: empty loop, output untouched', () => {
    const positionsNd = new Float32Array([1, 2, 3]);
    const displayDims = new Uint32Array([0, 1, 2]);
    const output = new Float32Array(3).fill(99); // sentinel
    extract_3d_positions(positionsNd, displayDims, 3, 0, output);
    expect(Array.from(output)).toEqual([99, 99, 99]);
  });
});
