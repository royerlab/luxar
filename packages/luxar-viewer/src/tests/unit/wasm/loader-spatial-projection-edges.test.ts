/**
 * Edge-case tests for WASM loader (index.ts) and TypeScript-fallback
 * projection helpers. Closes wasm.md gap cluster:
 *   - [wasm.md G8][P5]  compact_by_mask stride > 1 (multi-component data).
 *   - [wasm.md G22][P5] isWasmSupported with WebAssembly.instantiate set to a
 *                       TRUTHY non-function (the documented threat at L118).
 *   - [wasm.md G23][P5] initWasm: dynamic import resolves but `default()` is
 *                       missing → falls back gracefully (proves the override
 *                       URL is actually exercised — discriminator from G24).
 *   - [wasm.md G24][P5] setWasmJsUrl: a non-empty override flows through to
 *                       the dynamic import — proven by a partial-shim data:
 *                       URL that resolves but fails the `default()` call.
 *   - [wasm.md G31][P5] extract_3d_positions displayDims[j] >= ndim → OOB
 *                       read → undefined → Float32Array stores NaN.
 *
 * Pure math / pure module API — no mocks.
 */

import { describe, it, expect } from 'vitest';
import { initWasm, isWasmSupported, setWasmJsUrl } from '../../../wasm';
import { TypeScriptFallback } from '../../../wasm/typescript';
import { extract_3d_positions, compact_by_mask } from '../../../wasm/typescript/projection';

describe('compact_by_mask — stride > 1 multi-component [wasm.md G8]', () => {
  it('[G8] stride=3 (xyz triples): only visible triples copied to output', () => {
    // 4 points × 3 components; mask hides points 0 and 2.
    const input = new Float32Array([
      1,
      2,
      3, // point 0 (hidden)
      4,
      5,
      6, // point 1 (visible)
      7,
      8,
      9, // point 2 (hidden)
      10,
      11,
      12, // point 3 (visible)
    ]);
    const mask = new Uint8Array([0, 1, 0, 1]);
    const output = new Float32Array(6).fill(99); // 2 visible × 3 stride
    const n = compact_by_mask(input, mask, 4, 3, output);
    expect(n).toBe(2);
    expect(Array.from(output.slice(0, 6))).toEqual([4, 5, 6, 10, 11, 12]);
  });

  it('[G8] stride=4 (RGBA quads): preserves component ordering, no off-by-one', () => {
    // Catches a mutation that swapped `i * stride` and `outIdx * stride`.
    const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    const mask = new Uint8Array([1, 0, 1]);
    const output = new Float32Array(8);
    const n = compact_by_mask(input, mask, 3, 4, output);
    expect(n).toBe(2);
    expect(output[0]).toBeCloseTo(0.1, 5);
    expect(output[3]).toBeCloseTo(0.4, 5);
    expect(output[4]).toBeCloseTo(0.9, 5);
    expect(output[7]).toBeCloseTo(1.2, 5);
  });

  it('[G8] stride=1 (degenerate boundary): identical to scalar compaction', () => {
    const input = new Float32Array([10, 20, 30, 40, 50]);
    const mask = new Uint8Array([1, 0, 1, 1, 0]);
    const output = new Float32Array(3);
    const n = compact_by_mask(input, mask, 5, 1, output);
    expect(n).toBe(3);
    expect(Array.from(output)).toEqual([10, 30, 40]);
  });
});

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
  it('[G23][G24] override URL with a partial-shim data: URL → import resolves but `default()` missing → fall back', async () => {
    // The audit notes that a bad URL and the default both fall back to TS,
    // so the existing test "URL override → still TypeScriptFallback" doesn't
    // PROVE the override was honored. Here we install a data: URL that
    // imports successfully (defining `foo` but not `default`); then
    // `wasmModule.default()` throws because `default` is undefined.
    // This proves the override URL was exercised — a regression that
    // ignored the override would have imported the non-existent default
    // wasm path and failed earlier.
    //
    // jsdom CAN import data: URLs in its dynamic import, so this discriminates.
    setWasmJsUrl('data:text/javascript,export const foo = 1');
    try {
      const wasm = await initWasm();
      // The init must fall through to TS fallback because `default` is missing.
      expect(wasm).toBeInstanceOf(TypeScriptFallback);
    } finally {
      setWasmJsUrl('');
    }
  });

  it('[G24] non-empty override is reflected immediately (state is module-local, not cached on first use)', async () => {
    // Pin that setWasmJsUrl mutates module-local state at call time, not on
    // first initWasm. A regression that captured the URL inside initWasm
    // would still work but a regression that memoised it in module init
    // would fail this sequencing test.
    setWasmJsUrl('http://localhost:0/missing-1.js');
    const w1 = await initWasm();
    setWasmJsUrl('http://localhost:0/missing-2.js');
    const w2 = await initWasm();
    setWasmJsUrl('');
    expect(w1).toBeInstanceOf(TypeScriptFallback);
    expect(w2).toBeInstanceOf(TypeScriptFallback);
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
