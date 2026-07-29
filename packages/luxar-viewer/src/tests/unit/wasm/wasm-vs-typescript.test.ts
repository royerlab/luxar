/**
 * WASM vs TypeScript Implementation Comparison Tests
 *
 * These tests verify that the compiled WASM module produces identical
 * results to the TypeScript reference implementations.
 *
 * Test Strategy:
 * 1. Load both WASM module and TypeScript fallback
 * 2. Run identical inputs through both implementations
 * 3. Compare outputs for exact match (within floating-point tolerance)
 *
 * Note: Tests are skipped if WASM module is not built unless
 * LUXAR_REQUIRE_WASM_TESTS=1 is set.
 * Build with: pnpm build:wasm (or make build-wasm)
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { TypeScriptFallback } from '../../../wasm/typescript';
import { ArrayDecoder } from '../../../data/array-decoder/decoder';
import type { WasmModule } from '../../../wasm/types';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import {
  arraysEqual as sharedArraysEqual,
  arraysAlmostEqual as sharedArraysAlmostEqual,
} from '../../helpers/array-compare';

// WASM module reference (loaded dynamically)
let wasmModule: WasmModule | null = null;
let tsModule: WasmModule;

// Pre-check if WASM files exist (synchronous check at module load time)
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../../public/wasm/luxar_wasm_bg.wasm');
const wasmFilesExist = existsSync(wasmJsPath) && existsSync(wasmBinaryPath);
const requireWasmTests = process.env.LUXAR_REQUIRE_WASM_TESTS === '1';

/**
 * Local wrappers around the shared helpers — preserve this file's
 * historical defaults (1e-5 tolerance + a console.log diagnostic on
 * mismatch) without forcing every call site to pass explicit args.
 *
 * The shared implementations are in src/tests/helpers/array-compare.ts
 * (wasm.md O2/O13 dedup).
 *
 * NOTE (wasm.md C3): default epsilon `1e-5` is appropriate for typical
 * Float32 single-step operations. For multi-step algorithms that accumulate
 * rounding error per dimension (e.g. `mahalanobis_distance` does ndim
 * forward-substitution steps), call sites should pass a scaled epsilon —
 * e.g. `arraysAlmostEqual(a, b, 1e-5 * Math.sqrt(ndim))` for ndim > 3 —
 * to avoid silently missing WASM-vs-TS divergence at high dimensions.
 */
function arraysAlmostEqual(a: ArrayLike<number>, b: ArrayLike<number>, epsilon = 1e-5): boolean {
  const ok = sharedArraysAlmostEqual(a, b, epsilon);
  if (!ok && a.length === b.length) {
    // Diagnostic: log the FIRST mismatching index for debugging.
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i] - b[i]) > epsilon) {
        console.log(`Mismatch at index ${i}: ${a[i]} vs ${b[i]} (diff: ${Math.abs(a[i] - b[i])})`);
        break;
      }
    }
  }
  return ok;
}

function arraysEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  // Exact equality (epsilon=0 default in the shared helper).
  return sharedArraysEqual(a, b);
}

beforeAll(async () => {
  // Initialize TypeScript fallback (always available)
  tsModule = new TypeScriptFallback();

  // Skip WASM loading if files don't exist
  if (!wasmFilesExist) {
    console.log('[Test] WASM module not found at:', wasmJsPath);
    console.log('[Test] Build WASM with: pnpm build:wasm (or make build-wasm)');
    return;
  }

  // Try to load WASM module
  try {
    // Read WASM binary synchronously
    const wasmBinary = readFileSync(wasmBinaryPath);

    // Dynamic import the JS module
    const wasm = await import(wasmJsPath);

    // Use initSync with the binary buffer (works in Node.js without fetch)
    wasm.initSync({ module: wasmBinary });

    wasmModule = wasm as unknown as WasmModule;
    console.log('[Test] WASM module loaded successfully');
  } catch (error) {
    console.log('[Test] WASM module failed to load:', error);
    console.log('[Test] Build WASM with: pnpm build:wasm (or make build-wasm)');
  }
});

describe('WASM artifact requirement', () => {
  it.runIf(requireWasmTests)('loads built WASM artifacts when required', () => {
    expect(wasmFilesExist).toBe(true);
    expect(wasmModule).not.toBeNull();
  });
});

describe('WASM vs TypeScript Comparison', () => {
  // ============================================================================
  // LINES CLIPPING MODULE
  // ============================================================================
  describe('regression: lines clip near-parallel epsilon (W1)', () => {
    it.skipIf(!wasmFilesExist)('dv in [1e-10, 1e-7): WASM and TS agree', () => {
      // 2 vertices, ndim=3, display dims [0,1]; hidden dim 2 has a tiny delta.
      const positions = new Float32Array([0.0, 0.0, 0.05, 1.0, 1.0, 0.05 + 1e-8]);
      const segments = new Uint32Array([0, 1]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0]);
      const tolerance = new Float32Array([10.0, 10.0, 0.1]);
      const displayDims = new Uint32Array([0, 1]);
      const tsVis = new Uint8Array(1);
      const tsT1 = new Float32Array(1);
      const tsT2 = new Float32Array(1);
      const wVis = new Uint8Array(1);
      const wT1 = new Float32Array(1);
      const wT2 = new Float32Array(1);
      const tsCount = tsModule.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        3,
        1,
        tsVis,
        tsT1,
        tsT2
      );
      const wCount = wasmModule!.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        3,
        1,
        wVis,
        wT1,
        wT2
      );
      expect(wCount).toBe(tsCount);
      expect(arraysEqual(wVis, tsVis)).toBe(true);
      expect(arraysAlmostEqual(wT1, tsT1)).toBe(true);
      expect(arraysAlmostEqual(wT2, tsT2)).toBe(true);
    });
  });

  // ============================================================================
  // EFFECTIVE RADII MODULE
  // ============================================================================
  describe('effective_radii: calculate_effective_radii', () => {
    it.skipIf(!wasmFilesExist)('should produce identical effective radii', () => {
      const positions = new Float32Array([0.0, 0.0, 0.0, 0.6]);
      const radii = new Float32Array([1.0]);
      const displayDims = new Uint32Array([0, 1, 2]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
      const spatialExtend = new Uint8Array([1, 1, 1, 1]);

      const tsOutput = new Float32Array(1);
      const wasmOutput = new Float32Array(1);

      const tsVisible = tsModule.calculate_effective_radii(
        positions,
        radii,
        displayDims,
        slicePos,
        spatialExtend,
        4,
        1,
        tsOutput
      );
      const wasmVisible = wasmModule!.calculate_effective_radii(
        positions,
        radii,
        displayDims,
        slicePos,
        spatialExtend,
        4,
        1,
        wasmOutput
      );

      expect(wasmVisible).toBe(tsVisible);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('should match with large point counts', () => {
      const numPoints = 1000;
      const ndim = 5;

      const positions = new Float32Array(numPoints * ndim);
      const radii = new Float32Array(numPoints).fill(1.0);

      // Generate deterministic test data
      for (let i = 0; i < numPoints; i++) {
        for (let d = 0; d < ndim; d++) {
          positions[i * ndim + d] = Math.sin(i * 0.1 + d) * 0.5;
        }
      }

      const displayDims = new Uint32Array([0, 1, 2]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
      const spatialExtend = new Uint8Array([1, 1, 1, 1, 1]);

      const tsOutput = new Float32Array(numPoints);
      const wasmOutput = new Float32Array(numPoints);

      const tsVisible = tsModule.calculate_effective_radii(
        positions,
        radii,
        displayDims,
        slicePos,
        spatialExtend,
        ndim,
        numPoints,
        tsOutput
      );
      const wasmVisible = wasmModule!.calculate_effective_radii(
        positions,
        radii,
        displayDims,
        slicePos,
        spatialExtend,
        ndim,
        numPoints,
        wasmOutput
      );

      expect(wasmVisible).toBe(tsVisible);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // DEPTH SORT MODULE
  // ============================================================================
  describe('depth_sort: sort_splats_by_depth', () => {
    /**
     * Parity for a sorting kernel is EXACT-permutation equality: the TS
     * twin frounds every float step in the Rust op order, so both
     * backends must compute identical uint16 keys — a one-bucket key
     * difference would reorder splats, which `arraysAlmostEqual` cannot
     * excuse.
     */
    function runBoth(
      centers3: Float32Array,
      modelView: Float32Array,
      count: number
    ): { ts: Uint32Array; wasm: Uint32Array; tsSorted: number; wasmSorted: number } {
      const ts = new Uint32Array(count);
      const wasm = new Uint32Array(count);
      const tsSorted = tsModule.sort_splats_by_depth(centers3, modelView, ts, count);
      const wasmSorted = wasmModule!.sort_splats_by_depth(centers3, modelView, wasm, count);
      return { ts, wasm, tsSorted, wasmSorted };
    }

    const IDENTITY_MV = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

    it.skipIf(!wasmFilesExist)('identical ordering on a small mixed scene', () => {
      // Includes behind-camera, duplicate-depth, and near/far splats.
      const zs = [-1, 3, -10, -5, -5, 0, -2.5];
      const centers3 = new Float32Array(zs.length * 3);
      zs.forEach((z, i) => {
        centers3[i * 3] = i;
        centers3[i * 3 + 2] = z;
      });
      const { ts, wasm, tsSorted, wasmSorted } = runBoth(centers3, IDENTITY_MV, zs.length);
      expect(wasmSorted).toBe(tsSorted);
      expect(arraysEqual(wasm, ts)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('identical ordering at unit extremes (nm and km scales)', () => {
      for (const scale of [1e-6, 1.0, 1e6]) {
        const zs = [-1, -9, -5, -3, -7, -2].map((z) => z * scale);
        const centers3 = new Float32Array(zs.length * 3);
        zs.forEach((z, i) => {
          centers3[i * 3 + 2] = z;
        });
        const { ts, wasm, tsSorted, wasmSorted } = runBoth(centers3, IDENTITY_MV, zs.length);
        expect(wasmSorted).toBe(tsSorted);
        expect(arraysEqual(wasm, ts)).toBe(true);
      }
    });

    it.skipIf(!wasmFilesExist)('identical identity fallback on degenerate depth', () => {
      const centers3 = new Float32Array([0, 0, -4, 1, 0, -4, 2, 0, -4]);
      const { ts, wasm, tsSorted, wasmSorted } = runBoth(centers3, IDENTITY_MV, 3);
      expect(wasmSorted).toBe(0);
      expect(tsSorted).toBe(0);
      expect(arraysEqual(wasm, ts)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('identical ordering with non-finite centers (NaN/±Inf)', () => {
      // NaN z reaches the key math where Rust's f32::min(NaN, 65535)
      // returns 65535 — a naive Math.min twin would key it to 0 and
      // diverge (caught by this exact case). ±Inf exercises the
      // Inf-range → inv_range 0 → NaN-key collapse.
      for (const zs of [
        [NaN, -3, -8],
        [-1, -Infinity, -5],
        [Infinity, -3, -8],
        [NaN, Infinity, -Infinity, -2, -7],
        // Denormal z values: the range is denormal, so inv_range
        // overflows to Inf and every key collapses through Inf/NaN
        // (also exercises f32 denormal handling in both backends).
        [-1e-40, -2e-40, -3e-40],
      ]) {
        const centers3 = new Float32Array(zs.length * 3);
        zs.forEach((z, i) => {
          centers3[i * 3] = i;
          centers3[i * 3 + 2] = z;
        });
        const { ts, wasm, tsSorted, wasmSorted } = runBoth(centers3, IDENTITY_MV, zs.length);
        expect(wasmSorted).toBe(tsSorted);
        expect(arraysEqual(wasm, ts), `zs=${zs}: wasm=[${wasm}] ts=[${ts}]`).toBe(true);
      }
    });

    it.skipIf(!wasmFilesExist)(
      'identical ordering on 100k pseudo-random splats under a general model-view',
      () => {
        const count = 100_000;
        const centers3 = new Float32Array(count * 3);
        // Deterministic xorshift32 positions in [-500, 500]^3.
        let state = 0xdeadbeef;
        const next = () => {
          state ^= (state << 13) >>> 0;
          state >>>= 0;
          state ^= state >>> 17;
          state ^= (state << 5) >>> 0;
          state >>>= 0;
          return (state / 0xffffffff) * 1000 - 500;
        };
        for (let i = 0; i < centers3.length; i++) {
          centers3[i] = next();
        }
        // A rotation+translation model-view (columns are orthonormal-ish):
        // exercises all four z-row coefficients, not just m14.
        const mv = new Float32Array([
          0.866, 0, -0.5, 0, 0, 1, 0, 0, 0.5, 0, 0.866, 0, 10, -20, -1500, 1,
        ]);
        const { ts, wasm, tsSorted, wasmSorted } = runBoth(centers3, mv, count);
        expect(wasmSorted).toBe(tsSorted);
        expect(arraysEqual(wasm, ts)).toBe(true);
      }
    );
  });

  // ============================================================================
  // DECODE MODULE
  // ============================================================================
  describe('decode: quantized functions', () => {
    it.skipIf(!wasmFilesExist)('decode_quantized_u8 should match', () => {
      const data = new Uint8Array([0, 64, 128, 192, 255]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_quantized_u8(data, -10.0, 10.0, tsOutput);
      wasmModule!.decode_quantized_u8(data, -10.0, 10.0, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_quantized_u16 should match', () => {
      const data = new Uint16Array([0, 16384, 32768, 49152, 65535]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_quantized_u16(data, -1.0, 1.0, tsOutput);
      wasmModule!.decode_quantized_u16(data, -1.0, 1.0, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_log_scalar_u8 should match', () => {
      const data = new Uint8Array([0, 50, 100, 150, 255]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_log_scalar_u8(data, 5.0, tsOutput);
      wasmModule!.decode_log_scalar_u8(data, 5.0, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_log_scalar_u16 should match', () => {
      const data = new Uint16Array([0, 16384, 32768, 49152, 65535]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_log_scalar_u16(data, 5.0, tsOutput);
      wasmModule!.decode_log_scalar_u16(data, 5.0, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_geolog_scalar_u8 should match', () => {
      // level 0 = reserved exact zero; min/max-anchored true-log grid
      const data = new Uint8Array([0, 1, 50, 128, 255]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_geolog_scalar_u8(data, -7.5, 9.9, tsOutput);
      wasmModule!.decode_geolog_scalar_u8(data, -7.5, 9.9, wasmOutput);

      expect(wasmOutput[0]).toBe(0);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_geolog_scalar_u16 should match', () => {
      const data = new Uint16Array([0, 1, 16384, 49152, 65535]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_geolog_scalar_u16(data, -7.5, 9.9, tsOutput);
      wasmModule!.decode_geolog_scalar_u16(data, -7.5, 9.9, wasmOutput);

      expect(wasmOutput[0]).toBe(0);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    /**
     * Per-channel family: WASM, the TS reference, AND the main-thread
     * `makePerChannelDequant` closure must agree BIT-EXACTLY — all three do
     * the same f64 math on the same f64 scales with an f32 store, so any
     * difference is a divergence bug, not rounding. This matters because the
     * range loader decodes sub-threshold ranges on the main thread and larger
     * ones in the worker; the two paths must be indistinguishable.
     */
    const perChannelTriple = (
      kind: 'linear' | 'log' | 'signed_log' | 'geolog',
      dtype: 'u8' | 'u16',
      zeroLevel: boolean,
      colOffset: number
    ) => {
      const colLo = new Float64Array([-2.25, 0.5, -7.125]);
      const colHi = new Float64Array([3.5, 0.5, 9.875]); // column 1 constant (lo == hi)
      const cols = 3;
      const top = dtype === 'u8' ? 255 : 65535;
      const n = 61; // not a multiple of cols: exercises the rolling column counter
      const codes = Array.from({ length: n }, (_, i) => (i * 9973) % (top + 1));
      codes[0] = 0; // reserved-zero (or legacy lo) code
      codes[1] = top; // top code
      const tsOutput = new Float32Array(n);
      const wasmOutput = new Float32Array(n);

      if (dtype === 'u8') {
        const data = new Uint8Array(codes);
        if (kind === 'linear') {
          tsModule.decode_linear_perchannel_u8(data, colLo, colHi, colOffset, tsOutput);
          wasmModule!.decode_linear_perchannel_u8(data, colLo, colHi, colOffset, wasmOutput);
        } else if (kind === 'geolog') {
          tsModule.decode_geolog_perchannel_u8(data, colLo, colHi, colOffset, tsOutput);
          wasmModule!.decode_geolog_perchannel_u8(data, colLo, colHi, colOffset, wasmOutput);
        } else if (kind === 'log') {
          tsModule.decode_log_perchannel_u8(data, colLo, colHi, zeroLevel, colOffset, tsOutput);
          wasmModule!.decode_log_perchannel_u8(
            data,
            colLo,
            colHi,
            zeroLevel,
            colOffset,
            wasmOutput
          );
        } else {
          tsModule.decode_signed_log_perchannel_u8(
            data,
            colLo,
            colHi,
            zeroLevel,
            colOffset,
            tsOutput
          );
          wasmModule!.decode_signed_log_perchannel_u8(
            data,
            colLo,
            colHi,
            zeroLevel,
            colOffset,
            wasmOutput
          );
        }
      } else {
        const data = new Uint16Array(codes);
        if (kind === 'linear') {
          tsModule.decode_linear_perchannel_u16(data, colLo, colHi, colOffset, tsOutput);
          wasmModule!.decode_linear_perchannel_u16(data, colLo, colHi, colOffset, wasmOutput);
        } else if (kind === 'geolog') {
          tsModule.decode_geolog_perchannel_u16(data, colLo, colHi, colOffset, tsOutput);
          wasmModule!.decode_geolog_perchannel_u16(data, colLo, colHi, colOffset, wasmOutput);
        } else if (kind === 'log') {
          tsModule.decode_log_perchannel_u16(data, colLo, colHi, zeroLevel, colOffset, tsOutput);
          wasmModule!.decode_log_perchannel_u16(
            data,
            colLo,
            colHi,
            zeroLevel,
            colOffset,
            wasmOutput
          );
        } else {
          tsModule.decode_signed_log_perchannel_u16(
            data,
            colLo,
            colHi,
            zeroLevel,
            colOffset,
            tsOutput
          );
          wasmModule!.decode_signed_log_perchannel_u16(
            data,
            colLo,
            colHi,
            zeroLevel,
            colOffset,
            wasmOutput
          );
        }
      }

      // Third backend: the main-thread dequant closure (the sub-threshold and
      // worker-failure path in range-loader/perchannel.ts).
      const dequant = ArrayDecoder.makePerChannelDequant(
        {
          name: `${kind}_perchannel_${dtype}`,
          bits: dtype === 'u8' ? 8 : 16,
          col_lo: Array.from(colLo),
          col_hi: Array.from(colHi),
          zero_level: zeroLevel,
        },
        cols
      );
      const mainOutput = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        mainOutput[j] = dequant(codes[j], (colOffset + j) % cols);
      }

      expect(arraysEqual(wasmOutput, tsOutput)).toBe(true); // bit-exact
      expect(arraysEqual(wasmOutput, mainOutput)).toBe(true); // bit-exact
      if (zeroLevel && kind !== 'linear') expect(wasmOutput[0]).toBe(0);
    };

    it.skipIf(!wasmFilesExist)('decode_linear_perchannel_u16 matches (3 backends)', () => {
      perChannelTriple('linear', 'u16', false, 0);
    });

    it.skipIf(!wasmFilesExist)('decode_linear_perchannel_u8 matches with column phase', () => {
      perChannelTriple('linear', 'u8', false, 2);
    });

    it.skipIf(!wasmFilesExist)('decode_log_perchannel_u8 zero_level matches (3 backends)', () => {
      perChannelTriple('log', 'u8', true, 0);
    });

    it.skipIf(!wasmFilesExist)('decode_log_perchannel_u16 legacy matches (3 backends)', () => {
      perChannelTriple('log', 'u16', false, 0);
    });

    it.skipIf(!wasmFilesExist)(
      'decode_signed_log_perchannel_u16 zero_level matches with column phase',
      () => {
        perChannelTriple('signed_log', 'u16', true, 1);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'decode_signed_log_perchannel_u8 legacy matches (3 backends)',
      () => {
        perChannelTriple('signed_log', 'u8', false, 0);
      }
    );

    it.skipIf(!wasmFilesExist)('decode_geolog_perchannel_u16 matches (3 backends)', () => {
      // TRUE-log HDR-color encoding: zero level is always on (name contract).
      perChannelTriple('geolog', 'u16', true, 0);
    });

    it.skipIf(!wasmFilesExist)('decode_geolog_perchannel_u8 matches with column phase', () => {
      perChannelTriple('geolog', 'u8', true, 1);
    });

    it.skipIf(!wasmFilesExist)('decode_lut_scalar_u16 should match', () => {
      // Use larger indices to test u16 range
      const indices = new Uint16Array([0, 1000, 2000, 3000, 500]);
      // Create a LUT with 4096 entries
      const lut = new Float32Array(4096);
      for (let i = 0; i < 4096; i++) {
        lut[i] = Math.sin(i * 0.01) * 100;
      }

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_lut_scalar_u16(indices, lut, tsOutput);
      wasmModule!.decode_lut_scalar_u16(indices, lut, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_lut_row_u16 should match', () => {
      // Use larger indices to test u16 range
      const indices = new Uint16Array([0, 500, 1000, 1500]);
      // Create a LUT with 2000 RGB rows
      const rowSize = 3;
      const lut = new Float32Array(2000 * rowSize);
      for (let i = 0; i < 2000; i++) {
        lut[i * rowSize + 0] = Math.sin(i * 0.01);
        lut[i * rowSize + 1] = Math.cos(i * 0.01);
        lut[i * rowSize + 2] = Math.sin(i * 0.02);
      }

      const tsOutput = new Float32Array(indices.length * rowSize);
      const wasmOutput = new Float32Array(indices.length * rowSize);

      tsModule.decode_lut_row_u16(indices, lut, rowSize, tsOutput);
      wasmModule!.decode_lut_row_u16(indices, lut, rowSize, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_lut_scalar_u8 should match', () => {
      const indices = new Uint8Array([0, 2, 1, 3, 0]);
      const lut = new Float32Array([1.5, 2.5, 3.5, 4.5]);

      const tsOutput = new Float32Array(5);
      const wasmOutput = new Float32Array(5);

      tsModule.decode_lut_scalar_u8(indices, lut, tsOutput);
      wasmModule!.decode_lut_scalar_u8(indices, lut, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_lut_row_u8 should match', () => {
      const indices = new Uint8Array([0, 1, 2]);
      const lut = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); // RGB identity-ish

      const tsOutput = new Float32Array(9);
      const wasmOutput = new Float32Array(9);

      tsModule.decode_lut_row_u8(indices, lut, 3, tsOutput);
      wasmModule!.decode_lut_row_u8(indices, lut, 3, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('decode_broadcasted should match', () => {
      const value = new Float32Array([0.5, 0.6, 0.7]);

      const tsOutput = new Float32Array(15);
      const wasmOutput = new Float32Array(15);

      tsModule.decode_broadcasted(value, 5, 3, tsOutput);
      wasmModule!.decode_broadcasted(value, 5, 3, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // PROJECTION MODULE
  // ============================================================================
  describe('projection functions', () => {
    it.skipIf(!wasmFilesExist)('extract_3d_positions should match', () => {
      const positionsNd = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
      const displayDims = new Uint32Array([0, 2, 4]);

      const tsOutput = new Float32Array(9);
      const wasmOutput = new Float32Array(9);

      tsModule.extract_3d_positions(positionsNd, displayDims, 5, 3, tsOutput);
      wasmModule!.extract_3d_positions(positionsNd, displayDims, 5, 3, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('calculate_bounds_3d should match', () => {
      const positions = new Float32Array([-1, 2, 3, 4, -5, 6, 7, 8, -9]);

      const tsOutput = new Float32Array(6);
      const wasmOutput = new Float32Array(6);

      const tsCount = tsModule.calculate_bounds_3d(positions, 3, tsOutput);
      const wasmCount = wasmModule!.calculate_bounds_3d(positions, 3, wasmOutput);

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('compact_by_mask should match', () => {
      const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const mask = new Uint8Array([1, 0, 1]);

      const tsOutput = new Float32Array(6);
      const wasmOutput = new Float32Array(6);

      const tsCount = tsModule.compact_by_mask(input, mask, 3, 3, tsOutput);
      const wasmCount = wasmModule!.compact_by_mask(input, mask, 3, 3, wasmOutput);

      expect(wasmCount).toBe(tsCount);
      expect(
        arraysAlmostEqual(wasmOutput.slice(0, tsCount * 3), tsOutput.slice(0, tsCount * 3))
      ).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('count_visible should match', () => {
      const mask = new Uint8Array([1, 0, 1, 0, 1, 1, 0]);

      const tsCount = tsModule.count_visible(mask, 7);
      const wasmCount = wasmModule!.count_visible(mask, 7);

      expect(wasmCount).toBe(tsCount);
    });

    it.skipIf(!wasmFilesExist)('radii_to_visibility_mask should match', () => {
      const radii = new Float32Array([0.5, 0.001, 0.2, 0.0, 1.5]);

      const tsOutput = new Uint8Array(5);
      const wasmOutput = new Uint8Array(5);

      const tsCount = tsModule.radii_to_visibility_mask(radii, 0.01, 5, tsOutput);
      const wasmCount = wasmModule!.radii_to_visibility_mask(radii, 0.01, 5, wasmOutput);

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // GSPLATS PROCESSING MODULE
  // ============================================================================
  describe('gsplats_processing functions', () => {
    it.skipIf(!wasmFilesExist)('mahalanobis_distance should match', () => {
      const diff = new Float32Array([3.0, 4.0, 0.0]);
      const packedL = new Float32Array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]); // Identity

      const tsDist = tsModule.mahalanobis_distance(diff, packedL, 3);
      const wasmDist = wasmModule!.mahalanobis_distance(diff, packedL, 3);

      expect(Math.abs(wasmDist - tsDist)).toBeLessThan(1e-5);
    });

    // Audit C2/C3 fix: pin the high-ndim epsilon-scaling contract.
    // `mahalanobis_distance` does ndim forward-substitution steps, so
    // Float32 rounding accumulates per dimension. The tolerance must be
    // scaled by sqrt(ndim) — verified by running both implementations
    // against an identity covariance and a diff vector with magnitude 1.
    // A regression that returns an unscaled result, or a WASM build with
    // dimension-dependent precision loss, will surface here.
    it.skipIf(!wasmFilesExist).each([3, 8, 16] as const)(
      'mahalanobis_distance matches at ndim=%i within sqrt(ndim)*1e-5',
      (ndim) => {
        const diff = new Float32Array(ndim);
        for (let i = 0; i < ndim; i++) diff[i] = (i + 1) * 0.1;
        const packedSize = (ndim * (ndim + 1)) / 2;
        const packedL = new Float32Array(packedSize);
        // Build a lower-triangular identity in packed-row-major order.
        let p = 0;
        for (let row = 0; row < ndim; row++) {
          for (let col = 0; col <= row; col++) {
            packedL[p++] = col === row ? 1.0 : 0.0;
          }
        }

        const tsDist = tsModule.mahalanobis_distance(diff, packedL, ndim);
        const wasmDist = wasmModule!.mahalanobis_distance(diff, packedL, ndim);

        const scaledEps = 1e-5 * Math.sqrt(ndim);
        expect(Math.abs(wasmDist - tsDist)).toBeLessThan(scaledEps);
      }
    );

    it.skipIf(!wasmFilesExist)('extract_cholesky_submatrix should match', () => {
      const packed = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      const keepDims = new Uint32Array([0, 2]);

      const tsOutput = new Float32Array(3);
      const wasmOutput = new Float32Array(3);

      tsModule.extract_cholesky_submatrix(packed, keepDims, 2, tsOutput);
      wasmModule!.extract_cholesky_submatrix(packed, keepDims, 2, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('compute_gsplats_attenuation should match', () => {
      const positions = new Float32Array([0, 0, 0, 0, 0, 0, 0, 3]);
      const cholesky = new Float32Array([
        1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
      ]);
      const amplitudes = new Float32Array([1.0, 1.0]);
      const slicePos = new Float32Array([0, 0, 0, 0]);
      const hiddenDims = new Uint32Array([3]);

      const tsVisibility = new Uint8Array(2);
      const tsAttenuation = new Float32Array(2);
      const wasmVisibility = new Uint8Array(2);
      const wasmAttenuation = new Float32Array(2);

      const tsCount = tsModule.compute_gsplats_attenuation(
        positions,
        cholesky,
        amplitudes,
        slicePos,
        hiddenDims,
        4,
        2,
        0.01,
        3.0,
        tsVisibility,
        tsAttenuation
      );
      const wasmCount = wasmModule!.compute_gsplats_attenuation(
        positions,
        cholesky,
        amplitudes,
        slicePos,
        hiddenDims,
        4,
        2,
        0.01,
        3.0,
        wasmVisibility,
        wasmAttenuation
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmVisibility, tsVisibility)).toBe(true);
      expect(arraysAlmostEqual(wasmAttenuation, tsAttenuation)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('extract_visible_cholesky_3d should match', () => {
      const cholesky = new Float32Array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
      ]);
      const visibility = new Uint8Array([1, 1]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsOutput = new Float32Array(12);
      const wasmOutput = new Float32Array(12);

      const tsCount = tsModule.extract_visible_cholesky_3d(
        cholesky,
        visibility,
        displayDims,
        4,
        2,
        tsOutput
      );
      const wasmCount = wasmModule!.extract_visible_cholesky_3d(
        cholesky,
        visibility,
        displayDims,
        4,
        2,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('compact_attenuated_amplitudes should match', () => {
      const amplitudes = new Float32Array([1.0, 2.0, 3.0]);
      const attenuation = new Float32Array([0.5, 0.25, 0.75]);
      const visibility = new Uint8Array([1, 0, 1]);

      const tsOutput = new Float32Array(2);
      const wasmOutput = new Float32Array(2);

      const tsCount = tsModule.compact_attenuated_amplitudes(
        amplitudes,
        attenuation,
        visibility,
        3,
        tsOutput
      );
      const wasmCount = wasmModule!.compact_attenuated_amplitudes(
        amplitudes,
        attenuation,
        visibility,
        3,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    // Fused single-call projection (W5). The fused kernel writes compacted
    // outputs, so we compare the dense prefix [0, count*stride) only. epsilon is
    // scaled by sqrt(ndim) per the file's high-ndim accumulation convention.
    const runFused = (
      mod: WasmModule,
      args: {
        positions: Float32Array;
        cholesky: Float32Array;
        amplitudes: Float32Array;
        colors: Float32Array;
        discreteVisibility: Uint8Array;
        slicePosition: Float32Array;
        continuousHiddenDims: Uint32Array;
        displayDims: Uint32Array;
        ndim: number;
        splatCount: number;
        colorComponents?: number;
      }
    ): {
      count: number;
      centers: Float32Array;
      chol: Float32Array;
      amps: Float32Array;
      cols: Float32Array;
    } => {
      const n = args.splatCount;
      const k = args.colorComponents ?? 3;
      const centers = new Float32Array(n * 3);
      const chol = new Float32Array(n * 6);
      const amps = new Float32Array(n);
      const cols = new Float32Array(n * k);
      const count = mod.project_gsplats_nd_to_3d(
        args.positions,
        args.cholesky,
        args.amplitudes,
        args.colors,
        args.discreteVisibility,
        args.slicePosition,
        args.continuousHiddenDims,
        args.displayDims,
        args.ndim,
        n,
        k,
        1e-6,
        3.0,
        centers,
        chol,
        amps,
        cols
      );
      return { count, centers, chol, amps, cols };
    };

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d matches TS (correlated covariance)',
      () => {
        const ndim = 4;
        const splatCount = 3;
        const one = [2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0]; // correlated 4D packed
        const args = {
          positions: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0.3, 2, 2, 2, 50]),
          cholesky: new Float32Array([...one, ...one, ...one]),
          amplitudes: new Float32Array([1.0, 0.8, 0.5]),
          colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
          discreteVisibility: new Uint8Array([1, 1, 1]),
          slicePosition: new Float32Array([0, 0, 0, 0]),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);
        const eps = 1e-5 * Math.sqrt(ndim);
        expect(w.count).toBe(ts.count);
        expect(
          arraysAlmostEqual(
            w.centers.subarray(0, w.count * 3),
            ts.centers.subarray(0, ts.count * 3),
            eps
          )
        ).toBe(true);
        expect(
          arraysAlmostEqual(w.chol.subarray(0, w.count * 6), ts.chol.subarray(0, ts.count * 6), eps)
        ).toBe(true);
        expect(
          arraysAlmostEqual(w.amps.subarray(0, w.count), ts.amps.subarray(0, ts.count), eps)
        ).toBe(true);
        expect(
          arraysEqual(w.cols.subarray(0, w.count * 3), ts.cols.subarray(0, ts.count * 3))
        ).toBe(true);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d matches TS (RGBA colors — 4-channel stride)',
      () => {
        // The RGB cases above never exercise the 4th (alpha) channel of the
        // color compaction — the one path where the Rust copy_from_slice and
        // the TS c-loop could silently drift. Alpha is set DISTINCT from RGB
        // (alpha = 1 - r) so a stride/drop bug misaligns the compacted output.
        const ndim = 4;
        const splatCount = 3;
        const one = [2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0];
        const args = {
          positions: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0.3, 2, 2, 2, 50]),
          cholesky: new Float32Array([...one, ...one, ...one]),
          amplitudes: new Float32Array([1.0, 0.8, 0.5]),
          // (N, 4) RGBA — alpha column last, distinct from RGB.
          colors: new Float32Array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 0.3]),
          discreteVisibility: new Uint8Array([1, 1, 1]),
          slicePosition: new Float32Array([0, 0, 0, 0]),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          colorComponents: 4,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);
        expect(w.count).toBe(ts.count);
        // Bit-exact 4-channel color agreement, alpha included.
        expect(
          arraysEqual(w.cols.subarray(0, w.count * 4), ts.cols.subarray(0, ts.count * 4))
        ).toBe(true);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d matches TS (axis-permuted displayDims + discrete gate)',
      () => {
        const ndim = 4;
        const splatCount = 3;
        const one = [2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0];
        const args = {
          positions: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0, 2, 2, 2, 0]),
          cholesky: new Float32Array([...one, ...one, ...one]),
          amplitudes: new Float32Array([1.0, 1.0, 1.0]),
          colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
          discreteVisibility: new Uint8Array([1, 0, 1]), // splat 1 gated out
          slicePosition: new Float32Array([0, 0, 0, 0]),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([2, 0, 1]), // permuted X=dim2,Y=dim0,Z=dim1
          ndim,
          splatCount,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);
        const eps = 1e-5 * Math.sqrt(ndim);
        expect(w.count).toBe(ts.count);
        expect(w.count).toBe(2); // splat 1 gated
        expect(
          arraysAlmostEqual(
            w.centers.subarray(0, w.count * 3),
            ts.centers.subarray(0, ts.count * 3),
            eps
          )
        ).toBe(true);
        expect(
          arraysAlmostEqual(w.chol.subarray(0, w.count * 6), ts.chol.subarray(0, ts.count * 6), eps)
        ).toBe(true);
        expect(
          arraysEqual(w.cols.subarray(0, w.count * 3), ts.cols.subarray(0, ts.count * 3))
        ).toBe(true);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d matches TS (2D data — regression: display marginal read displayDims[2] OOB)',
      () => {
        // A 2D scene has displayDims.length === 2. The display-dims marginal
        // used to be computed with a hardcoded sub-ndim of 3: Rust read
        // display_dims[2] out of bounds and panicked (wasm `unreachable`,
        // killing every 2D gsplats load); TS read undefined and produced
        // NaN-driven garbage. Both must now emit the 2D marginal in the first
        // three packed slots and a scale-matched phantom z row.
        const ndim = 2;
        const splatCount = 2;
        const one = [2.0, 0.5, 1.5]; // correlated 2D packed [L00, L10, L11]
        const args = {
          positions: new Float32Array([0, 0, 5, -3]),
          cholesky: new Float32Array([...one, ...one]),
          amplitudes: new Float32Array([1.0, 0.8]),
          colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
          discreteVisibility: new Uint8Array([1, 1]),
          slicePosition: new Float32Array([0, 0]),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1]), // 2D scene: only two display dims
          ndim,
          splatCount,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);
        const eps = 1e-5 * Math.sqrt(ndim);
        expect(w.count).toBe(2);
        expect(ts.count).toBe(2);
        // Centers zero-padded in z.
        expect(Array.from(w.centers.subarray(0, 6))).toEqual([0, 0, 0, 5, -3, 0]);
        // Keeping ALL dims makes the marginal reproduce the input factor; the
        // phantom z diagonal is the geometric mean of the real diagonals, NOT
        // an epsilon (an ε-thin splat is invisible in sum mode — the shader
        // scales amplitude by the Gaussian extent along the view ray).
        const expectedPhantom = Math.sqrt(2.0 * 1.5);
        for (let s = 0; s < 2; s++) {
          const c = s * 6;
          expect(w.chol[c]).toBeCloseTo(2.0, 5);
          expect(w.chol[c + 1]).toBeCloseTo(0.5, 5);
          expect(w.chol[c + 2]).toBeCloseTo(1.5, 5);
          expect(w.chol[c + 3]).toBe(0); // L20
          expect(w.chol[c + 4]).toBe(0); // L21
          expect(w.chol[c + 5]).toBeCloseTo(expectedPhantom, 5); // L22 = √(L00·L11)
        }
        expect(arraysAlmostEqual(w.centers.subarray(0, 6), ts.centers.subarray(0, 6), eps)).toBe(
          true
        );
        expect(arraysAlmostEqual(w.chol.subarray(0, 12), ts.chol.subarray(0, 12), eps)).toBe(true);
        expect(arraysAlmostEqual(w.amps.subarray(0, 2), ts.amps.subarray(0, 2), eps)).toBe(true);
        expect(arraysEqual(w.cols.subarray(0, 6), ts.cols.subarray(0, 6))).toBe(true);
      }
    );

    it.skipIf(!wasmFilesExist)('extract_visible_cholesky_3d matches TS (2D data)', () => {
      const cholesky = new Float32Array([2.0, 0.5, 1.5]);
      const visibility = new Uint8Array([1]);
      const displayDims = new Uint32Array([0, 1]);
      const tsOutput = new Float32Array(6);
      const wasmOutput = new Float32Array(6);

      const tsCount = tsModule.extract_visible_cholesky_3d(
        cholesky,
        visibility,
        displayDims,
        2,
        1,
        tsOutput
      );
      const wasmCount = wasmModule!.extract_visible_cholesky_3d(
        cholesky,
        visibility,
        displayDims,
        2,
        1,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(wasmOutput[3]).toBe(0);
      expect(wasmOutput[4]).toBe(0);
      expect(wasmOutput[5]).toBeCloseTo(Math.sqrt(2.0 * 1.5), 5);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // LINES CLIPPING MODULE
  // ============================================================================
  describe('lines_clipping functions', () => {
    it.skipIf(!wasmFilesExist)('clip_segment_single should match', () => {
      const p1 = new Float32Array([0, 0, 0, 0]);
      const p2 = new Float32Array([10, 10, 10, 10]);
      const slicePos = new Float32Array([0, 0, 0, 5]);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsResult = tsModule.clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
      const wasmResult = wasmModule!.clip_segment_single(
        p1,
        p2,
        slicePos,
        tolerance,
        displayDims,
        4
      );

      expect(arraysAlmostEqual(wasmResult, tsResult)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('clip_segments_batch should match', () => {
      const positions = new Float32Array([0, 0, 0, 5, 10, 10, 10, 5, 20, 20, 20, 0]);
      const segments = new Uint32Array([0, 1, 1, 2]);
      const slicePos = new Float32Array([0, 0, 0, 5]);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsVisibility = new Uint8Array(2);
      const tsT1 = new Float32Array(2);
      const tsT2 = new Float32Array(2);
      const wasmVisibility = new Uint8Array(2);
      const wasmT1 = new Float32Array(2);
      const wasmT2 = new Float32Array(2);

      const tsCount = tsModule.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        4,
        2,
        tsVisibility,
        tsT1,
        tsT2
      );
      const wasmCount = wasmModule!.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        4,
        2,
        wasmVisibility,
        wasmT1,
        wasmT2
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmVisibility, tsVisibility)).toBe(true);
      expect(arraysAlmostEqual(wasmT1, tsT1)).toBe(true);
      expect(arraysAlmostEqual(wasmT2, tsT2)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('interpolate_clipped_positions should match', () => {
      const positions = new Float32Array([0, 0, 0, 0, 10, 20, 30, 10]);
      const segments = new Uint32Array([0, 1]);
      const visibility = new Uint8Array([1]);
      const t1Params = new Float32Array([0.25]);
      const t2Params = new Float32Array([0.75]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsStart = new Float32Array(3);
      const tsEnd = new Float32Array(3);
      const wasmStart = new Float32Array(3);
      const wasmEnd = new Float32Array(3);

      const tsCount = tsModule.interpolate_clipped_positions(
        positions,
        segments,
        visibility,
        t1Params,
        t2Params,
        displayDims,
        4,
        1,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.interpolate_clipped_positions(
        positions,
        segments,
        visibility,
        t1Params,
        t2Params,
        displayDims,
        4,
        1,
        wasmStart,
        wasmEnd
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmStart, tsStart)).toBe(true);
      expect(arraysAlmostEqual(wasmEnd, tsEnd)).toBe(true);
    });

    // wasm.md O8[P4]: prior version embedded a `for (const ...)` loop with
    // 4 cases. A single failing case was reported as "lerp should match" —
    // no diagnostic about WHICH case failed. Parametrize via `it.each` so
    // each case becomes its own named row and a single mismatch surfaces
    // by its `t` value.
    it.each([
      { a: 0, b: 10, t: 0 },
      { a: 0, b: 10, t: 1 },
      { a: 0, b: 10, t: 0.5 },
      { a: -10, b: 10, t: 0.25 },
    ])('lerp WASM↔TS parity: lerp($a, $b, $t)', ({ a, b, t }) => {
      if (!wasmFilesExist) return; // it.each doesn't support skipIf in this version
      const tsResult = tsModule.lerp(a, b, t);
      const wasmResult = wasmModule!.lerp(a, b, t);
      expect(Math.abs(wasmResult - tsResult)).toBeLessThan(1e-6);
    });

    it.skipIf(!wasmFilesExist)('lerp_vec3 should match', () => {
      const a = new Float32Array([0, 10, -5]);
      const b = new Float32Array([10, 0, 5]);

      const tsResult = tsModule.lerp_vec3(a, b, 0.3);
      const wasmResult = wasmModule!.lerp_vec3(a, b, 0.3);

      expect(arraysAlmostEqual(wasmResult, tsResult)).toBe(true);
    });

    // wasm.md O10 / Phase E5: previous version wrapped 3 `distance_3d`
    // wasm-vs-ts comparisons in a single `it` with a `for` loop. A
    // regression in just the negative-coordinate case would surface as
    // a generic "distance_3d should match" failure without naming the
    // offending input pair. Split via `it.each` so each row names its
    // {a, b} pair on failure. Preserves the original `skipIf(!wasmFilesExist)`
    // gate by chaining `skipIf().each()` — the test still reports as
    // skipped (not "passed") when the wasm artefact is unavailable.
    it.skipIf(!wasmFilesExist).each<{ a: number[]; b: number[]; label: string }>([
      { a: [0, 0, 0], b: [3, 4, 0], label: '3-4-5 (5)' },
      { a: [1, 1, 1], b: [2, 2, 2], label: 'unit diagonal (sqrt 3)' },
      { a: [-1, -1, -1], b: [1, 1, 1], label: 'symmetric across origin (sqrt 12)' },
    ])('distance_3d wasm-vs-ts: $label', ({ a, b }) => {
      const aArr = new Float32Array(a);
      const bArr = new Float32Array(b);
      const tsResult = tsModule.distance_3d(aArr, bArr);
      const wasmResult = wasmModule!.distance_3d(aArr, bArr);
      expect(Math.abs(wasmResult - tsResult)).toBeLessThan(1e-5);
    });

    it.skipIf(!wasmFilesExist)('interpolate_scalars_batch should match', () => {
      const values = new Float32Array([0.1, 0.5, 0.9]);
      const segments = new Uint32Array([0, 1, 1, 2]);
      const visibility = new Uint8Array([1, 1]);
      const t1Params = new Float32Array([0.0, 0.5]);
      const t2Params = new Float32Array([1.0, 1.0]);

      const tsStart = new Float32Array(2);
      const tsEnd = new Float32Array(2);
      const wasmStart = new Float32Array(2);
      const wasmEnd = new Float32Array(2);

      const tsCount = tsModule.interpolate_scalars_batch(
        values,
        segments,
        visibility,
        t1Params,
        t2Params,
        2,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.interpolate_scalars_batch(
        values,
        segments,
        visibility,
        t1Params,
        t2Params,
        2,
        wasmStart,
        wasmEnd
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmStart, tsStart)).toBe(true);
      expect(arraysAlmostEqual(wasmEnd, tsEnd)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('interpolate_colors_batch should match', () => {
      const colors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      const segments = new Uint32Array([0, 1, 1, 2]);
      const visibility = new Uint8Array([1, 1]);
      const t1Params = new Float32Array([0.0, 0.5]);
      const t2Params = new Float32Array([0.5, 1.0]);

      const tsStart = new Float32Array(6);
      const tsEnd = new Float32Array(6);
      const wasmStart = new Float32Array(6);
      const wasmEnd = new Float32Array(6);

      const tsCount = tsModule.interpolate_colors_batch(
        colors,
        segments,
        visibility,
        t1Params,
        t2Params,
        2,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.interpolate_colors_batch(
        colors,
        segments,
        visibility,
        t1Params,
        t2Params,
        2,
        wasmStart,
        wasmEnd
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmStart, tsStart)).toBe(true);
      expect(arraysAlmostEqual(wasmEnd, tsEnd)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('calculate_segment_lengths should match', () => {
      const startPos = new Float32Array([0, 0, 0, 1, 1, 1]);
      const endPos = new Float32Array([3, 4, 0, 2, 2, 2]);

      const tsOutput = new Float32Array(2);
      const wasmOutput = new Float32Array(2);

      tsModule.calculate_segment_lengths(startPos, endPos, 2, tsOutput);
      wasmModule!.calculate_segment_lengths(startPos, endPos, 2, wasmOutput);

      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)(
      'calculate_segment_lengths should match on huge coordinates (f32 squared-length overflow)',
      () => {
        // Regression (#793): Rust accumulated the squared length in f32,
        // overflowing to Infinity once a component delta passed
        // sqrt(f32::MAX) ≈ 1.8e19, while the TS mirror reads the same f32
        // inputs but widens to f64 and returned the true value. Both now
        // accumulate in f64 and must agree on the finite result.
        const startPos = new Float32Array([-1e30, 0, 0]);
        const endPos = new Float32Array([1e30, 0, 0]);
        const tsOutput = new Float32Array(1);
        const wasmOutput = new Float32Array(1);

        tsModule.calculate_segment_lengths(startPos, endPos, 1, tsOutput);
        wasmModule!.calculate_segment_lengths(startPos, endPos, 1, wasmOutput);

        expect(Number.isFinite(tsOutput[0])).toBe(true);
        expect(Number.isFinite(wasmOutput[0])).toBe(true);
        expect(wasmOutput[0]).toBe(tsOutput[0]);
        // Exact expected value: the f64 delta of the f32 inputs, rounded
        // back to f32 on store.
        expect(tsOutput[0]).toBe(Math.fround(endPos[0] - startPos[0]));
      }
    );

    it.skipIf(!wasmFilesExist)('compute_cap_suppression should match', () => {
      // A 4-segment chain that exercises every branch of the kernel in one
      // shot: a straight-through joint, a FRACTIONAL 45-degree bend, a
      // genuinely clipped endpoint (t2 < 1), and a culled neighbour.
      //   v0 -> v1 -> v2 (straight, +x), v2 -> v3 (45° towards +y, its far
      //   end slice-trimmed at t2 = 0.6), v3 -> v4 (culled)
      // The 45° joint pins a real fractional value (cos 45° ≈ 0.7071) on
      // BOTH backends — a Rust build that quantised the scalar to a boolean
      // would pass a 0/1-only fixture but fails here.
      const d = Math.SQRT1_2;
      const segments = new Uint32Array([0, 1, 1, 2, 2, 3, 3, 4]);
      const visibility = new Uint8Array([1, 1, 1, 0]);
      const t1Params = new Float32Array([0.0, 0.0, 0.0, 0.0]);
      const t2Params = new Float32Array([1.0, 1.0, 0.6, 1.0]);
      const startPositions = new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]);
      // Segment 2 heads 45° off +x; its stored end is the clipped position
      // at t = 0.6 along the way to v3 (direction is unchanged by the trim).
      const endPositions = new Float32Array([1, 0, 0, 2, 0, 0, 2 + 0.6 * d, 0.6 * d, 0]);

      const tsStart = new Float32Array(3);
      const tsEnd = new Float32Array(3);
      const wasmStart = new Float32Array(3);
      const wasmEnd = new Float32Array(3);

      const tsCount = tsModule.compute_cap_suppression(
        segments,
        visibility,
        t1Params,
        t2Params,
        4,
        5,
        startPositions,
        endPositions,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.compute_cap_suppression(
        segments,
        visibility,
        t1Params,
        t2Params,
        4,
        5,
        startPositions,
        endPositions,
        wasmStart,
        wasmEnd
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysAlmostEqual(wasmStart, tsStart)).toBe(true);
      expect(arraysAlmostEqual(wasmEnd, tsEnd)).toBe(true);
      // Pin the expected values on BOTH backends, so a matched-but-wrong
      // pair still fails — including the fractional 45° suppression.
      for (const [start, end] of [
        [tsStart, tsEnd],
        [wasmStart, wasmEnd],
      ]) {
        // seg0: free start; straight joint at v1.
        expect(start[0]).toBe(0);
        expect(end[0]).toBeCloseTo(1, 6);
        // seg1: straight joint at v1; 45° joint at v2 — fractional cos 45°.
        expect(start[1]).toBeCloseTo(1, 6);
        expect(end[1]).toBeCloseTo(d, 5);
        // seg2: 45° joint at v2; clipped far end (t2 = 0.6 < 1) -> 1.
        expect(start[2]).toBeCloseTo(d, 5);
        expect(end[2]).toBe(1);
      }
    });

    it.skipIf(!wasmFilesExist)(
      'compute_cap_suppression should match on huge coordinates (f32 squared-length overflow)',
      () => {
        // Regression: the direction loop accumulated the squared length in f32,
        // which overflows to infinity once a component delta passes
        // sqrt(f32::MAX) ≈ 1.8e19. Rust then zeroed the direction and dropped
        // the joint (suppression 0) while the TS mirror — reading the same f32
        // inputs but accumulating in f64 — kept it (suppression 1). Since the TS
        // backend is the PRODUCTION path above 16 dimensions, the same scene
        // rendered differently at 17D than at 16D. Both now accumulate in f64.
        const segments = new Uint32Array([0, 1, 1, 2]);
        const visibility = new Uint8Array([1, 1]);
        const t1Params = new Float32Array([0, 0]);
        const t2Params = new Float32Array([1, 1]);
        const C = 1e30; // well past the 1.8e19 f32 overflow threshold
        const startPositions = new Float32Array([-C, 0, 0, 0, 0, 0]);
        const endPositions = new Float32Array([0, 0, 0, C, 0, 0]);

        const tsStart = new Float32Array(2);
        const tsEnd = new Float32Array(2);
        const wasmStart = new Float32Array(2);
        const wasmEnd = new Float32Array(2);

        tsModule.compute_cap_suppression(
          segments,
          visibility,
          t1Params,
          t2Params,
          2,
          3,
          startPositions,
          endPositions,
          tsStart,
          tsEnd
        );
        wasmModule!.compute_cap_suppression(
          segments,
          visibility,
          t1Params,
          t2Params,
          2,
          3,
          startPositions,
          endPositions,
          wasmStart,
          wasmEnd
        );

        // The shared vertex is a straight-through joint at any scale.
        expect(tsEnd[0]).toBeCloseTo(1, 6);
        expect(wasmEnd[0]).toBeCloseTo(1, 6);
        expect(arraysAlmostEqual(wasmStart, tsStart)).toBe(true);
        expect(arraysAlmostEqual(wasmEnd, tsEnd)).toBe(true);
      }
    );
  });

  // ============================================================================
  // STRESS TESTS WITH RANDOM DATA
  // ============================================================================
  describe('stress tests with random data', () => {
    it.skipIf(!wasmFilesExist)('effective_radii with random data', () => {
      const numPoints = 5000;
      const ndim = 6;

      // Generate deterministic "random" data using sine waves
      const positions = new Float32Array(numPoints * ndim);
      const radii = new Float32Array(numPoints);

      for (let i = 0; i < numPoints; i++) {
        for (let d = 0; d < ndim; d++) {
          positions[i * ndim + d] = Math.sin(i * 0.037 + d * 1.23) * 2;
        }
        radii[i] = Math.abs(Math.sin(i * 0.071)) * 1.5 + 0.1;
      }

      const displayDims = new Uint32Array([0, 1, 2]);
      const slicePos = new Float32Array(ndim).fill(0);
      const spatialExtend = new Uint8Array(ndim).fill(1);

      const tsOutput = new Float32Array(numPoints);
      const wasmOutput = new Float32Array(numPoints);

      const tsVisible = tsModule.calculate_effective_radii(
        positions,
        radii,
        displayDims,
        slicePos,
        spatialExtend,
        ndim,
        numPoints,
        tsOutput
      );
      const wasmVisible = wasmModule!.calculate_effective_radii(
        positions,
        radii,
        displayDims,
        slicePos,
        spatialExtend,
        ndim,
        numPoints,
        wasmOutput
      );

      expect(wasmVisible).toBe(tsVisible);
      expect(arraysAlmostEqual(wasmOutput, tsOutput)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('clip_segments_batch with many segments', () => {
      const numSegments = 2000;
      const ndim = 5;

      // Generate deterministic test data
      const positions = new Float32Array((numSegments + 1) * ndim);
      const segments = new Uint32Array(numSegments * 2);

      for (let i = 0; i <= numSegments; i++) {
        for (let d = 0; d < ndim; d++) {
          positions[i * ndim + d] = Math.sin(i * 0.05 + d * 0.7) * 5;
        }
      }

      for (let i = 0; i < numSegments; i++) {
        segments[i * 2] = i;
        segments[i * 2 + 1] = i + 1;
      }

      const slicePos = new Float32Array(ndim).fill(0);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 1.0, 1.0]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsVisibility = new Uint8Array(numSegments);
      const tsT1 = new Float32Array(numSegments);
      const tsT2 = new Float32Array(numSegments);
      const wasmVisibility = new Uint8Array(numSegments);
      const wasmT1 = new Float32Array(numSegments);
      const wasmT2 = new Float32Array(numSegments);

      const tsCount = tsModule.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        ndim,
        numSegments,
        tsVisibility,
        tsT1,
        tsT2
      );
      const wasmCount = wasmModule!.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        ndim,
        numSegments,
        wasmVisibility,
        wasmT1,
        wasmT2
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmVisibility, tsVisibility)).toBe(true);
      expect(arraysAlmostEqual(wasmT1, tsT1)).toBe(true);
      expect(arraysAlmostEqual(wasmT2, tsT2)).toBe(true);
    });

    // The hand-written 2D cases above pin one matrix each. This sweeps many
    // correlated 2D covariances at once, so a Rust/TS divergence in the phantom
    // axis (ln/exp vs f32 rounding, or an index slip) can't hide behind a single
    // lucky value. The phantom is a transcendental (exp of a mean of logs), the
    // one place in this kernel where the backends could plausibly drift.
    it.skipIf(!wasmFilesExist)('project_gsplats_nd_to_3d 2D with random data', () => {
      const splatCount = 2000;
      const ndim = 2;

      const positions = new Float32Array(splatCount * ndim);
      const cholesky = new Float32Array(splatCount * 3); // packed 2D
      const amplitudes = new Float32Array(splatCount);
      const colors = new Float32Array(splatCount * 3);

      // Deterministic pseudo-random via sine waves (this file's convention).
      for (let i = 0; i < splatCount; i++) {
        positions[i * 2] = Math.sin(i * 0.37) * 100;
        positions[i * 2 + 1] = Math.cos(i * 0.71) * 100;
        // Valid Cholesky: strictly positive diagonals, spanning 4 decades of
        // scale so the geometric mean is exercised across magnitudes.
        const s = Math.pow(10, Math.sin(i * 0.11) * 2);
        cholesky[i * 3] = (0.5 + Math.abs(Math.sin(i * 0.23))) * s;
        cholesky[i * 3 + 1] = Math.sin(i * 0.53) * 0.4 * s;
        cholesky[i * 3 + 2] = (0.5 + Math.abs(Math.cos(i * 0.31))) * s;
        amplitudes[i] = 0.5 + Math.abs(Math.sin(i * 0.17));
        colors[i * 3] = Math.abs(Math.sin(i * 0.13));
        colors[i * 3 + 1] = Math.abs(Math.cos(i * 0.19));
        colors[i * 3 + 2] = Math.abs(Math.sin(i * 0.29));
      }

      const args = {
        positions,
        cholesky,
        amplitudes,
        colors,
        discreteVisibility: new Uint8Array(splatCount).fill(1),
        slicePosition: new Float32Array(ndim),
        continuousHiddenDims: new Uint32Array([]),
        displayDims: new Uint32Array([0, 1]),
        ndim,
        splatCount,
      };

      // `runFused` is scoped to the gsplats describe block above; call directly.
      const run = (mod: WasmModule) => {
        const centers = new Float32Array(splatCount * 3);
        const chol = new Float32Array(splatCount * 6);
        const amps = new Float32Array(splatCount);
        const cols = new Float32Array(splatCount * 3);
        const count = mod.project_gsplats_nd_to_3d(
          args.positions,
          args.cholesky,
          args.amplitudes,
          args.colors,
          args.discreteVisibility,
          args.slicePosition,
          args.continuousHiddenDims,
          args.displayDims,
          args.ndim,
          args.splatCount,
          3,
          1e-6,
          3.0,
          centers,
          chol,
          amps,
          cols
        );
        return { count, centers, chol };
      };
      const ts = run(tsModule);
      const w = run(wasmModule!);

      expect(w.count).toBe(ts.count);
      expect(w.count).toBe(splatCount);

      // Relative comparison: absolute epsilon is meaningless across 4 decades.
      for (let i = 0; i < w.count * 6; i++) {
        const a = w.chol[i];
        const b = ts.chol[i];
        expect(Number.isFinite(a)).toBe(true);
        const scale = Math.max(Math.abs(a), Math.abs(b), 1e-30);
        expect(Math.abs(a - b) / scale).toBeLessThan(1e-4);
      }
      // Every phantom diagonal must be positive (SPD) and scale-matched, never ~0.
      for (let s = 0; s < w.count; s++) {
        const l = s * 6;
        expect(w.chol[l + 3]).toBe(0);
        expect(w.chol[l + 4]).toBe(0);
        expect(w.chol[l + 5]).toBeGreaterThan(0);
        // Relative, not absolute: these diagonals span 4 decades, and the
        // phantom is an exp(mean(ln)) round-trip worth a few f32 ULP.
        const wantPhantom = Math.sqrt(w.chol[l] * w.chol[l + 2]);
        expect(Math.abs(w.chol[l + 5] - wantPhantom) / wantPhantom).toBeLessThan(1e-5);
      }
      expect(
        arraysAlmostEqual(w.centers.subarray(0, w.count * 3), ts.centers.subarray(0, ts.count * 3))
      ).toBe(true);
    });
  });
});
