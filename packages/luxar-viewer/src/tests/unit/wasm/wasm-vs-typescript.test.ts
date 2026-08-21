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
import {
  arraysEqual as sharedArraysEqual,
  arraysAlmostEqual as sharedArraysAlmostEqual,
} from '../../helpers/array-compare';
import {
  WASM_BUILD_HINT,
  tryLoadWasmArtifact,
  wasmArtifactExists,
  wasmJsPath,
} from '../../helpers/wasm-artifact';

// WASM module reference (loaded dynamically)
let wasmModule: WasmModule | null = null;
let tsModule: WasmModule;

// Pre-check if WASM files exist (synchronous check at module load time)
const wasmFilesExist = wasmArtifactExists();
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
    console.log(`[Test] ${WASM_BUILD_HINT}`);
    return;
  }

  // A load failure is a soft skip; a STALE build still throws by name (see
  // tests/helpers/wasm-artifact.ts).
  wasmModule = await tryLoadWasmArtifact((error) => {
    console.log('[Test] WASM module failed to load:', error);
    console.log(`[Test] ${WASM_BUILD_HINT}`);
  });
  if (wasmModule) console.log('[Test] WASM module loaded successfully');
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

  describe('regression: lines clip non-finite non-displayed dim (#806)', () => {
    it.skipIf(!wasmFilesExist)('NaN on a hidden dim: both backends agree (invisible)', () => {
      // 2 vertices, ndim=3, display dims [0,1]; hidden dim 2 carries a NaN on
      // one endpoint. f32::max/min in Rust would keep the t-params finite and
      // render, while TS propagated NaN — the two used to disagree. The fix
      // makes BOTH treat it as invisible.
      const positions = new Float32Array([0.0, 0.0, NaN, 1.0, 1.0, 0.05]);
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
      // Both must mark it invisible.
      expect(tsVis[0]).toBe(0);
      expect(wVis[0]).toBe(0);
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
        /** Ask both backends to record the surviving source indices (#1423). */
        recordSourceIndices?: boolean;
      }
    ): {
      count: number;
      centers: Float32Array;
      chol: Float32Array;
      amps: Float32Array;
      cols: Float32Array;
      src: Uint32Array;
    } => {
      const n = args.splatCount;
      const k = args.colorComponents ?? 3;
      const centers = new Float32Array(n * 3);
      const chol = new Float32Array(n * 6);
      const amps = new Float32Array(n);
      const cols = new Float32Array(n * k);
      // Empty = the recording opt-out; a real n-long array turns it on.
      const src = new Uint32Array(args.recordSourceIndices ? n : 0);
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
        cols,
        src
      );
      return { count, centers, chol, amps, cols, src };
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
      'project_gsplats_nd_to_3d matches TS (recorded source indices under compaction)',
      () => {
        // The map picking resolves on-disk element indices through (#1423) is
        // built from these, so the two backends must agree exactly — an
        // off-by-one on either side reports a neighbour's label.
        const ndim = 4;
        const splatCount = 6;
        const one = [2.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.5, 0.5, 0.0, 4.0];
        const args = {
          // Splats 1 and 4 sit far off-slice in the hidden dim (attenuated
          // out); splat 3 is discrete-gated. Survivors: 0, 2, 5 — a
          // NON-CONTIGUOUS set, the only shape where slot ≠ source index.
          // prettier-ignore
          positions: new Float32Array([
            0, 0, 0, 0,
            1, 1, 1, 50,
            2, 2, 2, 0.2,
            3, 3, 3, 0,
            4, 4, 4, 50,
            5, 5, 5, 0.1,
          ]),
          cholesky: new Float32Array(Array.from({ length: splatCount }, () => one).flat()),
          amplitudes: new Float32Array([1.0, 1.0, 0.9, 1.0, 1.0, 0.7]),
          colors: new Float32Array(splatCount * 3).fill(0.5),
          discreteVisibility: new Uint8Array([1, 1, 1, 0, 1, 1]),
          slicePosition: new Float32Array([0, 0, 0, 0]),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          recordSourceIndices: true,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);
        expect(w.count).toBe(ts.count);
        expect(w.count).toBe(3);
        expect(Array.from(w.src.subarray(0, w.count))).toEqual([0, 2, 5]);
        expect(Array.from(w.src.subarray(0, w.count))).toEqual(
          Array.from(ts.src.subarray(0, ts.count))
        );
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
  });

  // ============================================================================
  // f32 OPERATION-ORDER PARITY (#1820)
  // ============================================================================
  //
  // The cases above compare with `arraysAlmostEqual` at 1e-5 — loose enough to
  // pass while the TS reference did its multi-step arithmetic in f64 and Rust
  // did it in f32. The TS kernels are not a fallback (CLAUDE.md: they are the
  // PRODUCTION backend above 16 dimensions and whenever WASM is unavailable), so
  // "close" is the wrong contract. These cases are EXACT wherever exactness is
  // reachable and ULP/absolute-bounded, with a measured bound, where it is not.
  describe('f32 operation-order parity (#1820)', () => {
    /** Deterministic LCG — no Math.random, so a failure is reproducible. */
    function lcg(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
      };
    }

    /**
     * Bit-level mismatch summary: the total count plus the first few offenders.
     * Asserting on this instead of `toEqual` over the whole array keeps the
     * comparison exact while making a failure readable at 20 000 elements.
     */
    function exactMismatches(
      ts: ArrayLike<number>,
      wasm: ArrayLike<number>,
      limit = 4
    ): { total: number; first: Array<{ i: number; ts: number; wasm: number }> } {
      const first: Array<{ i: number; ts: number; wasm: number }> = [];
      let total = 0;
      for (let i = 0; i < ts.length; i++) {
        if (!Object.is(ts[i], wasm[i])) {
          total++;
          if (first.length < limit) first.push({ i, ts: ts[i], wasm: wasm[i] });
        }
      }
      return { total, first };
    }
    const NO_MISMATCHES = { total: 0, first: [] };

    /** Distance in f32 ulps between two values (both are already f32). */
    function ulpDistance(a: number, b: number): number {
      const f = new Float32Array([a, b]);
      const i = new Int32Array(f.buffer);
      const ordered = (x: number) => (x < 0 ? 0x80000000 - x : x);
      return Math.abs(ordered(i[0]) - ordered(i[1]));
    }

    function maxUlp(ts: ArrayLike<number>, wasm: ArrayLike<number>): number {
      let m = 0;
      for (let i = 0; i < ts.length; i++) m = Math.max(m, ulpDistance(ts[i], wasm[i]));
      return m;
    }

    function maxAbs(ts: ArrayLike<number>, wasm: ArrayLike<number>): number {
      let m = 0;
      for (let i = 0; i < ts.length; i++) m = Math.max(m, Math.abs(ts[i] - wasm[i]));
      return m;
    }

    // ------------------------------------------------------------------
    // effective_radii
    // ------------------------------------------------------------------
    it.skipIf(!wasmFilesExist)(
      'calculate_effective_radii is BIT-EXACT over a 20k randomized nD sweep',
      () => {
        // Every step of this kernel is a plain arithmetic op — no
        // transcendentals — so exactness is fully reachable and anything less
        // is a bug. Mixed spatial/discrete hidden dims exercise both the
        // Pythagorean accumulation and the tolerance gate.
        const numPoints = 20000;
        const ndim = 6;
        const positions = new Float32Array(numPoints * ndim);
        const radii = new Float32Array(numPoints);
        const rnd = lcg(0x1820);
        for (let i = 0; i < numPoints; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = (rnd() - 0.5) * 4;
          radii[i] = 0.5 + rnd() * 2;
        }
        const displayDims = new Uint32Array([0, 1, 2]);
        const slicePos = new Float32Array(ndim);
        const spatialExtend = new Uint8Array([1, 1, 1, 1, 1, 0]); // dim 5 discrete
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

        expect(tsVisible).toBe(wasmVisible);
        expect(exactMismatches(tsOutput, wasmOutput)).toEqual(NO_MISMATCHES);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'calculate_effective_radii: f32 rounding decides VISIBILITY, not just the digits',
      () => {
        // Hand-picked so the Pythagorean test lands on opposite sides of the
        // rim in the two precisions. With the hidden coords below,
        //   f32:  distanceSquared == radiusSquared (0.28985107) -> CULLED
        //   f64:  distanceSquared  < radiusSquared             -> VISIBLE
        // so `visibleCount` — which the caller uses to size its compaction —
        // differs by one between the backends, not by an ulp.
        const positions = new Float32Array([0, 0, 0, 0.5120331645011902, 0.166352316737175]);
        const radii = new Float32Array([0.5383781790733337]);
        const displayDims = new Uint32Array([0, 1, 2]);
        const slicePos = new Float32Array(5);
        const spatialExtend = new Uint8Array([1, 1, 1, 1, 1]);
        const tsOutput = new Float32Array(1);
        const wasmOutput = new Float32Array(1);

        const tsVisible = tsModule.calculate_effective_radii(
          positions,
          radii,
          displayDims,
          slicePos,
          spatialExtend,
          5,
          1,
          tsOutput
        );
        const wasmVisible = wasmModule!.calculate_effective_radii(
          positions,
          radii,
          displayDims,
          slicePos,
          spatialExtend,
          5,
          1,
          wasmOutput
        );

        expect(wasmVisible).toBe(0); // pin the f32 answer, not just agreement
        expect(tsVisible).toBe(wasmVisible);
        expect(tsOutput[0]).toBe(wasmOutput[0]);
        expect(tsOutput[0]).toBe(0);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'calculate_effective_radii: the DISCRETE tolerance gate is evaluated in f32',
      () => {
        // |value - target| is 0.5000000288709998 exactly (both operands are
        // f32, so the difference is exact in f64) but rounds to exactly 0.5 in
        // f32. Rust's `> 0.5` is therefore FALSE and the point survives the
        // discrete gate; an f64 comparison rejects it. Opposite direction to
        // the case above, so a fix that frounds one subtraction and not the
        // other still fails here.
        const target = 0.0012588808313012123;
        const value = 0.501258909702301;
        const positions = new Float32Array([0, 0, 0, value]);
        const radii = new Float32Array([1.0]);
        const displayDims = new Uint32Array([0, 1, 2]);
        const slicePos = new Float32Array([0, 0, 0, target]);
        const spatialExtend = new Uint8Array([1, 1, 1, 0]); // dim 3 discrete
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

        expect(wasmVisible).toBe(1); // f32 says "within tolerance"
        expect(tsVisible).toBe(wasmVisible);
        expect(tsOutput[0]).toBe(wasmOutput[0]);
      }
    );

    // ------------------------------------------------------------------
    // mahalanobis_distance
    // ------------------------------------------------------------------
    it.skipIf(!wasmFilesExist)(
      'mahalanobis_distance is BIT-EXACT over 5k randomized correlated factors',
      () => {
        // Forward substitution + a norm — no transcendentals, so the scalar the
        // TS twin returns must be the very same f64 wasm-bindgen hands back for
        // Rust's f32, not merely within sqrt(ndim)*1e-5 of it.
        const trials = 5000;
        const ndim = 8;
        const packedSize = (ndim * (ndim + 1)) / 2;
        const rnd = lcg(0xc0ffee);
        const diff = new Float32Array(ndim);
        const packedL = new Float32Array(packedSize);
        const mismatches: Array<{ trial: number; ts: number; wasm: number }> = [];
        for (let t = 0; t < trials; t++) {
          for (let i = 0; i < ndim; i++) diff[i] = (rnd() - 0.5) * 4;
          for (let row = 0; row < ndim; row++) {
            for (let col = 0; col <= row; col++) {
              packedL[(row * (row + 1)) / 2 + col] =
                col === row ? 0.5 + rnd() * 2 : (rnd() - 0.5) * 1.5;
            }
          }
          const tsDist = tsModule.mahalanobis_distance(diff, packedL, ndim);
          const wasmDist = wasmModule!.mahalanobis_distance(diff, packedL, ndim);
          if (!Object.is(tsDist, wasmDist) && mismatches.length < 4) {
            mismatches.push({ trial: t, ts: tsDist, wasm: wasmDist });
          }
        }
        expect(mismatches).toEqual([]);
      }
    );

    // ------------------------------------------------------------------
    // project_gsplats_nd_to_3d
    // ------------------------------------------------------------------
    /** Run the fused kernel with explicit `minAmplitude` / `truncate`. */
    const project = (
      mod: WasmModule,
      a: {
        positions: Float32Array;
        cholesky: Float32Array;
        amplitudes: Float32Array;
        discreteVisibility: Uint8Array;
        slicePosition: Float32Array;
        continuousHiddenDims: Uint32Array;
        displayDims: Uint32Array;
        ndim: number;
        splatCount: number;
        minAmplitude: number;
        truncate?: number;
      }
    ) => {
      const n = a.splatCount;
      const centers = new Float32Array(n * 3);
      const chol = new Float32Array(n * 6);
      const amps = new Float32Array(n);
      const cols = new Float32Array(n * 3);
      const count = mod.project_gsplats_nd_to_3d(
        a.positions,
        a.cholesky,
        a.amplitudes,
        new Float32Array(n * 3).fill(1),
        a.discreteVisibility,
        a.slicePosition,
        a.continuousHiddenDims,
        a.displayDims,
        a.ndim,
        n,
        3,
        a.minAmplitude,
        a.truncate ?? 3.0,
        centers,
        chol,
        amps,
        cols,
        new Uint32Array(0)
      );
      return { count, centers, chol, amps };
    };

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d is BIT-EXACT with no continuous hidden dims (no exp on the path)',
      () => {
        // Attenuation short-circuits to exactly 1.0, so the only float work is
        // the display marginal Cholesky — Σ_S dot products and a Crout
        // reduction, both pure arithmetic. Full exactness is reachable, and the
        // Crout step is where cancellation used to blow the f64-vs-f32
        // divergence out to thousands of ulps.
        const splatCount = 20000;
        const ndim = 3;
        const packedSize = 6;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = lcg(0xbeef);
        for (let i = 0; i < splatCount; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = rnd() * 10;
          const b = i * packedSize;
          cholesky[b] = 0.5 + rnd() * 3;
          cholesky[b + 1] = (rnd() - 0.5) * 2;
          cholesky[b + 2] = 0.5 + rnd() * 3;
          cholesky[b + 3] = (rnd() - 0.5) * 2;
          cholesky[b + 4] = (rnd() - 0.5) * 2;
          cholesky[b + 5] = 0.5 + rnd() * 3;
          amplitudes[i] = 0.2 + rnd();
        }
        const args = {
          positions,
          cholesky,
          amplitudes,
          discreteVisibility: new Uint8Array(splatCount).fill(1),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
        };
        const ts = project(tsModule, args);
        const w = project(wasmModule!, args);

        expect(ts.count).toBe(w.count);
        expect(ts.count).toBe(splatCount);
        expect(exactMismatches(ts.chol, w.chol)).toEqual(NO_MISMATCHES);
        expect(exactMismatches(ts.centers, w.centers)).toEqual(NO_MISMATCHES);
        expect(exactMismatches(ts.amps, w.amps)).toEqual(NO_MISMATCHES);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d: f32 accumulation decides the EMITTED SPLAT COUNT',
      () => {
        // One splat, two correlated continuous hidden dims. The f64 and f32
        // marginal-Cholesky + forward-substitution chains land ~1700 ulps apart
        // on the attenuated amplitude, and `minAmplitude` is placed strictly
        // between them: WASM emits the splat, an f64 TS chain culls it.
        const ndim = 5;
        const cholesky = new Float32Array([
          1.3153510093688965, -0.20379841327667236, 2.7451348304748535, -1.899763584136963,
          0.9705871343612671, 1.3426613807678223, -0.3823320269584656, -0.2775517702102661,
          1.0715609788894653, 2.4265294075012207, 0.28441232442855835, 0.37014153599739075,
          -1.196919560432434, -1.545201301574707, 0.83291095495224,
        ]);
        const args = {
          positions: new Float32Array([
            0.38872674107551575, -1.677039384841919, -1.0393322706222534, -1.6747952699661255,
            -1.6385165452957153,
          ]),
          cholesky,
          amplitudes: new Float32Array([1.2808760404586792]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([3, 4]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount: 1,
          minAmplitude: 0.00017888002912513912,
        };
        const ts = project(tsModule, args);
        const w = project(wasmModule!, args);

        expect(w.count).toBe(1); // pin the f32 answer
        expect(ts.count).toBe(w.count);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d: minAmplitude is thresholded as an f32',
      () => {
        // `min_amplitude: f32` on the Rust signature, so wasm-bindgen narrows
        // 0.25 + 1e-9 to exactly 0.25 at the boundary and the splat passes
        // `!(0.25 < 0.25)`. This backend gets the raw f64 and must narrow it
        // itself; comparing against the f64 culls a splat WASM emits. No
        // hidden dims, so attenuation is exactly 1 and nothing else can move.
        const ndim = 3;
        const args = {
          positions: new Float32Array([0, 0, 0]),
          cholesky: new Float32Array([1, 0, 1, 0, 0, 1]),
          amplitudes: new Float32Array([0.25]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount: 1,
          minAmplitude: 0.25 + 1e-9,
        };
        expect(Math.fround(args.minAmplitude)).toBe(0.25); // the premise
        const ts = project(tsModule, args);
        const w = project(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        expect(ts.amps[0]).toBe(w.amps[0]);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'compute_marginal_cholesky: the degeneracy floor clamps at f32::MIN_POSITIVE',
      () => {
        // Σ_S = diag(L00², 0) with L00 = 1e-14 gives maxDiag ≈ 1e-28, so
        // maxDiag × CHOLESKY_RELATIVE_EPSILON ≈ 1e-40 — a denormal f32, far
        // below 2⁻¹²⁶. Rust clamps to f32::MIN_POSITIVE and regularizes the
        // dead axis to √(2⁻¹²⁶) = 2⁻⁶³; clamping at `Number.MIN_VALUE`
        // (≈5e-324) instead leaves 1e-40 and regularizes to ~1e-20 — four
        // orders of magnitude apart, which is not a rounding difference.
        const ndim = 2;
        const args = {
          positions: new Float32Array([0, 0]),
          cholesky: new Float32Array([1e-14, 0, 0]), // packed 2D [L00, L10, L11]
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1]),
          ndim,
          splatCount: 1,
          minAmplitude: 1e-9,
        };
        const ts = project(tsModule, args);
        const w = project(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        // L11 is the regularized dead axis: √(f32::MIN_POSITIVE) = 2⁻⁶³.
        expect(w.chol[2]).toBe(Math.pow(2, -63));
        expect(Array.from(ts.chol.subarray(0, 6))).toEqual(Array.from(w.chol.subarray(0, 6)));
      }
    );

    // ------------------------------------------------------------------
    // The two paths where exactness is NOT reachable — bounded, with the
    // bound measured rather than guessed.
    // ------------------------------------------------------------------
    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d attenuation: exp residual only, bounded absolutely',
      () => {
        // The full attenuation path — correlated 5D factors, TWO continuous
        // hidden dims (so the marginal Cholesky and the forward substitution
        // both do real work) and 3 display dims.
        //
        // `Math.fround(Math.exp(x))` is not `f32::exp(x)` — V8's ieee754 kernel
        // and the wasm libm's `expf` are different approximations, so that ONE
        // step can never be made exact (porting musl's expf is out of scope).
        // Everything around it now is, which is what makes the residual
        // bounded: the attenuated amplitude is `amp · inv · (rawExp − shiftC)`
        // with rawExp ≤ 1, so a ≤1 ulp error in rawExp costs a few ulps of
        // `amp · inv` and nothing accumulates behind it.
        //
        // Measured on this exact fixture (20 000 splats, seed 555):
        //   before the f32 rounding: 16966/118014 display-Cholesky slots wrong
        //                            (up to 39828 ulp), amplitudes off by up to
        //                            5.36e-7 absolute (9·2⁻²⁴)
        //   after:                   0 Cholesky slots wrong, amplitudes off by
        //                            at most 1.79e-7 (3·2⁻²⁴)
        // The bound below is 6·2⁻²⁴ — 2× the measured worst case (headroom for
        // a libm change) and still comfortably under the pre-fix figure.
        //
        // A ULP bound is deliberately NOT used on the amplitudes: `rawExp −
        // shiftC` cancels catastrophically at the truncation radius, so the
        // RELATIVE error of a near-zero attenuated amplitude is unbounded while
        // its absolute error stays tiny. The Cholesky, which crosses no
        // transcendental, is held to exact equality instead.
        const splatCount = 20000;
        const ndim = 5;
        const packedSize = 15;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = lcg(555);
        for (let i = 0; i < splatCount; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = (rnd() - 0.5) * 6;
          for (let row = 0; row < ndim; row++) {
            for (let col = 0; col <= row; col++) {
              cholesky[i * packedSize + (row * (row + 1)) / 2 + col] =
                col === row ? 0.3 + rnd() * 3 : (rnd() - 0.5) * 3;
            }
          }
          amplitudes[i] = 0.2 + rnd();
        }
        const args = {
          positions,
          cholesky,
          amplitudes,
          discreteVisibility: new Uint8Array(splatCount).fill(1),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([3, 4]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
        };
        const ts = project(tsModule, args);
        const w = project(wasmModule!, args);

        // The visible SET must agree exactly even though the amplitudes cannot.
        expect(ts.count).toBe(w.count);
        expect(ts.count).toBeGreaterThan(splatCount / 2); // the fixture is not degenerate
        const n = ts.count;
        // Centers and the display marginal never touch `exp`: still exact.
        expect(
          exactMismatches(ts.centers.subarray(0, n * 3), w.centers.subarray(0, n * 3))
        ).toEqual(NO_MISMATCHES);
        expect(exactMismatches(ts.chol.subarray(0, n * 6), w.chol.subarray(0, n * 6))).toEqual(
          NO_MISMATCHES
        );
        expect(maxAbs(ts.amps.subarray(0, n), w.amps.subarray(0, n))).toBeLessThanOrEqual(
          6 * Math.pow(2, -24)
        );
      }
    );

    it.skipIf(!wasmFilesExist)(
      'computeDisplayCholesky3D 2D phantom: ln/exp residual only, within 2 ulp',
      () => {
        // The phantom z diagonal is `exp(mean(ln(Lii)))`, so it inherits BOTH
        // transcendental residuals — measured at ≤2 ulp (one from `ln`, one
        // from `exp`) over this 20 000-splat sweep, 3106/20000 differing. The
        // other five packed slots are pure arithmetic and must be exact; they
        // were 1247/100000 wrong before this fix.
        const splatCount = 20000;
        const ndim = 2;
        const packedSize = 3;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const rnd = lcg(4242);
        for (let i = 0; i < splatCount; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = rnd() * 10;
          const b = i * packedSize;
          cholesky[b] = 0.5 + rnd() * 3;
          cholesky[b + 1] = (rnd() - 0.5) * 2;
          cholesky[b + 2] = 0.5 + rnd() * 3;
        }
        const args = {
          positions,
          cholesky,
          amplitudes: new Float32Array(splatCount).fill(1),
          discreteVisibility: new Uint8Array(splatCount).fill(1),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
        };
        const ts = project(tsModule, args);
        const w = project(wasmModule!, args);
        expect(ts.count).toBe(w.count);
        expect(ts.count).toBe(splatCount);

        const tsPhantom = new Float32Array(splatCount);
        const wPhantom = new Float32Array(splatCount);
        const tsRest = new Float32Array(splatCount * 5);
        const wRest = new Float32Array(splatCount * 5);
        for (let i = 0; i < splatCount; i++) {
          tsPhantom[i] = ts.chol[i * 6 + 5];
          wPhantom[i] = w.chol[i * 6 + 5];
          for (let k = 0; k < 5; k++) {
            tsRest[i * 5 + k] = ts.chol[i * 6 + k];
            wRest[i * 5 + k] = w.chol[i * 6 + k];
          }
        }
        expect(exactMismatches(tsRest, wRest)).toEqual(NO_MISMATCHES);
        expect(maxUlp(tsPhantom, wPhantom)).toBeLessThanOrEqual(2);
      }
    );
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

    it.skipIf(!wasmFilesExist)('compute_joint_codes should match', () => {
      // One chain that exercises every branch of the kernel: a straight-through
      // joint, an end-to-end (opposing orientation) joint, a degree->=3 hub, a
      // slice-trimmed endpoint, a culled neighbour, and a self-registering
      // degenerate segment.
      //
      // Note what is NOT here any more: the old fixture pinned cos(45 deg) to
      // catch a backend that quantised the angle scalar, and a separate case
      // pinned huge coordinates because the direction loop's f32 squared length
      // overflowed to infinity in Rust while the TS mirror accumulated in f64 —
      // a real divergence, since TS is the PRODUCTION backend above 16
      // dimensions. Both are MOOT: the kernel reads no positions and computes no
      // angle, so its output is integer index arithmetic and the two backends
      // agree by construction rather than by careful matching of float order.
      //
      //   slots 0,1: v0 -> v1 -> v2   straight-through joint at v1
      //   slot   2:  v4 -> v3         END-to-END with slot 3 at v3
      //   slot   3:  v5 -> v3         (so both report the partner's END)
      //   slot   4:  v6 -> v7         start trimmed (t1 > 0)
      //   culled:    v7 -> v8         invisible, must not anchor a joint
      //   slot   5:  v9 -> v9         degenerate: must not name itself
      //   slots 6,7,8: v10 x3         degree-3 hub (one START, two ENDs)
      //
      // The hub is the branch where the two backends are written differently —
      // Rust `match`es on the degree, TypeScript runs two `if`s — so it has to
      // be in the CROSS-backend fixture and not only in the per-backend ones.
      const segments = new Uint32Array([
        0, 1, 1, 2, 4, 3, 5, 3, 6, 7, 7, 8, 9, 9, 10, 11, 12, 10, 13, 10,
      ]);
      const visibility = new Uint8Array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
      const t1Params = new Float32Array([0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0]);
      const t2Params = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

      const tsStart = new Float32Array(9).fill(99);
      const tsEnd = new Float32Array(9).fill(99);
      const wasmStart = new Float32Array(9).fill(99);
      const wasmEnd = new Float32Array(9).fill(99);

      const tsCount = tsModule.compute_joint_codes(
        segments,
        visibility,
        t1Params,
        t2Params,
        10,
        14,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.compute_joint_codes(
        segments,
        visibility,
        t1Params,
        t2Params,
        10,
        14,
        wasmStart,
        wasmEnd
      );

      expect(tsCount).toBe(9);
      expect(wasmCount).toBe(tsCount);
      // Codes are exact small integers, so this is toEqual, not almost-equal.
      expect(Array.from(wasmStart)).toEqual(Array.from(tsStart));
      expect(Array.from(wasmEnd)).toEqual(Array.from(tsEnd));

      // Pin the values too, so a change of encoding cannot pass by agreeing
      // with itself on both backends.
      expect(tsEnd[0]).toBe(2); // slot 0 end joins slot 1 at its START: +(1+1)
      expect(tsStart[1]).toBe(-3); // slot 1 start joins slot 0 at its END: -(0+3)
      expect(tsEnd[2]).toBe(-6); // slot 2 end joins slot 3 at its END: -(3+3)
      expect(tsEnd[3]).toBe(-5); // slot 3 end joins slot 2 at its END: -(2+3)
      expect(tsStart[4]).toBe(-1); // trimmed off its vertex
      expect(tsStart[5]).toBe(0); // degenerate self-registering segment
      expect(tsEnd[5]).toBe(0);

      // The hub, from all three of its endpoints. Asserted on BOTH backends'
      // arrays: the -2 sentinel has to be present, not merely agreed on.
      for (const out of [tsStart, wasmStart]) expect(out[6]).toBe(-2);
      for (const out of [tsEnd, wasmEnd]) {
        expect(out[7]).toBe(-2);
        expect(out[8]).toBe(-2);
      }
      expect(tsStart.includes(-2) || tsEnd.includes(-2)).toBe(true);
      expect(wasmStart.includes(-2) || wasmEnd.includes(-2)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('compute_joint_codes agrees on NaN clip params', () => {
      // A NaN t-param registers nothing (`t <= 0` is false for NaN) and must
      // therefore read nothing either — both passes run the SAME predicate. On
      // the old complementary spelling the endpoint read anyway and the
      // code-sum difference decoded to slot 3, a real but unrelated segment in
      // this 4-segment scene, so the `visibleCount` bound could not see it.
      //
      // TypeScript is the PRODUCTION backend above 16 dimensions, so this is
      // exactly the kind of input only one of the two backends ever sees in
      // the field — which is why it belongs in the cross-backend fixture.
      const segments = new Uint32Array([5, 8, 5, 6, 5, 7, 0, 1]);
      const visibility = new Uint8Array([1, 1, 1, 1]);
      const t1Params = new Float32Array([NaN, 0, 0, 0]);
      const t2Params = new Float32Array([1, 1, 1, 1]);

      const tsStart = new Float32Array(4).fill(99);
      const tsEnd = new Float32Array(4).fill(99);
      const wasmStart = new Float32Array(4).fill(99);
      const wasmEnd = new Float32Array(4).fill(99);

      tsModule.compute_joint_codes(segments, visibility, t1Params, t2Params, 4, 9, tsStart, tsEnd);
      wasmModule!.compute_joint_codes(
        segments,
        visibility,
        t1Params,
        t2Params,
        4,
        9,
        wasmStart,
        wasmEnd
      );

      expect(Array.from(wasmStart)).toEqual(Array.from(tsStart));
      expect(Array.from(wasmEnd)).toEqual(Array.from(tsEnd));
      expect(tsStart[0]).toBe(-1); // JOINT_CLIPPED — never registered, never reads
      expect(tsStart[1]).toBe(3); // the two that DID register pair with each other
      expect(tsStart[2]).toBe(2);
    });

    it.skipIf(!wasmFilesExist)(
      'compute_joint_codes agrees on duplicate and zero-length segments',
      () => {
        // Two IDENTICAL index pairs, plus a zero-length segment.
        //
        //   slots 0,1: v0 -> v1 twice   duplicates: each vertex reaches
        //                               degree 2, so they name each other
        //   slot   2:  v2 -> v2         zero length: registers both of its own
        //                               endpoints on one vertex
        //
        // The kernel is position-blind and matches by index, so a duplicate is
        // indistinguishable from an ordinary joint here — that is deliberate,
        // and the point of the case is that both backends say so identically.
        const segments = new Uint32Array([0, 1, 0, 1, 2, 2]);
        const visibility = new Uint8Array([1, 1, 1]);
        const t1Params = new Float32Array([0, 0, 0]);
        const t2Params = new Float32Array([1, 1, 1]);

        const tsStart = new Float32Array(3).fill(99);
        const tsEnd = new Float32Array(3).fill(99);
        const wasmStart = new Float32Array(3).fill(99);
        const wasmEnd = new Float32Array(3).fill(99);

        tsModule.compute_joint_codes(
          segments,
          visibility,
          t1Params,
          t2Params,
          3,
          3,
          tsStart,
          tsEnd
        );
        wasmModule!.compute_joint_codes(
          segments,
          visibility,
          t1Params,
          t2Params,
          3,
          3,
          wasmStart,
          wasmEnd
        );

        expect(Array.from(wasmStart)).toEqual(Array.from(tsStart));
        expect(Array.from(wasmEnd)).toEqual(Array.from(tsEnd));

        // v0 holds the two duplicates' STARTs, v1 their two ENDs.
        expect(tsStart[0]).toBe(2); // names slot 1 at its START: +(1 + 1)
        expect(tsStart[1]).toBe(1); // names slot 0 at its START: +(0 + 1)
        expect(tsEnd[0]).toBe(-4); // names slot 1 at its END: -(1 + 3)
        expect(tsEnd[1]).toBe(-3); // names slot 0 at its END: -(0 + 3)
        // The zero-length segment must not name itself.
        expect(tsStart[2]).toBe(0);
        expect(tsEnd[2]).toBe(0);
      }
    );
  });

  // ============================================================================
  // MESH CULLING
  // ============================================================================
  describe('mesh_culling functions', () => {
    /**
     * The case that fails WITHOUT `Math.fround` in the TS backend.
     *
     * Rust computes `slice - tolerance` in f32 and ROUNDS; JS reads two f32
     * values out of a Float32Array and subtracts them in f64, which is EXACT.
     * The gap is under half an ulp, so it is normally invisible — but when the
     * f32 rounding goes DOWN, the rounded bound is itself a legal f32 vertex
     * coordinate lying inside the gap, and a vertex sitting exactly there is
     * `>=` the f32 bound (WASM: visible) but `<` the exact f64 bound (an
     * unfrounded TS backend: culled).
     */
    it.skipIf(!wasmFilesExist)('agrees at a sub-ulp f32-vs-f64 slab boundary', () => {
      const slice = Math.fround(1.0);
      const tol = Math.fround(0.1);
      const exactBound = slice - tol; // f64, exact
      const f32Bound = Math.fround(exactBound); // what Rust computes

      // Assert the premise of this test rather than trusting it: the rounding
      // must go DOWN, or there is no observable divergence to detect.
      expect(f32Bound).toBeLessThan(exactBound);

      const positions = new Float32Array([0, 0, 0, f32Bound]);
      const slicePos = new Float32Array([0, 0, 0, slice]);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, tol]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsMask = new Uint8Array(1);
      const wasmMask = new Uint8Array(1);
      const tsVisible = tsModule.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        4,
        1,
        tsMask
      );
      const wasmVisible = wasmModule!.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        4,
        1,
        wasmMask
      );

      // The vertex sits exactly ON the f32 bound, so both must call it visible.
      expect(wasmVisible).toBe(1);
      expect(tsVisible).toBe(wasmVisible);
      expect(arraysEqual(tsMask, wasmMask)).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('agrees on a random 6D mesh with 3 hidden dims', () => {
      const numVertices = 4000;
      const numFaces = 3000;
      const ndim = 6;

      // Deterministic pseudo-random data (sine waves), as the sibling stress
      // tests do — but with a PER-DIMENSION frequency. A shared phase
      // (`sin(i * 0.037 + d * 1.23)`) correlates the three hidden coordinates,
      // and AND-ing three correlated slabs culls every vertex, which would make
      // the comparison below trivially true.
      const positions = new Float32Array(numVertices * ndim);
      for (let i = 0; i < numVertices; i++) {
        for (let d = 0; d < ndim; d++) {
          positions[i * ndim + d] = Math.sin(i * 0.037 * (d + 1) + d * 1.23) * 2;
        }
      }
      // Faces over CONSECUTIVE vertices, like a real indexed surface: with
      // uniformly random indices, needing all three vertices visible leaves ~0
      // faces and the compaction comparison would be vacuous too.
      const faces = new Uint32Array(numFaces * 3);
      for (let f = 0; f < numFaces; f++) {
        faces[f * 3] = f % numVertices;
        faces[f * 3 + 1] = (f + 1) % numVertices;
        faces[f * 3 + 2] = (f + 2) % numVertices;
      }

      const slicePos = new Float32Array([0, 0, 0, 0.5, -0.25, 1.0]);
      // Tolerances measured to leave ~9% of vertices and ~4% of faces visible —
      // a real mix, asserted explicitly below.
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 1.2, 1.2, 1.2]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsMask = new Uint8Array(numVertices);
      const wasmMask = new Uint8Array(numVertices);
      const tsVisible = tsModule.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        ndim,
        numVertices,
        tsMask
      );
      const wasmVisible = wasmModule!.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        ndim,
        numVertices,
        wasmMask
      );

      expect(wasmVisible).toBe(tsVisible);
      expect(arraysEqual(tsMask, wasmMask)).toBe(true);
      // Guard against a vacuous comparison: a mask that is all-0 or all-1 would
      // match trivially and prove nothing about the boundary logic.
      expect(tsVisible).toBeGreaterThan(0);
      expect(tsVisible).toBeLessThan(numVertices);

      const tsFaces = new Uint32Array(numFaces * 3);
      const wasmFaces = new Uint32Array(numFaces * 3);
      const tsKept = tsModule.compact_visible_faces(faces, tsMask, numFaces, tsFaces);
      const wasmKept = wasmModule!.compact_visible_faces(faces, wasmMask, numFaces, wasmFaces);

      expect(wasmKept).toBe(tsKept);
      expect(tsKept).toBeGreaterThan(0);
      expect(tsKept).toBeLessThan(numFaces);
      expect(
        arraysEqual(tsFaces.subarray(0, tsKept * 3), wasmFaces.subarray(0, wasmKept * 3))
      ).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('agrees on non-finite coordinates and infinite tolerance', () => {
      // Row per vertex: NaN / +Inf / -Inf on the hidden dim, then a finite one.
      // prettier-ignore
      const positions = new Float32Array([
        0, 0, 0, NaN,
        0, 0, 0, Infinity,
        0, 0, 0, -Infinity,
        0, 0, 0, 5,
      ]);
      const slicePos = new Float32Array([0, 0, 0, 5]);
      const displayDims = new Uint32Array([0, 1, 2]);

      // Both a finite EXTEND_TO_ALL sentinel and a genuinely infinite tolerance:
      // the latter makes sliceMax === +Infinity, where only an explicit
      // finite-check keeps an infinite coordinate culled.
      for (const tol of [1e10, Infinity]) {
        const tolerance = new Float32Array([1e10, 1e10, 1e10, tol]);
        const tsMask = new Uint8Array(4);
        const wasmMask = new Uint8Array(4);
        const tsVisible = tsModule.mesh_vertex_visibility_mask(
          positions,
          slicePos,
          tolerance,
          displayDims,
          4,
          4,
          tsMask
        );
        const wasmVisible = wasmModule!.mesh_vertex_visibility_mask(
          positions,
          slicePos,
          tolerance,
          displayDims,
          4,
          4,
          wasmMask
        );

        expect(wasmVisible).toBe(tsVisible);
        expect(arraysEqual(tsMask, wasmMask)).toBe(true);
        // Only the finite vertex survives, under either tolerance.
        expect(Array.from(wasmMask)).toEqual([0, 0, 0, 1]);
      }
    });

    it.skipIf(!wasmFilesExist)('agrees that the slab bounds are inclusive', () => {
      // prettier-ignore
      const positions = new Float32Array([
        0, 0, 0, 4.5, // exactly slice_min
        0, 0, 0, 5.5, // exactly slice_max
      ]);
      const slicePos = new Float32Array([0, 0, 0, 5]);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsMask = new Uint8Array(2);
      const wasmMask = new Uint8Array(2);
      const tsVisible = tsModule.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        4,
        2,
        tsMask
      );
      const wasmVisible = wasmModule!.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        4,
        2,
        wasmMask
      );

      expect(wasmVisible).toBe(2);
      expect(tsVisible).toBe(wasmVisible);
      expect(arraysEqual(tsMask, wasmMask)).toBe(true);
    });

    /**
     * Out-of-range face indices come from the STORE, so both backends must drop
     * the face. Without the range guard the WASM build would trap (the crate is
     * `panic = "abort"`, so the trap escapes as an uncatchable
     * `RuntimeError: unreachable`) while TS would read `undefined` — this test
     * is the only place that difference is observable.
     */
    it.skipIf(!wasmFilesExist)('agrees on out-of-range face indices without trapping', () => {
      const mask = new Uint8Array([1, 1, 1]); // valid indices 0..2
      // The trailing valid face proves the guard SKIPS a bad face rather than
      // stopping: a `break` would discard every valid face after the first bad
      // index, quietly losing most of a corrupt mesh.
      // prettier-ignore
      const faces = new Uint32Array([
        0, 1, 2,          // valid
        0, 1, 3,          // just past the end
        99, 0, 1,         // far past the end
        0, 1, 0xffffffff, // a signed -1 reinterpreted
        2, 1, 0,          // valid, AFTER the bad ones
      ]);
      const numFaces = 5;

      const tsFaces = new Uint32Array(numFaces * 3);
      const wasmFaces = new Uint32Array(numFaces * 3);
      const tsKept = tsModule.compact_visible_faces(faces, mask, numFaces, tsFaces);
      const wasmKept = wasmModule!.compact_visible_faces(faces, mask, numFaces, wasmFaces);

      expect(wasmKept).toBe(2);
      expect(tsKept).toBe(wasmKept);
      expect(arraysEqual(tsFaces.subarray(0, 6), wasmFaces.subarray(0, 6))).toBe(true);
      expect(Array.from(wasmFaces.subarray(0, 6))).toEqual([0, 1, 2, 2, 1, 0]);
    });

    /**
     * A SIGNED or fractional face-index source must land on the same answer in
     * both backends. wasm-bindgen copies `&[u32]` through `Uint32Array.set`, so
     * WASM sees ToUint32 values (`-1` → `0xffffffff`, `1.5` → `1`); the TS
     * reference reproduces that with `>>> 0`. Before it did, `-1` passed the TS
     * upper-bound guard and `vertexMask[-1]` read `undefined` (truthy), emitting a
     * negative-index face in TS and none in WASM — the #806 divergence pattern.
     */
    it.skipIf(!wasmFilesExist)('agrees on signed and fractional face indices', () => {
      const mask = new Uint8Array([1, 1, 1]);
      const cases: unknown[] = [
        new Int32Array([0, 1, -1]),
        new Int32Array([-1, -2, -3]),
        [0, 1, -1],
        [0.9, 1.2, 2.7],
        new Uint32Array([0, 1, 2]), // canonical path must be unaffected
      ];
      for (const faces of cases) {
        const tsOut = new Uint32Array(3);
        const wasmOut = new Uint32Array(3);
        type Compact = (f: unknown, m: Uint8Array, n: number, o: Uint32Array) => number;
        const tsKept = (tsModule.compact_visible_faces as unknown as Compact)(
          faces,
          mask,
          1,
          tsOut
        );
        const wasmKept = (wasmModule!.compact_visible_faces as unknown as Compact)(
          faces,
          mask,
          1,
          wasmOut
        );

        expect(tsKept).toBe(wasmKept);
        expect(arraysEqual(tsOut.subarray(0, tsKept * 3), wasmOut.subarray(0, wasmKept * 3))).toBe(
          true
        );
      }
    });

    it.skipIf(!wasmFilesExist)('agrees on the no-hidden-dims fast path', () => {
      const positions = new Float32Array([0, 0, 0, 1, 2, 3, -5, 9, 100]);
      const slicePos = new Float32Array([50, 50, 50]);
      const tolerance = new Float32Array([0, 0, 0]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsMask = new Uint8Array(3);
      const wasmMask = new Uint8Array(3);
      const tsVisible = tsModule.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        3,
        3,
        tsMask
      );
      const wasmVisible = wasmModule!.mesh_vertex_visibility_mask(
        positions,
        slicePos,
        tolerance,
        displayDims,
        3,
        3,
        wasmMask
      );

      expect(wasmVisible).toBe(3);
      expect(tsVisible).toBe(wasmVisible);
      expect(arraysEqual(tsMask, wasmMask)).toBe(true);
    });
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
          cols,
          new Uint32Array(0)
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
