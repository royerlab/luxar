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

import { readFileSync } from 'node:fs';
import { describe, it, expect, beforeAll } from 'vitest';
import { TypeScriptFallback } from '../../../wasm/typescript';
import { ArrayDecoder } from '../../../data/array-decoder/decoder';
import type { WasmModule } from '../../../wasm/types';
import {
  arraysEqual as sharedArraysEqual,
  arraysAlmostEqual as sharedArraysAlmostEqual,
  exactBitsEqual,
} from '../../helpers/array-compare';
import { mulberry32 } from '../../helpers/random';
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

const RUST_DECODE_SOURCE_URL = new URL('../../../wasm/rust/src/decode.rs', import.meta.url);
const TYPESCRIPT_DECODE_SOURCE_URL = new URL('../../../wasm/typescript/decode.ts', import.meta.url);

function exportedDecodeNames(source: string, language: 'rust' | 'typescript'): string[] {
  const pattern =
    language === 'rust'
      ? /^pub fn (decode_[a-z0-9_]+)/gm
      : /^export function (decode_[a-z0-9_]+)/gm;
  return [...source.matchAll(pattern)].map((match) => match[1]).sort();
}

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
 *
 * SUPERSEDED BY #1820 for the gsplats/effective-radii kernels. The premise of
 * the note above — that a multi-step algorithm must accumulate error, so the
 * tolerance has to grow with ndim — was a symptom of the TS reference doing its
 * arithmetic in f64 while Rust did it in f32. With the operation order matched,
 * `mahalanobis_distance` is BIT-EXACT against WASM at ndim 8 over 5000
 * randomized correlated factors (see the `f32 operation-order parity (#1820)`
 * block below), so no epsilon is needed there at all. The legacy cases that
 * assert the scaled epsilon are kept — a scaled epsilon is still a valid upper
 * bound, and they cover shapes the exact cases do not — but a NEW case in these
 * two kernels should assert exactness, not a scaled tolerance.
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

/** Inputs for {@link runFused}. */
interface FusedArgs {
  positions: Float32Array;
  cholesky: Float32Array;
  amplitudes: Float32Array;
  /** Defaults to an all-ones `(splatCount, colorComponents)` block. */
  colors?: Float32Array;
  discreteVisibility: Uint8Array;
  slicePosition: Float32Array;
  continuousHiddenDims: Uint32Array;
  displayDims: Uint32Array;
  ndim: number;
  splatCount: number;
  colorComponents?: number;
  /** Ask both backends to record the surviving source indices (#1423). */
  recordSourceIndices?: boolean;
  /** Rust `min_amplitude: f32`. */
  minAmplitude?: number;
  /** Rust `truncate: f32` — the per-dataset truncation radius. */
  truncate?: number;
}

/**
 * Fused single-call projection (W5), driven identically on either backend.
 *
 * The fused kernel writes COMPACTED outputs, so callers compare the dense
 * prefix `[0, count * stride)` only.
 *
 * `minAmplitude` / `truncate` default to the historical 1e-6 / 3.0 the original
 * W5 cases were written against, so those cases read unchanged; the #1820
 * parity block passes them explicitly because both are `f32` parameters whose
 * exact value is part of what is under test.
 */
function runFused(
  mod: WasmModule,
  args: FusedArgs
): {
  count: number;
  centers: Float32Array;
  chol: Float32Array;
  amps: Float32Array;
  cols: Float32Array;
  src: Uint32Array;
} {
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
    args.colors ?? new Float32Array(n * k).fill(1),
    args.discreteVisibility,
    args.slicePosition,
    args.continuousHiddenDims,
    args.displayDims,
    args.ndim,
    n,
    k,
    args.minAmplitude ?? 1e-6,
    args.truncate ?? 3.0,
    centers,
    chol,
    amps,
    cols,
    src
  );
  return { count, centers, chol, amps, cols, src };
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

describe('decode kernel source parity', () => {
  it('keeps Rust and TypeScript decode exports in sync', () => {
    const rustNames = exportedDecodeNames(readFileSync(RUST_DECODE_SOURCE_URL, 'utf8'), 'rust');
    const typescriptNames = exportedDecodeNames(
      readFileSync(TYPESCRIPT_DECODE_SOURCE_URL, 'utf8'),
      'typescript'
    );

    expect(rustNames, 'The Rust export scan must find the established decode kernels').toContain(
      'decode_quantized_u8'
    );
    expect(
      rustNames,
      'Rust and TypeScript decode_* exports must match; TypeScript is the production fallback'
    ).toEqual(typescriptNames);
  });
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
      expect(Array.from(tsVis)).toEqual(Array.from(wVis));
      expect(Array.from(tsT1)).toEqual(Array.from(wT1));
      expect(Array.from(tsT2)).toEqual(Array.from(wT2));
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
    /**
     * Every case runs with a non-trivial `shardCount` so the per-shard AABB
     * outputs (cross-node depth ordering) are covered by the SAME inputs that
     * already stress the ordering — behind-camera, duplicate depths, the
     * degenerate identity fallback, NaN/±Inf, denormals, and 100k random.
     *
     * Bounds parity is asserted EXACTLY (`exactBitsEqual`), not almost-equal.
     * That is not a stylistic choice: the boxes are pure min/max over the input
     * f32s, so no rounding is involved anywhere and the two backends must agree
     * bit-for-bit. A tolerance here would hide the one divergence that can
     * actually occur — `Math.min(NaN, x) === NaN` vs Rust's
     * `f32::min(NaN, x) === x`, which is why the twin uses plain comparisons.
     */
    const PARITY_SHARDS = 3;

    function runBoth(
      centers3: Float32Array,
      modelView: Float32Array,
      count: number,
      shardCount: number = PARITY_SHARDS
    ): {
      ts: Uint32Array;
      wasm: Uint32Array;
      tsSorted: number;
      wasmSorted: number;
      tsMin: Float32Array;
      tsMax: Float32Array;
      wasmMin: Float32Array;
      wasmMax: Float32Array;
      tsZMin: Float32Array;
      tsZMax: Float32Array;
    } {
      const ts = new Uint32Array(count);
      const wasm = new Uint32Array(count);
      const tsMin = new Float32Array(shardCount * 3);
      const tsMax = new Float32Array(shardCount * 3);
      const wasmMin = new Float32Array(shardCount * 3);
      const wasmMax = new Float32Array(shardCount * 3);
      const tsZMin = new Float32Array(shardCount);
      const tsZMax = new Float32Array(shardCount);
      const wasmZMin = new Float32Array(shardCount);
      const wasmZMax = new Float32Array(shardCount);
      const tsSorted = tsModule.sort_splats_by_depth(
        centers3,
        modelView,
        ts,
        count,
        shardCount,
        tsMin,
        tsMax,
        tsZMin,
        tsZMax
      );
      const wasmSorted = wasmModule!.sort_splats_by_depth(
        centers3,
        modelView,
        wasm,
        count,
        shardCount,
        wasmMin,
        wasmMax,
        wasmZMin,
        wasmZMax
      );
      // Asserted inside the harness so no case can forget it, and so a new
      // ordering case automatically extends bounds coverage too.
      expect(
        exactBitsEqual(tsMin, wasmMin),
        `shard bounds min diverged: ts=${Array.from(tsMin)} wasm=${Array.from(wasmMin)}`
      ).toBe(true);
      expect(
        exactBitsEqual(tsMax, wasmMax),
        `shard bounds max diverged: ts=${Array.from(tsMax)} wasm=${Array.from(wasmMax)}`
      ).toBe(true);
      // The MERGE KEY. Same exact-bits bar: it is min/max over the pass-1 view-z
      // scratch, so no rounding is involved and the two backends must agree
      // bit-for-bit.
      expect(
        exactBitsEqual(tsZMin, wasmZMin),
        `shard view-z min diverged: ts=${Array.from(tsZMin)} wasm=${Array.from(wasmZMin)}`
      ).toBe(true);
      expect(
        exactBitsEqual(tsZMax, wasmZMax),
        `shard view-z max diverged: ts=${Array.from(tsZMax)} wasm=${Array.from(wasmZMax)}`
      ).toBe(true);
      return {
        ts,
        wasm,
        tsSorted,
        wasmSorted,
        tsMin,
        tsMax,
        wasmMin,
        wasmMax,
        tsZMin,
        tsZMax,
      };
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

    it.skipIf(!wasmFilesExist)('shardCount 0 leaves both backends silent', () => {
      // The unsharded production path: neither backend may touch the outputs.
      const centers3 = new Float32Array([0, 0, -1, 1, 0, -2, 2, 0, -3]);
      const ts = new Uint32Array(3);
      const wasm = new Uint32Array(3);
      const tsMin = new Float32Array([7, 7, 7]);
      const tsMax = new Float32Array([9, 9, 9]);
      const wasmMin = new Float32Array([7, 7, 7]);
      const wasmMax = new Float32Array([9, 9, 9]);
      const untouched = new Float32Array([5, 5, 5]);
      const untouchedW = new Float32Array([5, 5, 5]);
      tsModule.sort_splats_by_depth(
        centers3,
        IDENTITY_MV,
        ts,
        3,
        0,
        tsMin,
        tsMax,
        untouched,
        untouched
      );
      wasmModule!.sort_splats_by_depth(
        centers3,
        IDENTITY_MV,
        wasm,
        3,
        0,
        wasmMin,
        wasmMax,
        untouchedW,
        untouchedW
      );
      expect(Array.from(untouched)).toEqual([5, 5, 5]);
      expect(Array.from(untouchedW)).toEqual([5, 5, 5]);
      expect(exactBitsEqual(tsMin, new Float32Array([7, 7, 7]))).toBe(true);
      expect(exactBitsEqual(tsMax, new Float32Array([9, 9, 9]))).toBe(true);
      expect(exactBitsEqual(wasmMin, new Float32Array([7, 7, 7]))).toBe(true);
      expect(exactBitsEqual(wasmMax, new Float32Array([9, 9, 9]))).toBe(true);
    });

    it.skipIf(!wasmFilesExist)('shard bounds agree with a shard count of 1', () => {
      // S == 1 must be the whole-node AABB on both backends — the self-check
      // that makes the otherwise-inert output non-vacuous.
      const centers3 = new Float32Array([
        3,
        -2,
        -1, //
        -5,
        7,
        -9, //
        1,
        0,
        -4,
      ]);
      const { tsMin, tsMax } = runBoth(centers3, IDENTITY_MV, 3, 1);
      expect(Array.from(tsMin)).toEqual([-5, -2, -9]);
      expect(Array.from(tsMax)).toEqual([3, 7, -1]);
    });

    it.skipIf(!wasmFilesExist)('shard bounds agree when NaN and Inf centers are mixed in', () => {
      // The one place the twins could diverge: `Math.min(NaN, x)` is NaN while
      // Rust's `f32::min(NaN, x)` is x, so the TS twin must use plain
      // comparisons. NaN is excluded from a box; ±Inf propagates into it.
      const centers3 = new Float32Array([
        NaN,
        0,
        -1, //
        Infinity,
        0,
        -2, //
        1,
        -Infinity,
        -3, //
        2,
        0,
        NaN, //
        3,
        4,
        -5, //
        -6,
        7,
        -8,
      ]);
      // Exercised at several shard counts so the NaN/Inf elements land in
      // different shards, including alone.
      for (const shards of [1, 2, 3, 6]) {
        const { ts, wasm, tsSorted, wasmSorted } = runBoth(centers3, IDENTITY_MV, 6, shards);
        expect(wasmSorted).toBe(tsSorted);
        expect(arraysEqual(wasm, ts)).toBe(true);
      }
    });

    it.skipIf(!wasmFilesExist)(
      'shard bounds agree when the shard count exceeds the element count',
      () => {
        // Surplus shards must reach the SAME empty sentinel on both backends,
        // not merely "some non-finite value".
        const centers3 = new Float32Array([0, 0, -1, 1, 0, -2]);
        const { tsMin, tsMax } = runBoth(centers3, IDENTITY_MV, 2, 5);
        expect(tsMin[6]).toBe(Infinity);
        expect(tsMax[6]).toBe(-Infinity);
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

    it.skipIf(!wasmFilesExist).each([
      { minVal: -12.3456789012345, maxVal: 98.7654321098765 },
      { minVal: 2.6769986296248494e-5, maxVal: 6.059071789834555 },
      { minVal: -9.391529450949973e-5, maxVal: 37.21074561637171 },
    ])(
      'decode_quantized_u16 should exactly match for bounds [$minVal, $maxVal]',
      ({ minVal, maxVal }) => {
        const data = Uint16Array.from({ length: 65536 }, (_, code) => code);
        const tsOutput = new Float32Array(data.length);
        const wasmOutput = new Float32Array(data.length);

        tsModule.decode_quantized_u16(data, minVal, maxVal, tsOutput);
        wasmModule!.decode_quantized_u16(data, minVal, maxVal, wasmOutput);

        expect(exactBitsEqual(tsOutput, wasmOutput)).toBe(true);
      }
    );

    it.skipIf(!wasmFilesExist).each([
      { minVal: -12.3456789012345, maxVal: 98.7654321098765 },
      { minVal: 2.6769986296248494e-5, maxVal: 6.059071789834555 },
      { minVal: -9.391529450949973e-5, maxVal: 37.21074561637171 },
      { minVal: -9860308715.14231, maxVal: 33712563323.788345 },
    ])(
      'decode_quantized_u8 should exactly match for bounds [$minVal, $maxVal]',
      ({ minVal, maxVal }) => {
        const data = Uint8Array.from({ length: 256 }, (_, code) => code);
        const tsOutput = new Float32Array(data.length);
        const wasmOutput = new Float32Array(data.length);

        tsModule.decode_quantized_u8(data, minVal, maxVal, tsOutput);
        wasmModule!.decode_quantized_u8(data, minVal, maxVal, wasmOutput);

        expect(exactBitsEqual(tsOutput, wasmOutput)).toBe(true);
      }
    );

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

    it
      .skipIf(!wasmFilesExist)
      .each(
        ['u8', 'u16']
          .flatMap((label) =>
            [Math.log(2), Math.log(10), Math.log(1e5)].map((maxLog) => ({ label, maxLog }))
          )
          .concat({ label: 'u8', maxLog: 1.3732879469562715 })
      )(
      'decode_log_scalar_$label should exactly match across every code at maxLog=$maxLog',
      ({ label, maxLog }) => {
        const data =
          label === 'u8'
            ? Uint8Array.from({ length: 256 }, (_, code) => code)
            : Uint16Array.from({ length: 65536 }, (_, code) => code);
        const tsOutput = new Float32Array(data.length);
        const wasmOutput = new Float32Array(data.length);

        if (label === 'u8') {
          tsModule.decode_log_scalar_u8(data as Uint8Array, maxLog, tsOutput);
          wasmModule!.decode_log_scalar_u8(data as Uint8Array, maxLog, wasmOutput);
        } else {
          tsModule.decode_log_scalar_u16(data as Uint16Array, maxLog, tsOutput);
          wasmModule!.decode_log_scalar_u16(data as Uint16Array, maxLog, wasmOutput);
        }

        expect(exactBitsEqual(tsOutput, wasmOutput)).toBe(true);
      }
    );

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

    // For these exhaustive bounds, the closest f64 result is 237 ULPs from a
    // float32 rounding boundary, so a one-ULP libm difference cannot change bits.
    it.skipIf(!wasmFilesExist)(
      'decode_geolog_scalar_u8 should exactly match for every code',
      () => {
        const data = Uint8Array.from({ length: 256 }, (_, code) => code);
        const tsOutput = new Float32Array(data.length);
        const wasmOutput = new Float32Array(data.length);
        const minLog = Math.log(1e-3);
        const maxLog = Math.log(1e3);

        tsModule.decode_geolog_scalar_u8(data, minLog, maxLog, tsOutput);
        wasmModule!.decode_geolog_scalar_u8(data, minLog, maxLog, wasmOutput);

        expect(exactBitsEqual(tsOutput, wasmOutput)).toBe(true);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'decode_geolog_scalar_u16 should exactly match for every code',
      () => {
        const data = Uint16Array.from({ length: 65536 }, (_, code) => code);
        const tsOutput = new Float32Array(data.length);
        const wasmOutput = new Float32Array(data.length);
        const minLog = Math.log(0.5);
        const maxLog = Math.log(12345.6789);

        tsModule.decode_geolog_scalar_u16(data, minLog, maxLog, tsOutput);
        wasmModule!.decode_geolog_scalar_u16(data, minLog, maxLog, wasmOutput);

        expect(exactBitsEqual(tsOutput, wasmOutput)).toBe(true);
      }
    );

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
    //
    // SUPERSEDED BY #1820, kept as a legacy upper bound. The "rounding
    // accumulates per dimension, so scale the epsilon" premise held only while
    // the TS reference accumulated in f64 and Rust in f32; once the operation
    // order matches, this kernel is BIT-EXACT at ndim 8 over 5000 randomized
    // correlated factors (see `mahalanobis_distance is BIT-EXACT over 5k
    // randomized correlated factors` below). A sqrt(ndim)-scaled epsilon is
    // still a true bound, so these cases stay; do not copy the pattern into a
    // new gsplats/effective-radii case, and do not read this comment as
    // evidence that exactness is unreachable at high ndim.
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

    // `runFused` (the shared W5 driver) is at module scope — see its doc
    // comment. These cases rely on its historical 1e-6 / 3.0 defaults.

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
  //
  // HOW TO READ THE NUMBERS. Every figure quoted below was measured against the
  // BUILT WASM artifact on the fixture it is attached to, and is written as
  // "<now> (was <then>)", where "then" is the same fixture with every
  // `Math.fround` stripped from the two kernels and `F32_MIN_POSITIVE` restored
  // to `Number.MIN_VALUE` — i.e. the pre-#1820 tree.
  //
  // WHY SEVERAL CASES ASSERT A COUNT OF DIFFERING ELEMENTS. A ulp/absolute bound
  // that a PARTIAL revert still satisfies guards nothing: the phantom diagonal,
  // for instance, is within 2 ulp both before and after this change. Where a
  // bound cannot separate the two, the number of elements that differ from WASM
  // can, so it is asserted too. Those thresholds are deliberately tight and each
  // one names the rounding it exists to catch; re-derive them (do not simply
  // relax them) if a V8 or libm upgrade moves the baseline.
  //
  // MUTATION COVERAGE. The cases below were validated by deleting each of the
  // 38 executable `Math.fround` calls in `effective-radii.ts` /
  // `gsplats-processing.ts` one at a time and re-running `src/tests/unit/wasm/`:
  // 32 of 38 turn a case red. The 6 survivors are provably inert and documented at their
  // definitions — `fround(-0.5 · x)` twice (halving an f32 is exact),
  // `fround(CHOLESKY_EPSILON)` and `fround(sqrt(CHOLESKY_EPSILON_F32))` (both
  // consumed only through a benign double-rounded sqrt), and the two phantom
  // roundings that are no-ops for the reachable `counted` ∈ {1, 2}. If you add
  // a rounding to either kernel, add the case that kills it.
  describe('f32 operation-order parity (#1820)', () => {
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
      // Iterate the LONGER side: a length mismatch must be reported, not
      // silently truncated away by looping over `ts.length` alone.
      const n = Math.max(ts.length, wasm.length);
      for (let i = 0; i < n; i++) {
        if (i >= ts.length || i >= wasm.length || !Object.is(ts[i], wasm[i])) {
          total++;
          if (first.length < limit) first.push({ i, ts: ts[i], wasm: wasm[i] });
        }
      }
      return { total, first };
    }
    const NO_MISMATCHES = { total: 0, first: [] };

    /** Number of positions where the two backends do not agree bit-for-bit. */
    function differingCount(ts: ArrayLike<number>, wasm: ArrayLike<number>): number {
      return exactMismatches(ts, wasm, 0).total;
    }

    /**
     * Distance in f32 ulps between two values (both are already f32).
     *
     * Monotone ACROSS the sign boundary: negatives map to their negated
     * magnitude and ±0 map to the same key. The more common
     * `bits < 0 ? 0x80000000 - bits : bits` form is only monotone within one
     * sign in JS — it sends -0 to 2³², so any sign crossing reports ~4.29e9.
     */
    function ulpDistance(a: number, b: number): number {
      const f = new Float32Array([a, b]);
      const u = new Uint32Array(f.buffer);
      const key = (x: number) => (x & 0x80000000 ? -(x & 0x7fffffff) : x);
      return Math.abs(key(u[0]) - key(u[1]));
    }

    function maxUlp(ts: ArrayLike<number>, wasm: ArrayLike<number>): number {
      let m = 0;
      const n = Math.max(ts.length, wasm.length);
      for (let i = 0; i < n; i++) {
        if (i >= ts.length || i >= wasm.length) return Infinity;
        m = Math.max(m, ulpDistance(ts[i], wasm[i]));
      }
      return m;
    }

    function maxAbs(ts: ArrayLike<number>, wasm: ArrayLike<number>): number {
      let m = 0;
      const n = Math.max(ts.length, wasm.length);
      for (let i = 0; i < n; i++) {
        if (i >= ts.length || i >= wasm.length) return Infinity;
        m = Math.max(m, Math.abs(ts[i] - wasm[i]));
      }
      return m;
    }

    it('metric helpers reject a length mismatch on either side', () => {
      const shorter = new Float32Array([1]);
      const longer = new Float32Array([1, 2]);

      expect(maxUlp(shorter, longer)).toBe(Infinity);
      expect(maxUlp(longer, shorter)).toBe(Infinity);
      expect(maxAbs(shorter, longer)).toBe(Infinity);
      expect(maxAbs(longer, shorter)).toBe(Infinity);
    });

    /** Independent recomputation of `invOneMinusC` for a truncation radius. */
    function invOneMinusC(truncate: number): number {
      const t = Math.fround(truncate);
      const shiftC = Math.fround(Math.exp(Math.fround(Math.fround(-0.5 * t) * t)));
      return Math.fround(1.0 / Math.fround(1.0 - shiftC));
    }

    // A slice position that is NOT all zeros, and whose components are not
    // f32-friendly. This matters: with `target === 0`, `value - target` is
    // exact, so `Math.fround` on it is literally a no-op and the sweeps below
    // would pass with that rounding deleted (measured: 0/20000 outputs change).
    // With these values, deleting the same rounding moves 212/20000 outputs by
    // up to 2527 ulp.
    const EFFECTIVE_RADII_SLICE = [
      0, 0, 0, 0.7913131713867188, -1.3313131332397461, 0.11313131079077721,
    ] as const;
    /** Same idea for the fused kernel's `diff[] = slicePosition[d] - center[d]`. */
    const FUSED_SLICE = [0, 0, 0, 0.7913131713867188, -1.3313131332397461] as const;

    // ------------------------------------------------------------------
    // effective_radii
    // ------------------------------------------------------------------
    it.skipIf(!wasmFilesExist)(
      'calculate_effective_radii is BIT-EXACT over a 20k randomized nD sweep',
      () => {
        // Every step of this kernel is a plain arithmetic op — no
        // transcendentals — so exactness is fully reachable and anything less
        // is a bug. Mixed spatial/discrete hidden dims exercise both the
        // Pythagorean accumulation and the tolerance gate, and the non-zero
        // slice position exercises the subtraction that feeds both.
        // Measured: 0/20000 differing (was 633/20000, up to 1278 ulp).
        const numPoints = 20000;
        const ndim = 6;
        const positions = new Float32Array(numPoints * ndim);
        const radii = new Float32Array(numPoints);
        const rnd = mulberry32(0x1820);
        for (let i = 0; i < numPoints; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = (rnd() - 0.5) * 4;
          radii[i] = 0.5 + rnd() * 2;
        }
        const displayDims = new Uint32Array([0, 1, 2]);
        const slicePos = new Float32Array(EFFECTIVE_RADII_SLICE);
        expect(slicePos.some((v) => v !== 0)).toBe(true); // the premise
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
        expect(wasmVisible).toBeGreaterThan(1000); // the fixture is not degenerate
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
        // Rust's f32, not merely within sqrt(ndim)*1e-5 of it. This is also the
        // case that disproves the file header's old "multi-step algorithms need
        // a sqrt(ndim)-scaled epsilon" note: at ndim 8, 0/5000 differ (was
        // 5000/5000, up to 3 ulp).
        const trials = 5000;
        const ndim = 8;
        const packedSize = (ndim * (ndim + 1)) / 2;
        const rnd = mulberry32(0xc0ffee);
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

    it.skipIf(!wasmFilesExist)(
      'mahalanobis_distance: the degenerate-pivot epsilon is compared as an f32',
      () => {
        // `mahalanobis_distance_internal` has a function-local
        // `const EPSILON: f32 = 1e-10`, and f32(1e-10) is 1.000000013351432e-10
        // — strictly GREATER than the f64 literal. A pivot sitting exactly on
        // that f32 therefore fails Rust's `diag > EPSILON` (the axis is treated
        // as degenerate, y[i] = 0) but passes an f64 `diag > 1e-10`, which
        // divides by 1e-10 instead and returns a distance ~1e10 times too big.
        // Nothing about this is an ulp: it is the whole answer.
        const eps32 = Math.fround(1e-10);
        expect(eps32 > 1e-10).toBe(true); // the premise
        const diff = new Float32Array([1, 1]);
        const packedL = new Float32Array([1, 0, eps32]); // L00 = 1, L10 = 0, L11 = eps32

        const tsDist = tsModule.mahalanobis_distance(diff, packedL, 2);
        const wasmDist = wasmModule!.mahalanobis_distance(diff, packedL, 2);

        expect(wasmDist).toBe(1); // second component zeroed => ||y|| = |y0| = 1
        expect(tsDist).toBe(wasmDist);
      }
    );

    // ------------------------------------------------------------------
    // project_gsplats_nd_to_3d
    // ------------------------------------------------------------------
    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d is BIT-EXACT with no continuous hidden dims (no exp on the path)',
      () => {
        // Attenuation short-circuits to exactly 1.0, so the only float work is
        // the display marginal Cholesky — Σ_S dot products and a Crout
        // reduction, both pure arithmetic. Full exactness is reachable, and the
        // Crout step is where cancellation used to blow the f64-vs-f32
        // divergence out to thousands of ulps: 0/120000 packed slots differ
        // (was 12495/120000, up to 3602 ulp).
        //
        // `slicePosition` is left all zeros here ON PURPOSE, unlike the sweeps
        // that carry `FUSED_SLICE`: with every dim displayed and no continuous
        // hidden dims, the kernel never reads it, so a non-zero value would add
        // noise to the fixture without exercising anything.
        const splatCount = 20000;
        const ndim = 3;
        const packedSize = 6;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = mulberry32(0xbeef);
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
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

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
        // between them: WASM emits the splat, an f64 TS chain culls it. The
        // driving mechanism here is the Σ_S dot-product accumulator, NOT the
        // final `amplitude × attenuation` product — that one has its own case
        // ('the amplitude × attenuation PRODUCT ...') below.
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
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

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
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        expect(ts.amps[0]).toBe(w.amps[0]);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d: truncate is narrowed to an f32 before the shift is built',
      () => {
        // The sibling of the `minAmplitude` case: `truncate: f32` is likewise
        // narrowed by wasm-bindgen and was likewise consumed raw here. Every
        // other fixture in this file passes truncate = 3.0, which IS f32-exact,
        // so the narrowing could never fire.
        //
        // The narrowing is only observable at small truncate. `shiftC` is built
        // as `exp(fround(fround(-0.5·t)·t))`, and for t near 3 the perturbation
        // a sub-half-ulp `t` can inject into that product (≤1.8e-7) is smaller
        // than half an ulp of the product itself (2.4e-7), so the rounding
        // absorbs it before `exp` ever sees it. At t ≈ 1.00001 it does not:
        // shiftC moves by an ulp and `invOneMinusC` (2.5417 vs 2.5415) moves
        // with it, which is enough to change whether a splat sitting on the
        // truncation shell clears `minAmplitude`.
        const truncate = 1.0000100731267594;
        const truncateF32 = 1.0000100135803223;
        expect(Math.fround(truncate)).toBe(truncateF32); // the premise
        expect(truncate).not.toBe(truncateF32);

        // One splat at mahalanobis distance ≈ 0.9 with an identity 4D factor —
        // deep enough into the shifted Gaussian that the two shifts separate.
        const packed = new Float32Array(10);
        let p = 0;
        for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) packed[p++] = r === c ? 1 : 0;
        const base = {
          positions: new Float32Array([0, 0, 0, 0.8999999761581421]),
          cholesky: packed,
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(4),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim: 4,
          splatCount: 1,
          truncate,
        };
        // Attenuation with `truncate` narrowed (what WASM computes) vs left raw
        // (what an un-narrowed TS backend computes). The two straddle the gate.
        const narrowed = 0.15363658964633942;
        const raw = 0.15363672375679016;
        expect(raw).toBeGreaterThan(narrowed);

        // minAmplitude == the narrowed attenuation: emitted, and the emitted
        // amplitude IS that value.
        const at = runFused(tsModule, { ...base, minAmplitude: narrowed });
        const aw = runFused(wasmModule!, { ...base, minAmplitude: narrowed });
        expect(aw.count).toBe(1);
        expect(at.count).toBe(aw.count);
        expect(aw.amps[0]).toBe(narrowed);
        expect(at.amps[0]).toBe(aw.amps[0]);

        // minAmplitude == the un-narrowed attenuation: culled on both backends.
        // An un-narrowed TS backend reaches exactly this value and emits.
        const bt = runFused(tsModule, { ...base, minAmplitude: raw });
        const bw = runFused(wasmModule!, { ...base, minAmplitude: raw });
        expect(bw.count).toBe(0);
        expect(bt.count).toBe(bw.count);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d: the amplitude × attenuation PRODUCT is rounded before the gate',
      () => {
        // The rounding this change headlines, and the only one on the fused
        // path that no sweep can reach: `attenuatedAmplitude` is compared
        // against `minAmplitude` and then WRITTEN OUT, so a sweep sees the
        // written f32 either way. It is only visible when the f32 product lands
        // exactly ON the gate while the f64 product sits strictly below it.
        //
        // truncate = 20 makes `shiftC` underflow to 0 and `invOneMinusC`
        // exactly 1, so attenuation == rawExp (an exact f32) and this is the
        // only rounding left in the chain.
        const atten = 0.7261490225791931;
        const amp = 0.5001000165939331;
        const minAmplitude = 0.36314713954925537;
        // The premise, restated as arithmetic: the f32 product IS the gate and
        // the f64 product is strictly under it.
        expect(Math.fround(amp * atten)).toBe(minAmplitude);
        expect(amp * atten).toBeLessThan(minAmplitude);

        const packed = new Float32Array(10);
        let p = 0;
        for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) packed[p++] = r === c ? 1 : 0;
        const args = {
          positions: new Float32Array([0, 0, 0, 0.800000011920929]),
          cholesky: packed,
          amplitudes: new Float32Array([amp]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(4),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim: 4,
          splatCount: 1,
          minAmplitude,
          truncate: 20,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        // Pin the intermediate too, so a fixture that drifts fails loudly
        // instead of quietly testing nothing.
        expect(
          runFused(wasmModule!, { ...args, amplitudes: new Float32Array([1]), minAmplitude: 0 })
            .amps[0]
        ).toBe(atten);
        expect(w.count).toBe(1); // f32 product == the gate => `!(x < min)` => emitted
        expect(ts.count).toBe(w.count);
        expect(ts.amps[0]).toBe(w.amps[0]);
      }
    );

    it.skipIf(!wasmFilesExist).each([
      [0.2007473260164261, 0.10000000149011612, 0.7499746084213257],
      [1.177423357963562, 0.5, 0.7649974226951599],
    ] as const)(
      'project_gsplats_nd_to_3d: the shiftC / invOneMinusC chain is rounded step by step (truncate %f)',
      (truncate, hiddenOffset, expectedAmplitude) => {
        // `invOneMinusC` multiplies EVERY splat's attenuation, so an ulp lost
        // building it is an ulp lost everywhere — but at the truncate the
        // sweeps use it is unreachable, because for shiftC ≈ 0.011 an ulp of
        // shiftC cannot move `1 − shiftC` at all. These two truncates are
        // chosen (by scanning f32 truncates in [0.2, 4]) so that exactly one of
        // the two roundings flips:
        //   0.2007473… — dropping the rounding on `-0.5·t·t` moves shiftC by an
        //                ulp and `invOneMinusC` from 50.13002 to 50.13018.
        //   1.177423…  — dropping the rounding on `1 − shiftC` (shiftC is
        //                almost exactly 0.5 here, so the subtraction sits on a
        //                binade edge) moves `invOneMinusC` by one ulp.
        // One splat with an identity 4D factor is enough: the emitted amplitude
        // is `1 · invOneMinusC · (rawExp − shiftC)`, so either flip changes it.
        const packed = new Float32Array(10);
        let p = 0;
        for (let r = 0; r < 4; r++) for (let c = 0; c <= r; c++) packed[p++] = r === c ? 1 : 0;
        const args = {
          positions: new Float32Array([0, 0, 0, hiddenOffset]),
          cholesky: packed,
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(4),
          continuousHiddenDims: new Uint32Array([3]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim: 4,
          splatCount: 1,
          minAmplitude: 1e-9,
          truncate,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        expect(w.amps[0]).toBe(expectedAmplitude); // pin the f32 answer
        expect(ts.amps[0]).toBe(w.amps[0]);
      }
    );

    // ------------------------------------------------------------------
    // The four constants Rust types as `f32`
    // ------------------------------------------------------------------
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
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        // L11 is the regularized dead axis: √(f32::MIN_POSITIVE) = 2⁻⁶³.
        expect(w.chol[2]).toBe(Math.pow(2, -63));
        expect(Array.from(ts.chol.subarray(0, 6))).toEqual(Array.from(w.chol.subarray(0, 6)));
      }
    );

    it.skipIf(!wasmFilesExist)(
      'compute_marginal_cholesky: CHOLESKY_RELATIVE_EPSILON multiplies maxDiag as an f32',
      () => {
        // `pub const CHOLESKY_RELATIVE_EPSILON: f32 = 1e-12`, so what actually
        // multiplies maxDiag in Rust is f32(1e-12), not the f64 literal, and
        // the PRODUCT is an f32 too. This L00 is chosen so BOTH roundings are
        // observable: the regularized dead axis differs by an ulp of its own
        // value if either the constant is left as the f64 literal or the
        // product is left unrounded.
        const L00 = 1.0000410079956055;
        const maxDiag = Math.fround(L00 * L00);
        const sqrtF32 = (v: number) => Math.fround(Math.sqrt(v));
        const withF32Eps = sqrtF32(Math.fround(maxDiag * Math.fround(1e-12)));
        const withF64Eps = sqrtF32(Math.fround(maxDiag * 1e-12));
        const withUnroundedProduct = sqrtF32(maxDiag * Math.fround(1e-12));
        expect(withF32Eps).not.toBe(withF64Eps); // the premise, both halves
        expect(withF32Eps).not.toBe(withUnroundedProduct);
        expect(withF32Eps).toBe(1.0000409247368225e-6);

        const ndim = 2;
        const args = {
          positions: new Float32Array([0, 0]),
          cholesky: new Float32Array([L00, 0, 0]), // Σ_S = diag(L00², 0)
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1]),
          ndim,
          splatCount: 1,
          minAmplitude: 1e-9,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        expect(w.chol[2]).toBe(withF32Eps);
        expect(ts.chol[2]).toBe(w.chol[2]);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'compute_marginal_cholesky: an all-zero Σ_S falls back to the ABSOLUTE CHOLESKY_EPSILON floor',
      () => {
        // maxDiag === 0 is the one input for which the relative floor has no
        // scale to anchor to, so the kernel falls back to CHOLESKY_EPSILON and
        // every diagonal is regularized to √(1e-10) = 1e-5. No other fixture in
        // this file constructs an all-zero covariance block, so this branch —
        // and the σ = 1e-5 band the loaders' chunk-fetch epsilon has to cover —
        // was previously unexecuted on either backend.
        //
        // NOTE: this case pins the VALUE, not the `Math.fround` on the
        // constant. That narrowing is provably unobservable —
        // fround(sqrt(fround(1e-10))) and fround(sqrt(1e-10)) are the same f32,
        // and the only other consumer is `sum > degenerateFloor`, which cannot
        // be reached with a positive `sum` when maxDiag is 0. See the constant's
        // doc comment in `gsplats-processing.ts`.
        const regularized = 9.999999747378752e-6;
        expect(Math.fround(Math.sqrt(Math.fround(1e-10)))).toBe(regularized);

        const ndim = 2;
        const args = {
          positions: new Float32Array([0, 0]),
          cholesky: new Float32Array([0, 0, 0]),
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0, 1]),
          ndim,
          splatCount: 1,
          minAmplitude: 1e-9,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        // L00 and L11 both floored; the off-diagonals stay 0.
        expect(Array.from(w.chol.subarray(0, 5))).toEqual([regularized, 0, regularized, 0, 0]);
        expect(Array.from(ts.chol.subarray(0, 5))).toEqual(Array.from(w.chol.subarray(0, 5)));
        // The phantom z diagonal is the geometric mean of the two floored
        // diagonals. Its mathematically equivalent logf/expf evaluation need not
        // reproduce the input bit pattern, but both backends must agree exactly.
        expect(ulpDistance(w.chol[5], regularized)).toBeLessThanOrEqual(8);
        expect(ts.chol[5]).toBe(w.chol[5]);
      }
    );

    // ------------------------------------------------------------------
    // Fewer than three display dims (CLAUDE.md hazard class)
    // ------------------------------------------------------------------
    it.skipIf(!wasmFilesExist)(
      'computeDisplayCholesky3D: a ONE-display-dim scene pads TWO phantom rows',
      () => {
        // `displayDims.length === 1` is the far corner of the <3-display-dims
        // hazard: the padding loop runs for rows 1 AND 2, so the phantom
        // diagonal is written to slots 2 and 5 and three off-diagonals to
        // 1, 3, 4. Nothing else in this file reaches that shape. With a single
        // real diagonal the geometric mean is a plain exp(ln(x)) round-trip,
        // which is where the two libms can still part company.
        const ndim = 2;
        const args = {
          positions: new Float32Array([1, 2]),
          cholesky: new Float32Array([2.5, 0.5, 1.5]),
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([0]),
          ndim,
          splatCount: 1,
          minAmplitude: 1e-9,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        expect(Array.from(w.centers.subarray(0, 3))).toEqual([1, 0, 0]);
        // Marginal over {dim 0} alone is [L00] = 2.5; both phantom diagonals
        // take that same value, and every off-diagonal is 0.
        expect(w.chol[0]).toBe(2.5);
        expect(w.chol[1]).toBe(0);
        expect(w.chol[3]).toBe(0);
        expect(w.chol[4]).toBe(0);
        expect(ulpDistance(w.chol[2], 2.5)).toBeLessThanOrEqual(2);
        expect(w.chol[5]).toBe(w.chol[2]);
        expect(Array.from(ts.chol.subarray(0, 6))).toEqual(Array.from(w.chol.subarray(0, 6)));
      }
    );

    it.skipIf(!wasmFilesExist)(
      'computeDisplayCholesky3D: a ZERO-display-dim scene uses the √CHOLESKY_EPSILON phantom',
      () => {
        // The `counted === 0` fallback. It is unreachable through the padding
        // path for any n ≥ 1, because the Crout step floors every diagonal to a
        // strictly positive value — so an empty `displayDims` is the only input
        // that exercises it. Both backends must produce the same degenerate but
        // FINITE 3×3, not a zero (invisible in sum projection) or a NaN.
        const ndim = 2;
        const args = {
          positions: new Float32Array([1, 2]),
          cholesky: new Float32Array([2.5, 0.5, 1.5]),
          amplitudes: new Float32Array([1]),
          discreteVisibility: new Uint8Array([1]),
          slicePosition: new Float32Array(ndim),
          continuousHiddenDims: new Uint32Array([]),
          displayDims: new Uint32Array([]),
          ndim,
          splatCount: 1,
          minAmplitude: 1e-9,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        const phantom = 9.999999747378752e-6; // √(f32 1e-10)
        expect(w.count).toBe(1);
        expect(ts.count).toBe(w.count);
        expect(Array.from(w.centers.subarray(0, 3))).toEqual([0, 0, 0]);
        expect(Array.from(w.chol.subarray(0, 6))).toEqual([phantom, 0, phantom, 0, 0, phantom]);
        expect(Array.from(ts.chol.subarray(0, 6))).toEqual(Array.from(w.chol.subarray(0, 6)));
      }
    );

    it.skipIf(!wasmFilesExist).each([1, 2] as const)(
      'project_gsplats_nd_to_3d: %i display dims WITH continuous hidden dims',
      (numDisplay) => {
        // The two features that the rest of the file only exercises separately:
        // a <3-display-dim marginal (phantom padding) AND a live attenuation
        // path (hidden marginal Cholesky + forward substitution + exp). The
        // hidden dims are the ones the display does not take, so the marginal
        // is genuinely 3×3 at numDisplay = 1.
        // The f32 transcendental ports from #1830 and the surrounding operation
        // ordering from #1836 make both the phantom diagonals and attenuation
        // bit-exact against WASM.
        const splatCount = 4000;
        const ndim = 4;
        const packedSize = 10;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = mulberry32(0x1d1d);
        for (let i = 0; i < splatCount; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = (rnd() - 0.5) * 5;
          for (let row = 0; row < ndim; row++) {
            for (let col = 0; col <= row; col++) {
              cholesky[i * packedSize + (row * (row + 1)) / 2 + col] =
                col === row ? 0.4 + rnd() * 2 : (rnd() - 0.5) * 2;
            }
          }
          amplitudes[i] = 0.3 + rnd();
        }
        const truncate = 3.0;
        const args = {
          positions,
          cholesky,
          amplitudes,
          discreteVisibility: new Uint8Array(splatCount).fill(1),
          slicePosition: new Float32Array([0, 0, 0.531313121318817, -0.7213131189346313]),
          continuousHiddenDims:
            numDisplay === 1 ? new Uint32Array([1, 2, 3]) : new Uint32Array([2, 3]),
          displayDims: numDisplay === 1 ? new Uint32Array([0]) : new Uint32Array([0, 1]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
          truncate,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(ts.count).toBe(w.count);
        expect(ts.count).toBeGreaterThan(1000);
        const n = ts.count;
        // Split the packed 3D factor into the phantom diagonals (which cross
        // ln/exp) and everything else (which does not, so must be exact).
        const phantomSlots = numDisplay === 1 ? [2, 5] : [5];
        const tsPhantom: number[] = [];
        const wPhantom: number[] = [];
        const tsRest: number[] = [];
        const wRest: number[] = [];
        for (let i = 0; i < n; i++) {
          for (let k = 0; k < 6; k++) {
            if (phantomSlots.includes(k)) {
              tsPhantom.push(ts.chol[i * 6 + k]);
              wPhantom.push(w.chol[i * 6 + k]);
            } else {
              tsRest.push(ts.chol[i * 6 + k]);
              wRest.push(w.chol[i * 6 + k]);
            }
          }
        }
        expect(exactMismatches(tsRest, wRest)).toEqual(NO_MISMATCHES);
        expect(exactMismatches(tsPhantom, wPhantom)).toEqual(NO_MISMATCHES);
        expect(exactMismatches(ts.amps.subarray(0, n), w.amps.subarray(0, n))).toEqual(
          NO_MISMATCHES
        );
      }
    );

    it.skipIf(!wasmFilesExist).each([3.0, 2.75])(
      'project_gsplats_nd_to_3d attenuation matches exactly at truncate = %f',
      (truncate) => {
        // The full attenuation path uses correlated 5D factors, two continuous
        // hidden dims (so the marginal Cholesky and forward substitution both do
        // real work), three display dims, a non-zero slice, and shifted-Gaussian
        // normalization. The f32 operation ordering and expf port make every
        // emitted value exact.
        //
        // The 2.75 case is GSPLAT_DEFAULT_TRUNCATION_RADIUS and is load-bearing:
        // it is the only parameter here where shiftC distinguishes expf
        // (0x3cbabadc) from fround(Math.exp(...)) (0x3cbabadd), changing 1,299 of
        // 19,067 emitted amplitudes if the production call site regresses.
        const splatCount = 20000;
        const ndim = 5;
        const packedSize = 15;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = mulberry32(555);
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
          slicePosition: new Float32Array(FUSED_SLICE),
          continuousHiddenDims: new Uint32Array([3, 4]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
          truncate,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

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
        const tsAmps = ts.amps.subarray(0, n);
        const wAmps = w.amps.subarray(0, n);
        expect(exactMismatches(tsAmps, wAmps)).toEqual(NO_MISMATCHES);
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d attenuation matches exactly with the shift underflowed away',
      () => {
        // Same fixture, truncate = 20. `shiftC = exp(-200)` underflows to
        // exactly 0 in f32, so `invOneMinusC` is exactly 1 and
        // `attenuation === rawExp` — the catastrophic `rawExp − shiftC`
        // cancellation is switched off. What remains is the mahalanobis chain
        // plus one expf, which is bit-exact against WASM.
        const splatCount = 20000;
        const ndim = 5;
        const packedSize = 15;
        const truncate = 20;
        expect(invOneMinusC(truncate)).toBe(1); // the premise: shiftC underflowed
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = mulberry32(555);
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
          slicePosition: new Float32Array(FUSED_SLICE),
          continuousHiddenDims: new Uint32Array([3, 4]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
          truncate,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(ts.count).toBe(w.count);
        expect(ts.count).toBeGreaterThan(splatCount / 2);
        const n = ts.count;
        expect(exactMismatches(ts.chol.subarray(0, n * 6), w.chol.subarray(0, n * 6))).toEqual(
          NO_MISMATCHES
        );
        expect(exactMismatches(ts.amps.subarray(0, n), w.amps.subarray(0, n))).toEqual(
          NO_MISMATCHES
        );
      }
    );

    it.skipIf(!wasmFilesExist)(
      'project_gsplats_nd_to_3d attenuation: exact at a small truncate under the f32 expf port (#1830)',
      () => {
        // Same fixture again at truncate = 0.5, where `invOneMinusC` is 8.51
        // instead of 1.01 — historically the case that most amplified the
        // transcendental residual. With `expf` (the musl f32 port, #1830) the
        // attenuation is bit-exact against WASM here too, so what used to be a
        // scale-dependent ULP bound is now exact equality. The shiftC /
        // invOneMinusC roundings are still exercised — they matter at a small
        // truncate (an ulp of shiftC cannot move 1 - shiftC at truncate = 3, but
        // does at 0.5).
        const splatCount = 20000;
        const ndim = 5;
        const packedSize = 15;
        const truncate = 0.5;
        expect(invOneMinusC(truncate)).toBeGreaterThan(8);
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const amplitudes = new Float32Array(splatCount);
        const rnd = mulberry32(555);
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
          slicePosition: new Float32Array(FUSED_SLICE),
          continuousHiddenDims: new Uint32Array([3, 4]),
          displayDims: new Uint32Array([0, 1, 2]),
          ndim,
          splatCount,
          minAmplitude: 1e-6,
          truncate,
        };
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);

        expect(ts.count).toBe(w.count);
        expect(ts.count).toBeGreaterThan(1000);
        const n = ts.count;
        expect(exactMismatches(ts.chol.subarray(0, n * 6), w.chol.subarray(0, n * 6))).toEqual(
          NO_MISMATCHES
        );
        const tsAmps = ts.amps.subarray(0, n);
        const wAmps = w.amps.subarray(0, n);
        // #1830: `expf` is bit-exact against Rust's `f32::exp`, so the
        // attenuation matches WASM exactly even at truncate = 0.5 — the residual
        // this case used to amplify (was 111/2529 differing, up to 6.56e-7) is gone.
        expect(maxAbs(tsAmps, wAmps)).toBe(0);
        expect(differingCount(tsAmps, wAmps)).toBe(0);
      }
    );

    it.skipIf(!wasmFilesExist).each([1, 1e-4, 1e-7, 1e6] as const)(
      'computeDisplayCholesky3D 2D phantom is exact at scene scale %f',
      (scale) => {
        // The phantom z diagonal is `expf(mean(logf(Lii)))`. The scale sweep
        // keeps the extreme-scene coverage while asserting the exact contract.
        // `slicePosition` stays all zeros: with no continuous hidden dims the
        // kernel never reads it. The `FUSED_SLICE` sweeps are the ones that
        // exercise the `diff[]` subtraction.
        const splatCount = 20000;
        const ndim = 2;
        const packedSize = 3;
        const positions = new Float32Array(splatCount * ndim);
        const cholesky = new Float32Array(splatCount * packedSize);
        const rnd = mulberry32(4242);
        for (let i = 0; i < splatCount; i++) {
          for (let d = 0; d < ndim; d++) positions[i * ndim + d] = rnd() * 10;
          const b = i * packedSize;
          cholesky[b] = Math.fround((0.5 + rnd() * 3) * scale);
          cholesky[b + 1] = Math.fround((rnd() - 0.5) * 2 * scale);
          cholesky[b + 2] = Math.fround((0.5 + rnd() * 3) * scale);
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
        const ts = runFused(tsModule, args);
        const w = runFused(wasmModule!, args);
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
        expect(exactMismatches(tsPhantom, wPhantom)).toEqual(NO_MISMATCHES);
      }
    );
  });

  // ============================================================================
  // LINES CLIPPING MODULE
  // ============================================================================
  describe('lines_clipping functions', () => {
    it.skipIf(!wasmFilesExist)('agrees exactly at a sub-ulp f32 slab boundary', () => {
      const slice = Math.fround(1.0);
      const tol = Math.fround(0.1);
      const exactBound = slice - tol;
      const f32Bound = Math.fround(exactBound);

      expect(f32Bound).toBeLessThan(exactBound);

      const p1 = new Float32Array([0, 0, 0, f32Bound]);
      const p2 = new Float32Array([1, 1, 1, slice]);
      const slicePos = new Float32Array([0, 0, 0, slice]);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, tol]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsSingle = tsModule.clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 4);
      const wasmSingle = wasmModule!.clip_segment_single(
        p1,
        p2,
        slicePos,
        tolerance,
        displayDims,
        4
      );
      expect(Array.from(tsSingle)).toEqual(Array.from(wasmSingle));

      const positions = new Float32Array([...p1, ...p2]);
      const segments = new Uint32Array([0, 1]);
      const tsVisibility = new Uint8Array(1);
      const tsT1 = new Float32Array(1);
      const tsT2 = new Float32Array(1);
      const wasmVisibility = new Uint8Array(1);
      const wasmT1 = new Float32Array(1);
      const wasmT2 = new Float32Array(1);

      const tsCount = tsModule.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        4,
        1,
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
        1,
        wasmVisibility,
        wasmT1,
        wasmT2
      );

      expect(tsCount).toBe(wasmCount);
      expect(Array.from(tsVisibility)).toEqual(Array.from(wasmVisibility));
      expect(Array.from(tsT1)).toEqual(Array.from(wasmT1));
      expect(Array.from(tsT2)).toEqual(Array.from(wasmT2));
    });

    it.skipIf(!wasmFilesExist)('matches f32 reciprocal and t-range accumulation exactly', () => {
      const positions = new Float32Array([
        0, 0, 0, -0.2653183341026306, -0.5240607261657715, 1, 1, 1, 0.3167182207107544,
        -0.13231058418750763,
      ]);
      const segments = new Uint32Array([0, 1]);
      const slicePos = new Float32Array(5);
      const tolerance = new Float32Array([
        1e10, 1e10, 1e10, 0.23722794651985168, 0.35342466831207275,
      ]);
      const displayDims = new Uint32Array([0, 1, 2]);
      const p1 = positions.subarray(0, 5);
      const p2 = positions.subarray(5, 10);
      const dv = Math.fround(p2[4] - p1[4]);
      const numerator = Math.fround(-tolerance[4] - p1[4]);
      const rustOrderT = Math.fround(numerator * Math.fround(1.0 / dv));
      const directDivisionT = Math.fround((-tolerance[4] - p1[4]) / (p2[4] - p1[4]));

      expect(rustOrderT).not.toBe(directDivisionT);

      const tsSingle = tsModule.clip_segment_single(p1, p2, slicePos, tolerance, displayDims, 5);
      const wasmSingle = wasmModule!.clip_segment_single(
        p1,
        p2,
        slicePos,
        tolerance,
        displayDims,
        5
      );
      expect(Array.from(tsSingle)).toEqual(Array.from(wasmSingle));

      const tsVisibility = new Uint8Array(1);
      const tsT1 = new Float32Array(1);
      const tsT2 = new Float32Array(1);
      const wasmVisibility = new Uint8Array(1);
      const wasmT1 = new Float32Array(1);
      const wasmT2 = new Float32Array(1);

      const tsCount = tsModule.clip_segments_batch(
        positions,
        segments,
        slicePos,
        tolerance,
        displayDims,
        5,
        1,
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
        5,
        1,
        wasmVisibility,
        wasmT1,
        wasmT2
      );

      expect(tsCount).toBe(wasmCount);
      expect(Array.from(tsVisibility)).toEqual(Array.from(wasmVisibility));
      expect(Array.from(tsT1)).toEqual(Array.from(wasmT1));
      expect(Array.from(tsT2)).toEqual(Array.from(wasmT2));
    });

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

      expect(Array.from(tsResult)).toEqual(Array.from(wasmResult));
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
      expect(Array.from(tsVisibility)).toEqual(Array.from(wasmVisibility));
      expect(Array.from(tsT1)).toEqual(Array.from(wasmT1));
      expect(Array.from(tsT2)).toEqual(Array.from(wasmT2));
    });

    it.skipIf(!wasmFilesExist)('interpolate_clipped_positions should match', () => {
      const positions = new Float32Array([
        -4.11237096786499, 3.7138946056365967, -2.3870370388031006, 0, 0.3802664279937744,
        -4.748578071594238, 1.6990193128585815, 0,
      ]);
      const segments = new Uint32Array([0, 1]);
      const visibility = new Uint8Array([1]);
      const t1Params = new Float32Array([0.8854133486747742]);
      const t2Params = new Float32Array([0.5823075771331787]);
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
      expect(Array.from(tsStart)).toEqual(Array.from(wasmStart));
      expect(Array.from(tsEnd)).toEqual(Array.from(wasmEnd));
    });

    it.skipIf(!wasmFilesExist)('interpolate_scalars_batch should match', () => {
      const values = new Float32Array([-4.11237096786499, 0.3802664279937744]);
      const segments = new Uint32Array([0, 1]);
      const visibility = new Uint8Array([1]);
      const t1Params = new Float32Array([0.8854133486747742]);
      const t2Params = new Float32Array([0.5823075771331787]);

      const tsStart = new Float32Array(1);
      const tsEnd = new Float32Array(1);
      const wasmStart = new Float32Array(1);
      const wasmEnd = new Float32Array(1);

      const tsCount = tsModule.interpolate_scalars_batch(
        values,
        segments,
        visibility,
        t1Params,
        t2Params,
        1,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.interpolate_scalars_batch(
        values,
        segments,
        visibility,
        t1Params,
        t2Params,
        1,
        wasmStart,
        wasmEnd
      );

      expect(wasmCount).toBe(tsCount);
      expect(Array.from(tsStart)).toEqual(Array.from(wasmStart));
      expect(Array.from(tsEnd)).toEqual(Array.from(wasmEnd));
    });

    it.skipIf(!wasmFilesExist)('interpolate_colors_batch should match', () => {
      const colors = new Float32Array([
        -4.11237096786499, 3.7138946056365967, -2.3870370388031006, 0.3802664279937744,
        -4.748578071594238, 1.6990193128585815,
      ]);
      const segments = new Uint32Array([0, 1]);
      const visibility = new Uint8Array([1]);
      const t1Params = new Float32Array([0.8854133486747742]);
      const t2Params = new Float32Array([0.5823075771331787]);

      const tsStart = new Float32Array(3);
      const tsEnd = new Float32Array(3);
      const wasmStart = new Float32Array(3);
      const wasmEnd = new Float32Array(3);

      const tsCount = tsModule.interpolate_colors_batch(
        colors,
        segments,
        visibility,
        t1Params,
        t2Params,
        1,
        tsStart,
        tsEnd
      );
      const wasmCount = wasmModule!.interpolate_colors_batch(
        colors,
        segments,
        visibility,
        t1Params,
        t2Params,
        1,
        wasmStart,
        wasmEnd
      );

      expect(wasmCount).toBe(tsCount);
      expect(Array.from(tsStart)).toEqual(Array.from(wasmStart));
      expect(Array.from(tsEnd)).toEqual(Array.from(wasmEnd));
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
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 2.5, 2.5]);
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

      const clippedCount = tsVisibility.reduce(
        (count, visible, index) =>
          count + (visible !== 0 && (tsT1[index] !== 0 || tsT2[index] !== 1) ? 1 : 0),
        0
      );

      expect(tsCount).toBe(253);
      expect(clippedCount).toBe(64);
      expect(wasmCount).toBe(tsCount);
      expect(Array.from(tsVisibility)).toEqual(Array.from(wasmVisibility));
      expect(Array.from(tsT1)).toEqual(Array.from(wasmT1));
      expect(Array.from(tsT2)).toEqual(Array.from(wasmT2));
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

      const ts = runFused(tsModule, args);
      const w = runFused(wasmModule!, args);

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
