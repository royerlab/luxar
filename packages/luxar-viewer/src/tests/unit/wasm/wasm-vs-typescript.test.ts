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
import type { WasmModule } from '../../../wasm/types';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';

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
 * Helper to compare Float32Arrays within tolerance.
 *
 * NOTE (wasm.md C3): default epsilon `1e-5` is appropriate for typical
 * Float32 single-step operations. For multi-step algorithms that accumulate
 * rounding error per dimension (e.g. `mahalanobis_distance` does ndim
 * forward-substitution steps), call sites should pass a scaled epsilon —
 * e.g. `arraysAlmostEqual(a, b, 1e-5 * Math.sqrt(ndim))` for ndim > 3 —
 * to avoid silently missing WASM-vs-TS divergence at high dimensions.
 */
function arraysAlmostEqual(a: ArrayLike<number>, b: ArrayLike<number>, epsilon = 1e-5): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > epsilon) {
      console.log(`Mismatch at index ${i}: ${a[i]} vs ${b[i]} (diff: ${Math.abs(a[i] - b[i])})`);
      return false;
    }
  }
  return true;
}

/**
 * Helper to compare Uint8Arrays exactly
 */
function arraysEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
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
  // SPATIAL MODULE
  // ============================================================================
  describe('spatial: query_chunks_for_view', () => {
    it.skipIf(!wasmFilesExist)('should produce identical results', () => {
      const chunkBounds = new Float32Array([
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0,
      ]);
      const slicePos = new Float32Array([0.5, 0.5, 0.5]);
      const tolerance = new Float32Array([0.6, 0.6, 0.6]);

      const tsOutput = new Uint32Array(3);
      const wasmOutput = new Uint32Array(3);

      const tsCount = tsModule.query_chunks_for_view(
        chunkBounds,
        slicePos,
        tolerance,
        3,
        3,
        tsOutput
      );
      const wasmCount = wasmModule!.query_chunks_for_view(
        chunkBounds,
        slicePos,
        tolerance,
        3,
        3,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // POINTS MODULE
  // ============================================================================
  describe('points: compute_nd_visibility_points', () => {
    it.skipIf(!wasmFilesExist)('should produce identical visibility masks', () => {
      const positions = new Float32Array([
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0,
        0.8, 0.0,
      ]);
      const radii = new Float32Array([1.0, 1.0, 1.0, 1.0]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
      const tolerance = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0]);

      const tsOutput = new Uint8Array(4);
      const wasmOutput = new Uint8Array(4);

      const tsCount = tsModule.compute_nd_visibility_points(
        positions,
        radii,
        slicePos,
        tolerance,
        5,
        4,
        tsOutput
      );
      const wasmCount = wasmModule!.compute_nd_visibility_points(
        positions,
        radii,
        slicePos,
        tolerance,
        5,
        4,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // LINES MODULE
  // ============================================================================
  describe('lines: compute_nd_visibility_lines', () => {
    it.skipIf(!wasmFilesExist)('should produce identical visibility masks', () => {
      const vertices = new Float32Array([
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 2.0, 2.0, 2.0, 10.0, 0.0, 3.0, 3.0, 3.0,
        0.0, 10.0,
      ]);
      const segments = new Uint32Array([0, 1, 1, 2, 2, 3]);
      const widths = new Float32Array([1.0, 1.0, 1.0, 1.0]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0]);
      const tolerance = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0]);

      const tsOutput = new Uint8Array(3);
      const wasmOutput = new Uint8Array(3);

      const tsCount = tsModule.compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        slicePos,
        tolerance,
        5,
        3,
        tsOutput
      );
      const wasmCount = wasmModule!.compute_nd_visibility_lines(
        vertices,
        segments,
        widths,
        slicePos,
        tolerance,
        5,
        3,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmOutput, tsOutput)).toBe(true);
    });
  });

  // ============================================================================
  // GSPLATS MODULE
  // ============================================================================
  describe('gsplats: compute_nd_visibility_gsplats', () => {
    it.skipIf(!wasmFilesExist)('should produce identical visibility masks', () => {
      const centers = new Float32Array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0]);
      const choleskyFactors = new Float32Array([
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
      ]);
      const slicePos = new Float32Array([0.0, 0.0, 0.0]);
      const tolerance = new Float32Array([2.0, 2.0, 2.0]);

      const tsOutput = new Uint8Array(2);
      const wasmOutput = new Uint8Array(2);

      const tsCount = tsModule.compute_nd_visibility_gsplats(
        centers,
        choleskyFactors,
        slicePos,
        tolerance,
        3,
        2,
        tsOutput
      );
      const wasmCount = wasmModule!.compute_nd_visibility_gsplats(
        centers,
        choleskyFactors,
        slicePos,
        tolerance,
        3,
        2,
        wasmOutput
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmOutput, tsOutput)).toBe(true);
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

    it.skipIf(!wasmFilesExist)('distance_3d should match', () => {
      const testCases = [
        { a: [0, 0, 0], b: [3, 4, 0] }, // 5
        { a: [1, 1, 1], b: [2, 2, 2] }, // sqrt(3)
        { a: [-1, -1, -1], b: [1, 1, 1] }, // sqrt(12)
      ];

      for (const { a, b } of testCases) {
        const aArr = new Float32Array(a);
        const bArr = new Float32Array(b);
        const tsResult = tsModule.distance_3d(aArr, bArr);
        const wasmResult = wasmModule!.distance_3d(aArr, bArr);
        expect(Math.abs(wasmResult - tsResult)).toBeLessThan(1e-5);
      }
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

    it.skipIf(!wasmFilesExist)('mark_clipped_endpoints should match', () => {
      const visibility = new Uint8Array([1, 1, 0, 1]);
      const t1Params = new Float32Array([0.0, 0.5, 0.0, 0.25]);
      const t2Params = new Float32Array([1.0, 0.75, 1.0, 1.0]);

      const tsStartClipped = new Uint8Array(3);
      const tsEndClipped = new Uint8Array(3);
      const wasmStartClipped = new Uint8Array(3);
      const wasmEndClipped = new Uint8Array(3);

      const tsCount = tsModule.mark_clipped_endpoints(
        visibility,
        t1Params,
        t2Params,
        4,
        tsStartClipped,
        tsEndClipped
      );
      const wasmCount = wasmModule!.mark_clipped_endpoints(
        visibility,
        t1Params,
        t2Params,
        4,
        wasmStartClipped,
        wasmEndClipped
      );

      expect(wasmCount).toBe(tsCount);
      expect(arraysEqual(wasmStartClipped, tsStartClipped)).toBe(true);
      expect(arraysEqual(wasmEndClipped, tsEndClipped)).toBe(true);
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
  });
});
