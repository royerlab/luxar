/**
 * WASM vs TypeScript Performance Benchmark
 *
 * Measures and compares the performance of WASM/Rust implementations
 * against their TypeScript reference/fallback implementations.
 *
 * Run with: pnpm bench:wasm (or make benchmark-wasm)
 */

import { TypeScriptFallback } from '../../wasm/typescript';
import type { WasmModule } from '../../wasm/types';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, readFileSync } from 'fs';

// ============================================================================
// Configuration
// ============================================================================

const SIZES = {
  small: 1_000,
  medium: 100_000,
  large: 1_000_000,
} as const;

const CONFIG = {
  iterations: 5, // Number of iterations for timing
  warmupIterations: 2, // Warmup runs before timing
  sizes: SIZES,
  defaultSize: 'medium' as keyof typeof SIZES,
};

// ============================================================================
// Types
// ============================================================================

interface BenchmarkResult {
  name: string;
  category: string;
  tsTime: number;
  wasmTime: number;
  speedup: number;
  /** If true, this is a utility function not used in production hot paths */
  utility?: boolean;
}

// ============================================================================
// Module Loading
// ============================================================================

let wasmModule: WasmModule | null = null;
let tsModule: WasmModule;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../public/wasm/luxar_wasm_bg.wasm');

async function loadModules(): Promise<boolean> {
  // Initialize TypeScript fallback (always available)
  tsModule = new TypeScriptFallback();

  // Check if WASM files exist
  if (!existsSync(wasmJsPath) || !existsSync(wasmBinaryPath)) {
    console.log('\x1b[33m[!] WASM module not found\x1b[0m');
    console.log(`    Expected at: ${wasmJsPath}`);
    console.log('    Build with: make build-wasm');
    return false;
  }

  try {
    const wasmBinary = readFileSync(wasmBinaryPath);
    const wasm = await import(wasmJsPath);
    wasm.initSync({ module: wasmBinary });
    wasmModule = wasm as unknown as WasmModule;
    return true;
  } catch (error) {
    console.log('\x1b[31m[!] Failed to load WASM module:\x1b[0m', error);
    return false;
  }
}

// ============================================================================
// Timing Utilities
// ============================================================================

function measureTime(fn: () => void, iterations: number, warmup: number): number {
  // Warmup
  for (let i = 0; i < warmup; i++) {
    fn();
  }

  // Measure
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = performance.now();

  return (end - start) / iterations;
}

function formatTime(ms: number): string {
  if (ms < 0.1) return `${(ms * 1000).toFixed(1)}us`;
  if (ms < 1) return `${(ms * 1000).toFixed(0)}us`;
  if (ms < 100) return `${ms.toFixed(2)}ms`;
  return `${ms.toFixed(0)}ms`;
}

function formatSpeedup(speedup: number): string {
  const color = speedup >= 3 ? '\x1b[32m' : speedup >= 1.5 ? '\x1b[33m' : '\x1b[31m';
  return `${color}${speedup.toFixed(1)}x\x1b[0m`;
}

// ============================================================================
// Test Data Generators
// ============================================================================

function generatePositions(count: number, ndim: number): Float32Array {
  const positions = new Float32Array(count * ndim);
  for (let i = 0; i < count; i++) {
    for (let d = 0; d < ndim; d++) {
      positions[i * ndim + d] = Math.sin(i * 0.037 + d * 1.23) * 10;
    }
  }
  return positions;
}

function generateRadii(count: number): Float32Array {
  const radii = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    radii[i] = Math.abs(Math.sin(i * 0.071)) * 1.5 + 0.1;
  }
  return radii;
}

function generateSegments(count: number): Uint32Array {
  const segments = new Uint32Array(count * 2);
  for (let i = 0; i < count; i++) {
    segments[i * 2] = i;
    segments[i * 2 + 1] = i + 1;
  }
  return segments;
}

function generateCholeskyFactors(count: number, ndim: number): Float32Array {
  const size = (ndim * (ndim + 1)) / 2;
  const factors = new Float32Array(count * size);
  for (let i = 0; i < count * size; i++) {
    factors[i] = Math.random() * 0.5 + 0.5;
  }
  return factors;
}

function generateQuantizedU8(count: number): Uint8Array {
  const data = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = Math.floor(Math.random() * 256);
  }
  return data;
}

function generateQuantizedU16(count: number): Uint16Array {
  const data = new Uint16Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = Math.floor(Math.random() * 65536);
  }
  return data;
}

// ============================================================================
// Benchmark Definitions
// ============================================================================

type BenchmarkFn = (size: number) => BenchmarkResult;

const benchmarks: Record<string, BenchmarkFn[]> = {
  'EFFECTIVE RADII': [
    (size) => {
      const ndim = 6;
      const positions = generatePositions(size, ndim);
      const radii = generateRadii(size);
      const displayDims = new Uint32Array([0, 1, 2]);
      const slicePos = new Float32Array(ndim).fill(0);
      const spatialExtend = new Uint8Array(ndim).fill(1);
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.calculate_effective_radii(
            positions,
            radii,
            displayDims,
            slicePos,
            spatialExtend,
            ndim,
            size,
            tsOutput
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.calculate_effective_radii(
            positions,
            radii,
            displayDims,
            slicePos,
            spatialExtend,
            ndim,
            size,
            wasmOutput
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'calculate_effective_radii',
        category: 'EFFECTIVE RADII',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
  ],

  DECODE: [
    (size) => {
      const data = generateQuantizedU8(size);
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.decode_quantized_u8(data, -10.0, 10.0, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_quantized_u8(data, -10.0, 10.0, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_quantized_u8',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const data = generateQuantizedU16(size);
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.decode_quantized_u16(data, -10.0, 10.0, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_quantized_u16(data, -10.0, 10.0, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_quantized_u16',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const data = generateQuantizedU8(size);
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.decode_log_scalar_u8(data, 5.0, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_log_scalar_u8(data, 5.0, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_log_scalar_u8',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const data = generateQuantizedU16(size);
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.decode_log_scalar_u16(data, 5.0, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_log_scalar_u16(data, 5.0, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_log_scalar_u16',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const lutSize = 256;
      const indices = generateQuantizedU8(size);
      const lut = new Float32Array(lutSize);
      for (let i = 0; i < lutSize; i++) {
        lut[i] = Math.sin(i * 0.01) * 100;
      }
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.decode_lut_scalar_u8(indices, lut, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_lut_scalar_u8(indices, lut, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_lut_scalar_u8',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const lutSize = 4096;
      const indices = generateQuantizedU16(size);
      // Clamp indices to LUT size
      for (let i = 0; i < size; i++) {
        indices[i] = indices[i] % lutSize;
      }
      const lut = new Float32Array(lutSize);
      for (let i = 0; i < lutSize; i++) {
        lut[i] = Math.sin(i * 0.01) * 100;
      }
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.decode_lut_scalar_u16(indices, lut, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_lut_scalar_u16(indices, lut, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_lut_scalar_u16',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const lutSize = 256;
      const indices = generateQuantizedU8(size);
      const rowSize = 3;
      const lut = new Float32Array(lutSize * rowSize);
      for (let i = 0; i < lutSize * rowSize; i++) {
        lut[i] = Math.random();
      }
      const tsOutput = new Float32Array(size * rowSize);
      const wasmOutput = new Float32Array(size * rowSize);

      const tsTime = measureTime(
        () => {
          tsModule.decode_lut_row_u8(indices, lut, rowSize, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_lut_row_u8(indices, lut, rowSize, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_lut_row_u8',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const lutSize = 4096;
      const indices = generateQuantizedU16(size);
      // Clamp indices to LUT size
      for (let i = 0; i < size; i++) {
        indices[i] = indices[i] % lutSize;
      }
      const rowSize = 3;
      const lut = new Float32Array(lutSize * rowSize);
      for (let i = 0; i < lutSize * rowSize; i++) {
        lut[i] = Math.random();
      }
      const tsOutput = new Float32Array(size * rowSize);
      const wasmOutput = new Float32Array(size * rowSize);

      const tsTime = measureTime(
        () => {
          tsModule.decode_lut_row_u16(indices, lut, rowSize, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_lut_row_u16(indices, lut, rowSize, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_lut_row_u16',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const value = new Float32Array([0.5, 0.6, 0.7]);
      const tsOutput = new Float32Array(size * 3);
      const wasmOutput = new Float32Array(size * 3);

      const tsTime = measureTime(
        () => {
          tsModule.decode_broadcasted(value, size, 3, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.decode_broadcasted(value, size, 3, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'decode_broadcasted',
        category: 'DECODE',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
  ],

  PROJECTION: [
    (size) => {
      const ndim = 5;
      const positionsNd = generatePositions(size, ndim);
      const displayDims = new Uint32Array([0, 2, 4]);
      const tsOutput = new Float32Array(size * 3);
      const wasmOutput = new Float32Array(size * 3);

      const tsTime = measureTime(
        () => {
          tsModule.extract_3d_positions(positionsNd, displayDims, ndim, size, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.extract_3d_positions(positionsNd, displayDims, ndim, size, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'extract_3d_positions',
        category: 'PROJECTION',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const positions = generatePositions(size, 3);
      const tsOutput = new Float32Array(6);
      const wasmOutput = new Float32Array(6);

      const tsTime = measureTime(
        () => {
          tsModule.calculate_bounds_3d(positions, size, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.calculate_bounds_3d(positions, size, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'calculate_bounds_3d',
        category: 'PROJECTION',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const stride = 3;
      const input = generatePositions(size, stride);
      // Create mask with ~50% visible
      const mask = new Uint8Array(size);
      for (let i = 0; i < size; i++) {
        mask[i] = Math.random() > 0.5 ? 1 : 0;
      }
      const tsOutput = new Float32Array(size * stride);
      const wasmOutput = new Float32Array(size * stride);

      const tsTime = measureTime(
        () => {
          tsModule.compact_by_mask(input, mask, size, stride, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.compact_by_mask(input, mask, size, stride, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'compact_by_mask',
        category: 'PROJECTION',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const mask = new Uint8Array(size);
      for (let i = 0; i < size; i++) {
        mask[i] = Math.random() > 0.5 ? 1 : 0;
      }

      const tsTime = measureTime(
        () => {
          tsModule.count_visible(mask, size);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.count_visible(mask, size);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'count_visible',
        category: 'PROJECTION',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const radii = generateRadii(size);
      const tsOutput = new Uint8Array(size);
      const wasmOutput = new Uint8Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.radii_to_visibility_mask(radii, 0.01, size, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.radii_to_visibility_mask(radii, 0.01, size, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'radii_to_visibility_mask',
        category: 'PROJECTION',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
  ],

  'LINES CLIPPING': [
    (_size) => {
      // clip_segment_single - single segment clipping (per-call benchmark)
      const ndim = 5;
      const p1 = new Float32Array([0, 0, 0, -0.5, -0.5]);
      const p2 = new Float32Array([1, 1, 1, 0.5, 0.5]);
      const slicePos = new Float32Array(ndim).fill(0);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 1.0, 1.0]);
      const displayDims = new Uint32Array([0, 1, 2]);

      const tsTime = measureTime(
        () => {
          for (let i = 0; i < 1000; i++) {
            tsModule.clip_segment_single(p1, p2, slicePos, tolerance, displayDims, ndim);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          for (let i = 0; i < 1000; i++) {
            wasmModule!.clip_segment_single(p1, p2, slicePos, tolerance, displayDims, ndim);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'clip_segment_single (1K calls)',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
        utility: true, // Not used in production - batch version inlines the logic
      };
    },
    (size) => {
      const ndim = 5;
      const positions = generatePositions(size + 1, ndim);
      const segments = generateSegments(size);
      const slicePos = new Float32Array(ndim).fill(0);
      const tolerance = new Float32Array([1e10, 1e10, 1e10, 1.0, 1.0]);
      const displayDims = new Uint32Array([0, 1, 2]);
      const tsVisibility = new Uint8Array(size);
      const tsT1 = new Float32Array(size);
      const tsT2 = new Float32Array(size);
      const wasmVisibility = new Uint8Array(size);
      const wasmT1 = new Float32Array(size);
      const wasmT2 = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.clip_segments_batch(
            positions,
            segments,
            slicePos,
            tolerance,
            displayDims,
            ndim,
            size,
            tsVisibility,
            tsT1,
            tsT2
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.clip_segments_batch(
            positions,
            segments,
            slicePos,
            tolerance,
            displayDims,
            ndim,
            size,
            wasmVisibility,
            wasmT1,
            wasmT2
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'clip_segments_batch',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      // interpolate_clipped_positions - batch position interpolation
      const ndim = 5;
      const positions = generatePositions(size + 1, ndim);
      const segments = generateSegments(size);
      const visibility = new Uint8Array(size);
      for (let i = 0; i < size; i++) visibility[i] = Math.random() > 0.3 ? 1 : 0;
      const t1Params = new Float32Array(size);
      const t2Params = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        t1Params[i] = Math.random() * 0.3;
        t2Params[i] = 0.7 + Math.random() * 0.3;
      }
      const displayDims = new Uint32Array([0, 1, 2]);
      const tsOutputStart = new Float32Array(size * 3);
      const tsOutputEnd = new Float32Array(size * 3);
      const wasmOutputStart = new Float32Array(size * 3);
      const wasmOutputEnd = new Float32Array(size * 3);

      const tsTime = measureTime(
        () => {
          tsModule.interpolate_clipped_positions(
            positions,
            segments,
            visibility,
            t1Params,
            t2Params,
            displayDims,
            ndim,
            size,
            tsOutputStart,
            tsOutputEnd
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.interpolate_clipped_positions(
            positions,
            segments,
            visibility,
            t1Params,
            t2Params,
            displayDims,
            ndim,
            size,
            wasmOutputStart,
            wasmOutputEnd
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'interpolate_clipped_positions',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      // interpolate_scalars_batch - batch scalar interpolation
      const values = generateRadii(size + 1);
      const segments = generateSegments(size);
      const visibility = new Uint8Array(size);
      for (let i = 0; i < size; i++) visibility[i] = Math.random() > 0.3 ? 1 : 0;
      const t1Params = new Float32Array(size);
      const t2Params = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        t1Params[i] = Math.random() * 0.3;
        t2Params[i] = 0.7 + Math.random() * 0.3;
      }
      const tsOutputStart = new Float32Array(size);
      const tsOutputEnd = new Float32Array(size);
      const wasmOutputStart = new Float32Array(size);
      const wasmOutputEnd = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.interpolate_scalars_batch(
            values,
            segments,
            visibility,
            t1Params,
            t2Params,
            size,
            tsOutputStart,
            tsOutputEnd
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.interpolate_scalars_batch(
            values,
            segments,
            visibility,
            t1Params,
            t2Params,
            size,
            wasmOutputStart,
            wasmOutputEnd
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'interpolate_scalars_batch',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      // interpolate_colors_batch - batch RGB color interpolation
      const colors = generatePositions(size + 1, 3); // Reuse position generator for RGB
      const segments = generateSegments(size);
      const visibility = new Uint8Array(size);
      for (let i = 0; i < size; i++) visibility[i] = Math.random() > 0.3 ? 1 : 0;
      const t1Params = new Float32Array(size);
      const t2Params = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        t1Params[i] = Math.random() * 0.3;
        t2Params[i] = 0.7 + Math.random() * 0.3;
      }
      const tsOutputStart = new Float32Array(size * 3);
      const tsOutputEnd = new Float32Array(size * 3);
      const wasmOutputStart = new Float32Array(size * 3);
      const wasmOutputEnd = new Float32Array(size * 3);

      const tsTime = measureTime(
        () => {
          tsModule.interpolate_colors_batch(
            colors,
            segments,
            visibility,
            t1Params,
            t2Params,
            size,
            tsOutputStart,
            tsOutputEnd
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.interpolate_colors_batch(
            colors,
            segments,
            visibility,
            t1Params,
            t2Params,
            size,
            wasmOutputStart,
            wasmOutputEnd
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'interpolate_colors_batch',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      const startPos = generatePositions(size, 3);
      const endPos = generatePositions(size, 3);
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.calculate_segment_lengths(startPos, endPos, size, tsOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.calculate_segment_lengths(startPos, endPos, size, wasmOutput);
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'calculate_segment_lengths',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      // mark_clipped_endpoints - mark which endpoints were clipped
      const visibility = new Uint8Array(size);
      for (let i = 0; i < size; i++) visibility[i] = Math.random() > 0.3 ? 1 : 0;
      const t1Params = new Float32Array(size);
      const t2Params = new Float32Array(size);
      for (let i = 0; i < size; i++) {
        t1Params[i] = Math.random() * 0.3;
        t2Params[i] = 0.7 + Math.random() * 0.3;
      }
      const tsStartClipped = new Uint8Array(size);
      const tsEndClipped = new Uint8Array(size);
      const wasmStartClipped = new Uint8Array(size);
      const wasmEndClipped = new Uint8Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.mark_clipped_endpoints(
            visibility,
            t1Params,
            t2Params,
            size,
            tsStartClipped,
            tsEndClipped
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.mark_clipped_endpoints(
            visibility,
            t1Params,
            t2Params,
            size,
            wasmStartClipped,
            wasmEndClipped
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'mark_clipped_endpoints',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (_size) => {
      // lerp - scalar linear interpolation (per-call benchmark)
      const tsTime = measureTime(
        () => {
          for (let i = 0; i < 10000; i++) {
            tsModule.lerp(0.0, 1.0, 0.5);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          for (let i = 0; i < 10000; i++) {
            wasmModule!.lerp(0.0, 1.0, 0.5);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'lerp (10K calls)',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
        utility: true, // Not used in production - batch functions inline the math
      };
    },
    (_size) => {
      // lerp_vec3 - 3D vector linear interpolation (per-call benchmark)
      const a = new Float32Array([0, 0, 0]);
      const b = new Float32Array([1, 2, 3]);

      const tsTime = measureTime(
        () => {
          for (let i = 0; i < 10000; i++) {
            tsModule.lerp_vec3(a, b, 0.5);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          for (let i = 0; i < 10000; i++) {
            wasmModule!.lerp_vec3(a, b, 0.5);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'lerp_vec3 (10K calls)',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
        utility: true, // Not used in production - batch functions inline the math
      };
    },
    (_size) => {
      // distance_3d - 3D Euclidean distance (per-call benchmark)
      const a = new Float32Array([0, 0, 0]);
      const b = new Float32Array([1, 2, 3]);

      const tsTime = measureTime(
        () => {
          for (let i = 0; i < 10000; i++) {
            tsModule.distance_3d(a, b);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          for (let i = 0; i < 10000; i++) {
            wasmModule!.distance_3d(a, b);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'distance_3d (10K calls)',
        category: 'LINES CLIPPING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
        utility: true, // Not used in production - batch functions inline the math
      };
    },
  ],

  'GSPLATS PROCESSING': [
    (size) => {
      const ndim = 4;
      const positions = generatePositions(size, ndim);
      const cholesky = generateCholeskyFactors(size, ndim);
      const amplitudes = generateRadii(size);
      const slicePos = new Float32Array(ndim).fill(0);
      const hiddenDims = new Uint32Array([3]);
      const tsVisibility = new Uint8Array(size);
      const tsAttenuation = new Float32Array(size);
      const wasmVisibility = new Uint8Array(size);
      const wasmAttenuation = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.compute_gsplats_attenuation(
            positions,
            cholesky,
            amplitudes,
            slicePos,
            hiddenDims,
            ndim,
            size,
            0.01,
            3.0,
            tsVisibility,
            tsAttenuation
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.compute_gsplats_attenuation(
            positions,
            cholesky,
            amplitudes,
            slicePos,
            hiddenDims,
            ndim,
            size,
            0.01,
            3.0,
            wasmVisibility,
            wasmAttenuation
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'compute_gsplats_attenuation',
        category: 'GSPLATS PROCESSING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (_size) => {
      // mahalanobis_distance - per-point operation, benchmark with many calls
      const ndim = 4;
      const diff = new Float32Array(ndim);
      const packedL = new Float32Array((ndim * (ndim + 1)) / 2);
      for (let i = 0; i < ndim; i++) diff[i] = Math.random() * 2 - 1;
      for (let i = 0; i < packedL.length; i++) packedL[i] = Math.random() * 0.5 + 0.5;

      const tsTime = measureTime(
        () => {
          for (let i = 0; i < 1000; i++) {
            tsModule.mahalanobis_distance(diff, packedL, ndim);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          for (let i = 0; i < 1000; i++) {
            wasmModule!.mahalanobis_distance(diff, packedL, ndim);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'mahalanobis_distance (1K calls)',
        category: 'GSPLATS PROCESSING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
        utility: true, // Not used in production - inlined by batch functions
      };
    },
    (_size) => {
      // extract_cholesky_submatrix - extract 3D from 4D
      const ndim = 4;
      const subNdim = 3;
      const packed = new Float32Array((ndim * (ndim + 1)) / 2);
      for (let i = 0; i < packed.length; i++) packed[i] = Math.random() * 0.5 + 0.5;
      const keepDims = new Uint32Array([0, 1, 2]);
      const tsOutput = new Float32Array((subNdim * (subNdim + 1)) / 2);
      const wasmOutput = new Float32Array((subNdim * (subNdim + 1)) / 2);

      const tsTime = measureTime(
        () => {
          for (let i = 0; i < 1000; i++) {
            tsModule.extract_cholesky_submatrix(packed, keepDims, subNdim, tsOutput);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          for (let i = 0; i < 1000; i++) {
            wasmModule!.extract_cholesky_submatrix(packed, keepDims, subNdim, wasmOutput);
          }
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'extract_cholesky_submatrix (1K)',
        category: 'GSPLATS PROCESSING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
        utility: true, // Not used in production - inlined by batch functions
      };
    },
    (size) => {
      // extract_visible_cholesky_3d - batch extraction for visible splats
      const ndim = 4;
      const cholesky = generateCholeskyFactors(size, ndim);
      const visibility = new Uint8Array(size);
      for (let i = 0; i < size; i++) visibility[i] = Math.random() > 0.5 ? 1 : 0;
      const displayDims = new Uint32Array([0, 1, 2]);
      const tsOutput = new Float32Array(size * 6);
      const wasmOutput = new Float32Array(size * 6);

      const tsTime = measureTime(
        () => {
          tsModule.extract_visible_cholesky_3d(
            cholesky,
            visibility,
            displayDims,
            ndim,
            size,
            tsOutput
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.extract_visible_cholesky_3d(
            cholesky,
            visibility,
            displayDims,
            ndim,
            size,
            wasmOutput
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'extract_visible_cholesky_3d',
        category: 'GSPLATS PROCESSING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
    (size) => {
      // compact_attenuated_amplitudes
      const amplitudes = generateRadii(size);
      const attenuation = new Float32Array(size);
      for (let i = 0; i < size; i++) attenuation[i] = Math.random();
      const visibility = new Uint8Array(size);
      for (let i = 0; i < size; i++) visibility[i] = Math.random() > 0.5 ? 1 : 0;
      const tsOutput = new Float32Array(size);
      const wasmOutput = new Float32Array(size);

      const tsTime = measureTime(
        () => {
          tsModule.compact_attenuated_amplitudes(
            amplitudes,
            attenuation,
            visibility,
            size,
            tsOutput
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      const wasmTime = measureTime(
        () => {
          wasmModule!.compact_attenuated_amplitudes(
            amplitudes,
            attenuation,
            visibility,
            size,
            wasmOutput
          );
        },
        CONFIG.iterations,
        CONFIG.warmupIterations
      );

      return {
        name: 'compact_attenuated_amplitudes',
        category: 'GSPLATS PROCESSING',
        tsTime,
        wasmTime,
        speedup: tsTime / wasmTime,
      };
    },
  ],
};

// ============================================================================
// Reporting
// ============================================================================

function printHeader(size: number): void {
  console.log('\n\x1b[1m\x1b[36m' + '='.repeat(70) + '\x1b[0m');
  console.log('\x1b[1m\x1b[36m  WASM vs TypeScript Performance Benchmark\x1b[0m');
  console.log('\x1b[1m\x1b[36m' + '='.repeat(70) + '\x1b[0m\n');

  const nodeVersion = process.version;
  console.log(`System: Node.js ${nodeVersion}, WASM module loaded \x1b[32m\u2713\x1b[0m`);
  console.log(
    `\nRunning benchmarks with ${size.toLocaleString()} elements, ${CONFIG.iterations} iterations each...\n`
  );
}

function printCategoryResults(category: string, results: BenchmarkResult[]): void {
  console.log(`\x1b[1m\x1b[33m${category}\x1b[0m`);
  console.log('\x1b[90m' + '-'.repeat(70) + '\x1b[0m');

  const nameWidth = 32;
  const colWidth = 12;

  // Header
  console.log(
    `${'Function'.padEnd(nameWidth)} ${'TypeScript'.padStart(colWidth)} ${'WASM'.padStart(colWidth)} ${'Speedup'.padStart(colWidth)}`
  );

  // Results
  for (const result of results) {
    let name = result.name;
    // Add utility marker for functions not used in production hot paths
    if (result.utility) {
      name = '\x1b[90m* ' + name + '\x1b[0m';
    }
    const displayName = name.length > nameWidth - 2 ? name.slice(0, nameWidth - 3) + '...' : name;
    console.log(
      `${displayName.padEnd(nameWidth + (result.utility ? 9 : 0))} ${formatTime(result.tsTime).padStart(colWidth)} ${formatTime(result.wasmTime).padStart(colWidth)} ${formatSpeedup(result.speedup).padStart(colWidth + 9)}`
    );
  }
  console.log('');
}

function printSummary(results: BenchmarkResult[]): void {
  console.log('\x1b[1m\x1b[36m' + '='.repeat(70) + '\x1b[0m');
  console.log('\x1b[1m\x1b[36m  SUMMARY\x1b[0m');
  console.log('\x1b[1m\x1b[36m' + '='.repeat(70) + '\x1b[0m\n');

  // Separate batch (production) vs utility functions
  const batchResults = results.filter((r) => !r.utility);
  const utilityResults = results.filter((r) => r.utility);

  // Batch function stats (what matters in production)
  const batchSpeedups = batchResults.map((r) => r.speedup);
  const batchAvg = batchSpeedups.reduce((a, b) => a + b, 0) / batchSpeedups.length;
  const batchMax = Math.max(...batchSpeedups);
  const batchMin = Math.min(...batchSpeedups);
  const batchMaxResult = batchResults.find((r) => r.speedup === batchMax)!;
  const batchMinResult = batchResults.find((r) => r.speedup === batchMin)!;

  console.log('\x1b[1m  Batch Functions (used in production):\x1b[0m');
  console.log(`   Functions:        ${batchResults.length}`);
  console.log(`   Average speedup:  ${formatSpeedup(batchAvg)}`);
  console.log(`   Fastest speedup:  ${formatSpeedup(batchMax)} (${batchMaxResult.name})`);
  console.log(`   Slowest speedup:  ${formatSpeedup(batchMin)} (${batchMinResult.name})`);
  console.log('');

  // Utility function stats (API completeness only)
  if (utilityResults.length > 0) {
    const utilitySpeedups = utilityResults.map((r) => r.speedup);
    const utilityAvg = utilitySpeedups.reduce((a, b) => a + b, 0) / utilitySpeedups.length;

    console.log('\x1b[90m  * Utility Functions (API completeness, not used in hot paths):\x1b[0m');
    console.log(`\x1b[90m   Functions:        ${utilityResults.length}\x1b[0m`);
    console.log(
      `\x1b[90m   Average speedup:  ${utilityAvg.toFixed(1)}x (expected <1x due to WASM call overhead)\x1b[0m`
    );
    console.log('');
  }

  // Explanation
  console.log(
    '\x1b[90m  Note: Utility functions (*) exist for API completeness and testing.\x1b[0m'
  );
  console.log(
    '\x1b[90m  In production, batch functions inline the math, avoiding per-call overhead.\x1b[0m'
  );
  console.log(
    '\x1b[90m  WASM call overhead dominates for trivial operations like lerp/distance.\x1b[0m'
  );
  console.log('');
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  // Parse command line args
  const sizeArg = process.argv.find((arg) => arg.startsWith('--size='));
  const sizeKey = sizeArg
    ? (sizeArg.split('=')[1] as keyof typeof CONFIG.sizes)
    : CONFIG.defaultSize;
  const size = CONFIG.sizes[sizeKey] || CONFIG.sizes[CONFIG.defaultSize];

  // Load modules
  const wasmLoaded = await loadModules();
  if (!wasmLoaded) {
    console.log('\n\x1b[33mCannot run benchmarks without WASM module.\x1b[0m');
    console.log('Build WASM with: make build-wasm\n');
    process.exit(1);
  }

  printHeader(size);

  // Run all benchmarks
  const allResults: BenchmarkResult[] = [];

  for (const [category, categoryBenchmarks] of Object.entries(benchmarks)) {
    const categoryResults: BenchmarkResult[] = [];

    for (const benchmark of categoryBenchmarks) {
      const result = benchmark(size);
      categoryResults.push(result);
      allResults.push(result);
    }

    printCategoryResults(category, categoryResults);
  }

  printSummary(allResults);
}

main().catch(console.error);
