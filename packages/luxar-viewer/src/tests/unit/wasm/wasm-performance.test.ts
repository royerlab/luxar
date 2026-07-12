/**
 * WASM vs TypeScript Performance Benchmarks
 *
 * Measures actual speedup of WASM implementations compared to TypeScript fallbacks.
 * Run with: pnpm test --run src/tests/unit/wasm/wasm-performance.test.ts
 *
 * Note: These are benchmarks, not unit tests. They measure performance but don't
 * assert correctness (that's covered by wasm-vs-typescript.test.ts). The audit
 * (wasm.md [O3][P9]) flagged that the header pointed to a non-existent
 * `.bench.ts` filename — header now matches the actual file path.
 */

import { describe, it, expect, beforeAll, vi } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import type { WasmModule } from '../../../wasm/types';
import { TypeScriptFallback } from '../../../wasm/typescript';

// Increase timeout for all benchmarks in this file - coverage instrumentation
// adds significant overhead to the tight loops used by performance benchmarks.
vi.setConfig({ testTimeout: 30_000 });

// Check if WASM files exist. Benchmarks are skipped when artifacts are absent
// unless LUXAR_REQUIRE_WASM_TESTS=1 is set, in which case a smoke test fails.
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../../public/wasm/luxar_wasm_bg.wasm');
const wasmFilesExist = existsSync(wasmJsPath) && existsSync(wasmBinaryPath);
const requireWasmTests = process.env.LUXAR_REQUIRE_WASM_TESTS === '1';

// Benchmark configuration
const WARMUP_ITERATIONS = 10;
const BENCHMARK_ITERATIONS = 100; // More iterations for stable timing

// Test data sizes (larger to get measurable times)
const SMALL_SIZE = 5_000;
const MEDIUM_SIZE = 50_000;
const LARGE_SIZE = 200_000;

interface BenchmarkResult {
  name: string;
  size: number;
  wasmTimeMs: number;
  tsTimeMs: number;
  speedup: number;
}

const results: BenchmarkResult[] = [];

/**
 * Run a benchmark comparing WASM and TypeScript implementations.
 */
function benchmark<T>(
  name: string,
  size: number,
  setupFn: () => { wasm: WasmModule; ts: TypeScriptFallback; args: T },
  wasmFn: (wasm: WasmModule, args: T) => void,
  tsFn: (ts: TypeScriptFallback, args: T) => void
): void {
  const { wasm, ts, args } = setupFn();

  // Warmup
  for (let i = 0; i < WARMUP_ITERATIONS; i++) {
    wasmFn(wasm, args);
    tsFn(ts, args);
  }

  // Benchmark WASM
  const wasmStart = performance.now();
  for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
    wasmFn(wasm, args);
  }
  const wasmTime = (performance.now() - wasmStart) / BENCHMARK_ITERATIONS;

  // Benchmark TypeScript
  const tsStart = performance.now();
  for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
    tsFn(ts, args);
  }
  const tsTime = (performance.now() - tsStart) / BENCHMARK_ITERATIONS;

  // performance.now() can quantize a fully-JIT'd 100-iteration loop to
  // 0ms (coarse timer resolution, fast machines, worker contention in
  // whole-suite runs). The old edge-case handling covered both-zero but
  // not tsTime=0 with wasmTime>0, which produced speedup=0 and failed
  // the `> 0` sanity assertions. Clamp both sides to ~timer resolution
  // so every degenerate combination yields a finite, positive ratio.
  const MIN_MEASURABLE_MS = 0.0005;
  const speedup = Math.max(tsTime, MIN_MEASURABLE_MS) / Math.max(wasmTime, MIN_MEASURABLE_MS);

  results.push({
    name,
    size,
    wasmTimeMs: wasmTime,
    tsTimeMs: tsTime,
    speedup,
  });
}

describe('WASM benchmark artifact requirement', () => {
  it.runIf(requireWasmTests)('has built WASM artifacts when benchmarks are required', () => {
    expect(wasmFilesExist).toBe(true);
  });
});

describe.skipIf(!wasmFilesExist)('WASM Performance Benchmarks', () => {
  let wasmModule: WasmModule;
  let tsModule: TypeScriptFallback;

  beforeAll(async () => {
    if (!wasmFilesExist) return;

    try {
      const wasmBinary = readFileSync(wasmBinaryPath);
      const wasm = await import(wasmJsPath);
      wasm.initSync({ module: wasmBinary });
      wasmModule = wasm as unknown as WasmModule;
      tsModule = new TypeScriptFallback();
      console.log('[Benchmark] WASM module loaded successfully');
    } catch (error) {
      console.error('Failed to load WASM module:', error);
    }
  });

  describe('decode_quantized_u8', () => {
    const runBenchmark = (size: number) => {
      benchmark(
        'decode_quantized_u8',
        size,
        () => {
          const data = new Uint8Array(size);
          for (let i = 0; i < size; i++) {
            data[i] = Math.floor(Math.random() * 256);
          }
          const output = new Float32Array(size);
          return { wasm: wasmModule, ts: tsModule, args: { data, output } };
        },
        (wasm, { data, output }) => {
          wasm.decode_quantized_u8(data, -10.0, 10.0, output);
        },
        (ts, { data, output }) => {
          ts.decode_quantized_u8(data, -10.0, 10.0, output);
        }
      );
    };

    it(`should benchmark with ${SMALL_SIZE.toLocaleString()} elements`, () => {
      runBenchmark(SMALL_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  decode_quantized_u8 (${SMALL_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });

    it(`should benchmark with ${MEDIUM_SIZE.toLocaleString()} elements`, () => {
      runBenchmark(MEDIUM_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  decode_quantized_u8 (${MEDIUM_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });

    it(`should benchmark with ${LARGE_SIZE.toLocaleString()} elements`, () => {
      runBenchmark(LARGE_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  decode_quantized_u8 (${LARGE_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });
  });

  describe('query_chunks_for_view', () => {
    const runBenchmark = (numChunks: number) => {
      const ndim = 3;
      benchmark(
        'query_chunks_for_view',
        numChunks,
        () => {
          // Create chunk bounds [minX, maxX, minY, maxY, minZ, maxZ] per chunk
          const chunkBounds = new Float32Array(numChunks * ndim * 2);
          for (let i = 0; i < numChunks; i++) {
            const x = (i % 100) * 10;
            const y = Math.floor(i / 100) * 10;
            const z = 0;
            chunkBounds[i * 6] = x; // minX
            chunkBounds[i * 6 + 1] = x + 10; // maxX
            chunkBounds[i * 6 + 2] = y; // minY
            chunkBounds[i * 6 + 3] = y + 10; // maxY
            chunkBounds[i * 6 + 4] = z; // minZ
            chunkBounds[i * 6 + 5] = z + 10; // maxZ
          }
          const slicePos = new Float32Array([500, 500, 5]);
          const tolerance = new Float32Array([Infinity, Infinity, Infinity]);
          const output = new Uint32Array(numChunks);
          return {
            wasm: wasmModule,
            ts: tsModule,
            args: { chunkBounds, slicePos, tolerance, ndim, numChunks, output },
          };
        },
        (wasm, { chunkBounds, slicePos, tolerance, ndim, numChunks, output }) => {
          wasm.query_chunks_for_view(chunkBounds, slicePos, tolerance, ndim, numChunks, output);
        },
        (ts, { chunkBounds, slicePos, tolerance, ndim, numChunks, output }) => {
          ts.query_chunks_for_view(chunkBounds, slicePos, tolerance, ndim, numChunks, output);
        }
      );
    };

    it(`should benchmark with ${SMALL_SIZE.toLocaleString()} chunks`, () => {
      runBenchmark(SMALL_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  query_chunks_for_view (${SMALL_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });

    it(`should benchmark with ${MEDIUM_SIZE.toLocaleString()} chunks`, () => {
      runBenchmark(MEDIUM_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  query_chunks_for_view (${MEDIUM_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });
  });

  describe('clip_segments_batch', () => {
    const runBenchmark = (numSegments: number) => {
      const ndim = 4;
      const numVertices = numSegments * 2; // Two vertices per segment
      benchmark(
        'clip_segments_batch',
        numSegments,
        () => {
          // Create vertex positions array [numVertices * ndim]
          const positions = new Float32Array(numVertices * ndim);
          for (let i = 0; i < numVertices; i++) {
            positions[i * ndim] = Math.random() * 100;
            positions[i * ndim + 1] = Math.random() * 100;
            positions[i * ndim + 2] = Math.random() * 100;
            positions[i * ndim + 3] = Math.random() * 10;
          }
          // Create segment indices [numSegments * 2]
          const segments = new Uint32Array(numSegments * 2);
          for (let i = 0; i < numSegments; i++) {
            segments[i * 2] = i * 2; // Start vertex
            segments[i * 2 + 1] = i * 2 + 1; // End vertex
          }
          const slicePos = new Float32Array([50, 50, 50, 5]);
          const tolerance = new Float32Array([Infinity, Infinity, Infinity, 2.0]);
          const displayDims = new Uint32Array([0, 1, 2]);
          const visibility = new Uint8Array(numSegments);
          const t1Params = new Float32Array(numSegments);
          const t2Params = new Float32Array(numSegments);
          return {
            wasm: wasmModule,
            ts: tsModule,
            args: {
              positions,
              segments,
              slicePos,
              tolerance,
              displayDims,
              ndim,
              numSegments,
              visibility,
              t1Params,
              t2Params,
            },
          };
        },
        (
          wasm,
          {
            positions,
            segments,
            slicePos,
            tolerance,
            displayDims,
            ndim,
            numSegments,
            visibility,
            t1Params,
            t2Params,
          }
        ) => {
          wasm.clip_segments_batch(
            positions,
            segments,
            slicePos,
            tolerance,
            displayDims,
            ndim,
            numSegments,
            visibility,
            t1Params,
            t2Params
          );
        },
        (
          ts,
          {
            positions,
            segments,
            slicePos,
            tolerance,
            displayDims,
            ndim,
            numSegments,
            visibility,
            t1Params,
            t2Params,
          }
        ) => {
          ts.clip_segments_batch(
            positions,
            segments,
            slicePos,
            tolerance,
            displayDims,
            ndim,
            numSegments,
            visibility,
            t1Params,
            t2Params
          );
        }
      );
    };

    it(`should benchmark with ${SMALL_SIZE.toLocaleString()} segments`, () => {
      runBenchmark(SMALL_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  clip_segments_batch (${SMALL_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });

    it(`should benchmark with ${MEDIUM_SIZE.toLocaleString()} segments`, () => {
      runBenchmark(MEDIUM_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  clip_segments_batch (${MEDIUM_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });
  });

  describe('calculate_effective_radii', () => {
    const runBenchmark = (numPoints: number) => {
      const ndim = 5;
      benchmark(
        'calculate_effective_radii',
        numPoints,
        () => {
          const positions = new Float32Array(numPoints * ndim);
          const radii = new Float32Array(numPoints);
          for (let i = 0; i < numPoints; i++) {
            positions[i * ndim] = Math.random() * 100;
            positions[i * ndim + 1] = Math.random() * 100;
            positions[i * ndim + 2] = Math.random() * 100;
            positions[i * ndim + 3] = Math.random() * 10;
            positions[i * ndim + 4] = Math.floor(Math.random() * 5); // Discrete dim
            radii[i] = 1.0 + Math.random() * 2;
          }
          const displayDims = new Uint32Array([0, 1, 2]);
          const slicePos = new Float32Array([50, 50, 50, 5, 2]);
          const spatialExtend = new Uint8Array([1, 1, 1, 1, 0]); // Last dim is discrete
          const output = new Float32Array(numPoints);
          return {
            wasm: wasmModule,
            ts: tsModule,
            args: {
              positions,
              radii,
              displayDims,
              slicePos,
              spatialExtend,
              ndim,
              numPoints,
              output,
            },
          };
        },
        (
          wasm,
          { positions, radii, displayDims, slicePos, spatialExtend, ndim, numPoints, output }
        ) => {
          wasm.calculate_effective_radii(
            positions,
            radii,
            displayDims,
            slicePos,
            spatialExtend,
            ndim,
            numPoints,
            output
          );
        },
        (
          ts,
          { positions, radii, displayDims, slicePos, spatialExtend, ndim, numPoints, output }
        ) => {
          ts.calculate_effective_radii(
            positions,
            radii,
            displayDims,
            slicePos,
            spatialExtend,
            ndim,
            numPoints,
            output
          );
        }
      );
    };

    it(`should benchmark with ${SMALL_SIZE.toLocaleString()} points`, () => {
      runBenchmark(SMALL_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  calculate_effective_radii (${SMALL_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });

    it(`should benchmark with ${MEDIUM_SIZE.toLocaleString()} points`, () => {
      runBenchmark(MEDIUM_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  calculate_effective_radii (${MEDIUM_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });

    it(`should benchmark with ${LARGE_SIZE.toLocaleString()} points`, () => {
      runBenchmark(LARGE_SIZE);
      const result = results[results.length - 1];
      console.log(
        `  calculate_effective_radii (${LARGE_SIZE.toLocaleString()}): ` +
          `WASM=${result.wasmTimeMs.toFixed(3)}ms, ` +
          `TS=${result.tsTimeMs.toFixed(3)}ms, ` +
          `speedup=${result.speedup.toFixed(2)}x`
      );
      expect(result.speedup).toBeGreaterThan(0);
    });
  });

  // Summary at the end
  describe('Summary', () => {
    it('should print benchmark summary', () => {
      console.log('\n========== WASM vs TypeScript Performance Summary ==========\n');
      console.log('| Function | Size | WASM (ms) | TS (ms) | Speedup |');
      console.log('|----------|------|-----------|---------|---------|');
      for (const r of results) {
        console.log(
          `| ${r.name.padEnd(30)} | ${r.size.toLocaleString().padStart(7)} | ` +
            `${r.wasmTimeMs.toFixed(3).padStart(9)} | ${r.tsTimeMs.toFixed(3).padStart(7)} | ` +
            `${r.speedup.toFixed(2).padStart(6)}x |`
        );
      }
      console.log('\n============================================================\n');

      // Calculate average speedup
      const avgSpeedup = results.reduce((sum, r) => sum + r.speedup, 0) / results.length;
      console.log(`Average speedup: ${avgSpeedup.toFixed(2)}x`);

      expect(results.length).toBeGreaterThan(0);
    });
  });
});
