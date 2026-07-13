/**
 * Performance-budget assertions for the WASM hot paths.
 *
 * The viewer relies on the WASM module to keep projection (with its
 * built-in nD visibility/culling) and decode workloads fast enough for
 * ~60 fps interaction at 100 K – 10 M elements. The TypeScript fallbacks
 * in `src/wasm/typescript/` are correct but slow; we ship them so a
 * missing WASM build does not break the app, not because we are happy
 * running on them.
 *
 * If a regression in the Rust source landed that pulled WASM down close to
 * the TS implementation (e.g. accidentally allocating in the hot loop or
 * regressing a SIMD path), a benchmark run would notice — but only if
 * someone happened to run `pnpm bench:wasm`. This test makes the budget
 * automatic: we run the same shape of comparison the benchmark suite does
 * and fail the build if any covered hot path is no longer comfortably
 * faster than the TS fallback.
 *
 * Skipped cleanly when `public/wasm/luxar_wasm_bg.wasm` is absent (CI or
 * dev box without `make build-wasm`), so this is opportunistic enforcement
 * rather than a hard prerequisite.
 *
 * **Threshold.** 1.15×, not 2×. The plan called for 2× but in practice
 * V8 autovectorizes the simpler TS fallbacks well enough that a healthy
 * WASM build lands at ~1.9× on most workloads in jsdom while gsplats
 * Cholesky/attenuation occasionally lands at 1.20× under concurrent
 * test-file CPU load. 1.15× still catches real regressions (a
 * SIMD/alloc-in-loop slip would push the ratio toward 1× or below)
 * without flaking. The 2× stretch target is checked manually via
 * `pnpm bench:wasm` (local/manual; not currently wired into a CI
 * workflow). A scheduled perf workflow is a future follow-up.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { TypeScriptFallback } from '../../../wasm/typescript';
import type { WasmModule } from '../../../wasm/types';

/** Minimum acceptable WASM speedup over the TypeScript fallback. See file comment. */
const MIN_SPEEDUP = 1.15;
/** Iterations averaged per run. */
const ITERATIONS = 5;
/** Warmup iterations before each run. */
const WARMUP = 2;
/**
 * Number of independent runs taken; the test asserts the **median**
 * speedup across runs. Two outlier runs cannot flip the verdict, which
 * keeps the test stable when vitest is squeezing CPU under parallel
 * test-file load.
 */
const RUNS = 5;
/**
 * Working-set size per benchmark. 1 M elements puts every covered hot
 * path comfortably above ~5 ms per call, well past the V8 timer
 * resolution and JIT warmup noise that dominates smaller workloads.
 */
const SIZE = 1_000_000;

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../../public/wasm/luxar_wasm_bg.wasm');
const wasmAvailable = existsSync(wasmJsPath) && existsSync(wasmBinaryPath);

let wasmModule: WasmModule | null = null;
let tsModule: WasmModule;

beforeAll(async () => {
  tsModule = new TypeScriptFallback();
  if (!wasmAvailable) return;

  // Use the same loader shape as the benchmark script: read the .wasm
  // bytes, then `initSync` from the JS shim. This bypasses any bundler
  // path resolution that vitest/jsdom would mangle.
  const wasmBinary = readFileSync(wasmBinaryPath);
  const wasm = await import(wasmJsPath);
  wasm.initSync({ module: wasmBinary });
  wasmModule = wasm as unknown as WasmModule;
});

function measure(fn: () => void): number {
  for (let i = 0; i < WARMUP; i++) fn();
  const start = performance.now();
  for (let i = 0; i < ITERATIONS; i++) fn();
  return (performance.now() - start) / ITERATIONS;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * Run `RUNS` independent timing pairs and return the median speedup ratio.
 * Each run does its own warmup so the JIT state is consistent across runs.
 */
function medianSpeedup(
  tsFn: () => void,
  wasmFn: () => void
): { speedup: number; tsTime: number; wasmTime: number } {
  const ratios: number[] = [];
  let lastTs = 0;
  let lastWasm = 0;
  for (let r = 0; r < RUNS; r++) {
    const tsTime = measure(tsFn);
    const wasmTime = measure(wasmFn);
    ratios.push(tsTime / wasmTime);
    lastTs = tsTime;
    lastWasm = wasmTime;
  }
  return { speedup: median(ratios), tsTime: lastTs, wasmTime: lastWasm };
}

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

const describeIfWasm = wasmAvailable ? describe : describe.skip;

function reportSpeedup(
  name: string,
  r: { speedup: number; tsTime: number; wasmTime: number }
): string {
  return `${name}: median speedup ${r.speedup.toFixed(2)}× (last: ts=${r.tsTime.toFixed(2)}ms, wasm=${r.wasmTime.toFixed(2)}ms)`;
}

describeIfWasm(
  `WASM perf budget — each hot path must have median speedup ≥ ${MIN_SPEEDUP}×`,
  () => {
    // Long timeout: under `--coverage` the v8 instrumentation slows the
    // TS path enough that the 1 M-element work exceeds the default 5 s test
    // budget. The ratio measurement is meaningless under coverage anyway
    // (TS is instrumented, WASM binary is not), but bumping the timeout
    // keeps the suite from going red.
    it('calculate_effective_radii', { timeout: 60_000 }, () => {
      const ndim = 6;
      const positions = generatePositions(SIZE, ndim);
      const radii = generateRadii(SIZE);
      const displayDims = new Uint32Array([0, 1, 2]);
      const slicePos = new Float32Array(ndim).fill(0);
      const spatialExtend = new Uint8Array(ndim).fill(1);
      const tsOut = new Float32Array(SIZE);
      const wasmOut = new Float32Array(SIZE);

      const r = medianSpeedup(
        () =>
          tsModule.calculate_effective_radii(
            positions,
            radii,
            displayDims,
            slicePos,
            spatialExtend,
            ndim,
            SIZE,
            tsOut
          ),
        () =>
          wasmModule!.calculate_effective_radii(
            positions,
            radii,
            displayDims,
            slicePos,
            spatialExtend,
            ndim,
            SIZE,
            wasmOut
          )
      );

      expect(r.speedup, reportSpeedup('calculate_effective_radii', r)).toBeGreaterThanOrEqual(
        MIN_SPEEDUP
      );
    });
  }
);

if (!wasmAvailable) {
  describe.skip('WASM perf budget (WASM module not built — run `make build-wasm`)', () => {
    it('skipped', () => {
      // Surfaced as a skipped test so the absence is visible in test output.
    });
  });
}
