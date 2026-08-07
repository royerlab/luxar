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
 * someone happened to run `pnpm bench:wasm`. This test turns the budget
 * into an assertion: we run the same shape of comparison the benchmark
 * suite does and fail if any covered hot path is no longer comfortably
 * faster than the TS fallback.
 *
 * It lives in the opt-in perf suite (`pnpm test:perf`) — the ratios are
 * timing-sensitive under parallel test-file load, so it is excluded from
 * the gating unit run. It also skips cleanly when
 * `public/wasm/luxar_wasm_bg.wasm` is absent (dev box without
 * `make build-wasm`), so this is opportunistic enforcement rather than a
 * hard prerequisite.
 *
 * **Threshold.** 1.15×, not 2×. The plan called for 2× but in practice
 * V8 autovectorizes the simpler TS fallbacks well enough that a healthy
 * WASM build lands at ~1.9× on most workloads in jsdom while some
 * occasionally land as low as 1.20× under concurrent test-file CPU load.
 * 1.15× still catches real regressions (a SIMD/alloc-in-loop slip would
 * push the ratio toward 1× or below) without flaking. The 2× stretch
 * target is checked manually via
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

    it('sort_splats_by_depth', { timeout: 60_000 }, () => {
      // Depth-sorting Phase 2 budget (spec §2/§5). Measured reality
      // (2026-07, M-class dev machine, release WASM): ~74 M splats/s at
      // 1 M WORST-CASE spatially-incoherent splats (~96-114 M/s native;
      // the gap is WASM execution overhead plus the wasm-bindgen boundary
      // copies). Real gsplat data is Morton-coherent and sorts faster.
      // The floor is set at 50 M/s — comfortably below the measured
      // worst case so parallel-test CPU load can't flake it, while still
      // catching real regressions (a debug build, an accidental
      // comparison sort, or an alloc-per-element slip all land far
      // below it). Spec §5's original 100 M/s target was a
      // pre-implementation estimate; the measured delta is recorded in
      // the spec's implementation-deltas note.
      const MIN_SPLATS_PER_SECOND = 50e6;
      const centers3 = generatePositions(SIZE, 3);
      // Push everything in front of the camera (view z < 0) so the sort
      // takes the full three-pass path, not the identity fallback.
      const modelView = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -100, 1]);
      const tsOut = new Uint32Array(SIZE);
      const wasmOut = new Uint32Array(SIZE);

      const r = medianSpeedup(
        () => tsModule.sort_splats_by_depth(centers3, modelView, tsOut, SIZE),
        () => wasmModule!.sort_splats_by_depth(centers3, modelView, wasmOut, SIZE)
      );

      expect(r.speedup, reportSpeedup('sort_splats_by_depth', r)).toBeGreaterThanOrEqual(
        MIN_SPEEDUP
      );

      const wasmTimes: number[] = [];
      for (let run = 0; run < RUNS; run++) {
        wasmTimes.push(
          measure(() => wasmModule!.sort_splats_by_depth(centers3, modelView, wasmOut, SIZE))
        );
      }
      const medianMs = median(wasmTimes);
      const splatsPerSecond = SIZE / (medianMs / 1000);
      console.log(
        `sort_splats_by_depth: ${(splatsPerSecond / 1e6).toFixed(0)} M splats/s ` +
          `(median ${medianMs.toFixed(2)} ms @ ${SIZE.toLocaleString()}; speedup ${r.speedup.toFixed(2)}×)`
      );
      expect(
        splatsPerSecond,
        `sort_splats_by_depth: ${(splatsPerSecond / 1e6).toFixed(0)} M splats/s ` +
          `(median ${medianMs.toFixed(2)} ms for ${SIZE.toLocaleString()} splats; ` +
          `floor ${(MIN_SPLATS_PER_SECOND / 1e6).toFixed(0)} M/s)`
      ).toBeGreaterThanOrEqual(MIN_SPLATS_PER_SECOND);
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
