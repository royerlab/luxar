/**
 * FULL-PATH depth-sort throughput bench (perf campaign, L3 lever harness).
 *
 * `perf-budget.test.ts` benchmarks the RAW `sort_splats_by_depth` kernel
 * through the wasm-bindgen shim on pre-allocated buffers. This bench
 * instead exercises the FULL worker-side `sortNode` path — registry
 * lookup, fresh `Uint32Array` ordering allocation per sort, and the
 * backend call with its wasm-bindgen boundary costs (per-sort
 * 12 B/splat centers copy-in + 4 B/splat ordering copy-in +
 * 4 B/splat copy-out + 3 mallocs) — so before/after comparisons of
 * boundary-copy optimizations show up here even when the raw kernel
 * number is unchanged.
 *
 * RECORD-ONLY: the numbers are console.log'd for manual before/after
 * comparison; the only assertion is an extremely generous floor
 * (≥ 10 M splats/s) so the test never flakes. It is NOT a regression
 * gate — that's perf-budget's job for the raw kernel.
 *
 * Runs in the opt-in perf suite (`pnpm test:perf`, vitest.perf.config.ts)
 * and skips cleanly when `public/wasm/luxar_wasm_bg.wasm` is absent.
 */
import { describe, it, expect, beforeAll, vi } from 'vitest';
import { existsSync, readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import type { WasmModule } from '../../../wasm/types';

vi.mock('../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  Modules: new Proxy({}, { get: (_t, p) => String(p) }),
}));

import { registerNode, sortNode } from '../../../workers/sort-worker/sorting';
import type { SortWorkerCtx } from '../../../workers/sort-worker/state';

/**
 * Extremely generous floor — the measured full path is well above
 * 40 M splats/s on any healthy build; 10 M/s only catches catastrophic
 * breakage (debug WASM, TS fallback silently selected, comparison sort).
 */
const MIN_SPLATS_PER_SECOND = 10e6;
/** Iterations averaged per run (each with a DIFFERENT rotation). */
const ITERATIONS = 5;
/** Warmup iterations before each run. */
const WARMUP = 2;
/** Independent runs; the reported number is the median across runs. */
const RUNS = 5;
/** Splat counts to bench. */
const SIZES = [1_000_000, 5_000_000];

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const wasmJsPath = join(__dirname, '../../../../public/wasm/luxar_wasm.js');
const wasmBinaryPath = join(__dirname, '../../../../public/wasm/luxar_wasm_bg.wasm');
const wasmAvailable = existsSync(wasmJsPath) && existsSync(wasmBinaryPath);

let wasmModule: WasmModule | null = null;

beforeAll(async () => {
  if (!wasmAvailable) return;
  // Same loader shape as perf-budget: read the .wasm bytes, initSync via
  // the JS shim — bypasses bundler path resolution in vitest/jsdom.
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

/** Spatially-incoherent centers — same generator shape as perf-budget. */
function generateCenters(count: number): Float32Array {
  const centers = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    for (let d = 0; d < 3; d++) {
      centers[i * 3 + d] = Math.sin(i * 0.037 + d * 1.23) * 10;
    }
  }
  return centers;
}

/**
 * Column-major model-view: rotation about Y by `theta`, then translate to
 * z = -100. Centers span |p| ≤ 10·√3 ≈ 17.3, so every view-space z lands
 * in [-117, -83] — all splats strictly in front of the camera and the
 * sort takes the full three-pass path, never the identity fallback.
 */
function rotatingModelView(theta: number): Float32Array {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  // prettier-ignore
  return new Float32Array([
    c, 0, -s, 0,
    0, 1, 0, 0,
    s, 0, c, 0,
    0, 0, -100, 1,
  ]);
}

const describeIfWasm = wasmAvailable ? describe : describe.skip;

describeIfWasm('Full-path sortNode throughput (record-only bench)', () => {
  for (const size of SIZES) {
    it(`sortNode full path @ ${size.toLocaleString()} splats`, { timeout: 300_000 }, () => {
      // Fresh worker ctx with the loaded module injected — exactly what
      // initialize() does in the live worker (ctx.wasm = await initWasm()),
      // so requireWasm() resolves to compiled WASM here.
      const ctx: SortWorkerCtx = { wasm: wasmModule, nodes: new Map() };
      registerNode(ctx, {
        nodeId: 'bench',
        generation: 1,
        centers3: generateCenters(size),
        count: size,
      });

      // A DIFFERENT rotation each sortNode call: precompute a pool of
      // model-views and cycle through it so no matrix is reused
      // back-to-back within a run (defeats any depth-cache shortcut and
      // matches the real camera-orbit workload).
      const poolSize = 64;
      const modelViews: Float32Array[] = [];
      for (let i = 0; i < poolSize; i++) {
        modelViews.push(rotatingModelView((i / poolSize) * 2 * Math.PI + 0.05));
      }
      let call = 0;

      const runOnce = () => {
        const result = sortNode(ctx, {
          nodeId: 'bench',
          generation: 1,
          modelView: modelViews[call++ % poolSize],
        });
        if (!result || result.ordering.length !== size) {
          throw new Error('sortNode returned no/short ordering — bench harness broken');
        }
      };

      const times: number[] = [];
      for (let run = 0; run < RUNS; run++) {
        times.push(measure(runOnce));
      }
      const medianMs = median(times);
      const splatsPerSecond = size / (medianMs / 1000);

      console.log(
        `sortNode full path: ${(splatsPerSecond / 1e6).toFixed(1)} M splats/s ` +
          `(median ${medianMs.toFixed(2)} ms @ ${size.toLocaleString()} splats, ` +
          `median of ${RUNS} runs × ${ITERATIONS} iters, rotating model-view)`
      );

      expect(
        splatsPerSecond,
        `sortNode full path: ${(splatsPerSecond / 1e6).toFixed(1)} M splats/s ` +
          `(median ${medianMs.toFixed(2)} ms for ${size.toLocaleString()} splats; ` +
          `generous floor ${(MIN_SPLATS_PER_SECOND / 1e6).toFixed(0)} M/s — record-only bench)`
      ).toBeGreaterThanOrEqual(MIN_SPLATS_PER_SECOND);
    });
  }
});

if (!wasmAvailable) {
  describe.skip('Full-path sortNode bench (WASM module not built — run `make build-wasm`)', () => {
    it('skipped', () => {
      // Surfaced as a skipped test so the absence is visible in output.
    });
  });
}
