/**
 * The range-loader's worker decode paths send ONE worker RPC per attribute,
 * not one per range.
 *
 * Measured on a Points timelapse: the per-channel path sent one
 * `decodePerChannel` message per range — 40k messages in a 6 s playback loop,
 * most of them a few hundred elements each. All ranges of one attribute share
 * the same scales and decode element-wise, so they pack into a single buffer
 * and a single call. These tests pin that (N ranges -> 1 call) for every
 * worker-decoded encoding the range loader owns (per-channel, quantized,
 * log-scalar, geolog-scalar, LUT scalar + row), and that the batched result is
 * identical to the main-thread decode — including the per-channel column
 * phase of a range that starts mid-row and a malformed (short) range.
 *
 * The spy pool runs the REAL worker task bodies in-process on the TypeScript
 * WASM fallback, so the only thing faked is the thread boundary.
 */

import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import type { LoadRange } from '../../../../../data/loaders/base-types';

vi.mock('../../../../../config', () => ({
  config: { dataLoading: { performance: { useWebWorkers: true } } },
}));

const calls: string[] = [];
const workerApi: Record<string, (p: never) => Promise<Float32Array>> = {};

vi.mock('../../../../../workers/worker-pool', () => ({
  getWorkerPool: () => ({
    runWithTimeout: (
      op: string,
      _kind: string,
      fn: (api: typeof workerApi) => Promise<Float32Array>
    ): Promise<Float32Array> => {
      calls.push(op);
      return fn(workerApi);
    },
  }),
}));

let backing: Uint8Array | Uint16Array = new Uint8Array(0);
let shortRange: { start: number; drop: number } | null = null;

vi.mock('zarrita', async () => {
  const actual = await vi.importActual<typeof import('zarrita')>('zarrita');
  return {
    ...actual,
    slice: (start: number | null, end?: number | null) => ({ start, end }),
    // A zarr read of rows [start, end) of a (rows, cols) or (rows,) array.
    get: vi.fn(async (array: { shape: number[] }, sel: { start: number; end: number }[]) => {
      const rowSize = array.shape.slice(1).reduce((a, b) => a * b, 1);
      let stop = sel[0].end * rowSize;
      if (shortRange && shortRange.start === sel[0].start) stop -= shortRange.drop;
      return { data: backing.slice(sel[0].start * rowSize, stop) };
    }),
  };
});

import {
  ArrayDecoder,
  ArrayRefRegistry,
  type ArrayMetadata,
} from '../../../../../data/array-decoder/decoder';
import { loadPerChannel } from '../../../../../data/loaders/spatial-query/range-loader/perchannel';
import { loadQuantized } from '../../../../../data/loaders/spatial-query/range-loader/quantized';
import { loadLUT } from '../../../../../data/loaders/spatial-query/range-loader/lut';
import { getFallback } from '../../../../../wasm';
import type { WasmCtx } from '../../../../../workers/data-worker/state';
import { decodePerChannel } from '../../../../../workers/data-worker/decode/perchannel';
import { decodeQuantized } from '../../../../../workers/data-worker/decode/quantized';
import { decodeLogScalar } from '../../../../../workers/data-worker/decode/log-scalar';
import { decodeGeologScalar } from '../../../../../workers/data-worker/decode/geolog-scalar';
import { decodeLUT } from '../../../../../workers/data-worker/decode/lut';

const RANGES: LoadRange[] = [
  { start: 0, end: 7 },
  { start: 11, end: 12 },
  { start: 20, end: 45 },
  { start: 50, end: 53 },
  { start: 60, end: 100 },
];
const ROWS = RANGES.reduce((s, r) => s + r.end - r.start, 0);

function ctx(workerThreshold: number) {
  return {
    config: { workerThreshold, logModule: 'test' as never },
    verbose: false,
    decoder: new ArrayDecoder(new ArrayRefRegistry()),
    signal: null,
  };
}

function fillBacking(n: number, bits: 8 | 16): void {
  backing = bits === 8 ? new Uint8Array(n) : new Uint16Array(n);
  for (let i = 0; i < n; i++) backing[i] = (i * 2654435761) % (bits === 8 ? 256 : 65536);
}

const array = (dtype: string, shape: number[]) => ({ dtype, shape, attrs: {} }) as never;

type Loader = (threshold: number, output: Float32Array) => Promise<number>;

/** Run `load` through the worker (threshold 0) and on the main thread (Infinity). */
async function workerVsMain(load: Loader, outLen: number) {
  calls.length = 0;
  const viaWorker = new Float32Array(outLen);
  await load(0, viaWorker);
  const workerCalls = [...calls];
  calls.length = 0;
  const onMain = new Float32Array(outLen);
  await load(Infinity, onMain);
  expect(calls).toEqual([]);
  return { viaWorker, onMain, workerCalls };
}

describe('range-loader worker decode batching', () => {
  beforeAll(() => {
    const wctx: WasmCtx = { wasm: getFallback(), tsFallback: getFallback() };
    workerApi.decodePerChannel = (p: never) => decodePerChannel(wctx, p);
    workerApi.decodeQuantized = (p: never) => decodeQuantized(wctx, p);
    workerApi.decodeLogScalar = (p: never) => decodeLogScalar(wctx, p);
    workerApi.decodeGeologScalar = (p: never) => decodeGeologScalar(wctx, p);
    workerApi.decodeLUT = (p: never) => decodeLUT(wctx, p);
  });

  beforeEach(() => {
    shortRange = null;
  });

  for (const bits of [8, 16] as const) {
    for (const kind of ['linear', 'log', 'signed_log', 'geolog'] as const) {
      it(`per-channel ${kind} u${bits}: ${RANGES.length} ranges -> 1 worker call, bit-identical`, async () => {
        // 1D storage with cols=3: ranges of 7/1/25/3/40 rows start mid-row,
        // so each range's column phase is non-zero — the case a batched pack
        // must keep exact.
        fillBacking(100, bits);
        const cols = 3;
        const attrs: ArrayMetadata = {
          encoding: {
            name: `${kind}_perchannel_u${bits}`,
            bits,
            col_lo: [-2, 0.5, 1],
            col_hi: [3, 4.5, 9],
            zero_level: kind !== 'linear' && kind !== 'geolog',
          },
        } as ArrayMetadata;
        const load: Loader = (t, out) =>
          loadPerChannel(ctx(t), array(`uint${bits}`, [100]), attrs, RANGES, out, cols);
        const { viaWorker, onMain, workerCalls } = await workerVsMain(load, ROWS);
        expect(workerCalls).toEqual(['decodePerChannel']);
        expect(viaWorker).toEqual(onMain);
      });
    }
  }

  it('per-channel keeps a short (malformed) range span unfilled, like the main thread', async () => {
    fillBacking(100 * 2, 16);
    shortRange = { start: 20, drop: 3 };
    const attrs = {
      encoding: { name: 'linear_perchannel_u16', bits: 16, col_lo: [0, 1], col_hi: [1, 2] },
    } as ArrayMetadata;
    const load: Loader = (t, out) =>
      loadPerChannel(ctx(t), array('uint16', [100, 2]), attrs, RANGES, out, 2);
    const { viaWorker, onMain, workerCalls } = await workerVsMain(load, ROWS * 2);
    expect(workerCalls).toEqual(['decodePerChannel']);
    expect(viaWorker).toEqual(onMain);
  });

  const scalarCases: [string, ArrayMetadata, string][] = [
    [
      'quantized',
      { encoding: { name: 'bounded_scalar_uint16', bounds: [0.1, 5.0] } } as ArrayMetadata,
      'decodeQuantized',
    ],
    [
      'log-scalar',
      { encoding: { name: 'log_scalar_uint16', max_log: 3.5 } } as ArrayMetadata,
      'decodeLogScalar',
    ],
    [
      'geolog-scalar',
      {
        encoding: { name: 'geolog_scalar_uint16', min_log: -1.5, max_log: 2.5 },
      } as ArrayMetadata,
      'decodeGeologScalar',
    ],
  ];
  for (const [label, attrs, op] of scalarCases) {
    it(`${label}: ${RANGES.length} ranges -> 1 worker call, bit-identical`, async () => {
      fillBacking(100, 16);
      const load: Loader = (t, out) =>
        loadQuantized(ctx(t), array('uint16', [100]), attrs, RANGES, out);
      const { viaWorker, onMain, workerCalls } = await workerVsMain(load, ROWS);
      expect(workerCalls).toEqual([op]);
      expect(viaWorker).toEqual(onMain);
    });
  }

  it('LUT scalar: N ranges -> 1 worker call, bit-identical', async () => {
    fillBacking(100, 8);
    backing = backing.map((v) => v % 4) as Uint8Array;
    const attrs = {
      encoding: { name: 'lut_uint8', lut: [0.5, 1.5, 2.5, 3.5], lut_mode: 'scalar' },
    } as ArrayMetadata;
    const load: Loader = (t, out) => loadLUT(ctx(t), array('uint8', [100]), attrs, RANGES, out);
    const { viaWorker, onMain, workerCalls } = await workerVsMain(load, ROWS);
    expect(workerCalls).toEqual(['decodeLUT']);
    expect(viaWorker).toEqual(onMain);
  });

  it('LUT row (k=3): N ranges -> 1 worker call, bit-identical', async () => {
    fillBacking(100, 8);
    backing = backing.map((v) => v % 2) as Uint8Array;
    shortRange = { start: 50, drop: 1 };
    const attrs = {
      encoding: {
        name: 'lut_uint8',
        lut: [
          [1, 0, 0],
          [0, 0.5, 1],
        ],
        lut_mode: 'row',
        original_shape: [100, 3],
      },
    } as ArrayMetadata;
    const load: Loader = (t, out) => loadLUT(ctx(t), array('uint8', [100]), attrs, RANGES, out);
    const { viaWorker, onMain, workerCalls } = await workerVsMain(load, ROWS * 3);
    expect(workerCalls).toEqual(['decodeLUT']);
    expect(viaWorker).toEqual(onMain);
  });
});
