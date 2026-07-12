/**
 * Direct tests for the worker `decodePerChannel` entry point — the WASM-side
 * decoder of the per-channel family (`linear_perchannel_*` /
 * `log_perchannel_*` / `signed_log_perchannel_*`).
 *
 * Uses the TypeScript reference backend as the ctx module (numerically
 * identical to compiled WASM by the parity suite in
 * `wasm/wasm-vs-typescript.test.ts`), so these tests focus on the worker
 * boundary: dispatch, the column-phase contract, and fail-loud validation.
 */

import { describe, expect, it } from 'vitest';
import { TypeScriptFallback } from '../../../../wasm/typescript';
import type { WasmCtx } from '../../../../workers/data-worker/state';
import { decodePerChannel } from '../../../../workers/data-worker/decode/perchannel';
import { ArrayDecoder } from '../../../../data/array-decoder/decoder';

const ctx: WasmCtx = {
  wasm: new TypeScriptFallback(),
  tsFallback: null,
};

const colLo = new Float64Array([-1.5, 0.25]);
const colHi = new Float64Array([2.0, 4.5]);

describe('decodePerChannel — worker entry point', () => {
  it('decodes signed-log zero_level identically to the main-thread dequant', async () => {
    const data = new Uint16Array([0, 1, 30000, 65535, 0, 42]);
    const decoded = await decodePerChannel(ctx, {
      data,
      kind: 'signed_log',
      colLo,
      colHi,
      zeroLevel: true,
      colOffset: 0,
      bits: 16,
    });

    const dequant = ArrayDecoder.makePerChannelDequant(
      {
        name: 'signed_log_perchannel_u16',
        bits: 16,
        col_lo: Array.from(colLo),
        col_hi: Array.from(colHi),
        zero_level: true,
      },
      2
    );
    for (let i = 0; i < data.length; i++) {
      expect(decoded[i]).toBe(Math.fround(dequant(data[i], i % 2)));
    }
    expect(decoded[0]).toBe(0); // reserved zero level
    expect(decoded[4]).toBe(0);
  });

  it('honors the column phase (colOffset) for mid-array ranges', async () => {
    // A range whose flat start is odd in a 2-column array: element 0 is
    // column 1. Compare against a whole-array decode shifted by one.
    const whole = new Uint8Array([10, 20, 30, 40]);
    const wholeDecoded = await decodePerChannel(ctx, {
      data: whole,
      kind: 'linear',
      colLo,
      colHi,
      zeroLevel: false,
      colOffset: 0,
      bits: 8,
    });
    const rangeDecoded = await decodePerChannel(ctx, {
      data: whole.slice(1),
      kind: 'linear',
      colLo,
      colHi,
      zeroLevel: false,
      colOffset: 1,
      bits: 8,
    });
    expect(Array.from(rangeDecoded)).toEqual(Array.from(wholeDecoded.slice(1)));
  });

  it('decodes geolog (TRUE-log HDR colors) identically to the main-thread dequant', async () => {
    const logLo = new Float64Array([Math.log(1e-3), Math.log(0.5)]);
    const logHi = new Float64Array([Math.log(10), Math.log(2.0)]);
    const data = new Uint16Array([0, 1, 65535, 30000, 0, 7]);
    const decoded = await decodePerChannel(ctx, {
      data,
      kind: 'geolog',
      colLo: logLo,
      colHi: logHi,
      zeroLevel: true,
      colOffset: 0,
      bits: 16,
    });
    const dequant = ArrayDecoder.makePerChannelDequant(
      {
        name: 'geolog_perchannel_u16',
        bits: 16,
        col_lo: Array.from(logLo),
        col_hi: Array.from(logHi),
        zero_level: true,
      },
      2
    );
    for (let i = 0; i < data.length; i++) {
      expect(decoded[i]).toBe(Math.fround(dequant(data[i], i % 2)));
    }
    expect(decoded[0]).toBe(0);
    expect(decoded[4]).toBe(0);
  });

  it('rejects a bits/container mismatch', async () => {
    await expect(
      decodePerChannel(ctx, {
        data: new Uint8Array([1, 2]),
        kind: 'log',
        colLo,
        colHi,
        zeroLevel: true,
        colOffset: 0,
        bits: 16, // container is Uint8Array
      })
    ).rejects.toThrow(/does not match data container/);
  });

  it('rejects mismatched or empty per-column scales', async () => {
    await expect(
      decodePerChannel(ctx, {
        data: new Uint8Array([1]),
        kind: 'log',
        colLo,
        colHi: new Float64Array([1.0]), // length 1 vs colLo length 2
        zeroLevel: true,
        colOffset: 0,
        bits: 8,
      })
    ).rejects.toThrow(/length mismatch/);
    await expect(
      decodePerChannel(ctx, {
        data: new Uint8Array([1]),
        kind: 'log',
        colLo: new Float64Array(0),
        colHi: new Float64Array(0),
        zeroLevel: true,
        colOffset: 0,
        bits: 8,
      })
    ).rejects.toThrow(/cols/);
  });

  it('rejects a negative or non-integer colOffset', async () => {
    await expect(
      decodePerChannel(ctx, {
        data: new Uint8Array([1]),
        kind: 'linear',
        colLo,
        colHi,
        zeroLevel: false,
        colOffset: -1,
        bits: 8,
      })
    ).rejects.toThrow(/non-negative integer/);
  });

  it('rejects an unknown kind', async () => {
    await expect(
      decodePerChannel(ctx, {
        data: new Uint8Array([1]),
        kind: 'bogus' as never,
        colLo,
        colHi,
        zeroLevel: false,
        colOffset: 0,
        bits: 8,
      })
    ).rejects.toThrow(/unknown kind/);
  });
});
