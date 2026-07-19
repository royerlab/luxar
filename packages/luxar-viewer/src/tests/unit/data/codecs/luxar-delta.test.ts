/**
 * Unit tests for the `luxar_delta_v1` zarrita codec (columnar delta+zigzag).
 *
 * The hand-computed wire-format vectors are IDENTICAL to the ones locked in
 * the Python tests (`encoding/tests/test_delta_codec.py::TestWireFormat`) —
 * if either side changes bytes, both suites fail. The cross-language fixture
 * test (Python-written zarr read through the registered codec) lives in
 * `zarr-delta-fixture.test.ts`.
 */

import { describe, expect, it } from 'vitest';

import { LuxarDeltaCodec } from '../../../../data/codecs/luxar-delta';
import { codecRegistry } from '../../../../data/zarr';

function chunk(data: Uint8Array | Uint16Array, shape: number[]) {
  // C-order stride for a (rows, cols) chunk — what zarrita hands the codec.
  const stride = shape.length === 2 ? [shape[1], 1] : [1];
  return { data, shape, stride };
}

describe('LuxarDeltaCodec wire format (locked against Python)', () => {
  it('matches the hand-computed u16 vector', () => {
    const codes = new Uint16Array([100, 5, 98, 5, 103, 65535]); // 3 rows x 2 cols
    const codec = new LuxarDeltaCodec(2, 16);
    const enc = codec.encode(chunk(codes, [3, 2]));
    // col 0 deltas: 100, -2, +5 -> zigzag 200, 3, 10
    // col 1 deltas: 5, 0, -6 (mod 2^16) -> zigzag 10, 0, 11
    expect(Array.from(enc.data)).toEqual([200, 3, 10, 10, 0, 11]);
    expect(Array.from(codec.decode(enc).data)).toEqual(Array.from(codes));
  });

  it('matches the hand-computed u8 wrap-around vector', () => {
    const codes = new Uint8Array([250, 3, 1]); // 3 rows x 1 col
    const codec = new LuxarDeltaCodec(1, 8);
    const enc = codec.encode(chunk(codes, [3, 1]));
    // deltas: -6 (250 >= 128), +9 (wrap 250->3), -2 -> zigzag 11, 18, 3
    expect(Array.from(enc.data)).toEqual([11, 18, 3]);
    expect(Array.from(codec.decode(enc).data)).toEqual(Array.from(codes));
  });

  it('emits columnar layout (all col-0 residuals, then col-1)', () => {
    // Constant col 0 (7), ramp col 1 (0..4), interleaved row-major input.
    const codes = new Uint16Array([7, 0, 7, 1, 7, 2, 7, 3, 7, 4]);
    const enc = new LuxarDeltaCodec(2, 16).encode(chunk(codes, [5, 2]));
    expect(Array.from(enc.data.slice(0, 5))).toEqual([14, 0, 0, 0, 0]);
    expect(Array.from(enc.data.slice(5))).toEqual([0, 2, 2, 2, 2]);
  });
});

describe('LuxarDeltaCodec round-trips', () => {
  it('round-trips random uint16 codes exactly (cols=3)', () => {
    const n = 1013;
    const codes = new Uint16Array(n * 3);
    let seed = 42;
    for (let i = 0; i < codes.length; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      codes[i] = seed & 0xffff;
    }
    const codec = new LuxarDeltaCodec(3, 16);
    const round = codec.decode(codec.encode(chunk(codes, [n, 3])));
    expect(round.data).toEqual(codes);
  });

  it('round-trips random uint8 codes exactly (cols=1)', () => {
    const codes = new Uint8Array(997);
    let seed = 7;
    for (let i = 0; i < codes.length; i++) {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      codes[i] = seed & 0xff;
    }
    const codec = new LuxarDeltaCodec(1, 8);
    const round = codec.decode(codec.encode(chunk(codes, [997])));
    expect(round.data).toEqual(codes);
  });

  it('handles a single-row chunk and preserves shape/stride', () => {
    const codes = new Uint16Array([9, 0, 65535]);
    const codec = new LuxarDeltaCodec(3, 16);
    const enc = codec.encode(chunk(codes, [1, 3]));
    const dec = codec.decode(enc);
    expect(Array.from(dec.data)).toEqual([9, 0, 65535]);
    expect(dec.shape).toEqual([1, 3]);
    expect(dec.stride).toEqual([3, 1]);
  });
});

describe('LuxarDeltaCodec validation and registration', () => {
  it('fromConfig resolves cols/bits and rejects dtype mismatches', () => {
    const codec = LuxarDeltaCodec.fromConfig({ cols: 3, bits: 16 }, { dataType: 'uint16' });
    expect(codec.kind).toBe('array_to_array');
    expect(() =>
      LuxarDeltaCodec.fromConfig({ cols: 3, bits: 16 }, { dataType: 'uint8' })
    ).toThrow(/does not match/);
    expect(() =>
      LuxarDeltaCodec.fromConfig({ cols: 3, bits: 16 }, { dataType: 'float32' })
    ).toThrow(/unsupported data type/);
  });

  it('fromConfig cross-checks cols against the chunk shape (fail-loud on corruption)', () => {
    // Matching shapes pass.
    LuxarDeltaCodec.fromConfig({ cols: 3, bits: 16 }, { dataType: 'uint16', shape: [4096, 3] });
    LuxarDeltaCodec.fromConfig({ cols: 1, bits: 16 }, { dataType: 'uint16', shape: [16384] });
    // A corrupted cols that still divides the chunk size must throw, not
    // silently decode garbage.
    expect(() =>
      LuxarDeltaCodec.fromConfig({ cols: 2, bits: 16 }, { dataType: 'uint16', shape: [4096, 3] })
    ).toThrow(/does not match chunk shape/);
    // Missing cols (config loss) defaults to 1 and must also be caught.
    expect(() =>
      LuxarDeltaCodec.fromConfig({ bits: 16 }, { dataType: 'uint16', shape: [4096, 3] })
    ).toThrow(/does not match chunk shape/);
  });

  it('rejects chunks whose size is not a multiple of cols', () => {
    const codec = new LuxarDeltaCodec(3, 16);
    expect(() => codec.decode(chunk(new Uint16Array(10), [10]))).toThrow(/multiple of cols/);
  });

  it('is registered as numcodecs.luxar_delta_v1 via the zarr facade', async () => {
    const thunk = codecRegistry.get('numcodecs.luxar_delta_v1');
    expect(thunk).toBeDefined();
    await expect(thunk!()).resolves.toBe(LuxarDeltaCodec);
  });
});
