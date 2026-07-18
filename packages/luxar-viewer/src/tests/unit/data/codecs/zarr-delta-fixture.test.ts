/**
 * Cross-language contract test for the `luxar_delta_v1` zarr filter.
 *
 * `tests/fixtures/test_delta_filter.luxar.zarr` is written by the PYTHON
 * encoder (`generate_test_data.py::generate_delta_filter_test`) with the
 * delta filter probe-enabled on the positions array and a zlib compressor
 * (Node-decodable, unlike blosc — see the fixture generator's docstring).
 * Reading it here exercises the real zarrita pipeline: zlib decompress →
 * `numcodecs.luxar_delta_v1` (registered by the zarr facade) → codes.
 *
 * The generic decode-parity check (sha256 + samples vs Python's own
 * ArrayDecoder) is covered by `array-roundtrip.test.ts` via
 * `roundtrip_expectations.json`, which includes this fixture automatically.
 * This file asserts the filter-specific behaviors: metadata resolution,
 * whole-array reads, and SUB-CHUNK random access (the reason the delta
 * lives in the codec pipeline at all).
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

import { FileSystemStore } from '@zarrita/storage';
import { beforeAll, describe, expect, it } from 'vitest';

import * as zarr from '../../../../data/zarr';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FIXTURE = path.resolve(
  __dirname,
  '../../../../../tests/fixtures/test_delta_filter.luxar.zarr'
);

describe('luxar_delta_v1 fixture (Python-written, TS-read)', () => {
  let positions: Awaited<ReturnType<typeof zarr.openArray>>;

  beforeAll(async () => {
    const store = new FileSystemStore(FIXTURE);
    const rootLoc = zarr.root(store);
    positions = await zarr.openArray(rootLoc.resolve('/points/positions'), { attrs: true });
  });

  it('fixture metadata carries the delta filter', () => {
    const zarray = JSON.parse(
      fs.readFileSync(path.join(FIXTURE, 'points/positions/.zarray'), 'utf-8')
    );
    expect(zarray.filters).toEqual([{ id: 'luxar_delta_v1', cols: 3, bits: 16 }]);
    expect(zarray.dtype).toContain('u2'); // uint16 codes
  });

  it('reads the whole array through zlib + delta and decodes sane codes', async () => {
    const { data, shape } = await zarr.readArray(positions);
    expect(shape).toEqual([20000, 3]);
    const codes = data as Uint16Array;
    // The Python writer quantizes each axis over its own [min, max] to
    // 65536 levels — a correct decode must hit both rails per axis. A
    // broken delta (wrong layout / anchor / modulus) produces codes that
    // miss the rails or scatter uniformly.
    for (let c = 0; c < 3; c++) {
      let min = 0xffff;
      let max = 0;
      for (let i = c; i < codes.length; i += 3) {
        const v = codes[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      expect(min).toBe(0);
      expect(max).toBe(65535);
    }
  });

  it('supports sub-chunk random access (mid-array slice)', async () => {
    // Read a narrow window that starts mid-chunk — zarrita reconstructs the
    // whole chunk through the codec, then slices. Cross-check against the
    // same rows from a whole-array read.
    const whole = await zarr.readArray(positions);
    const start = 7013; // deliberately not chunk-aligned
    const end = 7050;
    const window = await zarr.readArray(positions, [zarr.slice(start, end), zarr.slice(null)]);
    expect(window.shape).toEqual([end - start, 3]);
    const expected = (whole.data as Uint16Array).slice(start * 3, end * 3);
    expect(window.data as Uint16Array).toEqual(expected);
  });

  it('decoded positions match the writer extents after dequantization', async () => {
    const { data } = await zarr.readArray(positions);
    const codes = data as Uint16Array;
    const enc = (positions.attrs as Record<string, unknown>).encoding as {
      name: string;
      col_lo: number[];
      col_hi: number[];
    };
    expect(enc.name).toBe('linear_perchannel_u16');
    // Dequantize per axis; the fixture normalizes positions to extents
    // [300, 500, 800] so lo≈0 and hi≈extent must round-trip.
    const spans = [300, 500, 800];
    for (let c = 0; c < 3; c++) {
      const lo = enc.col_lo[c];
      const hi = enc.col_hi[c];
      expect(lo).toBeCloseTo(0, 3);
      expect(hi).toBeCloseTo(spans[c], 3);
      let min = Infinity;
      let max = -Infinity;
      for (let i = c; i < codes.length; i += 3) {
        const v = lo + (codes[i] / 65535) * (hi - lo);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      expect(min).toBeCloseTo(0, 2);
      expect(max).toBeCloseTo(spans[c], 2);
    }
  });
});
