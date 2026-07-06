/**
 * Unit tests for the shared color-attribute helpers.
 *
 * The helpers are pure (no DOM, no THREE). Direct (unencoded) color reads now
 * route through `RangeLoader.loadDirectTyped` (whose dtype-preserving copy core
 * is `copyDirectChunk`, tested directly below); `loadColorRanges` orchestration
 * is exercised against a fake RangeLoader.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  allocateColorBuffer,
  getExpectedColorType,
  colorBufferTypeMatches,
  restoreOriginalDtype,
  loadColorRanges,
  type ColorRange,
  type RangeLoader,
} from '../../../../data/loaders';
import {
  copyDirectChunk,
  type RangeNumericArray,
} from '../../../../data/loaders/spatial-query/range-loader/encoding-types';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    get: vi.fn(),
    slice: (start: number | null, end?: number | null) => ({ start, end }),
  };
});
import { get as zarrGet } from 'zarrita';
const mockZarrGet = vi.mocked(zarrGet);

describe('allocateColorBuffer', () => {
  it('returns a Float32Array for encoded arrays regardless of dtype', () => {
    expect(allocateColorBuffer(9, true, 'uint8')).toBeInstanceOf(Float32Array);
    expect(allocateColorBuffer(9, true, 'float32')).toBeInstanceOf(Float32Array);
    expect((allocateColorBuffer(9, true, 'uint8') as Float32Array).length).toBe(9);
  });

  it('returns a Uint8Array for any uint8 dtype variant', () => {
    for (const dtype of ['uint8', '|u1', '<u1', '>u1']) {
      const buf = allocateColorBuffer(6, false, dtype);
      expect(buf).toBeInstanceOf(Uint8Array);
      expect(buf.length).toBe(6);
    }
  });

  it('returns a Uint16Array for any uint16 dtype variant', () => {
    for (const dtype of ['uint16', '|u2', '<u2', '>u2']) {
      const buf = allocateColorBuffer(6, false, dtype);
      expect(buf).toBeInstanceOf(Uint16Array);
      expect(buf.length).toBe(6);
    }
  });

  it('falls back to Float32Array for unknown dtypes', () => {
    expect(allocateColorBuffer(3, false, 'float64')).toBeInstanceOf(Float32Array);
    expect(allocateColorBuffer(3, false, 'mystery')).toBeInstanceOf(Float32Array);
  });

  // [P5] unsupported numeric dtypes (no uint8/uint16 match) → Float32 fallback.
  it('falls back to Float32Array for unsupported dtypes (float64 / complex128)', () => {
    for (const dtype of ['float64', 'complex128']) {
      const buf = allocateColorBuffer(4, false, dtype);
      expect(buf).toBeInstanceOf(Float32Array);
      expect(buf.length).toBe(4);
    }
  });
});

describe('getExpectedColorType', () => {
  it('maps uint8 dtype variants to Uint8Array', () => {
    for (const dtype of ['uint8', '|u1', '<u1', '>u1']) {
      expect(getExpectedColorType(dtype)).toBe('Uint8Array');
    }
  });

  it('maps uint16 dtype variants to Uint16Array', () => {
    for (const dtype of ['uint16', '|u2', '<u2', '>u2']) {
      expect(getExpectedColorType(dtype)).toBe('Uint16Array');
    }
  });

  it('maps everything else to Float32Array', () => {
    expect(getExpectedColorType('float32')).toBe('Float32Array');
    expect(getExpectedColorType('mystery')).toBe('Float32Array');
  });

  // [P5] unsupported numeric dtypes also fall back to Float32Array.
  it('maps unsupported dtypes (float64 / complex128) to Float32Array', () => {
    expect(getExpectedColorType('float64')).toBe('Float32Array');
    expect(getExpectedColorType('complex128')).toBe('Float32Array');
  });
});

describe('colorBufferTypeMatches', () => {
  it('matches Uint8Array correctly', () => {
    expect(colorBufferTypeMatches(new Uint8Array(2), 'Uint8Array')).toBe(true);
    expect(colorBufferTypeMatches(new Float32Array(2), 'Uint8Array')).toBe(false);
    expect(colorBufferTypeMatches(new Uint16Array(2), 'Uint8Array')).toBe(false);
  });

  it('matches Uint16Array correctly', () => {
    expect(colorBufferTypeMatches(new Uint16Array(2), 'Uint16Array')).toBe(true);
    expect(colorBufferTypeMatches(new Uint8Array(2), 'Uint16Array')).toBe(false);
    expect(colorBufferTypeMatches(new Float32Array(2), 'Uint16Array')).toBe(false);
  });

  it('matches Float32Array correctly', () => {
    expect(colorBufferTypeMatches(new Float32Array(2), 'Float32Array')).toBe(true);
    expect(colorBufferTypeMatches(new Uint8Array(2), 'Float32Array')).toBe(false);
    expect(colorBufferTypeMatches(new Uint16Array(2), 'Float32Array')).toBe(false);
  });
});

describe('restoreOriginalDtype', () => {
  it('returns the input unchanged when no original dtype is recorded', () => {
    const decoded = new Float32Array([1.5, 2.5]);
    const out = restoreOriginalDtype(decoded, undefined, 2);
    expect(out).toBe(decoded);
  });

  it('returns the input unchanged for unknown dtypes', () => {
    const decoded = new Float32Array([1.5, 2.5]);
    const out = restoreOriginalDtype(decoded, 'float64', 2);
    expect(out).toBe(decoded);
  });

  it('rounds and clamps to uint8 [0, 255]', () => {
    const decoded = new Float32Array([-5, 0, 0.4, 0.6, 200.7, 999]);
    const out = restoreOriginalDtype(decoded, 'uint8', 6);
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([0, 0, 0, 1, 201, 255]);
  });

  it('rounds and clamps to uint16 [0, 65535]', () => {
    const decoded = new Float32Array([-50, 0, 1.4, 65535.6, 1000000]);
    const out = restoreOriginalDtype(decoded, 'uint16', 5);
    expect(out).toBeInstanceOf(Uint16Array);
    expect(Array.from(out as Uint16Array)).toEqual([0, 0, 1, 65535, 65535]);
  });

  it('accepts every recognized dtype string variant', () => {
    for (const dtype of ['uint8', '|u1', '<u1', '>u1']) {
      expect(restoreOriginalDtype(new Float32Array([10]), dtype, 1)).toBeInstanceOf(Uint8Array);
    }
    for (const dtype of ['uint16', '|u2', '<u2', '>u2']) {
      expect(restoreOriginalDtype(new Float32Array([10]), dtype, 1)).toBeInstanceOf(Uint16Array);
    }
  });

  // [P5] boundary clamping at the exact dtype edges.
  it('clamps uint8 at the 0 and 255 boundaries (incl. fractional overshoot)', () => {
    // 0 → 0, 255 → 255, 255.9 → 255 (clamp then round), -5 → 0 (clamp low).
    const decoded = new Float32Array([0, 255, 255.9, -5]);
    const out = restoreOriginalDtype(decoded, 'uint8', 4);
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([0, 255, 255, 0]);
  });

  it('clamps uint16 at the 0 and 65535 boundaries (incl. fractional overshoot)', () => {
    // 0 → 0, 65535 → 65535, 65535.9 → 65535 (clamp then round).
    const decoded = new Float32Array([0, 65535, 65535.9]);
    const out = restoreOriginalDtype(decoded, 'uint16', 3);
    expect(out).toBeInstanceOf(Uint16Array);
    expect(Array.from(out as Uint16Array)).toEqual([0, 65535, 65535]);
  });
});

// `copyDirectChunk` is the dtype-preserving copy core shared by every direct
// read (the unified RangeLoader.loadDirectTyped reader uses it). These pure
// cases pin the type-preservation contract that color correctness depends on.
describe('copyDirectChunk', () => {
  it('preserves uint8 bytes with zero conversion (same-kind copy)', () => {
    const out = new Uint8Array(9);
    const n = copyDirectChunk(out, new Uint8Array([10, 20, 30]), 0);
    expect(n).toBe(3);
    // Only the written prefix changes; the tail stays zero.
    expect(out).toEqual(new Uint8Array([10, 20, 30, 0, 0, 0, 0, 0, 0]));
  });

  it('preserves Float32 and accumulates at the given destOffset', () => {
    const out = new Float32Array(6);
    let off = 0;
    off += copyDirectChunk(out, new Float32Array([0.1, 0.2, 0.3]), off);
    off += copyDirectChunk(out, new Float32Array([0.4, 0.5, 0.6]), off);
    expect(off).toBe(6);
    expect(Array.from(out)).toEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((v) => Math.fround(v)));
  });

  it('widens a Uint8 source into a Float32 output (cross-kind conversion)', () => {
    const out = new Float32Array(3);
    copyDirectChunk(out, new Uint8Array([10, 20, 30]), 0);
    expect(Array.from(out)).toEqual([10, 20, 30]);
  });

  it('widens a BigInt64 source via Number() (int64 zarr arrays)', () => {
    const out = new Float32Array(2);
    const n = copyDirectChunk(out, new BigInt64Array([5n, 7n]) as RangeNumericArray, 0);
    expect(n).toBe(2);
    expect(Array.from(out)).toEqual([5, 7]);
  });

  it('preserves Uint32 for lines segments (same-kind copy)', () => {
    const out = new Uint32Array(4);
    copyDirectChunk(out, new Uint32Array([1, 2]), 0);
    copyDirectChunk(out, new Uint32Array([3, 4]), 2);
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
  });
});

describe('loadColorRanges (orchestrator)', () => {
  beforeEach(() => {
    mockZarrGet.mockReset();
  });

  /**
   * A RangeLoader stand-in. `loadRangesResolvingRef` (encoded path) is a spy;
   * `loadDirectTyped` (direct path) mirrors the real reader — it consumes the
   * queued `mockZarrGet` results and copies them into the caller's output via
   * `copyDirectChunk`, so the existing `mockResolvedValueOnce({data})` setups
   * still drive the direct-path tests and dtype-preservation is exercised.
   */
  function makeFakeRangeLoader() {
    return {
      loadRangesResolvingRef: vi.fn(async (_array, _attrs, _ranges, _out, _total, _epi, _store) => {
        return 0;
      }),
      loadDirectTyped: vi.fn(
        async (_array: unknown, ranges: ColorRange[], output: RangeNumericArray) => {
          let off = 0;
          for (let i = 0; i < ranges.length; i++) {
            const chunk = (await mockZarrGet(_array as never, [] as never)) as {
              data: RangeNumericArray;
            };
            off += copyDirectChunk(output as never, chunk.data, off);
          }
          return off;
        }
      ),
    } as unknown as RangeLoader & {
      loadRangesResolvingRef: ReturnType<typeof vi.fn>;
      loadDirectTyped: ReturnType<typeof vi.fn>;
    };
  }

  it('routes per-channel encoded colors (geolog_perchannel_u16 HDR) through the DECODE path', async () => {
    // Regression: the encoded-ness check used to be a hand-rolled
    // quantized/LUT/broadcasted triple that missed the per-channel family,
    // so HDR geolog_perchannel_u16 colors streamed their RAW u16 CODES as
    // SDR full-scale colors (rendered near-white). Per-channel colors must
    // decode to Float32 — never take the direct-typed path.
    const array = {
      dtype: 'uint16',
      shape: [10, 3],
      attrs: {
        encoding: {
          name: 'geolog_perchannel_u16',
          bits: 16,
          col_lo: [0, 0, 0],
          col_hi: [1, 1, 1],
          zero_level: true,
          original_dtype: 'float32',
        },
      },
    } as never;
    const rl = makeFakeRangeLoader();
    const ranges: ColorRange[] = [{ start: 0, end: 2 }];

    const out = await loadColorRanges(array, ranges, rl, {} as never, 'TEST');

    expect(out).toBeInstanceOf(Float32Array); // decoded floats, not raw codes
    expect(rl.loadRangesResolvingRef).toHaveBeenCalledTimes(1);
    expect(rl.loadDirectTyped).not.toHaveBeenCalled();
  });

  it('allocates a fresh buffer instead of reusing an UNDERSIZED Float32 target', async () => {
    // Regression (deep-check finding): an accumulator whose buffer was
    // swapped to an exact-size Float32Array by a smaller earlier load must
    // not be reused for a larger load — TypedArray out-of-bounds writes are
    // silently dropped, truncating colors with no error.
    const array = {
      dtype: 'uint16',
      shape: [10, 3],
      attrs: {
        encoding: {
          name: 'geolog_perchannel_u16',
          bits: 16,
          col_lo: [0, 0, 0],
          col_hi: [1, 1, 1],
          zero_level: true,
          original_dtype: 'float32',
        },
      },
    } as never;
    const rl = makeFakeRangeLoader();
    const undersized = new Float32Array(3); // one item; the load needs four
    const ranges: ColorRange[] = [{ start: 0, end: 4 }];

    const out = await loadColorRanges(array, ranges, rl, {} as never, 'TEST', undersized);

    expect(out).toBeInstanceOf(Float32Array);
    expect(out).not.toBe(undersized); // MUST have allocated a fitting buffer
    expect((out as Float32Array).length).toBe(12);
  });

  it('takes the direct path for unencoded uint8 colors and never invokes RangeLoader', async () => {
    mockZarrGet.mockResolvedValueOnce({ data: new Uint8Array([10, 20, 30, 40, 50, 60]) } as never);
    const array = {
      dtype: 'uint8',
      shape: [10, 3],
      attrs: {}, // no encoding
    } as never;
    const rl = makeFakeRangeLoader();
    const ranges: ColorRange[] = [{ start: 0, end: 2 }];

    const out = await loadColorRanges(array, ranges, rl, {} as never, 'TEST');

    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([10, 20, 30, 40, 50, 60]);
    expect(rl.loadRangesResolvingRef).not.toHaveBeenCalled();
  });

  it('reuses the target buffer on the direct path when its kind matches', async () => {
    mockZarrGet.mockResolvedValueOnce({ data: new Uint8Array([1, 2, 3]) } as never);
    const array = { dtype: 'uint8', shape: [10, 3], attrs: {} } as never;
    const rl = makeFakeRangeLoader();
    const target = new Uint8Array(3);
    const ranges: ColorRange[] = [{ start: 0, end: 1 }];

    const out = await loadColorRanges(array, ranges, rl, {} as never, 'TEST', target);

    expect(out).toBe(target);
    expect(Array.from(target)).toEqual([1, 2, 3]);
    expect(rl.loadRangesResolvingRef).not.toHaveBeenCalled();
  });

  it('skips the decode pipeline for rgb_uint8 with a Uint8Array target', async () => {
    mockZarrGet.mockResolvedValueOnce({ data: new Uint8Array([5, 6, 7]) } as never);
    const array = {
      dtype: 'uint8',
      shape: [10, 3],
      attrs: { encoding: { name: 'rgb_uint8' } },
    } as never;
    const rl = makeFakeRangeLoader();
    const target = new Uint8Array(3);

    const out = await loadColorRanges(array, [{ start: 0, end: 1 }], rl, {} as never, 'T', target);

    expect(out).toBe(target);
    expect(Array.from(target)).toEqual([5, 6, 7]);
    expect(rl.loadRangesResolvingRef).not.toHaveBeenCalled();
  });

  it('routes encoded colors through RangeLoader.loadRangesResolvingRef', async () => {
    const array = {
      dtype: 'float32',
      shape: [10, 3],
      attrs: {
        encoding: { name: 'bounded_scalar_uint8', original_dtype: 'uint8', min: 0, max: 1 },
      },
    } as never;
    const rl = makeFakeRangeLoader();
    rl.loadRangesResolvingRef.mockImplementationOnce(
      async (_a, _attrs, _ranges, output: Float32Array) => {
        // Pretend the loader filled the buffer with values that, when clamped
        // back to uint8, give a clear signal.
        output.set([0.5, 100.5, 250.5, 1000.0, -50.0, 128.7]);
        return 6;
      }
    );
    const ranges: ColorRange[] = [{ start: 0, end: 2 }];

    const out = await loadColorRanges(array, ranges, rl, {} as never, 'TEST');

    // original_dtype=uint8 → restoreOriginalDtype rounds and clamps [0,255].
    expect(out).toBeInstanceOf(Uint8Array);
    expect(Array.from(out as Uint8Array)).toEqual([1, 101, 251, 255, 0, 129]);
    expect(rl.loadRangesResolvingRef).toHaveBeenCalledTimes(1);
    expect(rl.loadRangesResolvingRef.mock.calls[0][7]).toBe('TEST'); // logPrefix
  });

  it('passes the supplied logPrefix through to the RangeLoader call', async () => {
    const array = {
      dtype: 'float32',
      shape: [10, 3],
      attrs: { encoding: { name: 'broadcasted' } },
    } as never;
    const rl = makeFakeRangeLoader();

    await loadColorRanges(array, [{ start: 0, end: 1 }], rl, {} as never, 'GSplats');

    expect(rl.loadRangesResolvingRef).toHaveBeenCalled();
    expect(rl.loadRangesResolvingRef.mock.calls[0][7]).toBe('GSplats');
  });

  // [P5] empty ranges → totalItems 0 → zero-length buffer, no zarr reads.
  it('returns a zero-length buffer for empty ranges (totalItems 0)', async () => {
    const array = { dtype: 'uint8', shape: [10, 3], attrs: {} } as never;
    const rl = makeFakeRangeLoader();

    const out = await loadColorRanges(array, [], rl, {} as never, 'TEST');

    expect(out.length).toBe(0);
    // No ranges to stream, so neither the RangeLoader nor zarr.get is touched.
    expect(rl.loadRangesResolvingRef).not.toHaveBeenCalled();
    expect(mockZarrGet).not.toHaveBeenCalled();
  });
});
