/**
 * Unit tests for the shared color-attribute helpers.
 *
 * The helpers are pure (no DOM, no THREE) except for `loadDirectColorRanges`,
 * which calls `zarr.get` against a supplied array — that is mocked here.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  allocateColorBuffer,
  getExpectedColorType,
  colorBufferTypeMatches,
  loadDirectColorRanges,
  restoreOriginalDtype,
  type ColorRange,
} from '../../../../data/loaders/color-attribute-utils';

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
});

describe('loadDirectColorRanges', () => {
  beforeEach(() => {
    mockZarrGet.mockReset();
  });

  it('preserves uint8 type with zero conversion', async () => {
    mockZarrGet.mockResolvedValueOnce({ data: new Uint8Array([10, 20, 30]) } as never);
    const out = new Uint8Array(9);
    const array = { dtype: 'uint8', shape: [10], attrs: {} } as never;
    const ranges: ColorRange[] = [{ start: 0, end: 1 }];

    await loadDirectColorRanges(array, ranges, out);
    expect(Array.from(out.slice(0, 3))).toEqual([10, 20, 30]);
  });

  it('preserves Float32 type and concatenates multiple ranges', async () => {
    mockZarrGet
      .mockResolvedValueOnce({ data: new Float32Array([0.1, 0.2, 0.3]) } as never)
      .mockResolvedValueOnce({ data: new Float32Array([0.4, 0.5, 0.6]) } as never);
    const out = new Float32Array(6);
    const array = { dtype: 'float32', shape: [10, 3], attrs: {} } as never;
    const ranges: ColorRange[] = [
      { start: 0, end: 1 },
      { start: 5, end: 6 },
    ];

    await loadDirectColorRanges(array, ranges, out);
    expect(Array.from(out)).toEqual([
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    ].map((v) => Math.fround(v)));
  });

  it('falls back to Float32 widening when source and output kinds disagree', async () => {
    // Stored as Uint8 but caller allocated Float32 (e.g. legacy mismatch).
    mockZarrGet.mockResolvedValueOnce({ data: new Uint8Array([10, 20, 30]) } as never);
    const out = new Float32Array(3);
    const array = { dtype: 'uint8', shape: [10], attrs: {} } as never;
    const ranges: ColorRange[] = [{ start: 0, end: 1 }];

    await loadDirectColorRanges(array, ranges, out);
    expect(Array.from(out)).toEqual([10, 20, 30]);
  });
});
