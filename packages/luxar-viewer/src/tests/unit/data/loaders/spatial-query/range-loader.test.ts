/**
 * Tests for RangeLoader - encoding detection and range-based decoding
 *
 * Tests cover:
 * - detectEncoding: All encoding types
 * - loadDirect: Unencoded float32/uint8 data
 * - loadBroadcasted: Single value expansion
 * - loadQuantized: Dequantization math (linear and log-space)
 * - loadLUT: Lookup table index-based decoding (row and scalar modes)
 *
 * Mocking strategy: Only zarr I/O (external) is mocked. RangeLoader's
 * internal logic (encoding detection, dequantization, broadcast, LUT)
 * is tested with real code.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import type { ArrayMetadata } from '../../../../../data/array-decoder/decoder';
import { ArrayRefRegistry } from '../../../../../data/array-decoder/decoder';
import {
  RangeLoader,
  getSharedRangeLoader,
  getSharedRefRegistry,
  resetSharedRangeLoader,
  type LoadRange,
} from '../../../../../data/loaders';

// ---------------------------------------------------------------------------
// Mock zarr I/O and worker infrastructure (external dependencies only)
// ---------------------------------------------------------------------------

// Mock the config to disable workers (test main-thread paths)
vi.mock('../../../../../config', () => ({
  config: {
    dataLoading: {
      performance: {
        useWebWorkers: false,
      },
    },
  },
}));

// Mock worker-pool so it's never called
vi.mock('../../../../../workers/worker-pool', () => ({
  getWorkerPool: () => {
    throw new Error('Workers should not be used in tests');
  },
}));

// Mock zarrita get/slice to return controlled data. open() and root() are
// mocked too so that loadRangesResolvingRef can be exercised without an
// actual zarr store.
vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    // get() is replaced per-test via mockZarrGet
    get: vi.fn(),
    slice: (start: number | null, end?: number | null) => ({ start, end }),
    open: vi.fn(),
    root: vi.fn(),
  };
});

import { get as zarrGet, open as zarrOpen, root as zarrRoot } from 'zarrita';
const mockZarrGet = vi.mocked(zarrGet);
const mockZarrOpen = vi.mocked(zarrOpen);
const mockZarrRoot = vi.mocked(zarrRoot);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a minimal mock zarr array with the given dtype and shape */
function mockZarrArray(dtype: string, shape: number[]) {
  return { dtype, shape, attrs: {} } as any;
}

/** Configure mockZarrGet to return the given typed array data for every call */
function setMockData(data: Float32Array | Uint8Array | Uint16Array) {
  mockZarrGet.mockResolvedValue({ data } as any);
}

/** Create a RangeLoader with workers disabled */
function createLoader(): RangeLoader {
  const loader = new RangeLoader(new ArrayRefRegistry(), { workerThreshold: Infinity });
  loader.setVerbose(false);
  return loader;
}

// ---------------------------------------------------------------------------
// detectEncoding
// ---------------------------------------------------------------------------

describe('RangeLoader.detectEncoding', () => {
  it('returns direct when attrs is undefined', () => {
    expect(RangeLoader.detectEncoding(undefined)).toBe('direct');
  });

  it('returns direct when encoding is absent', () => {
    expect(RangeLoader.detectEncoding({})).toBe('direct');
  });

  it('returns direct for float32 data with no encoding metadata', () => {
    const attrs: ArrayMetadata = { dtype: 'float32' };
    expect(RangeLoader.detectEncoding(attrs)).toBe('direct');
  });

  // --- Broadcasted ---

  it('detects broadcasted encoding', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'broadcasted', n_elements: 1000 },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('broadcasted');
  });

  // --- Array ref ---

  it('rejects encoding metadata without a name', () => {
    const attrs: ArrayMetadata = {
      encoding: { target: '/SharedNode/colors' },
    };
    expect(() => RangeLoader.detectEncoding(attrs)).toThrow('encoding.name is required');
  });

  it('detects array_ref when name is explicit', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'array_ref', target: '/SharedNode/colors' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('array_ref');
  });

  // --- LUT ---

  it('detects lut encoding (lut_uint8)', () => {
    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
        ],
        lut_mode: 'row',
        original_shape: [100, 3],
      },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('lut');
  });

  it('detects lut encoding (lut_uint16)', () => {
    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint16',
        original_dtype: 'float32',
        lut: [0.5, 1.5, 2.5],
        lut_mode: 'scalar',
        original_shape: [100, 1],
      },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('lut');
  });

  it('rejects malformed lut metadata', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'lut_uint8' },
    };
    expect(() => RangeLoader.detectEncoding(attrs)).toThrow('LUT encoding requires encoding.lut');
  });

  // --- Quantized ---

  it('detects quantized for rgb_uint8 with bounds', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'rgb_uint8', bounds: [0, 1], original_dtype: 'float32' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('quantized');
  });

  it('detects quantized for rgb_uint16 with min/max', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'rgb_uint16', min: 0, max: 1, original_dtype: 'float32' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('quantized');
  });

  it('detects quantized for bounded_scalar_uint8', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'bounded_scalar_uint8', bounds: [0.1, 5.0], original_dtype: 'float32' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('quantized');
  });

  it('detects quantized for bounded_scalar_uint16', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'bounded_scalar_uint16', min: 0, max: 10, original_dtype: 'float32' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('quantized');
  });

  it('detects quantized for log_scalar_uint8', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'log_scalar_uint8', max_log: 3.5, original_dtype: 'float32' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('quantized');
  });

  it('detects quantized for log_scalar_uint16', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'log_scalar_uint16', max_log: 5.0, original_dtype: 'float32' },
    };
    expect(RangeLoader.detectEncoding(attrs)).toBe('quantized');
  });

  it('rejects malformed quantized metadata', () => {
    expect(() =>
      RangeLoader.detectEncoding({ encoding: { name: 'bounded_scalar_uint8' } })
    ).toThrow('bounded_scalar encoding requires bounds or min/max');
    expect(() => RangeLoader.detectEncoding({ encoding: { name: 'log_scalar_uint8' } })).toThrow(
      'log_scalar encoding requires encoding.max_log'
    );
  });

  it('rejects nameless bounds metadata', () => {
    const boundsAttrs: ArrayMetadata = {
      encoding: { bounds: [0, 1] },
    };
    const minMaxAttrs: ArrayMetadata = {
      encoding: { min: 0, max: 100 },
    };

    expect(() => RangeLoader.detectEncoding(boundsAttrs)).toThrow('encoding.name is required');
    expect(() => RangeLoader.detectEncoding(minMaxAttrs)).toThrow('encoding.name is required');
  });

  it('treats dtype encodings as direct storage, not quantization', () => {
    for (const name of ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float16']) {
      const attrs: ArrayMetadata = { encoding: { name } };
      expect(RangeLoader.detectEncoding(attrs)).toBe('direct');
    }
  });

  it('rejects quantization bounds on dtype encodings', () => {
    const attrs: ArrayMetadata = { encoding: { name: 'uint8', bounds: [0, 1] } };
    expect(() => RangeLoader.detectEncoding(attrs)).toThrow(
      'bounds/min/max metadata is only valid for quantized encodings'
    );
  });

  it('rejects unknown encoding names', () => {
    const attrs: ArrayMetadata = { encoding: { name: 'mystery_encoder' } };
    expect(() => RangeLoader.detectEncoding(attrs)).toThrow('Unknown encoding name');
  });

  it('rejects malformed prefix-matching encoding names', () => {
    for (const name of [
      'lut_float32',
      'bounded_scalar_uint32',
      'log_scalar_float32',
      'rgb_uint32',
    ]) {
      const attrs: ArrayMetadata = { encoding: { name } };
      expect(() => RangeLoader.detectEncoding(attrs)).toThrow('Unknown encoding name');
    }
  });

  // --- Priority ---

  it('rejects target metadata on broadcasted encodings', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'broadcasted', target: '/foo' },
    };
    expect(() => RangeLoader.detectEncoding(attrs)).toThrow(
      'encoding.target is only valid for array_ref'
    );
  });

  it('rejects target metadata on non-array_ref encodings', () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'lut_uint8', target: '/foo', lut: [1, 2, 3] },
    };
    expect(() => RangeLoader.detectEncoding(attrs)).toThrow(
      'encoding.target is only valid for array_ref'
    );
  });
});

// ---------------------------------------------------------------------------
// loadDirect
// ---------------------------------------------------------------------------

describe('RangeLoader.loadDirect (via loadRanges)', () => {
  let loader: RangeLoader;

  beforeEach(() => {
    loader = createLoader();
    mockZarrGet.mockReset();
  });

  it('loads a single range of float32 data', async () => {
    const srcData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    setMockData(srcData);

    const output = new Float32Array(6);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('float32', [10, 3]);

    const written = await loader.loadRanges(array, undefined, ranges, output, 2, 3);

    expect(written).toBe(6);
    expect(Array.from(output)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('loads multiple disjoint ranges into a contiguous output', async () => {
    // First range returns [10, 20], second returns [30, 40]
    mockZarrGet
      .mockResolvedValueOnce({ data: new Float32Array([10, 20]) } as any)
      .mockResolvedValueOnce({ data: new Float32Array([30, 40]) } as any);

    const output = new Float32Array(4);
    const ranges: LoadRange[] = [
      { start: 0, end: 2 },
      { start: 5, end: 7 },
    ];
    const array = mockZarrArray('float32', [10]);

    const written = await loader.loadRanges(array, undefined, ranges, output, 4, 1);

    expect(written).toBe(4);
    expect(Array.from(output)).toEqual([10, 20, 30, 40]);
  });

  it('converts uint8 source data to float32', async () => {
    setMockData(new Uint8Array([0, 128, 255]));

    const output = new Float32Array(3);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint8', [100]);

    const written = await loader.loadRanges(array, undefined, ranges, output, 3, 1);

    expect(written).toBe(3);
    expect(output[0]).toBe(0);
    expect(output[1]).toBe(128);
    expect(output[2]).toBe(255);
  });

  it('returns 0 written for empty ranges', async () => {
    const output = new Float32Array(10);
    const ranges: LoadRange[] = [];
    const array = mockZarrArray('float32', [100]);

    const written = await loader.loadRanges(array, undefined, ranges, output, 0, 1);

    expect(written).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// loadBroadcasted
// ---------------------------------------------------------------------------

describe('RangeLoader.loadBroadcasted (via loadRanges)', () => {
  let loader: RangeLoader;

  beforeEach(() => {
    loader = createLoader();
    mockZarrGet.mockReset();
  });

  it('replicates a single scalar to all elements', async () => {
    // Broadcasted value: [0.5]
    setMockData(new Float32Array([0.5]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'broadcasted', n_elements: 4 },
    };
    const output = new Float32Array(4);
    const ranges: LoadRange[] = [{ start: 0, end: 4 }];
    const array = mockZarrArray('float32', [1]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 4, 1);

    expect(written).toBe(4);
    expect(Array.from(output)).toEqual([0.5, 0.5, 0.5, 0.5]);
  });

  it('replicates a 3-component vector (e.g. RGB color) to all elements', async () => {
    // Broadcasted color: [1.0, 0.0, 0.5]
    setMockData(new Float32Array([1.0, 0.0, 0.5]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'broadcasted', n_elements: 3 },
    };
    const output = new Float32Array(9);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('float32', [1, 3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 3);

    expect(written).toBe(9);
    // Each of the 3 elements should have [1.0, 0.0, 0.5]
    expect(Array.from(output)).toEqual([1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5]);
  });

  it('handles uint8 broadcast source by converting to float32', async () => {
    // Uint8 color [255, 0, 128] should be converted to Float32
    setMockData(new Uint8Array([255, 0, 128]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'broadcasted', n_elements: 2 },
    };
    const output = new Float32Array(6);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('uint8', [1, 3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 3);

    expect(written).toBe(6);
    expect(output[0]).toBe(255);
    expect(output[1]).toBe(0);
    expect(output[2]).toBe(128);
    expect(output[3]).toBe(255);
    expect(output[4]).toBe(0);
    expect(output[5]).toBe(128);
  });

  it('replicates single value to multi-component when source has fewer values', async () => {
    // If source is [7.0] but elementsPerItem is 3, each component fills with valueAsFloat32[0]
    setMockData(new Float32Array([7.0]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'broadcasted', n_elements: 2 },
    };
    const output = new Float32Array(6);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('float32', [1]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 3);

    expect(written).toBe(6);
    // valueAsFloat32[j] ?? valueAsFloat32[0] => all 7.0
    expect(Array.from(output)).toEqual([7, 7, 7, 7, 7, 7]);
  });
});

// ---------------------------------------------------------------------------
// loadQuantized
// ---------------------------------------------------------------------------

describe('RangeLoader.loadQuantized (via loadRanges)', () => {
  let loader: RangeLoader;

  beforeEach(() => {
    loader = createLoader();
    mockZarrGet.mockReset();
  });

  it('dequantizes uint8 data with bounds [0, 1]', async () => {
    // uint8 values: 0, 127, 255 => normalized: 0, 127/255, 1
    // with bounds [0, 1]: same as normalized
    setMockData(new Uint8Array([0, 127, 255]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'rgb_uint8',
        bounds: [0, 1] as [number, number],
        original_dtype: 'float32',
      },
    };
    const output = new Float32Array(3);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint8', [3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 1);

    expect(written).toBe(3);
    expect(output[0]).toBeCloseTo(0.0, 5);
    expect(output[1]).toBeCloseTo(127 / 255, 5);
    expect(output[2]).toBeCloseTo(1.0, 5);
  });

  it('dequantizes a non-symmetric interior uint8 value (64) with bounds [0, 1]', async () => {
    // Non-symmetric interior point: kills an inverted dequant formula
    // (1 - normalized), which would yield 191/255 instead of 64/255.
    setMockData(new Uint8Array([64]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'rgb_uint8',
        bounds: [0, 1] as [number, number],
        original_dtype: 'float32',
      },
    };
    const output = new Float32Array(1);
    const ranges: LoadRange[] = [{ start: 0, end: 1 }];
    const array = mockZarrArray('uint8', [1]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 1, 1);

    expect(written).toBe(1);
    expect(output[0]).toBeCloseTo(64 / 255, 5);
    // Guard against an inverted formula: 64/255 ≈ 0.251, 1 - 64/255 ≈ 0.749.
    expect(output[0]).not.toBeCloseTo(1 - 64 / 255, 5);
  });

  it('dequantizes uint8 data with arbitrary bounds [2.0, 10.0]', async () => {
    // uint8 0   => 2.0 + (0/255) * 8.0 = 2.0
    // uint8 128 => 2.0 + (128/255) * 8.0 ~= 6.016
    // uint8 255 => 2.0 + (255/255) * 8.0 = 10.0
    setMockData(new Uint8Array([0, 128, 255]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'bounded_scalar_uint8',
        bounds: [2.0, 10.0] as [number, number],
        original_dtype: 'float32',
      },
    };
    const output = new Float32Array(3);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint8', [3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 1);

    expect(written).toBe(3);
    expect(output[0]).toBeCloseTo(2.0, 5);
    expect(output[1]).toBeCloseTo(2.0 + (128 / 255) * 8.0, 4);
    expect(output[2]).toBeCloseTo(10.0, 5);
    // Asymmetric interior: distinguishes min + (max-min)*n from a flipped/biased map.
    expect(output[1]).toBeGreaterThan(6.0);
    expect(output[1]).toBeLessThan(6.05);
  });

  it('dequantizes uint16 data with bounds [0, 1]', async () => {
    // uint16 values: 0, 32767, 65535
    setMockData(new Uint16Array([0, 32767, 65535]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'rgb_uint16',
        bounds: [0, 1] as [number, number],
        original_dtype: 'float32',
      },
    };
    const output = new Float32Array(3);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint16', [3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 1);

    expect(written).toBe(3);
    expect(output[0]).toBeCloseTo(0.0, 5);
    expect(output[1]).toBeCloseTo(32767 / 65535, 4);
    expect(output[2]).toBeCloseTo(1.0, 5);
  });

  it('dequantizes with min/max format (not bounds)', async () => {
    // Include an asymmetric interior sample (64) to kill a hardcoded-range
    // mutant: with min=-5, max=5 the interior maps to -5 + (64/255)*10 ≈ -2.49,
    // which would be wrong for any other (min,max) pair.
    setMockData(new Uint8Array([0, 64, 255]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'bounded_scalar_uint8', min: -5.0, max: 5.0, original_dtype: 'float32' },
    };
    const output = new Float32Array(3);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint8', [3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 1);

    expect(written).toBe(3);
    expect(output[0]).toBeCloseTo(-5.0, 5);
    expect(output[1]).toBeCloseTo(-5.0 + (64 / 255) * 10.0, 4);
    expect(output[2]).toBeCloseTo(5.0, 5);
  });

  it('dequantizes 2D quantized data (e.g. rgb colors [N, 3])', async () => {
    // Two RGB pixels, uint8: [0,0,0] and [255,128,64]
    setMockData(new Uint8Array([0, 0, 0, 255, 128, 64]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'rgb_uint8',
        bounds: [0, 1] as [number, number],
        original_dtype: 'float32',
      },
    };
    const output = new Float32Array(6);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('uint8', [10, 3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 3);

    expect(written).toBe(6);
    expect(output[0]).toBeCloseTo(0.0, 5);
    expect(output[3]).toBeCloseTo(1.0, 5);
    expect(output[4]).toBeCloseTo(128 / 255, 4);
    expect(output[5]).toBeCloseTo(64 / 255, 4);
  });

  it('dequantizes log_scalar_uint8 (log-space encoding)', async () => {
    // log_scalar: normalized = val / 255, result = expm1(normalized * max_log)
    const maxLog = 3.5;
    // uint8 0 => expm1(0) = 0
    // uint8 255 => expm1(3.5) = e^3.5 - 1
    // uint8 64 => expm1((64/255) * 3.5)  (asymmetric interior)
    // uint8 128 => expm1((128/255) * 3.5)
    setMockData(new Uint8Array([0, 64, 128, 255]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'log_scalar_uint8', max_log: maxLog, original_dtype: 'float32' },
    };
    const output = new Float32Array(4);
    const ranges: LoadRange[] = [{ start: 0, end: 4 }];
    const array = mockZarrArray('uint8', [4]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 4, 1);

    expect(written).toBe(4);
    expect(output[0]).toBeCloseTo(0.0, 5);
    expect(output[1]).toBeCloseTo(Math.expm1((64 / 255) * maxLog), 4);
    expect(output[2]).toBeCloseTo(Math.expm1((128 / 255) * maxLog), 4);
    expect(output[3]).toBeCloseTo(Math.expm1(maxLog), 3);
  });

  it('dequantizes log_scalar_uint16', async () => {
    const maxLog = 5.0;
    setMockData(new Uint16Array([0, 65535]));

    const attrs: ArrayMetadata = {
      encoding: { name: 'log_scalar_uint16', max_log: maxLog, original_dtype: 'float32' },
    };
    const output = new Float32Array(2);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('uint16', [2]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 1);

    expect(written).toBe(2);
    expect(output[0]).toBeCloseTo(0.0, 5);
    expect(output[1]).toBeCloseTo(Math.expm1(maxLog), 2);
  });

  it('handles multiple ranges for quantized data', async () => {
    // First range: [0, 255], second range: [128]
    mockZarrGet
      .mockResolvedValueOnce({ data: new Uint8Array([0, 255]) } as any)
      .mockResolvedValueOnce({ data: new Uint8Array([128]) } as any);

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'bounded_scalar_uint8',
        bounds: [0, 10] as [number, number],
        original_dtype: 'float32',
      },
    };
    const output = new Float32Array(3);
    const ranges: LoadRange[] = [
      { start: 0, end: 2 },
      { start: 5, end: 6 },
    ];
    const array = mockZarrArray('uint8', [10]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 1);

    expect(written).toBe(3);
    expect(output[0]).toBeCloseTo(0.0, 5);
    expect(output[1]).toBeCloseTo(10.0, 5);
    expect(output[2]).toBeCloseTo((128 / 255) * 10.0, 3);
  });
});

// ---------------------------------------------------------------------------
// loadLUT
// ---------------------------------------------------------------------------

describe('RangeLoader.loadLUT (via loadRanges)', () => {
  let loader: RangeLoader;

  beforeEach(() => {
    loader = createLoader();
    mockZarrGet.mockReset();
  });

  it('decodes LUT row mode with 3-component vectors (e.g. RGB)', async () => {
    // LUT: index 0 -> [1.0, 0.0, 0.0] (red), index 1 -> [0.0, 1.0, 0.0] (green)
    // Indices: [0, 1, 0]
    setMockData(new Uint8Array([0, 1, 0]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
        ],
        lut_mode: 'row',
        original_shape: [3, 3],
      },
    };
    const output = new Float32Array(9);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint8', [3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 3);

    expect(written).toBe(9);
    // Element 0: red [1, 0, 0]
    expect(Array.from(output.subarray(0, 3))).toEqual([1.0, 0.0, 0.0]);
    // Element 1: green [0, 1, 0]
    expect(Array.from(output.subarray(3, 6))).toEqual([0.0, 1.0, 0.0]);
    // Element 2: red [1, 0, 0]
    expect(Array.from(output.subarray(6, 9))).toEqual([1.0, 0.0, 0.0]);
    // Per-channel pinning: index 0 must hit row 0 (red, channel 0 = 1) and
    // index 1 must hit row 1 (green, channel 1 = 1). A reversed-index lookup
    // (idx -> lut[len-1-idx]) would swap these, so assert the discriminating
    // channels are NOT cross-wired.
    expect(output[0]).toBe(1.0); // elem 0, red channel ON
    expect(output[1]).toBe(0.0); // elem 0, green channel OFF
    expect(output[3]).toBe(0.0); // elem 1, red channel OFF
    expect(output[4]).toBe(1.0); // elem 1, green channel ON
  });

  it('decodes LUT row mode with a non-palindromic index sequence (kills reversed-index lookup)', async () => {
    // Three distinct rows and indices [2, 0, 1] (not symmetric): a reversed
    // lookup (idx -> lut[2 - idx]) would map to rows [0, 2, 1], producing a
    // different output, so this pins forward index ordering.
    setMockData(new Uint8Array([2, 0, 1]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [
          [1.0, 0.0, 0.0], // row 0
          [0.0, 1.0, 0.0], // row 1
          [0.0, 0.0, 1.0], // row 2
        ],
        lut_mode: 'row',
        original_shape: [3, 3],
      },
    };
    const output = new Float32Array(9);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];
    const array = mockZarrArray('uint8', [3]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 3, 3);

    expect(written).toBe(9);
    // index 2 -> row 2, index 0 -> row 0, index 1 -> row 1
    expect(Array.from(output.subarray(0, 3))).toEqual([0.0, 0.0, 1.0]);
    expect(Array.from(output.subarray(3, 6))).toEqual([1.0, 0.0, 0.0]);
    expect(Array.from(output.subarray(6, 9))).toEqual([0.0, 1.0, 0.0]);
  });

  it('decodes LUT scalar mode', async () => {
    // LUT: flat list of scalar values [10.0, 20.0, 30.0]
    // Indices: [2, 0, 1, 2]
    setMockData(new Uint8Array([2, 0, 1, 2]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [10.0, 20.0, 30.0],
        lut_mode: 'scalar',
        original_shape: [4, 1],
      },
    };
    const output = new Float32Array(4);
    const ranges: LoadRange[] = [{ start: 0, end: 4 }];
    const array = mockZarrArray('uint8', [4]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 4, 1);

    expect(written).toBe(4);
    expect(Array.from(output)).toEqual([30.0, 10.0, 20.0, 30.0]);
  });

  it('decodes LUT with flat lut array in row mode', async () => {
    // LUT as flat array (not nested): [1,0, 0,1] representing 2 rows of k=2
    // Indices: [1, 0]
    setMockData(new Uint8Array([1, 0]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [1.0, 0.0, 0.0, 1.0], // flat: row0=[1,0], row1=[0,1]
        lut_mode: 'row',
        original_shape: [2, 2],
      },
    };
    const output = new Float32Array(4);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('uint8', [2]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 2);

    expect(written).toBe(4);
    // Index 1 -> [0.0, 1.0], Index 0 -> [1.0, 0.0]
    expect(Array.from(output)).toEqual([0.0, 1.0, 1.0, 0.0]);
  });

  it('decodes LUT with uint16 indices', async () => {
    setMockData(new Uint16Array([0, 2]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint16',
        original_dtype: 'float32',
        lut: [[100.0], [200.0], [300.0]],
        lut_mode: 'row',
        original_shape: [2, 1],
      },
    };
    const output = new Float32Array(2);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('uint16', [2]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 1);

    expect(written).toBe(2);
    expect(Array.from(output)).toEqual([100.0, 300.0]);
  });

  it('handles multiple ranges for LUT data', async () => {
    mockZarrGet
      .mockResolvedValueOnce({ data: new Uint8Array([0]) } as any)
      .mockResolvedValueOnce({ data: new Uint8Array([1]) } as any);

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [
          [1.0, 2.0],
          [3.0, 4.0],
        ],
        lut_mode: 'row',
        original_shape: [10, 2],
      },
    };
    const output = new Float32Array(4);
    const ranges: LoadRange[] = [
      { start: 0, end: 1 },
      { start: 5, end: 6 },
    ];
    const array = mockZarrArray('uint8', [10]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 2);

    expect(written).toBe(4);
    // Range 1: index 0 -> [1, 2], Range 2: index 1 -> [3, 4]
    expect(Array.from(output)).toEqual([1.0, 2.0, 3.0, 4.0]);
  });

  it('defaults to row mode when lut_mode is not specified', async () => {
    setMockData(new Uint8Array([0, 1]));

    const attrs: ArrayMetadata = {
      encoding: {
        name: 'lut_uint8',
        original_dtype: 'float32',
        lut: [[5.0], [10.0]],
        // lut_mode not set => defaults to 'row'
        original_shape: [2, 1],
      },
    };
    const output = new Float32Array(2);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];
    const array = mockZarrArray('uint8', [2]);

    const written = await loader.loadRanges(array, attrs, ranges, output, 2, 1);

    expect(written).toBe(2);
    expect(Array.from(output)).toEqual([5.0, 10.0]);
  });
});

// ---------------------------------------------------------------------------
// loadArrayRef
// ---------------------------------------------------------------------------

describe('RangeLoader.loadArrayRef (via loadRanges)', () => {
  let loader: RangeLoader;

  beforeEach(() => {
    loader = createLoader();
    mockZarrGet.mockReset();
  });

  it('throws when array_ref reaches RangeLoader without spatial-loader pre-resolution', async () => {
    const attrs: ArrayMetadata = {
      encoding: { name: 'array_ref', target: '/SharedNode/colors', hash: 'abc123' },
    };
    const output = new Float32Array(10);
    const ranges: LoadRange[] = [{ start: 0, end: 5 }];
    const array = mockZarrArray('float32', [100, 3]);

    await expect(loader.loadRanges(array, attrs, ranges, output, 5, 3)).rejects.toThrow(
      'Array reference encountered in RangeLoader but not pre-resolved'
    );
  });
});

// ---------------------------------------------------------------------------
// loadRangesResolvingRef
// ---------------------------------------------------------------------------

describe('RangeLoader.loadRangesResolvingRef', () => {
  let loader: RangeLoader;

  beforeEach(() => {
    loader = createLoader();
    mockZarrGet.mockReset();
    mockZarrOpen.mockReset();
    mockZarrRoot.mockReset();
  });

  it('passes non-ref attrs straight through to loadRanges (direct encoding)', async () => {
    setMockData(new Float32Array([1, 2, 3]));
    const output = new Float32Array(3);
    const array = mockZarrArray('float32', [10]);
    const ranges: LoadRange[] = [{ start: 0, end: 3 }];

    const written = await loader.loadRangesResolvingRef(
      array,
      undefined,
      ranges,
      output,
      3,
      1,
      {} as never
    );

    expect(written).toBe(3);
    expect(Array.from(output)).toEqual([1, 2, 3]);
    // No ref → never opened anything.
    expect(mockZarrOpen).not.toHaveBeenCalled();
    expect(mockZarrRoot).not.toHaveBeenCalled();
  });

  it('resolves array_ref by opening the target and delegating to loadRanges', async () => {
    // Wire up a fake target array that returns a deterministic chunk.
    const targetArray = {
      dtype: 'float32',
      shape: [100, 3],
      attrs: {},
    } as unknown as ReturnType<typeof mockZarrArray>;
    const fakeStore = { kind: 'mock-store' } as never;
    const fakeRootLocation = { resolve: vi.fn().mockReturnValue('resolved-target-loc') };
    mockZarrRoot.mockReturnValue(fakeRootLocation as never);
    mockZarrOpen.mockResolvedValue(targetArray as never);
    setMockData(new Float32Array([10, 20, 30, 40, 50, 60]));

    const refAttrs: ArrayMetadata = {
      encoding: { name: 'array_ref', target: '/Shared/colors', hash: 'sha-test' },
    };
    const placeholder = {} as ReturnType<typeof mockZarrArray>;
    const output = new Float32Array(6);
    const ranges: LoadRange[] = [{ start: 0, end: 2 }];

    const written = await loader.loadRangesResolvingRef(
      placeholder,
      refAttrs,
      ranges,
      output,
      2,
      999, // hint should be ignored — target shape says 3
      fakeStore
    );

    expect(mockZarrRoot).toHaveBeenCalledWith(fakeStore);
    expect(fakeRootLocation.resolve).toHaveBeenCalledWith('/Shared/colors');
    expect(mockZarrOpen).toHaveBeenCalledWith('resolved-target-loc', { kind: 'array' });
    expect(written).toBe(6);
    expect(Array.from(output)).toEqual([10, 20, 30, 40, 50, 60]);
  });

  it('uses target.shape[1] as elementsPerItem (overriding caller hint) for ref targets', async () => {
    // Same flow, but verify we do not pass `999` (caller hint) when there's
    // an array_ref. The output length tracks target shape semantics.
    const targetArray = {
      dtype: 'float32',
      shape: [10], // 1-D target → elementsPerItem = 1
      attrs: {},
    } as unknown as ReturnType<typeof mockZarrArray>;
    const fakeStore = {} as never;
    mockZarrRoot.mockReturnValue({ resolve: () => 'tloc' } as never);
    mockZarrOpen.mockResolvedValue(targetArray as never);
    setMockData(new Float32Array([7, 8]));

    const refAttrs: ArrayMetadata = {
      encoding: { name: 'array_ref', target: '/Shared/scalars', hash: 'h' },
    };
    const placeholder = {} as ReturnType<typeof mockZarrArray>;
    const output = new Float32Array(2);

    const written = await loader.loadRangesResolvingRef(
      placeholder,
      refAttrs,
      [{ start: 0, end: 2 }],
      output,
      2,
      999,
      fakeStore
    );

    expect(written).toBe(2);
    expect(Array.from(output)).toEqual([7, 8]);
  });
});

// ---------------------------------------------------------------------------
// Shared singleton helpers
// ---------------------------------------------------------------------------

describe('getSharedRangeLoader / getSharedRefRegistry / reset', () => {
  beforeEach(() => {
    resetSharedRangeLoader();
  });

  it('returns the same instance across calls (singleton)', () => {
    const a = getSharedRangeLoader();
    const b = getSharedRangeLoader();
    expect(a).toBe(b);
  });

  it('uses the supplied registry on first construction', () => {
    const reg = new ArrayRefRegistry();
    const a = getSharedRangeLoader(reg);
    const b = getSharedRangeLoader();
    expect(a).toBe(b);
    // The shared registry should match the one we supplied initially.
    expect(getSharedRefRegistry()).toBe(reg);
  });

  it('ignores a registry passed AFTER first construction', () => {
    const r1 = new ArrayRefRegistry();
    const r2 = new ArrayRefRegistry();
    getSharedRangeLoader(r1);
    getSharedRangeLoader(r2); // ignored
    expect(getSharedRefRegistry()).toBe(r1);
  });

  it('getSharedRefRegistry() lazily creates a registry when called first', () => {
    const reg = getSharedRefRegistry();
    expect(reg).toBeInstanceOf(ArrayRefRegistry);
    // Subsequent call returns the same one.
    expect(getSharedRefRegistry()).toBe(reg);
  });

  it('resetSharedRangeLoader() forces a fresh singleton on next access', () => {
    const before = getSharedRangeLoader();
    resetSharedRangeLoader();
    const after = getSharedRangeLoader();
    expect(after).not.toBe(before);
  });
});

describe('RangeLoader.getDecoder', () => {
  it('returns the underlying ArrayDecoder', () => {
    const reg = new ArrayRefRegistry();
    const loader = new RangeLoader(reg);
    const decoder = loader.getDecoder();
    expect(decoder).toBeDefined();
    // Two calls return the same instance (no rebuild per call).
    expect(loader.getDecoder()).toBe(decoder);
  });
});
