/**
 * Unit tests for LabelLoader's lazy per-element CSR reads.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { LabelLoader } from '../../../../../data/loaders';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    open: vi.fn(),
    get: vi.fn(),
    slice: vi.fn((start: number, end: number) => ({ start, end })),
  };
});

import { get as zarrGet, NotFoundError, open as zarrOpen } from 'zarrita';

const mockOpen = vi.mocked(zarrOpen);
const mockGet = vi.mocked(zarrGet);

interface MockArrays {
  offsets: BigUint64Array;
  bytes: Uint8Array;
  offsetsArray: { kind: string; shape: number[] };
  bytesArray: { kind: string; shape: number[] };
}

function encodeLabels(labels: string[]): Pick<MockArrays, 'offsets' | 'bytes'> {
  const encoded = labels.map((label) => new TextEncoder().encode(label));
  const totalBytes = encoded.reduce((sum, value) => sum + value.length, 0);
  const bytes = new Uint8Array(totalBytes);
  const offsets = new BigUint64Array(labels.length + 1);
  let cursor = 0;

  for (let index = 0; index < encoded.length; index++) {
    offsets[index] = BigInt(cursor);
    bytes.set(encoded[index], cursor);
    cursor += encoded[index].length;
  }
  offsets[labels.length] = BigInt(cursor);
  return { offsets, bytes };
}

function programArrays(labels: string[]): MockArrays {
  const encoded = encodeLabels(labels);
  const arrays: MockArrays = {
    ...encoded,
    offsetsArray: { kind: 'offsets-array', shape: [encoded.offsets.length] },
    bytesArray: { kind: 'bytes-array', shape: [encoded.bytes.length] },
  };

  mockOpen.mockResolvedValueOnce(arrays.offsetsArray as never);
  mockOpen.mockResolvedValueOnce(arrays.bytesArray as never);
  mockGet.mockImplementation(async (array, selection) => {
    if (!selection) throw new Error('full-array reads are forbidden');
    const range = selection[0] as unknown as { start: number; end: number };
    if (array === arrays.offsetsArray) {
      return { data: arrays.offsets.slice(range.start, range.end) } as never;
    }
    if (array === arrays.bytesArray) {
      return { data: arrays.bytes.slice(range.start, range.end) } as never;
    }
    throw new Error('unexpected array handle');
  });
  return arrays;
}

function makeLoader(maxCacheBytes = 1024 * 1024, channel: 'labels' | 'keys' = 'labels') {
  const fakeRoot = { resolve: (path: string) => `loc:${path}` } as never;
  return new LabelLoader(fakeRoot, channel, maxCacheBytes);
}

function selectedRange(callIndex: number): { start: number; end: number } {
  const selection = mockGet.mock.calls[callIndex][1] as unknown as Array<{
    start: number;
    end: number;
  }>;
  return selection[0];
}

describe('LabelLoader.hasLabels', () => {
  it('checks the configured channel metadata', () => {
    expect(makeLoader().hasLabels({ has_labels: true })).toBe(true);
    expect(makeLoader().hasLabels({ has_labels: false })).toBe(false);
    expect(makeLoader(1024, 'keys').hasLabels({ has_keys: true })).toBe(true);
    expect(makeLoader(1024, 'keys').hasLabels({ has_labels: true })).toBe(false);
  });
});

describe('LabelLoader.getLabel', () => {
  beforeEach(() => {
    mockOpen.mockReset();
    mockGet.mockReset();
  });

  it('slice-reads only one element from both CSR arrays', async () => {
    const loader = makeLoader();
    const arrays = programArrays(['zero', 'one', 'two']);

    expect(await loader.getLabel('/Points', 1)).toBe('one');
    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(2);
    expect(mockGet.mock.calls[0][0]).toBe(arrays.offsetsArray);
    expect(selectedRange(0)).toEqual({ start: 1, end: 3 });
    expect(mockGet.mock.calls[1][0]).toBe(arrays.bytesArray);
    expect(selectedRange(1)).toEqual({ start: 4, end: 7 });
  });

  it('does not issue a whole-array read for a multi-million-element node', async () => {
    const loader = makeLoader();
    const arrays = programArrays(['selected']);
    arrays.offsetsArray.shape = [5_000_001];

    expect(await loader.getLabel('/Huge', 0)).toBe('selected');
    expect(mockGet).toHaveBeenCalledTimes(2);
    expect(mockGet.mock.calls.every((call) => call[1] !== undefined)).toBe(true);
  });

  it('caches decoded labels but opens each node arrays only once', async () => {
    const loader = makeLoader();
    programArrays(['a', 'b']);

    expect(await loader.getLabel('/Points', 0)).toBe('a');
    expect(await loader.getLabel('/Points', 0)).toBe('a');
    expect(await loader.getLabel('/Points', 1)).toBe('b');

    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(4);
  });

  it('coalesces concurrent requests for the same element', async () => {
    const loader = makeLoader();
    programArrays(['shared']);

    await expect(
      Promise.all([
        loader.getLabel('/Points', 0),
        loader.getLabel('/Points', 0),
        loader.getLabel('/Points', 0),
      ])
    ).resolves.toEqual(['shared', 'shared', 'shared']);
    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(2);
  });

  it('coalesces array opens across concurrent reads of different elements', async () => {
    const loader = makeLoader();
    programArrays(['left', 'right']);

    await expect(
      Promise.all([loader.getLabel('/Points', 0), loader.getLabel('/Points', 1)])
    ).resolves.toEqual(['left', 'right']);
    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(4);
  });

  it('returns null for an empty label without reading label_bytes', async () => {
    const loader = makeLoader();
    programArrays(['filled', '']);

    expect(await loader.getLabel('/Points', 1)).toBeNull();
    expect(await loader.getLabel('/Points', 1)).toBeNull();
    expect(mockGet).toHaveBeenCalledTimes(1);
    expect(selectedRange(0)).toEqual({ start: 1, end: 3 });
  });

  it('rejects invalid indices before reading chunks', async () => {
    const loader = makeLoader();
    programArrays(['only-one']);

    expect(await loader.getLabel('/Points', -1)).toBeNull();
    expect(await loader.getLabel('/Points', 1)).toBeNull();
    expect(await loader.getLabel('/Points', 1.5)).toBeNull();
    expect(mockGet).not.toHaveBeenCalled();
  });

  it('uses a bounded LRU for decoded labels', async () => {
    const loader = makeLoader(64);
    programArrays(['aa', 'bb']);

    expect(await loader.getLabel('/Points', 0)).toBe('aa');
    expect(await loader.getLabel('/Points', 1)).toBe('bb');
    expect(await loader.getLabel('/Points', 0)).toBe('aa');

    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(6);
  });

  it('strips a leading slash and supports the keys channel', async () => {
    const loader = makeLoader(1024, 'keys');
    programArrays(['key']);

    expect(await loader.getLabel('/My/Node', 0)).toBe('key');
    expect(mockOpen.mock.calls[0][0]).toBe('loc:My/Node/key_offsets');
    expect(mockOpen.mock.calls[1][0]).toBe('loc:My/Node/key_bytes');
  });

  it('demotes and remembers a missing offsets array', async () => {
    const loader = makeLoader();
    mockOpen.mockRejectedValueOnce(new NotFoundError('v2 array', { path: '/x/.zarray' }));
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    try {
      expect(await loader.getLabel('/Unlabelled', 0)).toBeNull();
      expect(await loader.getLabel('/Unlabelled', 1)).toBeNull();
      expect(warnSpy).not.toHaveBeenCalled();
      expect(mockOpen).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does not evict decoded labels when an unlabelled node is hovered', async () => {
    const loader = makeLoader(64);
    programArrays(['retained']);

    expect(await loader.getLabel('/Labelled', 0)).toBe('retained');
    mockOpen.mockRejectedValueOnce(new NotFoundError('v2 array', { path: '/x/.zarray' }));
    expect(await loader.getLabel('/Unlabelled', 0)).toBeNull();
    expect(await loader.getLabel('/Unlabelled', 1)).toBeNull();
    expect(await loader.getLabel('/Labelled', 0)).toBe('retained');

    expect(mockGet).toHaveBeenCalledTimes(2);
  });

  it('warns and remembers when bytes metadata is missing', async () => {
    const loader = makeLoader();
    mockOpen.mockResolvedValueOnce({ kind: 'offsets-array', shape: [2] } as never);
    mockOpen.mockRejectedValueOnce(
      new NotFoundError('v2 array', { path: '/x/label_bytes/.zarray' })
    );
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    try {
      expect(await loader.getLabel('/Broken', 0)).toBeNull();
      expect(await loader.getLabel('/Broken', 1)).toBeNull();
      expect(warnSpy).toHaveBeenCalledTimes(1);
      expect(mockOpen).toHaveBeenCalledTimes(2);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('retries after a transient array-open failure', async () => {
    const loader = makeLoader();
    const arrays = encodeLabels(['recovered']);
    const offsetsArray = { kind: 'offsets-array', shape: [arrays.offsets.length] };
    const bytesArray = { kind: 'bytes-array', shape: [arrays.bytes.length] };
    mockOpen.mockRejectedValueOnce(new Error('HTTP 503: label_offsets metadata'));
    mockOpen.mockResolvedValueOnce(offsetsArray as never);
    mockOpen.mockResolvedValueOnce(bytesArray as never);
    mockGet.mockResolvedValueOnce({ data: arrays.offsets } as never);
    mockGet.mockResolvedValueOnce({ data: arrays.bytes } as never);
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    try {
      expect(await loader.getLabel('/TransientOpen', 0)).toBeNull();
      expect(await loader.getLabel('/TransientOpen', 0)).toBe('recovered');
      expect(mockOpen).toHaveBeenCalledTimes(3);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('retries after a transient sliced chunk read failure', async () => {
    const loader = makeLoader();
    mockOpen.mockResolvedValueOnce({ kind: 'offsets-array', shape: [2] } as never);
    mockOpen.mockResolvedValueOnce({ kind: 'bytes-array', shape: [5] } as never);
    mockGet.mockRejectedValueOnce(new Error('HTTP 503: /x/label_offsets/0'));
    mockGet.mockResolvedValueOnce({ data: new BigUint64Array([0n, 5n]) } as never);
    mockGet.mockResolvedValueOnce({ data: new TextEncoder().encode('retry') } as never);
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    try {
      expect(await loader.getLabel('/BrokenChunk', 0)).toBeNull();
      expect(await loader.getLabel('/BrokenChunk', 0)).toBe('retry');
      expect(mockGet).toHaveBeenCalledTimes(3);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('rejects descending offsets from a corrupt store', async () => {
    const loader = makeLoader();
    mockOpen.mockResolvedValueOnce({ kind: 'offsets-array', shape: [2] } as never);
    mockOpen.mockResolvedValueOnce({ kind: 'bytes-array', shape: [4] } as never);
    mockGet.mockResolvedValueOnce({ data: new BigUint64Array([4n, 2n]) } as never);
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    try {
      expect(await loader.getLabel('/Corrupt', 0)).toBeNull();
      expect(mockGet).toHaveBeenCalledTimes(1);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('rejects a truncated byte slice from a corrupt store', async () => {
    const loader = makeLoader();
    mockOpen.mockResolvedValueOnce({ kind: 'offsets-array', shape: [2] } as never);
    mockOpen.mockResolvedValueOnce({ kind: 'bytes-array', shape: [4] } as never);
    mockGet.mockResolvedValueOnce({ data: new BigUint64Array([0n, 4n]) } as never);
    mockGet.mockResolvedValueOnce({ data: new Uint8Array([1, 2]) } as never);
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);

    try {
      expect(await loader.getLabel('/Truncated', 0)).toBeNull();
      expect(mockGet).toHaveBeenCalledTimes(2);
      expect(warnSpy).toHaveBeenCalledTimes(1);
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('LabelLoader.dispose', () => {
  beforeEach(() => {
    mockOpen.mockReset();
    mockGet.mockReset();
  });

  it('clears array handles and decoded labels', async () => {
    const loader = makeLoader();
    programArrays(['first']);
    expect(await loader.getLabel('/Points', 0)).toBe('first');

    loader.dispose();
    programArrays(['second']);
    expect(await loader.getLabel('/Points', 0)).toBe('second');
    expect(mockOpen).toHaveBeenCalledTimes(4);
  });
});
