/**
 * Unit tests for LabelLoader.
 *
 * The loader fetches CSR-style label data from a zarr store. We mock the
 * zarrita boundary (`open` / `get`) to drive deterministic responses, then
 * exercise the public surface: caching, request coalescing, empty-label
 * handling, out-of-range indices, hasLabels metadata predicate, and
 * dispose.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LabelLoader } from '../../../../../data/loaders';

vi.mock('zarrita', async () => {
  const actual = await vi.importActual('zarrita');
  return {
    ...actual,
    open: vi.fn(),
    get: vi.fn(),
  };
});
import { open as zarrOpen, get as zarrGet } from 'zarrita';
const mockOpen = vi.mocked(zarrOpen);
const mockGet = vi.mocked(zarrGet);

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function utf8(s: string): Uint8Array {
  return new TextEncoder().encode(s);
}

/**
 * Configure mocks so a getLabel request for `nodePath` finds the supplied
 * `labels`. Each call to LabelLoader#loadNodeLabels invokes:
 *   - zarr.open twice (offsets, then bytes)
 *   - zarr.get twice (offsets data, then bytes data)
 * We push two open()s and two get()s onto the mock queue per request.
 */
function programOneLoad(labels: string[]): void {
  // Concatenated bytes + cumulative offsets in BigUint64 form.
  const concatenated: Uint8Array[] = labels.map(utf8);
  const totalLen = concatenated.reduce((a, b) => a + b.length, 0);
  const bytes = new Uint8Array(totalLen);
  const offsets = new BigUint64Array(labels.length + 1);
  let cursor = 0;
  for (let i = 0; i < labels.length; i++) {
    offsets[i] = BigInt(cursor);
    bytes.set(concatenated[i], cursor);
    cursor += concatenated[i].length;
  }
  offsets[labels.length] = BigInt(cursor);

  // Each open() resolves to a sentinel array; the type doesn't matter since
  // we only care that zarr.get receives the same handle back.
  mockOpen.mockResolvedValueOnce({ kind: 'offsets-array' } as never);
  mockOpen.mockResolvedValueOnce({ kind: 'bytes-array' } as never);
  mockGet.mockResolvedValueOnce({ data: offsets } as never);
  mockGet.mockResolvedValueOnce({ data: bytes } as never);
}

function makeLoader(): LabelLoader {
  const fakeRoot = { resolve: (p: string) => `loc:${p}` } as never;
  const fakeStore = {} as never;
  return new LabelLoader(fakeStore, fakeRoot);
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

describe('LabelLoader.hasLabels', () => {
  const loader = makeLoader();

  it('returns true only when has_labels === true', () => {
    expect(loader.hasLabels({ has_labels: true })).toBe(true);
  });

  it('returns false when has_labels is false, missing, or non-boolean truthy', () => {
    expect(loader.hasLabels({ has_labels: false })).toBe(false);
    expect(loader.hasLabels({})).toBe(false);
    // The check uses === true, so other truthy values are rejected.
    expect(loader.hasLabels({ has_labels: 1 } as unknown as Record<string, unknown>)).toBe(false);
    expect(loader.hasLabels({ has_labels: 'yes' } as unknown as Record<string, unknown>)).toBe(
      false
    );
  });
});

describe('LabelLoader.getLabel — fetch + cache', () => {
  beforeEach(() => {
    mockOpen.mockReset();
    mockGet.mockReset();
  });

  it('returns the decoded label for a valid index', async () => {
    const loader = makeLoader();
    programOneLoad(['hello', 'world', 'fizz']);

    expect(await loader.getLabel('/Points', 0)).toBe('hello');
  });

  it('caches subsequent reads against the same node — only one zarr fetch occurs', async () => {
    const loader = makeLoader();
    programOneLoad(['a', 'b', 'c']);

    expect(await loader.getLabel('/Points', 0)).toBe('a');
    expect(await loader.getLabel('/Points', 1)).toBe('b');
    expect(await loader.getLabel('/Points', 2)).toBe('c');

    // open + get were each called exactly twice (offsets+bytes, ONCE).
    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(2);

    // A DIFFERENT node path is cached independently: it triggers its own
    // fetch (two more open/get) and returns its own labels, not the first
    // node's cached value.
    programOneLoad(['x', 'y']);
    expect(await loader.getLabel('/Lines', 0)).toBe('x');
    expect(await loader.getLabel('/Lines', 1)).toBe('y');
    expect(mockOpen).toHaveBeenCalledTimes(4);
    expect(mockGet).toHaveBeenCalledTimes(4);
    // And the original node is still served from cache (no further fetches).
    expect(await loader.getLabel('/Points', 0)).toBe('a');
    expect(mockOpen).toHaveBeenCalledTimes(4);
  });

  it('returns null for empty labels (offsets[i] === offsets[i+1])', async () => {
    const loader = makeLoader();
    programOneLoad(['filled', '']);

    expect(await loader.getLabel('/Points', 0)).toBe('filled');
    expect(await loader.getLabel('/Points', 1)).toBeNull();
  });

  it('returns null for negative or out-of-range indices', async () => {
    const loader = makeLoader();
    programOneLoad(['only-one']);

    expect(await loader.getLabel('/Points', -1)).toBeNull();
    expect(await loader.getLabel('/Points', 1)).toBeNull();
    expect(await loader.getLabel('/Points', 100)).toBeNull();
  });

  it('coalesces concurrent loads of the same node into a single fetch', async () => {
    const loader = makeLoader();
    programOneLoad(['shared']);

    const [a, b, c] = await Promise.all([
      loader.getLabel('/Same', 0),
      loader.getLabel('/Same', 0),
      loader.getLabel('/Same', 0),
    ]);
    expect(a).toBe('shared');
    expect(b).toBe('shared');
    expect(c).toBe('shared');
    // Only one underlying load — open/get each called twice (one node).
    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockGet).toHaveBeenCalledTimes(2);
  });

  it('strips a leading slash from the node path before resolving', async () => {
    const loader = makeLoader();
    programOneLoad(['x']);

    await loader.getLabel('/My/Node', 0);

    // The fake root.resolve returns "loc:<path>", so we can read what was
    // resolved by inspecting how the array stand-in was opened. Both
    // resolves must use the cleaned path (no leading slash).
    expect(mockOpen).toHaveBeenCalledTimes(2);
    expect(mockOpen.mock.calls[0][0]).toBe('loc:My/Node/label_offsets');
    expect(mockOpen.mock.calls[1][0]).toBe('loc:My/Node/label_bytes');
  });

  it('returns null for any element when the underlying zarr fetch fails', async () => {
    const loader = makeLoader();
    mockOpen.mockRejectedValueOnce(new Error('not found'));

    // Default warn spy so the failure log doesn't pollute test output.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      // Returns empty array on error → any index falls through to the
      // out-of-range branch.
      expect(await loader.getLabel('/Missing', 0)).toBeNull();
      expect(warnSpy).toHaveBeenCalled();
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

  it('clears the cache so the next fetch re-runs', async () => {
    const loader = makeLoader();
    programOneLoad(['cached']);

    await loader.getLabel('/Points', 0);
    expect(mockOpen).toHaveBeenCalledTimes(2);

    loader.dispose();

    // Re-program for the second load.
    programOneLoad(['cached2']);
    const result = await loader.getLabel('/Points', 0);
    expect(result).toBe('cached2');
    // Two more open calls = four total now.
    expect(mockOpen).toHaveBeenCalledTimes(4);
  });
});
