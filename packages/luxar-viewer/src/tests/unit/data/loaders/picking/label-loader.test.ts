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
import { open as zarrOpen, get as zarrGet, NotFoundError } from 'zarrita';
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
    mockOpen.mockRejectedValueOnce(new Error('decode failed'));

    // Default warn spy so the failure log doesn't pollute test output.
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      // Returns empty array on error → any index falls through to the
      // out-of-range branch.
      expect(await loader.getLabel('/Broken', 0)).toBeNull();
      expect(warnSpy).toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does not warn when the node simply has no label arrays', async () => {
    // The picker calls getLabel for whatever it hit, and most nodes carry no
    // labels at all (every coarse LOD level of a labelled ladder, for one).
    // That is an ordinary miss, not a failure worth a console warning.
    // The real class zarrita throws for an absent node, so this pins the
    // `instanceof` branch of the shared isNotFoundError guard.
    const loader = makeLoader();
    mockOpen.mockRejectedValueOnce(new NotFoundError('v2 array', { path: '/x/.zarray' }));

    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    try {
      expect(await loader.getLabel('/Unlabelled', 0)).toBeNull();
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});

describe('LabelLoader.getLabel — a consecutive run of equal labels decodes once', () => {
  beforeEach(() => {
    mockOpen.mockReset();
    mockGet.mockReset();
  });

  it('decodes a 64-element broadcast node exactly once', async () => {
    const loader = makeLoader();
    // How a producer tags a whole node: one string repeated per element.
    const broadcast = 'AF_L — Arcuate fasciculus (left) · association';
    programOneLoad(new Array(64).fill(broadcast));

    const decodeSpy = vi.spyOn(TextDecoder.prototype, 'decode');
    try {
      expect(await loader.getLabel('/Tract', 0)).toBe(broadcast);
      expect(await loader.getLabel('/Tract', 63)).toBe(broadcast);
      // The whole node is decoded in one pass on first access; 63 of the 64
      // elements are served by reusing the previous element's string.
      expect(decodeSpy).toHaveBeenCalledTimes(1);
    } finally {
      decodeSpy.mockRestore();
    }
  });

  it('still decodes once per element when every label differs', async () => {
    // The reuse must not cost the all-distinct case anything, and must never
    // collapse two different labels into one.
    const loader = makeLoader();
    const distinct = Array.from({ length: 64 }, (_, i) => `label-${i}`);
    programOneLoad(distinct);

    const decodeSpy = vi.spyOn(TextDecoder.prototype, 'decode');
    try {
      expect(await loader.getLabel('/Distinct', 0)).toBe('label-0');
      expect(decodeSpy).toHaveBeenCalledTimes(64);
      expect(await loader.getLabel('/Distinct', 63)).toBe('label-63');
    } finally {
      decodeSpy.mockRestore();
    }
  });

  it('keeps distinct labels distinct when repeats and empties are interleaved', async () => {
    const loader = makeLoader();
    // No two ADJACENT entries are equal here, so this pins the run boundaries
    // without ever taking the reuse branch.
    programOneLoad(['red', 'green', 'red', '', 'green', 'blue', 'red']);

    expect(await loader.getLabel('/Mixed', 0)).toBe('red');
    expect(await loader.getLabel('/Mixed', 1)).toBe('green');
    expect(await loader.getLabel('/Mixed', 2)).toBe('red');
    expect(await loader.getLabel('/Mixed', 3)).toBeNull();
    expect(await loader.getLabel('/Mixed', 4)).toBe('green');
    expect(await loader.getLabel('/Mixed', 5)).toBe('blue');
    expect(await loader.getLabel('/Mixed', 6)).toBe('red');
  });

  it('does not let an empty label bridge a run across it', async () => {
    // The empty branch must still advance the previous-range bookkeeping. If
    // it does not, the element after an empty compares against the element
    // BEFORE it and wrongly reuses: ['red','',''] instead of ['red','','red'].
    const loader = makeLoader();
    programOneLoad(['red', '', 'red']);

    expect(await loader.getLabel('/Gap', 0)).toBe('red');
    expect(await loader.getLabel('/Gap', 1)).toBeNull();
    expect(await loader.getLabel('/Gap', 2)).toBe('red');
  });

  it('resumes a run correctly after an embedded empty label', async () => {
    // Same defect, with real runs either side: a stale previous range yields
    // ['red','red','','',''] instead of ['red','red','','red','red'].
    const loader = makeLoader();
    programOneLoad(['red', 'red', '', 'red', 'red']);

    expect(await loader.getLabel('/Runs', 0)).toBe('red');
    expect(await loader.getLabel('/Runs', 1)).toBe('red');
    expect(await loader.getLabel('/Runs', 2)).toBeNull();
    expect(await loader.getLabel('/Runs', 3)).toBe('red');
    expect(await loader.getLabel('/Runs', 4)).toBe('red');
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
