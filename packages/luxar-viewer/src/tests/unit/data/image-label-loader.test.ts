/**
 * Tests for ImageLabelLoader — lazy per-element image fetching from zarr.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock zarrita before importing the loader
vi.mock('zarrita', () => ({
  open: vi.fn(),
  root: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start: number, end: number) => ({ start, end })),
}));

import * as zarr from 'zarrita';
import { ImageLabelLoader } from '../../../data/image-label-loader';

// Helper: create a BigUint64Array of offsets
function makeOffsets(sizes: number[]): BigUint64Array {
  const offsets = new BigUint64Array(sizes.length + 1);
  let pos = BigInt(0);
  offsets[0] = pos;
  for (let i = 0; i < sizes.length; i++) {
    pos += BigInt(sizes[i]);
    offsets[i + 1] = pos;
  }
  return offsets;
}

// JPEG magic header bytes
const JPEG_HEADER = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
// PNG magic header bytes
const PNG_HEADER = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
// WebP magic header: 52 49 46 46 xx xx xx xx 57 45 42 50 (used by detectMimeType internally)

// Mock URL.createObjectURL and revokeObjectURL
let blobUrlCounter = 0;
const createdUrls = new Set<string>();
const revokedUrls = new Set<string>();

beforeEach(() => {
  vi.clearAllMocks();
  blobUrlCounter = 0;
  createdUrls.clear();
  revokedUrls.clear();

  vi.stubGlobal(
    'URL',
    new Proxy(globalThis.URL, {
      get(target, prop) {
        if (prop === 'createObjectURL') {
          return (_blob: Blob) => {
            const url = `blob:test-${blobUrlCounter++}`;
            createdUrls.add(url);
            return url;
          };
        }
        if (prop === 'revokeObjectURL') {
          return (url: string) => {
            revokedUrls.add(url);
          };
        }
        return Reflect.get(target, prop);
      },
    })
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function createLoader(maxCacheBytes = 50 * 1024 * 1024): ImageLabelLoader {
  const store = {} as zarr.Readable;
  const rootLoc = {
    resolve: vi.fn((path: string) => ({ path })),
  } as unknown as zarr.Location<zarr.Readable>;
  return new ImageLabelLoader(store, rootLoc, maxCacheBytes);
}

function setupMocks(
  offsets: BigUint64Array,
  imageData: Uint8Array
): void {
  const mockOffsetsArr = { dtype: 'uint64' };
  const mockBytesArr = { dtype: 'uint8' };

  (zarr.open as any).mockImplementation((loc: any) => {
    if (loc.path?.includes('image_label_offsets')) return Promise.resolve(mockOffsetsArr);
    if (loc.path?.includes('image_label_bytes')) return Promise.resolve(mockBytesArr);
    return Promise.reject(new Error(`Unknown array: ${loc.path}`));
  });

  (zarr.get as any).mockImplementation((arr: any, slices?: any) => {
    if (arr === mockOffsetsArr) {
      return Promise.resolve({ data: offsets });
    }
    if (arr === mockBytesArr && slices) {
      const start = slices[0].start;
      const end = slices[0].end;
      return Promise.resolve({ data: imageData.slice(start, end) });
    }
    return Promise.reject(new Error('Unexpected get call'));
  });
}

describe('ImageLabelLoader', () => {
  describe('getImageUrl', () => {
    it('returns a blob URL for a valid image', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6, 8, 6]); // 3 images
      const imageData = new Uint8Array([
        ...JPEG_HEADER,
        ...PNG_HEADER,
        ...JPEG_HEADER,
      ]);
      setupMocks(offsets, imageData);

      const url = await loader.getImageUrl('/node', 0);
      expect(url).toBeTruthy();
      expect(url).toMatch(/^blob:test-/);
      loader.dispose();
    });

    it('returns null for empty image (equal offsets)', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6, 0, 6]); // middle element has no image
      const imageData = new Uint8Array([...JPEG_HEADER, ...JPEG_HEADER]);
      setupMocks(offsets, imageData);

      const url = await loader.getImageUrl('/node', 1);
      expect(url).toBeNull();
      loader.dispose();
    });

    it('returns null for out-of-range index', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6]);
      const imageData = new Uint8Array([...JPEG_HEADER]);
      setupMocks(offsets, imageData);

      const url = await loader.getImageUrl('/node', 5);
      expect(url).toBeNull();
      loader.dispose();
    });

    it('returns null for negative index', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6]);
      const imageData = new Uint8Array([...JPEG_HEADER]);
      setupMocks(offsets, imageData);

      const url = await loader.getImageUrl('/node', -1);
      expect(url).toBeNull();
      loader.dispose();
    });

    it('caches results in LRU cache', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6]);
      const imageData = new Uint8Array([...JPEG_HEADER]);
      setupMocks(offsets, imageData);

      const url1 = await loader.getImageUrl('/node', 0);
      const url2 = await loader.getImageUrl('/node', 0);

      // Same URL returned (from cache)
      expect(url1).toBe(url2);
      // zarr.get for bytes should only be called once (offsets + bytes)
      // Second call hits cache
      loader.dispose();
    });
  });

  describe('request coalescing', () => {
    it('shares promise for concurrent requests to same element', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6]);
      const imageData = new Uint8Array([...JPEG_HEADER]);
      setupMocks(offsets, imageData);

      // Launch two concurrent requests
      const [url1, url2] = await Promise.all([
        loader.getImageUrl('/node', 0),
        loader.getImageUrl('/node', 0),
      ]);

      expect(url1).toBe(url2);
      loader.dispose();
    });
  });

  describe('LRU eviction', () => {
    it('revokes blob URLs when evicted from cache', async () => {
      // Very small cache: 10 bytes (will evict quickly)
      const loader = createLoader(10);
      const offsets = makeOffsets([6, 8]); // Two images, 6 + 8 = 14 bytes
      const imageData = new Uint8Array([...JPEG_HEADER, ...PNG_HEADER]);
      setupMocks(offsets, imageData);

      const url0 = await loader.getImageUrl('/node', 0);
      expect(url0).toBeTruthy();

      // Loading second image should evict first (cache too small for both)
      const url1 = await loader.getImageUrl('/node', 1);
      expect(url1).toBeTruthy();

      // First URL should have been revoked
      expect(revokedUrls.has(url0!)).toBe(true);
      loader.dispose();
    });
  });

  describe('dispose', () => {
    it('revokes all cached blob URLs on dispose', async () => {
      const loader = createLoader();
      const offsets = makeOffsets([6, 8]);
      const imageData = new Uint8Array([...JPEG_HEADER, ...PNG_HEADER]);
      setupMocks(offsets, imageData);

      const url0 = await loader.getImageUrl('/node', 0);
      const url1 = await loader.getImageUrl('/node', 1);

      loader.dispose();

      expect(revokedUrls.has(url0!)).toBe(true);
      expect(revokedUrls.has(url1!)).toBe(true);
    });
  });

  describe('hasImageLabels', () => {
    it('returns true when has_image_labels is set', () => {
      const loader = createLoader();
      expect(loader.hasImageLabels({ has_image_labels: true })).toBe(true);
      loader.dispose();
    });

    it('returns false when has_image_labels is not set', () => {
      const loader = createLoader();
      expect(loader.hasImageLabels({ has_labels: true })).toBe(false);
      expect(loader.hasImageLabels({})).toBe(false);
      loader.dispose();
    });
  });

  describe('error handling', () => {
    it('returns null when zarr load fails', async () => {
      const loader = createLoader();
      (zarr.open as any).mockRejectedValue(new Error('Network error'));

      const url = await loader.getImageUrl('/node', 0);
      expect(url).toBeNull();
      loader.dispose();
    });
  });
});
