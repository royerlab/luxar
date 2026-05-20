import { describe, it, expect, beforeEach } from 'vitest';
import {
  OPFSBucketCache,
  getBucket,
  keyToFileName,
} from '../../../../cache/multi-level-caching-store/opfs-store/buckets';

describe('getBucket', () => {
  it('returns a 2-char lowercase hex string', () => {
    for (const key of ['', 'a', 'metadata/.zattrs', 'points/positions/0.0.0']) {
      expect(getBucket(key)).toMatch(/^[0-9a-f]{2}$/);
    }
  });

  it('is deterministic for the same input', () => {
    expect(getBucket('foo/bar')).toBe(getBucket('foo/bar'));
    expect(getBucket('1234567890')).toBe(getBucket('1234567890'));
  });

  it('distributes ~uniformly across 256 buckets', () => {
    const counts = new Map<string, number>();
    for (let i = 0; i < 10_000; i++) {
      const bucket = getBucket(`points/positions/${i}.${i % 7}.${i % 13}`);
      counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
    }
    // 10k / 256 ≈ 39 per bucket; loose bound to absorb non-uniform hashes.
    for (const count of counts.values()) {
      expect(count).toBeLessThan(200);
    }
    // Should hit nearly every bucket.
    expect(counts.size).toBeGreaterThan(200);
  });
});

describe('keyToFileName', () => {
  it('produces filesystem-safe base64url (no +, /, =)', () => {
    for (const key of ['', 'a', 'a/b/c', 'foo.bar.baz', 'metadata/.zattrs']) {
      const fn = keyToFileName(key);
      expect(fn).not.toContain('+');
      expect(fn).not.toContain('/');
      expect(fn).not.toContain('=');
    }
  });

  it('round-trips back to the original key (UTF-8 base64url decode)', () => {
    function decode(b64url: string): string {
      // Pad and convert back to standard base64.
      const b64 = b64url.replace(/-/g, '+').replace(/_/g, '/');
      const padding = (4 - (b64.length % 4)) % 4;
      const binary = atob(b64 + '='.repeat(padding));
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return new TextDecoder().decode(bytes);
    }
    for (const key of ['', 'points/positions/0.0.0', 'metadata/.zattrs', 'a-b_c=d+e/f']) {
      expect(decode(keyToFileName(key))).toBe(key);
    }
  });

  it('handles non-ASCII keys (CJK + emoji)', () => {
    const cjk = keyToFileName('日本語/データ');
    const emoji = keyToFileName('🎉/✨');
    expect(cjk).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(emoji).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  it('is deterministic and known-stable (regression anchor)', () => {
    // Stable encoding — if this ever changes, OPFS_ENCODING_VERSION must
    // bump.
    expect(keyToFileName('points/positions/0.0.0')).toBe('cG9pbnRzL3Bvc2l0aW9ucy8wLjAuMA');
  });
});

describe('OPFSBucketCache', () => {
  /**
   * Minimal FileSystemDirectoryHandle stub: returns a fresh object per
   * `getDirectoryHandle` call so the cache effect is visible (caller
   * gets the cached one, not a fresh one). Track call count to assert
   * caching behaviour.
   */
  function mockRoot() {
    const dirCallCount = new Map<string, number>();
    const dirHandles = new Map<string, FileSystemDirectoryHandle>();
    const root = {
      async getDirectoryHandle(name: string, opts?: { create?: boolean }) {
        dirCallCount.set(name, (dirCallCount.get(name) ?? 0) + 1);
        // Simulate "doesn't exist" when create is falsy and we've not
        // returned it before in this mock session.
        if (!opts?.create && !dirHandles.has(name)) {
          throw new Error('not found');
        }
        let h = dirHandles.get(name);
        if (!h) {
          h = {
            async getFileHandle(_fname: string, _o?: { create?: boolean }) {
              return {} as FileSystemFileHandle;
            },
          } as unknown as FileSystemDirectoryHandle;
          dirHandles.set(name, h);
        }
        return h;
      },
    } as unknown as FileSystemDirectoryHandle;
    return { root, dirCallCount };
  }

  let cache: OPFSBucketCache;
  beforeEach(() => {
    cache = new OPFSBucketCache();
  });

  it('caches the handle for a bucket across repeat calls', async () => {
    const { root, dirCallCount } = mockRoot();
    const h1 = await cache.getHandle(root, 'aa', true);
    const h2 = await cache.getHandle(root, 'aa', true);
    expect(h1).toBe(h2);
    // Second call should NOT have re-invoked getDirectoryHandle.
    expect(dirCallCount.get('aa')).toBe(1);
  });

  it('returns null when getDirectoryHandle throws (create=false + missing)', async () => {
    const { root } = mockRoot();
    const h = await cache.getHandle(root, '00', false);
    expect(h).toBeNull();
  });

  it('invalidate() drops only the named bucket', async () => {
    const { root, dirCallCount } = mockRoot();
    await cache.getHandle(root, 'aa', true);
    await cache.getHandle(root, 'bb', true);
    cache.invalidate('aa');
    // Re-fetching 'aa' must call through again; 'bb' should still be cached.
    await cache.getHandle(root, 'aa', true);
    await cache.getHandle(root, 'bb', true);
    expect(dirCallCount.get('aa')).toBe(2);
    expect(dirCallCount.get('bb')).toBe(1);
  });

  it('clear() drops every cached handle', async () => {
    const { root, dirCallCount } = mockRoot();
    await cache.getHandle(root, 'aa', true);
    await cache.getHandle(root, 'bb', true);
    cache.clear();
    await cache.getHandle(root, 'aa', true);
    await cache.getHandle(root, 'bb', true);
    expect(dirCallCount.get('aa')).toBe(2);
    expect(dirCallCount.get('bb')).toBe(2);
  });

  it('navigateToFile resolves to the right bucket + filename', async () => {
    const key = 'points/positions/0.0.0';
    const bucket = getBucket(key);
    const expectedFileName = keyToFileName(key);

    let observedBucket = '';
    let observedFileName = '';
    const spyRoot = {
      async getDirectoryHandle(name: string) {
        observedBucket = name;
        return {
          async getFileHandle(fname: string) {
            observedFileName = fname;
            return {} as FileSystemFileHandle;
          },
        } as unknown as FileSystemDirectoryHandle;
      },
    } as unknown as FileSystemDirectoryHandle;

    await cache.navigateToFile(spyRoot, key, true);
    expect(observedBucket).toBe(bucket);
    expect(observedFileName).toBe(expectedFileName);
  });

  it('navigateToFile throws when the bucket cannot be accessed', async () => {
    const failingRoot = {
      async getDirectoryHandle() {
        throw new Error('opfs failure');
      },
    } as unknown as FileSystemDirectoryHandle;
    await expect(cache.navigateToFile(failingRoot, 'foo', false)).rejects.toThrow(
      /Cannot access bucket/
    );
  });
});
