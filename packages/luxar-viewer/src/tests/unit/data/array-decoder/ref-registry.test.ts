/**
 * Unit tests for ArrayRefRegistry (data/array-decoder/ref-registry.ts).
 *
 * The registry is the in-memory cache that resolves `array_ref`
 * deduplication: the Python encoder writes one copy of a Float32Array
 * under a hash and replaces every duplicate occurrence with a reference.
 * These tests exercise the real public API directly — no mocking — since
 * the unit has no external boundary (it is a thin Map wrapper).
 */

import { describe, it, expect } from 'vitest';
import { ArrayRefRegistry } from '../../../../data/array-decoder/ref-registry';

describe('ArrayRefRegistry', () => {
  describe('register / get / has', () => {
    it('registers an array under a hash and retrieves it', () => {
      const registry = new ArrayRefRegistry();
      const arr = new Float32Array([1, 2, 3]);

      expect(registry.has('h1')).toBe(false);
      expect(registry.get('h1')).toBeUndefined();

      registry.register('h1', arr);

      expect(registry.has('h1')).toBe(true);
      // get() returns the exact same instance (reference reuse, not a copy)
      expect(registry.get('h1')).toBe(arr);
      expect(Array.from(registry.get('h1')!)).toEqual([1, 2, 3]);
    });

    it('keeps distinct hashes independent', () => {
      const registry = new ArrayRefRegistry();
      const a = new Float32Array([1, 2]);
      const b = new Float32Array([9, 8, 7]);

      registry.register('a', a);
      registry.register('b', b);

      expect(registry.get('a')).toBe(a);
      expect(registry.get('b')).toBe(b);
      expect(registry.has('a')).toBe(true);
      expect(registry.has('b')).toBe(true);
      expect(registry.has('c')).toBe(false);
    });
  });

  describe('duplicate registration', () => {
    it('does NOT overwrite the first array on duplicate hash (guarded by !has)', () => {
      const registry = new ArrayRefRegistry();
      const first = new Float32Array([1, 1, 1]);
      const second = new Float32Array([2, 2, 2]);

      registry.register('dup', first);
      registry.register('dup', second); // should be a no-op

      // The first array must still be the one stored.
      expect(registry.get('dup')).toBe(first);
      expect(registry.get('dup')).not.toBe(second);
      expect(Array.from(registry.get('dup')!)).toEqual([1, 1, 1]);

      // Count is still 1 — the duplicate was not added.
      expect(registry.getStats().count).toBe(1);
    });
  });

  describe('clear', () => {
    it('empties the registry so has() returns false afterward', () => {
      const registry = new ArrayRefRegistry();
      registry.register('x', new Float32Array([1]));
      registry.register('y', new Float32Array([2, 3]));

      expect(registry.getStats().count).toBe(2);

      registry.clear();

      expect(registry.has('x')).toBe(false);
      expect(registry.has('y')).toBe(false);
      expect(registry.get('x')).toBeUndefined();
      expect(registry.getStats().count).toBe(0);
      expect(registry.getStats().totalBytes).toBe(0);
    });
  });

  describe('getStats', () => {
    it('returns count 0 and totalBytes 0 for an empty registry', () => {
      const registry = new ArrayRefRegistry();
      expect(registry.getStats()).toEqual({ count: 0, totalBytes: 0 });
    });

    it('returns correct count and summed byteLength for registered arrays', () => {
      const registry = new ArrayRefRegistry();
      const a = new Float32Array(3); // 3 * 4 = 12 bytes
      const b = new Float32Array(5); // 5 * 4 = 20 bytes

      registry.register('a', a);
      registry.register('b', b);

      const stats = registry.getStats();
      expect(stats.count).toBe(2);
      expect(stats.totalBytes).toBe(a.byteLength + b.byteLength);
      expect(stats.totalBytes).toBe(12 + 20);
    });

    it('does not double-count bytes for a rejected duplicate registration', () => {
      const registry = new ArrayRefRegistry();
      const a = new Float32Array(4); // 16 bytes

      registry.register('a', a);
      registry.register('a', new Float32Array(100)); // ignored

      const stats = registry.getStats();
      expect(stats.count).toBe(1);
      expect(stats.totalBytes).toBe(16);
    });
  });
});
