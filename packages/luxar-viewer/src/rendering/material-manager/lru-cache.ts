/**
 * Generic LRU helpers for `Map`-backed material caches.
 *
 * `Map` preserves insertion order in JS — the first key returned by
 * `keys()` is the least-recently-used entry. `lruGet` promotes a hit
 * by deleting + re-inserting; `lruSet` evicts from the front until
 * the cache is under `maxSize` and then appends.
 *
 * Extracted from `material-manager.ts` in P6/step 4.2. The original
 * methods were private class members that captured `this.registered
 * Materials`, `this.evictionCount`, and the `log` helper directly;
 * here the eviction-time side effects are threaded through an
 * `onEvict` callback so the helpers are agnostic to who owns the
 * cache.
 *
 * @module rendering/material-manager/lru-cache
 */

/**
 * LRU-aware cache lookup. On hit, promote the entry to
 * most-recently-used by deleting + re-inserting it. Returns the
 * cached value or undefined.
 */
export function lruGet<T>(cache: Map<string, T>, key: string): T | undefined {
  const value = cache.get(key);
  if (value !== undefined) {
    cache.delete(key);
    cache.set(key, value);
  }
  return value;
}

/**
 * LRU-aware cache insert. If `maxSize > 0` and the cache is at or
 * above the bound, evict the LRU entry (first key in insertion order)
 * and invoke `onEvict` with the evicted (key, value) so the owner can
 * dispose the resource and update its counters / registries. Repeat
 * until under the bound, then insert the new entry.
 *
 * `maxSize === 0` disables eviction (unbounded growth).
 */
export function lruSet<T>(
  cache: Map<string, T>,
  key: string,
  value: T,
  maxSize: number,
  onEvict: (key: string, value: T) => void
): void {
  if (maxSize > 0) {
    while (cache.size >= maxSize) {
      const lruKey = cache.keys().next().value;
      if (lruKey === undefined) break;
      const lruValue = cache.get(lruKey);
      cache.delete(lruKey);
      if (lruValue !== undefined) onEvict(lruKey, lruValue);
    }
  }
  cache.set(key, value);
}
