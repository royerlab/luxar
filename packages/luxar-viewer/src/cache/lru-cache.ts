/**
 * LRU Cache using Map for O(1) get/set/delete operations.
 * Map maintains insertion order, enabling efficient LRU tracking.
 */
export class LRUCache<V> {
  private cache = new Map<string, V>();
  private maxSize: number;
  private currentSize = 0;
  private getSize: (v: V) => number;
  private onEvict?: (key: string, value: V) => void;

  // Pinned keys are protected from ROUTINE eviction: victim selection skips
  // them so a caller can keep a just-inserted entry alive for a bounded window
  // (e.g. a prefetched next-frame slice that must survive until the foreground
  // consumes it). If EVERY remaining entry is pinned and the cache is still
  // over budget, a pinned entry is evicted as a last resort so the byte
  // invariant is never violated.
  private pinned = new Set<string>();

  // Hit/miss tracking for monitoring
  private hits = 0;
  private misses = 0;
  private evictions = 0;

  constructor(
    maxSize: number,
    getSize: (v: V) => number,
    onEvict?: (key: string, value: V) => void
  ) {
    this.maxSize = maxSize;
    this.getSize = getSize;
    this.onEvict = onEvict;
  }

  /**
   * Get value from cache with LRU promotion.
   *
   * On hit, moves item to end of Map (most recently used position).
   * This ensures least recently used items are at the beginning for eviction.
   *
   * @param key - Cache key to lookup
   * @returns Cached value if present, undefined if not found
   *
   * @example
   * ```typescript
   * const cache = new LRUCache<Uint8Array>(1024 * 1024, v => v.byteLength);
   * const chunk = cache.get('positions/0.0.0');
   * if (chunk) {
   *   // Cache hit - chunk moved to MRU position
   *   console.log(`Hit: ${chunk.byteLength} bytes`);
   * } else {
   *   // Cache miss - need to fetch
   * }
   * ```
   *
   * @param options.countStats - `false` promotes as usual but leaves the
   *   hit/miss counters alone (a prefetch read that must not move the
   *   hit-rate statistic). Defaults to `true`.
   * @remarks Performance: O(1) - two Map operations (delete + set for reordering)
   */
  get(key: string, options?: { countStats?: boolean }): V | undefined {
    const count = options?.countStats !== false;
    const value = this.cache.get(key);
    if (value !== undefined) {
      if (count) this.hits++;
      // Move to end (most recently used) - O(1) with Map
      this.cache.delete(key);
      this.cache.set(key, value);
    } else if (count) {
      this.misses++;
    }
    return value;
  }

  /**
   * Set value in cache with automatic LRU eviction.
   *
   * If key exists, updates value and recalculates size.
   * If cache is full, evicts least recently used items until space available.
   * New item is always added at end (most recently used position).
   *
   * Eviction policy: Remove items from beginning of Map (oldest) until
   * sufficient space. JavaScript Map maintains insertion order.
   *
   * @param key - Cache key
   * @param value - Value to cache (size calculated via getSize function)
   * @returns Whether `value` was stored. `false` means it exceeds the whole
   *   budget; any previous value under `key` has then been dropped too.
   *
   * @example
   * ```typescript
   * const cache = new LRUCache<Uint8Array>(64 * 1024, v => v.byteLength);
   *
   * // Add chunk (may trigger eviction if cache full)
   * const chunk = new Uint8Array(32 * 1024); // 32KB
   * cache.set('positions/0.0.0', chunk);
   * console.log(`Cache: ${cache.count} items, ${cache.size} bytes`);
   * ```
   *
   * @example
   * ```typescript
   * // Eviction demonstration
   * cache.set('a', new Uint8Array(30 * 1024)); // 30KB
   * cache.set('b', new Uint8Array(30 * 1024)); // 30KB, total 60KB
   * cache.set('c', new Uint8Array(30 * 1024)); // 30KB, evicts 'a' (LRU)
   * console.log(cache.has('a')); // false - evicted
   * console.log(cache.has('b')); // true - still cached
   * ```
   *
   * @remarks Performance: O(k) where k = number of evictions needed (typically 0-2)
   */
  set(key: string, value: V, opts?: { evictMostRecent?: boolean }): boolean {
    const size = this.getSize(value);

    // Reject items that exceed total cache capacity to avoid permanent
    // size invariant violation (currentSize > maxSize). Nothing ELSE is
    // evicted for it — but an existing entry under the same key is dropped
    // (through `delete`, so `onEvict` sees it): the caller has just replaced
    // that value, and keeping the old one would serve stale bytes under the
    // key, and let a following `pin(key)` protect them.
    if (size > this.maxSize) {
      this.delete(key);
      return false;
    }

    // If exists, remove old
    if (this.cache.has(key)) {
      this.currentSize -= this.getSize(this.cache.get(key)!);
      this.cache.delete(key);
    }

    // Evict until space available. Default policy: least-recently-used
    // (Map head). Scan policy (`evictMostRecent`, opted into per-store by
    // callers that KNOW they are in a sequential scan, e.g. dimension
    // playback): evict the MOST-recently-used entry instead — during a
    // cyclic scan the entry just behind the playhead is the one needed
    // farthest in the future, so evicting it (and keeping the oldest
    // prefix resident) converts the classic 0%-hit LRU scan pathology
    // into hits ≈ budget/working-set clustered right after the wrap.
    // The key being inserted is never a victim (it's not in the Map here).
    while (this.currentSize + size > this.maxSize && this.cache.size > 0) {
      const victimKey = this.selectVictim(opts?.evictMostRecent ?? false);
      if (!victimKey) break; // Safety check (should never happen)
      const victimValue = this.cache.get(victimKey)!;
      this.onEvict?.(victimKey, victimValue);
      this.currentSize -= this.getSize(victimValue);
      this.cache.delete(victimKey);
      this.pinned.delete(victimKey); // a last-resort eviction clears its pin
      this.evictions++;
    }

    this.cache.set(key, value);
    this.currentSize += size;
    return true;
  }

  /**
   * Choose the next eviction victim. Prefers an UNPINNED key — the oldest
   * (Map head) under the default LRU policy, or the newest (Map tail) under the
   * scan policy (`evictMostRecent`). If every entry is pinned, falls back to the
   * plain oldest/newest so the byte budget can still be honored (pinned entries
   * are eviction-exempt only best-effort, never a hard reservation).
   *
   * **Fast path when nothing is pinned** (the norm — only the SliceCache pins,
   * and only during active prefetch): the LRU victim is the Map head in O(1);
   * this keeps L0/L1 chunk eviction — which never pins and can hold tens of
   * thousands of entries — cheap. Only when pins exist do we pay the O(n) scan
   * to skip them (SliceCache, n at most a few hundred multi-MB slices). The MRU
   * branch is O(n) regardless — a Map has no reverse iterator — matching the
   * pre-pin behavior.
   */
  private selectVictim(evictMostRecent: boolean): string | undefined {
    if (this.pinned.size === 0) {
      if (!evictMostRecent) return this.cache.keys().next().value; // O(1) Map head
      let last: string | undefined;
      for (const k of this.cache.keys()) last = k; // Map tail (no reverse iterator)
      return last;
    }
    // Pins present: skip them (best-effort), falling back to the plain
    // oldest/newest only if EVERY entry is pinned.
    let first: string | undefined;
    let last: string | undefined;
    let firstUnpinned: string | undefined;
    let lastUnpinned: string | undefined;
    for (const k of this.cache.keys()) {
      if (first === undefined) first = k;
      last = k;
      if (!this.pinned.has(k)) {
        if (firstUnpinned === undefined) firstUnpinned = k;
        lastUnpinned = k;
      }
    }
    return evictMostRecent ? (lastUnpinned ?? last) : (firstUnpinned ?? first);
  }

  /** Protect `key` from routine eviction until {@link unpin} (best-effort). */
  pin(key: string): void {
    if (this.cache.has(key)) this.pinned.add(key);
  }

  /** Release a pin so `key` is eligible for eviction again. */
  unpin(key: string): void {
    this.pinned.delete(key);
  }

  /**
   * Read a value WITHOUT LRU promotion and WITHOUT touching the hit/miss
   * counters. For bookkeeping reads (e.g. "is the stored entry longer than
   * what I'm about to store?") that must not perturb eviction order or the
   * hit-rate statistic the monitor reports.
   */
  peek(key: string): V | undefined {
    return this.cache.get(key);
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }

  /**
   * Swap the value stored under an EXISTING key in place: its recency
   * position, pin, and the hit/miss/eviction counters are all untouched, and
   * NOTHING is evicted. Returns false (storing nothing) when `key` is absent
   * or when the new size would push the cache over budget — the caller makes
   * room first if it wants the swap to land. For callers that grow or shrink
   * an entry without it counting as a fresh use (see `SliceCache.setStage`).
   */
  replace(key: string, value: V): boolean {
    const old = this.cache.get(key);
    if (old === undefined) return false;
    const next = this.currentSize - this.getSize(old) + this.getSize(value);
    if (next > this.maxSize) return false;
    // Map.set on an existing key keeps its insertion (recency) position.
    this.cache.set(key, value);
    this.currentSize = next;
    return true;
  }

  /** Iterate `[key, value]` pairs from least to most recently used (no promotion). */
  entries(): IterableIterator<[string, V]> {
    return this.cache.entries();
  }

  delete(key: string): boolean {
    if (!this.cache.has(key)) {
      return false;
    }
    const value = this.cache.get(key)!;
    this.onEvict?.(key, value);
    this.currentSize -= this.getSize(value);
    this.pinned.delete(key);
    return this.cache.delete(key);
  }

  clear(): void {
    if (this.onEvict) {
      for (const [key, value] of this.cache) {
        this.onEvict(key, value);
      }
    }
    this.cache.clear();
    this.pinned.clear();
    this.currentSize = 0;
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  get size(): number {
    return this.currentSize;
  }

  get count(): number {
    return this.cache.size;
  }

  get hitCount(): number {
    return this.hits;
  }

  get missCount(): number {
    return this.misses;
  }

  get evictionCount(): number {
    return this.evictions;
  }
}
