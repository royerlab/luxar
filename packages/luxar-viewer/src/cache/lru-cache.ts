/**
 * LRU Cache using Map for O(1) get/set/delete operations.
 * Map maintains insertion order, enabling efficient LRU tracking.
 */
export class LRUCache<V> {
  private cache = new Map<string, V>();
  private maxSize: number;
  private currentSize = 0;
  private getSize: (v: V) => number;

  // Hit/miss tracking for monitoring
  private hits = 0;
  private misses = 0;
  private evictions = 0;

  constructor(maxSize: number, getSize: (v: V) => number) {
    this.maxSize = maxSize;
    this.getSize = getSize;
  }

  get(key: string): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      this.hits++;
      // Move to end (most recently used) - O(1) with Map
      this.cache.delete(key);
      this.cache.set(key, value);
    } else {
      this.misses++;
    }
    return value;
  }

  set(key: string, value: V): void {
    // If exists, remove old
    if (this.cache.has(key)) {
      this.currentSize -= this.getSize(this.cache.get(key)!);
      this.cache.delete(key);
    }

    const size = this.getSize(value);

    // Evict LRU until space available
    while (this.currentSize + size > this.maxSize && this.cache.size > 0) {
      const oldestKey = this.cache.keys().next().value;
      if (!oldestKey) break; // Safety check (should never happen)
      this.currentSize -= this.getSize(this.cache.get(oldestKey)!);
      this.cache.delete(oldestKey);
      this.evictions++;
    }

    this.cache.set(key, value);
    this.currentSize += size;
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }

  delete(key: string): boolean {
    const value = this.cache.get(key);
    if (value) {
      this.currentSize -= this.getSize(value);
      return this.cache.delete(key);
    }
    return false;
  }

  clear(): void {
    this.cache.clear();
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
