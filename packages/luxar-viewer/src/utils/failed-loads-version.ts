/** Shared version for the loader failures, lazy LOD failures, and archive fault. */
let version = 0;

export function bumpFailedLoadsVersion(): void {
  version++;
}

export function failedLoadsVersion(): number {
  return version;
}

/** Track every mutation, including a changed reason for an already failed path. */
export class FailedLoadsMap<K, V> extends Map<K, V> {
  override set(key: K, value: V): this {
    bumpFailedLoadsVersion();
    return super.set(key, value);
  }

  override delete(key: K): boolean {
    const removed = super.delete(key);
    if (removed) bumpFailedLoadsVersion();
    return removed;
  }

  override clear(): void {
    if (this.size > 0) bumpFailedLoadsVersion();
    super.clear();
  }
}
