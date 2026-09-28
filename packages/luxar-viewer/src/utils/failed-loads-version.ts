/**
 * Shared version for loader failures, lazy LOD failures, and archive faults.
 * New scenes must not restart at a version a still-mounted UI already applied.
 */
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
