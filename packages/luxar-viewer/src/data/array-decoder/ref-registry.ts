/**
 * Global registry for `array_ref` deduplication.
 *
 * The Python encoder writes the same Float32Array bytes under a single
 * hash and replaces every duplicate occurrence with a reference. The
 * registry resolves the reference back to a concrete buffer the first
 * time it's seen, so subsequent ranges reuse the same memory.
 *
 * Lives in its own module so consumers that only need the registry
 * (SceneLoader, loader-factory, base-types) don't pull in the full
 * 900-line ArrayDecoder body.
 */

import { log, Modules } from '../../utils/log';

/**
 * Global registry for array references (deduplication)
 */
export class ArrayRefRegistry {
  private registry = new Map<string, Float32Array>();

  /**
   * Register an array with its hash
   */
  register(hash: string, array: Float32Array): void {
    if (!this.registry.has(hash)) {
      this.registry.set(hash, array);
      log.info(Modules.ZARR_LOADER, `Registered array ref: ${hash} (${array.length} elements)`);
    }
  }

  /**
   * Get an array by hash
   */
  get(hash: string): Float32Array | undefined {
    return this.registry.get(hash);
  }

  /**
   * Check if hash exists
   */
  has(hash: string): boolean {
    return this.registry.has(hash);
  }

  /**
   * Clear all registered arrays
   */
  clear(): void {
    this.registry.clear();
  }

  /**
   * Get statistics
   */
  getStats(): { count: number; totalBytes: number } {
    let totalBytes = 0;
    for (const arr of this.registry.values()) {
      totalBytes += arr.byteLength;
    }
    return {
      count: this.registry.size,
      totalBytes,
    };
  }
}
