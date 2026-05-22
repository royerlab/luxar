import { ArrayRefRegistry } from '../../../array-decoder/decoder';
import { RangeLoader } from '../range-loader';

let sharedRangeLoader: RangeLoader | null = null;
let sharedRefRegistry: ArrayRefRegistry | null = null;

/**
 * Get the singleton RangeLoader. Optional `registry` parameter only takes
 * effect on the first call (subsequent calls return the existing instance).
 */
export function getSharedRangeLoader(registry?: ArrayRefRegistry): RangeLoader {
  if (!sharedRangeLoader) {
    sharedRefRegistry = registry ?? new ArrayRefRegistry();
    sharedRangeLoader = new RangeLoader(sharedRefRegistry);
  }
  return sharedRangeLoader;
}

export function getSharedRefRegistry(): ArrayRefRegistry {
  if (!sharedRefRegistry) sharedRefRegistry = new ArrayRefRegistry();
  return sharedRefRegistry;
}

/** Reset shared instances (test-only). */
export function resetSharedRangeLoader(): void {
  sharedRangeLoader = null;
  sharedRefRegistry = null;
}
