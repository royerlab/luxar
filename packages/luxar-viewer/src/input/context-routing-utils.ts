/**
 * Pure helpers for context-aware keyboard routing.
 *
 * Extracted from input-context-manager.ts so the allow/block filter and the
 * priority-ordering of fallback contexts can be unit tested without setting
 * up a full InputContextManager instance with bindings, listeners, and a
 * keyboard event source.
 *
 * @module input/context-routing-utils
 */

/**
 * Minimum shape needed to evaluate whether a key is permitted in a context.
 * The full `ContextConfig` includes `name`, `priority`, and `passthrough`,
 * but the allow/block decision only depends on the two filter arrays.
 */
export interface KeyFilterConfig {
  allowedKeys?: string[];
  blockedKeys?: string[];
}

/**
 * Decide whether a key is allowed in a context based on its filter config.
 *
 * Rules (matching the existing context manager behavior):
 * - If `blockedKeys` is non-empty and contains `key`, the key is rejected.
 * - If `allowedKeys` is non-empty and does NOT contain `key`, the key is rejected.
 * - Otherwise the key is allowed.
 *
 * `blockedKeys` always wins over `allowedKeys` when both are present and
 * both list the key (defense in depth).
 *
 * @param key - The key to test (use `event.key`, normalized to the case the
 *   filter arrays use; the existing manager uses lowercase consistently).
 * @param config - The filter config.
 * @returns true if the key is allowed.
 */
export function isKeyAllowedInContext(key: string, config: KeyFilterConfig): boolean {
  if (config.blockedKeys && config.blockedKeys.includes(key)) {
    return false;
  }
  if (config.allowedKeys && !config.allowedKeys.includes(key)) {
    return false;
  }
  return true;
}

/**
 * Minimum shape for a context entry that can be ordered by priority.
 */
export interface PriorityConfig {
  priority?: number;
}

/**
 * Sort a map's entries by descending priority, optionally excluding the
 * currently-active context. Used by passthrough handling: when the active
 * context doesn't claim a key, the remaining contexts are tried in priority
 * order until one accepts it.
 *
 * Stable for equal priorities (preserves Map insertion order). A missing
 * `priority` is treated as 0.
 *
 * @param configs - The full set of registered contexts and their configs.
 * @param excludeContext - The currently-active context to skip, or
 *   undefined to include all entries.
 * @returns Entries sorted by priority descending.
 */
export function sortContextsByPriority<K, V extends PriorityConfig>(
  configs: Map<K, V>,
  excludeContext?: K
): Array<[K, V]> {
  return Array.from(configs.entries())
    .filter(([ctx]) => ctx !== excludeContext)
    .sort((a, b) => (b[1].priority ?? 0) - (a[1].priority ?? 0));
}
