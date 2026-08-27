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

/** Normalize a binding chord to the registry's case-insensitive sorted form. */
export function canonicalizeBindingKey(bindingKey: string): string {
  return bindingKey.toLowerCase().split('+').sort().join('+');
}

/**
 * Decide whether a key binding is allowed in a context based on its filter config.
 *
 * Rules (matching the existing context manager behavior):
 * - If `blockedKeys` contains the canonical binding key, the binding is rejected.
 * - If `allowedKeys` contains neither the base key nor canonical binding key,
 *   the binding is rejected.
 * - Otherwise the key is allowed.
 *
 * `blockedKeys` always wins over `allowedKeys` when both are present and
 * both list the key (defense in depth).
 *
 * Bare blocked keys do not reject modified bindings on the same key. This lets
 * a context block bare ArrowUp for fly-control passthrough while still owning
 * Shift+ArrowUp. Bare allowlist entries continue to admit every modifier
 * variant so fly-control bindings such as Shift+W remain reachable.
 *
 * @param key - The base key to test (usually `event.key`).
 * @param config - The filter config.
 * @param bindingKey - Canonical modifier-aware binding key. Defaults to the
 *   normalized base key for callers that do not use modifiers.
 * @returns true if the key is allowed.
 */
export function isKeyAllowedInContext(
  key: string,
  config: KeyFilterConfig,
  bindingKey = key.toLowerCase()
): boolean {
  const normalizedKey = key.toLowerCase();
  const normalizedBindingKey = canonicalizeBindingKey(bindingKey);

  if (
    config.blockedKeys?.some(
      (blockedKey) => canonicalizeBindingKey(blockedKey) === normalizedBindingKey
    )
  ) {
    return false;
  }
  if (
    config.allowedKeys &&
    !config.allowedKeys.some((allowedKey) => {
      const normalizedAllowedKey = canonicalizeBindingKey(allowedKey);
      return (
        normalizedAllowedKey === normalizedKey || normalizedAllowedKey === normalizedBindingKey
      );
    })
  ) {
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
 * currently-active context. Passthrough handling applies this ordering to the
 * active context's declared fallback set; contexts outside that route are not
 * consulted.
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
