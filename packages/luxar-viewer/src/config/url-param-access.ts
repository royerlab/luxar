/** Small alias-aware query reader shared by bootstrap and the early WebGPU entry. */
import type { UrlParamKey } from './url-params';

/**
 * Deprecated spellings still accepted for a parameter, mapped to the current
 * key. Empty today: the 2026-09 camelCase rename was a pre-release hard cut.
 * This table is the mechanism a POST-release rename uses — add
 * `'old-spelling': URL_PARAM_KEYS.newSpelling`, and `readUrlParams` (`url-params.ts`)
 * keeps honoring the old URL while warning once per page load, for the
 * deprecation window the compatibility policy sets (two releases or six
 * months, whichever is longer).
 */
export const URL_PARAM_ALIASES: Readonly<Record<string, UrlParamKey>> = {};

/**
 * Resolve the spelling under which `key` is present in `params`: the key
 * itself, else a deprecated alias of it (warning once per alias), else null.
 * `aliases` is injectable so the mechanism stays testable while the shipped
 * table is empty.
 */
function resolveParamKey(
  params: URLSearchParams,
  key: UrlParamKey,
  aliases: Readonly<Record<string, UrlParamKey>>,
  onAlias?: (alias: string, canonical: UrlParamKey) => void
): string | null {
  if (params.has(key)) return key;
  for (const [alias, canonical] of Object.entries(aliases)) {
    if (canonical !== key || !params.has(alias)) continue;
    onAlias?.(alias, canonical);
    return alias;
  }
  return null;
}

/**
 * Whether flag `key` (or a deprecated alias of it) is present in `params`.
 * The alias-aware replacement for `params.has(key)`.
 */
export function hasParam(
  params: URLSearchParams,
  key: UrlParamKey,
  aliases: Readonly<Record<string, UrlParamKey>> = URL_PARAM_ALIASES,
  onAlias?: (alias: string, canonical: UrlParamKey) => void
): boolean {
  return resolveParamKey(params, key, aliases, onAlias) !== null;
}

/**
 * The value of `key` (or of a deprecated alias of it) in `params`, or null
 * when absent. The alias-aware replacement for `params.get(key)`.
 */
export function getParam(
  params: URLSearchParams,
  key: UrlParamKey,
  aliases: Readonly<Record<string, UrlParamKey>> = URL_PARAM_ALIASES,
  onAlias?: (alias: string, canonical: UrlParamKey) => void
): string | null {
  const present = resolveParamKey(params, key, aliases, onAlias);
  return present === null ? null : params.get(present);
}
