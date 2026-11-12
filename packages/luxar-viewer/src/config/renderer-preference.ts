/** Renderer preference rules shared by standalone bootstrap and its early prefetch entry. */
import { getParam } from './url-param-access';
import type { UserSettings } from './user-settings';
import { StorageKeys } from '../utils/storage-keys';

/** Current persisted settings schema; old versions fall back to defaults. */
export const SETTINGS_VERSION = 1 as const;
export const DEFAULT_RENDERER_PREFERENCE = 'auto' as const;

/** Accept the same case-insensitive URL spellings in both startup paths. */
export function normalizeRendererParam(raw: string | null): 'webgl' | 'webgpu' | null {
  if (raw === null) return null;
  const value = raw.trim().toLowerCase();
  if (value === 'webgl' || value === 'webgl2') return 'webgl';
  if (value === 'webgpu') return 'webgpu';
  return null;
}

/** Invalid stored values have the same meaning as the Settings default, auto. */
export function sanitizeRendererPreference(value: unknown): UserSettings['advanced']['renderer'] {
  return value === 'webgl' || value === 'webgpu' ? value : DEFAULT_RENDERER_PREFERENCE;
}

/** Read only the one setting needed before the main viewer bundle is available. */
export function storedRendererPreference(
  storage?: Pick<Storage, 'getItem'>
): UserSettings['advanced']['renderer'] {
  try {
    const raw: unknown = JSON.parse(
      (storage ?? localStorage).getItem(StorageKeys.settings) ?? 'null'
    );
    if (typeof raw !== 'object' || raw === null) return DEFAULT_RENDERER_PREFERENCE;
    const settings = raw as { version?: unknown; advanced?: { renderer?: unknown } };
    if (settings.version !== SETTINGS_VERSION) return DEFAULT_RENDERER_PREFERENCE;
    return sanitizeRendererPreference(settings.advanced?.renderer);
  } catch {
    return DEFAULT_RENDERER_PREFERENCE;
  }
}

/** URL wins over stored Settings; auto leaves the build/default decision open. */
export function resolveRendererOverride(
  urlRenderer: 'webgl' | 'webgpu' | null,
  storedRenderer: UserSettings['advanced']['renderer']
): 'webgl' | 'webgpu' | undefined {
  return urlRenderer ?? (storedRenderer === 'auto' ? undefined : storedRenderer);
}

/** Alias-aware early equivalent of the renderer field passed through bootstrap. */
export function earlyRendererOverride(
  search: string,
  storage?: Pick<Storage, 'getItem'>
): 'webgl' | 'webgpu' | undefined {
  const urlRenderer = normalizeRendererParam(getParam(new URLSearchParams(search), 'renderer'));
  return urlRenderer ?? resolveRendererOverride(null, storedRendererPreference(storage));
}
