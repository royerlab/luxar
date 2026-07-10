/**
 * Persisted global viewer preferences (the Settings rail popover's model).
 *
 * One localStorage document (`StorageKeys.settings` = `luxar.settings`),
 * global across scenes — unlike the per-scene rendering settings
 * (`luxar.rendering.<sceneId>`). Applied in TWO ways, matching how each
 * consumer reads its value:
 *
 *   - LIVE values (read from `config` at use time — FOV wheel sensitivity per
 *     wheel event, idle timeout at timer re-arm, worker toggles per load):
 *     {@link applyLiveConfigOverrides} mutates the corresponding `config`
 *     fields. The config object is intentionally mutable (plain literal, not
 *     frozen); this module is the single choke point for such mutations.
 *
 *   - STARTUP values (captured once — cache tier gates and pool budget,
 *     renderer backend, worker pool size): `core/bootstrap.ts` threads them
 *     into `LuxarAppOptions` exactly like URL parameters, with the precedence
 *     **URL param > stored setting > built-in default**. Changing one of these
 *     from the Settings UI therefore requires a reload; {@link reloadRequired}
 *     tells the UI when to surface that.
 *
 * Only `bootstrapStandalone` consumes this module at startup — library
 * embedders configure the viewer through `LuxarAppOptions` directly and are
 * unaffected by a host page's stored preferences.
 */

import { config } from './index';
import { StorageKeys } from '../utils/storage-keys';
import { log, Modules } from '../utils/log';

/** Current schema version — bump on breaking shape changes (old versions → defaults). */
const SETTINGS_VERSION = 1 as const;

export interface UserSettings {
  version: typeof SETTINGS_VERSION;
  input: {
    /** Ctrl/⌘+wheel FOV change per wheel-delta unit (live). */
    fovSensitivity: number;
  };
  performance: {
    /** Power-save pause after this many ms of idle (live at timer re-arm). */
    idleTimeoutMs: number;
    /** Route heavy decode/projection through web workers (live per load). */
    useWebWorkers: boolean;
    /** Worker pool size; 0 = auto (cores-based). Applied at pool creation → reload. */
    workerCount: number;
    /** Concurrent prefetch fetches. Applied at scene-loader construction → reload. */
    networkMaxConcurrent: number;
  };
  caching: {
    /** Master in-memory/disk cache switch (→ loaderConfig.noCache) → reload. */
    enabled: boolean;
    /** Decoded-slice S-cache (→ loaderConfig.noSliceCache) → reload. */
    sliceCache: boolean;
    /** Adjacent-chunk prefetch (→ loaderConfig.noPrefetch) → reload. */
    prefetch: boolean;
    /** 'auto' = heap-aware/device-class budgets; 'custom' = explicit pool. */
    budgetMode: 'auto' | 'custom';
    /** Total cache pool (L0+L1+S-cache) in MB when budgetMode='custom'. */
    budgetMB: number;
  };
  advanced: {
    /** Render backend; 'auto' = the built-in default chain → reload. */
    renderer: 'auto' | 'webgl' | 'webgpu';
  };
}

/**
 * UI/sanitization ranges — shared by the Settings popover sliders. Frozen at
 * runtime (like `StorageKeys`): `as const` only gives compile-time readonly
 * tags, and a mutated range would silently change what sanitization admits.
 */
export const USER_SETTINGS_RANGES = Object.freeze({
  fovSensitivity: Object.freeze({ min: 0.01, max: 0.2 }),
  idleTimeoutMs: Object.freeze({ min: 500, max: 10_000 }),
  workerCount: Object.freeze({ min: 0, max: 16 }),
  networkMaxConcurrent: Object.freeze({ min: 1, max: 12 }),
  budgetMB: Object.freeze({ min: 128, max: 4096 }),
} as const);

/**
 * Built-in values of the live-read config fields, captured ONCE at module
 * load — i.e. before {@link applyLiveConfigOverrides} can mutate them.
 * `defaultUserSettings()` must read from this snapshot, not from `config`:
 * reading `config` at call time would return the user's own applied values
 * as "defaults", so Reset All could never restore the built-ins and
 * sanitize fallbacks would drift toward whatever was last applied.
 * `Object.freeze` locks the snapshot at runtime too — a write into it would
 * reintroduce exactly the drift this baseline exists to prevent.
 */
const BUILTIN_LIVE_DEFAULTS = Object.freeze({
  fovSensitivity: config.camera.fovSensitivity,
  idleTimeoutMs: config.animation.idleTimeoutMs,
  useWebWorkers: config.dataLoading.performance.useWebWorkers,
  workerCount: config.dataLoading.performance.workerCount,
  networkMaxConcurrent: config.dataLoading.network.maxConcurrent,
} as const);

/** Defaults derived from the built-in config (so the two never drift). */
export function defaultUserSettings(): UserSettings {
  return {
    version: SETTINGS_VERSION,
    input: {
      fovSensitivity: BUILTIN_LIVE_DEFAULTS.fovSensitivity,
    },
    performance: {
      idleTimeoutMs: BUILTIN_LIVE_DEFAULTS.idleTimeoutMs,
      useWebWorkers: BUILTIN_LIVE_DEFAULTS.useWebWorkers,
      workerCount: BUILTIN_LIVE_DEFAULTS.workerCount,
      networkMaxConcurrent: BUILTIN_LIVE_DEFAULTS.networkMaxConcurrent,
    },
    caching: {
      enabled: true,
      sliceCache: true,
      prefetch: true,
      budgetMode: 'auto',
      budgetMB: 1024,
    },
    advanced: {
      renderer: 'auto',
    },
  };
}

const clampOrDefault = (
  value: unknown,
  fallback: number,
  range: { min: number; max: number }
): number => {
  const num = typeof value === 'number' ? value : NaN;
  if (!Number.isFinite(num)) return fallback;
  return Math.min(range.max, Math.max(range.min, num));
};

const booleanOrDefault = (value: unknown, fallback: boolean): boolean =>
  typeof value === 'boolean' ? value : fallback;

const enumOrDefault = <T extends string>(value: unknown, allowed: readonly T[], fallback: T): T =>
  typeof value === 'string' && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;

/**
 * Sanitize an untrusted parsed value into a well-formed UserSettings.
 * Every field is individually clamped/validated so a corrupt or partial
 * document degrades per-field instead of discarding everything.
 */
function sanitizeUserSettings(raw: unknown): UserSettings {
  const d = defaultUserSettings();
  if (typeof raw !== 'object' || raw === null) return d;
  const r = raw as Record<string, Record<string, unknown>>;
  if ((r as { version?: unknown }).version !== SETTINGS_VERSION) return d;
  const input = r.input ?? {};
  const perf = r.performance ?? {};
  const caching = r.caching ?? {};
  const advanced = r.advanced ?? {};
  return {
    version: SETTINGS_VERSION,
    input: {
      fovSensitivity: clampOrDefault(
        input.fovSensitivity,
        d.input.fovSensitivity,
        USER_SETTINGS_RANGES.fovSensitivity
      ),
    },
    performance: {
      idleTimeoutMs: clampOrDefault(
        perf.idleTimeoutMs,
        d.performance.idleTimeoutMs,
        USER_SETTINGS_RANGES.idleTimeoutMs
      ),
      useWebWorkers: booleanOrDefault(perf.useWebWorkers, d.performance.useWebWorkers),
      workerCount: Math.round(
        clampOrDefault(
          perf.workerCount,
          d.performance.workerCount,
          USER_SETTINGS_RANGES.workerCount
        )
      ),
      networkMaxConcurrent: Math.round(
        clampOrDefault(
          perf.networkMaxConcurrent,
          d.performance.networkMaxConcurrent,
          USER_SETTINGS_RANGES.networkMaxConcurrent
        )
      ),
    },
    caching: {
      enabled: booleanOrDefault(caching.enabled, d.caching.enabled),
      sliceCache: booleanOrDefault(caching.sliceCache, d.caching.sliceCache),
      prefetch: booleanOrDefault(caching.prefetch, d.caching.prefetch),
      budgetMode: enumOrDefault(caching.budgetMode, ['auto', 'custom'], d.caching.budgetMode),
      budgetMB: Math.round(
        clampOrDefault(caching.budgetMB, d.caching.budgetMB, USER_SETTINGS_RANGES.budgetMB)
      ),
    },
    advanced: {
      renderer: enumOrDefault(advanced.renderer, ['auto', 'webgl', 'webgpu'], d.advanced.renderer),
    },
  };
}

/** Load stored settings (quota-safe). Corrupt / missing / wrong version → defaults. */
export function loadUserSettings(): UserSettings {
  let stored: string | null = null;
  try {
    stored = localStorage.getItem(StorageKeys.settings);
  } catch {
    /* storage blocked (private mode / embedder policy) — run on defaults */
  }
  if (!stored) return defaultUserSettings();
  try {
    return sanitizeUserSettings(JSON.parse(stored));
  } catch (err) {
    log.warning(Modules.CONFIG, 'Corrupt luxar.settings — using defaults', err);
    return defaultUserSettings();
  }
}

/** Persist settings (quota-safe — a failed write only logs). */
export function saveUserSettings(settings: UserSettings): void {
  try {
    localStorage.setItem(StorageKeys.settings, JSON.stringify(settings));
  } catch (err) {
    log.warning(Modules.CONFIG, 'Failed to save luxar.settings', err);
  }
}

/**
 * Apply the LIVE-read preferences onto the mutable config object. Consumers
 * of these fields dereference `config` at use time (per wheel event, per
 * timer re-arm, per load / pool creation), so mutation is the correct and
 * immediate application mechanism. The single sanctioned config-mutation
 * site in the codebase — keep it that way.
 */
export function applyLiveConfigOverrides(settings: UserSettings): void {
  config.camera.fovSensitivity = settings.input.fovSensitivity;
  config.animation.idleTimeoutMs = settings.performance.idleTimeoutMs;
  config.dataLoading.performance.useWebWorkers = settings.performance.useWebWorkers;
  // Worker pool size is read from config at pool creation, so mutating it
  // before app init is effective; after init it needs a reload.
  config.dataLoading.performance.workerCount = settings.performance.workerCount;
  config.dataLoading.network.maxConcurrent = settings.performance.networkMaxConcurrent;
}

/**
 * Snapshot of the reload-relevant values as they were APPLIED at boot.
 * `reloadRequired` diffs the current settings against this, so the "reload
 * to apply" hint survives popover close/reopen and even setting a value
 * back to its stored-at-boot state clears the hint correctly.
 */
let bootSnapshot: UserSettings | null = null;

/** Reload-relevant projection of a settings object (order-stable for compare). */
function reloadKey(s: UserSettings): string {
  return JSON.stringify([
    s.performance.workerCount,
    s.performance.networkMaxConcurrent,
    s.caching.enabled,
    s.caching.sliceCache,
    s.caching.prefetch,
    s.caching.budgetMode,
    s.caching.budgetMode === 'custom' ? s.caching.budgetMB : null,
    s.advanced.renderer,
  ]);
}

/** True when `current` differs from the boot snapshot on any reload-only key. */
export function reloadRequired(current: UserSettings): boolean {
  if (!bootSnapshot) return false;
  return reloadKey(current) !== reloadKey(bootSnapshot);
}

/**
 * Boot entry point: load, apply live overrides, and freeze the boot
 * snapshot for {@link reloadRequired}. Called once by `bootstrapStandalone`
 * before the first config read.
 */
export function initUserSettings(): UserSettings {
  const settings = loadUserSettings();
  applyLiveConfigOverrides(settings);
  bootSnapshot = structuredClone(settings);
  return settings;
}

/** Test hook: reset the boot snapshot (module state) between tests. */
export function resetUserSettingsForTests(): void {
  bootSnapshot = null;
}
