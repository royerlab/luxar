/**
 * Unit tests for config/user-settings.ts — the persisted global viewer
 * preferences behind the Settings popover.
 *
 * Covers: defaults derivation from config, quota-safe + corrupt-input load
 * paths, per-field sanitization (clamps / enum / boolean), save/load
 * round-trip, live config-override application, and the reload-required
 * boot-snapshot diff.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  defaultUserSettings,
  loadUserSettings,
  saveUserSettings,
  applyLiveConfigOverrides,
  initUserSettings,
  reloadRequired,
  resetUserSettingsForTests,
  USER_SETTINGS_RANGES,
  type UserSettings,
} from '../../../config/user-settings';
import { config } from '../../../config';
import { StorageKeys } from '../../../utils/storage-keys';

const KEY = StorageKeys.settings;

/** Snapshot of the config fields the module mutates, restored after each test. */
const configSnapshot = {
  fovSensitivity: config.camera.fovSensitivity,
  idleTimeoutMs: config.animation.idleTimeoutMs,
  useWebWorkers: config.dataLoading.performance.useWebWorkers,
  workerCount: config.dataLoading.performance.workerCount,
  maxConcurrent: config.dataLoading.network.maxConcurrent,
};

beforeEach(() => {
  localStorage.clear();
  resetUserSettingsForTests();
});

afterEach(() => {
  config.camera.fovSensitivity = configSnapshot.fovSensitivity;
  config.animation.idleTimeoutMs = configSnapshot.idleTimeoutMs;
  config.dataLoading.performance.useWebWorkers = configSnapshot.useWebWorkers;
  config.dataLoading.performance.workerCount = configSnapshot.workerCount;
  config.dataLoading.network.maxConcurrent = configSnapshot.maxConcurrent;
  vi.restoreAllMocks();
});

describe('defaultUserSettings', () => {
  it('derives live-value defaults from the built-in config (no drift)', () => {
    const d = defaultUserSettings();
    expect(d.version).toBe(1);
    expect(d.input.fovSensitivity).toBe(config.camera.fovSensitivity);
    expect(d.performance.idleTimeoutMs).toBe(config.animation.idleTimeoutMs);
    expect(d.performance.networkMaxConcurrent).toBe(config.dataLoading.network.maxConcurrent);
    expect(d.caching).toEqual({
      enabled: true,
      sliceCache: true,
      prefetch: true,
      budgetMode: 'auto',
      budgetMB: 1024,
    });
    expect(d.advanced.renderer).toBe('auto');
  });
});

describe('loadUserSettings', () => {
  it('returns defaults when nothing is stored', () => {
    expect(loadUserSettings()).toEqual(defaultUserSettings());
  });

  it('returns defaults on corrupt JSON', () => {
    localStorage.setItem(KEY, '{not json');
    expect(loadUserSettings()).toEqual(defaultUserSettings());
  });

  it('returns defaults on a wrong schema version', () => {
    const other = { ...defaultUserSettings(), version: 99 };
    localStorage.setItem(KEY, JSON.stringify(other));
    expect(loadUserSettings()).toEqual(defaultUserSettings());
  });

  it('returns defaults when storage access throws (private mode)', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(loadUserSettings()).toEqual(defaultUserSettings());
  });

  it('clamps out-of-range numbers per field and rounds integer fields', () => {
    const stored = defaultUserSettings() as unknown as Record<string, Record<string, unknown>>;
    stored.input = { fovSensitivity: 999 };
    stored.performance = {
      idleTimeoutMs: -5,
      useWebWorkers: 'yes', // non-boolean → default
      workerCount: 3.7,
      networkMaxConcurrent: 100,
    };
    stored.caching = {
      enabled: true,
      sliceCache: true,
      prefetch: true,
      budgetMode: 'weird', // invalid enum → default
      budgetMB: NaN, // non-finite → default
    };
    stored.advanced = { renderer: 'vulkan' }; // invalid enum → default
    localStorage.setItem(KEY, JSON.stringify(stored));

    const loaded = loadUserSettings();
    const d = defaultUserSettings();
    expect(loaded.input.fovSensitivity).toBe(USER_SETTINGS_RANGES.fovSensitivity.max);
    expect(loaded.performance.idleTimeoutMs).toBe(USER_SETTINGS_RANGES.idleTimeoutMs.min);
    expect(loaded.performance.useWebWorkers).toBe(d.performance.useWebWorkers);
    expect(loaded.performance.workerCount).toBe(4); // 3.7 rounded
    expect(loaded.performance.networkMaxConcurrent).toBe(
      USER_SETTINGS_RANGES.networkMaxConcurrent.max
    );
    expect(loaded.caching.budgetMode).toBe('auto');
    expect(loaded.caching.budgetMB).toBe(d.caching.budgetMB);
    expect(loaded.advanced.renderer).toBe('auto');
  });

  it('round-trips a saved settings object', () => {
    const s = defaultUserSettings();
    s.input.fovSensitivity = 0.12;
    s.caching.budgetMode = 'custom';
    s.caching.budgetMB = 512;
    s.advanced.renderer = 'webgpu';
    saveUserSettings(s);
    expect(loadUserSettings()).toEqual(s);
  });

  it('save is quota-safe (a throwing setItem only warns)', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    expect(() => saveUserSettings(defaultUserSettings())).not.toThrow();
  });
});

describe('applyLiveConfigOverrides', () => {
  it('mutates the five live-read config paths', () => {
    const s = defaultUserSettings();
    s.input.fovSensitivity = 0.15;
    s.performance.idleTimeoutMs = 4000;
    s.performance.useWebWorkers = false;
    s.performance.workerCount = 2;
    s.performance.networkMaxConcurrent = 8;

    applyLiveConfigOverrides(s);

    expect(config.camera.fovSensitivity).toBe(0.15);
    expect(config.animation.idleTimeoutMs).toBe(4000);
    expect(config.dataLoading.performance.useWebWorkers).toBe(false);
    expect(config.dataLoading.performance.workerCount).toBe(2);
    expect(config.dataLoading.network.maxConcurrent).toBe(8);
  });

  it('defaults stay the BUILT-IN values even after live overrides mutated config (Reset All)', () => {
    // Regression: defaultUserSettings() used to read config at call time, so
    // after applying a user value the "default" became the user value and
    // Reset All could never restore the built-ins.
    const builtinFov = defaultUserSettings().input.fovSensitivity;
    const s = defaultUserSettings();
    s.input.fovSensitivity = 0.19;
    s.performance.idleTimeoutMs = 9000;
    applyLiveConfigOverrides(s);

    const d = defaultUserSettings();
    expect(d.input.fovSensitivity).toBe(builtinFov);
    expect(d.input.fovSensitivity).not.toBe(0.19);
    expect(d.performance.idleTimeoutMs).not.toBe(9000);
  });
});

describe('initUserSettings + reloadRequired', () => {
  it('is false right after boot, true after a reload-key change, false when reverted', () => {
    const s = defaultUserSettings();
    s.caching.budgetMode = 'custom';
    s.caching.budgetMB = 512;
    saveUserSettings(s);

    const boot = initUserSettings();
    expect(reloadRequired(boot)).toBe(false);

    const edited: UserSettings = structuredClone(boot);
    edited.advanced.renderer = 'webgl';
    expect(reloadRequired(edited)).toBe(true);

    edited.advanced.renderer = 'auto';
    expect(reloadRequired(edited)).toBe(false);
  });

  it('live-only changes do NOT flag a reload', () => {
    const boot = initUserSettings();
    const edited: UserSettings = structuredClone(boot);
    edited.input.fovSensitivity = 0.19;
    edited.performance.idleTimeoutMs = 9000;
    edited.performance.useWebWorkers = !edited.performance.useWebWorkers;
    expect(reloadRequired(edited)).toBe(false);
  });

  it('budgetMB only matters in custom mode', () => {
    const boot = initUserSettings(); // defaults: auto mode
    const edited: UserSettings = structuredClone(boot);
    edited.caching.budgetMB = 4096; // ignored while mode=auto
    expect(reloadRequired(edited)).toBe(false);
    edited.caching.budgetMode = 'custom';
    expect(reloadRequired(edited)).toBe(true);
  });

  it('returns false before initUserSettings has run (library embedding)', () => {
    expect(reloadRequired(defaultUserSettings())).toBe(false);
  });

  it('applies the stored live overrides at init', () => {
    const s = defaultUserSettings();
    s.input.fovSensitivity = 0.11;
    saveUserSettings(s);
    initUserSettings();
    expect(config.camera.fovSensitivity).toBe(0.11);
  });
});
