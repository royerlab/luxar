/**
 * Unit tests for the rendering-controls settings-persistence helpers.
 *
 * The helpers are pure (modulo localStorage I/O), so we can hammer
 * them with the jsdom localStorage and a couple of stub zarr configs.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  buildBaseDefaults,
  buildResetDefaults,
  saveSettingsToStorage,
  clearStoredSettings,
  loadSettingsFromStorage,
} from '../../../ui/rendering-controls/settings-persistence';
import { config } from '../../../config';
import { StorageKeys } from '../../../utils/storage-keys';

describe('settings-persistence — buildBaseDefaults', () => {
  it('matches config defaults plus the five fly-control fields', () => {
    const defaults = buildBaseDefaults();
    expect(defaults.flyMovementSpeed).toBe(config.controls.fly.movement.speed.default);
    expect(defaults.flyRotationSpeed).toBe(config.controls.fly.rotation.speed.default);
    expect(defaults.flyInertialMode).toBe(config.controls.fly.inertialMode.default);
    expect(defaults.flyDamping).toBe(config.controls.fly.movement.damping.default);
    expect(defaults.flyRotationDamping).toBe(config.controls.fly.rotation.damping.default);
    // A representative non-fly default to confirm the spread copied correctly.
    expect(defaults.toneMapping).toBe(config.renderingControls.defaults.toneMapping);
  });

  it('returns a fresh object each call (no shared mutation)', () => {
    const a = buildBaseDefaults();
    const b = buildBaseDefaults();
    expect(a).not.toBe(b);
    a.flyMovementSpeed = 999;
    expect(b.flyMovementSpeed).not.toBe(999);
  });
});

describe('settings-persistence — buildResetDefaults', () => {
  it('with no zarrViewerConfig returns the same as buildBaseDefaults', () => {
    expect(buildResetDefaults()).toEqual(buildBaseDefaults());
  });

  it('overlays viewer_config.rendering keys on top of base defaults', () => {
    // viewer_config keys are walked by extractRenderingOverrides; smoke-test
    // by passing a config with one known override and confirming it lands.
    const zarrConfig = {
      camera: { fov: 99 },
    } as const;
    const defaults = buildResetDefaults(zarrConfig);
    expect(defaults.fov).toBe(99);
    // Untouched defaults still come from config.
    expect(defaults.flyMovementSpeed).toBe(config.controls.fly.movement.speed.default);
  });
});

describe('settings-persistence — localStorage I/O', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('saveSettingsToStorage writes a string under the StorageKeys.rendering(sceneId) key', () => {
    const sceneId = 'demo_scene_42';
    saveSettingsToStorage(sceneId, buildBaseDefaults());
    const raw = localStorage.getItem(StorageKeys.rendering(sceneId));
    expect(raw).toBeTruthy();
  });

  it('saveSettingsToStorage with empty sceneId is a no-op', () => {
    saveSettingsToStorage('', buildBaseDefaults());
    expect(localStorage.length).toBe(0);
  });

  it('loadSettingsFromStorage returns { stored: false, loaded: null } when nothing saved', () => {
    expect(loadSettingsFromStorage('missing')).toEqual({ stored: false, loaded: null });
  });

  it('loadSettingsFromStorage round-trips saved settings', () => {
    const sceneId = 'demo_scene_42';
    saveSettingsToStorage(sceneId, buildBaseDefaults());
    const result = loadSettingsFromStorage(sceneId);
    expect(result.stored).toBe(true);
    expect(result.loaded).toBeTruthy();
    expect(result.loaded?.toneMapping).toBe(config.renderingControls.defaults.toneMapping);
  });

  it('loadSettingsFromStorage returns { stored: true, loaded: null } on parse failure', () => {
    const sceneId = 'corrupt';
    localStorage.setItem(StorageKeys.rendering(sceneId), 'not-json');
    const result = loadSettingsFromStorage(sceneId);
    expect(result.stored).toBe(true);
    expect(result.loaded).toBeNull();
  });

  it('clearStoredSettings removes the key', () => {
    const sceneId = 'demo';
    saveSettingsToStorage(sceneId, buildBaseDefaults());
    clearStoredSettings(sceneId);
    expect(localStorage.getItem(StorageKeys.rendering(sceneId))).toBeNull();
  });

  it('clearStoredSettings with empty sceneId is a no-op', () => {
    saveSettingsToStorage('keep', buildBaseDefaults());
    clearStoredSettings('');
    expect(localStorage.getItem(StorageKeys.rendering('keep'))).toBeTruthy();
  });

  it('save / load tolerates a setItem that throws (quota / disabled storage)', () => {
    const sceneId = 'demo';
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota exceeded');
    });
    expect(() => saveSettingsToStorage(sceneId, buildBaseDefaults())).not.toThrow();
    setItemSpy.mockRestore();
  });

  it('load tolerates a getItem that throws', () => {
    const sceneId = 'demo';
    const getItemSpy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('disabled');
    });
    const result = loadSettingsFromStorage(sceneId);
    expect(result).toEqual({ stored: false, loaded: null });
    getItemSpy.mockRestore();
  });
});
