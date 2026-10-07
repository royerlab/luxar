/**
 * Unit tests for the shared tone-mapping name ↔ THREE enum mapping.
 *
 * This module replaced five independent copies of that translation (GUI
 * dropdown, settings applier, cinematic toggle, post-processing resource
 * builder, and the two mega-shader materials' hardcoded ACES fallback),
 * so these tests guard the contract every one of them now relies on.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  TONE_MAPPING_BY_NAME,
  TONE_MAPPING_NAMES,
  luxarToneMappingMode,
  resolveToneMappingDefault,
  toneMappingFromLuxarMode,
  toneMappingFromName,
} from '../../../../rendering/post-processing/tone-mapping';
import { config } from '../../../../config';

describe('TONE_MAPPING_BY_NAME', () => {
  it('covers the seven supported tone-mapping modes', () => {
    expect(Object.keys(TONE_MAPPING_BY_NAME).sort()).toEqual([
      'ACES',
      'AgX',
      'Cineon',
      'Linear',
      'Neutral',
      'None',
      'Reinhard',
    ]);
  });

  it('maps each name to the matching THREE constant', () => {
    expect(TONE_MAPPING_BY_NAME.None).toBe(THREE.NoToneMapping);
    expect(TONE_MAPPING_BY_NAME.Linear).toBe(THREE.LinearToneMapping);
    expect(TONE_MAPPING_BY_NAME.Reinhard).toBe(THREE.ReinhardToneMapping);
    expect(TONE_MAPPING_BY_NAME.Cineon).toBe(THREE.CineonToneMapping);
    expect(TONE_MAPPING_BY_NAME.ACES).toBe(THREE.ACESFilmicToneMapping);
    expect(TONE_MAPPING_BY_NAME.AgX).toBe(THREE.AgXToneMapping);
    expect(TONE_MAPPING_BY_NAME.Neutral).toBe(THREE.NeutralToneMapping);
  });
});

describe('TONE_MAPPING_NAMES', () => {
  it('lists exactly the mapped names (the GUI dropdown cannot drift)', () => {
    expect([...TONE_MAPPING_NAMES].sort()).toEqual(Object.keys(TONE_MAPPING_BY_NAME).sort());
  });
});

describe('resolveToneMappingDefault', () => {
  it('returns a valid THREE.ToneMapping enum value', () => {
    expect(Object.values(TONE_MAPPING_BY_NAME)).toContain(resolveToneMappingDefault());
  });

  it("resolves the current config default ('ACES') to THREE.ACESFilmicToneMapping", () => {
    // The config default is pinned to 'ACES' in
    // src/config/sections/rendering-controls/data.ts. If a future commit
    // changes that default, this test fails so the change becomes deliberate.
    expect(config.renderingControls.defaults.toneMapping).toBe('ACES');
    expect(resolveToneMappingDefault()).toBe(THREE.ACESFilmicToneMapping);
  });
});

describe('toneMappingFromName', () => {
  it('resolves every supported name', () => {
    for (const name of TONE_MAPPING_NAMES) {
      expect(toneMappingFromName(name)).toBe(TONE_MAPPING_BY_NAME[name]);
    }
  });

  it('falls back to the config default for unknown names', () => {
    // Persisted settings / zarr viewer_config / URL params are untrusted:
    // a stale or misspelled name must not leave tone mapping undefined.
    expect(toneMappingFromName('Filmic')).toBe(resolveToneMappingDefault());
    expect(toneMappingFromName('')).toBe(resolveToneMappingDefault());
  });

  it('falls back for Object.prototype member names, not the inherited value', () => {
    // Untrusted input could name a prototype member ('constructor',
    // 'toString', …); the lookup must treat those as unknown and fall back,
    // never return the inherited function.
    for (const name of ['constructor', 'toString', 'hasOwnProperty', 'valueOf']) {
      expect(toneMappingFromName(name)).toBe(resolveToneMappingDefault());
    }
  });
});

describe('luxarToneMappingMode / toneMappingFromLuxarMode (the mega-shader modes)', () => {
  it('maps every named mode to a stable compact id and back', () => {
    const cases: Array<[THREE.ToneMapping, number]> = [
      [THREE.LinearToneMapping, 1],
      [THREE.ReinhardToneMapping, 2],
      [THREE.CineonToneMapping, 3],
      [THREE.ACESFilmicToneMapping, 4],
      [THREE.AgXToneMapping, 5],
      [THREE.NeutralToneMapping, 6],
    ];
    for (const [mapping, mode] of cases) {
      expect(luxarToneMappingMode(mapping)).toBe(mode);
      expect(toneMappingFromLuxarMode(mode)).toBe(mapping);
    }
  });

  it('aliases None to Linear (the shader still clamps) and anything else to Neutral', () => {
    expect(luxarToneMappingMode(THREE.NoToneMapping)).toBe(1);
    expect(luxarToneMappingMode(THREE.CustomToneMapping)).toBe(6);
    expect(toneMappingFromLuxarMode(0)).toBe(THREE.NeutralToneMapping);
    expect(toneMappingFromLuxarMode(Number.NaN)).toBe(THREE.NeutralToneMapping);
  });
});
