/**
 * Unit tests for the tone-mapping handler helpers.
 *
 * Mode-map conversions are pure; the apply/read helpers are exercised
 * against a tiny stub of {@link LuxarToneMappingEffect} (just the four
 * mutable properties — `mode`, `exposure`, `globalOffset`, `globalGamma`).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { ToneMappingMode } from 'postprocessing';
import {
  threeToPmndrsToneMapping,
  pmndrsToThreeToneMapping,
  applyToneMapping,
  readToneMapping,
  applyExposure,
  applyGlobalOffset,
  applyGlobalGamma,
} from '../../../../rendering/post-processing/tone-mapping-handler';

interface ToneMappingEffectStub {
  mode: ToneMappingMode;
  exposure: number;
  globalOffset: number;
  globalGamma: number;
}

function makeEffectStub(): ToneMappingEffectStub {
  return {
    mode: ToneMappingMode.ACES_FILMIC,
    exposure: 0,
    globalOffset: 0,
    globalGamma: 1,
  };
}

describe('threeToPmndrsToneMapping', () => {
  it('maps each known THREE constant to its pmndrs equivalent', () => {
    expect(threeToPmndrsToneMapping(THREE.NoToneMapping)).toBe(ToneMappingMode.LINEAR);
    expect(threeToPmndrsToneMapping(THREE.LinearToneMapping)).toBe(ToneMappingMode.LINEAR);
    expect(threeToPmndrsToneMapping(THREE.ReinhardToneMapping)).toBe(ToneMappingMode.REINHARD);
    expect(threeToPmndrsToneMapping(THREE.CineonToneMapping)).toBe(
      ToneMappingMode.OPTIMIZED_CINEON
    );
    expect(threeToPmndrsToneMapping(THREE.ACESFilmicToneMapping)).toBe(ToneMappingMode.ACES_FILMIC);
    expect(threeToPmndrsToneMapping(THREE.AgXToneMapping)).toBe(ToneMappingMode.AGX);
    expect(threeToPmndrsToneMapping(THREE.NeutralToneMapping)).toBe(ToneMappingMode.NEUTRAL);
  });

  it('falls back to ACES_FILMIC for unknown values', () => {
    expect(threeToPmndrsToneMapping(-1 as unknown as THREE.ToneMapping)).toBe(
      ToneMappingMode.ACES_FILMIC
    );
    expect(threeToPmndrsToneMapping(9999 as unknown as THREE.ToneMapping)).toBe(
      ToneMappingMode.ACES_FILMIC
    );
  });
});

describe('pmndrsToThreeToneMapping', () => {
  it('maps each pmndrs mode to the closest THREE constant', () => {
    expect(pmndrsToThreeToneMapping(ToneMappingMode.LINEAR)).toBe(THREE.LinearToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.REINHARD)).toBe(THREE.ReinhardToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.REINHARD2)).toBe(THREE.ReinhardToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.REINHARD2_ADAPTIVE)).toBe(
      THREE.ReinhardToneMapping
    );
    expect(pmndrsToThreeToneMapping(ToneMappingMode.UNCHARTED2)).toBe(THREE.CineonToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.OPTIMIZED_CINEON)).toBe(
      THREE.CineonToneMapping
    );
    expect(pmndrsToThreeToneMapping(ToneMappingMode.CINEON)).toBe(THREE.CineonToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.ACES_FILMIC)).toBe(THREE.ACESFilmicToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.AGX)).toBe(THREE.AgXToneMapping);
    expect(pmndrsToThreeToneMapping(ToneMappingMode.NEUTRAL)).toBe(THREE.NeutralToneMapping);
  });

  it('falls back to ACESFilmicToneMapping for unknown values', () => {
    expect(pmndrsToThreeToneMapping(-1 as unknown as ToneMappingMode)).toBe(
      THREE.ACESFilmicToneMapping
    );
  });
});

describe('round-trip conversions', () => {
  it('THREE → pmndrs → THREE preserves the value for canonical THREE constants', () => {
    const canonical: THREE.ToneMapping[] = [
      THREE.LinearToneMapping,
      THREE.ReinhardToneMapping,
      THREE.CineonToneMapping,
      THREE.ACESFilmicToneMapping,
      THREE.AgXToneMapping,
      THREE.NeutralToneMapping,
    ];
    for (const mode of canonical) {
      expect(pmndrsToThreeToneMapping(threeToPmndrsToneMapping(mode))).toBe(mode);
    }
  });
});

describe('applyToneMapping', () => {
  it('writes the mapped pmndrs mode to the effect', () => {
    const effect = makeEffectStub();
    applyToneMapping(effect, THREE.ReinhardToneMapping);
    expect(effect.mode).toBe(ToneMappingMode.REINHARD);
    applyToneMapping(effect, THREE.AgXToneMapping);
    expect(effect.mode).toBe(ToneMappingMode.AGX);
  });

  it('is a no-op when the effect is null or undefined', () => {
    expect(() => applyToneMapping(null, THREE.ReinhardToneMapping)).not.toThrow();
    expect(() => applyToneMapping(undefined, THREE.ReinhardToneMapping)).not.toThrow();
  });

  it('falls back to ACES_FILMIC for an unknown THREE constant', () => {
    const effect = makeEffectStub();
    effect.mode = ToneMappingMode.LINEAR;
    applyToneMapping(effect, -1 as unknown as THREE.ToneMapping);
    expect(effect.mode).toBe(ToneMappingMode.ACES_FILMIC);
  });
});

describe('readToneMapping', () => {
  it('returns the THREE equivalent of the effect mode', () => {
    const effect = makeEffectStub();
    effect.mode = ToneMappingMode.AGX;
    expect(readToneMapping(effect)).toBe(THREE.AgXToneMapping);
    effect.mode = ToneMappingMode.NEUTRAL;
    expect(readToneMapping(effect)).toBe(THREE.NeutralToneMapping);
  });

  it('returns ACESFilmicToneMapping when the effect is null or undefined', () => {
    expect(readToneMapping(null)).toBe(THREE.ACESFilmicToneMapping);
    expect(readToneMapping(undefined)).toBe(THREE.ACESFilmicToneMapping);
  });
});

describe('apply{Exposure,GlobalOffset,GlobalGamma}', () => {
  it('writes the value to the corresponding effect property', () => {
    const effect = makeEffectStub();
    applyExposure(effect, 1.5);
    expect(effect.exposure).toBe(1.5);
    applyGlobalOffset(effect, -0.25);
    expect(effect.globalOffset).toBe(-0.25);
    applyGlobalGamma(effect, 2.2);
    expect(effect.globalGamma).toBe(2.2);
  });

  it('all three are no-ops when the effect is absent', () => {
    expect(() => applyExposure(null, 1)).not.toThrow();
    expect(() => applyGlobalOffset(undefined, 1)).not.toThrow();
    expect(() => applyGlobalGamma(null, 1)).not.toThrow();
  });
});
