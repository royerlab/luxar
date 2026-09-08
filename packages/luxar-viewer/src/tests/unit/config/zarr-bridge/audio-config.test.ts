import { describe, it, expect } from 'vitest';
import { extractAudioConfig } from '../../../../config/zarr-bridge/audio-config';

describe('extractAudioConfig', () => {
  it('returns {} for absent or malformed blocks', () => {
    expect(extractAudioConfig(undefined)).toEqual({});
    expect(extractAudioConfig(null)).toEqual({});
    expect(extractAudioConfig('loud')).toEqual({});
  });

  it('passes valid fields through camelCased', () => {
    expect(
      extractAudioConfig({
        enabled: true,
        master_gain: 0.8,
        panning_model: 'HRTF',
        buses: { ambient: 0.6, voice: 1.0, effects: 0.8 },
        duck_db: -9,
      })
    ).toEqual({
      enabled: true,
      masterGain: 0.8,
      panningModel: 'HRTF',
      buses: { ambient: 0.6, voice: 1.0, effects: 0.8 },
      duckDb: -9,
    });
  });

  it('clamps gains to [0, 2] and duck_db to [-60, 0]', () => {
    expect(extractAudioConfig({ master_gain: 7, duck_db: 5, buses: { voice: -1 } })).toEqual({
      masterGain: 2,
      duckDb: 0,
      buses: { voice: 0 },
    });
    expect(extractAudioConfig({ duck_db: -200 })).toEqual({ duckDb: -60 });
  });

  it('drops a bad panning model, unknown buses and non-numbers', () => {
    expect(
      extractAudioConfig({
        panning_model: 'binaural',
        buses: { music: 0.5, ambient: 'loud' },
        master_gain: 'x',
        duck_db: null,
        enabled: 'yes',
      })
    ).toEqual({});
  });
});
