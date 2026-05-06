/**
 * Unit tests for the tone-mapping mode-name table.
 */

import { describe, it, expect } from 'vitest';
import { ToneMappingMode } from 'postprocessing';
import { toneMappingModeName } from '../../../../rendering/post-processing/tone-mapping-mode-names';

describe('toneMappingModeName', () => {
  it('returns the canonical name for each known mode', () => {
    expect(toneMappingModeName(ToneMappingMode.LINEAR)).toBe('Linear');
    expect(toneMappingModeName(ToneMappingMode.REINHARD)).toBe('Reinhard');
    expect(toneMappingModeName(ToneMappingMode.OPTIMIZED_CINEON)).toBe('Cineon');
    expect(toneMappingModeName(ToneMappingMode.ACES_FILMIC)).toBe('ACES Filmic');
    expect(toneMappingModeName(ToneMappingMode.AGX)).toBe('AgX');
    expect(toneMappingModeName(ToneMappingMode.NEUTRAL)).toBe('Neutral');
  });

  it('returns "Off" for null and undefined', () => {
    expect(toneMappingModeName(null)).toBe('Off');
    expect(toneMappingModeName(undefined)).toBe('Off');
  });

  it('returns "Unknown" for a value outside the table', () => {
    // -1 is not a defined ToneMappingMode value — defensive fallback.
    expect(toneMappingModeName(-1 as unknown as ToneMappingMode)).toBe('Unknown');
  });
});
