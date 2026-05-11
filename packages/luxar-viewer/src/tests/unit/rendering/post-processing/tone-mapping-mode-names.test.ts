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

  it('returns "Unknown" for null, undefined, and out-of-table values', () => {
    // The helper now returns 'Unknown' for every non-canonical input; the
    // distinction between "no effect" and "effect with unknown mode" is
    // resolved at the call site (e.g. status snapshot in the manager).
    expect(toneMappingModeName(null)).toBe('Unknown');
    expect(toneMappingModeName(undefined)).toBe('Unknown');
    expect(toneMappingModeName(-1 as unknown as ToneMappingMode)).toBe('Unknown');
  });
});
