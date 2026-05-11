/**
 * getEffectiveAA reports which AA pass the orchestrator will
 * actually install. Used by rendering-controls UI so the user sees
 * the real AA mode instead of two flags being independently true.
 */
import { describe, it, expect } from 'vitest';
import { getEffectiveAA } from '../../../../rendering/post-processing/effect-orchestrator';

describe('getEffectiveAA', () => {
  it('SMAA wins when both flags are true', () => {
    expect(getEffectiveAA(true, true)).toBe('smaa');
  });
  it('returns fxaa when only FXAA is enabled', () => {
    expect(getEffectiveAA(false, true)).toBe('fxaa');
  });
  it('returns smaa when only SMAA is enabled', () => {
    expect(getEffectiveAA(true, false)).toBe('smaa');
  });
  it("returns 'none' when both are disabled", () => {
    expect(getEffectiveAA(false, false)).toBe('none');
  });
});
