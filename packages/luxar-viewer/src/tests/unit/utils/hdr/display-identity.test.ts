/**
 * Unit tests for utils/hdr/display-identity.ts — the model table and the
 * measured-headroom reader (PROTOTYPE, `?hdr`).
 */
import { describe, expect, it } from 'vitest';
import {
  displayKey,
  identifyDisplay,
  measuredHeadroom,
  type DisplaySignals,
} from '../../../../utils/hdr/display-identity';

const M4_MAX = 'ANGLE (Apple, ANGLE Metal Renderer: Apple M4 Max, Unspecified Version)';
const M4 = 'ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified Version)';

function signals(over: Partial<DisplaySignals> = {}): DisplaySignals {
  return {
    platform: 'macOS',
    gpuRenderer: M4_MAX,
    width: 1512,
    height: 982,
    dpr: 2,
    label: null,
    isInternal: null,
    ...over,
  };
}

describe('identifyDisplay', () => {
  it('identifies a Pro-class MacBook panel as XDR without any permission', () => {
    expect(identifyDisplay(signals())?.name).toBe('MacBook Pro Liquid Retina XDR');
    // 16" default scaled size: shape + Pro-class chip.
    expect(identifyDisplay(signals({ width: 1728, height: 1117 }))?.headroom).toBe(3);
  });

  it('does not take an external monitor on a MacBook Pro for its XDR panel', () => {
    expect(identifyDisplay(signals({ width: 1920, height: 1080 }))).toBeNull();
    expect(identifyDisplay(signals({ isInternal: false, label: 'DELL U2723QE' }))).toBeNull();
  });

  it('uses the 14" Pro size list to tell a base-chip MacBook Pro from an Air', () => {
    expect(identifyDisplay(signals({ gpuRenderer: M4 }))?.name).toContain('XDR');
    // 13" Air default scaled size: same panel shape, no XDR.
    expect(identifyDisplay(signals({ gpuRenderer: M4, width: 1470, height: 956 }))).toBeNull();
  });

  it('survives page zoom (devicePixelRatio includes it)', () => {
    expect(identifyDisplay(signals({ dpr: 2.2 }))?.name).toContain('XDR');
    expect(displayKey(signals({ dpr: 2.2 }))).toBe(displayKey(signals()));
    // A 1x external monitor never takes the MacBook rule.
    expect(identifyDisplay(signals({ dpr: 1, width: 1536, height: 997 }))).toBeNull();
  });

  it('trusts isInternal over the panel shape when the label is known', () => {
    const internal = signals({ width: 1920, height: 1200, isInternal: true });
    expect(identifyDisplay(internal)?.name).toContain('XDR');
  });

  it('looks external monitors up by EDID model name', () => {
    expect(identifyDisplay(signals({ label: 'M28U', isInternal: false }))?.headroom).toBe(1.5);
    expect(
      identifyDisplay(signals({ label: 'Pro Display XDR', isInternal: false }))?.headroom
    ).toBe(3);
  });

  it('knows nothing off macOS / Apple silicon', () => {
    expect(identifyDisplay(signals({ platform: 'Windows' }))).toBeNull();
    expect(identifyDisplay(signals({ gpuRenderer: null }))).toBeNull();
    // Inherited object keys are not table entries.
    expect(identifyDisplay(signals({ label: 'constructor', platform: 'Windows' }))).toBeNull();
  });
});

describe('measuredHeadroom', () => {
  it('reads hdrHeadroom in stops (as measured on a MacBook Pro)', () => {
    expect(measuredHeadroom({ hdrHeadroom: 2.666576385498047 })).toBeCloseTo(6.349, 2);
  });

  it('prefers hdrHeadroom over the legacy linear field', () => {
    expect(measuredHeadroom({ hdrHeadroom: 1, highDynamicRangeHeadroom: 9 })).toBe(2);
  });

  it('falls back to the legacy linear highDynamicRangeHeadroom', () => {
    expect(measuredHeadroom({ highDynamicRangeHeadroom: 6.35 })).toBeCloseTo(6.35);
  });

  it('returns null when the browser exposes neither', () => {
    expect(measuredHeadroom({ label: 'M28U' })).toBeNull();
    expect(measuredHeadroom(null)).toBeNull();
  });
});

describe('displayKey', () => {
  it('keys by label when known, else by screen geometry', () => {
    expect(displayKey(signals({ label: 'M28U' }))).toBe('label:M28U');
    expect(displayKey(signals())).toBe('screen:1512x982');
  });
});
