import { describe, expect, it } from 'vitest';
import { biasArmsForScene, parseLodBiasArms, summarizeSelection } from '../e2e/perf-audit-config';

describe('parseLodBiasArms', () => {
  it('preserves the existing audit scenario when the sweep is unset', () => {
    expect(parseLodBiasArms(undefined)).toEqual([{ value: null, scenarioSuffix: '', query: '' }]);
  });

  it('builds distinct URL and result-key arms for the requested sweep', () => {
    expect(parseLodBiasArms('1, 2,4,2')).toEqual([
      { value: 1, scenarioSuffix: '', query: '' },
      { value: 2, scenarioSuffix: '-lod-bias-2', query: '&lod-bias=2' },
      { value: 4, scenarioSuffix: '-lod-bias-4', query: '&lod-bias=4' },
    ]);
  });

  it.each(['0', '-1', 'NaN', 'Infinity', '1,nope'])('rejects invalid sweep %j', (raw) => {
    expect(() => parseLodBiasArms(raw)).toThrow(/positive finite numbers/);
  });
});

describe('biasArmsForScene', () => {
  const arms = parseLodBiasArms('1,2,4');

  it('crosses laddered scenes with every requested arm', () => {
    expect(biasArmsForScene(true, arms)).toEqual(arms);
  });

  it('keeps non-ladder scenes on the neutral arm', () => {
    expect(biasArmsForScene(false, parseLodBiasArms('2,4'))).toEqual([
      { value: null, scenarioSuffix: '', query: '' },
    ]);
  });
});

describe('summarizeSelection', () => {
  const lodGroups = [
    {
      name: '/lod',
      levelCount: 3,
      activeLevel: 1,
      selector: 'screen-area' as const,
      footprintStamped: true,
    },
  ];

  it('sums visible descendants of substitutive groups', () => {
    const snapshot = summarizeSelection({ lodGroups }, [
      { path: '/lod/l0', bucket: 'opaque', depthWrite: true, renderOrder: 0, elements: 12 },
      { path: '/other', bucket: 'transparent', depthWrite: false, renderOrder: 1, elements: 8 },
    ]);
    expect(snapshot).toEqual({
      visibleElements: 12,
      activeLevels: '/lod:1/2[screen-area,footprint-stamped]',
    });
  });

  it('sums the whole draw order when the scene has no substitutive group', () => {
    const snapshot = summarizeSelection({ lodGroups: [] }, [
      { path: '/points', bucket: 'opaque', depthWrite: true, renderOrder: 0, elements: 12 },
      { path: '/lines', bucket: 'opaque', depthWrite: true, renderOrder: 0, elements: 8 },
    ]);
    expect(snapshot.visibleElements).toBe(20);
  });

  it('does not count a leaf hidden by an ancestor', () => {
    const snapshot = summarizeSelection({ lodGroups }, [
      { path: '/lod/coarse', bucket: 'opaque', depthWrite: true, renderOrder: 0, elements: 600 },
    ]);
    expect(snapshot.visibleElements).toBe(600);
  });

  it('returns null metrics when the debug hook is unavailable', () => {
    expect(summarizeSelection(undefined, undefined)).toEqual({
      visibleElements: null,
      activeLevels: null,
    });
  });
});
