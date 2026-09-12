import { describe, expect, it } from 'vitest';
import { parseLodBiasArms } from '../e2e/perf-audit-config';

describe('parseLodBiasArms', () => {
  it('preserves the existing audit scenario when the sweep is unset', () => {
    expect(parseLodBiasArms(undefined)).toEqual([{ value: null, scenarioSuffix: '', query: '' }]);
  });

  it('builds distinct URL and result-key arms for the requested sweep', () => {
    expect(parseLodBiasArms('1, 2,4,2')).toEqual([
      { value: 1, scenarioSuffix: '-lod-bias-1', query: '&lod-bias=1' },
      { value: 2, scenarioSuffix: '-lod-bias-2', query: '&lod-bias=2' },
      { value: 4, scenarioSuffix: '-lod-bias-4', query: '&lod-bias=4' },
    ]);
  });

  it.each(['0', '-1', 'NaN', 'Infinity', '1,nope'])('rejects invalid sweep %j', (raw) => {
    expect(() => parseLodBiasArms(raw)).toThrow(/positive finite numbers/);
  });
});
