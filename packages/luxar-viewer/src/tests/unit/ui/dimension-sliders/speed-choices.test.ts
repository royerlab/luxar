/**
 * The Speed section's rate choices (`ui/dimension-sliders/speed-choices.ts`):
 * period labels, the two views' chips, and how a playing rate is matched.
 */

import { describe, it, expect } from 'vitest';
import {
  customRate,
  describePeriod,
  formatPeriod,
  isPresetRate,
  periodInBestUnit,
  rateInBestUnit,
  rateUnitLabel,
  speedChoices,
} from '../../../../ui/dimension-sliders/speed-choices';
import { config } from '../../../../config';

describe('formatPeriod', () => {
  it('reads whole minutes in minutes and anything else in seconds', () => {
    expect(formatPeriod(20)).toBe('20s');
    expect(formatPeriod(60)).toBe('1m');
    expect(formatPeriod(90)).toBe('90s');
    expect(formatPeriod(600)).toBe('10m');
  });
});

describe('typed periods', () => {
  it('read in the largest whole unit, else rounded in the largest that fits', () => {
    expect(periodInBestUnit(90)).toEqual({ amount: 90, unit: 's' });
    expect(periodInBestUnit(7200)).toEqual({ amount: 2, unit: 'h' });
    expect(periodInBestUnit(5400)).toEqual({ amount: 90, unit: 'min' });
    expect(periodInBestUnit(2 * 86400)).toEqual({ amount: 2, unit: 'd' });
    expect(periodInBestUnit(5000)).toEqual({ amount: 5000, unit: 's' }); // exact, not 1.39 h
    expect(periodInBestUnit(5400.5)).toEqual({ amount: 1.5, unit: 'h' });
    expect(periodInBestUnit(0.5)).toEqual({ amount: 0.5, unit: 's' });
    expect(formatPeriod(7200)).toBe('2h');
    expect(describePeriod(3 * 86400)).toBe('1 step every 3 d');
  });

  it('a typed rate is read per step (period) or per unit time (fps)', () => {
    expect(customRate('period', 3, 'h')).toBeCloseTo(1 / 10800, 15);
    expect(customRate('period', 30, 'd')).toBeCloseTo(1 / (30 * 86400), 15);
    expect(customRate('fps', 24, 's')).toBe(24);
    expect(customRate('fps', 3, 'min')).toBeCloseTo(1 / 20, 15);
    expect(customRate('fps', 1, 'd')).toBeCloseTo(1 / 86400, 15);
  });

  it('accepts only rates the viewer plays: one step every 30 days to 120 a second', () => {
    expect(customRate('period', 31, 'd')).toBeNull();
    expect(customRate('period', 0.001, 's')).toBeNull();
    expect(customRate('fps', 121, 's')).toBeNull();
    expect(customRate('fps', 0.5, 'd')).toBeCloseTo(1 / (2 * 86400), 15); // every 2 days
    expect(customRate('fps', 0.01, 'd')).toBeNull(); // every 100 days
    for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(customRate('period', bad, 's')).toBeNull();
      expect(customRate('fps', bad, 's')).toBeNull();
    }
  });

  it('a rate seeds the field in the unit it reads best in, in either view', () => {
    expect(rateInBestUnit('period', 1 / 7200)).toEqual({ amount: 2, unit: 'h' });
    expect(rateInBestUnit('fps', 24)).toEqual({ amount: 24, unit: 's' });
    expect(rateInBestUnit('fps', 1 / 20)).toEqual({ amount: 3, unit: 'min' });
    expect(rateInBestUnit('fps', 1 / 86400)).toEqual({ amount: 1, unit: 'd' });
    expect(rateUnitLabel('period', 'min')).toBe('min');
    expect(rateUnitLabel('fps', 'min')).toBe('/min');
  });

  it('a preset rate is recognised per view, a typed one is not', () => {
    expect(isPresetRate('period', 1 / 20)).toBe(true);
    expect(isPresetRate('period', 1 / 3600)).toBe(false);
    expect(isPresetRate('fps', 10)).toBe(true);
    expect(isPresetRate('fps', 24)).toBe(false);
  });
});

describe('speedChoices', () => {
  it('the period view offers every period preset, each setting 1 / period', () => {
    const choices = speedChoices('period');
    const { periodSeconds } = config.dimensionAnimation.presets;
    expect(choices.map((c) => c.label)).toEqual(periodSeconds.map(formatPeriod));
    expect(choices.map((c) => c.fps)).toEqual(periodSeconds.map((s) => 1 / s));
    const twoMinutes = choices.find((c) => c.label === '2m');
    expect(twoMinutes?.tooltip).toBe('1 step every 2 min');
  });

  it('the fps view offers the fps presets, sub-1 rates as fractions', () => {
    const choices = speedChoices('fps');
    expect(choices.map((c) => c.fps)).toEqual(config.dimensionAnimation.presets.fps);
    expect(choices.find((c) => c.fps === 0.5)?.label).toBe('1/2');
  });

  it('a period chip matches its rate despite 1/p being inexact, and nothing else', () => {
    const twenty = speedChoices('period').find((c) => c.label === '20s')!;
    expect(twenty.matches(1 / 20)).toBe(true);
    expect(twenty.matches(1 / 20.000001)).toBe(true);
    expect(twenty.matches(1 / 21)).toBe(false);
    expect(twenty.matches(0)).toBe(false);
  });
});
