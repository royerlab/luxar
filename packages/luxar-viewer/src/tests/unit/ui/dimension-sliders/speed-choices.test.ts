/**
 * The Speed section's rate choices (`ui/dimension-sliders/speed-choices.ts`):
 * period labels, the two views' chips, and how a playing rate is matched.
 */

import { describe, it, expect } from 'vitest';
import { formatPeriod, speedChoices } from '../../../../ui/dimension-sliders/speed-choices';
import { config } from '../../../../config';

describe('formatPeriod', () => {
  it('reads whole minutes in minutes and anything else in seconds', () => {
    expect(formatPeriod(20)).toBe('20s');
    expect(formatPeriod(60)).toBe('1m');
    expect(formatPeriod(90)).toBe('90s');
    expect(formatPeriod(600)).toBe('10m');
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
