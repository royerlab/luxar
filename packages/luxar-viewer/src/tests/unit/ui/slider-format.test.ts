/**
 * Unit tests for the GUI value-formatting helpers.
 */

import { describe, it, expect } from 'vitest';
import { decimalsForStep, formatSliderValue } from '../../../ui/slider-kit';
import { clamp } from '../../../utils/clamp';

describe('clamp', () => {
  it('returns value unchanged when within range', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(0, 0, 10)).toBe(0);
    expect(clamp(10, 0, 10)).toBe(10);
  });

  it('clamps to min when value is below', () => {
    expect(clamp(-5, 0, 10)).toBe(0);
    expect(clamp(-100, 0, 10)).toBe(0);
  });

  it('clamps to max when value is above', () => {
    expect(clamp(15, 0, 10)).toBe(10);
    expect(clamp(1000, 0, 10)).toBe(10);
  });

  it('only applies min when max is undefined', () => {
    expect(clamp(-5, 0)).toBe(0);
    expect(clamp(5, 0)).toBe(5);
    expect(clamp(1e9, 0)).toBe(1e9); // unbounded above
  });

  it('only applies max when min is undefined', () => {
    expect(clamp(15, undefined, 10)).toBe(10);
    expect(clamp(5, undefined, 10)).toBe(5);
    expect(clamp(-1e9, undefined, 10)).toBe(-1e9); // unbounded below
  });

  it('returns the value unchanged when both bounds are undefined', () => {
    expect(clamp(42)).toBe(42);
    expect(clamp(-42)).toBe(-42);
  });

  it('handles negative bounds correctly', () => {
    expect(clamp(-5, -10, -1)).toBe(-5);
    expect(clamp(-15, -10, -1)).toBe(-10);
    expect(clamp(0, -10, -1)).toBe(-1);
  });
});

describe('slider formatting', () => {
  it('derives decimal precision from fixed and exponent-form steps', () => {
    expect(decimalsForStep(1)).toBe(0);
    expect(decimalsForStep(0.1)).toBe(1);
    expect(decimalsForStep(0.001)).toBe(3);
    expect(decimalsForStep(1e-5)).toBe(5);
  });

  it('returns zero precision for invalid steps', () => {
    expect(decimalsForStep(0)).toBe(0);
    expect(decimalsForStep(-0.1)).toBe(0);
    expect(decimalsForStep(Number.NaN)).toBe(0);
  });

  it('widens off-grid slider readouts without changing on-grid precision', () => {
    expect(formatSliderValue(1.23, 0.01, 0, 0)).toBe('1.23');
    expect(formatSliderValue(1.2345, 0.01, 0, 0)).toBe('1.2345');
    expect(formatSliderValue(-3.1415, 0.01, -5, 0)).toBe('-3.1415');
  });
});
