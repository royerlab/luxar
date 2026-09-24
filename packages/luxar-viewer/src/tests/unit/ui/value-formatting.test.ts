/**
 * Unit tests for the GUI value-formatting helpers.
 */

import { describe, it, expect } from 'vitest';
import { formatNumber } from '../../../ui/slider-kit';
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

describe('formatNumber', () => {
  it('returns String(value) when step is undefined', () => {
    expect(formatNumber(3.14159)).toBe('3.14159');
    expect(formatNumber(42)).toBe('42');
    expect(formatNumber(0.1)).toBe('0.1');
  });

  it('uses 0 decimals for integer step', () => {
    expect(formatNumber(3.7, 1)).toBe('4');
    expect(formatNumber(2.4, 1)).toBe('2');
    expect(formatNumber(3.14, 2)).toBe('3');
  });

  it('uses 1 decimal for step=0.1', () => {
    expect(formatNumber(3.14, 0.1)).toBe('3.1');
    expect(formatNumber(0.05, 0.1)).toBe('0.1');
  });

  it('uses 2 decimals for step=0.01', () => {
    expect(formatNumber(3.14159, 0.01)).toBe('3.14');
    expect(formatNumber(0.005, 0.01)).toBe('0.01');
  });

  it('uses 3 decimals for step=0.001', () => {
    expect(formatNumber(0.12345, 0.001)).toBe('0.123');
  });

  it('handles exponent-form steps', () => {
    expect(formatNumber(0.00003, 1e-5)).toBe('0.00003');
  });

  it('formats negative values correctly', () => {
    expect(formatNumber(-3.14159, 0.01)).toBe('-3.14');
  });

  it('handles zero value', () => {
    expect(formatNumber(0, 0.01)).toBe('0.00');
    expect(formatNumber(0, 1)).toBe('0');
  });
});
