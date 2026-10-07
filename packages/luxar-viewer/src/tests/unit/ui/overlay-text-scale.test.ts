// @vitest-environment jsdom
/**
 * Unit tests for ui/overlay-text-scale.ts — the session-wide multiplier on
 * overlay type: precedence (URL over scene), clamping, the custom property on
 * the container, and the font-size expression text overlays use.
 */

import { describe, it, expect } from 'vitest';
import {
  TEXT_SCALE_PROPERTY,
  applyTextScale,
  resolveTextScale,
  scaledFontSize,
  textScaleOf,
} from '../../../ui/overlay-text-scale';

describe('resolveTextScale', () => {
  it('prefers the URL value over the authored one', () => {
    expect(resolveTextScale(0.7, 0.9)).toBe(0.7);
  });

  it('falls back to the authored value, then to 1', () => {
    expect(resolveTextScale(null, 0.8)).toBe(0.8);
    expect(resolveTextScale(undefined, undefined)).toBe(1);
  });

  it('ignores a non-number or non-positive authored value', () => {
    expect(resolveTextScale(null, '0.8')).toBe(1);
    expect(resolveTextScale(null, 0)).toBe(1);
    expect(resolveTextScale(null, Number.NaN)).toBe(1);
  });

  it('clamps to [0.25, 4]', () => {
    expect(resolveTextScale(0.01, null)).toBe(0.25);
    expect(resolveTextScale(10, null)).toBe(4);
  });
});

describe('applyTextScale', () => {
  it('writes the custom property, and clears it again at 1', () => {
    const container = document.createElement('div');
    applyTextScale(0.8, container);
    expect(container.style.getPropertyValue(TEXT_SCALE_PROPERTY)).toBe('0.8');
    expect(textScaleOf(container)).toBe(0.8);
    applyTextScale(1, container);
    expect(container.style.getPropertyValue(TEXT_SCALE_PROPERTY)).toBe('');
  });

  it('reads 1 from an element that inherits no scale', () => {
    expect(textScaleOf(document.createElement('div'))).toBe(1);
  });
});

describe('scaledFontSize', () => {
  it('multiplies the viewport-height size by the property, defaulting to 1', () => {
    expect(scaledFontSize(0.02)).toBe('calc(2vh * var(--luxar-text-scale, 1))');
  });
});
