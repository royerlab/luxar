/**
 * Unit tests for RobustVignetteEffect.
 *
 * The effect is a thin pmndrs `Effect` subclass: constructor reads the
 * uniform values, get/set on `darkness` and `offset` mutate them.
 * Tests don't need a renderer.
 */

import { describe, it, expect } from 'vitest';
import {
  RobustVignetteEffect,
  isRobustVignetteEffect,
} from '../../../../rendering/post-processing/robust-vignette-effect';
import { BlendFunction } from 'postprocessing';

describe('RobustVignetteEffect', () => {
  it('uses documented defaults when constructed with no options', () => {
    const fx = new RobustVignetteEffect();
    expect(fx.darkness).toBe(0.5);
    expect(fx.offset).toBe(0.5);
  });

  it('applies provided darkness and offset', () => {
    const fx = new RobustVignetteEffect({ darkness: 0.7, offset: 0.3 });
    expect(fx.darkness).toBe(0.7);
    expect(fx.offset).toBe(0.3);
  });

  it('respects an explicit blendFunction option', () => {
    const fx = new RobustVignetteEffect({ blendFunction: BlendFunction.MULTIPLY });
    expect(fx.blendMode.blendFunction).toBe(BlendFunction.MULTIPLY);
  });

  it('darkness setter mutates the uniform', () => {
    const fx = new RobustVignetteEffect();
    fx.darkness = 0.9;
    expect(fx.darkness).toBe(0.9);
  });

  it('offset setter mutates the uniform', () => {
    const fx = new RobustVignetteEffect();
    fx.offset = 0.1;
    expect(fx.offset).toBe(0.1);
  });

  it('accepts boundary values (0 and 1) without throwing', () => {
    expect(() => new RobustVignetteEffect({ darkness: 0, offset: 0 })).not.toThrow();
    expect(() => new RobustVignetteEffect({ darkness: 1, offset: 1 })).not.toThrow();
  });
});

describe('isRobustVignetteEffect', () => {
  it('returns true for an actual RobustVignetteEffect instance', () => {
    expect(isRobustVignetteEffect(new RobustVignetteEffect())).toBe(true);
  });

  it('returns false for plain objects', () => {
    expect(isRobustVignetteEffect({})).toBe(false);
    expect(isRobustVignetteEffect(null)).toBe(false);
    expect(isRobustVignetteEffect(undefined)).toBe(false);
    expect(isRobustVignetteEffect({ darkness: 0.5, offset: 0.5 })).toBe(false);
  });
});
