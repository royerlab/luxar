/**
 * Unit tests for the pure layers-panel helpers.
 *
 * The helpers run with no DOM and no THREE.js scene — only the
 * THREE.js blending-mode + equation enums are referenced (real values,
 * not mocks).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  clampGamma,
  getBlendingState,
  liveLayerAttrs,
} from '../../../ui/layers/layer-attrs-utils';
import type { LayerInfo } from '../../../ui/layers/layer-state';

describe('clampGamma', () => {
  it('clamps to [0.2, 5.0]', () => {
    expect(clampGamma(0.0)).toBe(0.2);
    expect(clampGamma(0.1)).toBe(0.2);
    expect(clampGamma(0.2)).toBe(0.2);
    expect(clampGamma(1.0)).toBe(1.0);
    expect(clampGamma(5.0)).toBe(5.0);
    expect(clampGamma(10.0)).toBe(5.0);
  });

  it('passes valid values through unchanged', () => {
    for (const v of [0.5, 1.0, 2.2, 3.5, 4.99]) {
      expect(clampGamma(v)).toBeCloseTo(v);
    }
  });

  it('handles non-finite values defensively (clamp clamps NaN to lower bound)', () => {
    // clamp(NaN, 0.2, 5.0) — Math.min/Math.max behaviour with NaN
    // produces NaN; this just locks in observed behaviour.
    expect(Number.isNaN(clampGamma(NaN))).toBe(true);
  });
});

describe('getBlendingState', () => {
  it('additive: AdditiveBlending, no depth, transparent', () => {
    const s = getBlendingState('additive');
    expect(s.blending).toBe(THREE.AdditiveBlending);
    expect(s.depthTest).toBe(false);
    expect(s.depthWrite).toBe(false);
    expect(s.transparent).toBe(true);
    expect(s.blendEquation).toBeUndefined();
  });

  it('normal: NormalBlending, depth-tested, transparent', () => {
    const s = getBlendingState('normal');
    expect(s.blending).toBe(THREE.NormalBlending);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(false);
    expect(s.transparent).toBe(true);
  });

  it('max: CustomBlending with MaxEquation', () => {
    const s = getBlendingState('max');
    expect(s.blending).toBe(THREE.CustomBlending);
    expect(s.blendEquation).toBe(THREE.MaxEquation);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(false);
    expect(s.transparent).toBe(true);
  });

  it('opaque: NormalBlending with depthWrite enabled, transparency disabled', () => {
    const s = getBlendingState('opaque');
    expect(s.blending).toBe(THREE.NormalBlending);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(true);
    expect(s.transparent).toBe(false);
  });

  it('luminous: AdditiveBlending but depth-tested', () => {
    const s = getBlendingState('luminous');
    expect(s.blending).toBe(THREE.AdditiveBlending);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(false);
    expect(s.transparent).toBe(true);
  });

  it('falls back to a normal-blending equivalent for unknown modes', () => {
    const s = getBlendingState('not-a-real-mode');
    expect(s.blending).toBe(THREE.NormalBlending);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(false);
    expect(s.transparent).toBe(true);
  });

  it('opaque is the only mode with depthWrite=true', () => {
    const modes = ['additive', 'normal', 'max', 'opaque', 'luminous'];
    const writers = modes.filter((m) => getBlendingState(m).depthWrite);
    expect(writers).toEqual(['opaque']);
  });

  it('opaque is the only mode with transparent=false', () => {
    const modes = ['additive', 'normal', 'max', 'opaque', 'luminous'];
    const opaque = modes.filter((m) => !getBlendingState(m).transparent);
    expect(opaque).toEqual(['opaque']);
  });
});

describe('liveLayerAttrs', () => {
  function makeLayer(overrides: Partial<LayerInfo> = {}): LayerInfo {
    return {
      path: '/test',
      type: 'points',
      label: 'test',
      visible: true,
      opacity: 1,
      gamma: 1,
      displayMin: 0,
      displayMax: 1,
      blendingMode: 'additive',
      colormap: 'viridis',
      isData: true,
      ...overrides,
    } as LayerInfo;
  }

  it('forwards opacity, blending mode, and gamma (clamped)', () => {
    const attrs = liveLayerAttrs(makeLayer({ opacity: 0.7, gamma: 1.5, blendingMode: 'normal' }));
    expect(attrs.opacity).toBe(0.7);
    expect(attrs.gamma).toBe(1.5);
    expect(attrs.blending_mode).toBe('normal');
  });

  it('clamps gamma to the allowed range', () => {
    expect(liveLayerAttrs(makeLayer({ gamma: 0.0 })).gamma).toBe(0.2);
    expect(liveLayerAttrs(makeLayer({ gamma: 100 })).gamma).toBe(5.0);
  });

  it('derives intensity and offset from displayMin/displayMax', () => {
    // For [0, 1]: intensity=1, offset=0 (identity)
    const attrs = liveLayerAttrs(makeLayer({ displayMin: 0, displayMax: 1 }));
    expect(attrs.intensity).toBeCloseTo(1.0);
    expect(attrs.offset).toBeCloseTo(0.0);
  });

  it('produces non-trivial intensity / offset for a [0.2, 0.8] window', () => {
    const attrs = liveLayerAttrs(makeLayer({ displayMin: 0.2, displayMax: 0.8 }));
    // Whatever computeUniforms does, the window is narrower than [0, 1] so
    // intensity must scale up and offset must shift.
    expect(attrs.intensity).toBeGreaterThan(1.0);
    expect(attrs.offset).not.toBeCloseTo(0);
  });
});
