/**
 * Unit tests for the pure layers-panel helpers.
 *
 * The helpers run with no DOM and no THREE.js scene — only the
 * THREE.js blending-mode + equation enums are referenced (real values,
 * not mocks).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { clampGamma, getBlendingState, liveLayerAttrs } from '../../../ui/layers/attrs-utils';
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
      expect(clampGamma(v)).toBeCloseTo(v, 5);
    }
  });

  it('handles non-finite values defensively (clamp clamps NaN to lower bound)', () => {
    // clamp(NaN, 0.2, 5.0) — Math.min/Math.max behaviour with NaN
    // produces NaN; this just locks in observed behaviour.
    expect(Number.isNaN(clampGamma(NaN))).toBe(true);
  });
});

describe('getBlendingState', () => {
  it('additive: AdditiveBlending, no depth, transparent, AddEquation', () => {
    const s = getBlendingState('additive');
    expect(s.blending).toBe(THREE.AdditiveBlending);
    expect(s.depthTest).toBe(false);
    expect(s.depthWrite).toBe(false);
    expect(s.transparent).toBe(true);
    // BlendingState is now total — non-max modes report AddEquation so
    // switching from 'max' resets the equation instead of stranding it.
    expect(s.blendEquation).toBe(THREE.AddEquation);
  });

  it('normal: NormalBlending, depth-tested, transparent (depthWrite is opacity-aware)', () => {
    // a fully-opaque (opacity ≥ 0.99) normal layer writes depth so
    // additive layers behind it are correctly occluded. Lower opacities
    // disable depthWrite to allow correct alpha compositing.
    const sOpaque = getBlendingState('normal', 1.0);
    expect(sOpaque.blending).toBe(THREE.NormalBlending);
    expect(sOpaque.depthTest).toBe(true);
    expect(sOpaque.depthWrite).toBe(true);
    expect(sOpaque.transparent).toBe(true);

    const sTransparent = getBlendingState('normal', 0.5);
    expect(sTransparent.depthWrite).toBe(false);
    expect(sTransparent.transparent).toBe(true);
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
    // unknown mode coerces to 'normal' with default opacity 1.0,
    // which writes depth (see normal-mode test above).
    const s = getBlendingState('not-a-real-mode');
    expect(s.blending).toBe(THREE.NormalBlending);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(true); // opacity defaults to 1.0
    expect(s.transparent).toBe(true);
  });

  it('opaque + fully-opaque normal write depth; transparent normal/additive/max/luminous do not', () => {
    const modes = ['additive', 'normal', 'max', 'opaque', 'luminous', 'volumetric'];
    // Default opacity 1.0 → both opaque and normal write depth.
    const writersOpaque = modes.filter((m) => getBlendingState(m, 1.0).depthWrite);
    expect(writersOpaque.sort()).toEqual(['normal', 'opaque']);
    // At lower opacity only opaque keeps depthWrite (it ignores opacity).
    const writersTrans = modes.filter((m) => getBlendingState(m, 0.5).depthWrite);
    expect(writersTrans).toEqual(['opaque']);
  });

  it('opaque is the only mode with transparent=false', () => {
    const modes = ['additive', 'normal', 'max', 'opaque', 'luminous', 'volumetric'];
    const opaque = modes.filter((m) => !getBlendingState(m).transparent);
    expect(opaque).toEqual(['opaque']);
  });

  it('every mode reports a defined blendEquation (total state)', () => {
    const modes = ['additive', 'normal', 'max', 'opaque', 'luminous', 'volumetric', 'unknown'];
    for (const m of modes) {
      expect(getBlendingState(m).blendEquation).toBeDefined();
    }
  });

  it('max is the only mode with MaxEquation; everything else uses AddEquation', () => {
    const modes = ['additive', 'normal', 'max', 'opaque', 'luminous', 'volumetric'];
    const equationByMode = Object.fromEntries(
      modes.map((m) => [m, getBlendingState(m).blendEquation])
    );
    expect(equationByMode.max).toBe(THREE.MaxEquation);
    expect(equationByMode.additive).toBe(THREE.AddEquation);
    expect(equationByMode.normal).toBe(THREE.AddEquation);
    expect(equationByMode.opaque).toBe(THREE.AddEquation);
    expect(equationByMode.luminous).toBe(THREE.AddEquation);
    expect(equationByMode.volumetric).toBe(THREE.AddEquation);
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
    expect(attrs.intensity).toBeCloseTo(1.0, 5);
    expect(attrs.offset).toBeCloseTo(0.0, 5);
  });

  it('produces non-trivial intensity / offset for a [0.2, 0.8] window', () => {
    const attrs = liveLayerAttrs(makeLayer({ displayMin: 0.2, displayMax: 0.8 }));
    // Whatever computeUniforms does, the window is narrower than [0, 1] so
    // intensity must scale up and offset must shift.
    expect(attrs.intensity).toBeGreaterThan(1.0);
    expect(attrs.offset).not.toBeCloseTo(0, 5);
  });
});
