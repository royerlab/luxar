/**
 * Unit tests for the blending-state module.
 *
 * Covers discriminator predicates, canonical state assertions for each
 * mode (additive/normal/max/opaque/luminous), idempotency, max-mode
 * round-trip, and opacity boundary tests.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  isAdditiveMode,
  isLuminousMode,
  isMaxMode,
  isNormalMode,
  isOpaqueMode,
  getCompleteBlendingState,
  getGSplatNormalBlendingState,
} from '../../../rendering/blending-state';
import type { BlendingMode } from '../../../rendering/material-manager';

describe('BlendingMode predicates', () => {
  const all: BlendingMode[] = ['additive', 'normal', 'max', 'opaque', 'luminous'];

  it('isAdditiveMode is true only for additive', () => {
    for (const mode of all) {
      expect(isAdditiveMode(mode)).toBe(mode === 'additive');
    }
  });

  it('isNormalMode is true only for normal', () => {
    for (const mode of all) {
      expect(isNormalMode(mode)).toBe(mode === 'normal');
    }
  });

  it('isMaxMode is true only for max', () => {
    for (const mode of all) {
      expect(isMaxMode(mode)).toBe(mode === 'max');
    }
  });

  it('isOpaqueMode is true only for opaque', () => {
    for (const mode of all) {
      expect(isOpaqueMode(mode)).toBe(mode === 'opaque');
    }
  });

  it('isLuminousMode is true only for luminous', () => {
    for (const mode of all) {
      expect(isLuminousMode(mode)).toBe(mode === 'luminous');
    }
  });

  it('exactly one predicate fires per mode', () => {
    for (const mode of all) {
      const hits = [
        isAdditiveMode(mode),
        isNormalMode(mode),
        isMaxMode(mode),
        isOpaqueMode(mode),
        isLuminousMode(mode),
      ].filter(Boolean).length;
      expect(hits).toBe(1);
    }
  });
});

describe('H.1 — getCompleteBlendingState canonical state per mode', () => {
  it('additive: AdditiveBlending, depthTest=false, alpha-weighted', () => {
    const state = getCompleteBlendingState('additive');
    expect(state.blending).toBe(THREE.AdditiveBlending);
    expect(state.depthTest).toBe(false);
    expect(state.depthWrite).toBe(false);
    expect(state.transparent).toBe(true);
    expect(state.shaderOutputMode).toBe('alpha-weighted');
  });

  it('luminous: AdditiveBlending with depthTest=true', () => {
    const state = getCompleteBlendingState('luminous');
    expect(state.blending).toBe(THREE.AdditiveBlending);
    expect(state.depthTest).toBe(true);
    expect(state.depthWrite).toBe(false);
    expect(state.shaderOutputMode).toBe('alpha-weighted');
  });

  it('max: CustomBlending + MaxEquation + OneFactor + rgb-contribution', () => {
    const state = getCompleteBlendingState('max');
    expect(state.blending).toBe(THREE.CustomBlending);
    expect(state.blendEquation).toBe(THREE.MaxEquation);
    expect(state.blendSrc).toBe(THREE.OneFactor);
    expect(state.blendDst).toBe(THREE.OneFactor);
    expect(state.depthWrite).toBe(false);
    expect(state.shaderOutputMode).toBe('rgb-contribution');
  });

  it('opaque: NormalBlending + depthTest + depthWrite', () => {
    const state = getCompleteBlendingState('opaque');
    expect(state.blending).toBe(THREE.NormalBlending);
    expect(state.depthTest).toBe(true);
    expect(state.depthWrite).toBe(true);
    expect(state.transparent).toBe(false);
    expect(state.shaderOutputMode).toBe('opaque');
  });

  it('normal: depthWrite gated by opacity ≥ 0.99', () => {
    const at99 = getCompleteBlendingState('normal', 0.99);
    expect(at99.depthWrite).toBe(true);

    const just_below = getCompleteBlendingState('normal', 0.989);
    expect(just_below.depthWrite).toBe(false);

    const at_one = getCompleteBlendingState('normal', 1.0);
    expect(at_one.depthWrite).toBe(true);

    const half = getCompleteBlendingState('normal', 0.5);
    expect(half.depthWrite).toBe(false);
  });

  it('max-mode round-trip: max → additive → max returns identical state', () => {
    // Regression lock-in for the LineMaterial "stranded OneFactor"
    // issue: switching out of max and back must produce the same
    // canonical state, not stranded fields.
    const first = getCompleteBlendingState('max');
    void getCompleteBlendingState('additive');
    const second = getCompleteBlendingState('max');
    expect(second).toEqual(first);
  });
});

describe('getGSplatNormalBlendingState (premultiplied alpha-over for gsplats)', () => {
  it('CustomBlending + AddEquation + One/OneMinusSrcAlpha, premultiplied-alpha output', () => {
    const s = getGSplatNormalBlendingState();
    expect(s.blending).toBe(THREE.CustomBlending);
    expect(s.blendEquation).toBe(THREE.AddEquation);
    expect(s.blendSrc).toBe(THREE.OneFactor); // shader premultiplies — SrcAlpha would double-multiply
    expect(s.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(s.shaderOutputMode).toBe('premultiplied-alpha');
  });

  it('symmetric alpha channel: no separate alpha equation/factors (WebGPU-bridge safe)', () => {
    const s = getGSplatNormalBlendingState();
    expect(s.blendEquationAlpha).toBeUndefined();
    expect(s.blendSrcAlpha).toBeUndefined();
    expect(s.blendDstAlpha).toBeUndefined();
  });

  it('transparent, depthTest on, depthWrite unconditionally OFF (no opacity gate)', () => {
    // Unlike the generic normal entry (opacity >= 0.99 flips depthWrite),
    // coverage-alpha splat fragments must never write depth — a fragment
    // with alpha ~1e-4 writing depth punches occlusion halos.
    const s = getGSplatNormalBlendingState();
    expect(s.transparent).toBe(true);
    expect(s.depthTest).toBe(true);
    expect(s.depthWrite).toBe(false);
  });

  it('differs from the generic normal state exactly where intended', () => {
    const generic = getCompleteBlendingState('normal', 1.0);
    const gsplat = getGSplatNormalBlendingState();
    // Same alpha-over intent…
    expect(gsplat.blendEquation).toBe(generic.blendEquation);
    expect(gsplat.blendDst).toBe(generic.blendDst);
    expect(gsplat.transparent).toBe(generic.transparent);
    expect(gsplat.depthTest).toBe(generic.depthTest);
    // …but premultiplied source factor and no opacity-gated depth write.
    expect(generic.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(gsplat.blendSrc).toBe(THREE.OneFactor);
    expect(generic.depthWrite).toBe(true); // opacity 1.0 gates it on
    expect(gsplat.depthWrite).toBe(false);
  });
});

// [rendering.md/G11][P5] Opacity boundary tests for the normal-mode
// `opacity >= 0.99` gate. NaN / Infinity / -0 / negative inputs are not
// guarded in the source — pinning the IEEE-754 contract makes any future
// clamp introduction deliberate.
describe('normal-mode opacity boundary inputs', () => {
  it('opacity=NaN ⇒ depthWrite=false (NaN >= 0.99 is false)', () => {
    const s = getCompleteBlendingState('normal', Number.NaN);
    expect(s.depthWrite).toBe(false);
  });

  it('opacity=+Infinity ⇒ depthWrite=true (Inf >= 0.99 is true)', () => {
    const s = getCompleteBlendingState('normal', Number.POSITIVE_INFINITY);
    expect(s.depthWrite).toBe(true);
  });

  it('opacity=-Infinity ⇒ depthWrite=false (-Inf >= 0.99 is false)', () => {
    const s = getCompleteBlendingState('normal', Number.NEGATIVE_INFINITY);
    expect(s.depthWrite).toBe(false);
  });

  it('opacity=-0 is treated the same as +0 ⇒ depthWrite=false', () => {
    // IEEE-754: -0 == 0, and 0 >= 0.99 is false. A mutant that
    // special-cased the sign would be visible here.
    const s = getCompleteBlendingState('normal', -0);
    expect(s.depthWrite).toBe(false);
  });

  it('negative opacity ⇒ depthWrite=false (preserves the >= 0.99 gate)', () => {
    const s = getCompleteBlendingState('normal', -0.5);
    expect(s.depthWrite).toBe(false);
  });
});
