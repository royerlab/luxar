/**
 * Unit tests for the blending-state module.
 *
 * Covers discriminator predicates, canonical state assertions for each
 * mode (additive/normal/max/opaque/luminous/volumetric), idempotency,
 * max-mode round-trip, and opacity boundary tests.
 */
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  isAdditiveMode,
  isLuminousMode,
  isMaxMode,
  isNormalMode,
  isOpaqueMode,
  isVolumetricMode,
  usesPeakProjection,
  needsDepthSort,
  BLENDING_MODES,
  normalizeBlendingMode,
  getCompleteBlendingState,
  getGSplatNormalBlendingState,
  getPointBlendingState,
} from '../../../rendering/blending-state';
import type { BlendingMode } from '../../../rendering/material-manager';

describe('BlendingMode predicates', () => {
  const all: BlendingMode[] = ['additive', 'normal', 'max', 'opaque', 'luminous', 'volumetric'];

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

  it('isVolumetricMode is true only for volumetric', () => {
    for (const mode of all) {
      expect(isVolumetricMode(mode)).toBe(mode === 'volumetric');
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
        isVolumetricMode(mode),
      ].filter(Boolean).length;
      expect(hits).toBe(1);
    }
  });

  it('usesPeakProjection groups the surface modes (max/normal/opaque) against the emissive ones', () => {
    // PR #561 taxonomy: surface modes project the 2D-Gaussian peak;
    // emissive modes (additive/luminous/volumetric) integrate the ray.
    expect(usesPeakProjection('max')).toBe(true);
    expect(usesPeakProjection('normal')).toBe(true);
    expect(usesPeakProjection('opaque')).toBe(true);
    expect(usesPeakProjection('additive')).toBe(false);
    expect(usesPeakProjection('luminous')).toBe(false);
    expect(usesPeakProjection('volumetric')).toBe(false);
  });

  it('needsDepthSort is true exactly for the order-dependent modes (normal, volumetric)', () => {
    // Volumetric is the first SUM-projected SORTED mode — the two
    // taxonomies (projection vs order-dependence) are deliberately
    // independent predicates. Truth table pins both.
    expect(needsDepthSort('normal')).toBe(true);
    expect(needsDepthSort('volumetric')).toBe(true);
    expect(needsDepthSort('additive')).toBe(false);
    expect(needsDepthSort('luminous')).toBe(false);
    expect(needsDepthSort('max')).toBe(false);
    expect(needsDepthSort('opaque')).toBe(false);
  });
});

describe('normalizeBlendingMode', () => {
  it('passes every canonical mode through unchanged', () => {
    for (const mode of BLENDING_MODES) {
      expect(normalizeBlendingMode(mode)).toBe(mode);
    }
  });

  it("maps undefined to 'additive' (the composition identity)", () => {
    expect(normalizeBlendingMode(undefined)).toBe('additive');
  });

  it("coerces unknown strings to 'normal' (depth-sortable fallthrough)", () => {
    expect(normalizeBlendingMode('bogus-mode-a')).toBe('normal');
    expect(normalizeBlendingMode('')).toBe('normal');
    expect(normalizeBlendingMode('Additive')).toBe('normal'); // case-sensitive
  });

  it('warns exactly once per distinct unknown string', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      // Unique strings — the once-only Set is module-level, so reuse
      // across tests would make this pass vacuously.
      normalizeBlendingMode('warn-once-x');
      normalizeBlendingMode('warn-once-x');
      normalizeBlendingMode('warn-once-x');
      const forX = warnSpy.mock.calls.filter((c) => String(c).includes('warn-once-x'));
      expect(forX.length).toBe(1);

      normalizeBlendingMode('warn-once-y');
      const forY = warnSpy.mock.calls.filter((c) => String(c).includes('warn-once-y'));
      expect(forY.length).toBe(1);

      // Valid + undefined inputs never warn.
      warnSpy.mockClear();
      for (const mode of BLENDING_MODES) normalizeBlendingMode(mode);
      normalizeBlendingMode(undefined);
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
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

  it('volumetric: One/OneMinusSrcAlpha premultiplied state, depthWrite unconditionally OFF', () => {
    const state = getCompleteBlendingState('volumetric');
    expect(state.blending).toBe(THREE.CustomBlending);
    expect(state.blendEquation).toBe(THREE.AddEquation);
    expect(state.blendSrc).toBe(THREE.OneFactor);
    expect(state.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(state.depthTest).toBe(true);
    expect(state.depthWrite).toBe(false);
    expect(state.transparent).toBe(true);
    expect(state.shaderOutputMode).toBe('premultiplied-alpha');

    // No normalModeDepthWrite coupling: opacity never flips depthWrite
    // (unlike 'normal' — volumetric is smooth in opacity by design).
    expect(getCompleteBlendingState('volumetric', 1.0).depthWrite).toBe(false);
    expect(getCompleteBlendingState('volumetric', 0.99).depthWrite).toBe(false);

    // Identical framebuffer state to the gsplat-normal helper — the
    // semantic difference lives in the fragment shader, not the state.
    expect(state).toEqual(getGSplatNormalBlendingState());
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
    // CompleteBlendingState deliberately declares NO alpha-channel
    // fields (they had no producer and applyBlendingStateToMaterial
    // never applied them) — the symmetric-alpha guarantee is
    // structural. Guard against the fields being re-added and set.
    const s = getGSplatNormalBlendingState() as unknown as Record<string, unknown>;
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

describe('getPointBlendingState (points never depth-write in normal mode)', () => {
  const NON_DEPTHWRITE_FIELDS = [
    'blending',
    'blendEquation',
    'blendSrc',
    'blendDst',
    'depthTest',
    'transparent',
    'shaderOutputMode',
  ] as const;

  for (const opacity of [1.0, 0.5]) {
    it(`normal at opacity=${opacity}: depthWrite OFF, other fields match generic`, () => {
      const point = getPointBlendingState('normal', opacity);
      const generic = getCompleteBlendingState('normal', opacity);
      expect(point.depthWrite).toBe(false);
      for (const field of NON_DEPTHWRITE_FIELDS) {
        expect(point[field], `field '${field}'`).toBe(generic[field]);
      }
    });
  }

  it('contrasts with the generic normal state at opacity 1.0 (only depthWrite differs)', () => {
    expect(getCompleteBlendingState('normal', 1.0).depthWrite).toBe(true);
    expect(getPointBlendingState('normal', 1.0).depthWrite).toBe(false);
  });

  for (const mode of ['additive', 'luminous', 'max', 'opaque', 'volumetric'] as const) {
    it(`'${mode}' is identical to getCompleteBlendingState (only normal diverges)`, () => {
      expect(getPointBlendingState(mode, 1.0)).toEqual(getCompleteBlendingState(mode, 1.0));
    });
  }
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
