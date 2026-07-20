/**
 * LineMaterial / LineTSLMaterial `applyBlendingMode` parity tests.
 *
 * Sibling of `materials/point/blending-mode.test.ts` (three-geometry
 * test symmetry). Ensures runtime UI transitions produce the same
 * complete blend state as material creation, on BOTH backends. Both
 * wrappers draw from the SHARED `getCompleteBlendingState` +
 * `applyBlendingStateToMaterial` helpers, so for every mode the GLSL
 * and TSL materials carry identical blending / blend-factor / depth /
 * transparency state (the convergence suite below pins that).
 *
 * Both toggle the LUXAR_MAX_RGB_CONTRIBUTION shader define when
 * entering max and clear it when leaving.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { LineMaterial } from '../../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';
import { getCompleteBlendingState } from '../../../../../rendering/blending-state';
import type { BlendingMode } from '../../../../../rendering/material-manager';

describe('LineMaterial.applyBlendingMode (GLSL)', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
    expect(mat.userData.blendingMode).toBe('max');
  });

  it('additive mode resets CustomBlending state via the shared helper (SrcAlpha/One)', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // Shared getCompleteBlendingState: SrcAlpha/One (resets stranded
    // OneFactor from the previous max state; inert under the
    // AdditiveBlending preset, and now byte-identical to the TSL twin).
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal at opacity=1.0 writes depth; at 0.5 does not', () => {
    const mat = new LineMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(true);

    mat.uniforms.uOpacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('opaque mode disables transparency and writes depth', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('luminous mode is additive but depth-tested', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('luminous');
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    // THREE.Material.needsUpdate is a setter that increments `version`.
    // Read via `version` to detect whether a recompile would be triggered.
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('mode changes that toggle defines bump material.version (max→additive)', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled (LUXAR_MAX_RGB_CONTRIBUTION removed) → recompile required
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
  });

  it('line fragment shader contains LUXAR_MAX_RGB_CONTRIBUTION guard', () => {
    const mat = new LineMaterial();
    expect(mat.fragmentShader).toContain('#ifdef LUXAR_MAX_RGB_CONTRIBUTION');
    expect(mat.fragmentShader).toContain('gammaColor * a');
  });
});

describe('LineTSLMaterial.applyBlendingMode (TSL)', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
    expect(mat.userData.blendingMode).toBe('max');
  });

  it('additive mode resets CustomBlending state via the shared helper (SrcAlpha/One)', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // TSL wrapper applies getCompleteBlendingState: SrcAlpha/One (the
    // GLSL wrapper's hand-rolled reset differs in blendDst; both are
    // inert under the AdditiveBlending preset).
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal at opacity=1.0 writes depth; at 0.5 does not', () => {
    const mat = new LineTSLMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(true);

    mat.uniforms.uOpacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('opaque mode disables transparency and writes depth', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('luminous mode is additive but depth-tested', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.userData.blendingMode).toBe('luminous');
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('mode changes that toggle defines bump material.version (max→additive)', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled → rebuildGraph → needsUpdate → version bump
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
  });
});

// Both wrappers delegate to getCompleteBlendingState +
// applyBlendingStateToMaterial, so for EVERY mode the applied THREE
// state must equal the helper's canonical values and the two backends
// must agree field-for-field. This is the convergence contract that
// replaced the GLSL wrapper's hand-rolled per-mode dispatch.
describe('LineMaterial ↔ LineTSLMaterial blending-state convergence', () => {
  const ALL_MODES: BlendingMode[] = ['additive', 'normal', 'max', 'opaque', 'luminous'];
  const STATE_FIELDS = [
    'blending',
    'blendEquation',
    'blendSrc',
    'blendDst',
    'depthTest',
    'depthWrite',
    'transparent',
  ] as const;

  for (const mode of ALL_MODES) {
    it(`'${mode}': GLSL state equals getCompleteBlendingState and matches TSL`, () => {
      const glsl = new LineMaterial();
      const tsl = new LineTSLMaterial();
      glsl.applyBlendingMode(mode);
      tsl.applyBlendingMode(mode);
      const expected = getCompleteBlendingState(mode, 1.0);
      for (const field of STATE_FIELDS) {
        expect(glsl[field], `GLSL ${field} for '${mode}'`).toBe(expected[field]);
        expect(tsl[field], `TSL ${field} for '${mode}'`).toBe(expected[field]);
      }
      expect(glsl.userData.blendingMode).toBe(mode);
      expect(tsl.userData.blendingMode).toBe(mode);
    });
  }

  it("'volumetric': phase-1 fallback applies the ADDITIVE state, userData keeps 'volumetric'", () => {
    // Lines don't implement the emission–absorption fragment math yet
    // (VOLUMETRIC_BLENDING_SPEC.md phases 3–4). The material intercepts
    // the mode and applies additive — the exact κ=0 limit — while the
    // REQUESTED mode stays in userData so stored scenes upgrade
    // automatically when the line implementation lands.
    const expected = getCompleteBlendingState('additive', 1.0);
    for (const mat of [new LineMaterial(), new LineTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      for (const field of STATE_FIELDS) {
        expect(mat[field], `${mat.constructor.name} ${field}`).toBe(expected[field]);
      }
      expect(mat.userData.blendingMode).toBe('volumetric');
    }
  });

  it('volumetric fallback survives TSL CONSTRUCTION and graph REBUILDS (factory-tail interception)', () => {
    // Mirror of the point twin: the TSL factory tail is the only state
    // writer at construction and re-runs on every rebuildGraph, so it
    // must intercept volumetric itself (pre-fix it applied the raw
    // premultiplied state under the alpha-weighted line shader).
    const expected = getCompleteBlendingState('additive', 1.0);
    const constructed = new LineTSLMaterial({ blendingMode: 'volumetric' });
    for (const field of STATE_FIELDS) {
      expect(constructed[field], `constructed ${field}`).toBe(expected[field]);
    }
    expect(constructed.userData.blendingMode).toBe('volumetric');

    const switched = new LineTSLMaterial({ blendingMode: 'max' });
    switched.applyBlendingMode('volumetric');
    for (const field of STATE_FIELDS) {
      expect(switched[field], `post-switch ${field}`).toBe(expected[field]);
    }

    switched.updateGamma(2.2); // gamma crossing 1.0 → rebuildGraph
    for (const field of STATE_FIELDS) {
      expect(switched[field], `post-rebuild ${field}`).toBe(expected[field]);
    }
  });
});

// H — TSL constructors honor explicit transparent/depthTest overrides
// AFTER the factory tail's mode-derived state, exactly like the GLSL
// twins (additive state otherwise forces depthTest=false).
describe('LineTSLMaterial constructor explicit overrides', () => {
  it('depthTest: true survives additive construction', () => {
    const mat = new LineTSLMaterial({ blendingMode: 'additive', depthTest: true });
    expect(mat.depthTest).toBe(true);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('transparent: false survives additive construction', () => {
    const mat = new LineTSLMaterial({ blendingMode: 'additive', transparent: false });
    expect(mat.transparent).toBe(false);
  });
});
