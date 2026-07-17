/**
 * LineMaterial / LineTSLMaterial `applyBlendingMode` parity tests.
 *
 * Sibling of `materials/point/blending-mode.test.ts` (three-geometry
 * test symmetry). Ensures runtime UI transitions produce the same
 * complete blend state as material creation, on BOTH backends:
 *
 *   - GLSL (`LineMaterial`): hand-rolled per-mode dispatch. Non-max
 *     modes reset the CustomBlending factors to
 *     SrcAlpha / OneMinusSrcAlpha (the preset blending modes ignore
 *     them — this only prevents stranded MaxEquation state).
 *   - TSL (`LineTSLMaterial`): draws from the shared
 *     `getCompleteBlendingState`, so additive/luminous pin
 *     SrcAlpha / One instead.
 *
 * Both toggle the LUXAR_MAX_RGB_CONTRIBUTION shader define when
 * entering max and clear it when leaving.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { LineMaterial } from '../../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';

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

  it('additive mode resets CustomBlending state (no stranded MaxEquation/OneFactor)', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // GLSL line resets to SrcAlpha/OneMinusSrcAlpha (unlike the shared
    // helper's SrcAlpha/One) — inert under the AdditiveBlending preset,
    // pinned so a stranded OneFactor can never reappear.
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
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
