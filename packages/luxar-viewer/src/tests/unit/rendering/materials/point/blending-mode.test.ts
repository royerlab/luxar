/**
 * PointMaterial.applyBlendingMode parity tests.
 *
 * Ensures runtime UI transitions produce the same complete blend state
 * as material creation. Also verifies the
 * LUXAR_MAX_RGB_CONTRIBUTION shader define is set when entering max
 * and cleared when leaving.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../../../../rendering/materials/point/material-tsl';

describe('PointMaterial.applyBlendingMode', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor', () => {
    const mat = new PointMaterial();
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
    const mat = new PointMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // Additive uses SrcAlpha/One (resets stranded OneFactor)
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal at opacity=1.0 writes depth; at 0.5 does not', () => {
    const mat = new PointMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.depthWrite).toBe(true);

    mat.uniforms.opacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('opaque mode disables transparency and writes depth', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
  });

  it('luminous mode is additive but depth-tested', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    // THREE.Material.needsUpdate is a setter that increments `version`.
    // Read via `version` to detect whether a recompile would be triggered.
    const mat = new PointMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('mode changes that toggle defines bump material.version (max→additive)', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled (LUXAR_MAX_RGB_CONTRIBUTION removed) → recompile required
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
  });

  it('point fragment shader contains LUXAR_MAX_RGB_CONTRIBUTION guard', () => {
    const mat = new PointMaterial();
    expect(mat.fragmentShader).toContain('#ifdef LUXAR_MAX_RGB_CONTRIBUTION');
    expect(mat.fragmentShader).toContain('finalColor * alpha');
  });

  it("'volumetric': phase-1 fallback applies the ADDITIVE state, userData keeps 'volumetric'", () => {
    // Points don't implement the emission–absorption fragment math yet
    // (VOLUMETRIC_BLENDING_SPEC.md phases 3–4). The material intercepts
    // the mode and applies additive — the exact κ=0 limit — while the
    // REQUESTED mode stays in userData so stored scenes upgrade
    // automatically when the point implementation lands.
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      expect(mat.blending).toBe(THREE.AdditiveBlending);
      expect(mat.blendEquation).toBe(THREE.AddEquation);
      expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
      expect(mat.blendDst).toBe(THREE.OneFactor);
      expect(mat.depthTest).toBe(false);
      expect(mat.depthWrite).toBe(false);
      expect(mat.transparent).toBe(true);
      expect(mat.userData.blendingMode).toBe('volumetric');
    }
  });
});

// H — TSL constructor honors explicit transparent/depthTest overrides
// AFTER the factory tail's mode-derived state, exactly like the GLSL
// twin (additive state otherwise forces depthTest=false).
describe('PointTSLMaterial constructor explicit overrides', () => {
  it('depthTest: true survives additive construction', () => {
    const mat = new PointTSLMaterial({ blendingMode: 'additive', depthTest: true });
    expect(mat.depthTest).toBe(true);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('transparent: false survives additive construction', () => {
    const mat = new PointTSLMaterial({ blendingMode: 'additive', transparent: false });
    expect(mat.transparent).toBe(false);
  });
});
