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
    expect(mat.fragmentShader).toContain('defined(LUXAR_MAX_RGB_CONTRIBUTION)');
    expect(mat.fragmentShader).toContain('finalColor * alpha');
  });

  it('point fragment shader contains the volumetric emission–absorption branch', () => {
    // Shader-text pins for the LUXAR_VOLUMETRIC output branch — a
    // pure-TS state test can't guard the emitted GLSL (the blending
    // campaign's mutation lesson). τ = κ·density·chord, S(τ) series
    // branch, physical absorption alpha, and the color-discard bypass
    // (a black point still absorbs) are each distinct generated code.
    const mat = new PointMaterial();
    expect(mat.fragmentShader).toContain('#if defined(LUXAR_VOLUMETRIC)');
    expect(mat.fragmentShader).toContain('float tau = uAbsorption * alpha * vRadius *');
    expect(mat.fragmentShader).toContain('float volAlpha = 1.0 - exp(-tau);');
    expect(mat.fragmentShader).toContain('tau < 1e-4) discard');
    expect(mat.fragmentShader).toContain('fragColor = vec4(finalColor * alpha * screen, volAlpha)');
    // Per-point alpha → optical depth map, gated by uHasElementAlpha.
    expect(mat.fragmentShader).toContain('-log(1.0 - min(vAlpha,');
    expect(mat.fragmentShader).toContain('uHasElementAlpha');
  });

  it("'volumetric' applies the REAL emission–absorption state (phase 3), userData keeps 'volumetric'", () => {
    // Points implement the volumetric fragment math since phase 3
    // (VOLUMETRIC_BLENDING_SPEC.md): premultiplied self-screened
    // emission over One/OneMinusSrcAlpha — the same framebuffer state
    // as gsplat volumetric — never depth-writes, depth-tested.
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      expect(mat.blending).toBe(THREE.CustomBlending);
      expect(mat.blendEquation).toBe(THREE.AddEquation);
      expect(mat.blendSrc).toBe(THREE.OneFactor);
      expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
      expect(mat.depthTest).toBe(true);
      expect(mat.depthWrite).toBe(false);
      expect(mat.transparent).toBe(true);
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBe('');
      expect(mat.userData.blendingMode).toBe('volumetric');
    }
  });

  it('every non-volumetric transition clears LUXAR_VOLUMETRIC (no stranded define)', () => {
    // Risk #6 of the spec: a volumetric→normal switch must not strand
    // the define — the normal branch would then never be reached.
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBe('');
      mat.applyBlendingMode('normal');
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBeUndefined();
      mat.applyBlendingMode('volumetric');
      mat.applyBlendingMode('additive');
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBeUndefined();
    }
  });

  it('volumetric state survives TSL CONSTRUCTION and graph REBUILDS (factory-tail interception)', () => {
    // The TSL factory tail is the ONLY state writer at construction
    // (the ctor never calls applyBlendingMode, unlike GLSL) and re-runs
    // on every rebuildGraph — so it must derive the volumetric state
    // (and output branch) itself from config.blendingMode.
    const constructed = new PointTSLMaterial({ blendingMode: 'volumetric' });
    expect(constructed.blending).toBe(THREE.CustomBlending);
    expect(constructed.blendSrc).toBe(THREE.OneFactor);
    expect(constructed.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(constructed.defines?.LUXAR_VOLUMETRIC).toBe('');
    expect(constructed.userData.blendingMode).toBe('volumetric');

    // max→volumetric toggles BOTH defines → definesChanged →
    // rebuildGraph — the tail must re-derive the volumetric state, not
    // clobber it with a stale-mode default.
    const switched = new PointTSLMaterial({ blendingMode: 'max' });
    switched.applyBlendingMode('volumetric');
    expect(switched.blending).toBe(THREE.CustomBlending);
    expect(switched.blendSrc).toBe(THREE.OneFactor);
    expect(switched.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(switched.defines?.LUXAR_VOLUMETRIC).toBe('');

    // Any later rebuild while volumetric (e.g. gamma crossing 1.0)
    // must not clobber the state either.
    switched.updateGamma(2.2);
    expect(switched.blending).toBe(THREE.CustomBlending);
    expect(switched.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
  });

  it('clone() carries uAbsorption and uHasElementAlpha (both backends)', () => {
    // The layers panel clones on first interaction; a clone that reset
    // κ to 1.0 or dropped the RGBA-alpha flag would silently change the
    // volumetric render (the gsplat phase-1 review caught the same bug
    // class in its clones).
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      mat.updateAbsorption(2.5);
      mat.updateHasElementAlpha(true);
      const cloned = mat.clone();
      expect(cloned.uniforms.uAbsorption.value).toBe(2.5);
      expect(cloned.uniforms.uHasElementAlpha.value).toBe(1);
      expect(cloned.userData.blendingMode).toBe('volumetric');
      expect(cloned.blending).toBe(THREE.CustomBlending);
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
