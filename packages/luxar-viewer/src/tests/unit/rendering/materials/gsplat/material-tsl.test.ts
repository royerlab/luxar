/**
 * Unit tests for GSplatTSLMaterial clone semantics.
 *
 * P8 three-geometry symmetry: mirrors `materials/line/material-tsl.test.ts`.
 * GSplat TSL has two graph-specialized configs that the constructor
 * picks up from `userData.blendingMode` and `defines.USE_COLORMAP`.
 * The clone must:
 *
 *   - Preserve `userData.blendingMode` (drives sum-vs-max projection
 *     selection in the factory — a clone that lost it would silently
 *     fall back to additive sum projection and break max-projection
 *     gsplat rendering, as called out by rendering.md G3).
 *   - Preserve the `uProjectionMode` uniform value (0 = sum, 1 = max).
 *   - Preserve `USE_COLORMAP` define + the scalar-range uniforms when
 *     the source material has a colormap attached.
 *   - Resync key camera uniforms (`uFx`, `uFy`, `uResolution`).
 *
 * The TSL factory's `rebuildGraph` is invoked inside the constructor
 * so the clone — constructed via `new GSplatTSLMaterial(config)` —
 * picks up the right graph specialization automatically. These tests
 * pin that contract (a regression that dropped the `userData` stamp
 * before `rebuildGraph` would silently revert to additive sum mode).
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GSplatTSLMaterial } from '../../../../../rendering/materials/gsplat/material-tsl';
import { getGSplatNormalBlendingState } from '../../../../../rendering/blending-state';

describe('GSplatTSLMaterial clone', () => {
  it('preserves the max-projection blending mode through clone', () => {
    // P8 symmetry with LineTSLMaterial.clone preserving
    // LUXAR_SHARPNESS_TWO: both are graph-specialized configs.
    const original = new GSplatTSLMaterial({ blendingMode: 'max' });
    expect(original.userData.blendingMode).toBe('max');
    expect(original.uniforms.uProjectionMode.value).toBe(1);

    const cloned = original.clone();

    expect(cloned.userData.blendingMode).toBe('max');
    expect(cloned.uniforms.uProjectionMode.value).toBe(1);
  });

  it('preserves the additive (sum-projection) blending mode through clone', () => {
    const original = new GSplatTSLMaterial({ blendingMode: 'additive' });
    expect(original.uniforms.uProjectionMode.value).toBe(0);

    const cloned = original.clone();

    expect(cloned.userData.blendingMode).toBe('additive');
    expect(cloned.uniforms.uProjectionMode.value).toBe(0);
  });

  it('preserves gamma, opacity, intensity, offset numerics through clone', () => {
    // P7 numeric tolerances: gamma is clamped + inverted, so we check
    // the inverse-gamma uniform with Float32 tolerance (~1e-5).
    const original = new GSplatTSLMaterial({
      gamma: 2.2,
      opacity: 0.7,
      intensity: 1.5,
      offset: 0.25,
    });

    const cloned = original.clone();

    expect(cloned.uniforms.uOpacity.value).toBeCloseTo(0.7, 5);
    expect(cloned.uniforms.uIntensity.value).toBeCloseTo(1.5, 5);
    expect(cloned.uniforms.uOffset.value).toBeCloseTo(0.25, 5);
    expect(cloned.uniforms.uInvGamma.value).toBeCloseTo(1 / 2.2, 5);
    expect(cloned.userData.gamma).toBeCloseTo(2.2, 5);
  });

  it('preserves truncation radius (and derived shift/inv-one-minus-c) through clone', () => {
    const original = new GSplatTSLMaterial({ truncationRadius: 2.5 });
    const expectedShiftC = Math.exp(-0.5 * 2.5 * 2.5);
    const expectedInvOneMinusC = 1.0 / (1.0 - expectedShiftC);

    const cloned = original.clone();

    expect(cloned.uniforms.uTruncate.value).toBeCloseTo(2.5, 5);
    expect(cloned.uniforms.uTruncateSq.value).toBeCloseTo(6.25, 5);
    expect(cloned.uniforms.uShiftC.value).toBeCloseTo(expectedShiftC, 5);
    expect(cloned.uniforms.uInvOneMinusC.value).toBeCloseTo(expectedInvOneMinusC, 5);
  });

  it('resyncs camera uniforms (uFx, uFy, uResolution) from source onto clone', () => {
    const original = new GSplatTSLMaterial();
    original.updateCameraParams(Math.PI / 3, new THREE.Vector2(1600, 900), false);
    const srcFx = original.uniforms.uFx.value as number;
    const srcRes = original.uniforms.uResolution.value as THREE.Vector2;

    const cloned = original.clone();

    expect(cloned.uniforms.uFx.value).toBeCloseTo(srcFx, 5);
    expect(cloned.uniforms.uFy.value).toBeCloseTo(srcFx, 5);
    const clonedRes = cloned.uniforms.uResolution.value as THREE.Vector2;
    expect(clonedRes.x).toBeCloseTo(srcRes.x, 5);
    expect(clonedRes.y).toBeCloseTo(srcRes.y, 5);
  });

  it('preserves the USE_COLORMAP define when source has a colormap texture', () => {
    // A 2-row 256-pixel DataTexture is the minimum that survives
    // GSplat colormap binding (matches PointTSL clone test fixture).
    const lut = new THREE.DataTexture(
      new Uint8Array(256 * 2 * 4),
      256,
      2,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    );
    lut.needsUpdate = true;

    const original = new GSplatTSLMaterial({
      colormapTexture: lut,
      scalarRange: [0.1, 0.9],
    });
    expect('USE_COLORMAP' in (original.defines ?? {})).toBe(true);

    const cloned = original.clone();

    expect('USE_COLORMAP' in (cloned.defines ?? {})).toBe(true);
    // scalarRange is stamped on userData and re-derived into uScalarMin /
    // uScalarScale by the constructor. Verify both made it across.
    expect(cloned.uniforms.uScalarMin?.value).toBeCloseTo(0.1, 5);
    // scalarScale = 1/(max-min) = 1/0.8 = 1.25.
    expect(cloned.uniforms.uScalarScale?.value).toBeCloseTo(1.25, 5);

    lut.dispose();
  });

  it('clone of a default material has no colormap define and no scalar uniforms', () => {
    const original = new GSplatTSLMaterial();
    const cloned = original.clone();

    expect('USE_COLORMAP' in (cloned.defines ?? {})).toBe(false);
    expect(cloned.uniforms.uScalarMin).toBeUndefined();
    expect(cloned.uniforms.uScalarScale).toBeUndefined();
  });

  it('cloned material has independent uniform identity (mutation does not leak back)', () => {
    // P1/P8: clone must produce a deep-enough copy that uniform writes
    // on the clone do not affect the source. This is the cheapest
    // mutation-killer for "clone forgot to construct fresh TSLNodes".
    const original = new GSplatTSLMaterial({ blendingMode: 'max' });
    const cloned = original.clone();

    cloned.uniforms.uOpacity.value = 0.123;
    expect(original.uniforms.uOpacity.value).not.toBe(0.123);
  });

  it('clone preserves a tuned uMaxExtentFactor (was silently reset to 0.33)', () => {
    const original = new GSplatTSLMaterial({ maxExtentFactor: 0.7 });
    expect(original.uniforms.uMaxExtentFactor.value).toBe(0.7);

    const cloned = original.clone();
    expect(cloned.uniforms.uMaxExtentFactor.value).toBe(0.7);
  });
});

describe('GSplatTSLMaterial normal mode — premultiplied coverage alpha', () => {
  it('constructor normal: gsplat-specific state (CustomBlending One/OneMinusSrcAlpha, no depth write)', () => {
    const material = new GSplatTSLMaterial({ blendingMode: 'normal' });

    const expected = getGSplatNormalBlendingState();
    expect(material.blending).toBe(expected.blending);
    expect(material.blendEquation).toBe(expected.blendEquation);
    expect(material.blendSrc).toBe(expected.blendSrc);
    expect(material.blendDst).toBe(expected.blendDst);
    expect(material.transparent).toBe(true);
    expect(material.depthTest).toBe(true);
    expect(material.depthWrite).toBe(false);
    // Never the premultipliedAlpha flag — NodeMaterial.setup() would
    // auto-inject a second RGB×alpha transform on this path.
    expect(material.premultipliedAlpha).toBe(false);
    expect(material.uniforms.uProjectionMode.value).toBe(0);
  });

  it('depthWrite stays OFF at opacity 1.0 (no generic opacity>=0.99 gate)', () => {
    const material = new GSplatTSLMaterial({ blendingMode: 'normal', opacity: 1.0 });
    expect(material.depthWrite).toBe(false);
  });

  it('round-trips normal → additive → normal restoring both states exactly', () => {
    const material = new GSplatTSLMaterial({ blendingMode: 'normal' });

    material.applyBlendingMode('additive');
    expect(material.blending).toBe(THREE.AdditiveBlending);
    expect(material.depthTest).toBe(false);
    expect(material.userData.blendingMode).toBe('additive');

    material.applyBlendingMode('normal');
    const expected = getGSplatNormalBlendingState();
    expect(material.blending).toBe(expected.blending);
    expect(material.blendSrc).toBe(expected.blendSrc);
    expect(material.blendDst).toBe(expected.blendDst);
    expect(material.depthWrite).toBe(false);
    expect(material.transparent).toBe(true);
    expect(material.userData.blendingMode).toBe('normal');
  });

  it('clone preserves normal mode with the premultiplied state', () => {
    const original = new GSplatTSLMaterial({ blendingMode: 'normal' });
    const cloned = original.clone();

    expect(cloned.userData.blendingMode).toBe('normal');
    const expected = getGSplatNormalBlendingState();
    expect(cloned.blending).toBe(expected.blending);
    expect(cloned.blendSrc).toBe(expected.blendSrc);
    expect(cloned.blendDst).toBe(expected.blendDst);
    expect(cloned.depthWrite).toBe(false);
  });
});
