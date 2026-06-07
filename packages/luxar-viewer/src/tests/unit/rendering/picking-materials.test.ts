/**
 * Unit tests for GPU picking materials.
 *
 * Verifies that picking materials instantiate correctly, implement
 * CameraAwareMaterial, and have the expected uniforms.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointPickingMaterial } from '../../../rendering/picking/point/material';
import { LinePickingMaterial } from '../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../rendering/picking/line/material-tsl';
import { GSplatPickingMaterial } from '../../../rendering/picking/gsplat/material';
import { GSplatPickingTSLMaterial } from '../../../rendering/picking/gsplat/material-tsl';

describe('PointPickingMaterial', () => {
  it('instantiates with correct nodeId uniform', () => {
    const material = new PointPickingMaterial({ nodeId: 42 });
    expect(material.uniforms.uNodeId.value).toBe(42);
    expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    material.dispose();
  });

  it('uses correct material settings for picking', () => {
    const material = new PointPickingMaterial({ nodeId: 1 });
    expect(material.transparent).toBe(false);
    expect(material.depthTest).toBe(true);
    expect(material.depthWrite).toBe(true);
    expect(material.blending).toBe(THREE.NoBlending);
    material.dispose();
  });

  it('applies radiusScale', () => {
    const material = new PointPickingMaterial({
      nodeId: 1,
      radiusScale: 0.5,
    });
    expect(material.uniforms.radiusScale.value).toBe(0.5);
    material.dispose();
  });

  it('implements updateCameraParams', () => {
    const material = new PointPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);
    const fov = (60 * Math.PI) / 180;

    material.updateCameraParams(fov, resolution, false);

    expect(material.uniforms.uIsOrtho.value).toBe(0);
    expect(material.uniforms.maxPointSize.value).toBe(540); // 1080 * 0.5
    expect(material.uniforms.pointSizeFactor.value).toBeGreaterThan(0);
    material.dispose();
  });

  it('handles orthographic camera params', () => {
    const material = new PointPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(800, 600);
    const frustumHeight = 10; // world units

    material.updateCameraParams(frustumHeight, resolution, true);

    expect(material.uniforms.uIsOrtho.value).toBe(1);
    expect(material.uniforms.pointSizeFactor.value).toBe(240); // 2 * 600 / (10 * 0.5)
    material.dispose();
  });
});

describe('LinePickingMaterial', () => {
  it('instantiates with correct nodeId uniform', () => {
    const material = new LinePickingMaterial({ nodeId: 7 });
    expect(material.uniforms.uNodeId.value).toBe(7);
    expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    material.dispose();
  });

  it('uses correct material settings for picking', () => {
    const material = new LinePickingMaterial({ nodeId: 1 });
    expect(material.transparent).toBe(false);
    expect(material.depthTest).toBe(true);
    expect(material.blending).toBe(THREE.NoBlending);
    expect(material.side).toBe(THREE.DoubleSide);
    material.dispose();
  });

  it('implements updateCameraParams', () => {
    const material = new LinePickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false);

    expect(material.uniforms.uFOV.value).toBe(1.0);
    expect(material.uniforms.uResolution.value.x).toBe(1920);
    expect(material.uniforms.uResolution.value.y).toBe(1080);
    material.dispose();
  });

  // The sharpness profile is now a shifted-truncated super-Gaussian
  // (beta = 2^(6s - 2)) with no per-buffer fast path, so the picking
  // shader carries no LUXAR_SHARPNESS_TWO define.
  it('GLSL line picking shader uses the super-Gaussian falloff, no LUXAR_SHARPNESS_TWO', () => {
    const material = new LinePickingMaterial({ nodeId: 1 });
    expect(material.fragmentShader).toContain('exp2(6.0 * vSharpness - 2.0)');
    expect(material.fragmentShader).toContain('exp(-K * pow(p, beta))');
    expect(material.fragmentShader).not.toContain('LUXAR_SHARPNESS_TWO');
    expect('LUXAR_SHARPNESS_TWO' in (material.defines ?? {})).toBe(false);
    material.dispose();
  });
});

describe('GSplatPickingMaterial', () => {
  it('instantiates with correct nodeId and tighter truncation', () => {
    const material = new GSplatPickingMaterial({ nodeId: 99 });
    expect(material.uniforms.uNodeId.value).toBe(99);
    // Tighter truncation: 1.5σ instead of 3.0σ
    expect(material.uniforms.uTruncate.value).toBe(1.5);
    expect(material.uniforms.uTruncateSq.value).toBe(2.25);
    material.dispose();
  });

  it('uses max projection mode for picking (no uProjectionMode uniform; shader hard-codes max)', () => {
    // The picking shader hard-codes max projection — it has no
    // sum-projection ray-integral path — so neither the GLSL nor the
    // TSL picking materials bind a `uProjectionMode` uniform.
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    expect(material.uniforms.uProjectionMode).toBeUndefined();
    material.dispose();
  });

  it('implements updateCameraParams with nearCull', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false, 0.5);

    expect(material.uniforms.uNearCull.value).toBe(0.5);
    expect(material.uniforms.uFx.value).toBeGreaterThan(0);
    material.dispose();
  });

  // rendering.md G5 fix: GSplatPickingMaterial previously only had
  // an instantiation test. Orthographic branch (uIsOrtho=1) is an
  // independent code path in `computeFocalLength` (frustumHeight
  // semantics, not tan(fov/2)), so it must be exercised separately
  // to kill mutations to the `isOrtho ? 1 : 0` flag and the fy=fx
  // assignment.
  it('handles orthographic camera params (uIsOrtho=1)', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(800, 600);
    const frustumHeight = 10; // world units, ortho semantics

    material.updateCameraParams(frustumHeight, resolution, true);

    expect(material.uniforms.uIsOrtho.value).toBe(1);
    // For ortho: focal = resolution.y / frustumHeight = 600 / 10 = 60.
    // Both fx and fy must be set identically (square pixels assumption).
    expect(material.uniforms.uFx.value).toBeCloseTo(60, 5);
    expect(material.uniforms.uFy.value).toBeCloseTo(60, 5);
    expect(material.uniforms.uFx.value).toBe(material.uniforms.uFy.value);
    material.dispose();
  });

  it('updateCameraParams without nearCull leaves uNearCull at its default', () => {
    // Pins the contract that nearCull is opt-in (mirrors LinePicking
    // and the visual GSplat material).
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    const initialNearCull = material.uniforms.uNearCull.value;
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false);

    expect(material.uniforms.uNearCull.value).toBe(initialNearCull);
    material.dispose();
  });
});

// rendering.md G3, G4 fix: GSplatPickingTSLMaterial had ZERO direct
// tests despite being referenced by material-manager/factories.ts.
// Mirrors the LinePickingTSLMaterial block above one-for-one — same
// nodeId / camera-params / ortho-branch coverage (project memory
// "three-geometry symmetry rule": Points/Lines/GSplats parallel tests).
describe('GSplatPickingTSLMaterial', () => {
  it('instantiates with correct nodeId uniform', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 99 });
    expect(material.uniforms.uNodeId.value).toBe(99);
    // Tighter truncation: 1.5σ — must match GLSL picking path.
    expect(material.uniforms.uTruncate.value).toBe(1.5);
    expect(material.uniforms.uTruncateSq.value).toBe(2.25);
    material.dispose();
  });

  it('uses max projection mode (no uProjectionMode uniform)', () => {
    // Symmetric with GSplatPickingMaterial (GLSL): picking shader
    // hard-codes max projection; uProjectionMode is intentionally
    // NOT exposed (see material-tsl.ts module preamble).
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    expect(material.uniforms.uProjectionMode).toBeUndefined();
    material.dispose();
  });

  it('updateCameraParams (perspective) sets fx=fy and uIsOrtho=0', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false, 0.5);

    expect(material.uniforms.uIsOrtho.value).toBe(0);
    expect(material.uniforms.uNearCull.value).toBe(0.5);
    expect(material.uniforms.uFx.value).toBeGreaterThan(0);
    expect(material.uniforms.uFx.value).toBe(material.uniforms.uFy.value);
    material.dispose();
  });

  it('updateCameraParams (orthographic) sets uIsOrtho=1 and matching ortho focal', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(800, 600);
    const frustumHeight = 10;

    material.updateCameraParams(frustumHeight, resolution, true);

    expect(material.uniforms.uIsOrtho.value).toBe(1);
    // Mirror the GLSL ortho test: focal = res.y / frustumHeight = 60.
    expect(material.uniforms.uFx.value).toBeCloseTo(60, 5);
    expect(material.uniforms.uFy.value).toBeCloseTo(60, 5);
    material.dispose();
  });

  it('updateCameraParams without nearCull preserves uNearCull default', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    const initial = material.uniforms.uNearCull.value;
    material.updateCameraParams(1.0, new THREE.Vector2(800, 600), false);
    expect(material.uniforms.uNearCull.value).toBe(initial);
    material.dispose();
  });

  it('uniform proxy writes land on the underlying TSLNode (no .onUpdate bridge)', () => {
    // Documents the IUniform-proxy contract called out in the TSL
    // material's module preamble: mutating uniforms.uX.value must
    // mutate node.value directly. Symmetric with how LinePickingTSL
    // uniforms behave — and a cheap mutation-killer for the proxy.
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    material.uniforms.uNodeId.value = 42;
    expect(material.uniforms.uNodeId.value).toBe(42);
    material.dispose();
  });
});

describe('LinePickingTSLMaterial sharpness', () => {
  // The sharpness fast path (LUXAR_SHARPNESS_TWO / setSharpnessAllTwo)
  // was removed when the perpendicular profile became a shifted-truncated
  // super-Gaussian (beta = 2^(6s - 2)). The material no longer exposes a
  // setter and carries no such define.

  it('exposes no setSharpnessAllTwo and no LUXAR_SHARPNESS_TWO define', () => {
    const material = new LinePickingTSLMaterial({ nodeId: 1 });
    expect(
      (material as unknown as { setSharpnessAllTwo?: unknown }).setSharpnessAllTwo
    ).toBeUndefined();
    expect('LUXAR_SHARPNESS_TWO' in (material.defines ?? {})).toBe(false);
    material.dispose();
  });
});

// [rendering.md/O3][P10] computePickBufferSize tests live in their canonical
// location at tests/unit/rendering/picking/picking-system.test.ts. The
// previous duplicate block here covered the same algorithm with overlapping
// inputs; all unique cases (asymmetric cap, floor-fractional, negative-clamp)
// have been merged into the canonical file.
