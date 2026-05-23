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
import {
  MAX_PICK_BUFFER_DIM,
  computePickBufferSize,
} from '../../../rendering/picking/picking-system';

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

  it('applies radiusScale and sharpnessScale', () => {
    const material = new PointPickingMaterial({
      nodeId: 1,
      radiusScale: 0.5,
      sharpnessScale: 0.25,
    });
    expect(material.uniforms.radiusScale.value).toBe(0.5);
    expect(material.uniforms.sharpnessScale.value).toBe(0.25);
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

  // rendering.md G2 fix: setSharpnessAllTwo was tested for the TSL Line
  // picking material but not for the GLSL one. create-lines-node.ts:93 calls
  // it on every line node, so a regression in the GLSL define-toggle would
  // silently break sharpness=2 fast-path picking.
  it('setSharpnessAllTwo(true) sets the LUXAR_SHARPNESS_TWO define on the GLSL material', () => {
    const material = new LinePickingMaterial({ nodeId: 1 });
    material.setSharpnessAllTwo(true);
    // Source uses `defines.LUXAR_SHARPNESS_TWO = ''` as a presence flag
    // (the GLSL `#ifdef` keys on presence, not value).
    expect('LUXAR_SHARPNESS_TWO' in material.defines).toBe(true);
    material.dispose();
  });

  it('setSharpnessAllTwo(false) clears the LUXAR_SHARPNESS_TWO define on the GLSL material', () => {
    const material = new LinePickingMaterial({ nodeId: 1 });
    material.setSharpnessAllTwo(true);
    material.setSharpnessAllTwo(false);
    expect('LUXAR_SHARPNESS_TWO' in material.defines).toBe(false);
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

describe('LinePickingTSLMaterial sharpness fast path', () => {
  // Pins the LUXAR_SHARPNESS_TWO contract: `setSharpnessAllTwo(true)`
  // stores the define as an empty string (matching the GLSL3 `#define`
  // shape), so `_currentConfig` must use `'KEY' in defines`, not
  // truthiness. Before the fix, the truthy check disabled the
  // advertised TSL sharpness=2 fast path even though the define was
  // present.

  it('setSharpnessAllTwo(true) sets the define AND rebuilds the graph', () => {
    const material = new LinePickingTSLMaterial({ nodeId: 1 });
    const v0 = material.version;

    material.setSharpnessAllTwo(true);

    expect(material.defines).toBeDefined();
    expect('LUXAR_SHARPNESS_TWO' in (material.defines as Record<string, unknown>)).toBe(true);
    // The set value is '' (falsey) — what matters is its presence.
    expect((material.defines as Record<string, string>).LUXAR_SHARPNESS_TWO).toBe('');
    // Graph rebuild bumps version (set via `this.needsUpdate = true`).
    expect(material.version).toBeGreaterThan(v0);

    material.dispose();
  });

  it('setSharpnessAllTwo(true)→(false) clears the define and rebuilds', () => {
    const material = new LinePickingTSLMaterial({ nodeId: 1 });
    material.setSharpnessAllTwo(true);
    const v1 = material.version;

    material.setSharpnessAllTwo(false);

    expect('LUXAR_SHARPNESS_TWO' in (material.defines as Record<string, unknown>)).toBe(false);
    expect(material.version).toBeGreaterThan(v1);

    material.dispose();
  });

  it('setSharpnessAllTwo(true) twice is idempotent (no second rebuild)', () => {
    const material = new LinePickingTSLMaterial({ nodeId: 1 });
    material.setSharpnessAllTwo(true);
    const v1 = material.version;

    material.setSharpnessAllTwo(true);
    expect(material.version).toBe(v1);

    material.dispose();
  });
});

describe('computePickBufferSize', () => {
  it('halves both dimensions for normal-sized buffers', () => {
    expect(computePickBufferSize(1920, 1080)).toEqual({ w: 960, h: 540 });
  });

  it('caps oversized 4K buffer at MAX_PICK_BUFFER_DIM on the wide axis only', () => {
    // 3840 / 2 = 1920 -> capped at 1024; 2160 / 2 = 1080 -> capped at 1024
    expect(computePickBufferSize(3840, 2160)).toEqual({
      w: MAX_PICK_BUFFER_DIM,
      h: MAX_PICK_BUFFER_DIM,
    });
  });

  it('caps only the axis that exceeds the limit', () => {
    // 5120 / 2 = 2560 -> capped; 1440 / 2 = 720 -> kept
    expect(computePickBufferSize(5120, 1440)).toEqual({
      w: MAX_PICK_BUFFER_DIM,
      h: 720,
    });
  });

  it('floors fractional values', () => {
    expect(computePickBufferSize(1921, 1081)).toEqual({ w: 960, h: 540 });
  });

  it('clamps to a minimum of 1 pixel', () => {
    expect(computePickBufferSize(0, 0)).toEqual({ w: 1, h: 1 });
    expect(computePickBufferSize(1, 1)).toEqual({ w: 1, h: 1 });
  });

  // rendering.md G8: negative inputs aren't an expected runtime case
  // (canvas dimensions are always ≥ 0), but the contract is that the
  // function never returns a value below the documented floor of 1.
  // Pins the defensive behaviour against future "fast-path" tweaks
  // that drop the Math.max clamp.
  it('clamps negative inputs to 1 pixel (defensive contract)', () => {
    expect(computePickBufferSize(-100, -100)).toEqual({ w: 1, h: 1 });
    expect(computePickBufferSize(-1, 1080)).toEqual({ w: 1, h: 540 });
    expect(computePickBufferSize(1920, -1)).toEqual({ w: 960, h: 1 });
  });
});
