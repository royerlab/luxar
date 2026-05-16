/**
 * Unit tests for GPU picking materials.
 *
 * Verifies that picking materials instantiate correctly, implement
 * CameraAwareMaterial, and have the expected uniforms.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointPickingMaterial } from '../../../rendering/picking/point-picking-material';
import { LinePickingMaterial } from '../../../rendering/picking/line-picking-material';
import { GSplatPickingMaterial } from '../../../rendering/picking/gsplat-picking-material';
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
});
