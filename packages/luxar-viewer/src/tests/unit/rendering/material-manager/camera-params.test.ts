/**
 * Regression tests for camera-parameter propagation to NEWLY CREATED
 * cached materials.
 *
 * `MaterialManager.updateCameraParams` stores the latest
 * (fov, resolution, isOrtho, nearCull) and broadcasts to registered
 * materials — but the three `getXMaterial` factories must ALSO apply
 * the full stored state to a material created afterwards. Historically
 * they dropped `currentNearCull`, so a material created after camera
 * setup kept its constructor-default near-cull until the next
 * resize/FOV event ("splats missing until the camera moves" on scenes
 * whose world scale differs from the 0.1 default).
 *
 * Uses real material classes (no `vi.mock('three')`) — same pattern as
 * material-cache-lru.test.ts.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  MaterialManager,
  __resetMaterialManagerForTests,
} from '../../../../rendering/material-manager';
import type { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import type { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';

const baseProps = (over: Record<string, number | boolean | string> = {}) =>
  ({
    opacity: 1.0,
    gamma: 1.0,
    intensity: 1.0,
    offset: 0.0,
    blendingMode: 'additive',
    ...over,
  }) as Parameters<MaterialManager['getLineMaterial']>[0];

describe('MaterialManager camera params on newly created materials', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  it('applies the stored nearCull to a line material created after updateCameraParams', () => {
    const mm = new MaterialManager();
    mm.updateCameraParams(1.0, new THREE.Vector2(800, 600), false, 0.42);

    const line = mm.getLineMaterial(baseProps()) as LineMaterial;
    expect(line.uniforms.uNearCull.value).toBe(0.42);
  });

  it('applies the stored nearCull to a gsplat material created after updateCameraParams', () => {
    const mm = new MaterialManager();
    mm.updateCameraParams(1.0, new THREE.Vector2(800, 600), false, 0.42);

    const gsplat = mm.getGSplatMaterial(baseProps()) as GSplatMaterial;
    expect(gsplat.uniforms.uNearCull.value).toBe(0.42);
  });

  it('applies the stored resolution/fov/isOrtho to materials created after updateCameraParams', () => {
    const mm = new MaterialManager();
    mm.updateCameraParams(1.25, new THREE.Vector2(1234, 777), true, 0.5);

    const line = mm.getLineMaterial(baseProps()) as LineMaterial;
    expect(line.uniforms.uResolution.value.x).toBe(1234);
    expect(line.uniforms.uResolution.value.y).toBe(777);
    expect(line.uniforms.uIsOrtho.value).toBe(1);

    const gsplat = mm.getGSplatMaterial(baseProps()) as GSplatMaterial;
    expect(gsplat.uniforms.uResolution.value.x).toBe(1234);
    expect(gsplat.uniforms.uIsOrtho.value).toBe(1);
  });

  it('keeps constructor-default near-cull when no camera update has happened yet', () => {
    const mm = new MaterialManager();

    const line = mm.getLineMaterial(baseProps()) as LineMaterial;
    // 0.1 — aligned with the point/gsplat ctor default (deep-campaign
    // ctor-default-drift fix; the value only matters pre-first-broadcast).
    expect(line.uniforms.uNearCull.value).toBe(0.1);

    const gsplat = mm.getGSplatMaterial(baseProps()) as GSplatMaterial;
    expect(gsplat.uniforms.uNearCull.value).toBe(0.1);
  });
});
