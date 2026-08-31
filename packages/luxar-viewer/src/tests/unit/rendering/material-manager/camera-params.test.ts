/**
 * Regression tests for camera-parameter propagation to NEWLY CREATED
 * cached materials.
 *
 * `MaterialManager.updateCameraParams` stores the latest
 * (fov, resolution, isOrtho, nearCull) and broadcasts to registered
 * materials — but the four `getXMaterial` factories must ALSO apply
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
import type { MeshMaterial } from '../../../../rendering/materials/mesh/material-glsl';

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
    mm.updateCameraParams(1.0, new THREE.Vector2(800, 600), false, 0.42, 1);

    const line = mm.getLineMaterial(baseProps()) as LineMaterial;
    expect(line.uniforms.uNearCull.value).toBe(0.42);
  });

  it('applies the stored nearCull to a gsplat material created after updateCameraParams', () => {
    const mm = new MaterialManager();
    mm.updateCameraParams(1.0, new THREE.Vector2(800, 600), false, 0.42, 1);

    const gsplat = mm.getGSplatMaterial(baseProps()) as GSplatMaterial;
    expect(gsplat.uniforms.uNearCull.value).toBe(0.42);
  });

  it('applies the stored nearCull to a mesh material created after updateCameraParams', () => {
    // Mesh joined the broadcast with the near fade (#1431). It reads only the two
    // fade inputs of the contract, and the same "created after camera setup" hazard
    // applies: left out, a mesh would fade against the 0.1 default on a scene whose
    // world scale is nothing like it, and only correct itself on the next resize.
    const mm = new MaterialManager();
    mm.updateCameraParams(1.0, new THREE.Vector2(800, 600), true, 0.42, 1);

    const mesh = mm.getMeshMaterial(baseProps({ blendingMode: 'opaque' })) as MeshMaterial;
    expect(mesh.uniforms.uNearCull.value).toBe(0.42);
    expect(mesh.uniforms.uIsOrtho.value).toBe(1);
  });

  it('applies the stored resolution/fov/isOrtho to materials created after updateCameraParams', () => {
    const mm = new MaterialManager();
    mm.updateCameraParams(1.25, new THREE.Vector2(1234, 777), true, 0.5, 1);

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

    const mesh = mm.getMeshMaterial(baseProps({ blendingMode: 'opaque' })) as MeshMaterial;
    expect(mesh.uniforms.uNearCull.value).toBe(0.1);
  });
});
