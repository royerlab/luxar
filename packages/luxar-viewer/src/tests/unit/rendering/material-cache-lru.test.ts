/**
 * Unit tests for the PER-NODE line-material contract on MaterialManager.
 *
 * The historical line-material LRU (the last cached material kind) died
 * with the lines texture-storage migration: line materials now carry a
 * per-node `uLineTex` texture uniform, so two nodes can never share a
 * material — sharing would rebind one node's texture onto another's
 * mesh at every commit. `getLineMaterial` therefore creates a fresh
 * material on EVERY call (mirroring `getPointMaterial` /
 * `getGSplatMaterial`), and no material cache exists at all.
 *
 * This suite pins that contract: distinct instances per call, immediate
 * + ongoing camera-param registration for every created material, one
 * construction per call across many creations, and dispose() reaching
 * every per-node material through the registry.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  MaterialManager,
  __resetMaterialManagerForTests,
} from '../../../rendering/material-manager';

const baseProps = (over: Record<string, number | boolean | string> = {}) =>
  ({
    opacity: 1.0,
    gamma: 1.0,
    intensity: 1.0,
    offset: 0.0,
    blendingMode: 'additive',
    ...over,
  }) as Parameters<MaterialManager['getLineMaterial']>[0];

describe('MaterialManager per-node line materials (no LRU cache)', () => {
  beforeEach(() => {
    __resetMaterialManagerForTests();
  });

  it('returns a DISTINCT instance per call, even for identical props (per-node)', () => {
    const mm = new MaterialManager();
    const props = baseProps({ opacity: 0.42 });
    const m1 = mm.getLineMaterial(props);
    const m2 = mm.getLineMaterial(props);
    const m3 = mm.getLineMaterial(props);
    expect(m2).not.toBe(m1);
    expect(m3).not.toBe(m1);
    expect(m3).not.toBe(m2);
  });

  it('per-node materials receive the current camera params immediately at creation', () => {
    const mm = new MaterialManager();
    const fov = Math.PI / 4;
    mm.updateCameraParams(fov, new THREE.Vector2(2560, 1440), false, 0.33);
    const material = mm.getLineMaterial(baseProps());
    // updateCameraParams flowed inside getLineMaterial — the material
    // starts life with the manager's current resolution/FOV/nearCull,
    // not the constructor defaults.
    expect(material.uniforms.uResolution.value.x).toBe(2560);
    expect(material.uniforms.uResolution.value.y).toBe(1440);
    expect(material.uniforms.uNearCull.value).toBe(0.33);
    const expectedScale = 1440 / Math.max(Math.tan(fov * 0.5), 1e-4);
    expect(material.uniforms.uPerspectiveLineScale.value).toBeCloseTo(expectedScale, 5);
  });

  it('EVERY created material is registered and keeps receiving camera broadcasts', () => {
    const mm = new MaterialManager();
    const materials = [
      mm.getLineMaterial(baseProps({ opacity: 0.1 })),
      mm.getLineMaterial(baseProps({ opacity: 0.1 })), // identical props — still its own registration
      mm.getLineMaterial(baseProps({ opacity: 0.9 })),
    ];
    expect(mm.getCacheStats().totalRegistered).toBe(3);

    mm.updateCameraParams(0.9, new THREE.Vector2(640, 480), false, 0.25);
    for (const m of materials) {
      expect(m.uniforms.uResolution.value.x).toBe(640);
      expect(m.uniforms.uResolution.value.y).toBe(480);
      expect(m.uniforms.uNearCull.value).toBe(0.25);
    }
  });

  it('constructs one material per call and registers every one of them', () => {
    // The observable consequence of having no cache: N calls means N
    // constructions and N registrations, never fewer.
    const mm = new MaterialManager();
    for (let i = 0; i < 25; i++) {
      mm.getLineMaterial(baseProps({ opacity: i / 25 }));
    }
    const stats = mm.getCacheStats();
    expect(stats.createCount).toBe(25);
    expect(stats.totalRegistered).toBe(25);
  });

  it('every material kind is per node, including points and gsplats', () => {
    // All three kinds carry their own element texture, so no creation
    // path may return a shared instance — each call adds a registration.
    const mm = new MaterialManager();
    mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    mm.getPointMaterial(baseProps({ opacity: 0.1 }));
    mm.getPointMaterial(baseProps({ opacity: 0.2 }));
    mm.getGSplatMaterial(baseProps({ opacity: 0.3 }));
    const stats = mm.getCacheStats();
    expect(stats.createCount).toBe(4);
    expect(stats.totalRegistered).toBe(4);
  });

  it('manager dispose() reaches every per-node material through the registry', () => {
    // Per-node materials sit in NO cache, so the registry is the ONLY
    // path teardown has to them — a registry miss would leak the GPU
    // program on embedder re-init.
    const mm = new MaterialManager();
    const a = mm.getLineMaterial(baseProps({ opacity: 0.1 }));
    const b = mm.getLineMaterial(baseProps({ opacity: 0.2 }));
    let aDisposed = false;
    let bDisposed = false;
    const aOrig = a.dispose.bind(a);
    const bOrig = b.dispose.bind(b);
    a.dispose = () => {
      aDisposed = true;
      aOrig();
    };
    b.dispose = () => {
      bDisposed = true;
      bOrig();
    };
    mm.dispose();
    expect(aDisposed).toBe(true);
    expect(bDisposed).toBe(true);
    expect(mm.getCacheStats().totalRegistered).toBe(0);
  });
});
