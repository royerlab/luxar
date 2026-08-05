/**
 * Unit tests for the pure registration helpers in
 * `picking-system/registration.ts`.
 *
 * These cover the dispose + materialManager-unregister paths that the
 * orchestrator's `unregisterNode` / `clearRegistrationsForRebuild`
 * use under the hood. The orchestrator's higher-level tests
 * (`picking-system.test.ts`) exercise these helpers indirectly; this
 * file pins them as pure functions of their arguments.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  disposePickMaterial,
  isEffectivelyVisible,
  unregisterAllPickMaterials,
  type PickNodeEntry,
} from '../../../../../rendering/picking/picking-system/registration';
import { materialManager } from '../../../../../rendering/material-manager';

/**
 * Build a fake CameraAwareMaterial — a real THREE.Material with
 * `updateCameraParams` grafted on so `isCameraAwareMaterial` returns
 * true. Same pattern as the existing context-restore tests in
 * picking-system.test.ts.
 */
function makeCameraAwareMaterial(): THREE.Material {
  const m = new THREE.MeshBasicMaterial();
  (m as unknown as { updateCameraParams: () => void }).updateCameraParams = vi.fn();
  return m;
}

/** Wrap a material (or array) in a pick-mesh entry. */
function makeEntry(material: THREE.Material | THREE.Material[]): PickNodeEntry {
  const pick = new THREE.Mesh(new THREE.BufferGeometry(), material);
  return { main: new THREE.Object3D(), pick };
}

describe('disposePickMaterial', () => {
  it('calls dispose() on a single material', () => {
    const dispose = vi.fn();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), {
      dispose,
    } as unknown as THREE.Material);

    disposePickMaterial(mesh);

    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('calls dispose() on every entry of a material array', () => {
    const a = vi.fn();
    const b = vi.fn();
    const c = vi.fn();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), [
      { dispose: a },
      { dispose: b },
      { dispose: c },
    ] as unknown as THREE.Material[]);

    disposePickMaterial(mesh);

    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
    expect(c).toHaveBeenCalledTimes(1);
  });

  it('is a no-op when a material-array slot is null/undefined', () => {
    const dispose = vi.fn();
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), [
      null,
      { dispose },
      undefined,
    ] as unknown as THREE.Material[]);

    expect(() => disposePickMaterial(mesh)).not.toThrow();
    expect(dispose).toHaveBeenCalledTimes(1);
  });
});

describe('unregisterAllPickMaterials', () => {
  it('unregisters each isCameraAwareMaterial-tagged material from materialManager', () => {
    const a = makeCameraAwareMaterial();
    const b = makeCameraAwareMaterial();
    materialManager.register(a);
    materialManager.register(b);
    const baseline = materialManager.getCacheStats().totalRegistered;

    const nodeMap = new Map<number, PickNodeEntry>();
    nodeMap.set(1, makeEntry(a));
    nodeMap.set(2, makeEntry(b));

    unregisterAllPickMaterials(nodeMap);

    expect(materialManager.getCacheStats().totalRegistered).toBe(baseline - 2);
  });

  it('handles material arrays — every CameraAwareMaterial inside is unregistered', () => {
    const a = makeCameraAwareMaterial();
    const b = makeCameraAwareMaterial();
    materialManager.register(a);
    materialManager.register(b);
    const baseline = materialManager.getCacheStats().totalRegistered;

    const nodeMap = new Map<number, PickNodeEntry>();
    nodeMap.set(1, makeEntry([a, b]));

    unregisterAllPickMaterials(nodeMap);

    expect(materialManager.getCacheStats().totalRegistered).toBe(baseline - 2);
  });

  it('unregisters a non-camera-aware pick material (mesh pick, in staticMaterials)', () => {
    // Plain material without the camera-aware tag stands in for the mesh pick
    // material: register() routes it into staticMaterials (no
    // updateCameraParams). It must STILL be unregistered — otherwise it leaks
    // across every context-restore cycle (issue #1284).
    const plain = new THREE.MeshBasicMaterial();
    materialManager.register(plain);
    const baseline = materialManager.getCacheStats().totalRegistered;

    const nodeMap = new Map<number, PickNodeEntry>();
    nodeMap.set(1, makeEntry(plain));

    unregisterAllPickMaterials(nodeMap);

    expect(materialManager.getCacheStats().totalRegistered).toBe(baseline - 1);
  });

  it('is a no-op for an empty nodeMap', () => {
    const baseline = materialManager.getCacheStats().totalRegistered;
    unregisterAllPickMaterials(new Map());
    expect(materialManager.getCacheStats().totalRegistered).toBe(baseline);
  });
});

describe('isEffectivelyVisible', () => {
  it('returns true for a parentless visible node', () => {
    expect(isEffectivelyVisible(new THREE.Object3D())).toBe(true);
  });

  it('returns false when the node itself is hidden', () => {
    const node = new THREE.Object3D();
    node.visible = false;
    expect(isEffectivelyVisible(node)).toBe(false);
  });

  it('returns false when ANY ancestor is hidden (own flag stays true)', () => {
    // The LOD registry hides the LEVEL object, which can be a group
    // (partition tiles) — the member mesh keeps visible=true.
    const grandparent = new THREE.Group();
    const parent = new THREE.Group();
    const node = new THREE.Object3D();
    grandparent.add(parent);
    parent.add(node);
    grandparent.visible = false;

    expect(node.visible).toBe(true);
    expect(isEffectivelyVisible(node)).toBe(false);
  });

  it('returns true when the whole ancestor chain is visible', () => {
    const parent = new THREE.Group();
    const node = new THREE.Object3D();
    parent.add(node);
    expect(isEffectivelyVisible(node)).toBe(true);
  });
});
