/**
 * Unit tests for PickingSystem.
 *
 * The class wraps a WebGLRenderTarget and depends on
 * THREE.WebGLRenderer for the actual readRenderTargetPixels
 * / getDrawingBufferSize / domElement.getBoundingClientRect calls. Under
 * jsdom we can:
 *   - stub the renderer with the bare-minimum methods used outside
 *     performPick()
 *   - construct a real WebGLRenderTarget (its constructor is pure JS;
 *     no GL context needed)
 *
 * Tests focus on the registration/lifecycle surface and the exported
 * pure helpers (computePickBufferSize / MAX_PICK_BUFFER_DIM). The
 * actual pick-render path needs a real WebGL context — out of scope.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  PickingSystem,
  computePickBufferSize,
  MAX_PICK_BUFFER_DIM,
} from '../../../../rendering/picking/picking-system';

function makeStubRenderer(): THREE.WebGLRenderer {
  // PickingSystem ctor + the lifecycle methods we test only need
  // `renderer` as an opaque reference. domElement is read by
  // onMouseMove via getBoundingClientRect; not exercised here.
  return {
    domElement: document.createElement('canvas'),
    getDrawingBufferSize: vi.fn(),
    readRenderTargetPixels: vi.fn(),
  } as unknown as THREE.WebGLRenderer;
}

function makeCamera(): THREE.PerspectiveCamera {
  return new THREE.PerspectiveCamera(60, 1.0, 0.1, 1000);
}

describe('computePickBufferSize', () => {
  it('returns half the drawing-buffer size', () => {
    expect(computePickBufferSize(1920, 1080)).toEqual({ w: 960, h: 540 });
  });

  it('clamps both axes to MAX_PICK_BUFFER_DIM', () => {
    // 4K wide → /2 = 1920 → clamp to MAX (1024)
    expect(computePickBufferSize(3840, 2160)).toEqual({
      w: MAX_PICK_BUFFER_DIM,
      h: MAX_PICK_BUFFER_DIM,
    });
  });

  it('floors odd input to integer pixels', () => {
    expect(computePickBufferSize(101, 99)).toEqual({ w: 50, h: 49 });
  });

  it('returns at least 1 pixel per axis even for tiny inputs', () => {
    expect(computePickBufferSize(0, 0)).toEqual({ w: 1, h: 1 });
    expect(computePickBufferSize(1, 1)).toEqual({ w: 1, h: 1 });
  });

  it('handles asymmetric clamping (one axis under cap, other over)', () => {
    expect(computePickBufferSize(1500, 5000)).toEqual({
      w: 750,
      h: MAX_PICK_BUFFER_DIM,
    });
  });
});

describe('PickingSystem — registration', () => {
  let system: PickingSystem;
  const onPickResult = vi.fn();

  beforeEach(() => {
    system = new PickingSystem(makeStubRenderer(), makeCamera(), onPickResult);
    onPickResult.mockClear();
  });

  it('starts with zero registered nodes', () => {
    expect(system.registeredNodeCount).toBe(0);
  });

  it('allocatePickId returns sequentially starting at 1', () => {
    expect(system.allocatePickId()).toBe(1);
    expect(system.allocatePickId()).toBe(2);
    expect(system.allocatePickId()).toBe(3);
  });

  it('registerNode increments registeredNodeCount', () => {
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id);
    expect(system.registeredNodeCount).toBe(1);
  });

  it('registerNode disables matrixAutoUpdate on the pick node', () => {
    const pickNode = new THREE.Object3D();
    system.registerNode(new THREE.Object3D(), pickNode, system.allocatePickId());
    expect(pickNode.matrixAutoUpdate).toBe(false);
    expect(pickNode.matrixWorldAutoUpdate).toBe(false);
  });

  it('unregisterNode decrements registeredNodeCount', () => {
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id);
    expect(system.registeredNodeCount).toBe(1);

    system.unregisterNode(id);
    expect(system.registeredNodeCount).toBe(0);
  });

  it('unregisterNode disposes the pick mesh material', () => {
    const dispose = vi.fn();
    const pickMesh = new THREE.Mesh(
      new THREE.BufferGeometry(),
      { dispose } as unknown as THREE.Material
    );
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), pickMesh, id);

    system.unregisterNode(id);
    expect(dispose).toHaveBeenCalled();
  });

  it('unregisterNode disposes every entry in a material array', () => {
    const a = vi.fn();
    const b = vi.fn();
    const pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), [
      { dispose: a },
      { dispose: b },
    ] as unknown as THREE.Material[]);
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), pickMesh, id);

    system.unregisterNode(id);
    expect(a).toHaveBeenCalled();
    expect(b).toHaveBeenCalled();
  });

  it('unregisterNode for an unknown ID is a no-op', () => {
    expect(() => system.unregisterNode(9999)).not.toThrow();
    expect(system.registeredNodeCount).toBe(0);
  });
});

describe('PickingSystem — context-restore registration drop', () => {
  let system: PickingSystem;

  beforeEach(() => {
    system = new PickingSystem(makeStubRenderer(), makeCamera(), vi.fn());
  });

  it('clearRegistrationsForRebuild empties the node map without disposing materials', () => {
    const dispose = vi.fn();
    const pickMesh = new THREE.Mesh(
      new THREE.BufferGeometry(),
      { dispose } as unknown as THREE.Material
    );
    system.registerNode(
      new THREE.Object3D(),
      pickMesh,
      system.allocatePickId()
    );
    expect(system.registeredNodeCount).toBe(1);

    system.clearRegistrationsForRebuild();
    expect(system.registeredNodeCount).toBe(0);
    // Material dispose must NOT be called — its shader is already
    // invalid in a context-loss scenario; calling dispose can throw
    // on some drivers.
    expect(dispose).not.toHaveBeenCalled();
  });

  it('clearRegistrationsForRebuild unregisters CameraAwareMaterial from materialManager', async () => {
    // Pre-fix: pick materials registered with materialManager via
    // node-factory's `materialManager.register(pickMaterial)`. After
    // context-restore, clearRegistrationsForRebuild dropped the
    // node map but left the registry entries. Repeated cycles
    // accumulated stale references; camera-uniform updates would
    // target dead materials and `getCacheStats().totalRegistered`
    // would grow unboundedly.
    const { materialManager } = await import(
      '../../../../rendering/material-manager'
    );

    // Use a real THREE.Material so EventDispatcher is wired (the
    // materialManager subscribes to the 'dispose' event), then graft
    // `updateCameraParams` onto it so isCameraAwareMaterial returns
    // true.
    const baseMaterial = new THREE.MeshBasicMaterial();
    (baseMaterial as unknown as { updateCameraParams: () => void }).updateCameraParams = vi.fn();
    const disposeSpy = vi.spyOn(baseMaterial, 'dispose');
    const pickMaterial = baseMaterial as unknown as THREE.Material;

    // Match what node-factory does: register the pick material with
    // materialManager at construction.
    const before = materialManager.getCacheStats().totalRegistered;
    materialManager.register(
      pickMaterial as unknown as Parameters<typeof materialManager.register>[0]
    );
    expect(materialManager.getCacheStats().totalRegistered).toBe(before + 1);

    const pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), pickMaterial);
    system.registerNode(new THREE.Object3D(), pickMesh, system.allocatePickId());

    system.clearRegistrationsForRebuild();

    // pick material is now unregistered from
    // materialManager — count returns to baseline.
    expect(materialManager.getCacheStats().totalRegistered).toBe(before);
    // Material's own dispose() was NOT called — the shader is
    // already invalid in a context-restore scenario.
    expect(disposeSpy).not.toHaveBeenCalled();
    disposeSpy.mockRestore();
  });
});

describe('PickingSystem — camera + suppression', () => {
  let system: PickingSystem;

  beforeEach(() => {
    system = new PickingSystem(makeStubRenderer(), makeCamera(), vi.fn());
  });

  it('setCamera replaces the camera reference (no throw)', () => {
    expect(() => system.setCamera(makeCamera())).not.toThrow();
  });

  it('setPostProcessing accepts null (no throw)', () => {
    expect(() => system.setPostProcessing(null)).not.toThrow();
  });

  it('markDirty does not throw when called outside a frame', () => {
    expect(() => system.markDirty()).not.toThrow();
  });

  it('suppress(true) clears any pending debounce timer', () => {
    expect(() => system.suppress(true)).not.toThrow();
    expect(() => system.suppress(false)).not.toThrow();
  });
});

describe('PickingSystem.dispose', () => {
  it('disposes the underlying pick render target', () => {
    const system = new PickingSystem(makeStubRenderer(), makeCamera(), vi.fn());
    expect(() => system.dispose()).not.toThrow();
  });

  it('double dispose is safe', () => {
    const system = new PickingSystem(makeStubRenderer(), makeCamera(), vi.fn());
    system.dispose();
    expect(() => system.dispose()).not.toThrow();
  });
});
