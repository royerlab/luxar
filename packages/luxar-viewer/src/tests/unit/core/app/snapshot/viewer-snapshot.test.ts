import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import * as THREE from 'three';
import {
  captureSnapshot,
  restoreSnapshot,
  VIEWER_SNAPSHOT_VERSION,
  type ViewerSnapshot,
} from '../../../../../core/app/snapshot/viewer-snapshot';
import { sceneDimsManager } from '../../../../../scene/scene-dims-manager';

interface FakeControls {
  getFocusTarget: () => THREE.Vector3;
  setTarget: (v: THREE.Vector3) => void;
  reinitialize: () => void;
  dispatchEvent: Mock<(event: { type: string }) => void>;
}

interface FakeSceneManager {
  camera: THREE.Camera;
  controls: FakeControls;
  commitCameraChange: ReturnType<typeof vi.fn>;
}

/** Like production: run the write, sync matrices, publish ONE controls `change`. */
function withCommit(camera: THREE.Camera, controls: FakeControls): FakeSceneManager {
  return {
    camera,
    controls,
    commitCameraChange: vi.fn((write?: () => void) => {
      write?.();
      camera.updateMatrixWorld();
      controls.dispatchEvent({ type: 'change' });
    }),
  };
}

function makePerspectiveSceneManager(): FakeSceneManager {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(10, 20, 30);
  camera.up.set(0, 1, 0);
  const target = new THREE.Vector3(1, 2, 3);
  return withCommit(camera, {
    getFocusTarget: () => target.clone(),
    setTarget: (v) => target.copy(v),
    reinitialize: vi.fn(),
    dispatchEvent: vi.fn(),
  });
}

function makeOrthoSceneManager(): FakeSceneManager {
  const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 100);
  camera.position.set(0, 0, 50);
  camera.up.set(0, 1, 0);
  camera.zoom = 2.0;
  const target = new THREE.Vector3(0, 0, 0);
  return withCommit(camera, {
    getFocusTarget: () => target.clone(),
    setTarget: (v) => target.copy(v),
    reinitialize: vi.fn(),
    dispatchEvent: vi.fn(),
  });
}

function loadDimsScene(ndim: number, currentStep: number[]): THREE.Scene {
  const scene = new THREE.Scene();
  const dimensions = Array.from({ length: ndim }, (_, i) => ({
    name: `d${i}`,
    unit: '',
    range: [0, 10] as [number, number],
    step: 1,
    display: i < 3,
  }));
  scene.userData.sceneDimensions = { dimensions };
  sceneDimsManager.initFromScene(scene);
  // Override the auto-init currentStep with the values we want.
  const dims = sceneDimsManager.getDims();
  expect(dims).not.toBeNull();
  for (let i = 0; i < currentStep.length; i++) {
    sceneDimsManager.setDimensionValue(i, currentStep[i]);
  }
  return scene;
}

describe('captureSnapshot', () => {
  beforeEach(() => {
    sceneDimsManager.reset();
  });

  it('captures perspective camera state', () => {
    const sm = makePerspectiveSceneManager();
    const snap = captureSnapshot(sm as unknown as Parameters<typeof captureSnapshot>[0]);

    expect(snap.version).toBe(VIEWER_SNAPSHOT_VERSION);
    expect(snap.camera.position).toEqual([10, 20, 30]);
    expect(snap.camera.target).toEqual([1, 2, 3]);
    expect(snap.camera.up).toEqual([0, 1, 0]);
    expect(snap.camera.isOrtho).toBe(false);
    expect(snap.camera.fov).toBe(60);
    expect(snap.camera.zoom).toBeUndefined();
  });

  it('captures orthographic camera state', () => {
    const sm = makeOrthoSceneManager();
    const snap = captureSnapshot(sm as unknown as Parameters<typeof captureSnapshot>[0]);

    expect(snap.camera.isOrtho).toBe(true);
    expect(snap.camera.zoom).toBe(2.0);
    expect(snap.camera.fov).toBeUndefined();
  });

  it('omits dims block when no scene is loaded', () => {
    const sm = makePerspectiveSceneManager();
    const snap = captureSnapshot(sm as unknown as Parameters<typeof captureSnapshot>[0]);
    expect(snap.dims).toBeUndefined();
  });

  it('captures dims when a scene is loaded', () => {
    loadDimsScene(5, [0, 0, 0, 4, 2]);
    const sm = makePerspectiveSceneManager();
    const snap = captureSnapshot(sm as unknown as Parameters<typeof captureSnapshot>[0]);

    expect(snap.dims).toBeDefined();
    expect(snap.dims?.ndim).toBe(5);
    expect(snap.dims?.displayed.length).toBeLessThanOrEqual(3);
    expect(snap.dims?.currentStep[3]).toBe(4);
    expect(snap.dims?.currentStep[4]).toBe(2);
  });

  it('produces a JSON-serialisable object', () => {
    loadDimsScene(4, [0, 0, 0, 3]);
    const sm = makePerspectiveSceneManager();
    const snap = captureSnapshot(sm as unknown as Parameters<typeof captureSnapshot>[0]);

    const json = JSON.stringify(snap);
    const round = JSON.parse(json);
    expect(round.version).toBe(VIEWER_SNAPSHOT_VERSION);
    expect(round.camera.position).toEqual([10, 20, 30]);
    expect(round.dims.currentStep[3]).toBe(3);
  });
});

describe('restoreSnapshot', () => {
  beforeEach(() => {
    sceneDimsManager.reset();
  });

  it('leaves near/far to the per-frame updater while dynamic clipping is on', () => {
    // A pose's planes describe the distance it was captured at; under dynamic
    // clipping writing them clipped geometry until the next frame's update.
    const sm = makePerspectiveSceneManager() as FakeSceneManager & {
      getDynamicClippingState: () => { enabled: boolean; near: number; far: number };
    };
    sm.getDynamicClippingState = () => ({ enabled: true, near: 0.1, far: 1000 });
    const snapshot: ViewerSnapshot = {
      version: VIEWER_SNAPSHOT_VERSION,
      camera: {
        position: [100, 200, 300],
        target: [9, 8, 7],
        up: [0, 1, 0],
        isOrtho: false,
        fov: 45,
        near: 0.5,
        far: 5000,
      },
    };

    restoreSnapshot(sm as unknown as Parameters<typeof restoreSnapshot>[0], snapshot);

    const persp = sm.camera as THREE.PerspectiveCamera;
    expect(persp.near).toBe(0.1);
    expect(persp.far).toBe(1000);
    // Everything else still applies.
    expect(sm.camera.position.toArray()).toEqual([100, 200, 300]);
    expect(persp.fov).toBe(45);
  });

  it('writes camera position, target, up, near, far back to camera', () => {
    const sm = makePerspectiveSceneManager();
    const snapshot: ViewerSnapshot = {
      version: VIEWER_SNAPSHOT_VERSION,
      camera: {
        position: [100, 200, 300],
        target: [9, 8, 7],
        up: [0, 0, 1],
        isOrtho: false,
        fov: 45,
        near: 0.5,
        far: 5000,
      },
    };

    const result = restoreSnapshot(
      sm as unknown as Parameters<typeof restoreSnapshot>[0],
      snapshot
    );

    expect(result.cameraApplied).toBe(true);
    expect(result.dimsApplied).toBe(false);
    expect(sm.camera.position.toArray()).toEqual([100, 200, 300]);
    expect(sm.camera.up.toArray()).toEqual([0, 0, 1]);
    const persp = sm.camera as THREE.PerspectiveCamera;
    expect(persp.near).toBe(0.5);
    expect(persp.far).toBe(5000);
    expect(persp.fov).toBe(45);
    expect(sm.controls.getFocusTarget().toArray()).toEqual([9, 8, 7]);
    expect(sm.controls.reinitialize).toHaveBeenCalledOnce();
    // The programmatic path must fire the same CONTROLS 'change' event an
    // interactive camera move produces — it wakes the render loop (per-frame
    // LOD evaluation sees the new pose), refreshes ortho materials, and
    // dirties the picking system (regression: setCameraPose left the
    // previous LOD level pinned).
    expect(sm.commitCameraChange).toHaveBeenCalledOnce();
    expect(sm.controls.dispatchEvent).toHaveBeenCalledWith({ type: 'change' });
  });

  it('restores ortho-specific zoom when projection is ortho', () => {
    const sm = makeOrthoSceneManager();
    const snapshot: ViewerSnapshot = {
      version: VIEWER_SNAPSHOT_VERSION,
      camera: {
        position: [0, 0, 50],
        target: [0, 0, 0],
        up: [0, 1, 0],
        isOrtho: true,
        zoom: 5.5,
        near: 0.1,
        far: 100,
      },
    };

    restoreSnapshot(sm as unknown as Parameters<typeof restoreSnapshot>[0], snapshot);

    expect((sm.camera as THREE.OrthographicCamera).zoom).toBe(5.5);
  });

  it('skips dims when ndim mismatches the loaded scene', () => {
    // core.md W11 strengthening: previously only asserted dimsApplied===false.
    // Three orthogonal contracts to assert on the ndim-mismatch path:
    //   1. result.dimsApplied is false (skip happened).
    //   2. result.cameraApplied is TRUE — camera restore is independent
    //      of the dims-block decision (mutation guard: a refactor that
    //      bailed early on dim mismatch would break the camera path).
    //   3. sceneDimsManager.currentStep is UNCHANGED by the failed restore
    //      (the dim-3 value remains 1, not 0).
    loadDimsScene(4, [0, 0, 0, 1]);
    const sm = makePerspectiveSceneManager();
    const spy = vi.spyOn(sceneDimsManager, 'setDimensionValue');

    const snapshot: ViewerSnapshot = {
      version: VIEWER_SNAPSHOT_VERSION,
      camera: {
        position: [42, 0, 0],
        target: [0, 0, 0],
        up: [0, 1, 0],
        isOrtho: false,
        near: 0.1,
        far: 1000,
      },
      dims: {
        ndim: 5, // Mismatch — loaded scene is 4D.
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 0, 0],
      },
    };

    const result = restoreSnapshot(
      sm as unknown as Parameters<typeof restoreSnapshot>[0],
      snapshot
    );
    expect(result.dimsApplied).toBe(false);
    expect(result.cameraApplied).toBe(true); // camera still applied
    expect(sm.camera.position.x).toBe(42); // camera DID move
    // No per-dim writes were attempted.
    expect(spy).not.toHaveBeenCalled();
    // Existing currentStep is preserved exactly.
    const dims = sceneDimsManager.getDims();
    expect(dims?.currentStep[3]).toBe(1);

    spy.mockRestore();
  });

  it('forwards currentStep values to sceneDimsManager when ndim matches', () => {
    loadDimsScene(4, [0, 0, 0, 1]);
    const sm = makePerspectiveSceneManager();

    const snapshot: ViewerSnapshot = {
      version: VIEWER_SNAPSHOT_VERSION,
      camera: {
        position: [0, 0, 0],
        target: [0, 0, 0],
        up: [0, 1, 0],
        isOrtho: false,
        near: 0.1,
        far: 1000,
      },
      dims: {
        ndim: 4,
        displayed: [0, 1, 2],
        currentStep: [0, 0, 0, 7],
      },
    };

    const result = restoreSnapshot(
      sm as unknown as Parameters<typeof restoreSnapshot>[0],
      snapshot
    );
    expect(result.dimsApplied).toBe(true);
    const dims = sceneDimsManager.getDims()!;
    expect(dims.currentStep[3]).toBe(7);
  });

  it('returns false for both when version mismatches', () => {
    const sm = makePerspectiveSceneManager();
    const originalPos = sm.camera.position.clone();

    const snapshot = {
      version: 99 as unknown as 1,
      camera: {
        position: [1, 1, 1] as [number, number, number],
        target: [0, 0, 0] as [number, number, number],
        up: [0, 1, 0] as [number, number, number],
        isOrtho: false,
        near: 0.1,
        far: 1000,
      },
    } as ViewerSnapshot;

    const result = restoreSnapshot(
      sm as unknown as Parameters<typeof restoreSnapshot>[0],
      snapshot
    );
    expect(result.cameraApplied).toBe(false);
    expect(result.dimsApplied).toBe(false);
    // Camera unchanged.
    expect(sm.camera.position.toArray()).toEqual(originalPos.toArray());
  });

  it('round-trips: capture then restore produces an identical snapshot', () => {
    loadDimsScene(4, [0, 0, 0, 5]);
    const sm = makePerspectiveSceneManager();
    const before = captureSnapshot(sm as unknown as Parameters<typeof captureSnapshot>[0]);

    const sm2 = makePerspectiveSceneManager();
    sm2.camera.position.set(0, 0, 0);
    restoreSnapshot(sm2 as unknown as Parameters<typeof restoreSnapshot>[0], before);
    const after = captureSnapshot(sm2 as unknown as Parameters<typeof captureSnapshot>[0]);

    expect(after.camera.position).toEqual(before.camera.position);
    expect(after.camera.target).toEqual(before.camera.target);
    expect(after.camera.up).toEqual(before.camera.up);
    expect(after.camera.fov).toEqual(before.camera.fov);
    expect(after.dims?.currentStep).toEqual(before.dims?.currentStep);
  });
});
