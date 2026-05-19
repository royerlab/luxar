/**
 * Unit tests for camera-framing helpers.
 *
 * Two helpers, two test groups:
 *
 *   - `computeSceneBoundingBox` — verify each rendering shape
 *     (Points, InstancedMesh, Lines / GSplats Mesh+IBG) contributes
 *     bounds and primitive count, and that empty scenes / zero-count
 *     primitives produce an empty box.
 *
 *   - `fitCameraToBounds` — verify perspective and orthographic
 *     camera paths (distance vs zoom), the controls fan-out
 *     (setSceneScale / setTarget / reinitialize / update / saveState),
 *     the zero-extent early return, and the `preserveControlsTarget`
 *     option (matching `autoFrameCamera`'s preserveTarget flag).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  computeSceneBoundingBox,
  fitCameraToBounds,
  ZOOM_RANGE_FACTOR,
} from '../../../../scene/scene-manager/camera-framing';
import type { ControlsManager } from '../../../../controls/controls-manager';
import type { BoundingBox } from '../../../../scene/scene-manager/scene-manager-utils';

function makeControls(focusTarget = new THREE.Vector3(0, 0, 0)): {
  controls: ControlsManager;
  setSceneScale: ReturnType<typeof vi.fn>;
  setDistanceLimits: ReturnType<typeof vi.fn>;
  setZoomLimits: ReturnType<typeof vi.fn>;
  setTarget: ReturnType<typeof vi.fn>;
  reinitialize: ReturnType<typeof vi.fn>;
  update: ReturnType<typeof vi.fn>;
  saveState: ReturnType<typeof vi.fn>;
} {
  const setSceneScale = vi.fn();
  const setDistanceLimits = vi.fn();
  const setZoomLimits = vi.fn();
  const setTarget = vi.fn();
  const reinitialize = vi.fn();
  const update = vi.fn();
  const saveState = vi.fn();
  const getFocusTarget = vi.fn(() => focusTarget);
  const controls = {
    setSceneScale,
    setDistanceLimits,
    setZoomLimits,
    setTarget,
    reinitialize,
    update,
    saveState,
    getFocusTarget,
  } as unknown as ControlsManager;
  return {
    controls,
    setSceneScale,
    setDistanceLimits,
    setZoomLimits,
    setTarget,
    reinitialize,
    update,
    saveState,
  };
}

function makeBounds(min: [number, number, number], max: [number, number, number]): BoundingBox {
  return {
    min: { x: min[0], y: min[1], z: min[2] },
    max: { x: max[0], y: max[1], z: max[2] },
  };
}

describe('computeSceneBoundingBox', () => {
  it('returns an empty box and zero primitive count for an empty scene', () => {
    const scene = new THREE.Scene();
    const result = computeSceneBoundingBox(scene);
    expect(result.box.isEmpty()).toBe(true);
    expect(result.primitiveCount).toBe(0);
  });

  it('aggregates Points bounds and counts instances as primitives', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.InstancedBufferGeometry();
    // Quad template — the geometry's bounding box reflects the
    // per-point world extents the loader stamps in via
    // commit-points-geometry. We stamp them manually here.
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0]), 3)
    );
    geometry.setAttribute(
      'aCenter',
      new THREE.InstancedBufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1, -1, 0, 2]), 3)
    );
    geometry.instanceCount = 3;
    geometry.boundingBox = new THREE.Box3(new THREE.Vector3(-1, 0, 0), new THREE.Vector3(1, 1, 2));
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    mesh.userData.nodeType = 'points';
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    const result = computeSceneBoundingBox(scene);
    expect(result.box.isEmpty()).toBe(false);
    expect(result.primitiveCount).toBe(3);
    expect(result.box.min.x).toBeCloseTo(-1, 6);
    expect(result.box.max.z).toBeCloseTo(2, 6);
  });

  it('aggregates InstancedMesh bounds and counts via mesh.count', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.BoxGeometry(2, 2, 2);
    const material = new THREE.MeshBasicMaterial();
    const mesh = new THREE.InstancedMesh(geometry, material, 5);
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    const result = computeSceneBoundingBox(scene);
    expect(result.primitiveCount).toBe(5);
    expect(result.box.isEmpty()).toBe(false);
  });

  it('handles Lines (Mesh + InstancedBufferGeometry, nodeType=lines)', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1]), 3)
    );
    geometry.instanceCount = 7;
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    mesh.userData.nodeType = 'lines';
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    const result = computeSceneBoundingBox(scene);
    expect(result.primitiveCount).toBe(7);
  });

  it('handles GSplats (Mesh + InstancedBufferGeometry, nodeType=gsplats)', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 1, 1]), 3)
    );
    geometry.instanceCount = 11;
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    mesh.userData.nodeType = 'gsplats';
    scene.add(mesh);
    scene.updateMatrixWorld(true);

    const result = computeSceneBoundingBox(scene);
    expect(result.primitiveCount).toBe(11);
  });

  it('skips zero-count Points (no instances to bound)', () => {
    const scene = new THREE.Scene();
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, 1, 0]), 3)
    );
    geometry.instanceCount = 0;
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
    mesh.userData.nodeType = 'points';
    scene.add(mesh);

    const result = computeSceneBoundingBox(scene);
    expect(result.box.isEmpty()).toBe(true);
    expect(result.primitiveCount).toBe(0);
  });

  it('combines bounds from multiple primitives in the same scene', () => {
    const scene = new THREE.Scene();
    // Points mesh near origin
    const g1 = new THREE.InstancedBufferGeometry();
    g1.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, 1, 0]), 3)
    );
    g1.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array([0, 0, 0]), 3));
    g1.instanceCount = 1;
    g1.boundingBox = new THREE.Box3(new THREE.Vector3(-1, -1, 0), new THREE.Vector3(1, 1, 0));
    const pointsMesh = new THREE.Mesh(g1, new THREE.MeshBasicMaterial());
    pointsMesh.userData.nodeType = 'points';
    scene.add(pointsMesh);
    // InstancedMesh far away
    const g2 = new THREE.BoxGeometry(1, 1, 1);
    const im = new THREE.InstancedMesh(g2, new THREE.MeshBasicMaterial(), 1);
    im.position.set(10, 10, 10);
    scene.add(im);
    scene.updateMatrixWorld(true);

    const result = computeSceneBoundingBox(scene);
    expect(result.box.min.x).toBeLessThanOrEqual(0);
    expect(result.box.max.x).toBeGreaterThanOrEqual(10);
    expect(result.primitiveCount).toBe(2); // 1 point + 1 instance
  });
});

describe('fitCameraToBounds', () => {
  let perspectiveCamera: THREE.PerspectiveCamera;

  beforeEach(() => {
    perspectiveCamera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  });

  it('returns 0 and short-circuits on a zero-extent box', () => {
    const { controls, setSceneScale } = makeControls();
    const result = fitCameraToBounds(
      perspectiveCamera,
      controls,
      makeBounds([0, 0, 0], [0, 0, 0]),
      {
        lookAtTarget: new THREE.Vector3(),
      }
    );
    expect(result).toBe(0);
    expect(setSceneScale).not.toHaveBeenCalled();
  });

  it('perspective: positions camera at target + (0, 0, distance), sets distance limits', () => {
    const {
      controls,
      setSceneScale,
      setDistanceLimits,
      setTarget,
      reinitialize,
      update,
      saveState,
    } = makeControls();
    const target = new THREE.Vector3(5, 0, 0);
    const diagonal = fitCameraToBounds(
      perspectiveCamera,
      controls,
      makeBounds([0, -1, -1], [10, 1, 1]),
      { lookAtTarget: target }
    );
    expect(diagonal).toBeGreaterThan(0);
    expect(setSceneScale).toHaveBeenCalledTimes(1);
    expect(perspectiveCamera.position.x).toBe(5);
    expect(perspectiveCamera.position.y).toBe(0);
    expect(perspectiveCamera.position.z).toBeGreaterThan(0); // some positive distance from origin
    expect(setDistanceLimits).toHaveBeenCalledTimes(1);
    const [near, far] = setDistanceLimits.mock.calls[0];
    expect(far / near).toBeCloseTo(ZOOM_RANGE_FACTOR ** 2, 6);
    expect(setTarget).toHaveBeenCalledTimes(1);
    expect(reinitialize).toHaveBeenCalledTimes(1);
    expect(update).toHaveBeenCalledTimes(1);
    expect(saveState).toHaveBeenCalledTimes(1);
  });

  it('orthographic: computes zoom from frustum + sizes, sets zoom limits', () => {
    const ortho = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
    const { controls, setZoomLimits, setDistanceLimits } = makeControls();
    const diagonal = fitCameraToBounds(ortho, controls, makeBounds([0, 0, 0], [4, 4, 4]), {
      lookAtTarget: new THREE.Vector3(2, 2, 2),
    });
    expect(diagonal).toBeGreaterThan(0);
    expect(setZoomLimits).toHaveBeenCalledTimes(1);
    expect(setDistanceLimits).not.toHaveBeenCalled(); // ortho path uses zoom, not distance
  });

  it('preserveControlsTarget=true suppresses controls.setTarget but still reinitializes', () => {
    const { controls, setTarget, reinitialize } = makeControls();
    fitCameraToBounds(perspectiveCamera, controls, makeBounds([0, 0, 0], [1, 1, 1]), {
      lookAtTarget: new THREE.Vector3(),
      preserveControlsTarget: true,
    });
    expect(setTarget).not.toHaveBeenCalled();
    expect(reinitialize).toHaveBeenCalledTimes(1);
  });
});
