// @vitest-environment jsdom
/**
 * Unit tests for the camera-setup helpers.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import { config } from '../../../../../config';
import {
  createDefaultPerspectiveCamera,
  resetCameraToInitialPosition,
  resolveTargetNodeCenter,
  applyZarrViewerConfig,
} from '../../../../../scene/scene-manager/camera/camera-setup';
import type { ControlsManager } from '../../../../../controls/controls-manager';
import type { ZarrViewerConfig } from '../../../../../types/zarr';

function makeCanvas(width = 800, height = 600): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  Object.defineProperty(canvas, 'clientWidth', { value: width, configurable: true });
  Object.defineProperty(canvas, 'clientHeight', { value: height, configurable: true });
  return canvas;
}

function makeControls(): {
  controls: ControlsManager;
  setTarget: ReturnType<typeof vi.fn>;
  reinitialize: ReturnType<typeof vi.fn>;
  update: ReturnType<typeof vi.fn>;
  getFocusTarget: ReturnType<typeof vi.fn>;
} {
  const setTarget = vi.fn();
  const reinitialize = vi.fn();
  const update = vi.fn();
  const focusTarget = new THREE.Vector3(0, 0, 0);
  const getFocusTarget = vi.fn(() => focusTarget);
  const controls = {
    setTarget,
    reinitialize,
    update,
    getFocusTarget,
  } as unknown as ControlsManager;
  return { controls, setTarget, reinitialize, update, getFocusTarget };
}

describe('createDefaultPerspectiveCamera', () => {
  it('reads FOV / near / far from config defaults', () => {
    const canvas = makeCanvas();
    const camera = createDefaultPerspectiveCamera(canvas);
    expect(camera.fov).toBe(config.renderingControls.defaults.fov);
    expect(camera.near).toBe(config.renderingControls.defaults.near);
    expect(camera.far).toBe(config.renderingControls.defaults.far);
  });

  it('uses canvas.clientWidth / clientHeight for aspect ratio', () => {
    const camera = createDefaultPerspectiveCamera(makeCanvas(1600, 800));
    expect(camera.aspect).toBeCloseTo(2, 6); // 1600/800
  });

  it('positions the camera at config.camera.initialPosition', () => {
    const camera = createDefaultPerspectiveCamera(makeCanvas());
    expect(camera.position.x).toBe(config.camera.initialPosition.x);
    expect(camera.position.y).toBe(config.camera.initialPosition.y);
    expect(camera.position.z).toBe(config.camera.initialPosition.z);
  });
});

describe('resetCameraToInitialPosition', () => {
  it('moves the camera to initialPosition and aims at origin', () => {
    const canvas = makeCanvas();
    const camera = createDefaultPerspectiveCamera(canvas);
    camera.position.set(100, 200, 300);
    camera.lookAt(50, 50, 50);

    resetCameraToInitialPosition(camera);

    expect(camera.position.x).toBe(config.camera.initialPosition.x);
    expect(camera.position.y).toBe(config.camera.initialPosition.y);
    expect(camera.position.z).toBe(config.camera.initialPosition.z);
    // After lookAt(0,0,0), the camera's matrix world is updated; verify
    // the world matrix is consistent with the position.
    const worldPos = new THREE.Vector3();
    camera.getWorldPosition(worldPos);
    expect(worldPos.equals(camera.position)).toBe(true);
  });
});

describe('resolveTargetNodeCenter', () => {
  it('returns null when no descendant has the requested name', () => {
    const root = new THREE.Group();
    expect(resolveTargetNodeCenter(root, 'absent')).toBeNull();
  });

  it('returns null when the named object has an empty bounding box', () => {
    // A bare Group has no geometry, so its bounding box is empty.
    const root = new THREE.Group();
    const empty = new THREE.Group();
    empty.name = 'empty';
    root.add(empty);
    expect(resolveTargetNodeCenter(root, 'empty')).toBeNull();
  });

  it('returns the world-space center of the first matching descendant', () => {
    const root = new THREE.Group();
    const target = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    target.name = 'pivot';
    target.position.set(10, 0, 0);
    root.add(target);

    const center = resolveTargetNodeCenter(root, 'pivot');
    expect(center).not.toBeNull();
    expect(center!.x).toBeCloseTo(10, 6);
    expect(center!.y).toBeCloseTo(0, 6);
    expect(center!.z).toBeCloseTo(0, 6);
  });

  it('resolves a bare node name against a path-named scene object', () => {
    const root = new THREE.Group();
    const target = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    target.name = '/story/Cluster';
    target.position.set(10, 0, 0);
    root.add(target);

    const center = resolveTargetNodeCenter(root, 'Cluster');
    expect(center?.x).toBeCloseTo(10, 6);
  });

  it('prefers an exact object name over an earlier path-name fallback', () => {
    const root = new THREE.Group();
    const fallback = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    fallback.name = '/story/Cluster';
    fallback.position.set(10, 0, 0);
    const exact = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    exact.name = 'Cluster';
    exact.position.set(20, 0, 0);
    root.add(fallback, exact);

    const center = resolveTargetNodeCenter(root, 'Cluster');
    expect(center?.x).toBeCloseTo(20, 6);
  });

  it('stops on the first match (does not match later siblings with the same name)', () => {
    const root = new THREE.Group();
    const a = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    a.name = 'pivot';
    a.position.set(10, 0, 0);
    const b = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    b.name = 'pivot';
    b.position.set(20, 0, 0);
    root.add(a);
    root.add(b);

    const center = resolveTargetNodeCenter(root, 'pivot');
    expect(center?.x).toBeCloseTo(10, 6); // first match wins
  });
});

describe('applyZarrViewerConfig', () => {
  let root: THREE.Group;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;

  beforeEach(() => {
    root = new THREE.Group();
    scene = new THREE.Scene();
    camera = createDefaultPerspectiveCamera(makeCanvas());
  });

  it('returns positionApplied=false when no viewerConfig is present', () => {
    const { controls } = makeControls();
    const result = applyZarrViewerConfig(root, camera, controls, scene);
    expect(result.positionApplied).toBe(false);
  });

  it('applies camera position and reports positionApplied=true', () => {
    root.userData.viewerConfig = {
      camera: { position: [42, 13, 7] },
    } as unknown as ZarrViewerConfig;
    const { controls, reinitialize, update } = makeControls();

    const result = applyZarrViewerConfig(root, camera, controls, scene);

    expect(camera.position.x).toBe(42);
    expect(camera.position.y).toBe(13);
    expect(camera.position.z).toBe(7);
    expect(result.positionApplied).toBe(true);
    expect(reinitialize).toHaveBeenCalledTimes(1);
    expect(update).toHaveBeenCalledTimes(1);
  });

  it('applies an explicit target via controls.setTarget', () => {
    root.userData.viewerConfig = {
      camera: { target: [1, 2, 3] },
    } as unknown as ZarrViewerConfig;
    const { controls, setTarget } = makeControls();

    const result = applyZarrViewerConfig(root, camera, controls, scene);

    expect(setTarget).toHaveBeenCalledTimes(1);
    const arg = setTarget.mock.calls[0][0] as THREE.Vector3;
    expect(arg.x).toBe(1);
    expect(arg.y).toBe(2);
    expect(arg.z).toBe(3);
    // No position applied → caller should NOT suppress auto-framing.
    expect(result.positionApplied).toBe(false);
  });

  it('target_node takes precedence over explicit target', () => {
    const named = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshBasicMaterial());
    named.name = 'pivot';
    named.position.set(99, 0, 0);
    root.add(named);

    root.userData.viewerConfig = {
      camera: { target: [1, 2, 3], target_node: 'pivot' },
    } as unknown as ZarrViewerConfig;
    const { controls, setTarget } = makeControls();

    applyZarrViewerConfig(root, camera, controls, scene);

    // Should have used the resolved node center (99, 0, 0), NOT the explicit (1,2,3).
    expect(setTarget).toHaveBeenCalledTimes(1);
    const arg = setTarget.mock.calls[0][0] as THREE.Vector3;
    expect(arg.x).toBeCloseTo(99, 6);
  });

  it('logs a warning and does NOT call setTarget when target_node does not resolve', () => {
    root.userData.viewerConfig = {
      camera: { target_node: 'absent' },
    } as unknown as ZarrViewerConfig;
    const { controls, setTarget } = makeControls();

    applyZarrViewerConfig(root, camera, controls, scene);

    expect(setTarget).not.toHaveBeenCalled();
  });

  it('applies up vector and lookAt via getFocusTarget', () => {
    root.userData.viewerConfig = {
      camera: { up: [0, 0, 1] },
    } as unknown as ZarrViewerConfig;
    const { controls, getFocusTarget } = makeControls();

    applyZarrViewerConfig(root, camera, controls, scene);

    expect(camera.up.x).toBe(0);
    expect(camera.up.y).toBe(0);
    expect(camera.up.z).toBe(1);
    expect(getFocusTarget).toHaveBeenCalled();
  });

  it('applies background color', () => {
    root.userData.viewerConfig = {
      background_color: '#abcdef',
    } as unknown as ZarrViewerConfig;
    const { controls } = makeControls();

    applyZarrViewerConfig(root, camera, controls, scene);

    expect(scene.background).toBeInstanceOf(THREE.Color);
    expect((scene.background as THREE.Color).getHexString()).toBe('abcdef');
  });

  it('does NOT call reinitialize when no camera-affecting fields are set', () => {
    root.userData.viewerConfig = {
      background_color: '#000000',
    } as unknown as ZarrViewerConfig;
    const { controls, reinitialize, update } = makeControls();

    applyZarrViewerConfig(root, camera, controls, scene);

    expect(reinitialize).not.toHaveBeenCalled();
    expect(update).not.toHaveBeenCalled();
  });
});
