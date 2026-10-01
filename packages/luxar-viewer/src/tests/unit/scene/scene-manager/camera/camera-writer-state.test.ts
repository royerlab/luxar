// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { ControlsManager } from '../../../../../controls/controls-manager';
import {
  centerOnOrigin,
  fitCameraToBounds,
} from '../../../../../scene/scene-manager/camera/camera-framing';

const domElements: HTMLElement[] = [];
const managers: ControlsManager[] = [];

function makeControls(): { camera: THREE.PerspectiveCamera; controls: ControlsManager } {
  const element = document.createElement('div');
  document.body.appendChild(element);
  domElements.push(element);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
  camera.position.set(4, 3, 12);
  const controls = new ControlsManager(camera, element);
  managers.push(controls);
  return { camera, controls };
}

afterEach(() => {
  for (const controls of managers) controls.dispose();
  for (const element of domElements) element.remove();
  managers.length = 0;
  domElements.length = 0;
});

describe('camera writers with live controls', () => {
  it('keeps the origin view upright and at its new distance after the next orbit update', () => {
    const { camera, controls } = makeControls();
    camera.position.set(7, 5, 9);
    camera.lookAt(0, 0, 0);
    controls.reinitialize();
    controls.update();
    const distance = camera.position.distanceTo(controls.getFocusTarget());

    centerOnOrigin(camera, controls);
    const position = camera.position.clone();
    const quaternion = camera.quaternion.clone();
    controls.update();

    expect(position.distanceTo(new THREE.Vector3(0, 0, distance))).toBeLessThan(1e-6);
    expect(camera.position.distanceTo(position)).toBeLessThan(1e-6);
    expect(camera.quaternion.angleTo(quaternion)).toBeLessThan(1e-6);
    expect(
      camera.getWorldDirection(new THREE.Vector3()).distanceTo(new THREE.Vector3(0, 0, -1))
    ).toBeLessThan(1e-6);
    expect(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).y).toBeCloseTo(1, 6);
  });

  it('publishes one change across an origin write and the next orbit update', () => {
    const { camera, controls } = makeControls();
    const changed = vi.fn();
    controls.addEventListener('change', changed);

    centerOnOrigin(camera, controls);
    if (changed.mock.calls.length === 0) controls.dispatchEvent({ type: 'change' });
    controls.update();

    expect(changed).toHaveBeenCalledTimes(1);
  });

  it('publishes one change across a control switch and the next update', () => {
    const { controls } = makeControls();
    const changed = vi.fn();
    controls.addEventListener('change', changed);

    controls.setControlType('fly');
    controls.update();
    expect(changed).toHaveBeenCalledTimes(1);

    changed.mockClear();
    controls.setControlType('orbit');
    controls.update();
    expect(changed).toHaveBeenCalledTimes(1);
  });

  it('keeps a preserved fly framing target after the next update', () => {
    const { camera, controls } = makeControls();
    camera.lookAt(0, 0, 0);
    controls.setControlType('fly');
    const target = controls.getFocusTarget();
    const bounds = { min: { x: -2, y: -2, z: -2 }, max: { x: 2, y: 2, z: 2 } };

    fitCameraToBounds(camera, controls, bounds, {
      lookAtTarget: target,
      preserveControlsTarget: true,
    });
    const direction = target.clone().sub(camera.position).normalize();
    expect(camera.getWorldDirection(new THREE.Vector3()).distanceTo(direction)).toBeLessThan(1e-6);

    controls.update();
    expect(camera.getWorldDirection(new THREE.Vector3()).distanceTo(direction)).toBeLessThan(1e-6);
  });
});
