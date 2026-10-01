/**
 * #2944 review A1: the dimension-animation per-frame callback must stop
 * holding the render loop awake once nothing plays. It is registered
 * `continuous: true`, and only dispose() used to remove it, so after one
 * play → pause the controller's idle check reported "keep animating" forever.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import { AnimationController } from '../../../scene/animation/animation-controller';
import type { ControlsManager } from '../../../controls/controls-manager';
import type { PostProcessingManager } from '../../../rendering';

function setup() {
  const scene = new THREE.Scene();
  scene.userData.sceneDimensions = {
    dimensions: [
      { name: 'x', unit: 'um', range: [0, 100], step: 1, display: true },
      { name: 'y', unit: 'um', range: [0, 100], step: 1, display: true },
      { name: 'z', unit: 'um', range: [0, 50], step: 1, display: true },
      { name: 'time', unit: 's', range: [0, 10], step: 1, display: false, discrete: true },
      { name: 'channel', unit: '', range: [0, 3], step: 1, display: false, discrete: true },
    ],
  };
  const dims = new SceneDimsManager();
  dims.initFromScene(scene);
  const controls = {
    isAutoRotateActive: () => false,
    isAutoDollyActive: () => false,
    isGestureActive: () => false,
  } as unknown as ControlsManager;
  const post = { needsContinuousAnimation: () => false } as unknown as PostProcessingManager;
  const controller = new AnimationController(controls, post);
  vi.spyOn(controller, 'startAnimation').mockImplementation(() => {});
  const manager = new DimensionAnimationManager(dims, controller);
  const keepsAwake = (): boolean =>
    (controller as unknown as { shouldContinueAnimating(): boolean }).shouldContinueAnimating();
  return { manager, controller, keepsAwake };
}

describe('DimensionAnimationManager: render-loop liveness (#2944 A1)', () => {
  it('holds the loop awake while playing', () => {
    const { manager, keepsAwake } = setup();
    manager.play(3, { targetFPS: 10 });
    expect(keepsAwake()).toBe(true);
  });

  it('lets the loop idle after play → pause', () => {
    const { manager, keepsAwake } = setup();
    manager.play(3, { targetFPS: 10 });
    manager.pause(3);
    expect(keepsAwake()).toBe(false);
  });

  it('lets the loop idle after play → stop', () => {
    const { manager, keepsAwake } = setup();
    manager.play(3, { targetFPS: 10 });
    manager.stop(3);
    expect(keepsAwake()).toBe(false);
  });

  it('stays awake while another dimension still plays, then idles', () => {
    const { manager, keepsAwake } = setup();
    manager.play(3, { targetFPS: 10 });
    manager.play(4, { targetFPS: 10 });
    manager.pause(3);
    expect(keepsAwake()).toBe(true);
    manager.pause(4);
    expect(keepsAwake()).toBe(false);
  });

  it('re-arms the callback on a second play', () => {
    const { manager, controller, keepsAwake } = setup();
    manager.play(3, { targetFPS: 10 });
    manager.pause(3);
    manager.play(3, { targetFPS: 10 });
    expect(keepsAwake()).toBe(true);
    expect(controller.hasPerFrameCallback('dimension-animation')).toBe(true);
  });
});
