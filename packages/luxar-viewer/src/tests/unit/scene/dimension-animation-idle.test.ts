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

describe('DimensionAnimationManager: step reporting (#2944 review B)', () => {
  it('a tick that leaves the playhead where it was reports no step', () => {
    // A single-timepoint discrete dim playing in loop mode: every tick wraps
    // back onto the same value, which setDimensionValue ignores. The frame
    // callback must not report a new slice for it.
    const scene = new THREE.Scene();
    scene.userData.sceneDimensions = {
      dimensions: [
        { name: 'x', unit: 'um', range: [0, 100], step: 1, display: true },
        { name: 'y', unit: 'um', range: [0, 100], step: 1, display: true },
        { name: 'z', unit: 'um', range: [0, 50], step: 1, display: true },
        { name: 'time', unit: 's', range: [2, 2], step: 1, display: false, discrete: true },
      ],
    };
    const dims = new SceneDimsManager();
    dims.initFromScene(scene);
    const controller = new AnimationController(
      {} as unknown as ControlsManager,
      {} as unknown as PostProcessingManager
    );
    vi.spyOn(controller, 'startAnimation').mockImplementation(() => {});
    const register = vi.spyOn(controller, 'addPerFrameCallback');
    let now = 1000;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    try {
      const manager = new DimensionAnimationManager(dims, controller);
      manager.play(3, { targetFPS: 10, loopMode: 'loop' });
      const onFrame = register.mock.calls[0][1];
      now += 200; // past the 100 ms tick period: a tick is due
      expect(dims.getDims()!.currentStep[3]).toBe(2);
      expect(onFrame()).toBe(false);
      expect(dims.getDims()!.currentStep[3]).toBe(2);
    } finally {
      vi.restoreAllMocks();
    }
  });
});
