// @vitest-environment jsdom
/**
 * The data-server-unreachable hold against the REAL controls and the REAL
 * wake path: the app wires every controls `change` to
 * `animationController.startAnimation()` (via SceneManager,
 * `core/app/init/pipeline.ts`), and auto-rotate emits `change` on every step —
 * so without suppressing the turntable, one wake (a mouse-over, a focus) let
 * auto-rotate keep the loop alive forever under the hold. Dimension playback
 * had the same escape through its dims listeners.
 *
 * Only the post-processing renderer is stubbed; frames run through the real
 * RafDriver by invoking the armed requestAnimationFrame callback, with fake
 * timers (performance.now included) advancing 16 ms per frame.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import { ControlsManager } from '../../../controls/controls-manager';
import { AnimationController } from '../../../scene/animation/animation-controller';
import { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import type { PostProcessingManager } from '../../../rendering';
import { createTestCamera } from '../../test-config';
import { config } from '../../../config';

const REASON = 'data-server-unreachable';
const FRAME_MS = 16;

describe('self-sustain hold with real controls', () => {
  let camera: THREE.PerspectiveCamera;
  let dom: HTMLElement;
  let controls: ControlsManager;
  let controller: AnimationController;
  let pending: FrameRequestCallback | null;
  let rendered: number;

  /** Run `n` frames (16 ms apart); returns how many actually ticked. */
  function runFrames(n: number): number {
    let ticks = 0;
    for (let i = 0; i < n; i++) {
      vi.advanceTimersByTime(FRAME_MS);
      const cb = pending;
      pending = null;
      if (cb) {
        cb(performance.now());
        ticks++;
      }
    }
    return ticks;
  }

  beforeEach(() => {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'performance', 'Date'],
    });
    pending = null;
    rendered = 0;
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      pending = cb;
      return 1;
    });
    vi.stubGlobal('cancelAnimationFrame', () => {
      pending = null;
    });
    camera = createTestCamera(1);
    camera.position.set(0, 0, 5);
    dom = document.createElement('div');
    document.body.appendChild(dom);
    controls = new ControlsManager(camera, dom, new THREE.Scene());
    const post = {
      render: () => {
        rendered++;
      },
      needsContinuousAnimation: () => false,
    } as unknown as PostProcessingManager;
    controller = new AnimationController(controls, post);
    // The app's wake path: controls change → startAnimation.
    controls.addEventListener('change', () => controller.startAnimation());
  });

  afterEach(() => {
    controller.dispose();
    controls.dispose();
    dom.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  const seconds = (s: number): number => Math.ceil((s * 1000) / FRAME_MS);

  it('auto-rotate keeps the loop alive and moves the camera when not held', () => {
    controls.setAutoRotate(true);
    const before = camera.position.clone();
    controller.startAnimation();
    const ticks = runFrames(seconds(3));
    expect(ticks).toBeGreaterThan(seconds(2.5));
    expect(camera.position.distanceTo(before)).toBeGreaterThan(1e-3);
    expect(controller.isActive).toBe(true);
  });

  it('under the hold, a wake does not let auto-rotate run forever', () => {
    controls.setAutoRotate(true);
    controller.suspendSelfSustainedAnimation(REASON);
    const before = camera.position.clone();
    controller.startAnimation(); // a mouse-over / focus wake
    runFrames(seconds(1));
    const lateTicks = runFrames(seconds(3));
    // Idles out after idleTimeoutMs; the turntable never drove the camera.
    expect(lateTicks).toBeLessThan(seconds(config.animation.idleTimeoutMs / 1000));
    expect(controller.isActive).toBe(false);
    expect(camera.position.distanceTo(before)).toBeLessThan(1e-9);
    // The setting itself is untouched.
    expect(controls.getAutoRotate()).toBe(true);
  });

  it('auto-rotate resumes when the hold is released', () => {
    controls.setAutoRotate(true);
    controller.suspendSelfSustainedAnimation(REASON);
    controller.resumeSelfSustainedAnimation(REASON);
    const before = camera.position.clone();
    const ticks = runFrames(seconds(3));
    expect(ticks).toBeGreaterThan(seconds(2.5));
    expect(camera.position.distanceTo(before)).toBeGreaterThan(1e-3);
  });

  it('a capture lifts the hold: auto-rotate and the loop keep running', () => {
    controls.setAutoRotate(true);
    controller.setSelfSustainExemptPredicate(() => true);
    controller.suspendSelfSustainedAnimation(REASON);
    controller.startAnimation();
    const before = camera.position.clone();
    const ticks = runFrames(seconds(3));
    expect(ticks).toBeGreaterThan(seconds(2.5));
    expect(rendered).toBeGreaterThan(0);
    expect(camera.position.distanceTo(before)).toBeGreaterThan(1e-3);
  });

  it('dimension playback holds its playhead and lets the loop idle under the hold', () => {
    const scene = new THREE.Scene();
    scene.userData.sceneDimensions = {
      dimensions: [
        { name: 'x', unit: 'um', range: [0, 100], step: 1, display: true },
        { name: 'y', unit: 'um', range: [0, 100], step: 1, display: true },
        { name: 'z', unit: 'um', range: [0, 50], step: 1, display: true },
        { name: 'time', unit: 's', range: [0, 1000], step: 1, display: false, discrete: true },
      ],
    };
    const dims = new SceneDimsManager();
    dims.initFromScene(scene);
    // The app's dims listener wakes the loop on every step.
    dims.addListener(() => controller.startAnimation());
    const playback = new DimensionAnimationManager(dims, controller);
    try {
      controller.suspendSelfSustainedAnimation(REASON);
      playback.play(3, { targetFPS: 30 });
      const start = dims.getDims()!.currentStep[3];
      runFrames(seconds(4));
      expect(dims.getDims()!.currentStep[3]).toBe(start);
      expect(controller.isActive).toBe(false);

      // Released: playback steps again.
      controller.resumeSelfSustainedAnimation(REASON);
      runFrames(seconds(1));
      expect(dims.getDims()!.currentStep[3]).not.toBe(start);
    } finally {
      playback.dispose();
    }
  });
});
