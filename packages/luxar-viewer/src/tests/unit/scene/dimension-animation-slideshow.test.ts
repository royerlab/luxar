/**
 * Slow ("slideshow") dimension playback — the auto-advance a kiosk tour runs
 * on its story dimension (one step every 20 s):
 *
 * - someone else moving a playing dimension (a control-panel tap, the
 *   slider, a key) re-arms its clock, so the next step is one full period
 *   after the move — the playhead's own steps do not;
 * - while the slideshow hold is on (a waypoint flight in the air) the clock
 *   does not run, so the period counts from the landing; frame-paced
 *   playback is never held;
 * - a slideshow is not under the playback-LOD directives: each step loads
 *   like a manual one;
 * - the keyboard speed ladder reaches the period presets, and the Speed
 *   view follows the rate unless one was chosen.
 *
 * Same harness as dimension-animation-manager.test.ts: a real
 * SceneDimsManager and AnimationController, the per-frame callback captured
 * and driven by hand under a mocked `performance.now`.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  DimensionAnimationManager,
  speedLadder,
} from '../../../scene/animation/dimension-animation-manager';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import {
  AnimationController,
  type PerFrameCallback,
} from '../../../scene/animation/animation-controller';
import type { ControlsManager } from '../../../controls/controls-manager';
import type { PostProcessingManager } from '../../../rendering';
import { config } from '../../../config';
import { MAX_ANIMATION_PERIOD_S } from '../../../types/zarr';

const STORY = 3; // discrete 0..10, hidden
const TIME = 4; // discrete 0..3, hidden

it('uses the authored animation period bound as its slowest rate', () => {
  expect(config.dimensionAnimation.presets.customMin).toBe(1 / MAX_ANIMATION_PERIOD_S);
});

describe('DimensionAnimationManager slideshow playback', () => {
  let dims: SceneDimsManager;
  let manager: DimensionAnimationManager;
  let frame: PerFrameCallback | null = null;
  let now = 0;
  let held = false;

  const value = (dim: number): number => dims.getDims()!.currentStep[dim];
  const tick = (advanceMs: number): void => {
    now += advanceMs;
    frame?.();
  };

  beforeEach(() => {
    const scene = new THREE.Scene();
    scene.userData.sceneDimensions = {
      dimensions: [
        { name: 'x', range: [0, 100], step: 1, display: true },
        { name: 'y', range: [0, 100], step: 1, display: true },
        { name: 'z', range: [0, 50], step: 1, display: true },
        { name: 'story', range: [0, 10], step: 1, display: false, discrete: true },
        { name: 'time', range: [0, 3], step: 1, display: false, discrete: true },
      ],
    };
    dims = new SceneDimsManager();
    dims.initFromScene(scene);
    vi.spyOn(dims, 'waitForUpdate').mockResolvedValue(undefined);

    const controller = new AnimationController({} as ControlsManager, {} as PostProcessingManager);
    vi.spyOn(controller, 'startAnimation').mockImplementation(() => {});
    vi.spyOn(controller, 'addPerFrameCallback').mockImplementation((_id, cb) => {
      frame = cb;
    });
    vi.spyOn(controller, 'removePerFrameCallback').mockImplementation(() => {
      frame = null;
      return true;
    });

    now = 1000;
    held = false;
    vi.spyOn(performance, 'now').mockImplementation(() => now);
    manager = new DimensionAnimationManager(dims, controller, undefined, () => held);
  });

  afterEach(() => {
    manager.dispose();
    vi.restoreAllMocks();
    frame = null;
  });

  /** Let pending `waitForUpdate` continuations release the pacing gate. */
  const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  it('steps once per period', async () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    tick(19_000);
    expect(value(STORY)).toBe(0);
    tick(1_000);
    expect(value(STORY)).toBe(1);
    await settle();
    tick(19_000);
    expect(value(STORY)).toBe(1);
    tick(1_000);
    expect(value(STORY)).toBe(2);
  });

  it('pausing a slideshow does not force a reload of its refined step', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    tick(20_000);
    const setValue = vi.spyOn(dims, 'setDimensionValue');
    expect(manager.pause(STORY)).toBe(true);
    expect(value(STORY)).toBe(1);
    expect(setValue).not.toHaveBeenCalled();
  });

  it('an outside move re-arms the clock: the next step is a full period after it', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    tick(19_000);
    // A visitor taps story 5 one second before the step was due.
    dims.setDimensionValue(STORY, 5);
    tick(1_000);
    expect(value(STORY)).toBe(5); // NOT 6: the tap restarted the period
    tick(18_000);
    expect(value(STORY)).toBe(5);
    tick(1_000);
    expect(value(STORY)).toBe(6);
  });

  it('every event of a drag re-arms: the period counts from the last one', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    for (const v of [2, 3, 4]) {
      tick(15_000);
      dims.setDimensionValue(STORY, v);
    }
    tick(19_000);
    expect(value(STORY)).toBe(4);
    tick(1_000);
    expect(value(STORY)).toBe(5);
  });

  it("the playhead's own steps do not re-arm, and other dims' moves leave it alone", async () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    tick(20_000);
    expect(value(STORY)).toBe(1);
    await settle();
    // Moving a dimension that is NOT playing does not touch the story clock.
    tick(10_000);
    dims.setDimensionValue(TIME, 2);
    tick(10_000);
    expect(value(STORY)).toBe(2);
  });

  it('the hold stops a slideshow clock; the period starts when it lifts', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    held = true; // a flight is in the air
    tick(15_000);
    tick(15_000);
    expect(value(STORY)).toBe(0);
    held = false; // landed
    tick(19_000);
    expect(value(STORY)).toBe(0);
    tick(1_000);
    expect(value(STORY)).toBe(1);
  });

  it('frame-paced playback is never held', () => {
    manager.play(TIME, { targetFPS: 10 });
    held = true;
    tick(150);
    expect(value(TIME)).toBe(1);
  });

  it('a held or re-armed slideshow is behind its rate by design: no cadence warning', async () => {
    const warnings: unknown[] = [];
    manager.addEventListener('fpsWarning', (e) => warnings.push(e));
    manager.play(STORY, { targetFPS: 1 / 20 });
    for (let stop = 0; stop < 4; stop++) {
      held = true; // a 10 s flight
      tick(5_000);
      tick(5_000);
      held = false;
      tick(20_000);
      await settle();
    }
    expect(value(STORY)).toBe(4);
    expect(warnings).toEqual([]);

    // Frame-paced playback still reports a slipped cadence.
    manager.play(TIME, { targetFPS: 60 });
    for (let i = 0; i < 4; i++) {
      tick(600); // ~1.7 fps achieved of 60 requested
      await settle();
    }
    expect(warnings.length).toBeGreaterThan(0);
  });

  it('starts a fresh cadence window when switching a slideshow to frame-paced playback', async () => {
    const warnings: unknown[] = [];
    manager.addEventListener('fpsWarning', (e) => warnings.push(e));
    manager.play(STORY, { targetFPS: 1 / 20 });
    for (let i = 0; i < 3; i++) {
      tick(20_000);
      await settle();
    }
    expect(value(STORY)).toBe(3);

    manager.setTargetFPS(STORY, 10);
    for (let i = 0; i < 5; i++) {
      tick(100);
      await settle();
    }
    expect(value(STORY)).toBe(8);
    expect(warnings).toEqual([]);
  });

  it('a slideshow is not under the playback-LOD directives; frame-paced playback is', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    expect(manager.isAnyPlaying()).toBe(true);
    expect(manager.isFramePacedPlaying()).toBe(false);
    expect(manager.getFramePacedDimIndices()).toEqual([]);
    expect(manager.getFrameBudgetMs()).toBeNull();
    expect(manager.getPlaybackPeriodMs()).toBeNull();
    manager.setLadderDepth(STORY, 3);
    expect(manager.getPlaybackLadderDepth()).toBeNull();

    manager.play(TIME, { targetFPS: 10 });
    expect(manager.isFramePacedPlaying()).toBe(true);
    expect(manager.getFramePacedDimIndices()).toEqual([TIME]);
    expect(manager.getPlaybackPeriodMs()).toBeCloseTo(100, 9);
    expect(manager.getFrameBudgetMs()).not.toBeNull();
  });

  it('the slideshow threshold is the configured period', () => {
    const thresholdFPS = 1000 / config.dimensionAnimation.timing.slideshowPeriodMs;
    manager.play(STORY, { targetFPS: thresholdFPS * 1.01 });
    expect(manager.isFramePacedPlaying()).toBe(true);
    manager.setTargetFPS(STORY, thresholdFPS);
    expect(manager.isFramePacedPlaying()).toBe(false);
  });

  it('the speed ladder merges fps and period presets, slowest first, and the keys reach it', () => {
    const ladder = speedLadder();
    expect(ladder).toEqual([...ladder].sort((a, b) => a - b));
    expect(new Set(ladder).size).toBe(ladder.length);
    expect(ladder).toContain(1 / 20);
    expect(ladder[0]).toBeCloseTo(1 / 600, 12);

    manager.play(STORY, { targetFPS: 1 / 10 });
    manager.decreaseSpeed(STORY);
    expect(manager.getState(STORY)?.targetFPS).toBeCloseTo(1 / 20, 12);
    manager.increaseSpeed(STORY);
    expect(manager.getState(STORY)?.targetFPS).toBeCloseTo(1 / 10, 12);
  });

  it('the Speed view follows the rate until one is chosen, and an authored view sticks', () => {
    expect(manager.getSpeedView(STORY)).toBe('fps'); // default 10 fps
    manager.setTargetFPS(STORY, 0.5);
    expect(manager.getSpeedView(STORY)).toBe('period');
    manager.setSpeedView(STORY, 'fps');
    expect(manager.getSpeedView(STORY)).toBe('fps');

    manager.play(TIME, { targetFPS: 2, speedView: 'period' });
    expect(manager.getSpeedView(TIME)).toBe('period');
  });
});
