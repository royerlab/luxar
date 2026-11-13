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
const DEPTH = 5; // continuous 0..100, hidden, no authored step

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
        { name: 'depth', range: [0, 100], display: false },
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

  it('holds a narrated stop until its own long clip ends plus the pad', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/story-0/voice']);
    tick(600);
    manager.narrationStarted('/story-0/voice', 43_000);
    tick(42_400);
    expect(value(STORY)).toBe(0);
    manager.narrationEnded('/story-0/voice');
    tick(1_999);
    expect(value(STORY)).toBe(0);
    tick(1);
    expect(value(STORY)).toBe(1);
  });

  it('releases a started clip that never ends at its duration plus pad and slack', async () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/interrupted']);
    tick(1_000);
    manager.narrationStarted('/interrupted', 30_000);
    tick(32_999);
    expect(value(STORY)).toBe(0);
    tick(1);
    expect(value(STORY)).toBe(1);
    await settle();
    manager.narrationEnded('/interrupted');
    tick(19_990);
    expect(value(STORY)).toBe(1);
    tick(10);
    expect(value(STORY)).toBe(2);
  });

  it('does not extend the cap when a late end arrives before the next frame', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/interrupted']);
    manager.narrationStarted('/interrupted', 30_000);
    now += 33_000;
    manager.narrationEnded('/interrupted');
    tick(0);
    expect(value(STORY)).toBe(1);
  });

  it('keeps the authored minimum when a short clip never ends', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/interrupted']);
    manager.narrationStarted('/interrupted', 5_000);
    tick(19_000);
    expect(value(STORY)).toBe(0);
    tick(1_000);
    expect(value(STORY)).toBe(1);
  });

  it('keeps a running narration held across pause and resume', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/voice']);
    manager.narrationStarted('/voice', 60_000);
    tick(5_000);
    manager.pause(STORY);
    manager.play(STORY);
    tick(20_000);
    expect(value(STORY)).toBe(0);
    manager.narrationEnded('/voice');
    tick(1_999);
    expect(value(STORY)).toBe(0);
    tick(1);
    expect(value(STORY)).toBe(1);
  });

  it('does not restore a paused stop after a visitor moves the playhead', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/old']);
    manager.narrationStarted('/old', 60_000);
    manager.pause(STORY);
    dims.setDimensionValue(STORY, 5);
    dims.setDimensionValue(STORY, 0);
    manager.play(STORY);
    tick(20_000);
    expect(value(STORY)).toBe(1);
  });

  it('keeps the period for a short clip or one that never starts', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/short']);
    manager.narrationStarted('/short', 5_000);
    tick(5_000);
    manager.narrationEnded('/short');
    tick(14_999);
    expect(value(STORY)).toBe(0);
    tick(1);
    expect(value(STORY)).toBe(1);
  });

  it('falls back to the period when autoplay blocks the clip', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/blocked']);
    tick(20_000);
    expect(value(STORY)).toBe(1);
  });

  it('waits for every voice of a spatial narration to end', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/spatial']);
    manager.narrationStarted('/spatial', 30_000);
    manager.narrationStarted('/spatial', 30_000);
    tick(25_000);
    manager.narrationEnded('/spatial');
    tick(2_000);
    expect(value(STORY)).toBe(0);
    manager.narrationEnded('/spatial');
    tick(2_000);
    expect(value(STORY)).toBe(1);
  });

  it('restarts at a new landing and ignores the previous narration ending', () => {
    manager.play(STORY, { targetFPS: 1 / 20 });
    manager.beginNarratedStop(STORY, ['/old']);
    manager.narrationStarted('/old', 30_000);
    tick(10_000);
    dims.setDimensionValue(STORY, 5);
    manager.beginNarratedStop(STORY, ['/new']);
    manager.narrationEnded('/old');
    manager.narrationStarted('/new', 30_000);
    tick(19_999);
    expect(value(STORY)).toBe(5);
    tick(1);
    expect(value(STORY)).toBe(5);
    manager.narrationEnded('/new');
    tick(1_999);
    expect(value(STORY)).toBe(5);
    tick(1);
    expect(value(STORY)).toBe(6);
  });

  it('frame-paced playback is never held', () => {
    manager.play(TIME, { targetFPS: 10 });
    held = true;
    manager.beginNarratedStop(TIME, ['/voice']);
    manager.narrationStarted('/voice', 30_000);
    tick(150);
    expect(value(TIME)).toBe(1);
  });

  it('switching a slideshow to a frame-paced rate measures the new rate from the switch', async () => {
    const warnings: unknown[] = [];
    manager.addEventListener('fpsWarning', (e) => warnings.push(e));
    manager.play(STORY, { targetFPS: 1 / 20 });
    tick(20_000);
    await settle();
    tick(19_000); // a step is 1 s away when the visitor picks 10 fps
    manager.setTargetFPS(STORY, 10);
    for (let i = 0; i < 12; i++) {
      tick(100); // 10 fps achieved, as requested
      await settle();
    }
    expect(value(STORY)).toBeGreaterThan(1);
    expect(warnings).toEqual([]);
  });

  it('play() with a faster rate on a playing slideshow takes effect from now', async () => {
    const warnings: unknown[] = [];
    manager.addEventListener('fpsWarning', (e) => warnings.push(e));
    manager.play(STORY, { targetFPS: 1 / 20 });
    tick(5_000);
    manager.play(STORY, { targetFPS: 10 }); // e.g. a viewer_config re-apply
    tick(100);
    expect(value(STORY)).toBe(1); // not 15 s later, when the old period ran out
    for (let i = 0; i < 12; i++) {
      await settle();
      tick(100);
    }
    expect(warnings).toEqual([]);
  });

  it('a held slideshow is behind its rate by design: no cadence warning', async () => {
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

  it("a continuous dim's slideshow tick is a manual step, not a range-scaled jump", async () => {
    // Frame-paced: the range in continuousTraverseSeconds (10 s), so 1 fps is 10 a tick.
    const start = value(DEPTH);
    manager.play(DEPTH, { targetFPS: 1 });
    tick(1_000);
    expect(value(DEPTH)).toBeCloseTo(start + 10, 9);
    await settle();
    // A slideshow tick takes the [ / ] key's step (1% of the range, no authored
    // step), where the rate-derived increment would be 200 — twice the range.
    manager.setTargetFPS(DEPTH, 1 / 20);
    expect(manager.peekNextValue(DEPTH)).toBeCloseTo(start + 11, 9);
    tick(20_000);
    expect(value(DEPTH)).toBeCloseTo(start + 11, 9);
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
    const slowest = Math.max(...config.dimensionAnimation.presets.periodSeconds);
    expect(ladder[0]).toBeCloseTo(1 / slowest, 12);

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
