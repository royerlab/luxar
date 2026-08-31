/**
 * Animation-sync concerns extracted from `recording-panel.ts`.
 *
 * Three concerns split out:
 *
 *   1. `getTurntableInfo(speed, fps)` — pure formatter for the
 *      "duration, frame count" line shown in the GUI.
 *   2. `getNavigableDimensionOptions()` — pure read of the scene
 *      dimensions manager to build the dropdown of non-displayed
 *      dimensions for slider-sync.
 *   3. `SliderSyncCoordinator` — owns the slider-sync lifecycle:
 *      register a "complete" listener on the animation manager, kick
 *      off playback after a small delay, and cancel both halves
 *      cleanly on stop / dispose.
 *
 * The turntable rotation loop stays in the recording-panel because it
 * shares `savedAutoRotate` state with the offline capture path, so
 * splitting it here would just push the coupling around.
 *
 * @module ui/recording-panel/animation-sync
 */

import type { DimensionAnimationManager } from '../../scene/animation/dimension-animation-manager';
import { sceneDimsManager } from '../../scene/scene-dims-manager';
import { log, Modules } from '../../utils/log';

/**
 * Seconds one recorded turn takes, from the stored turntable SPEED — and back.
 *
 * `RecordingOptions.turntableSpeed` is degrees per second, so a full 360°
 * takes `360/speed` seconds. Like the navigation popover's rotation row, the
 * UI shows the DURATION (the same unit as every other timing control) while
 * the option keeps the rate it has always had, so saved presets and the
 * capture strategies are untouched.
 *
 * A non-positive or non-finite input falls back to the other unit's identity
 * rather than dividing to `Infinity`/`NaN`, which would make `totalFrames`
 * either zero or unbounded.
 */
export function turnSecondsFromDegPerSec(degPerSec: number): number {
  if (!Number.isFinite(degPerSec) || degPerSec <= 0) return 360;
  return 360 / degPerSec;
}

/** Inverse of {@link turnSecondsFromDegPerSec} (10 s → 36 °/s). */
export function degPerSecFromTurnSeconds(seconds: number): number {
  if (!Number.isFinite(seconds) || seconds <= 0) return 360;
  return 360 / seconds;
}

/**
 * Format the turntable duration / frame-count info string used by the
 * GUI ("12.0s, 720 frames"). Pure given speed and target FPS.
 */
export function getTurntableInfo(turntableSpeedDegPerSec: number, videoFPS: number): string {
  const duration = 360 / turntableSpeedDegPerSec;
  const frames = Math.ceil(duration * videoFPS);
  return `${duration.toFixed(1)}s, ${frames} frames`;
}

/**
 * Build the dropdown options for the "sync to slider" feature: a
 * map of human-readable dimension name → dimension index, restricted
 * to non-displayed dimensions. Returns a `(no dimensions)` placeholder
 * mapped to `-1` if every dim is currently displayed.
 */
export function getNavigableDimensionOptions(): Record<string, number> {
  const options: Record<string, number> = {};
  const dims = sceneDimsManager.getDims();
  if (dims) {
    const names = sceneDimsManager.getDimensionNames();
    for (let i = 0; i < dims.ndim; i++) {
      if (!dims.displayed.includes(i)) {
        options[names[i] || `dim ${i}`] = i;
      }
    }
  }
  if (Object.keys(options).length === 0) {
    options['(no dimensions)'] = -1;
  }
  return options;
}

/**
 * Owns the slider-sync lifecycle: registers an animation-complete
 * listener and a play-after-delay timeout, and tears both down
 * cleanly. Pulled into its own class because the panel had two
 * mirrored half-states (`syncCompleteHandler`, `syncPlayTimeout`)
 * each needing its own cleanup branch.
 */
export class SliderSyncCoordinator {
  private completeHandler: (() => void) | null = null;
  private playTimeout: ReturnType<typeof setTimeout> | null = null;

  /**
   * Snap the dimension to its minimum value, register the
   * "play complete → onComplete()" listener, then schedule the
   * actual play call after a short delay so the snap propagates
   * before playback starts.
   *
   * `playDelayMs` defaults to 100 ms.
   * `isAlive` is consulted right before play() so a dispose during
   * the delay window short-circuits cleanly.
   */
  start(
    dimIndex: number,
    animationManager: DimensionAnimationManager,
    onComplete: () => void,
    isAlive: () => boolean,
    playDelayMs = 100
  ): void {
    if (dimIndex < 0) return;

    this.cleanup(animationManager);

    const ranges = sceneDimsManager.getDimensionRanges();
    if (ranges && ranges[dimIndex]) {
      const [min] = ranges[dimIndex];
      sceneDimsManager.setDimensionValue(dimIndex, min);
    }

    this.completeHandler = () => {
      log.info(Modules.RECORDING, 'Slider sync complete — stopping recording');
      onComplete();
    };
    // The animation manager's typings expect a specific event-name
    // string union; we cast through `never` here to register the
    // synthetic 'complete' event the recording flow listens for.
    animationManager.addEventListener('complete', this.completeHandler as never);

    this.playTimeout = setTimeout(() => {
      this.playTimeout = null;
      if (isAlive()) {
        animationManager.play(dimIndex, { loopMode: 'once', direction: 'forward' });
      }
    }, playDelayMs);
  }

  /**
   * Cancel a pending start delay and unhook the complete listener.
   * Idempotent — safe to call from dispose, on stop, and on a fresh
   * start before re-arming.
   */
  cleanup(animationManager: DimensionAnimationManager | null): void {
    if (this.playTimeout !== null) {
      clearTimeout(this.playTimeout);
      this.playTimeout = null;
    }
    if (this.completeHandler && animationManager) {
      animationManager.removeEventListener('complete', this.completeHandler as never);
      this.completeHandler = null;
    }
  }
}
