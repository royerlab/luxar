/**
 * Animation-shortcut concern extracted from `input/input-handler.ts`.
 *
 * Owns the five keyboard bindings that drive
 * DimensionAnimationManager — all of them target the currently
 * selected dimension and live on the `NAVIGATION` input context:
 *
 *   - K          : Toggle play/pause for the selected dim.
 *   - Home / End : Jump to the first / last frame of the selected dim.
 *   - Shift+↑    : Increase the animation FPS for the selected dim.
 *   - Shift+↓    : Decrease the animation FPS for the selected dim.
 *
 * Each binding short-circuits when no dim is selected
 * (`getSelectedDimensionIndex` returns `-1`) or when the animation
 * manager isn't constructed yet — both checks live inside the
 * handler bodies so the InputHandler can wire the registration
 * before scene load (the bindings just become no-ops until both
 * sides are ready).
 *
 * Behavior is identical to the inlined `registerAnimationShortcuts`
 * — same key codes, same NAVIGATION context, same log emojis +
 * messages, same preventDefault setting.
 *
 * @module input/handlers/animation-shortcuts
 */

import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { InputContext, type InputContextManager } from '../context-manager';
import { getSelectedDimensionIndex } from '../dimension-navigation/compute-step';
import { log, Modules } from '../../../utils/log';

/**
 * Read-only access to the two pieces of InputHandler state the
 * shortcuts read on every key press.
 *
 * Functions (not properties) so the handler captures the latest
 * value at dispatch time — `selectedDimension` mutates while the
 * scene is open, and `animationManager` is constructed lazily
 * (`initAnimationManager()`).
 */
export interface AnimationShortcutsContext {
  /** 0-based index into the navigable-dimensions list. */
  getSelectedDimension(): number;
  /** May be `undefined` until `initAnimationManager()` runs. */
  getAnimationManager(): DimensionAnimationManager | undefined;
}

export class AnimationShortcuts {
  constructor(
    private contextManager: InputContextManager,
    private ctx: AnimationShortcutsContext
  ) {}

  /**
   * Register the five animation-shortcut bindings on the
   * `NAVIGATION` context. Idempotency is the InputContextManager's
   * responsibility — the InputHandler only ever calls this once
   * (from `initAnimationManager()`).
   */
  register(): void {
    // K — Toggle play/pause
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'k',
      handler: () => this.toggleSelectedDimensionPlayback(),
      preventDefault: true,
      description: 'Toggle dimension animation (K)',
    });

    // Home — Jump to start
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'Home',
      handler: () => this.jumpSelectedDimensionToBound('start'),
      preventDefault: true,
      description: 'Jump to dimension start (Home)',
    });

    // End — Jump to end
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'End',
      handler: () => this.jumpSelectedDimensionToBound('end'),
      preventDefault: true,
      description: 'Jump to dimension end (End)',
    });

    // Shift+↑ — Increase speed
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'ArrowUp',
      modifiers: { shift: true },
      handler: () => this.adjustSelectedDimensionSpeed(+1),
      preventDefault: true,
      description: 'Increase animation speed (Shift+↑)',
    });

    // Shift+↓ — Decrease speed
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'ArrowDown',
      modifiers: { shift: true },
      handler: () => this.adjustSelectedDimensionSpeed(-1),
      preventDefault: true,
      description: 'Decrease animation speed (Shift+↓)',
    });

    log.success(Modules.ANIMATION, 'Animation keyboard shortcuts registered');
  }

  // -- shortcut implementations -------------------------------------------

  /**
   * Resolve the currently selected actual dimension index, or `-1`
   * when no dim is selectable. Centralizes the
   * `selectedDimension → dims → navigableDims[i]` lookup so the
   * bindings can read it without re-implementing the SimpleDims
   * indirection.
   */
  private resolveDimensionIndex(): number {
    return getSelectedDimensionIndex(this.ctx.getSelectedDimension(), sceneDimsManager.getDims());
  }

  private toggleSelectedDimensionPlayback(): void {
    const dimIndex = this.resolveDimensionIndex();
    const animManager = this.ctx.getAnimationManager();
    if (dimIndex < 0 || !animManager) return;
    const isPlaying = animManager.togglePlay(dimIndex);
    log.info(Modules.ANIMATION, `Dimension ${dimIndex} ${isPlaying ? 'playing' : 'paused'}`);
  }

  private jumpSelectedDimensionToBound(which: 'start' | 'end'): void {
    const dimIndex = this.resolveDimensionIndex();
    if (dimIndex < 0) return;
    const ranges = sceneDimsManager.getDimensionRanges();
    if (!ranges) return;
    const value = which === 'start' ? ranges[dimIndex][0] : ranges[dimIndex][1];
    sceneDimsManager.setDimensionValue(dimIndex, value);
    log.info(Modules.ANIMATION, `Jumped to ${which} of dimension ${dimIndex}`);
  }

  private adjustSelectedDimensionSpeed(direction: 1 | -1): void {
    const dimIndex = this.resolveDimensionIndex();
    const animManager = this.ctx.getAnimationManager();
    if (dimIndex < 0 || !animManager) return;
    if (direction > 0) {
      animManager.increaseSpeed(dimIndex);
    } else {
      animManager.decreaseSpeed(dimIndex);
    }
    const fps = animManager.getState(dimIndex)?.targetFPS;
    log.info(Modules.ANIMATION, `${direction > 0 ? 'Increased' : 'Decreased'} speed to ${fps} FPS`);
  }
}
