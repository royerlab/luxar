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
 * before scene load (the bindings decline until both sides are ready).
 *
 * The bindings retain the inlined `registerAnimationShortcuts` key codes,
 * NAVIGATION context, logging, and preventDefault settings.
 *
 * @module input/handlers/animation-shortcuts
 */

import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import type { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { InputContext, type InputContextManager } from '../context-manager';
import { getSelectedDimensionIndex } from '../dimension-navigation/compute-step';
import { log, Modules } from '../../../utils/log';
import { KeyAction } from './actions';

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
   * `NAVIGATION` context. `registerAllKeyBindings()` calls this once
   * during InputHandler startup; the handlers decline until a scene
   * provides a selected dimension and animation manager.
   */
  register(): void {
    // K — Toggle play/pause
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: KeyAction.toggleAnimation,
      key: 'k',
      handler: () => this.toggleSelectedDimensionPlayback(),
      preventDefault: true,
      description: 'Play / pause dimension animation',
      help: { section: 'dimensions', group: 'animation-toggle', order: 50 },
    });

    // Home — Jump to start
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: KeyAction.jumpAnimation,
      actionParameter: 'start',
      key: 'Home',
      handler: () => this.jumpSelectedDimensionToBound('start'),
      preventDefault: true,
      description: 'Jump to dimension start / end',
      help: {
        section: 'dimensions',
        group: 'animation-bounds',
        keys: ['Home', 'End'],
        order: 60,
      },
    });

    // End — Jump to end
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: KeyAction.jumpAnimation,
      actionParameter: 'end',
      key: 'End',
      handler: () => this.jumpSelectedDimensionToBound('end'),
      preventDefault: true,
      description: 'Jump to dimension start / end',
      help: {
        section: 'dimensions',
        group: 'animation-bounds',
        keys: ['Home', 'End'],
        order: 60,
      },
    });

    // Shift+↑ — Increase speed
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: KeyAction.adjustAnimationSpeed,
      actionParameter: 'increase',
      key: 'ArrowUp',
      modifiers: { shift: true },
      handler: () => this.adjustSelectedDimensionSpeed(+1),
      preventDefault: true,
      description: 'Animation speed up / down',
      help: {
        section: 'dimensions',
        group: 'animation-speed',
        keys: ['⇧', '↑ / ↓'],
        order: 70,
      },
    });

    // Shift+↓ — Decrease speed
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      actionId: KeyAction.adjustAnimationSpeed,
      actionParameter: 'decrease',
      key: 'ArrowDown',
      modifiers: { shift: true },
      handler: () => this.adjustSelectedDimensionSpeed(-1),
      preventDefault: true,
      description: 'Animation speed up / down',
      help: {
        section: 'dimensions',
        group: 'animation-speed',
        keys: ['⇧', '↑ / ↓'],
        order: 70,
      },
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

  private toggleSelectedDimensionPlayback(): boolean {
    const dimIndex = this.resolveDimensionIndex();
    const animManager = this.ctx.getAnimationManager();
    if (dimIndex < 0 || !animManager) return false;
    const isPlaying = animManager.togglePlay(dimIndex);
    log.info(Modules.ANIMATION, `Dimension ${dimIndex} ${isPlaying ? 'playing' : 'paused'}`);
    return true;
  }

  private jumpSelectedDimensionToBound(which: 'start' | 'end'): boolean {
    const dimIndex = this.resolveDimensionIndex();
    if (dimIndex < 0) return false;
    const ranges = sceneDimsManager.getDimensionRanges();
    if (!ranges) return false;
    const value = which === 'start' ? ranges[dimIndex][0] : ranges[dimIndex][1];
    sceneDimsManager.setDimensionValue(dimIndex, value);
    log.info(Modules.ANIMATION, `Jumped to ${which} of dimension ${dimIndex}`);
    return true;
  }

  private adjustSelectedDimensionSpeed(direction: 1 | -1): boolean {
    const dimIndex = this.resolveDimensionIndex();
    const animManager = this.ctx.getAnimationManager();
    if (dimIndex < 0 || !animManager) return false;
    if (direction > 0) {
      animManager.increaseSpeed(dimIndex);
    } else {
      animManager.decreaseSpeed(dimIndex);
    }
    const fps = animManager.getState(dimIndex)?.targetFPS;
    log.info(Modules.ANIMATION, `${direction > 0 ? 'Increased' : 'Decreased'} speed to ${fps} FPS`);
    return true;
  }
}
