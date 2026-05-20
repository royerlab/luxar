/**
 * Camera control-mode command bodies for the V (cycle) and I (inertial)
 * shortcuts, plus the pure `nextControlType` cycle helper. Extracted
 * from input-handler.ts so the orchestrator delegates rather than
 * inlining the SceneManager + InputContextManager + RenderingControls
 * coordination.
 *
 * @module input/input-handler/commands/control-mode
 */

import { log, Modules, LogEmoji } from '../../../utils/log';
import { InputContext, type InputContextManager } from '../context-manager';
import type { SceneManager } from '../../../scene/scene-manager';
import type { RenderingControls } from '../../../ui/rendering-controls';

/** Camera control modes the V key cycles through. */
export type ControlType = 'orbit' | 'fly' | 'ortho';

/**
 * Return the next control type in the cycle: orbit → fly → ortho → orbit.
 * Defensive: any unknown current mode (shouldn't happen in production)
 * resets to `orbit`.
 */
export function nextControlType(current: ControlType | string): ControlType {
  switch (current) {
    case 'orbit':
      return 'fly';
    case 'fly':
      return 'ortho';
    case 'ortho':
      return 'orbit';
    default:
      return 'orbit';
  }
}

export interface ControlModeCtx {
  sceneManager: SceneManager;
  contextManager: InputContextManager;
  renderingControls: RenderingControls | undefined;
}

/**
 * Cycle camera control modes: Orbit → Fly → Ortho → Orbit. Updates the
 * input context to FLY_CONTROLS when switching to fly mode (so WASD
 * keys are enabled); back to NAVIGATION otherwise. Syncs rendering
 * controls display if active.
 */
export function toggleControlMode(ctx: ControlModeCtx): void {
  const currentType = ctx.sceneManager.controls.getControlType();
  log.custom(LogEmoji.CONTROLS, Modules.INPUT, `toggleControlMode called: ${currentType} → ?`);

  const newType = nextControlType(currentType);

  // Use sceneManager.setControlType for ortho (handles camera swap)
  ctx.sceneManager.setControlType(newType);

  // Update input context based on control mode
  if (newType === 'fly') {
    ctx.contextManager.setContext(InputContext.FLY_CONTROLS);
  } else {
    ctx.contextManager.setContext(InputContext.NAVIGATION);
  }

  // Sync rendering controls if they exist
  if (ctx.renderingControls) {
    ctx.renderingControls.syncCurrentState();
  }

  log.custom(
    LogEmoji.CONTROLS,
    Modules.CONTROLS,
    `Switched to ${newType} controls (press V to toggle)`
  );
}

/**
 * Toggle inertial mode for fly controls (momentum-based movement). Only
 * functional in fly control mode; logs an info message and returns
 * otherwise.
 */
export function toggleInertialMode(ctx: ControlModeCtx): void {
  const flyControls = ctx.sceneManager.controls.getFlyControls();
  if (flyControls) {
    const currentInertial = flyControls.inertialMode;
    flyControls.setInertialMode(!currentInertial);

    // Sync rendering controls if they exist
    if (ctx.renderingControls) {
      ctx.renderingControls.syncCurrentState();
    }

    log.custom(
      LogEmoji.ROCKET,
      Modules.CONTROLS,
      `Fly controls inertial mode: ${!currentInertial ? 'ON' : 'OFF'}`
    );
  } else {
    log.info(
      Modules.INPUT,
      'Inertial mode is only available in fly control mode (press V to switch)'
    );
  }
}
