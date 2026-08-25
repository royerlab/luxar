/**
 * Fly-mode bindings (FLY_CONTROLS context). Movement keys (WASD, Q, E)
 * need both keydown and keyup handlers — fly controls hold the key
 * state themselves and integrate movement per-frame. Every WASD key is
 * registered with all four supported modifier combinations because the
 * fly controls differentiate behavior on Shift (speed boost) and Alt
 * (vertical for W/S only). Arrow look keys + Shift speed-boost binding
 * round it out.
 */

import { InputContext } from '../context-manager';
import type { KeyBindingsDeps } from './register-all';
import { KeyAction } from './actions';

export function registerFlyControlBindings(deps: KeyBindingsDeps): void {
  const { contextManager, sceneManager } = deps;
  const getFlyControls = () => sceneManager.controls.getFlyControls();
  const forwardToFly = () => ({
    handler: (event: KeyboardEvent) => {
      const controls = getFlyControls();
      if (!controls) return false;
      controls.handleKeyDown(event);
      return true;
    },
    keyupHandler: (event: KeyboardEvent) => {
      const controls = getFlyControls();
      if (!controls) return false;
      controls.handleKeyUp(event);
      return true;
    },
  });

  const flyMovementKeys = [
    ['w', 'forward'],
    ['a', 'left'],
    ['s', 'backward'],
    ['d', 'right'],
    ['q', 'roll-left'],
    ['e', 'roll-right'],
  ] as const;
  const modifierCombinations = [
    {}, // No modifiers
    { shift: true }, // Shift only (speed boost)
    { alt: true }, // Alt only (vertical for W/S)
    { shift: true, alt: true }, // Shift+Alt (fast vertical)
  ];

  for (const [key, direction] of flyMovementKeys) {
    for (const modifiers of modifierCombinations) {
      const modifierId = `${modifiers.shift ? 'fast-' : ''}${modifiers.alt ? 'alt' : 'normal'}`;
      const isBaseMovement = Object.keys(modifiers).length === 0;
      const isVertical = modifiers.alt && (key === 'w' || key === 's');
      const isRoll = key === 'q' || key === 'e';
      const description = isBaseMovement
        ? isRoll
          ? 'Roll left / right'
          : 'Move forward / left / back / right'
        : isVertical
          ? 'Move up / down'
          : `Fly: ${key.toUpperCase()}${modifiers.shift ? '+Shift' : ''}${
              modifiers.alt ? '+Alt' : ''
            }`;
      contextManager.registerBinding(InputContext.FLY_CONTROLS, {
        actionId: KeyAction.flyMove,
        actionParameter: `${direction}.${modifierId}`,
        key,
        modifiers: Object.keys(modifiers).length > 0 ? modifiers : undefined,
        ...forwardToFly(),
        description,
        help: isBaseMovement
          ? {
              section: 'fly',
              group: isRoll ? 'fly-roll' : 'fly-move',
              keys: isRoll ? ['Q / E'] : ['W', 'A', 'S', 'D'],
              order: isRoll ? 50 : 10,
            }
          : isVertical
            ? {
                section: 'fly',
                group: 'fly-vertical',
                keys: ['⌥', 'W / S'],
                order: 20,
              }
            : false,
      });
    }
  }

  // Arrow keys for look direction (with and without Shift)
  const arrowKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
  for (const key of arrowKeys) {
    contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      actionId: KeyAction.flyLook,
      actionParameter: key.toLowerCase(),
      key,
      ...forwardToFly(),
      description: 'Look around',
      help: {
        section: 'fly',
        group: 'fly-look',
        keys: ['↑ ↓ ← →'],
        order: 40,
      },
    });
    contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      actionId: KeyAction.flyLook,
      actionParameter: `${key.toLowerCase()}.fast`,
      key,
      modifiers: { shift: true },
      ...forwardToFly(),
      description: `Fly look: ${key}+Shift`,
      help: false,
    });
  }

  // Shift speed boost. Must forward to the fly controls so its internal
  // `speedBoost` flag is set — the WASD bindings above include Shift+key
  // combos that already accelerate movement, but `setSpeedBoost` is what
  // actually doubles the velocity multiplier in physics.ts. (Shift+wheel
  // roll needs no gating here: the fly wheel handler branches on the
  // event's own shiftKey flag.)
  contextManager.registerBinding(InputContext.FLY_CONTROLS, {
    actionId: KeyAction.flySpeedBoost,
    key: 'Shift',
    ...forwardToFly(),
    description: 'Hold for 2× speed boost',
    help: {
      section: 'fly',
      group: 'fly-speed',
      keys: ['⇧'],
      order: 30,
    },
  });
}
