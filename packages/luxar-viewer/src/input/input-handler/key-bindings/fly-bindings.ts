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

export function registerFlyControlBindings(deps: KeyBindingsDeps): void {
  const { contextManager, sceneManager } = deps;
  const getFlyControls = () => sceneManager.controls.getFlyControls();

  const flyMovementKeys = ['w', 'a', 's', 'd', 'q', 'e'];
  const modifierCombinations = [
    {}, // No modifiers
    { shift: true }, // Shift only (speed boost)
    { alt: true }, // Alt only (vertical for W/S)
    { shift: true, alt: true }, // Shift+Alt (fast vertical)
  ];

  for (const key of flyMovementKeys) {
    for (const modifiers of modifierCombinations) {
      contextManager.registerBinding(InputContext.FLY_CONTROLS, {
        key,
        modifiers: Object.keys(modifiers).length > 0 ? modifiers : undefined,
        handler: (event) => getFlyControls()?.handleKeyDown(event),
        keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
        description: `Fly: ${key.toUpperCase()}${
          modifiers.shift ? '+Shift' : ''
        }${modifiers.alt ? '+Alt' : ''}`,
      });
    }
  }

  // Arrow keys for look direction (with and without Shift)
  const arrowKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
  for (const key of arrowKeys) {
    contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key,
      handler: (event) => getFlyControls()?.handleKeyDown(event),
      keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
      description: `Fly look: ${key}`,
    });
    contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key,
      modifiers: { shift: true },
      handler: (event) => getFlyControls()?.handleKeyDown(event),
      keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
      description: `Fly look: ${key}+Shift`,
    });
  }

  // Shift speed boost. Must forward to the fly controls so its internal
  // `speedBoost` flag is set — the WASD bindings above include Shift+key
  // combos that already accelerate movement, but `setSpeedBoost` is what
  // actually doubles the velocity multiplier in physics.ts. (Shift+wheel
  // roll needs no gating here: the fly wheel handler branches on the
  // event's own shiftKey flag.)
  contextManager.registerBinding(InputContext.FLY_CONTROLS, {
    key: 'Shift',
    handler: (event) => getFlyControls()?.handleKeyDown(event),
    keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
    description: 'Speed boost',
  });
}
