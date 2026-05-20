/**
 * Keyboard input handlers for LuxarFlyControls.
 * Extracted from `luxar-fly-controls.ts` so the orchestrator stays
 * focused on lifecycle and the per-frame physics loop.
 *
 * The orchestrator owns the moveState/lookState records (object refs)
 * so the helpers can mutate them in place. Primitive state
 * (`speedBoost`) is read/written via callbacks. Event dispatch goes
 * back through the orchestrator via `ctx.dispatch` to preserve the
 * manager as the dispatch site (Non-Goal 2).
 */

export type FlyMouseAction = 'none' | 'strafe' | 'rotate';

export interface FlyMoveState {
  forward: number;
  back: number;
  left: number;
  right: number;
  up: number;
  down: number;
}

export interface FlyLookState {
  horizontal: number; // -1 for left, 1 for right
  vertical: number; // -1 for up, 1 for down
  roll: number; // -1 for Q (roll left), 1 for E (roll right)
}

export interface FlyKeyboardCtx {
  enabled: boolean;
  moveState: FlyMoveState;
  lookState: FlyLookState;
  setSpeedBoost: (v: boolean) => void;
  dispatch: (type: 'change') => void;
}

export function handleKeyDown(ctx: FlyKeyboardCtx, event: KeyboardEvent): void {
  if (!ctx.enabled) return;

  // Only prevent default for arrow keys (always used for camera look)
  if (event.key.startsWith('Arrow')) {
    event.preventDefault();
  }

  // Only prevent default for WASD if we're not typing in an input field
  const activeElement = document.activeElement;
  const isTyping =
    activeElement &&
    (activeElement.tagName === 'INPUT' ||
      activeElement.tagName === 'TEXTAREA' ||
      activeElement.getAttribute('contenteditable') === 'true');

  if (
    !isTyping &&
    ['w', 'a', 's', 'd', 'q', 'e', 'W', 'A', 'S', 'D', 'Q', 'E'].includes(event.key)
  ) {
    event.preventDefault();
  }

  // WASD for movement
  switch (event.key.toLowerCase()) {
    case 'w':
      if (event.altKey || event.metaKey) {
        ctx.moveState.up = 1; // Alt/Option+W for up
      } else {
        ctx.moveState.forward = 1; // W for forward
      }
      break;
    case 's':
      if (event.altKey || event.metaKey) {
        ctx.moveState.down = 1; // Alt/Option+S for down
      } else {
        ctx.moveState.back = 1; // S for backward
      }
      break;
    case 'a':
      ctx.moveState.left = 1; // A for strafe left
      break;
    case 'd':
      ctx.moveState.right = 1; // D for strafe right
      break;
    case 'q':
      ctx.lookState.roll = -1; // Q for roll left
      break;
    case 'e':
      ctx.lookState.roll = 1; // E for roll right
      break;
  }

  if (event.key === 'Shift') {
    ctx.setSpeedBoost(true);
  }

  // Arrow keys for camera look direction
  switch (event.key) {
    case 'ArrowUp':
      startLookChange(ctx.lookState, 0, -1); // Look up
      break;
    case 'ArrowDown':
      startLookChange(ctx.lookState, 0, 1); // Look down
      break;
    case 'ArrowLeft':
      startLookChange(ctx.lookState, -1, 0); // Look left
      break;
    case 'ArrowRight':
      startLookChange(ctx.lookState, 1, 0); // Look right
      break;
  }

  ctx.dispatch('change');
}

export function handleKeyUp(ctx: FlyKeyboardCtx, event: KeyboardEvent): void {
  if (!ctx.enabled) return;

  // WASD movement release
  switch (event.key.toLowerCase()) {
    case 'w':
      ctx.moveState.forward = 0;
      ctx.moveState.up = 0; // Also clear up in case Alt was held
      break;
    case 's':
      ctx.moveState.back = 0;
      ctx.moveState.down = 0; // Also clear down in case Alt was held
      break;
    case 'a':
      ctx.moveState.left = 0;
      break;
    case 'd':
      ctx.moveState.right = 0;
      break;
    case 'q':
      ctx.lookState.roll = 0;
      break;
    case 'e':
      ctx.lookState.roll = 0;
      break;
  }

  // Release speed boost
  if (event.key === 'Shift') {
    ctx.setSpeedBoost(false);
  }

  // Arrow keys for camera look release
  switch (event.key) {
    case 'ArrowUp':
    case 'ArrowDown':
      ctx.lookState.vertical = 0;
      break;
    case 'ArrowLeft':
    case 'ArrowRight':
      ctx.lookState.horizontal = 0;
      break;
  }

  ctx.dispatch('change');
}

/**
 * Start continuous look change with arrow keys.
 * @param horizontal - Horizontal look direction (-1 left, 1 right)
 * @param vertical - Vertical look direction (-1 up, 1 down)
 */
function startLookChange(lookState: FlyLookState, horizontal: number, vertical: number): void {
  lookState.horizontal = horizontal;
  lookState.vertical = vertical;
}
