/**
 * Unit tests for luxar-fly-controls/input/keyboard.ts.
 *
 * Symmetric to luxar-orbit-controls/input/keyboard.test.ts (P8).
 *
 * The fly orchestrator test file already drives these helpers via real
 * KeyboardEvents (`controls.handleKeyDown(...)`), so direct coverage
 * here is narrow: it pins the move-state / look-state mappings as
 * pure-function contracts, ensuring future refactors don't silently
 * change which keys map to which move/look field.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  handleKeyDown,
  handleKeyUp,
  type FlyKeyboardCtx,
  type FlyMoveState,
  type FlyLookState,
} from '../../../../../controls/luxar-fly-controls/input/keyboard';

function makeCtx(overrides: Partial<FlyKeyboardCtx> = {}): {
  ctx: FlyKeyboardCtx;
  moveState: FlyMoveState;
  lookState: FlyLookState;
  speedBoost: { value: boolean };
} {
  const moveState: FlyMoveState = {
    forward: 0,
    back: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0,
  };
  const lookState: FlyLookState = { horizontal: 0, vertical: 0, roll: 0 };
  const speedBoost = { value: false };
  const ctx: FlyKeyboardCtx = {
    enabled: true,
    moveState,
    lookState,
    setSpeedBoost: (v) => {
      speedBoost.value = v;
    },
    dispatch: vi.fn(),
    ...overrides,
  };
  return { ctx, moveState, lookState, speedBoost };
}

describe('handleKeyDown — WASD movement state mapping (P8 symmetry)', () => {
  it('W → forward = 1', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'w' }));
    expect(moveState.forward).toBe(1);
  });

  it('S → back = 1', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 's' }));
    expect(moveState.back).toBe(1);
  });

  it('A → left = 1', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'a' }));
    expect(moveState.left).toBe(1);
  });

  it('D → right = 1', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'd' }));
    expect(moveState.right).toBe(1);
  });

  it('case-insensitive: uppercase W maps the same as lowercase w', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'W' }));
    expect(moveState.forward).toBe(1);
  });
});

describe('handleKeyDown — Alt+W/S → vertical', () => {
  it('Alt+W → up = 1 (NOT forward)', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'w', altKey: true }));
    expect(moveState.up).toBe(1);
    expect(moveState.forward).toBe(0);
  });

  it('Alt+S → down = 1 (NOT back)', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 's', altKey: true }));
    expect(moveState.down).toBe(1);
    expect(moveState.back).toBe(0);
  });

  it('Meta+W also triggers up (macOS Option/Command parity)', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'w', metaKey: true }));
    expect(moveState.up).toBe(1);
    expect(moveState.forward).toBe(0);
  });
});

describe('handleKeyDown — Q/E roll', () => {
  it('Q → roll = -1', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'q' }));
    expect(lookState.roll).toBe(-1);
  });

  it('E → roll = +1 (opposite sign to Q)', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'e' }));
    expect(lookState.roll).toBe(1);
  });
});

describe('handleKeyDown — Arrow keys → look state', () => {
  it('ArrowUp → vertical = -1, horizontal = 0', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'ArrowUp' }));
    expect(lookState.vertical).toBe(-1);
    expect(lookState.horizontal).toBe(0);
  });

  it('ArrowDown → vertical = +1', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'ArrowDown' }));
    expect(lookState.vertical).toBe(1);
  });

  it('ArrowLeft → horizontal = -1', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    expect(lookState.horizontal).toBe(-1);
  });

  it('ArrowRight → horizontal = +1', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(lookState.horizontal).toBe(1);
  });
});

describe('handleKeyDown — Shift speed boost', () => {
  it('Shift → setSpeedBoost(true)', () => {
    const { ctx, speedBoost } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'Shift' }));
    expect(speedBoost.value).toBe(true);
  });
});

describe('handleKeyDown — gating', () => {
  it('does nothing when disabled', () => {
    const { ctx, moveState } = makeCtx({ enabled: false });
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'w' }));
    expect(moveState.forward).toBe(0);
  });

  it('dispatches "change" on every recognized keydown', () => {
    const { ctx } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'w' }));
    expect(ctx.dispatch).toHaveBeenCalledWith('change');
  });
});

describe('handleKeyUp — releases state', () => {
  it('keyup w clears BOTH forward and up (Alt may have been held during down)', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'w', altKey: true })); // up=1
    handleKeyUp(ctx, new KeyboardEvent('keyup', { key: 'w' }));
    expect(moveState.up).toBe(0);
    expect(moveState.forward).toBe(0);
  });

  it('keyup s clears BOTH back and down (symmetric to w)', () => {
    const { ctx, moveState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 's', altKey: true })); // down=1
    handleKeyUp(ctx, new KeyboardEvent('keyup', { key: 's' }));
    expect(moveState.down).toBe(0);
    expect(moveState.back).toBe(0);
  });

  it('keyup ArrowUp / ArrowDown clears vertical', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'ArrowUp' }));
    handleKeyUp(ctx, new KeyboardEvent('keyup', { key: 'ArrowUp' }));
    expect(lookState.vertical).toBe(0);
  });

  it('keyup ArrowLeft / ArrowRight clears horizontal', () => {
    const { ctx, lookState } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    handleKeyUp(ctx, new KeyboardEvent('keyup', { key: 'ArrowRight' }));
    expect(lookState.horizontal).toBe(0);
  });

  it('keyup Shift releases the speed boost', () => {
    const { ctx, speedBoost } = makeCtx();
    handleKeyDown(ctx, new KeyboardEvent('keydown', { key: 'Shift' }));
    handleKeyUp(ctx, new KeyboardEvent('keyup', { key: 'Shift' }));
    expect(speedBoost.value).toBe(false);
  });
});
