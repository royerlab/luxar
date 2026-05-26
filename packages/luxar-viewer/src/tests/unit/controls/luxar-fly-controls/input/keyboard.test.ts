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

  it('[controls.md G28] dispatches "change" on UNRECOGNIZED keys too (Space, Tab, /)', () => {
    // controls.md G28[P8]: orbit's keyboard test has 5 cases including
    // "non-arrow"; fly's keyboard previously had 4 ArrowKey tests but no
    // "no recognised key" symmetry coverage. The fly handler unconditionally
    // dispatches `change` at the end of every keydown (keyboard.ts L125).
    // Pin this behaviour so a regression that wrapped dispatch in
    // `if (recognized) {...}` would surface.
    for (const k of ['Space', 'Tab', '/', 'Backquote', 'F1']) {
      const { ctx, moveState } = makeCtx();
      handleKeyDown(ctx, new KeyboardEvent('keydown', { key: k }));
      // moveState DOES NOT change (no recognized binding).
      expect(moveState.forward).toBe(0);
      expect(moveState.back).toBe(0);
      // But dispatch IS called — that's the documented contract.
      expect(ctx.dispatch).toHaveBeenCalledWith('change');
    }
  });
});

describe('handleKeyDown — preventDefault contract (controls.md G13, G14)', () => {
  it('[G13] calls preventDefault on every ArrowKey', () => {
    // controls.md G13: arrow keys are unconditionally consumed by the
    // camera-look pipeline; the orbit-side has this covered, fly did not.
    for (const k of ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) {
      const { ctx } = makeCtx();
      const evt = new KeyboardEvent('keydown', { key: k, cancelable: true });
      const spy = vi.spyOn(evt, 'preventDefault');
      handleKeyDown(ctx, evt);
      expect(spy).toHaveBeenCalledTimes(1);
    }
  });

  it('[G13] calls preventDefault on WASD/QE when NOT typing in an input', () => {
    // controls.md G13: WASD/QE only consume when document.activeElement
    // is not an input/textarea/contenteditable. jsdom default activeElement
    // is body, so the gate falls through to preventDefault.
    for (const k of ['w', 'a', 's', 'd', 'q', 'e', 'W', 'A', 'S', 'D', 'Q', 'E']) {
      const { ctx } = makeCtx();
      const evt = new KeyboardEvent('keydown', { key: k, cancelable: true });
      const spy = vi.spyOn(evt, 'preventDefault');
      handleKeyDown(ctx, evt);
      expect(spy).toHaveBeenCalledTimes(1);
    }
  });

  it('[G14] does NOT call preventDefault on WASD when an <input> has focus', () => {
    // controls.md G14: keyboard.ts:47-59 isTyping gate. If the user is
    // typing in a form field, WASD must pass through to the browser
    // (otherwise the user can't type 'w', 'a', 's', 'd', etc.).
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    try {
      for (const k of ['w', 'a', 's', 'd']) {
        const { ctx } = makeCtx();
        const evt = new KeyboardEvent('keydown', { key: k, cancelable: true });
        const spy = vi.spyOn(evt, 'preventDefault');
        handleKeyDown(ctx, evt);
        expect(spy).not.toHaveBeenCalled();
      }
    } finally {
      input.remove();
    }
  });

  it('[G14] does NOT call preventDefault on WASD when a <textarea> has focus', () => {
    const ta = document.createElement('textarea');
    document.body.appendChild(ta);
    ta.focus();
    try {
      const { ctx } = makeCtx();
      const evt = new KeyboardEvent('keydown', { key: 'w', cancelable: true });
      const spy = vi.spyOn(evt, 'preventDefault');
      handleKeyDown(ctx, evt);
      expect(spy).not.toHaveBeenCalled();
    } finally {
      ta.remove();
    }
  });

  it('[G14] does NOT call preventDefault on WASD when contenteditable element has focus', () => {
    const el = document.createElement('div');
    el.setAttribute('contenteditable', 'true');
    el.tabIndex = 0;
    document.body.appendChild(el);
    el.focus();
    try {
      const { ctx } = makeCtx();
      const evt = new KeyboardEvent('keydown', { key: 'd', cancelable: true });
      const spy = vi.spyOn(evt, 'preventDefault');
      handleKeyDown(ctx, evt);
      expect(spy).not.toHaveBeenCalled();
    } finally {
      el.remove();
    }
  });

  it('[G14] ArrowKeys STILL call preventDefault even when an input has focus (arrow gate is unconditional)', () => {
    // controls.md G14 (symmetric): the isTyping gate in keyboard.ts only
    // applies to WASD/QE. Arrow keys are caught BEFORE the gate (lines
    // 42-44) — they always preventDefault. Pin that distinction.
    const input = document.createElement('input');
    document.body.appendChild(input);
    input.focus();
    try {
      const { ctx } = makeCtx();
      const evt = new KeyboardEvent('keydown', { key: 'ArrowLeft', cancelable: true });
      const spy = vi.spyOn(evt, 'preventDefault');
      handleKeyDown(ctx, evt);
      expect(spy).toHaveBeenCalledTimes(1);
    } finally {
      input.remove();
    }
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
