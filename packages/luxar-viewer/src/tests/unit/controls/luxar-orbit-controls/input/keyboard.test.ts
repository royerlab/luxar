// @vitest-environment jsdom
/**
 * Unit tests for luxar-orbit-controls/input/keyboard.ts.
 *
 * Targets audit finding G12 (attachKeyboardPan — arrow-key pan dispatch
 * 100% untested) and G28 (orbit ↔ fly keyboard module symmetry).
 */

import { describe, it, expect, vi } from 'vitest';
import {
  attachKeyboardPan,
  type OrbitKeyboardCtx,
} from '../../../../../controls/luxar-orbit-controls/input/keyboard';

function makeCtx(overrides: Partial<OrbitKeyboardCtx> = {}): {
  ctx: OrbitKeyboardCtx;
  pan: ReturnType<typeof vi.fn>;
  state: { enabled: boolean; enablePan: boolean; keyPanSpeed: number };
} {
  const state = { enabled: true, enablePan: true, keyPanSpeed: 7 };
  const pan = vi.fn();
  const ctx: OrbitKeyboardCtx = {
    enabled: () => state.enabled,
    enablePan: () => state.enablePan,
    keyPanSpeed: () => state.keyPanSpeed,
    pan,
    ...overrides,
  };
  return { ctx, pan, state };
}

describe('attachKeyboardPan — arrow dispatch', () => {
  it('ArrowUp dispatches pan(0, +speed)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }));
    expect(pan).toHaveBeenCalledWith(0, 7);

    disposer();
    document.body.removeChild(el);
  });

  it('ArrowDown dispatches pan(0, -speed)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowDown', cancelable: true }));
    expect(pan).toHaveBeenCalledWith(0, -7);

    disposer();
    document.body.removeChild(el);
  });

  it('ArrowLeft dispatches pan(+speed, 0)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowLeft', cancelable: true }));
    expect(pan).toHaveBeenCalledWith(7, 0);

    disposer();
    document.body.removeChild(el);
  });

  it('ArrowRight dispatches pan(-speed, 0)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowRight', cancelable: true }));
    expect(pan).toHaveBeenCalledWith(-7, 0);

    disposer();
    document.body.removeChild(el);
  });

  it('non-arrow keys are ignored', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', cancelable: true }));
    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'Space', cancelable: true }));
    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'Enter', cancelable: true }));

    expect(pan).not.toHaveBeenCalled();

    disposer();
    document.body.removeChild(el);
  });
});

describe('attachKeyboardPan — gating', () => {
  it('does nothing when enabled() returns false', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan, state } = makeCtx();
    state.enabled = false;
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }));
    expect(pan).not.toHaveBeenCalled();

    disposer();
    document.body.removeChild(el);
  });

  it('does nothing when enablePan() returns false', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan, state } = makeCtx();
    state.enablePan = false;
    const disposer = attachKeyboardPan(el, ctx);

    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }));
    expect(pan).not.toHaveBeenCalled();

    disposer();
    document.body.removeChild(el);
  });

  it('reads keyPanSpeed at dispatch time (mutations to the live field take effect)', () => {
    // The contract is that keyPanSpeed is a getter; changing it after
    // attach must affect subsequent dispatches.
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan, state } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    state.keyPanSpeed = 7;
    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }));
    state.keyPanSpeed = 13;
    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }));

    expect(pan).toHaveBeenNthCalledWith(1, 0, 7);
    expect(pan).toHaveBeenNthCalledWith(2, 0, 13);

    disposer();
    document.body.removeChild(el);
  });

  it('disposer detaches the listener (subsequent dispatches are ignored)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx, pan } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    disposer();
    el.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true }));
    expect(pan).not.toHaveBeenCalled();

    document.body.removeChild(el);
  });

  it('calls preventDefault on arrow events (no page scroll)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const { ctx } = makeCtx();
    const disposer = attachKeyboardPan(el, ctx);

    const evt = new KeyboardEvent('keydown', { code: 'ArrowUp', cancelable: true });
    const spy = vi.spyOn(evt, 'preventDefault');
    el.dispatchEvent(evt);
    expect(spy).toHaveBeenCalled();

    disposer();
    document.body.removeChild(el);
  });
});
