// @vitest-environment jsdom
/**
 * Unit tests for luxar-fly-controls/listeners.ts.
 *
 * Targets audit finding G13 (attachListeners only smoke-tested via the
 * orchestrator). Direct tests verify:
 * (1) keyboard listeners are attached / detached based on
 *     externalInputManagement.
 * (2) mouse + wheel listeners are always attached and always detached.
 * (3) the returned disposer cleanly tears everything down.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  attachListeners,
  type FlyListenersCtx,
} from '../../../../controls/luxar-fly-controls/listeners';

function makeCtx(overrides: Partial<FlyListenersCtx> = {}): FlyListenersCtx {
  return {
    domElement: document.createElement('div'),
    externalInputManagement: false,
    onKeyDown: vi.fn(),
    onKeyUp: vi.fn(),
    onMouseDown: vi.fn(),
    onMouseUp: vi.fn(),
    onMouseMove: vi.fn(),
    onWheel: vi.fn(),
    onPointerDown: vi.fn(),
    onPointerMove: vi.fn(),
    onPointerUp: vi.fn(),
    ...overrides,
  };
}

describe('attachListeners — externalInputManagement = false (default)', () => {
  it('attaches window keydown/keyup handlers', () => {
    const ctx = makeCtx({ externalInputManagement: false });
    const disposer = attachListeners(ctx);

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w' }));
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'w' }));
    expect(ctx.onKeyDown).toHaveBeenCalledTimes(1);
    expect(ctx.onKeyUp).toHaveBeenCalledTimes(1);

    disposer();
  });

  it('disposer detaches window keyboard handlers', () => {
    const ctx = makeCtx({ externalInputManagement: false });
    const disposer = attachListeners(ctx);

    disposer();
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w' }));
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'w' }));
    expect(ctx.onKeyDown).not.toHaveBeenCalled();
    expect(ctx.onKeyUp).not.toHaveBeenCalled();
  });
});

describe('attachListeners — externalInputManagement = true', () => {
  it('does NOT attach window keydown/keyup (boundary)', () => {
    const ctx = makeCtx({ externalInputManagement: true });
    const disposer = attachListeners(ctx);

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w' }));
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'w' }));
    expect(ctx.onKeyDown).not.toHaveBeenCalled();
    expect(ctx.onKeyUp).not.toHaveBeenCalled();

    disposer();
  });

  it('still attaches mouse + wheel handlers (those are NOT gated by the flag)', () => {
    const domElement = document.createElement('div');
    document.body.appendChild(domElement);
    const ctx = makeCtx({ externalInputManagement: true, domElement });
    const disposer = attachListeners(ctx);

    domElement.dispatchEvent(new MouseEvent('mousedown', { button: 0 }));
    expect(ctx.onMouseDown).toHaveBeenCalled();

    disposer();
    document.body.removeChild(domElement);
  });
});

describe('attachListeners — mouse + wheel (always attached)', () => {
  it('attaches mousedown on domElement', () => {
    const domElement = document.createElement('div');
    document.body.appendChild(domElement);
    const ctx = makeCtx({ domElement });
    const disposer = attachListeners(ctx);

    domElement.dispatchEvent(new MouseEvent('mousedown', { button: 0 }));
    expect(ctx.onMouseDown).toHaveBeenCalledTimes(1);

    disposer();
    document.body.removeChild(domElement);
  });

  it('attaches mouseup and mousemove on window (so drag can extend outside the canvas)', () => {
    const ctx = makeCtx();
    const disposer = attachListeners(ctx);

    window.dispatchEvent(new MouseEvent('mouseup', { button: 0 }));
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 50, clientY: 50 }));
    expect(ctx.onMouseUp).toHaveBeenCalledTimes(1);
    expect(ctx.onMouseMove).toHaveBeenCalledTimes(1);

    disposer();
  });

  it('attaches wheel on domElement', () => {
    const domElement = document.createElement('div');
    document.body.appendChild(domElement);
    const ctx = makeCtx({ domElement });
    const disposer = attachListeners(ctx);

    domElement.dispatchEvent(new WheelEvent('wheel', { deltaY: -100 }));
    expect(ctx.onWheel).toHaveBeenCalledTimes(1);

    disposer();
    document.body.removeChild(domElement);
  });

  it('attaches contextmenu handler that calls preventDefault', () => {
    const domElement = document.createElement('div');
    document.body.appendChild(domElement);
    const ctx = makeCtx({ domElement });
    const disposer = attachListeners(ctx);

    const evt = new Event('contextmenu', { cancelable: true });
    const pdSpy = vi.spyOn(evt, 'preventDefault');
    domElement.dispatchEvent(evt);
    expect(pdSpy).toHaveBeenCalled();

    disposer();
    document.body.removeChild(domElement);
  });
});

describe('attachListeners — disposer is comprehensive', () => {
  it('after dispose, NONE of the registered events propagate to handlers', () => {
    const domElement = document.createElement('div');
    document.body.appendChild(domElement);
    const ctx = makeCtx({ domElement, externalInputManagement: false });
    const disposer = attachListeners(ctx);

    disposer();

    // Fire one of each event class.
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w' }));
    window.dispatchEvent(new KeyboardEvent('keyup', { key: 'w' }));
    domElement.dispatchEvent(new MouseEvent('mousedown'));
    window.dispatchEvent(new MouseEvent('mouseup'));
    window.dispatchEvent(new MouseEvent('mousemove'));
    domElement.dispatchEvent(new WheelEvent('wheel'));

    expect(ctx.onKeyDown).not.toHaveBeenCalled();
    expect(ctx.onKeyUp).not.toHaveBeenCalled();
    expect(ctx.onMouseDown).not.toHaveBeenCalled();
    expect(ctx.onMouseUp).not.toHaveBeenCalled();
    expect(ctx.onMouseMove).not.toHaveBeenCalled();
    expect(ctx.onWheel).not.toHaveBeenCalled();

    document.body.removeChild(domElement);
  });
});

describe('attachListeners — touch (pointer events, always attached)', () => {
  it('attaches pointerdown on domElement and pointermove/up/cancel on window', () => {
    const domElement = document.createElement('div');
    document.body.appendChild(domElement);
    const ctx = makeCtx({ domElement });
    const disposer = attachListeners(ctx);

    domElement.dispatchEvent(
      new PointerEvent('pointerdown', { pointerId: 1, pointerType: 'touch' })
    );
    window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 1, pointerType: 'touch' }));
    window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 1, pointerType: 'touch' }));
    window.dispatchEvent(new PointerEvent('pointercancel', { pointerId: 2, pointerType: 'touch' }));
    expect(ctx.onPointerDown).toHaveBeenCalledTimes(1);
    expect(ctx.onPointerMove).toHaveBeenCalledTimes(1);
    expect(ctx.onPointerUp).toHaveBeenCalledTimes(2); // up + cancel

    disposer();
    domElement.dispatchEvent(
      new PointerEvent('pointerdown', { pointerId: 3, pointerType: 'touch' })
    );
    window.dispatchEvent(new PointerEvent('pointermove', { pointerId: 3, pointerType: 'touch' }));
    window.dispatchEvent(new PointerEvent('pointerup', { pointerId: 3, pointerType: 'touch' }));
    expect(ctx.onPointerDown).toHaveBeenCalledTimes(1);
    expect(ctx.onPointerMove).toHaveBeenCalledTimes(1);
    expect(ctx.onPointerUp).toHaveBeenCalledTimes(2);
    document.body.removeChild(domElement);
  });
});
