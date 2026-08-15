// @vitest-environment jsdom
/**
 * Unit tests for FocusManager — outside-click + canvas-refocus owner
 * extracted from the rendering-controls facade.
 *
 * The class is tightly DOM-bound, so the tests drive it through the
 * lifecycle methods (`onPanelShown`, `onPanelHidden`, `dispose`) and
 * synthesised mousedown / focus events. Fake timers cover the
 * deferred-install setTimeout so we can assert the listener is
 * attached at exactly the documented 100 ms boundary.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { FocusManager } from '../../../../ui/rendering-controls/focus-manager';

let panel: HTMLElement;
let canvas: HTMLElement;

beforeEach(() => {
  panel = document.createElement('div');
  document.body.appendChild(panel);

  const innerInput = document.createElement('input');
  innerInput.type = 'text';
  innerInput.id = 'panel-input';
  panel.appendChild(innerInput);

  canvas = document.createElement('canvas');
  canvas.tabIndex = 0; // canvas focus requires tabindex
  document.body.appendChild(canvas);
});

afterEach(() => {
  panel.remove();
  canvas.remove();
});

describe('FocusManager — onPanelShown', () => {
  it('schedules the outside-click listener with a 100 ms delay', () => {
    vi.useFakeTimers();
    const addSpy = vi.spyOn(document, 'addEventListener');
    const fm = new FocusManager({ panel, canvas });

    fm.onPanelShown();

    // Not yet — pre-100ms.
    expect(addSpy).not.toHaveBeenCalledWith('mousedown', expect.any(Function), true);

    vi.advanceTimersByTime(99);
    expect(addSpy).not.toHaveBeenCalledWith('mousedown', expect.any(Function), true);

    vi.advanceTimersByTime(2);
    expect(addSpy).toHaveBeenCalledWith('mousedown', expect.any(Function), true);

    addSpy.mockRestore();
    fm.dispose();
    vi.useRealTimers();
  });

  it('calling onPanelShown twice cancels the first deferred install', () => {
    vi.useFakeTimers();
    const addSpy = vi.spyOn(document, 'addEventListener');
    const fm = new FocusManager({ panel, canvas });

    fm.onPanelShown();
    vi.advanceTimersByTime(50);
    fm.onPanelShown();
    vi.advanceTimersByTime(150);

    // Only the second install should have fired.
    const mousedownCalls = addSpy.mock.calls.filter((c) => c[0] === 'mousedown');
    expect(mousedownCalls.length).toBe(1);

    addSpy.mockRestore();
    fm.dispose();
    vi.useRealTimers();
  });
});

describe('FocusManager — outside-click handling', () => {
  it('clicking outside the panel blurs an input inside the panel', () => {
    vi.useFakeTimers();
    const fm = new FocusManager({ panel, canvas });
    const input = panel.querySelector('input') as HTMLInputElement;
    input.focus();
    expect(document.activeElement).toBe(input);

    fm.onPanelShown();
    vi.advanceTimersByTime(150);

    const outside = document.createElement('div');
    document.body.appendChild(outside);
    outside.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));

    expect(document.activeElement).not.toBe(input);
    outside.remove();
    fm.dispose();
    vi.useRealTimers();
  });

  it('clicking inside the panel does NOT blur the input', () => {
    vi.useFakeTimers();
    const fm = new FocusManager({ panel, canvas });
    const input = panel.querySelector('input') as HTMLInputElement;
    input.focus();

    fm.onPanelShown();
    vi.advanceTimersByTime(150);

    panel.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(document.activeElement).toBe(input);

    fm.dispose();
    vi.useRealTimers();
  });
});

describe('FocusManager — onPanelHidden', () => {
  it('blurs any focused element and refocuses the canvas', () => {
    vi.useFakeTimers();
    const fm = new FocusManager({ panel, canvas });
    const input = panel.querySelector('input') as HTMLInputElement;
    input.focus();
    expect(document.activeElement).toBe(input);

    fm.onPanelHidden();
    expect(document.activeElement).toBe(canvas);

    fm.dispose();
    vi.useRealTimers();
  });

  it('removes the outside-click listener after a prior onPanelShown', () => {
    vi.useFakeTimers();
    const addSpy = vi.spyOn(document, 'addEventListener');
    const removeSpy = vi.spyOn(document, 'removeEventListener');
    const fm = new FocusManager({ panel, canvas });

    fm.onPanelShown();
    vi.advanceTimersByTime(150);
    const installedHandler = addSpy.mock.calls.find((c) => c[0] === 'mousedown')?.[1];

    fm.onPanelHidden();
    expect(removeSpy).toHaveBeenCalledWith('mousedown', installedHandler, true);

    addSpy.mockRestore();
    removeSpy.mockRestore();
    fm.dispose();
    vi.useRealTimers();
  });

  it('cancels a pending deferred install when the panel hides early', () => {
    vi.useFakeTimers();
    const addSpy = vi.spyOn(document, 'addEventListener');
    const fm = new FocusManager({ panel, canvas });

    fm.onPanelShown();
    vi.advanceTimersByTime(50);
    fm.onPanelHidden();
    vi.advanceTimersByTime(200);

    expect(addSpy).not.toHaveBeenCalledWith('mousedown', expect.any(Function), true);
    addSpy.mockRestore();
    fm.dispose();
    vi.useRealTimers();
  });
});

describe('FocusManager — dispose', () => {
  it('idempotent: calling dispose twice does not throw and does not re-call removeEventListener', () => {
    vi.useFakeTimers();
    const removeSpy = vi.spyOn(document, 'removeEventListener');
    const fm = new FocusManager({ panel, canvas });

    fm.onPanelShown();
    vi.advanceTimersByTime(150);

    fm.dispose();
    const firstCount = removeSpy.mock.calls.filter((c) => c[0] === 'mousedown').length;
    expect(() => fm.dispose()).not.toThrow();
    const secondCount = removeSpy.mock.calls.filter((c) => c[0] === 'mousedown').length;
    expect(secondCount).toBe(firstCount);

    removeSpy.mockRestore();
    vi.useRealTimers();
  });

  it('cleans up everything: pending timer + active listener', () => {
    vi.useFakeTimers();
    const addSpy = vi.spyOn(document, 'addEventListener');
    const fm = new FocusManager({ panel, canvas });

    // Mid-deferred-install timer + then dispose immediately.
    fm.onPanelShown();
    vi.advanceTimersByTime(50);
    fm.dispose();
    vi.advanceTimersByTime(200);

    // Listener must NOT have been installed.
    const mousedownCalls = addSpy.mock.calls.filter((c) => c[0] === 'mousedown');
    expect(mousedownCalls.length).toBe(0);

    addSpy.mockRestore();
    vi.useRealTimers();
  });
});
