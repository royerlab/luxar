// @vitest-environment jsdom
/**
 * Guard: the Shift+F10 and ContextMenu bindings stay synchronized with the
 * canvas-actions listener contract (issue #1917).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { OPEN_ELEMENT_MENU_EVENT } from '../../../../../core/app/interaction/canvas-actions';
import {
  InputContext,
  InputContextManager,
} from '../../../../../input/input-handler/context-manager';
import { registerNavigationBindings } from '../../../../../input/input-handler/key-bindings/navigation-bindings';
import type { KeyBindingsDeps } from '../../../../../input/input-handler/key-bindings/register-all';

function registerBindings(canvas: HTMLCanvasElement): InputContextManager {
  const contextManager = new InputContextManager();
  registerNavigationBindings({
    contextManager,
    sceneManager: { renderer: { domElement: canvas } },
    debugConsole: {},
    panels: {},
    commands: {},
  } as unknown as KeyBindingsDeps);
  return contextManager;
}

describe('element-menu keybinding ↔ canvas-actions listener', () => {
  let canvas: HTMLCanvasElement;

  beforeEach(() => {
    canvas = document.createElement('canvas');
    canvas.tabIndex = 0;
    document.body.appendChild(canvas);
    canvas.focus();
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  it.each([
    ['F10', { shiftKey: true }],
    ['ContextMenu', {}],
  ])('dispatches OPEN_ELEMENT_MENU_EVENT for %s', (key, init) => {
    const contextManager = registerBindings(canvas);
    const listener = vi.fn();
    window.addEventListener(OPEN_ELEMENT_MENU_EVENT, listener, { once: true });

    const handled = contextManager.handleKeyEvent(
      new KeyboardEvent('keydown', { key, bubbles: true, ...init }),
      'down'
    );

    expect(handled).toBe(true);
    expect(listener).toHaveBeenCalledOnce();
  });

  it('does not dispatch when focus is on a panel control', () => {
    const contextManager = registerBindings(canvas);
    const button = document.createElement('button');
    document.body.appendChild(button);
    button.focus();
    const listener = vi.fn();
    window.addEventListener(OPEN_ELEMENT_MENU_EVENT, listener, { once: true });

    contextManager.handleKeyEvent(
      new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, bubbles: true }),
      'down'
    );

    expect(listener).not.toHaveBeenCalled();
  });

  it('registers both shortcuts on the navigation context', () => {
    const bindings = registerBindings(canvas).getRegisteredShortcutBindings();

    expect(bindings.get(InputContext.NAVIGATION)).toEqual(
      expect.arrayContaining(['f10+shift', 'contextmenu'])
    );
  });
});
