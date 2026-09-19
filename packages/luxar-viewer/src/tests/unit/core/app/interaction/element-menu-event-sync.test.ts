// @vitest-environment jsdom
/**
 * Guard: the element-menu keyboard path stays wired end to end (issue #1917).
 *
 * The event name is necessarily written twice. The listener lives in
 * `core/app/interaction/canvas-actions.ts`, the dispatch in the InputHandler's
 * shared command surface (`input/input-handler.ts`), and the layer contract in
 * `.dependency-cruiser.cjs` forbids `input/` from importing `core/` — so the
 * command cannot import the constant and hard-codes the string instead.
 * (`OPEN_DATASET_BROWSER_EVENT` is duplicated the same way, for the same reason —
 * see `dataset-browser-event-sync.test.ts`.)
 *
 * The bindings no longer dispatch anything themselves: they gate on focus and
 * route into `commands.openElementMenu`, the same command the on-screen
 * affordances call. That splits the path in two, so both halves are checked
 * here — the command's dispatched event name, and the two chords actually
 * reaching the command.
 *
 * Divergence fails SILENTLY and in the least visible way possible: the keyboard
 * path is the one route with no pointer feedback, so a rename on either side
 * just means Shift+F10 quietly stops working, with the mouse path still fine
 * and every unit test still green.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { OPEN_ELEMENT_MENU_EVENT } from '../../../../../core/app/interaction/canvas-actions';
import { InputHandler } from '../../../../../input';
import {
  InputContext,
  InputContextManager,
} from '../../../../../input/input-handler/context-manager';
import { registerNavigationBindings } from '../../../../../input/input-handler/key-bindings/navigation-bindings';
import type {
  KeyBindingsCommands,
  KeyBindingsDeps,
} from '../../../../../input/input-handler/key-bindings/register-all';

/**
 * Register the navigation bindings against a stub command surface. The
 * injected `openElementMenu` is what the routing tests assert on: the real
 * command's dispatch is covered separately, so stubbing it here keeps the
 * routing assertions about the binding table and its focus gate alone.
 */
function registerBindings(
  canvas: HTMLCanvasElement,
  openElementMenu: KeyBindingsCommands['openElementMenu']
): InputContextManager {
  const contextManager = new InputContextManager();
  registerNavigationBindings({
    contextManager,
    sceneManager: { renderer: { domElement: canvas } },
    debugConsole: {},
    panels: {},
    commands: { openElementMenu },
  } as unknown as KeyBindingsDeps);
  return contextManager;
}

/** A real InputHandler over stub collaborators, so the command runs for real. */
function makeHandler(): InputHandler {
  return new InputHandler(
    {
      renderer: { domElement: document.createElement('canvas') },
      controls: { addEventListener: vi.fn(), removeEventListener: vi.fn() },
    } as never,
    { startAnimation: vi.fn() } as never,
    { toggle: vi.fn(), cycleMode: vi.fn(), visible: false } as never,
    { toggle: vi.fn(), getIsVisible: vi.fn(() => false), dispose: vi.fn() } as never
  );
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

  it('the command dispatches exactly OPEN_ELEMENT_MENU_EVENT', () => {
    const handler = makeHandler();
    handler.init();
    const listener = vi.fn();
    window.addEventListener(OPEN_ELEMENT_MENU_EVENT, listener);
    const event = new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, cancelable: true });

    handler.getUiActions().commands.openElementMenu(event);

    expect(listener).toHaveBeenCalledOnce();
    expect(event.defaultPrevented).toBe(true);
    window.removeEventListener(OPEN_ELEMENT_MENU_EVENT, listener);
    handler.dispose();
  });

  it.each([
    ['F10', { shiftKey: true }],
    ['ContextMenu', {}],
  ])('routes %s to the element-menu command', (key, init) => {
    const openElementMenu = vi.fn();
    const contextManager = registerBindings(canvas, openElementMenu);
    const event = new KeyboardEvent('keydown', { key, bubbles: true, ...init });

    const handled = contextManager.handleKeyEvent(event, 'down');

    expect(handled).toBe(true);
    expect(openElementMenu).toHaveBeenCalledOnce();
    expect(openElementMenu).toHaveBeenCalledWith(event);
  });

  it('does not route when focus is on a panel control', () => {
    // Layer rows bind these same two keys and stop propagation, so the guard is
    // for the other focusable controls outside the canvas: Tab-reaching a panel
    // keeps the cached scene pick alive (the pointer never left the canvas, so
    // no `mouseleave` advanced `pickGeneration`) while `openContextMenu` is
    // module-global — an ungated route would open a canvas menu over unrelated
    // UI. Body focus is deliberately NOT gated: `isFocusOnSceneCanvas` counts
    // `document.body` as scene focus.
    const openElementMenu = vi.fn();
    const contextManager = registerBindings(canvas, openElementMenu);
    const button = document.createElement('button');
    document.body.appendChild(button);
    button.focus();

    contextManager.handleKeyEvent(
      new KeyboardEvent('keydown', { key: 'F10', shiftKey: true, bubbles: true }),
      'down'
    );

    expect(openElementMenu).not.toHaveBeenCalled();
  });

  it('registers both shortcuts on the navigation context', () => {
    // FLY_CONTROLS filters keys through an `allowedKeys` whitelist rebuilt from
    // its own registrations — neither F10 nor ContextMenu is in it — and only
    // reaches a NAVIGATION binding via passthrough, so registering elsewhere
    // would silently disable the shortcut in fly mode.
    const bindings = registerBindings(canvas, vi.fn()).getRegisteredShortcutBindings();

    const chords = (bindings.get(InputContext.NAVIGATION) ?? []).map((binding) => binding.key);
    expect(chords).toEqual(expect.arrayContaining(['f10+shift', 'contextmenu']));
  });
});
