// @vitest-environment jsdom
/**
 * Guard: the element-menu command's dispatched event name stays equal to
 * `OPEN_ELEMENT_MENU_EVENT` (issue #1917).
 *
 * The input layer cannot import the core-layer constant, so the event name is
 * necessarily written twice. A mismatch fails silently while the mouse path
 * still works; this file checks the command event plus the two binding guards.
 */
import { describe, it, expect, vi } from 'vitest';
import { InputHandler } from '../../../../../input/input-handler';
import { OPEN_ELEMENT_MENU_EVENT } from '../../../../../core/app/interaction/canvas-actions';

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

  it('the dispatch is gated on focus being on the scene, not a panel', async () => {
    const source =
      await import('../../../../../input/input-handler/key-bindings/navigation-bindings?raw');
    expect(source.default).toContain('isFocusOnSceneCanvas');
    expect(source.default).toContain('commands.openElementMenu(event)');
  });

  it('both keys are bound on the NAVIGATION context', async () => {
    const source = (
      await import('../../../../../input/input-handler/key-bindings/navigation-bindings?raw')
    ).default;
    const f10 = source.match(
      /registerBinding\(\s*InputContext\.NAVIGATION\s*,\s*\{[^}]*key:\s*'F10'[^}]*\}/s
    );
    const ctxMenu = source.match(
      /registerBinding\(\s*InputContext\.NAVIGATION\s*,\s*\{[^}]*key:\s*'ContextMenu'/s
    );
    expect(f10, 'Shift+F10 not registered on InputContext.NAVIGATION').toBeTruthy();
    expect(ctxMenu, 'ContextMenu not registered on InputContext.NAVIGATION').toBeTruthy();
    expect(f10![0]).toContain('shift: true');
  });
});
