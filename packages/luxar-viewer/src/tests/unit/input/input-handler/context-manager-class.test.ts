// @vitest-environment jsdom
/**
 * Unit tests for InputContextManager
 *
 * Tests the input context management system that prevents
 * keyboard conflicts between different UI modes.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  InputContextManager,
  InputContext,
  MAX_KEY_EVENT_DEPTH,
} from '../../../../input/input-handler/context-manager';
import { log } from '../../../../utils/log';
import { registerTestBinding } from './context-manager-test-utils';

describe('InputContextManager', () => {
  let manager: InputContextManager;

  beforeEach(() => {
    manager = new InputContextManager();
  });

  describe('initialization', () => {
    it('should initialize with navigation context', () => {
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
    });

    it('should initialize with empty context stack', () => {
      const debugInfo = manager.getDebugInfo();
      expect(debugInfo.contextStack).toHaveLength(0);
    });

    // input.md O5 / Phase E7: the previous `should be enabled by default`
    // test bundled THREE behaviors (default-enabled dispatch, disable
    // suppression, re-enable resumption) into one `it`. A regression
    // that broke only the re-enable path surfaced as a generic
    // "should be enabled by default" failure that doesn't match the
    // broken behavior. Split into two tests: one pins the documented
    // default; the other pins the `setEnabled` toggle round-trip
    // (true → false → true).
    it('is enabled by default — registered handler fires without an explicit setEnabled call', () => {
      // input.md W3 fix: strengthen the assertion — pin the default enabled
      // state by observing that a registered handler fires when the manager
      // is freshly constructed. A mutation that flipped the default to
      // `false` would otherwise be hidden behind two "returns false"
      // outcomes that look identical.
      //
      // Use a stock navigation shortcut.
      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event, 'down')).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('setEnabled toggles dispatch: enabled → disabled suppresses, disabled → enabled resumes', () => {
      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      // After explicit disable: dispatch is suppressed.
      manager.setEnabled(false);
      const event2 = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event2, 'down')).toBe(false);
      expect(handler).toHaveBeenCalledTimes(0);

      // Re-enable: dispatch resumes.
      manager.setEnabled(true);
      const event3 = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event3, 'down')).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
    });
  });

  describe('context switching', () => {
    it('should switch contexts', () => {
      manager.setContext(InputContext.FLY_CONTROLS);
      expect(manager.getContext()).toBe(InputContext.FLY_CONTROLS);

      manager.setContext(InputContext.TYPING);
      expect(manager.getContext()).toBe(InputContext.TYPING);
    });

    it('should push context to stack', () => {
      manager.pushContext(InputContext.FLY_CONTROLS);
      expect(manager.getContext()).toBe(InputContext.FLY_CONTROLS);

      const debugInfo = manager.getDebugInfo();
      expect(debugInfo.contextStack).toContain(InputContext.NAVIGATION);
    });

    it('should pop context from stack', () => {
      manager.pushContext(InputContext.FLY_CONTROLS);
      manager.pushContext(InputContext.TYPING);

      manager.popContext();
      expect(manager.getContext()).toBe(InputContext.FLY_CONTROLS);

      manager.popContext();
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
    });

    it('should not push same context twice', () => {
      // input.md W4 fix: pin not just the stack-length invariant, but
      // (a) the actual stack contents (no duplicate FLY_CONTROLS) and
      // (b) that popping returns to the original context exactly once.
      // The pre-strengthening test only asserted "length unchanged",
      // which a mutation that grew the stack with a duplicate would
      // still satisfy if the no-op guard short-circuited differently.
      manager.pushContext(InputContext.FLY_CONTROLS);
      const before = manager.getDebugInfo().contextStack.slice();
      expect(before).toEqual([InputContext.NAVIGATION]); // saved prior

      manager.pushContext(InputContext.FLY_CONTROLS); // Same context — no-op
      const after = manager.getDebugInfo().contextStack;
      expect(after).toEqual(before); // identical contents, not just length

      // Single pop returns to the saved prior context — confirms no
      // duplicate FLY_CONTROLS was silently pushed.
      manager.popContext();
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
      expect(manager.getDebugInfo().contextStack).toHaveLength(0);
    });

    it('should handle empty stack on pop', () => {
      // Should not throw
      expect(() => manager.popContext()).not.toThrow();
      // Should remain in current context
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
    });
  });

  describe('custom context lifecycle', () => {
    it('rejects an empty context identifier', () => {
      expect(() => manager.registerContext('', { priority: 5 })).toThrow(
        'Input context identifier cannot be empty'
      );
    });

    it('registers a copied config and routes push → handle → pop with explicit fallback', () => {
      const allowedKeys = ['x'];
      const fallbackContexts = [InputContext.NAVIGATION];
      manager.registerContext('annotation', {
        priority: 5,
        passthrough: true,
        allowedKeys,
        fallbackContexts,
      });
      const annotationHandler = vi.fn();
      const navigationHandler = vi.fn();
      registerTestBinding(manager, 'annotation', { key: 'x', handler: annotationHandler });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: navigationHandler,
      });

      allowedKeys.splice(0, allowedKeys.length, 'y');
      fallbackContexts.length = 0;
      manager.pushContext('annotation');

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'x' }), 'down')).toBe(true);
      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'h' }), 'down')).toBe(true);
      expect(annotationHandler).toHaveBeenCalledOnce();
      expect(navigationHandler).toHaveBeenCalledOnce();
      manager.popContext();
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
    });

    it('rejects duplicate custom and built-in context names', () => {
      manager.registerContext('annotation', { priority: 5 });
      expect(() => manager.registerContext('annotation', { priority: 6 })).toThrow(
        'Input context "annotation" is already registered'
      );
      expect(() => manager.registerContext(InputContext.TYPING, { priority: 1 })).toThrow(
        'Input context "typing" is already registered'
      );
    });

    it('rejects authored and derived allowed-key filters together', () => {
      expect(() =>
        manager.registerContext('annotation', {
          priority: 5,
          allowedKeys: ['x'],
          allowRegisteredBindings: true,
        })
      ).toThrow(
        'Input context "annotation" cannot define allowedKeys with allowRegisteredBindings'
      );
    });

    it('rejects unknown fallback contexts but allows built-in priority ties', () => {
      expect(() =>
        manager.registerContext('annotation', {
          priority: 5,
          fallbackContexts: ['misspelled-navigation'],
        })
      ).toThrow('Input context "misspelled-navigation" is not registered');

      expect(() => manager.registerContext('annotation', { priority: 5 })).not.toThrow();
    });

    it('leaves the context stack unchanged when an unknown push is rejected', () => {
      expect(() => manager.pushContext('missing')).toThrow(
        'Input context "missing" is not registered'
      );
      expect(manager.getDebugInfo().contextStack).toEqual([]);
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
    });

    it('unregisters custom bindings and rejects later activation', () => {
      manager.registerContext('annotation', { priority: 5 });
      registerTestBinding(manager, 'annotation', { key: 'x', handler: vi.fn() });
      manager.pushContext('annotation');
      manager.popContext();
      manager.unregisterContext('annotation');

      expect(manager.getRegisteredShortcutBindings().has('annotation')).toBe(false);
      expect(() => manager.pushContext('annotation')).toThrow(
        'Input context "annotation" is not registered'
      );
      expect(() => manager.unregisterContext(InputContext.NAVIGATION)).toThrow(
        'Built-in input context "navigation" cannot be unregistered'
      );
    });

    it('treats unregistering an unknown custom context as a no-op', () => {
      expect(() => manager.unregisterContext('missing')).not.toThrow();
      expect(manager.getContext()).toBe(InputContext.NAVIGATION);
    });

    it('removes an unregistered custom context from custom fallback routes', () => {
      const handler = vi.fn();
      manager.registerContext('annotation-base', { priority: 4 });
      manager.registerContext('annotation-overlay', {
        priority: 5,
        passthrough: true,
        fallbackContexts: ['annotation-base'],
      });
      registerTestBinding(manager, 'annotation-base', { key: 'x', handler });
      manager.unregisterContext('annotation-base');
      manager.pushContext('annotation-overlay');

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'x' }), 'down')).toBe(
        false
      );
      expect(handler).not.toHaveBeenCalled();
    });

    it('copies blocked keys instead of retaining the caller array', () => {
      const blockedKeys = ['x'];
      const handler = vi.fn();
      manager.registerContext('annotation', { priority: 5, blockedKeys });
      registerTestBinding(manager, 'annotation', { key: 'x', handler });
      blockedKeys.length = 0;
      manager.pushContext('annotation');

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'x' }), 'down')).toBe(
        false
      );
      expect(handler).not.toHaveBeenCalled();
    });

    it('refuses to unregister a context that is still active', () => {
      manager.registerContext('annotation', { priority: 5 });
      manager.pushContext('annotation');

      expect(() => manager.unregisterContext('annotation')).toThrow(
        'Input context "annotation" cannot be unregistered while active'
      );
      expect(manager.getContext()).toBe('annotation');
    });

    it('reset removes custom contexts as well as their bindings', () => {
      manager.registerContext('annotation', { priority: 5 });
      registerTestBinding(manager, 'annotation', { key: 'x', handler: vi.fn() });
      manager.reset();

      expect(() => manager.setContext('annotation')).toThrow(
        'Input context "annotation" is not registered'
      );
      expect(manager.getRegisteredShortcutBindings().has('annotation')).toBe(false);
    });

    it('keeps built-in Escape precedence while typing', () => {
      const navigationEscape = vi.fn();
      const customEscape = vi.fn();
      manager.registerContext('annotation', { priority: 100 });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: navigationEscape,
      });
      registerTestBinding(manager, 'annotation', { key: 'Escape', handler: customEscape });
      manager.setContext(InputContext.TYPING);

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'Escape' }), 'down')).toBe(
        true
      );
      expect(navigationEscape).toHaveBeenCalledOnce();
      expect(customEscape).not.toHaveBeenCalled();
    });

    it('ignores inactive custom Escape bindings when built-in handlers decline', () => {
      const customEscape = vi.fn();
      manager.registerContext('annotation', { priority: 100 });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: () => false,
      });
      registerTestBinding(manager, 'annotation', { key: 'Escape', handler: customEscape });
      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'Escape' }), 'down')).toBe(
        false
      );
      expect(customEscape).not.toHaveBeenCalled();
      input.remove();
    });

    it('uses the active custom Escape binding when built-in handlers decline', () => {
      const customEscape = vi.fn();
      manager.registerContext('annotation', { priority: 100 });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: () => false,
      });
      registerTestBinding(manager, 'annotation', { key: 'Escape', handler: customEscape });
      manager.pushContext('annotation');
      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'Escape' }), 'down')).toBe(
        true
      );
      expect(customEscape).toHaveBeenCalledOnce();
      input.remove();
    });

    it('warns after registration when an authored filter makes the binding unreachable', () => {
      const warning = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
      manager.registerContext('annotation', { priority: 5, allowedKeys: ['y'] });

      registerTestBinding(manager, 'annotation', { key: 'x', handler: vi.fn() });

      expect(warning).toHaveBeenCalledWith(
        expect.anything(),
        'Key binding x is unreachable in context annotation'
      );
      warning.mockRestore();
    });
  });

  describe('key binding registration', () => {
    it('reports canonical registered binding keys by context', () => {
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 's',
        modifiers: { ctrl: true, shift: true },
        handler: vi.fn(),
      });

      expect(manager.getRegisteredShortcutBindings().get(InputContext.NAVIGATION)).toEqual([
        expect.objectContaining({
          key: 'ctrl+s+shift',
          description: 'Test binding',
          help: false,
        }),
      ]);
    });

    it('rebinds a stable action identity to a new chord', () => {
      const first = vi.fn();
      const second = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        actionId: 'test.rebind',
        key: 'h',
        handler: first,
      });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        actionId: 'test.rebind',
        key: 'j',
        handler: second,
      });

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'h' }), 'down')).toBe(
        false
      );
      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'j' }), 'down')).toBe(true);
      expect(first).not.toHaveBeenCalled();
      expect(second).toHaveBeenCalledOnce();
      expect(manager.getShortcutLabel('test.rebind')).toBe('J');
    });

    it.each([
      ['F10', undefined, 'F10'],
      ['Home', undefined, 'Home'],
      ['End', undefined, 'End'],
      ['ArrowUp', { shift: true }, 'Shift+ArrowUp'],
      ['ContextMenu', undefined, 'ContextMenu'],
      [' ', { shift: true }, 'Shift+Space'],
    ])('formats %s bindings for display', (key, modifiers, expected) => {
      registerTestBinding(manager, InputContext.NAVIGATION, {
        actionId: 'test.format',
        key,
        modifiers,
        handler: vi.fn(),
      });

      expect(manager.getShortcutLabel('test.format')).toBe(expected);
    });

    it('resolves duplicate actions from the active context before fallbacks', () => {
      registerTestBinding(manager, InputContext.NAVIGATION, {
        actionId: 'test.shared',
        key: 'n',
        handler: vi.fn(),
      });
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        actionId: 'test.shared',
        key: 'f',
        handler: vi.fn(),
      });

      manager.setContext(InputContext.FLY_CONTROLS);
      expect(manager.getShortcutLabel('test.shared')).toBe('F');
      manager.setContext(InputContext.NAVIGATION);
      expect(manager.getShortcutLabel('test.shared')).toBe('N');
    });
    it('should register key bindings', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
        preventDefault: true,
        description: 'Show help',
      });

      const debugInfo = manager.getDebugInfo();
      const navBindings = debugInfo.registeredBindings.get(InputContext.NAVIGATION);
      expect(navBindings).toContainEqual(expect.objectContaining({ key: 'h' }));
    });

    it('should register bindings with modifiers', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'l',
        modifiers: { ctrl: true },
        handler,
        preventDefault: true,
      });

      const debugInfo = manager.getDebugInfo();
      const navBindings = debugInfo.registeredBindings.get(InputContext.NAVIGATION);
      expect(navBindings).toContainEqual(expect.objectContaining({ key: 'ctrl+l' }));
    });

    it('[input.md C5] registered binding actually fires when matching event is dispatched (end-to-end)', () => {
      // input.md C5[P2]: prior tests only assert via getDebugInfo() that a
      // bindingKey appears in the registered list — a single-channel
      // probe. They don't end-to-end the lookup: a regression that wrote
      // bindings into a different Map but kept the debugInfo accessor
      // honest would survive. Pin the lookup-plus-dispatch contract.
      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
        preventDefault: true,
      });
      // Use a stock navigation shortcut.
      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');
      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
      expect(handler).toHaveBeenCalledWith(event);
    });

    it('[input.md C5] modifier-binding fires only when the modifier matches (Ctrl+l)', () => {
      // Symmetric end-to-end for the modifier-registration path: a
      // ctrl+l binding must NOT fire on plain 'l' (different bindingKey),
      // AND must fire on Ctrl+l. Catches a regression that ignored
      // modifiers when composing the lookup key.
      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'l',
        modifiers: { ctrl: true },
        handler,
        preventDefault: true,
      });

      // Plain 'l' — wrong bindingKey, no dispatch.
      const plain = new KeyboardEvent('keydown', { key: 'l' });
      expect(manager.handleKeyEvent(plain, 'down')).toBe(false);
      expect(handler).not.toHaveBeenCalled();

      // Ctrl+l — matches.
      const ctrlL = new KeyboardEvent('keydown', { key: 'l', ctrlKey: true });
      expect(manager.handleKeyEvent(ctrlL, 'down')).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('fires a modifier-only binding when the modifier is pressed alone', () => {
      // When Shift (or any other modifier) is pressed alone, the keydown
      // event reports BOTH `event.key === 'Shift'` AND `event.shiftKey === true`.
      // A naive lookup that concatenates both produces "shift+shift", which
      // never matches the registered "shift" binding key and silently breaks
      // modifier-only handlers (e.g. fly-mode speed boost). The lookup must
      // skip the modifier flag when it duplicates the pressed key.
      const shiftHandler = vi.fn();
      const ctrlHandler = vi.fn();
      const altHandler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Shift',
        handler: shiftHandler,
      });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Control',
        handler: ctrlHandler,
      });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Alt',
        handler: altHandler,
      });

      expect(
        manager.handleKeyEvent(
          new KeyboardEvent('keydown', { key: 'Shift', shiftKey: true }),
          'down'
        )
      ).toBe(true);
      expect(shiftHandler).toHaveBeenCalledTimes(1);

      expect(
        manager.handleKeyEvent(
          new KeyboardEvent('keydown', { key: 'Control', ctrlKey: true }),
          'down'
        )
      ).toBe(true);
      expect(ctrlHandler).toHaveBeenCalledTimes(1);

      expect(
        manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'Alt', altKey: true }), 'down')
      ).toBe(true);
      expect(altHandler).toHaveBeenCalledTimes(1);
    });

    it('should warn about conflicts', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: handler1,
      });

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: handler2,
      });

      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Key binding conflict'));

      warnSpy.mockRestore();
    });

    it('should unregister bindings', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      manager.unregisterBinding(InputContext.NAVIGATION, 'h');

      const debugInfo = manager.getDebugInfo();
      const navBindings = debugInfo.registeredBindings.get(InputContext.NAVIGATION);
      expect(navBindings).not.toContain('h');
    });
  });

  describe('key event handling', () => {
    it('routes a key UI_INTERACTION does not claim through to NAVIGATION, reporting it handled', () => {
      const globalShortcut = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, { key: 'h', handler: globalShortcut });
      manager.pushContext(InputContext.UI_INTERACTION);

      expect(manager.handleKeyEvent(new KeyboardEvent('keydown', { key: 'h' }), 'down')).toBe(true);
      expect(globalShortcut).toHaveBeenCalledTimes(1);
    });

    it('falls through when a matching handler explicitly declines the event', () => {
      const higher = vi.fn(() => false);
      const lower = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, { key: 'h', handler: higher });
      registerTestBinding(manager, InputContext.NAVIGATION, { key: 'h', handler: lower });
      manager.setContext(InputContext.UI_INTERACTION);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event, 'down')).toBe(true);
      expect(higher).toHaveBeenCalledWith(event);
      expect(lower).toHaveBeenCalledWith(event);
    });

    it('does not visit contexts outside the active fallback route', () => {
      const intermediate = vi.fn(() => false);
      const lower = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'h',
        handler: intermediate,
      });
      registerTestBinding(manager, InputContext.NAVIGATION, { key: 'h', handler: lower });
      manager.setContext(InputContext.FLY_CONTROLS);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event, 'down')).toBe(true);
      expect(intermediate).not.toHaveBeenCalled();
      expect(lower).toHaveBeenCalledWith(event);
    });

    it('falls through on keyup when a keyup handler declines', () => {
      const higher = vi.fn(() => false);
      const lower = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'h',
        handler: vi.fn(),
        keyupHandler: higher,
      });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: vi.fn(),
        keyupHandler: lower,
      });
      manager.setContext(InputContext.UI_INTERACTION);

      const event = new KeyboardEvent('keyup', { key: 'h' });
      expect(manager.handleKeyEvent(event, 'up')).toBe(true);
      expect(higher).toHaveBeenCalledWith(event);
      expect(lower).toHaveBeenCalledWith(event);
    });

    it('falls through on keyup when the current binding has no keyup handler', () => {
      const lower = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'h',
        handler: vi.fn(),
      });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: vi.fn(),
        keyupHandler: lower,
      });
      manager.setContext(InputContext.UI_INTERACTION);

      const event = new KeyboardEvent('keyup', { key: 'h' });
      expect(manager.handleKeyEvent(event, 'up')).toBe(true);
      expect(lower).toHaveBeenCalledWith(event);
    });

    it('skips lower-context bindings without keyup handlers', () => {
      const lower = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'h',
        handler: vi.fn(),
      });
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: vi.fn(),
        keyupHandler: lower,
      });
      manager.setContext(InputContext.FLY_CONTROLS);

      const event = new KeyboardEvent('keyup', { key: 'h' });
      expect(manager.handleKeyEvent(event, 'up')).toBe(true);
      expect(lower).toHaveBeenCalledWith(event);
    });

    it('continues Escape routing from typing when a higher context declines', () => {
      const higher = vi.fn(() => false);
      const lower = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'Escape',
        handler: higher,
      });
      registerTestBinding(manager, InputContext.NAVIGATION, { key: 'Escape', handler: lower });
      manager.setContext(InputContext.TYPING);

      const event = new KeyboardEvent('keydown', { key: 'Escape' });
      expect(manager.handleKeyEvent(event, 'down')).toBe(true);
      expect(higher).toHaveBeenCalledWith(event);
      expect(lower).toHaveBeenCalledWith(event);
    });

    it('does not prevent default when a handler declines the event', () => {
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: () => false,
        preventDefault: true,
      });
      const event = new KeyboardEvent('keydown', { key: 'h', cancelable: true });

      expect(manager.handleKeyEvent(event, 'down')).toBe(false);
      expect(event.defaultPrevented).toBe(false);
    });
    it('should handle registered key events', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
        preventDefault: true,
      });

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const preventDefaultSpy = vi.spyOn(event, 'preventDefault');

      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalledWith(event);
      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should ignore unregistered keys', () => {
      const event = new KeyboardEvent('keydown', { key: 'x' });

      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(false);
    });

    it('should handle keys with modifiers', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'l',
        modifiers: { ctrl: true },
        handler,
      });

      const event = new KeyboardEvent('keydown', {
        key: 'l',
        ctrlKey: true,
      });

      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalledWith(event);
    });

    it('should not handle when disabled', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      manager.setEnabled(false);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('typing context', () => {
    it('should block keys in typing context', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      manager.setContext(InputContext.TYPING);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true); // Blocked
      expect(handler).not.toHaveBeenCalled();
    });

    it('should allow Escape in typing context', () => {
      manager.setContext(InputContext.TYPING);

      const event = new KeyboardEvent('keydown', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(false); // Not blocked
    });

    it('should detect typing in input elements', () => {
      // Create an input element and focus it
      const input = document.createElement('input');
      document.body.appendChild(input);
      input.focus();

      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true); // Blocked because typing
      expect(handler).not.toHaveBeenCalled();

      document.body.removeChild(input);
    });

    it('should detect typing in textarea', () => {
      const textarea = document.createElement('textarea');
      document.body.appendChild(textarea);
      textarea.focus();

      const event = new KeyboardEvent('keydown', { key: 'a' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true); // Blocked

      document.body.removeChild(textarea);
    });

    it('should detect contenteditable elements', () => {
      const div = document.createElement('div');
      div.setAttribute('contenteditable', 'true');
      document.body.appendChild(div);
      div.focus();

      const event = new KeyboardEvent('keydown', { key: 'a' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true); // Blocked

      document.body.removeChild(div);
    });
  });

  describe('context-specific key filtering', () => {
    it('should respect allowed keys', () => {
      // FLY_CONTROLS context has specific allowed keys
      manager.setContext(InputContext.FLY_CONTROLS);

      const handler = vi.fn();
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'w',
        handler,
      });

      // 'w' is in allowed keys for FLY_CONTROLS
      const event = new KeyboardEvent('keydown', { key: 'w' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });

    it('dispatches a NAVIGATION binding that shares a chord with FLY_CONTROLS', () => {
      manager.setContext(InputContext.NAVIGATION);

      const navigationHandler = vi.fn();
      const flyHandler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'w',
        handler: navigationHandler,
      });
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'w',
        handler: flyHandler,
      });

      const event = new KeyboardEvent('keydown', { key: 'w' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(navigationHandler).toHaveBeenCalledOnce();
      expect(flyHandler).not.toHaveBeenCalled();
    });
  });

  describe('context priority and passthrough', () => {
    it('should pass through keys not in allowed list when passthrough enabled', () => {
      const handler = vi.fn();

      // Register in NAVIGATION context
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      // Switch to FLY_CONTROLS which only allows WASD and arrow keys
      // BUT has passthrough enabled, so 'h' should fall through to NAVIGATION
      manager.setContext(InputContext.FLY_CONTROLS);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      // 'h' is not in FLY_CONTROLS allowedKeys, but passthrough is enabled
      // so it should fall through to NAVIGATION and be handled there
      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });

    it('should pass through unhandled keys in permissive contexts', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: '[',
        handler,
      });

      manager.setContext(InputContext.UI_INTERACTION);

      const event = new KeyboardEvent('keydown', { key: '[' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
    });
  });

  describe('debug information', () => {
    it('should provide debug info', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: handler1,
      });

      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'w',
        handler: handler2,
      });

      manager.pushContext(InputContext.FLY_CONTROLS);

      const debugInfo = manager.getDebugInfo();

      expect(debugInfo.currentContext).toBe(InputContext.FLY_CONTROLS);
      expect(debugInfo.contextStack).toContain(InputContext.NAVIGATION);
      expect(debugInfo.registeredBindings.get(InputContext.NAVIGATION)).toContainEqual(
        expect.objectContaining({ key: 'h' })
      );
      expect(debugInfo.registeredBindings.get(InputContext.FLY_CONTROLS)).toContainEqual(
        expect.objectContaining({ key: 'w' })
      );
    });
  });

  describe('cleanup', () => {
    it('should clear context bindings', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      manager.clearContextBindings(InputContext.NAVIGATION);

      const debugInfo = manager.getDebugInfo();
      const navBindings = debugInfo.registeredBindings.get(InputContext.NAVIGATION);
      expect(navBindings).toBeUndefined();
    });

    it('should reset to default state', () => {
      // Make changes
      manager.setContext(InputContext.FLY_CONTROLS);
      manager.pushContext(InputContext.TYPING);
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: vi.fn(),
      });

      // Reset
      manager.reset();

      expect(manager.getContext()).toBe(InputContext.NAVIGATION);

      const debugInfo = manager.getDebugInfo();
      expect(debugInfo.contextStack).toHaveLength(0);
      expect(debugInfo.registeredBindings.size).toBe(0);
    });
  });

  describe('case sensitivity', () => {
    it('should handle keys case-insensitively', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h', // lowercase
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'H' }); // uppercase
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });
  });

  // input.md G8 fix: pin the exact value of the MAX_KEY_EVENT_DEPTH
  // constant so a refactor that changed it (e.g. to 5 or 100) would
  // surface the impact on documented behavior. 10 is "comfortably
  // above any realistic UI depth" per the source comment.
  describe('MAX_KEY_EVENT_DEPTH boundary', () => {
    it('is exactly 10 (the documented depth ceiling)', () => {
      expect(MAX_KEY_EVENT_DEPTH).toBe(10);
    });

    it('bails with a logged error when handleKeyEvent recurses past the limit', () => {
      const localMgr = new InputContextManager();
      // Register a handler that re-dispatches the same keyboard event
      // through handleKeyEvent — would recurse infinitely without the
      // depth guard.
      const handler = vi.fn((event: KeyboardEvent) => {
        localMgr.handleKeyEvent(event, 'down');
      });
      registerTestBinding(localMgr, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const result = localMgr.handleKeyEvent(event, 'down');
      // The outer call dispatches, the recursion fires up to the cap,
      // then handleKeyEvent returns false at the cap. Result of the
      // outermost call is `true` (the binding fired).
      expect(result).toBe(true);
      // Handler should have been called exactly MAX_KEY_EVENT_DEPTH
      // times before the guard short-circuits.
      expect(handler.mock.calls.length).toBe(MAX_KEY_EVENT_DEPTH);
    });
  });

  // input.md G6 fix: Escape from a typing context is routed through
  // dispatchEscapeFromTypingContext, which fires the first matching Escape
  // binding across every context (including the current one) so a panel can
  // close. This pins that post-refactor contract.
  describe('Escape in typing context (post-refactor contract)', () => {
    it('dispatches Escape through dispatchEscapeFromTypingContext to the navigation binding', () => {
      const escapeHandler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: escapeHandler,
      });
      manager.setContext(InputContext.TYPING);

      const event = new KeyboardEvent('keydown', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'down');

      // Post-refactor: Escape MUST be dispatched to the registered
      // NAVIGATION binding so the panel can close.
      expect(handled).toBe(true);
      expect(escapeHandler).toHaveBeenCalledWith(event);
    });
  });

  // input.md G7 fix: passthrough cascade — when the current context
  // has passthrough enabled and the key isn't bound there, the event
  // should walk lower-priority contexts. Pin the cascade ordering.
  describe('passthrough cascade', () => {
    it('cascades a key through context priority order until a binding fires', () => {
      const navHandler = vi.fn();
      const dimHandler = vi.fn();

      // Register on the lowest-priority context (NAVIGATION).
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: navHandler,
      });

      // Stand in FLY_CONTROLS (priority 1, passthrough). 'h' is not
      // in the FLY_CONTROLS registered-chord allowlist, so it falls through.
      manager.setContext(InputContext.FLY_CONTROLS);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(navHandler).toHaveBeenCalledWith(event);
      expect(dimHandler).not.toHaveBeenCalled();
    });

    it('does not route from navigation into fly controls', () => {
      const flyHandler = vi.fn();
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'ArrowUp',
        handler: flyHandler,
      });

      const event = new KeyboardEvent('keydown', { key: 'ArrowUp' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(false);
      expect(flyHandler).not.toHaveBeenCalled();
    });

    it('does NOT cascade when the current context disables passthrough', () => {
      const navHandler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler: navHandler,
      });

      // TYPING context has passthrough: false, allowedKeys: [].
      // The "Escape special-case" routes through dispatchEscapeFromTypingContext;
      // for any non-Escape key, the typing branch returns true without
      // dispatching to other contexts.
      manager.setContext(InputContext.TYPING);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true); // Blocked by typing
      expect(navHandler).not.toHaveBeenCalled();
    });
  });

  // input.md G13 fix: typing-context detection examines
  // document.activeElement directly. The existing tests pin the
  // contenteditable / input / textarea cases; the missing branches
  // are (a) range / checkbox / radio inputs MUST NOT block shortcuts,
  // and (b) the select element MUST block.
  describe('typing-context detection — input type variants', () => {
    let probe: HTMLElement | null = null;
    afterEach(() => {
      if (probe?.parentNode) probe.parentNode.removeChild(probe);
      probe = null;
    });

    it('does NOT treat range inputs (sliders) as typing context', () => {
      const range = document.createElement('input');
      range.type = 'range';
      document.body.appendChild(range);
      range.focus();
      probe = range;

      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      // Shortcut still fires — range is interactive, not text entry.
      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });

    it('does NOT treat checkbox inputs as typing context', () => {
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      document.body.appendChild(cb);
      cb.focus();
      probe = cb;

      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });

    it('treats <select> as typing context (blocks shortcuts)', () => {
      const select = document.createElement('select');
      document.body.appendChild(select);
      select.focus();
      probe = select;

      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true); // blocked
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('keydown vs keyup behavior', () => {
    it('should call handler on keydown', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'c',
        handler,
      });

      const keydownEvent = new KeyboardEvent('keydown', { key: 'c' });
      const handled = manager.handleKeyEvent(keydownEvent, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('should NOT call handler on keyup if no keyupHandler provided', () => {
      const handler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'c',
        handler,
        // No keyupHandler
      });

      const keyupEvent = new KeyboardEvent('keyup', { key: 'c' });
      const handled = manager.handleKeyEvent(keyupEvent, 'up');

      // Should return false (not handled) because no keyupHandler
      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });

    it('should NOT double-call handler for toggle actions (correct behavior)', () => {
      let toggleState = false;
      const toggleHandler = vi.fn(() => {
        toggleState = !toggleState;
      });

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'c',
        handler: toggleHandler,
        // No keyupHandler - toggle actions only trigger on keydown
      });

      // Keydown - should toggle to true
      const keydownEvent = new KeyboardEvent('keydown', { key: 'c' });
      manager.handleKeyEvent(keydownEvent, 'down');
      expect(toggleState).toBe(true);
      expect(toggleHandler).toHaveBeenCalledTimes(1);

      // Keyup - should NOT call handler (no keyupHandler provided)
      const keyupEvent = new KeyboardEvent('keyup', { key: 'c' });
      const handled = manager.handleKeyEvent(keyupEvent, 'up');

      // Handler should NOT be called on keyup
      expect(handled).toBe(false);
      expect(toggleHandler).toHaveBeenCalledTimes(1); // Still only once
      expect(toggleState).toBe(true); // Stays ON (doesn't toggle back)
    });
  });

  // ─────────────────────────────────────────────────────────────────────
  // HIGH-9 regression: typing-detection paths must not diverge.
  //
  // Before this fix, context-manager.ts had its own inlined DOM check
  // duplicating focus-utils.ts::isTypingInInput. A fix to one path
  // would silently miss the other. Both paths now delegate to the
  // canonical focus-utils helper.
  //
  // Strategy: for each canonical activeElement classification, assert
  // that InputContextManager's typing-context branch (observable via
  // handleKeyEvent dropping non-Escape keys while a typing surface is
  // focused) AGREES with the focus-utils helper's verdict.
  // ─────────────────────────────────────────────────────────────────────
  describe('HIGH-9: typing-detection agrees with focus-utils helper', () => {
    afterEach(() => {
      document.body.innerHTML = '';
    });

    // Helper: probe the manager's view of "am I typing?" by registering
    // a non-Escape binding and observing whether handleKeyEvent
    // suppresses it. In a typing context, the manager intercepts
    // non-Escape keys and returns true without invoking the handler.
    const probeIsTyping = (mgr: InputContextManager): boolean => {
      const handler = vi.fn();
      registerTestBinding(mgr, InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = mgr.handleKeyEvent(event, 'down');
      // In typing context: handled === true and handler NOT called.
      // In non-typing context: handler called.
      return handled && handler.mock.calls.length === 0;
    };

    const cases: Array<{ name: string; build: () => HTMLElement; typing: boolean }> = [
      {
        name: 'plain text input',
        build: () => {
          const el = document.createElement('input');
          el.type = 'text';
          return el;
        },
        typing: true,
      },
      {
        name: 'textarea',
        build: () => document.createElement('textarea'),
        typing: true,
      },
      {
        name: 'contenteditable=true div',
        build: () => {
          const el = document.createElement('div');
          el.setAttribute('contenteditable', 'true');
          el.tabIndex = 0;
          return el;
        },
        typing: true,
      },
      {
        name: 'button',
        build: () => document.createElement('button'),
        typing: false,
      },
      {
        name: 'plain div',
        build: () => {
          const el = document.createElement('div');
          el.tabIndex = 0;
          return el;
        },
        typing: false,
      },
      {
        name: 'input type=range',
        build: () => {
          const el = document.createElement('input');
          el.type = 'range';
          return el;
        },
        typing: false,
      },
      {
        name: 'input type=checkbox',
        build: () => {
          const el = document.createElement('input');
          el.type = 'checkbox';
          return el;
        },
        typing: false,
      },
    ];

    for (const c of cases) {
      it(`agrees with focus-utils for ${c.name}`, async () => {
        // Import focus-utils canonical helper synchronously via dynamic
        // import to avoid a hoisting hazard with the mocked top-level.
        const { isTypingInInput } = await import('../../../../utils/dom/focus');

        const el = c.build();
        document.body.appendChild(el);
        el.focus();

        // Sanity: jsdom focus might not always set activeElement on
        // every node. If activeElement is body for a non-typing case
        // it's still a valid agreement (body is not typing).
        const helperVerdict = isTypingInInput(document.activeElement);
        expect(helperVerdict).toBe(c.typing);

        const managerVerdict = probeIsTyping(new InputContextManager());
        expect(managerVerdict).toBe(c.typing);

        expect(managerVerdict).toBe(helperVerdict);
      });
    }
  });

  // ─────────────────────────────────────────────────────────────────────
  // MED-3 regression: Escape-from-typing must route both `handler` and
  // `keyupHandler`. Previously, in a typing context, an Escape keyup
  // would fall through to `return false` instead of looking up
  // additional contexts for a keyupHandler match.
  // ─────────────────────────────────────────────────────────────────────
  describe('MED-3: dispatchEscapeFromTypingContext routes keyupHandler', () => {
    afterEach(() => {
      document.body.innerHTML = '';
    });

    const enterTypingContext = (): void => {
      const input = document.createElement('input');
      input.type = 'text';
      document.body.appendChild(input);
      input.focus();
    };

    it('invokes Escape keyupHandler from a typing context', () => {
      enterTypingContext();
      const keydownHandler = vi.fn();
      const keyupHandler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: keydownHandler,
        keyupHandler,
      });

      const event = new KeyboardEvent('keyup', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'up');

      expect(handled).toBe(true);
      expect(keyupHandler).toHaveBeenCalledTimes(1);
      expect(keydownHandler).not.toHaveBeenCalled();
    });

    it('still invokes Escape main handler on keydown from typing context', () => {
      enterTypingContext();
      const keydownHandler = vi.fn();

      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: keydownHandler,
      });

      const event = new KeyboardEvent('keydown', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(keydownHandler).toHaveBeenCalledTimes(1);
    });

    it('skips bindings without a keyupHandler and finds a match in another context', () => {
      enterTypingContext();
      const navKeydownOnly = vi.fn();
      const uiKeyupHandler = vi.fn();

      // NAVIGATION binding has only a keydown handler — should be
      // skipped on keyup dispatch.
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'Escape',
        handler: navKeydownOnly,
      });

      // UI_INTERACTION binding has a keyupHandler — should fire.
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'Escape',
        handler: vi.fn(),
        keyupHandler: uiKeyupHandler,
      });

      const event = new KeyboardEvent('keyup', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'up');

      expect(handled).toBe(true);
      expect(uiKeyupHandler).toHaveBeenCalledTimes(1);
      expect(navKeydownOnly).not.toHaveBeenCalled();
    });
  });

  describe('dispatchEscapeFromTypingContext fallback [input.md G19]', () => {
    // input.md G19[P5]: when called from a typing context with NO matching
    // Escape binding in any context, the dispatch must return false (no
    // handler ran, swallowing skipped). Prior tests always registered at
    // least one Escape binding — the all-misses path was uncovered.
    it('[G19] Escape from TYPING context with NO Escape binding registered: handleKeyEvent returns false', () => {
      manager.setContext(InputContext.TYPING);
      // No bindings at all — TYPING allowedKeys is [] so nothing fires
      // from current context; dispatchEscapeFromTypingContext walks the
      // priority-ordered list and finds nothing.
      const event = new KeyboardEvent('keydown', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'down');
      expect(handled).toBe(false);
    });

    it('[G19] Escape from TYPING with bindings on OTHER keys: returns false (no Escape match)', () => {
      manager.setContext(InputContext.TYPING);
      const handler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'h', // not Escape
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'down');
      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });

    it('[G19] Escape keyup with a binding that has handler but NO keyupHandler: continues searching, returns false', () => {
      // Pins the inner `continue` at context-manager.ts L560: a binding
      // without a keyupHandler does NOT consume the keyup; the search
      // continues to lower-priority contexts.
      manager.setContext(InputContext.TYPING);
      const downOnly = vi.fn();
      registerTestBinding(manager, InputContext.UI_INTERACTION, {
        key: 'Escape',
        handler: downOnly, // keydown handler only — no keyupHandler
      });
      const event = new KeyboardEvent('keyup', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'up');
      expect(handled).toBe(false);
      expect(downOnly).not.toHaveBeenCalled();
    });
  });

  describe('NAVIGATION context allows modified bindings [input.md G20]', () => {
    it('dispatches a modified NAVIGATION binding when the bare key is blocked', () => {
      const navigationHandler = vi.fn();
      const flyHandler = vi.fn();
      registerTestBinding(manager, InputContext.NAVIGATION, {
        key: 'ArrowUp',
        modifiers: { shift: true },
        handler: navigationHandler,
      });
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'ArrowUp',
        modifiers: { shift: true },
        handler: flyHandler,
      });

      const event = new KeyboardEvent('keydown', { key: 'ArrowUp', shiftKey: true });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(navigationHandler).toHaveBeenCalledWith(event);
      expect(flyHandler).not.toHaveBeenCalled();
    });

    it('does not route a bare fly key from navigation into fly controls', () => {
      const flyHandler = vi.fn();
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'ArrowUp',
        handler: flyHandler,
      });

      const event = new KeyboardEvent('keydown', { key: 'ArrowUp' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(false);
      expect(flyHandler).not.toHaveBeenCalled();
    });

    it('does not route UI interaction into fly controls', () => {
      const flyHandler = vi.fn();
      registerTestBinding(manager, InputContext.FLY_CONTROLS, {
        key: 'ArrowUp',
        modifiers: { shift: true },
        handler: flyHandler,
      });
      manager.setContext(InputContext.UI_INTERACTION);

      const event = new KeyboardEvent('keydown', { key: 'ArrowUp', shiftKey: true });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(false);
      expect(flyHandler).not.toHaveBeenCalled();
    });
  });
});
