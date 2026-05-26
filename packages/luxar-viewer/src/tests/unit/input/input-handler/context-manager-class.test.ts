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

    it('should be enabled by default', () => {
      // input.md W3 fix: strengthen the assertion — pin the default enabled
      // state by observing that a registered handler fires when the manager
      // is freshly constructed (without an explicit setEnabled call). A
      // mutation that flipped the default to `false` would otherwise be
      // hidden behind two "returns false" outcomes that look identical.
      //
      // Use 'h' — 'a' is in flyModeKeys (blocked in NAVIGATION).
      const handler = vi.fn();
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: 'h' });
      // Default state: the manager dispatches the binding.
      expect(manager.handleKeyEvent(event, 'down')).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);

      // After explicit disable: dispatch is suppressed.
      manager.setEnabled(false);
      const event2 = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event2, 'down')).toBe(false);
      expect(handler).toHaveBeenCalledTimes(1); // unchanged

      // Re-enable: dispatch resumes.
      manager.setEnabled(true);
      const event3 = new KeyboardEvent('keydown', { key: 'h' });
      expect(manager.handleKeyEvent(event3, 'down')).toBe(true);
      expect(handler).toHaveBeenCalledTimes(2);
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

  describe('key binding registration', () => {
    it('should register key bindings', () => {
      const handler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler,
        preventDefault: true,
        description: 'Show help',
      });

      const debugInfo = manager.getDebugInfo();
      const navBindings = debugInfo.registeredBindings.get(InputContext.NAVIGATION);
      expect(navBindings).toContain('h');
    });

    it('should register bindings with modifiers', () => {
      const handler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'l',
        modifiers: { ctrl: true },
        handler,
        preventDefault: true,
      });

      const debugInfo = manager.getDebugInfo();
      const navBindings = debugInfo.registeredBindings.get(InputContext.NAVIGATION);
      expect(navBindings).toContain('ctrl+l');
    });

    it('[input.md C5] registered binding actually fires when matching event is dispatched (end-to-end)', () => {
      // input.md C5[P2]: prior tests only assert via getDebugInfo() that a
      // bindingKey appears in the registered list — a single-channel
      // probe. They don't end-to-end the lookup: a regression that wrote
      // bindings into a different Map but kept the debugInfo accessor
      // honest would survive. Pin the lookup-plus-dispatch contract.
      const handler = vi.fn();
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler,
        preventDefault: true,
      });
      // Use 'h' (not in flyModeKeys → not in NAVIGATION blockedKeys).
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
      manager.registerBinding(InputContext.NAVIGATION, {
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

    it('should warn about conflicts', () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler: handler1,
      });

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler: handler2,
      });

      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Key binding conflict'));

      warnSpy.mockRestore();
    });

    it('should unregister bindings', () => {
      const handler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
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
    it('should handle registered key events', () => {
      const handler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.FLY_CONTROLS, {
        key: 'w',
        handler,
      });

      // 'w' is in allowed keys for FLY_CONTROLS
      const event = new KeyboardEvent('keydown', { key: 'w' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });

    it('should respect blocked keys', () => {
      // NAVIGATION context blocks WASD keys
      manager.setContext(InputContext.NAVIGATION);

      const handler = vi.fn();
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'w',
        handler,
      });

      const event = new KeyboardEvent('keydown', { key: 'w' });
      const handled = manager.handleKeyEvent(event, 'down');

      // Should be blocked
      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('context priority and passthrough', () => {
    it('should pass through keys not in allowed list when passthrough enabled', () => {
      const handler = vi.fn();

      // Register in NAVIGATION context
      manager.registerBinding(InputContext.NAVIGATION, {
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

      // Register a handler in a lower priority context
      manager.registerBinding(InputContext.DIMENSION_NAV, {
        key: '[',
        handler,
      });

      // DIMENSION_NAV allows '[' and has passthrough
      manager.setContext(InputContext.DIMENSION_NAV);

      const event = new KeyboardEvent('keydown', { key: '[' });
      const handled = manager.handleKeyEvent(event, 'down');

      // Should handle the key since it's registered and allowed
      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalled();
    });
  });

  describe('debug information', () => {
    it('should provide debug info', () => {
      const handler1 = vi.fn();
      const handler2 = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler: handler1,
      });

      manager.registerBinding(InputContext.FLY_CONTROLS, {
        key: 'w',
        handler: handler2,
      });

      manager.pushContext(InputContext.FLY_CONTROLS);

      const debugInfo = manager.getDebugInfo();

      expect(debugInfo.currentContext).toBe(InputContext.FLY_CONTROLS);
      expect(debugInfo.contextStack).toContain(InputContext.NAVIGATION);
      expect(debugInfo.registeredBindings.get(InputContext.NAVIGATION)).toContain('h');
      expect(debugInfo.registeredBindings.get(InputContext.FLY_CONTROLS)).toContain('w');
    });
  });

  describe('cleanup', () => {
    it('should clear context bindings', () => {
      const handler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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
      localMgr.registerBinding(InputContext.NAVIGATION, {
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

  // input.md G6 fix: the existing "should allow Escape in typing
  // context" test (lines 238-245) reflects the OLD behavior where
  // Escape from a typing context was passed-through unhandled. The
  // current code routes Escape through dispatchEscapeFromTypingContext,
  // which fires the first matching Escape binding across every
  // context (including the current one). This `it.skip` documents the
  // post-refactor contract; un-skip when the OOS production bug is
  // fixed.
  describe('Escape in typing context (post-refactor contract)', () => {
    it.skip('dispatches Escape through dispatchEscapeFromTypingContext to the navigation binding', () => {
      const escapeHandler = vi.fn();
      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler: navHandler,
      });

      // Stand in FLY_CONTROLS (priority 1, passthrough). 'h' is not
      // in flyModeKeys/arrows allowedKeys so it falls through.
      manager.setContext(InputContext.FLY_CONTROLS);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      expect(handled).toBe(true);
      expect(navHandler).toHaveBeenCalledWith(event);
      expect(dimHandler).not.toHaveBeenCalled();
    });

    it('does NOT cascade when the current context disables passthrough', () => {
      const navHandler = vi.fn();
      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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
      mgr.registerBinding(InputContext.NAVIGATION, {
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
        const { isTypingInInput } = await import(
          '../../../../input/input-handler/commands/focus-utils'
        );

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

      manager.registerBinding(InputContext.NAVIGATION, {
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

      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'Escape',
        handler: navKeydownOnly,
      });

      // UI_INTERACTION binding has a keyupHandler — should fire.
      manager.registerBinding(InputContext.UI_INTERACTION, {
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
      manager.registerBinding(InputContext.NAVIGATION, {
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
      manager.registerBinding(InputContext.UI_INTERACTION, {
        key: 'Escape',
        handler: downOnly, // keydown handler only — no keyupHandler
      });
      const event = new KeyboardEvent('keyup', { key: 'Escape' });
      const handled = manager.handleKeyEvent(event, 'up');
      expect(handled).toBe(false);
      expect(downOnly).not.toHaveBeenCalled();
    });
  });

  describe('NAVIGATION context allows Shift [input.md G20]', () => {
    // input.md G20[P5]: flyModeKeysWithoutShift filters Shift OUT of
    // NAVIGATION's blockedKeys. No test directly asserts that Shift is
    // allowed in NAVIGATION. A regression that re-added Shift to the
    // blocklist would survive (Shift+wheel uses WindowEventHandler, not
    // this manager).
    it('[G20] Shift is NOT in NAVIGATION blockedKeys (probed via private field cast)', () => {
      // The binding-system uses modifier-prefixed bindingKeys ("Shift+Shift"
      // for a Shift event with shiftKey=true), which makes a direct
      // dispatch test brittle. Instead, pin the contract by reading the
      // private contextConfigs map directly via cast. A regression that
      // re-added 'Shift' to NAVIGATION's blockedKeys list would surface.
      const configs = (manager as unknown as {
        contextConfigs: Map<InputContext, { blockedKeys?: string[] }>;
      }).contextConfigs;
      const navConfig = configs.get(InputContext.NAVIGATION);
      const blocked = navConfig?.blockedKeys ?? [];
      expect(blocked).not.toContain('Shift');
    });

    it('[G20] WASD keys ARE blocked in NAVIGATION (sanity that the filter is correct)', () => {
      // Symmetric to G20: pin that the OTHER fly-mode keys (WASD) ARE
      // blocked in NAVIGATION. This guards the test from a mutation that
      // empties the blockedKeys array entirely (which would let G20 pass
      // for the wrong reason).
      const handler = vi.fn();
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'w',
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: 'w' });
      const handled = manager.handleKeyEvent(event, 'down');
      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('DIMENSION_NAV context [input.md G21]', () => {
    // input.md G21[P5]: DIMENSION_NAV has allowedKeys = config.input.keyboard.dimensionKeys.
    // No test set the context to DIMENSION_NAV and exercised a key from
    // dimensionKeys to confirm the allowlist works.
    it('[G21] DIMENSION_NAV context: a registered binding on a dimensionKey fires', () => {
      manager.setContext(InputContext.DIMENSION_NAV);
      const handler = vi.fn();
      // '1' is in dimensionKeys per config.
      manager.registerBinding(InputContext.DIMENSION_NAV, {
        key: '1',
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: '1' });
      const handled = manager.handleKeyEvent(event, 'down');
      expect(handled).toBe(true);
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('[G21] DIMENSION_NAV context: a NON-dimensionKey ("z") is NOT allowed even if registered', () => {
      // 'z' is not in dimensionKeys → allowlist rejects it.
      manager.setContext(InputContext.DIMENSION_NAV);
      const handler = vi.fn();
      manager.registerBinding(InputContext.DIMENSION_NAV, {
        key: 'z',
        handler,
      });
      const event = new KeyboardEvent('keydown', { key: 'z' });
      manager.handleKeyEvent(event, 'down');
      // The binding registers but the allowlist gates dispatch; current
      // context's allowedKeys filters BEFORE the binding lookup.
      // (If 'z' has a binding elsewhere with passthrough, it could fire
      // from a lower context — but no such binding here, so we expect 0.)
      expect(handler).not.toHaveBeenCalled();
    });
  });
});
