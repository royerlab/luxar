/**
 * Unit tests for InputContextManager
 *
 * Tests the input context management system that prevents
 * keyboard conflicts between different UI modes.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InputContextManager, InputContext } from '../input/input-context-manager';

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
      const event = new KeyboardEvent('keydown', { key: 'a' });
      // If disabled, would return false immediately
      manager.setEnabled(false);
      expect(manager.handleKeyEvent(event, 'down')).toBe(false);

      manager.setEnabled(true);
      // Now it processes the event (returns false because no binding, but processes)
      expect(manager.handleKeyEvent(event, 'down')).toBe(false);
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
      manager.pushContext(InputContext.FLY_CONTROLS);
      const debugInfo1 = manager.getDebugInfo();
      const stackLength1 = debugInfo1.contextStack.length;

      manager.pushContext(InputContext.FLY_CONTROLS); // Same context
      const debugInfo2 = manager.getDebugInfo();

      expect(debugInfo2.contextStack.length).toBe(stackLength1);
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
    it('should block keys not in allowed list for restrictive contexts', () => {
      const handler = vi.fn();

      // Register in NAVIGATION context
      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler,
      });

      // Switch to FLY_CONTROLS which only allows WASD and arrow keys
      manager.setContext(InputContext.FLY_CONTROLS);

      const event = new KeyboardEvent('keydown', { key: 'h' });
      const handled = manager.handleKeyEvent(event, 'down');

      // 'h' is not in allowed keys for FLY_CONTROLS, so it's not handled
      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
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
});
