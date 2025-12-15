/**
 * Tests for keyupHandler functionality in InputContextManager
 *
 * The keyupHandler feature allows bindings to have separate handlers
 * for keydown and keyup events. This is critical for:
 * - Shift key (disable zoom on down, enable on up)
 * - Fly controls (start movement on down, stop on up)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { InputContextManager, InputContext } from '../../../input/input-context-manager';

describe('InputContextManager - keyupHandler Feature', () => {
  let manager: InputContextManager;

  beforeEach(() => {
    manager = new InputContextManager();
  });

  describe('keyupHandler execution', () => {
    it('should call main handler on keydown event', () => {
      const keydownHandler = vi.fn();
      const keyupHandler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'Shift',
        handler: keydownHandler,
        keyupHandler,
      });

      const event = new KeyboardEvent('keydown', { key: 'Shift' });
      manager.handleKeyEvent(event, 'down');

      expect(keydownHandler).toHaveBeenCalledTimes(1);
      expect(keydownHandler).toHaveBeenCalledWith(event);
      expect(keyupHandler).not.toHaveBeenCalled();
    });

    it('should call keyupHandler on keyup event when provided', () => {
      const keydownHandler = vi.fn();
      const keyupHandler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'Shift',
        handler: keydownHandler,
        keyupHandler,
      });

      const event = new KeyboardEvent('keyup', { key: 'Shift' });
      manager.handleKeyEvent(event, 'up');

      expect(keyupHandler).toHaveBeenCalledTimes(1);
      expect(keyupHandler).toHaveBeenCalledWith(event);
      expect(keydownHandler).not.toHaveBeenCalled();
    });

    it('should NOT call handler on keyup if keyupHandler not provided', () => {
      const handler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'h',
        handler,
        // No keyupHandler
      });

      const keyupEvent = new KeyboardEvent('keyup', { key: 'h' });
      const handled = manager.handleKeyEvent(keyupEvent, 'up');

      // Should NOT handle keyup without keyupHandler
      expect(handled).toBe(false);
      expect(handler).not.toHaveBeenCalled();
    });
  });

  describe('Shift key behavior via bindings', () => {
    it('should handle Shift keydown and keyup with separate handlers', () => {
      const keydownHandler = vi.fn();
      const keyupHandler = vi.fn();

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'Shift',
        handler: keydownHandler,
        keyupHandler,
      });

      // Press Shift - should call keydown handler
      const keydownEvent = new KeyboardEvent('keydown', { key: 'Shift' });
      const downHandled = manager.handleKeyEvent(keydownEvent, 'down');

      expect(downHandled).toBe(true);
      expect(keydownHandler).toHaveBeenCalledTimes(1);
      expect(keyupHandler).not.toHaveBeenCalled();

      // Release Shift - should call keyup handler
      const keyupEvent = new KeyboardEvent('keyup', { key: 'Shift' });
      const upHandled = manager.handleKeyEvent(keyupEvent, 'up');

      expect(upHandled).toBe(true);
      expect(keyupHandler).toHaveBeenCalledTimes(1);
      expect(keydownHandler).toHaveBeenCalledTimes(1); // Still only once from before
    });
  });

  describe('Fly controls keyup behavior', () => {
    it('should call separate keyup handler for fly control keys', () => {
      const keydownHandler = vi.fn();
      const keyupHandler = vi.fn();

      manager.setContext(InputContext.FLY_CONTROLS);
      manager.registerBinding(InputContext.FLY_CONTROLS, {
        key: 'w',
        handler: keydownHandler,
        keyupHandler,
      });

      // Press W - starts movement
      const keydownEvent = new KeyboardEvent('keydown', { key: 'w' });
      manager.handleKeyEvent(keydownEvent, 'down');
      expect(keydownHandler).toHaveBeenCalledTimes(1);
      expect(keyupHandler).not.toHaveBeenCalled();

      // Release W - stops movement
      const keyupEvent = new KeyboardEvent('keyup', { key: 'w' });
      manager.handleKeyEvent(keyupEvent, 'up');
      expect(keyupHandler).toHaveBeenCalledTimes(1);
      expect(keydownHandler).toHaveBeenCalledTimes(1); // Still only called once
    });

    it('should handle Shift+W with separate up/down handlers', () => {
      const keydownHandler = vi.fn();
      const keyupHandler = vi.fn();

      manager.setContext(InputContext.FLY_CONTROLS);
      manager.registerBinding(InputContext.FLY_CONTROLS, {
        key: 'w',
        modifiers: { shift: true },
        handler: keydownHandler,
        keyupHandler,
      });

      // Press Shift+W
      const keydownEvent = new KeyboardEvent('keydown', { key: 'w', shiftKey: true });
      manager.handleKeyEvent(keydownEvent, 'down');
      expect(keydownHandler).toHaveBeenCalledTimes(1);

      // Release W (Shift still held)
      const keyupEvent = new KeyboardEvent('keyup', { key: 'w', shiftKey: true });
      manager.handleKeyEvent(keyupEvent, 'up');
      expect(keyupHandler).toHaveBeenCalledTimes(1);
    });
  });

  describe('Toggle actions should NOT double-trigger', () => {
    it('should only call handler on keydown for toggle actions without keyupHandler', () => {
      let cinematicMode = false;
      const toggleHandler = vi.fn(() => {
        cinematicMode = !cinematicMode;
      });

      manager.registerBinding(InputContext.NAVIGATION, {
        key: 'c',
        handler: toggleHandler,
        // No keyupHandler - toggle actions don't need it
      });

      // Press C - toggle ON
      const keydownEvent = new KeyboardEvent('keydown', { key: 'c' });
      manager.handleKeyEvent(keydownEvent, 'down');
      expect(cinematicMode).toBe(true);
      expect(toggleHandler).toHaveBeenCalledTimes(1);

      // Release C - should NOT call handler (no keyupHandler)
      const keyupEvent = new KeyboardEvent('keyup', { key: 'c' });
      const handled = manager.handleKeyEvent(keyupEvent, 'up');

      // Keyup should NOT be handled (returns false) - CORRECT behavior
      expect(handled).toBe(false);
      expect(toggleHandler).toHaveBeenCalledTimes(1); // Only called once
      expect(cinematicMode).toBe(true); // Stays ON (doesn't toggle back)
    });
  });
});
