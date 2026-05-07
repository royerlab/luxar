/**
 * Unit tests for Debug Console
 * Tests critical fixes: XSS vulnerability, memory leaks from resize/drag
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DebugConsole } from '../../../ui/panels/debug-console';

// Mock console interceptor
vi.mock('../../../utils/console-interceptor', () => ({
  consoleInterceptor: {
    addListener: vi.fn(),
    removeListener: vi.fn(),
    getBufferedMessages: vi.fn(() => []),
    getStats: vi.fn(() => ({
      total: 0,
      types: { log: 0, warn: 0, error: 0, info: 0, debug: 0 },
    })),
    clearBuffer: vi.fn(),
  },
}));

// Mock logger
vi.mock('../../../utils/log', () => ({
  log: {
    custom: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    data: vi.fn(),
  },
  Modules: {
    DEBUG_CONSOLE: 'DEBUG_CONSOLE',
  },
  LogEmoji: {
    CONSOLE: '🖥️',
  },
}));

beforeEach(() => {
  document.body.innerHTML = '';
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('DebugConsole - Critical Fixes', () => {
  describe('Memory Leak Prevention - Resize Handlers', () => {
    it('registers a non-empty EventGroup on construction and clears it on dispose', () => {
      const debugConsole = new DebugConsole();
      const events = (debugConsole as unknown as { events: { size: number } }).events;
      // makeDraggable + makeResizable + the toolbar buttons attach a
      // double-digit number of listeners; we don't enumerate exactly to
      // avoid over-coupling, just verify the group has tracked listeners.
      expect(events.size).toBeGreaterThan(0);

      debugConsole.dispose();
      expect(events.size).toBe(0);
    });

    it('clean up all global drag + resize listeners on dispose', () => {
      const debugConsole = new DebugConsole();

      // Verify dispose removes listeners
      const removeEventSpy = vi.spyOn(document, 'removeEventListener');

      debugConsole.dispose();

      // Should remove 4 listeners: 2 for drag (mousemove, mouseup) + 2 for resize (mousemove, mouseup)
      const mousemoveCalls = removeEventSpy.mock.calls.filter((call) => call[0] === 'mousemove');
      const mouseupCalls = removeEventSpy.mock.calls.filter((call) => call[0] === 'mouseup');

      expect(mousemoveCalls.length).toBe(2); // drag + resize
      expect(mouseupCalls.length).toBe(2); // drag + resize

      removeEventSpy.mockRestore();
    });
  });

  describe('XSS Vulnerability Fix', () => {
    it('should safely render console messages without innerHTML', () => {
      const debugConsole = new DebugConsole();
      debugConsole.show();

      // Access private method via any cast for testing
      const console_any = debugConsole as any;

      // Test with malicious string
      const maliciousArg = '<script>alert("XSS")</script>';
      const element = console_any.formatArgAsDOMElement(maliciousArg);

      // Should be rendered as text, not executed
      expect(element.textContent).toContain('<script>');
      expect(element.innerHTML).not.toContain('<script>alert');

      // Verify it's a text node, not executable HTML
      expect(element.querySelector('script')).toBeNull();

      debugConsole.dispose();
    });

    it('should handle objects without innerHTML injection', () => {
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as any;

      const maliciousObject = {
        toString: () => '<img src=x onerror=alert(1)>',
      };

      const element = console_any.formatArgAsDOMElement(maliciousObject);

      // Should JSON.stringify the object, not execute toString as HTML
      expect(element.textContent).toBeTruthy();
      expect(element.querySelector('img')).toBeNull();

      debugConsole.dispose();
    });

    it('should use textContent for all user-controlled data', () => {
      const debugConsole = new DebugConsole();
      debugConsole.show();

      const contentArea = document.querySelector('.luxar-debug-console__content');
      expect(contentArea).toBeTruthy();

      // Verify no innerHTML is set on message rendering
      const console_any = debugConsole as any;
      const message = {
        type: 'log' as const,
        timestamp: new Date(),
        args: ['<b>Bold text</b>', '<script>evil()</script>'],
        formatted: 'test',
      };

      console_any.renderMessage(message);

      // The malicious content should be escaped as text
      const messageElements = document.querySelectorAll('.luxar-console-message');
      expect(messageElements.length).toBe(1);

      const messageText = messageElements[0].textContent || '';
      expect(messageText).toContain('<b>Bold text</b>'); // Should see raw HTML
      expect(messageText).toContain('<script>evil()</script>');

      // Should NOT have actual script or bold elements
      expect(document.querySelector('.luxar-console-message script')).toBeNull();
      expect(document.querySelector('.luxar-console-message b')).toBeNull();

      debugConsole.dispose();
    });
  });

  describe('Dispose Cleanup', () => {
    it('should fully dispose debug console without leaks', () => {
      const debugConsole = new DebugConsole();

      const panel = document.querySelector('.luxar-debug-console');
      expect(panel).toBeTruthy();

      debugConsole.dispose();

      // Panel should be removed.
      expect(document.querySelector('.luxar-debug-console')).toBeNull();
    });
  });
});
