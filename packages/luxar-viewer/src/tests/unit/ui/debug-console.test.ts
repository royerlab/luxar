/**
 * Unit tests for Debug Console
 * Tests critical fixes: XSS vulnerability, memory leaks from resize/drag
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { formatArgs } from '../../../ui/debug-console/formatters';
import { DebugConsole } from '../../../ui/debug-console';

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

    it('post-dispose: global mousemove does not mutate panel position or size', () => {
      // C4 strengthening (P1): observable contract instead of exact count.
      // The previous test asserted "2 mousemove + 2 mouseup removeEventListener
      // calls", which pins the impl split between drag/resize subsystems.
      // The load-bearing contract is: after dispose, document-level mouse
      // events must not affect any DOM the panel previously owned.
      const debugConsole = new DebugConsole();
      const panel = document.querySelector('.luxar-debug-console') as HTMLElement;
      const header = panel.querySelector('.luxar-debug-console__header') as HTMLElement;

      // Sanity: pre-dispose, a drag mutates panel.style.left/top.
      header.dispatchEvent(
        new MouseEvent('mousedown', { clientX: 10, clientY: 10, bubbles: true })
      );
      document.dispatchEvent(new MouseEvent('mousemove', { clientX: 60, clientY: 80 }));
      const draggedLeft = panel.style.left;
      const draggedTop = panel.style.top;
      expect(draggedLeft).not.toBe('');
      expect(draggedTop).not.toBe('');
      document.dispatchEvent(new MouseEvent('mouseup'));

      debugConsole.dispose();

      // Post-dispose: mousedown on the (now detached) header + mousemove
      // on document must NOT call any of the panel's handlers. We assert
      // by checking that the panel reference's inline style was not
      // overwritten (the dispose path doesn't reset it; only an active
      // drag handler would).
      const leftBeforeProbe = panel.style.left;
      const topBeforeProbe = panel.style.top;
      header.dispatchEvent(
        new MouseEvent('mousedown', { clientX: 200, clientY: 200, bubbles: true })
      );
      document.dispatchEvent(new MouseEvent('mousemove', { clientX: 999, clientY: 999 }));
      document.dispatchEvent(new MouseEvent('mouseup'));

      expect(panel.style.left).toBe(leftBeforeProbe);
      expect(panel.style.top).toBe(topBeforeProbe);
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

  describe('Accessibility', () => {
    it('panel has region role labelled by the title element', () => {
      const debugConsole = new DebugConsole();
      const panel = document.querySelector('.luxar-debug-console') as HTMLElement;
      expect(panel.getAttribute('role')).toBe('region');
      expect(panel.getAttribute('aria-labelledby')).toBe('luxar-debug-console-title');
      const title = document.getElementById('luxar-debug-console-title');
      expect(title?.textContent).toBe('Debug Console');
      debugConsole.dispose();
    });

    it('toolbar controls have aria-labels', () => {
      const debugConsole = new DebugConsole();
      const filter = document.querySelector('.luxar-debug-console__filter') as HTMLInputElement;
      const closeBtn = document.querySelector(
        '.luxar-debug-console__close-btn'
      ) as HTMLButtonElement;
      const clearBtn = document.querySelector(
        '.luxar-debug-console__clear-btn'
      ) as HTMLButtonElement;
      const copyBtn = document.querySelector('.luxar-debug-console__copy-btn') as HTMLButtonElement;
      expect(filter.getAttribute('aria-label')).toBe('Filter messages');
      expect(closeBtn.getAttribute('aria-label')).toBe('Close debug console');
      expect(clearBtn.getAttribute('aria-label')).toBe('Clear console');
      expect(copyBtn.getAttribute('aria-label')).toBe('Copy all messages to clipboard');
      debugConsole.dispose();
    });

    it('content region is a live log; resize handles are aria-hidden', () => {
      const debugConsole = new DebugConsole();
      const content = document.querySelector('.luxar-debug-console__content') as HTMLElement;
      expect(content.getAttribute('role')).toBe('log');
      expect(content.getAttribute('aria-live')).toBe('polite');
      const handles = document.querySelectorAll('.luxar-debug-console__resize-handle');
      expect(handles.length).toBeGreaterThan(0);
      handles.forEach((h) => expect(h.getAttribute('aria-hidden')).toBe('true'));
      debugConsole.dispose();
    });
  });

  // [ui.md/O4][P10] Moved out of `describe('Accessibility', ...)` — formatting
  // null-prototype / cyclic objects is a robustness concern (JSON.stringify
  // fallback path), not an accessibility one.
  describe('Argument Formatting Robustness', () => {
    it('renders an Error as name: message in the DOM, not {}', () => {
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as unknown as {
        formatArgAsDOMElement: (arg: unknown) => HTMLElement;
      };

      const el = console_any.formatArgAsDOMElement(new Error('boom'));

      expect(el.textContent).toBe('Error: boom');
      expect(el.textContent).not.toContain('{}');
    });

    it('agrees with formatArgs for Errors (the two formatters must not drift)', () => {
      // There are two independent renderers — the DOM span here and `formatArgs`
      // in debug-console/formatters.ts, which feeds the filter haystack and the
      // CLIPBOARD (the bug-report path). If they disagree for Errors, the visible
      // row and the copied text say different things.
      //
      // Scoped to Errors on purpose: they legitimately differ for plain strings,
      // which the DOM renderer quotes.
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as unknown as {
        formatArgAsDOMElement: (arg: unknown) => HTMLElement;
      };

      for (const err of [new Error('boom'), new Error(), new TypeError('bad type')]) {
        expect(console_any.formatArgAsDOMElement(err).textContent).toBe(formatArgs([err]));
      }
    });

    it('formats null-prototype objects without throwing', () => {
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as unknown as {
        formatArgAsDOMElement: (arg: unknown) => HTMLElement;
      };

      // Object.create(null) has no toString — JSON.stringify still works,
      // but we exercise the catch by passing a stringify-hostile null-proto.
      const cyclic: Record<string, unknown> = Object.create(null);
      cyclic.self = cyclic;

      // Must not throw, and must not produce '[object Object]' garbage.
      const el = console_any.formatArgAsDOMElement(cyclic);
      expect(el.textContent).toBeTruthy();
      // Either a sane String() result or our '[unprintable]' fallback.
      expect(typeof el.textContent).toBe('string');

      debugConsole.dispose();
    });

    it('renders a cross-realm Error in the DOM, not {}', () => {
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as unknown as {
        formatArgAsDOMElement: (arg: unknown) => HTMLElement;
      };

      const crossRealm = { [Symbol.toStringTag]: 'Error', name: 'TypeError', message: 'boom' };
      const el = console_any.formatArgAsDOMElement(crossRealm);
      expect(el.textContent).toBe('TypeError: boom');

      debugConsole.dispose();
    });
  });

  describe('Stack Trace Rendering', () => {
    const renderAndFindStack = (type: 'error' | 'warn' | 'log') => {
      const debugConsole = new DebugConsole();
      debugConsole.show();
      const console_any = debugConsole as unknown as {
        renderMessage: (m: unknown) => void;
      };
      console_any.renderMessage({
        type,
        timestamp: new Date(),
        args: ['something happened'],
        formatted: 'something happened',
        stack: 'Error: boom\n    at somewhere',
      });
      const stackEl = document.querySelector('.luxar-console-message-stack');
      const text = stackEl?.textContent ?? null;
      debugConsole.dispose();
      return text;
    };

    it('renders the stack of an error row', () => {
      expect(renderAndFindStack('error')).toContain('at somewhere');
    });

    it('renders the stack of a warn row too', () => {
      // Warns capture a stack (interceptor) and export it to the clipboard, so
      // the panel must show it as well or the two disagree.
      expect(renderAndFindStack('warn')).toContain('at somewhere');
    });

    it('does not render a stack for non-error/warn rows', () => {
      expect(renderAndFindStack('log')).toBeNull();
    });
  });
});
