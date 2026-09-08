/**
 * In-app debug console panel for viewing browser console output.
 *
 * Displays all console.log(), console.warn(), console.error() messages in a
 * draggable, resizable panel within the application. Uses the global console
 * interceptor to capture ALL messages from app startup, including those that
 * occurred before the panel was created.
 *
 * Key features:
 * - Message history from app startup (ring buffer)
 * - Syntax-highlighted output (objects, numbers, strings, etc.)
 * - Filtering by keyword
 * - Copy to clipboard
 * - Auto-scroll option
 * - Draggable and resizable
 * - Toggle with Ctrl+L keyboard shortcut
 *
 * Useful for debugging WebGL issues, understanding data loading, and
 * diagnosing problems without opening browser DevTools.
 *
 * @example
 * ```typescript
 * const debugConsole = new DebugConsole();
 *
 * // Toggle visibility with Ctrl+L
 * // Or programmatically:
 * debugConsole.show();  // Display with all buffered messages
 * debugConsole.hide();  // Hide panel
 * ```
 *
 * @module ui/debug-console
 */

import { consoleInterceptor, type BufferedMessage } from '../utils/console-interceptor';
import { config } from '../config';
import { getViewerContainer } from '../utils/viewer-container';
import { formatErrorForDisplay, isErrorLike } from '../utils/format-error';
import { log, Modules, LogEmoji } from '../utils/log';
import { EventGroup } from '../utils/cross-layer/event-group';
import {
  formatArgs as formatArgsImpl,
  formatConsoleTimestamp,
  formatFallbackValue,
  messageMatchesFilter,
} from './debug-console/formatters';

/**
 * A single captured console entry as rendered by the {@link DebugConsole}.
 */
export interface ConsoleMessage {
  /** Console severity/channel the message came in on. */
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  /** Wall-clock time the message was recorded. */
  timestamp: Date;
  /** Original console arguments, preserved for re-formatting. */
  args: unknown[];
  /** Flattened string form of `args`, used for filter/search matching. */
  formatted: string;
  /** Stack trace, when available — captured for errors and warnings. */
  stack?: string;
}

/**
 * Debug console UI component with console message capture and display.
 *
 * Integrates with the global consoleInterceptor to show all browser console
 * output in a styled panel. Messages are buffered globally, so opening the
 * console shows full history from app startup.
 */
export class DebugConsole {
  private panel: HTMLElement;
  private contentArea: HTMLElement;
  private isVisible = false;
  private autoScroll = true;
  private filter: string = '';
  private messageListenerCallback: ((message: BufferedMessage) => void) | null = null;

  /**
   * Cleanup group covering every DOM listener attached by this panel:
   * the toolbar buttons, drag/resize global listeners, and the
   * blur/visibilitychange watchers that stop in-flight drag/resize when
   * the window loses focus (otherwise the state stays "active"
   * indefinitely — if the user releases the mouse off-window, we never
   * see the mouseup and the next mousemove would resume dragging).
   */
  private events: EventGroup = new EventGroup();

  /**
   * Create and initialize debug console panel.
   *
   * Builds UI, registers with global console interceptor, and adds to DOM.
   * Starts hidden - use toggle() or show() to display.
   *
   * The console captures ALL messages logged since app startup, including
   * those before this constructor was called (via global interceptor).
   *
   * @example
   * ```typescript
   * const debugConsole = new DebugConsole();
   * // Panel created but hidden
   *
   * // Later, user presses Ctrl+L to show
   * debugConsole.toggle();
   * ```
   */
  constructor() {
    // Create UI
    this.panel = this.createPanel();
    this.contentArea = this.panel.querySelector('.luxar-debug-console__content') as HTMLElement;

    // Set up listener for new messages from the global interceptor
    this.messageListenerCallback = (message: BufferedMessage) => {
      if (this.isVisible) {
        this.renderBufferedMessage(message);
        this.updateStatus();
        if (this.autoScroll) {
          this.scrollToBottom();
        }
      }
    };

    // Register with the global interceptor
    consoleInterceptor.addListener(this.messageListenerCallback);

    // Add to DOM
    getViewerContainer().appendChild(this.panel);

    // Start hidden
    this.hide();

    // Log that debug console is ready
    log.custom(LogEmoji.CONSOLE, Modules.DEBUG_CONSOLE, 'Debug console ready (Ctrl+L to open)');
  }

  /**
   * Create the debug console panel UI
   */
  private createPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'luxar-debug-console luxar-glass-surface luxar-panel-pop';
    panel.setAttribute('role', 'region');
    panel.setAttribute('aria-labelledby', 'luxar-debug-console-title');
    // Note: Resize handles are now child elements (not pseudo-elements)
    // to allow ::before/::after for liquid glass effect
    panel.innerHTML = `
      <div class="luxar-glass-refraction" aria-hidden="true"></div>
      <div class="luxar-debug-console__resize-handle luxar-debug-console__resize-handle--top" aria-hidden="true"></div>
      <div class="luxar-debug-console__resize-handle luxar-debug-console__resize-handle--left" aria-hidden="true"></div>
      <div class="luxar-debug-console__resize-handle luxar-debug-console__resize-handle--corner" aria-hidden="true"></div>
      <div class="luxar-debug-console__scroll">
        <div class="luxar-debug-console__header luxar-panel-header">
          <div id="luxar-debug-console-title" class="luxar-debug-console__title">Debug Console</div>
          <div class="luxar-debug-console__controls">
            <input type="text" class="luxar-debug-console__filter" placeholder="Filter..." aria-label="Filter messages" />
            <button type="button" class="luxar-debug-console__clear-btn" title="Clear console" aria-label="Clear console">Clear</button>
            <button type="button" class="luxar-debug-console__copy-btn" title="Copy all to clipboard" aria-label="Copy all messages to clipboard">Copy</button>
            <label class="luxar-debug-console__autoscroll">
              <input type="checkbox" checked /> Auto-scroll
            </label>
            <button type="button" class="luxar-debug-console__close-btn luxar-panel-close" title="Close (Ctrl+L)" aria-label="Close debug console"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12"/><path d="M18 6L6 18"/></svg></button>
          </div>
        </div>
        <div class="luxar-debug-console__content" role="log" aria-live="polite" aria-atomic="false"></div>
        <div class="luxar-debug-console__status" aria-live="polite">
          <span class="luxar-debug-console__message-count">0 messages</span>
          <span class="luxar-debug-console__filter-status"></span>
        </div>
      </div>
    `;

    // Styles now in src/styles/components/debug-console.css

    // Setup event handlers
    this.setupEventHandlers(panel);

    return panel;
  }

  /**
   * Setup event handlers for the panel
   */
  private setupEventHandlers(panel: HTMLElement): void {
    const closeBtn = panel.querySelector<HTMLElement>('.luxar-debug-console__close-btn');
    if (closeBtn) this.events.on(closeBtn, 'click', () => this.hide());

    const clearBtn = panel.querySelector<HTMLElement>('.luxar-debug-console__clear-btn');
    if (clearBtn) this.events.on(clearBtn, 'click', () => this.clear());

    const copyBtn = panel.querySelector<HTMLElement>('.luxar-debug-console__copy-btn');
    if (copyBtn) this.events.on(copyBtn, 'click', () => this.copyToClipboard());

    const filterInput = panel.querySelector<HTMLInputElement>('.luxar-debug-console__filter');
    if (filterInput) {
      this.events.on(filterInput, 'input', (e) => {
        this.filter = (e.target as HTMLInputElement).value;
        this.applyFilter();
      });
    }

    const autoScrollCheckbox = panel.querySelector<HTMLInputElement>(
      '.luxar-debug-console__autoscroll input'
    );
    if (autoScrollCheckbox) {
      this.events.on(autoScrollCheckbox, 'change', (e) => {
        this.autoScroll = (e.target as HTMLInputElement).checked;
      });
    }

    // Make panel draggable
    this.makeDraggable(panel);

    // Make panel resizable
    this.makeResizable(panel);
  }

  /**
   * Make the panel draggable
   */
  private makeDraggable(panel: HTMLElement): void {
    const header = panel.querySelector('.luxar-debug-console__header') as HTMLElement;
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let initialX = 0;
    let initialY = 0;

    header.style.cursor = 'move';

    this.events.on(header, 'mousedown', (e) => {
      if (
        (e.target as HTMLElement).tagName === 'BUTTON' ||
        (e.target as HTMLElement).tagName === 'INPUT'
      )
        return;

      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = panel.getBoundingClientRect();
      initialX = rect.left;
      initialY = rect.top;

      e.preventDefault();
    });

    const dragMouseMove = (e: MouseEvent): void => {
      if (!isDragging) return;
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      panel.style.left = `${initialX + deltaX}px`;
      panel.style.top = `${initialY + deltaY}px`;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    };

    const dragMouseUp = (): void => {
      isDragging = false;
    };

    this.events.on(document, 'mousemove', dragMouseMove);
    this.events.on(document, 'mouseup', dragMouseUp);

    // Stop the in-flight drag if focus leaves the window — see comment on
    // `events` field for the reasoning.
    const stopOnBlur = (): void => dragMouseUp();
    this.events.on(window, 'blur', stopOnBlur);
    this.events.on(document, 'visibilitychange', stopOnBlur);
  }

  /**
   * Make the panel resizable
   */
  private makeResizable(panel: HTMLElement): void {
    let isResizing = false;
    let resizeDirection = '';
    let startX = 0;
    let startY = 0;
    let startWidth = 0;
    let startHeight = 0;

    this.events.on(panel, 'mousedown', (e) => {
      const rect = panel.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (y < config.ui.debugConsole.resize.borderWidth) {
        isResizing = true;
        resizeDirection = 'n';
        startY = e.clientY;
        startHeight = rect.height;
        e.preventDefault();
      } else if (x < config.ui.debugConsole.resize.borderWidth) {
        isResizing = true;
        resizeDirection = 'w';
        startX = e.clientX;
        startWidth = rect.width;
        e.preventDefault();
      }
    });

    const resizeMouseMove = (e: MouseEvent): void => {
      if (!isResizing) return;
      if (resizeDirection === 'n') {
        const deltaY = startY - e.clientY;
        const newHeight = Math.max(
          config.ui.debugConsole.panel.minHeight,
          Math.min(config.ui.debugConsole.panel.maxHeight, startHeight + deltaY)
        );
        panel.style.height = `${newHeight}px`;
      } else if (resizeDirection === 'w') {
        const deltaX = startX - e.clientX;
        const newWidth = Math.max(
          config.ui.debugConsole.panel.minWidth,
          Math.min(config.ui.debugConsole.panel.maxWidth, startWidth + deltaX)
        );
        panel.style.width = `${newWidth}px`;
      }
    };

    const resizeMouseUp = (): void => {
      isResizing = false;
      resizeDirection = '';
    };

    this.events.on(document, 'mousemove', resizeMouseMove);
    this.events.on(document, 'mouseup', resizeMouseUp);

    // makeDraggable's blur/visibilitychange listeners already invoke the
    // shared "drag is over" reset; resize state needs the same — wire a
    // dedicated stop-on-blur for the resize path.
    const stopResizeOnBlur = (): void => resizeMouseUp();
    this.events.on(window, 'blur', stopResizeOnBlur);
    this.events.on(document, 'visibilitychange', stopResizeOnBlur);
  }

  /**
   * Render a buffered message from the interceptor
   */
  private renderBufferedMessage(message: BufferedMessage): void {
    const consoleMessage: ConsoleMessage = {
      type: message.type,
      timestamp: message.timestamp,
      args: message.args,
      formatted: this.formatArgs(message.args),
      stack: message.stack,
    };
    this.renderMessage(consoleMessage);
  }

  /**
   * Format arguments for display
   */
  private formatArgs(args: unknown[]): string {
    return formatArgsImpl(args);
  }

  /**
   * Render a single message
   */
  private renderMessage(message: ConsoleMessage): void {
    const messageEl = document.createElement('div');
    messageEl.className = `luxar-console-message luxar-console-message-${message.type}`;

    const timestamp = formatConsoleTimestamp(message.timestamp);

    // Create timestamp element safely (no innerHTML)
    const timestampEl = document.createElement('span');
    timestampEl.className = 'luxar-console-message-timestamp';
    timestampEl.textContent = timestamp;
    messageEl.appendChild(timestampEl);

    // Format each argument with appropriate styling using DOM methods
    message.args.forEach((arg, index) => {
      if (index > 0) {
        messageEl.appendChild(document.createTextNode(' '));
      }
      const argEl = this.formatArgAsDOMElement(arg);
      messageEl.appendChild(argEl);
    });

    // Add stack trace if present. Warns capture a stack too (see the
    // interceptor's console.warn override), and the clipboard export renders it
    // regardless of type — so include warn here, not just error, or a captured
    // warn stack would leave via copy but never show in the panel.
    if (message.stack && (message.type === 'error' || message.type === 'warn')) {
      const stackEl = document.createElement('div');
      stackEl.className = 'luxar-console-message-stack';
      stackEl.textContent = message.stack;
      messageEl.appendChild(stackEl);
    }

    if (!messageMatchesFilter(message.formatted, this.filter)) {
      messageEl.style.display = 'none';
    }

    this.contentArea.appendChild(messageEl);
  }

  /**
   * Format an argument as a DOM element (safe, no XSS)
   */
  private formatArgAsDOMElement(arg: unknown): HTMLElement {
    const span = document.createElement('span');

    if (arg === undefined) {
      span.className = 'luxar-console-message-undefined';
      span.textContent = 'undefined';
    } else if (arg === null) {
      span.className = 'luxar-console-message-undefined';
      span.textContent = 'null';
    } else if (typeof arg === 'string') {
      span.className = 'luxar-console-message-string';
      span.textContent = `"${arg}"`;
    } else if (typeof arg === 'number') {
      span.className = 'luxar-console-message-number';
      span.textContent = String(arg);
    } else if (typeof arg === 'boolean') {
      span.className = 'luxar-console-message-boolean';
      span.textContent = String(arg);
    } else if (isErrorLike(arg)) {
      // Before the object branch: an Error's name/message/stack are
      // non-enumerable, so JSON.stringify would render it as `{}`. `isErrorLike`
      // also catches a cross-realm Error (which fails `instanceof Error`). Kept
      // in lockstep with `debug-console/formatters.ts::formatArgs` — the two must
      // agree for Errors or the visible row and the copied text disagree.
      span.className = 'luxar-console-message-object';
      span.textContent = formatErrorForDisplay(arg as Error);
    } else if (typeof arg === 'object') {
      span.className = 'luxar-console-message-object';
      try {
        const json = JSON.stringify(arg, null, 2);
        span.textContent = json;
      } catch {
        // The shared fallback preserves ordinary object stringification while
        // containing null-prototype objects and hostile toString methods.
        try {
          span.textContent = formatFallbackValue(arg);
        } catch {
          span.textContent = '[unprintable]';
        }
      }
    } else {
      span.textContent = formatFallbackValue(arg);
    }

    return span;
  }

  /**
   * Apply filter to existing messages
   */
  private applyFilter(): void {
    const messages = this.contentArea.querySelectorAll('.luxar-console-message');
    messages.forEach((el) => {
      const messageEl = el as HTMLElement;
      const text = messageEl.textContent || '';
      if (!messageMatchesFilter(text, this.filter)) {
        messageEl.style.display = 'none';
      } else {
        messageEl.style.display = '';
      }
    });
    this.updateStatus();
  }

  /**
   * Update status bar
   */
  private updateStatus(): void {
    const countEl = this.panel.querySelector('.luxar-debug-console__message-count');
    const filterEl = this.panel.querySelector('.luxar-debug-console__filter-status');

    if (countEl) {
      const visibleCount = this.contentArea.querySelectorAll(
        '.luxar-console-message:not([style*="display: none"])'
      ).length;
      const stats = consoleInterceptor.getStats();
      countEl.textContent = this.filter
        ? `${visibleCount} of ${stats.total} messages`
        : `${stats.total} messages`;
    }

    if (filterEl) {
      filterEl.textContent = this.filter ? `(filtered: "${this.filter}")` : '';
    }
  }

  /**
   * Scroll to bottom of content
   */
  private scrollToBottom(): void {
    this.contentArea.scrollTop = this.contentArea.scrollHeight;
  }

  /**
   * Clear all console messages from display and buffer.
   *
   * Clears both the UI and the global console interceptor buffer. This is
   * a destructive operation - messages cannot be recovered. The message
   * count resets to 0.
   *
   * @example
   * ```typescript
   * // User clicks "Clear" button
   * debugConsole.clear();
   * // All messages removed, fresh start
   * ```
   */
  clear(): void {
    // Clear the global buffer
    consoleInterceptor.clearBuffer();
    // Clear the UI
    this.contentArea.innerHTML = '';
    this.updateStatus();
    log.info(Modules.DEBUG_CONSOLE, 'Debug console cleared');
  }

  /**
   * Copy all messages to clipboard
   */
  private copyToClipboard(): void {
    const messages = consoleInterceptor.getBufferedMessages();
    const text = messages
      .map((m) => {
        const formatted = this.formatArgs(m.args);
        const line = `[${m.timestamp.toISOString()}] [${m.type.toUpperCase()}] ${formatted}`;
        // Include the stack. This is the bug-report path — someone pastes this
        // into an issue — and it was dropped entirely, so even a captured
        // `console.error` stack never left the app.
        return m.stack ? `${line}\n${m.stack}` : line;
      })
      .join('\n');

    navigator.clipboard
      .writeText(text)
      .then(() => {
        log.info(Modules.DEBUG_CONSOLE, 'Console output copied to clipboard');
      })
      .catch((err) => {
        log.error(Modules.DEBUG_CONSOLE, 'Failed to copy to clipboard:', err);
      });
  }

  /**
   * Show debug console with complete message history.
   *
   * Displays all buffered messages from app startup, not just messages
   * since last show(). This is key behavior - you can see what happened
   * during initialization, data loading, etc. even if you open console later.
   *
   * Triggered by Ctrl+L keyboard shortcut when console is hidden.
   *
   * @example
   * ```typescript
   * // Open console after app has been running
   * debugConsole.show();
   * // See ALL messages since startup, not just recent ones
   * ```
   */
  show(): void {
    this.panel.style.display = 'flex';
    this.isVisible = true;

    // Clear and render ALL buffered messages from the global interceptor
    this.contentArea.innerHTML = '';

    // Get ALL messages from the beginning of the app
    const allMessages = consoleInterceptor.getBufferedMessages();

    // Log stats for debugging
    const stats = consoleInterceptor.getStats();
    log.data(
      Modules.DEBUG_CONSOLE,
      `Loading ${stats.total} buffered messages (${stats.types.log} log, ${stats.types.warn} warn, ${stats.types.error} error)`
    );

    // Render each message
    allMessages.forEach((msg) => {
      this.renderBufferedMessage(msg);
    });

    this.updateStatus();

    if (this.autoScroll) {
      this.scrollToBottom();
    }
  }

  /**
   * Hide debug console panel.
   *
   * Messages continue to be buffered by the global interceptor while hidden.
   * Next show() will display complete history.
   *
   * Triggered by Ctrl+L when console is visible, close button, or Escape key.
   */
  hide(): void {
    this.panel.style.display = 'none';
    this.isVisible = false;
  }

  /**
   * Toggle debug console visibility (show ↔ hide).
   *
   * Primary method for Ctrl+L keyboard binding.
   *
   * @example
   * ```typescript
   * // User presses Ctrl+L
   * debugConsole.toggle();
   * ```
   */
  toggle(): void {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Check if debug console is currently visible.
   *
   * @returns true if console panel is shown, false if hidden
   */
  getIsVisible(): boolean {
    return this.isVisible;
  }

  /**
   * Clean up debug console resources and remove from DOM.
   *
   * Unregisters from global console interceptor, removes panel from DOM,
   * and cleans up styles. Should be called during application teardown.
   *
   * After calling dispose(), the DebugConsole instance cannot be reused.
   */
  dispose(): void {
    // Remove listener from global interceptor
    if (this.messageListenerCallback) {
      consoleInterceptor.removeListener(this.messageListenerCallback);
    }

    // Tear down every DOM listener (toolbar buttons, drag/resize, blur watchers).
    this.events.dispose();

    // Remove panel from DOM. Styles live in src/styles/components/
    // debug-console.css and are loaded by Vite — there is no inline
    // <style> element to clean up here.
    this.panel.remove();

    log.info(Modules.DEBUG_CONSOLE, 'Debug console disposed');
  }
}
