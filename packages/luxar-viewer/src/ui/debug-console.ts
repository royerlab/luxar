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
import { log, Modules, LogEmoji } from '../utils/log';

export interface ConsoleMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  formatted: string;
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

  // Store bound resize handlers for cleanup
  private boundDragMouseMove: ((e: MouseEvent) => void) | null = null;
  private boundDragMouseUp: (() => void) | null = null;
  private boundResizeMouseMove: ((e: MouseEvent) => void) | null = null;
  private boundResizeMouseUp: (() => void) | null = null;

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
    document.body.appendChild(this.panel);

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
    panel.className = 'luxar-debug-console';
    panel.innerHTML = `
      <div class="luxar-debug-console__header">
        <div class="luxar-debug-console__title">Debug Console</div>
        <div class="luxar-debug-console__controls">
          <input type="text" class="luxar-debug-console__filter" placeholder="Filter..." />
          <button class="luxar-debug-console__clear-btn" title="Clear console">Clear</button>
          <button class="luxar-debug-console__copy-btn" title="Copy all to clipboard">Copy</button>
          <label class="luxar-debug-console__autoscroll">
            <input type="checkbox" checked /> Auto-scroll
          </label>
          <button class="luxar-debug-console__close-btn" title="Close (Ctrl+L)">×</button>
        </div>
      </div>
      <div class="luxar-debug-console__content"></div>
      <div class="luxar-debug-console__status">
        <span class="message-count">0 messages</span>
        <span class="filter-status"></span>
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
    // Close button
    panel.querySelector('.luxar-debug-console__close-btn')?.addEventListener('click', () => {
      this.hide();
    });

    // Clear button
    panel.querySelector('.luxar-debug-console__clear-btn')?.addEventListener('click', () => {
      this.clear();
    });

    // Copy button
    panel.querySelector('.luxar-debug-console__copy-btn')?.addEventListener('click', () => {
      this.copyToClipboard();
    });

    // Filter input
    const filterInput = panel.querySelector('.luxar-debug-console__filter') as HTMLInputElement;
    filterInput?.addEventListener('input', (e) => {
      this.filter = (e.target as HTMLInputElement).value;
      this.applyFilter();
    });

    // Auto-scroll checkbox
    const autoScrollCheckbox = panel.querySelector(
      '.luxar-debug-console__autoscroll input'
    ) as HTMLInputElement;
    autoScrollCheckbox?.addEventListener('change', (e) => {
      this.autoScroll = (e.target as HTMLInputElement).checked;
    });

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

    header.addEventListener('mousedown', (e) => {
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

    // Create bound handlers for cleanup
    this.boundDragMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;

      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;

      panel.style.left = `${initialX + deltaX}px`;
      panel.style.top = `${initialY + deltaY}px`;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    };

    this.boundDragMouseUp = () => {
      isDragging = false;
    };

    document.addEventListener('mousemove', this.boundDragMouseMove);
    document.addEventListener('mouseup', this.boundDragMouseUp);
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

    panel.addEventListener('mousedown', (e) => {
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

    // Create bound handlers for cleanup
    this.boundResizeMouseMove = (e: MouseEvent) => {
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

    this.boundResizeMouseUp = () => {
      isResizing = false;
      resizeDirection = '';
    };

    document.addEventListener('mousemove', this.boundResizeMouseMove);
    document.addEventListener('mouseup', this.boundResizeMouseUp);
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
  private formatArgs(args: any[]): string {
    return args
      .map((arg) => {
        if (arg === undefined) return 'undefined';
        if (arg === null) return 'null';
        if (typeof arg === 'string') return arg;
        if (typeof arg === 'number') return arg.toString();
        if (typeof arg === 'boolean') return arg.toString();
        if (typeof arg === 'object') {
          try {
            return JSON.stringify(arg, null, 2);
          } catch {
            return arg.toString();
          }
        }
        return String(arg);
      })
      .join(' ');
  }

  /**
   * Render a single message
   */
  private renderMessage(message: ConsoleMessage): void {
    const messageEl = document.createElement('div');
    messageEl.className = `luxar-console-message luxar-console-message-${message.type}`;

    // Format timestamp
    const timestamp = message.timestamp.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
    });

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

    // Add stack trace if present
    if (message.stack && message.type === 'error') {
      const stackEl = document.createElement('div');
      stackEl.className = 'luxar-console-message-stack';
      stackEl.textContent = message.stack;
      messageEl.appendChild(stackEl);
    }

    // Check filter
    if (this.filter && !message.formatted.toLowerCase().includes(this.filter.toLowerCase())) {
      messageEl.style.display = 'none';
    }

    this.contentArea.appendChild(messageEl);
  }

  /**
   * Format an argument as a DOM element (safe, no XSS)
   */
  private formatArgAsDOMElement(arg: any): HTMLElement {
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
    } else if (typeof arg === 'object') {
      span.className = 'luxar-console-message-object';
      try {
        const json = JSON.stringify(arg, null, 2);
        span.textContent = json;
      } catch {
        span.textContent = arg.toString();
      }
    } else {
      span.textContent = String(arg);
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
      if (this.filter && !text.toLowerCase().includes(this.filter.toLowerCase())) {
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
    const countEl = this.panel.querySelector('.message-count');
    const filterEl = this.panel.querySelector('.filter-status');

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
        return `[${m.timestamp.toISOString()}] [${m.type.toUpperCase()}] ${formatted}`;
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

    // Remove global event listeners for dragging and resizing
    if (this.boundDragMouseMove) {
      document.removeEventListener('mousemove', this.boundDragMouseMove);
      this.boundDragMouseMove = null;
    }
    if (this.boundDragMouseUp) {
      document.removeEventListener('mouseup', this.boundDragMouseUp);
      this.boundDragMouseUp = null;
    }
    if (this.boundResizeMouseMove) {
      document.removeEventListener('mousemove', this.boundResizeMouseMove);
      this.boundResizeMouseMove = null;
    }
    if (this.boundResizeMouseUp) {
      document.removeEventListener('mouseup', this.boundResizeMouseUp);
      this.boundResizeMouseUp = null;
    }

    // Remove panel from DOM
    this.panel.remove();

    // Remove styles
    const style = document.getElementById('debug-console-styles');
    style?.remove();

    log.info(Modules.DEBUG_CONSOLE, 'Debug console disposed');
  }
}
