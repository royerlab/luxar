/**
 * Debug Console Panel
 * 
 * Displays browser console output in an in-app panel.
 * Uses the global console interceptor to ensure all messages are captured
 * from the very beginning of the application lifecycle.
 */

import { consoleInterceptor, type BufferedMessage } from '../utils/console-interceptor';
import { DEBUG_CONSOLE_CONFIG } from '../config/debug-console';

export interface ConsoleMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  formatted: string;
  stack?: string;
}

export class DebugConsole {
  private panel: HTMLElement;
  private contentArea: HTMLElement;
  private isVisible = false;
  private autoScroll = true;
  private filter: string = '';
  private messageListenerCallback: ((message: BufferedMessage) => void) | null = null;

  constructor() {
    // Create UI
    this.panel = this.createPanel();
    this.contentArea = this.panel.querySelector('.debug-console-content') as HTMLElement;
    
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
    console.log('🔧 [Luxar] Debug console ready (Ctrl+L to open)');
  }

  /**
   * Create the debug console panel UI
   */
  private createPanel(): HTMLElement {
    const panel = document.createElement('div');
    panel.className = 'debug-console-panel';
    panel.innerHTML = `
      <div class="debug-console-header">
        <div class="debug-console-title">🔧 Debug Console</div>
        <div class="debug-console-controls">
          <input type="text" class="debug-console-filter" placeholder="Filter..." />
          <button class="debug-console-clear" title="Clear console">Clear</button>
          <button class="debug-console-copy" title="Copy all to clipboard">Copy</button>
          <label class="debug-console-autoscroll">
            <input type="checkbox" checked /> Auto-scroll
          </label>
          <button class="debug-console-close" title="Close (Ctrl+L)">✕</button>
        </div>
      </div>
      <div class="debug-console-content"></div>
      <div class="debug-console-status">
        <span class="message-count">0 messages</span>
        <span class="filter-status"></span>
      </div>
    `;

    // Apply styles
    this.applyStyles();

    // Setup event handlers
    this.setupEventHandlers(panel);

    return panel;
  }

  /**
   * Apply CSS styles to the panel
   */
  private applyStyles(): void {
    const styleId = 'debug-console-styles';
    if (document.getElementById(styleId)) return;

    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = `
      .debug-console-panel {
        position: fixed;
        bottom: ${DEBUG_CONSOLE_CONFIG.panel.bottomOffset}px;
        right: ${DEBUG_CONSOLE_CONFIG.panel.rightOffset}px;
        width: ${DEBUG_CONSOLE_CONFIG.panel.defaultWidth}px;
        height: ${DEBUG_CONSOLE_CONFIG.panel.defaultHeight}px;
        background: ${DEBUG_CONSOLE_CONFIG.style.backgroundColor};
        border: 1px solid ${DEBUG_CONSOLE_CONFIG.style.borderColor};
        border-radius: ${DEBUG_CONSOLE_CONFIG.style.borderRadius}px;
        display: flex;
        flex-direction: column;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 12px;
        z-index: 10000;
        backdrop-filter: blur(${DEBUG_CONSOLE_CONFIG.style.backdropBlur}px);
        box-shadow: ${DEBUG_CONSOLE_CONFIG.style.boxShadow};
      }

      .debug-console-header {
        padding: 10px;
        background: rgba(30, 30, 30, 0.9);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 8px 8px 0 0;
      }

      .debug-console-title {
        color: #e0e0e0;
        font-weight: bold;
        font-size: 14px;
      }

      .debug-console-controls {
        display: flex;
        gap: 10px;
        align-items: center;
      }

      .debug-console-filter {
        padding: 4px 8px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        color: #e0e0e0;
        width: 150px;
      }

      .debug-console-clear,
      .debug-console-copy,
      .debug-console-close {
        padding: 4px 12px;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        color: #e0e0e0;
        cursor: pointer;
        transition: background 0.2s;
      }

      .debug-console-clear:hover,
      .debug-console-copy:hover,
      .debug-console-close:hover {
        background: rgba(255, 255, 255, 0.2);
      }

      .debug-console-autoscroll {
        color: #e0e0e0;
        display: flex;
        align-items: center;
        gap: 5px;
        cursor: pointer;
      }

      .debug-console-autoscroll input {
        cursor: pointer;
      }

      .debug-console-content {
        flex: 1;
        overflow-y: auto;
        overflow-x: auto;
        padding: 10px;
        background: rgba(10, 10, 10, 0.5);
      }

      .debug-console-status {
        padding: 5px 10px;
        background: rgba(30, 30, 30, 0.9);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        color: #808080;
        font-size: 11px;
        border-radius: 0 0 8px 8px;
      }

      .console-message {
        margin: 2px 0;
        padding: 4px 8px;
        border-radius: 3px;
        word-wrap: break-word;
        font-family: inherit;
        line-height: 1.4;
        position: relative;
        transition: background 0.1s;
      }

      .console-message:hover {
        background: rgba(255, 255, 255, 0.05);
      }

      .console-message-timestamp {
        color: #606060;
        margin-right: 8px;
        font-size: 10px;
      }

      .console-message-log {
        color: #e0e0e0;
      }

      .console-message-info {
        color: #4CAF50;
      }

      .console-message-warn {
        color: #FFC107;
        background: rgba(255, 193, 7, 0.1);
      }

      .console-message-error {
        color: #f44336;
        background: rgba(244, 67, 54, 0.1);
      }

      .console-message-debug {
        color: #9E9E9E;
        font-style: italic;
      }

      .console-message-stack {
        margin-top: 4px;
        padding-left: 20px;
        color: #808080;
        font-size: 10px;
        white-space: pre-wrap;
      }

      .console-message-object {
        color: #64B5F6;
      }

      .console-message-number {
        color: #FF9800;
      }

      .console-message-boolean {
        color: #E91E63;
      }

      .console-message-string {
        color: #8BC34A;
      }

      .console-message-undefined {
        color: #9E9E9E;
        font-style: italic;
      }

      /* Scrollbar styling */
      .debug-console-content::-webkit-scrollbar {
        width: 8px;
        height: 8px;
      }

      .debug-console-content::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
      }

      .debug-console-content::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
      }

      .debug-console-content::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
      }

      /* Resize handle */
      .debug-console-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        cursor: ns-resize;
      }

      .debug-console-panel::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        width: 4px;
        cursor: ew-resize;
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Setup event handlers for the panel
   */
  private setupEventHandlers(panel: HTMLElement): void {
    // Close button
    panel.querySelector('.debug-console-close')?.addEventListener('click', () => {
      this.hide();
    });

    // Clear button
    panel.querySelector('.debug-console-clear')?.addEventListener('click', () => {
      this.clear();
    });

    // Copy button
    panel.querySelector('.debug-console-copy')?.addEventListener('click', () => {
      this.copyToClipboard();
    });

    // Filter input
    const filterInput = panel.querySelector('.debug-console-filter') as HTMLInputElement;
    filterInput?.addEventListener('input', (e) => {
      this.filter = (e.target as HTMLInputElement).value;
      this.applyFilter();
    });

    // Auto-scroll checkbox
    const autoScrollCheckbox = panel.querySelector('.debug-console-autoscroll input') as HTMLInputElement;
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
    const header = panel.querySelector('.debug-console-header') as HTMLElement;
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let initialX = 0;
    let initialY = 0;

    header.style.cursor = 'move';

    header.addEventListener('mousedown', (e) => {
      if ((e.target as HTMLElement).tagName === 'BUTTON' || 
          (e.target as HTMLElement).tagName === 'INPUT') return;
      
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = panel.getBoundingClientRect();
      initialX = rect.left;
      initialY = rect.top;
      
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      panel.style.left = `${initialX + deltaX}px`;
      panel.style.top = `${initialY + deltaY}px`;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
    });
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

      if (y < DEBUG_CONSOLE_CONFIG.resize.borderWidth) {
        isResizing = true;
        resizeDirection = 'n';
        startY = e.clientY;
        startHeight = rect.height;
        e.preventDefault();
      } else if (x < DEBUG_CONSOLE_CONFIG.resize.borderWidth) {
        isResizing = true;
        resizeDirection = 'w';
        startX = e.clientX;
        startWidth = rect.width;
        e.preventDefault();
      }
    });

    document.addEventListener('mousemove', (e) => {
      if (!isResizing) return;

      if (resizeDirection === 'n') {
        const deltaY = startY - e.clientY;
        const newHeight = Math.max(DEBUG_CONSOLE_CONFIG.panel.minHeight, Math.min(DEBUG_CONSOLE_CONFIG.panel.maxHeight, startHeight + deltaY));
        panel.style.height = `${newHeight}px`;
      } else if (resizeDirection === 'w') {
        const deltaX = startX - e.clientX;
        const newWidth = Math.max(DEBUG_CONSOLE_CONFIG.panel.minWidth, Math.min(DEBUG_CONSOLE_CONFIG.panel.maxWidth, startWidth + deltaX));
        panel.style.width = `${newWidth}px`;
      }
    });

    document.addEventListener('mouseup', () => {
      isResizing = false;
      resizeDirection = '';
    });
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
      stack: message.stack
    };
    this.renderMessage(consoleMessage);
  }

  /**
   * Format arguments for display
   */
  private formatArgs(args: any[]): string {
    return args.map(arg => {
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
    }).join(' ');
  }

  /**
   * Render a single message
   */
  private renderMessage(message: ConsoleMessage): void {
    const messageEl = document.createElement('div');
    messageEl.className = `console-message console-message-${message.type}`;
    
    // Format timestamp
    const timestamp = message.timestamp.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });

    // Build message HTML
    let html = `<span class="console-message-timestamp">${timestamp}</span>`;
    
    // Format each argument with appropriate styling
    message.args.forEach((arg, index) => {
      if (index > 0) html += ' ';
      html += this.formatArgWithStyle(arg);
    });

    messageEl.innerHTML = html;

    // Add stack trace if present
    if (message.stack && message.type === 'error') {
      const stackEl = document.createElement('div');
      stackEl.className = 'console-message-stack';
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
   * Format an argument with appropriate styling
   */
  private formatArgWithStyle(arg: any): string {
    if (arg === undefined) {
      return '<span class="console-message-undefined">undefined</span>';
    }
    if (arg === null) {
      return '<span class="console-message-undefined">null</span>';
    }
    if (typeof arg === 'string') {
      return `<span class="console-message-string">"${this.escapeHtml(arg)}"</span>`;
    }
    if (typeof arg === 'number') {
      return `<span class="console-message-number">${arg}</span>`;
    }
    if (typeof arg === 'boolean') {
      return `<span class="console-message-boolean">${arg}</span>`;
    }
    if (typeof arg === 'object') {
      try {
        const json = JSON.stringify(arg, null, 2);
        return `<span class="console-message-object">${this.escapeHtml(json)}</span>`;
      } catch {
        return `<span class="console-message-object">${this.escapeHtml(arg.toString())}</span>`;
      }
    }
    return this.escapeHtml(String(arg));
  }

  /**
   * Escape HTML for safe display
   */
  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Apply filter to existing messages
   */
  private applyFilter(): void {
    const messages = this.contentArea.querySelectorAll('.console-message');
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
      const visibleCount = this.contentArea.querySelectorAll('.console-message:not([style*="display: none"])').length;
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
   * Clear all messages
   */
  clear(): void {
    // Clear the global buffer
    consoleInterceptor.clearBuffer();
    // Clear the UI
    this.contentArea.innerHTML = '';
    this.updateStatus();
    console.log('Debug console cleared');
  }

  /**
   * Copy all messages to clipboard
   */
  private copyToClipboard(): void {
    const messages = consoleInterceptor.getBufferedMessages();
    const text = messages
      .map(m => {
        const formatted = this.formatArgs(m.args);
        return `[${m.timestamp.toISOString()}] [${m.type.toUpperCase()}] ${formatted}`;
      })
      .join('\n');
    
    navigator.clipboard.writeText(text).then(() => {
      console.log('Console output copied to clipboard');
    }).catch(err => {
      console.error('Failed to copy to clipboard:', err);
    });
  }

  /**
   * Show the debug console
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
    console.log(`📊 [Luxar] Loading ${stats.total} buffered messages (${stats.types.log} log, ${stats.types.warn} warn, ${stats.types.error} error)`);
    
    // Render each message
    allMessages.forEach(msg => {
      this.renderBufferedMessage(msg);
    });
    
    this.updateStatus();
    
    if (this.autoScroll) {
      this.scrollToBottom();
    }
  }

  /**
   * Hide the debug console
   */
  hide(): void {
    this.panel.style.display = 'none';
    this.isVisible = false;
  }

  /**
   * Toggle visibility
   */
  toggle(): void {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Check if console is visible
   */
  getIsVisible(): boolean {
    return this.isVisible;
  }

  /**
   * Clean up the debug console
   */
  dispose(): void {
    // Remove listener from global interceptor
    if (this.messageListenerCallback) {
      consoleInterceptor.removeListener(this.messageListenerCallback);
    }
    
    // Remove panel from DOM
    this.panel.remove();
    
    // Remove styles
    const style = document.getElementById('debug-console-styles');
    style?.remove();
    
    console.log('Debug console disposed');
  }
}