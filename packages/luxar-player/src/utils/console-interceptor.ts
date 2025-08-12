/**
 * Early Console Interceptor
 *
 * This module intercepts console methods at the very beginning of the application
 * lifecycle to ensure no messages are missed. It maintains a global buffer that
 * the DebugConsole can later consume.
 *
 * IMPORTANT: This must be imported before any other code that uses console methods.
 */

export interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: any[];
  stack?: string;
}

class ConsoleInterceptor {
  private static instance: ConsoleInterceptor;

  /** Ring buffer for messages - automatically handles overflow */
  private messageBuffer: BufferedMessage[] = [];

  /** Current write position in ring buffer */
  private bufferIndex = 0;

  /** Maximum messages to buffer (configurable) */
  private readonly maxBufferSize = 10000; // TODO: Import from config when circular dependency is resolved

  /** Whether buffer has wrapped around */
  private hasWrapped = false;

  /** Original console methods */
  private originalConsole: {
    log: typeof console.log;
    warn: typeof console.warn;
    error: typeof console.error;
    info: typeof console.info;
    debug: typeof console.debug;
  };

  /** Callbacks for new messages */
  private listeners: Set<(message: BufferedMessage) => void> = new Set();

  /** Flag to ensure we only intercept once */
  private isIntercepting = false;

  private constructor() {
    // Store original console methods immediately
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      info: console.info.bind(console),
      debug: console.debug.bind(console),
    };

    // Start interception immediately
    this.startInterception();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): ConsoleInterceptor {
    if (!ConsoleInterceptor.instance) {
      ConsoleInterceptor.instance = new ConsoleInterceptor();
    }
    return ConsoleInterceptor.instance;
  }

  /**
   * Start intercepting console methods
   */
  private startInterception(): void {
    if (this.isIntercepting) return;
    this.isIntercepting = true;

    // Override console.log
    console.log = (...args: any[]) => {
      this.captureMessage('log', args);
      this.originalConsole.log(...args);
    };

    // Override console.warn
    console.warn = (...args: any[]) => {
      this.captureMessage('warn', args);
      this.originalConsole.warn(...args);
    };

    // Override console.error
    console.error = (...args: any[]) => {
      const error = args[0];
      const stack = this.extractStack(error);
      this.captureMessage('error', args, stack);
      this.originalConsole.error(...args);
    };

    // Override console.info
    console.info = (...args: any[]) => {
      this.captureMessage('info', args);
      this.originalConsole.info(...args);
    };

    // Override console.debug
    console.debug = (...args: any[]) => {
      this.captureMessage('debug', args);
      this.originalConsole.debug(...args);
    };

    // Log that interception has started (using original console)
    this.originalConsole.log('🎬 [Luxar] Console interception started - capturing all output');
  }

  /**
   * Extract stack trace from error object or create one
   */
  private extractStack(error: any): string | undefined {
    if (error?.stack) {
      return error.stack;
    }

    // Create a stack trace if it's an error message without stack
    if (typeof error === 'string' && error.toLowerCase().includes('error')) {
      const tempError = new Error();
      return tempError.stack;
    }

    return undefined;
  }

  /**
   * Capture a console message using ring buffer pattern
   */
  private captureMessage(type: BufferedMessage['type'], args: any[], stack?: string): void {
    const message: BufferedMessage = {
      type,
      timestamp: new Date(),
      args: [...args], // Clone args to prevent mutation
      stack,
    };

    // Ring buffer implementation - overwrite oldest when full
    if (this.messageBuffer.length < this.maxBufferSize) {
      // Buffer not full yet, just append
      this.messageBuffer.push(message);
      this.bufferIndex = this.messageBuffer.length;
    } else {
      // Buffer full, overwrite oldest message
      this.messageBuffer[this.bufferIndex] = message;
      this.bufferIndex = (this.bufferIndex + 1) % this.maxBufferSize;
      this.hasWrapped = true;
    }

    // Notify listeners
    this.listeners.forEach((listener) => {
      try {
        listener(message);
      } catch (err) {
        // Use original console to avoid recursion
        this.originalConsole.error('Error in console listener:', err);
      }
    });
  }

  /**
   * Get all buffered messages in chronological order
   */
  getBufferedMessages(): BufferedMessage[] {
    if (!this.hasWrapped) {
      // Buffer hasn't wrapped, return as-is
      return [...this.messageBuffer];
    }

    // Buffer has wrapped, reconstruct in chronological order
    // Oldest messages are from bufferIndex to end, newest are from 0 to bufferIndex-1
    const oldestPart = this.messageBuffer.slice(this.bufferIndex);
    const newestPart = this.messageBuffer.slice(0, this.bufferIndex);
    return [...oldestPart, ...newestPart];
  }

  /**
   * Clear the message buffer
   */
  clearBuffer(): void {
    this.messageBuffer = [];
    this.bufferIndex = 0;
    this.hasWrapped = false;
  }

  /**
   * Add a listener for new messages
   */
  addListener(callback: (message: BufferedMessage) => void): void {
    this.listeners.add(callback);
  }

  /**
   * Remove a message listener
   */
  removeListener(callback: (message: BufferedMessage) => void): void {
    this.listeners.delete(callback);
  }

  /**
   * Get original console methods
   */
  getOriginalConsole() {
    return this.originalConsole;
  }

  /**
   * Get buffer statistics
   */
  getStats() {
    const typeCounts = {
      log: 0,
      warn: 0,
      error: 0,
      info: 0,
      debug: 0,
    };

    this.messageBuffer.forEach((msg) => {
      typeCounts[msg.type]++;
    });

    return {
      total: this.messageBuffer.length,
      maxSize: this.maxBufferSize,
      types: typeCounts,
      oldestMessage: this.messageBuffer[0]?.timestamp,
      newestMessage: this.messageBuffer[this.messageBuffer.length - 1]?.timestamp,
    };
  }

  /**
   * Restore original console methods (for cleanup)
   */
  restore(): void {
    if (!this.isIntercepting) return;

    console.log = this.originalConsole.log;
    console.warn = this.originalConsole.warn;
    console.error = this.originalConsole.error;
    console.info = this.originalConsole.info;
    console.debug = this.originalConsole.debug;

    this.isIntercepting = false;
    this.originalConsole.log('🛑 [Luxar] Console interception stopped - restored original methods');
  }
}

// Create and export singleton instance immediately
export const consoleInterceptor = ConsoleInterceptor.getInstance();

// Also export the type for the singleton
export type { ConsoleInterceptor };
