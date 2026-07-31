/**
 * Early Console Interceptor
 *
 * This module intercepts console methods at the very beginning of the application
 * lifecycle to ensure no messages are missed. It maintains a global buffer that
 * the DebugConsole can later consume.
 *
 * IMPORTANT: This must be imported before any other code that uses console methods.
 */

import { getErrorStack } from './format-error';

export interface BufferedMessage {
  type: 'log' | 'warn' | 'error' | 'info' | 'debug';
  timestamp: Date;
  args: unknown[];
  stack?: string;
}

/**
 * Default ring-buffer capacity used until the bootstrap calls
 * {@link ConsoleInterceptor.setMaxBufferSize} with the config value. Kept as
 * an exported constant so consumers (config, tests) can reference the same
 * number without a magic literal.
 */
export const DEFAULT_MAX_BUFFER_SIZE = 10000;

class ConsoleInterceptor {
  private static instance: ConsoleInterceptor;

  /** Ring buffer for messages - automatically handles overflow */
  private messageBuffer: BufferedMessage[] = [];

  /** Current write position in ring buffer */
  private bufferIndex = 0;

  /** Maximum messages to buffer. The default mirrors
   * `config.ui.debugConsole.interceptor.maxBufferSize` — importing `config`
   * here is unsafe (this singleton is constructed at module-load time, before
   * the config module is guaranteed to have finished evaluating). Instead, the
   * bootstrap can call {@link setMaxBufferSize} after both modules are loaded
   * to push the canonical value, keeping the two in sync without a manual edit.
   *
   * @see {@link DEFAULT_MAX_BUFFER_SIZE} — the constant.
   * @see {@link setMaxBufferSize} — the setter the config layer uses.
   */
  private maxBufferSize = DEFAULT_MAX_BUFFER_SIZE;

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
    // Capture the platform's original console methods at construction time so
    // we can later restore them. Construction does NOT patch console — call
    // {@link patch} explicitly to start intercepting.
    //
    // This deliberate separation matters for embedding: importing this
    // module from a published library must not silently monkey-patch the
    // host page's console.
    this.originalConsole = {
      log: console.log.bind(console),
      warn: console.warn.bind(console),
      error: console.error.bind(console),
      info: console.info.bind(console),
      debug: console.debug.bind(console),
    };
  }

  /**
   * Get singleton instance.
   *
   * Construction is lazy and side-effect-free. The singleton stores the
   * platform's original console methods, but does NOT patch them — call
   * {@link patch} on the returned instance to start intercepting.
   */
  static getInstance(): ConsoleInterceptor {
    if (!ConsoleInterceptor.instance) {
      ConsoleInterceptor.instance = new ConsoleInterceptor();
    }
    return ConsoleInterceptor.instance;
  }

  /**
   * Dispose the singleton instance, unpatching console first if necessary.
   *
   * After this call, the next {@link getInstance} returns a fresh interceptor
   * with an empty buffer. Used at app shutdown and between tests.
   */
  static disposeInstance(): void {
    if (ConsoleInterceptor.instance) {
      ConsoleInterceptor.instance.dispose();
      ConsoleInterceptor.instance = undefined as unknown as ConsoleInterceptor;
    }
  }

  /**
   * Whether console.* is currently being intercepted by this instance.
   */
  get isPatched(): boolean {
    return this.isIntercepting;
  }

  /**
   * Start intercepting console methods. Idempotent — safe to call repeatedly.
   *
   * After patch(), every console.log/warn/error/info/debug call also lands in
   * the ring buffer. Reverse with {@link dispose}.
   */
  patch(): void {
    if (this.isIntercepting) return;
    this.isIntercepting = true;

    // Override console.log
    console.log = (...args: unknown[]) => {
      this.captureMessage('log', args);
      this.originalConsole.log(...args);
    };

    // Override console.warn
    console.warn = (...args: unknown[]) => {
      // Warnings get a stack too: ~30 `log.warning(…, error)` sites pass a real
      // Error, and without this the in-app console had neither its message (see
      // the formatters' Error branch) nor any trace to fall back on.
      this.captureMessage('warn', args, this.extractStack(args));
      this.originalConsole.warn(...args);
    };

    // Override console.error
    console.error = (...args: unknown[]) => {
      this.captureMessage('error', args, this.extractStack(args));
      this.originalConsole.error(...args);
    };

    // Override console.info
    console.info = (...args: unknown[]) => {
      this.captureMessage('info', args);
      this.originalConsole.info(...args);
    };

    // Override console.debug
    console.debug = (...args: unknown[]) => {
      this.captureMessage('debug', args);
      this.originalConsole.debug(...args);
    };

    // Log that interception has started (using original console)
    // Note: We can't use the log utility here since it would create a circular dependency
    this.originalConsole.log(
      '[🎬] [ConsoleInterceptor] Console interception started - capturing all output'
    );
  }

  /**
   * The stack of the first real `Error` among `args`, falling back to the first
   * duck-typed stack carrier, or `undefined`.
   *
   * Scans ALL args rather than just the first: every `log.*` call formats its
   * message into `args[0]` as a STRING and passes the error along behind it, so
   * looking only at `args[0]` could never find one.
   *
   * Two passes, and the order matters: a plain object that merely carries a
   * `stack` string (a context bag such as `{ stack: 'phase: decode' }`) must not
   * shadow the real `Error` behind it. Real `Error` instances win — including
   * cross-realm ones (another window/iframe), caught by the `[object Error]`
   * brand check where `instanceof` fails; the
   * duck-typed carriers `getErrorStack` also accepts (some Firefox
   * `DOMException`s, thrown plain objects) are only the fallback.
   *
   * Deliberately does NOT fabricate a stack. This used to synthesize
   * `new Error().stack` whenever a message merely contained the word "error",
   * which produced a plausible-looking trace rooted inside this interceptor —
   * worse than no stack, because someone reading the bug report would follow it.
   */
  private extractStack(args: readonly unknown[]): string | undefined {
    for (const arg of args) {
      // Guard the checks: `instanceof` walks [[GetPrototypeOf]] and the brand
      // check's Symbol.toStringTag lookup does a [[Get]] — both throw for a
      // revoked Proxy. This runs inside the patched console.warn/error BEFORE
      // the original call, so a throw here would swallow the diagnostic being
      // logged — the same never-throw contract `getErrorStack` upholds.
      let isError = false;
      try {
        // `instanceof` alone is realm-dependent: an Error created in another
        // window/iframe has a foreign Error.prototype and fails it, letting an
        // earlier context bag shadow the real stack. The spec brand check sees
        // the [[ErrorData]] internal slot regardless of realm.
        isError = arg instanceof Error || Object.prototype.toString.call(arg) === '[object Error]';
      } catch {
        isError = false;
      }
      if (isError) {
        const stack = getErrorStack(arg);
        if (stack) return stack;
      }
    }
    for (const arg of args) {
      const stack = getErrorStack(arg);
      if (stack) return stack;
    }
    return undefined;
  }

  /**
   * Capture a console message using ring buffer pattern
   */
  private captureMessage(type: BufferedMessage['type'], args: unknown[], stack?: string): void {
    const message: BufferedMessage = {
      type,
      timestamp: new Date(),
      args: [...args], // Clone args to prevent mutation
      stack,
    };

    // Ring buffer: while filling, append. Once full, overwrite at bufferIndex
    // and advance modulo maxBufferSize. The push branch must wrap on the
    // exact fill boundary (length === maxBufferSize) so the next overwrite
    // hits index 0 (the oldest entry) rather than maxBufferSize (out of
    // bounds, which previously grew the array by one and stranded index 0).
    if (this.messageBuffer.length < this.maxBufferSize) {
      this.messageBuffer.push(message);
      this.bufferIndex = this.messageBuffer.length % this.maxBufferSize;
    } else {
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
   * Override the ring-buffer capacity. The config layer calls this once after
   * both modules are loaded so the interceptor uses the canonical value from
   * `config.ui.debugConsole.interceptor.maxBufferSize` without forcing a
   * cycle-prone import at module-load time. Shrinking trims oldest-first to
   * preserve chronological order; growing leaves existing messages intact and
   * just allows more headroom before wrap.
   *
   * @param size - New maximum buffer size; must be a positive integer.
   */
  setMaxBufferSize(size: number): void {
    if (!Number.isInteger(size) || size <= 0) {
      this.originalConsole.warn(
        `[ConsoleInterceptor] setMaxBufferSize ignored: expected positive integer, got ${size}`
      );
      return;
    }
    if (size === this.maxBufferSize) return;

    // Reconstruct messages in chronological order first so we can trim from
    // the head when shrinking (oldest-first eviction).
    const chronological = this.getBufferedMessages();
    if (chronological.length > size) {
      this.messageBuffer = chronological.slice(chronological.length - size);
    } else {
      this.messageBuffer = chronological;
    }
    this.maxBufferSize = size;
    this.hasWrapped = this.messageBuffer.length === size;
    this.bufferIndex = this.hasWrapped ? 0 : this.messageBuffer.length;
  }

  /** Current ring-buffer capacity (for diagnostics / tests). */
  getMaxBufferSize(): number {
    return this.maxBufferSize;
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
   * Restore the original console methods and stop intercepting.
   *
   * After dispose(), captureMessage() is no longer reachable through
   * `console.log` etc. The buffer is preserved (callers can still read
   * `getBufferedMessages()`), and the singleton slot is left intact —
   * call {@link ConsoleInterceptor.disposeInstance} to clear it.
   */
  dispose(): void {
    if (!this.isIntercepting) return;

    console.log = this.originalConsole.log;
    console.warn = this.originalConsole.warn;
    console.error = this.originalConsole.error;
    console.info = this.originalConsole.info;
    console.debug = this.originalConsole.debug;

    this.isIntercepting = false;
    this.originalConsole.log(
      '[🛑] [ConsoleInterceptor] Console interception stopped - restored original methods'
    );
  }
}

/**
 * Lazy proxy for the singleton. Property access on this object resolves to
 * the live `ConsoleInterceptor` (constructed on first use). Importing the
 * symbol is itself side-effect-free — no console patching, no allocations.
 *
 * Patch the host console explicitly via `consoleInterceptor.patch()` from
 * the bootstrap path that wants buffered console output (e.g. main.ts for
 * the standalone app, LuxarApp.init({ debug: true }) for embedded use).
 */
export const consoleInterceptor: ConsoleInterceptor = new Proxy({} as ConsoleInterceptor, {
  get(_target, prop, receiver) {
    const instance = ConsoleInterceptor.getInstance();
    const value = Reflect.get(instance, prop, receiver);
    return typeof value === 'function' ? value.bind(instance) : value;
  },
  set(_target, prop, value, receiver) {
    const instance = ConsoleInterceptor.getInstance();
    return Reflect.set(instance, prop, value, receiver);
  },
  has(_target, prop) {
    return prop in ConsoleInterceptor.getInstance();
  },
});

/**
 * Dispose the console-interceptor singleton, restoring the host's original
 * `console.*` methods (if patched) and clearing the buffer. Idempotent and
 * side-effect-free when never patched. Called from the app dispose pipeline
 * so a mount/unmount cycle leaves the host console exactly as it found it.
 */
export function disposeConsoleInterceptor(): void {
  ConsoleInterceptor.disposeInstance();
}

// Also export the type for the singleton
export type { ConsoleInterceptor };
