/**
 * Update Profiler - Hierarchical timing for scene updates
 *
 * Provides low-overhead profiling of the scene update pipeline with:
 * - Hierarchical timing breakdown (parent/child relationships)
 * - Exponential moving average (alpha=0.1) for smooth averages
 * - Metadata tracking (chunks, cache hits, point counts)
 * - 60fps budget awareness (>16ms highlighted)
 * - Context/ambient pattern with stack-based session management
 * - Convenient time() helper for wrapping async operations
 *
 * Usage:
 * ```typescript
 * // Simple case - wrap with time()
 * const data = await profiler.time('Spatial Query', () =>
 *   this.queryChunks(viewState)
 * );
 *
 * // Complex case - manual begin/end with metadata
 * const session = profiler.begin('Chunk Load');
 * session.setMetadata({ chunks: ranges.length });
 * // ... do work ...
 * session.end();
 * ```
 */

import { log, Modules } from '../utils/log';

/**
 * Metadata that can be attached to timing entries
 */
export interface TimingMetadata {
  /** Number of chunks loaded */
  chunks?: number;
  /** Cache hits (L1 + L2) */
  cacheHits?: number;
  /** Cache misses (network fetches) */
  cacheMisses?: number;
  /** Points visible/loaded */
  points?: number;
  /** Segments visible/loaded */
  segments?: number;
  /** Splats visible/loaded */
  splats?: number;
  /** Whether this operation was skipped */
  skipped?: boolean;
  /** Reason for skipping */
  skipReason?: string;
  /** Custom string data */
  info?: string;
}

/**
 * A single timing entry in the hierarchy
 */
export interface TimingEntry {
  /** Name of this timing entry */
  name: string;
  /** Last measured duration in ms */
  lastMs: number;
  /** Exponential moving average in ms */
  avgMs: number;
  /** Number of measurements (for debugging) */
  count: number;
  /** Child timing entries */
  children: TimingEntry[];
  /** Optional metadata */
  metadata?: TimingMetadata;
  /** Whether this entry exceeds 60fps budget (>16ms) */
  overBudget?: boolean;
}

/**
 * Session for recording a single update cycle
 */
export interface UpdateSession {
  /** Begin a child timing entry */
  begin(name: string): UpdateSession;
  /** End this timing entry */
  end(): void;
  /** Set metadata on this entry */
  setMetadata(meta: Partial<TimingMetadata>): void;
  /** Mark this entry as skipped */
  markSkipped(reason: string): void;
}

// EMA alpha for smoothing (0.1 = slow adaptation, stable averages)
const EMA_ALPHA = 0.1;

// 60fps frame budget in ms
const FRAME_BUDGET_MS = 16.67;

/**
 * Internal session implementation
 */
class SessionImpl implements UpdateSession {
  private readonly entry: TimingEntry;
  private readonly startTime: number;
  private ended = false;
  private readonly parent: SessionImpl | null;
  private readonly profiler: UpdateProfiler;

  constructor(name: string, parent: SessionImpl | null, profiler: UpdateProfiler) {
    this.entry = {
      name,
      lastMs: 0,
      avgMs: 0,
      count: 0,
      children: [],
    };
    this.startTime = performance.now();
    this.parent = parent;
    this.profiler = profiler;

    // Add to parent's children if we have a parent
    if (parent) {
      parent.entry.children.push(this.entry);
    }
  }

  begin(name: string): UpdateSession {
    return new SessionImpl(name, this, this.profiler);
  }

  end(): void {
    if (this.ended) return;
    this.ended = true;

    const duration = performance.now() - this.startTime;
    this.entry.lastMs = duration;
    this.entry.count++;

    // Update EMA
    if (this.entry.count === 1) {
      this.entry.avgMs = duration;
    } else {
      this.entry.avgMs = EMA_ALPHA * duration + (1 - EMA_ALPHA) * this.entry.avgMs;
    }

    // Check budget
    this.entry.overBudget = duration > FRAME_BUDGET_MS;

    // Merge into profiler's persistent state
    this.profiler._mergeEntry(this.entry, this.parent?.entry.name);
  }

  setMetadata(meta: Partial<TimingMetadata>): void {
    this.entry.metadata = { ...this.entry.metadata, ...meta };
  }

  markSkipped(reason: string): void {
    this.entry.metadata = {
      ...this.entry.metadata,
      skipped: true,
      skipReason: reason,
    };
    // Skipped entries have 0 duration
    this.entry.lastMs = 0;
    this.entry.avgMs = 0;
  }

  getEntry(): TimingEntry {
    return this.entry;
  }
}

/**
 * Root session that tracks the entire update
 */
export class RootSession extends SessionImpl {
  constructor(profiler: UpdateProfiler) {
    super('Total Update', null, profiler);
  }
}

/**
 * No-op session returned when no update is active
 * Allows code to call profiler methods without checking if profiling is active
 */
class NoOpSession implements UpdateSession {
  begin(_name: string): UpdateSession {
    return this;
  }
  end(): void {
    // No-op
  }
  setMetadata(_meta: Partial<TimingMetadata>): void {
    // No-op
  }
  markSkipped(_reason: string): void {
    // No-op
  }
}

const NOOP_SESSION = new NoOpSession();

// ScopedSession class removed - no longer needed with explicit parent tracking

/**
 * Main profiler class
 *
 * Supports two usage patterns:
 *
 * 1. Context/ambient pattern (recommended):
 *    ```typescript
 *    profiler.beginUpdate();
 *    await profiler.time('Points', async () => {
 *      await profiler.time('Spatial Query', () => query());
 *      await profiler.time('GPU Upload', () => upload());
 *    });
 *    profiler.endUpdate();
 *    ```
 *
 * 2. Manual session management:
 *    ```typescript
 *    const session = profiler.beginUpdate();
 *    const pointsSession = session.begin('Points');
 *    // ... do work ...
 *    pointsSession.end();
 *    session.end();
 *    ```
 *
 * NOTE: Supports concurrent async operations. Each time() call tracks its
 * parent context at call time, so parallel operations work correctly.
 */
export class UpdateProfiler {
  // Persistent timing state (survives across updates)
  private rootEntry: TimingEntry = {
    name: 'Total Update',
    lastMs: 0,
    avgMs: 0,
    count: 0,
    children: [],
  };

  // Current active session (null when not profiling)
  private activeSession: RootSession | null = null;

  // Track active sessions by ID for concurrent async support
  // Each async operation gets its own context that doesn't interfere with others
  private activeSessions = new Map<number, UpdateSession>();
  private nextSessionId = 0;

  // Track the current session context for nested operations
  // Uses AsyncLocalStorage-like pattern: each sync execution path has its own context
  private currentSessionContext: UpdateSession | null = null;

  // Listeners for UI updates
  private listeners = new Set<() => void>();

  /**
   * Begin a new update cycle (root session)
   * Call this at the start of each update pipeline
   */
  beginUpdate(): UpdateSession {
    // End any previous session that wasn't properly closed
    if (this.activeSession) {
      log.warning(Modules.PERFORMANCE, 'Previous session was not ended properly');
    }

    this.activeSession = new RootSession(this);
    this.currentSessionContext = this.activeSession;
    return this.activeSession;
  }

  /**
   * End the current update cycle
   * Call this at the end of each update pipeline
   */
  endUpdate(): void {
    if (this.activeSession) {
      this.activeSession.end();
      this.currentSessionContext = null;
      this.activeSessions.clear();
    }
  }

  /**
   * Begin a child timing entry directly under the root
   * Use this for top-level parallel operations (Points, Lines, GSplats)
   *
   * @param name - Name of this timing entry
   * @returns Session that should be passed to nested operations
   */
  beginTopLevel(name: string): UpdateSession {
    if (!this.activeSession) {
      return NOOP_SESSION;
    }
    // Create child directly under root
    return this.activeSession.begin(name);
  }

  /**
   * Begin a child timing entry of the current innermost session
   * This is the context/ambient pattern - no need to pass session around
   *
   * NOTE: For concurrent async operations at the top level, use beginTopLevel()
   * or timeTopLevel() instead to avoid session stack corruption.
   *
   * @param name - Name of this timing entry
   * @returns Session
   */
  begin(name: string): UpdateSession {
    const parent = this.currentSessionContext;
    if (!parent) {
      // No active update - return no-op session
      return NOOP_SESSION;
    }

    return parent.begin(name);
  }

  /**
   * Convenience: run a top-level parallel operation with timing
   * Creates a direct child of root, safe for concurrent use with Promise.all
   *
   * @param name - Name of this timing entry (e.g., 'Points (/path)')
   * @param fn - Async function to time
   * @returns Result of fn()
   */
  async timeTopLevel<T>(name: string, fn: (session: UpdateSession) => Promise<T>): Promise<T> {
    if (!this.activeSession) {
      return fn(NOOP_SESSION);
    }

    // Create session directly under root (not using shared stack)
    const session = this.activeSession.begin(name);
    const sessionId = this.nextSessionId++;
    this.activeSessions.set(sessionId, session);

    try {
      // Session is passed explicitly to fn — do NOT modify currentSessionContext
      // here since concurrent timeTopLevel calls (via Promise.all in scene-loader)
      // would corrupt the save/restore interleaving.
      const result = await fn(session);

      session.end();
      this.activeSessions.delete(sessionId);

      return result;
    } catch (e) {
      session.end();
      this.activeSessions.delete(sessionId);
      throw e;
    }
  }

  /**
   * Convenience: run a function with timing
   * Automatically handles begin/end and preserves return value
   *
   * NOTE: For top-level parallel operations, use timeTopLevel() instead.
   *
   * @param name - Name of this timing entry
   * @param fn - Sync or async function to time
   * @returns Result of fn()
   */
  time<T>(name: string, fn: () => T): T {
    const session = this.begin(name);
    try {
      const result = fn();
      // Handle promises
      if (result instanceof Promise) {
        return result.finally(() => session.end()) as T;
      }
      session.end();
      return result;
    } catch (e) {
      session.end();
      throw e;
    }
  }

  /**
   * Convenience: run an async function with timing and metadata
   * Allows setting metadata before the operation completes
   *
   * @param name - Name of this timing entry
   * @param fn - Function that receives session for metadata and returns result
   * @returns Result of fn()
   */
  timeWithMeta<T>(name: string, fn: (session: UpdateSession) => T): T {
    const session = this.begin(name);
    try {
      const result = fn(session);
      // Handle promises
      if (result instanceof Promise) {
        return result.finally(() => session.end()) as T;
      }
      session.end();
      return result;
    } catch (e) {
      session.end();
      throw e;
    }
  }

  /**
   * Mark an operation as skipped (for extend_to_all, etc.)
   * Creates a timing entry with 0 duration and skip reason
   *
   * @param name - Name of the skipped operation
   * @param reason - Why it was skipped
   */
  skip(name: string, reason: string): void {
    const session = this.begin(name);
    session.markSkipped(reason);
    session.end();
  }

  /**
   * Check if profiling is currently active
   * Useful for conditional instrumentation
   */
  isActive(): boolean {
    return this.activeSession !== null;
  }

  /**
   * Get the current innermost session (for setting metadata)
   * Returns NoOpSession if no update is active
   *
   * @example
   * profiler.current().setMetadata({ points: 50000 });
   */
  current(): UpdateSession {
    return this.currentSessionContext ?? NOOP_SESSION;
  }

  /**
   * Get the current timing hierarchy (for UI display)
   */
  getTimings(): TimingEntry {
    return this.rootEntry;
  }

  /**
   * Reset all timing data
   */
  reset(): void {
    this.rootEntry = {
      name: 'Total Update',
      lastMs: 0,
      avgMs: 0,
      count: 0,
      children: [],
    };
    this.notifyListeners();
  }

  /**
   * Add a listener for timing updates
   */
  addListener(listener: () => void): void {
    this.listeners.add(listener);
  }

  /**
   * Remove a listener
   */
  removeListener(listener: () => void): void {
    this.listeners.delete(listener);
  }

  /**
   * Internal: Merge a completed entry into persistent state
   * Called by SessionImpl.end()
   */
  _mergeEntry(entry: TimingEntry, parentName: string | undefined): void {
    if (!parentName) {
      // This is the root entry
      this.mergeIntoRoot(entry);
    } else {
      // Find parent and merge child
      this.mergeChild(this.rootEntry, entry, parentName);
    }

    // Notify listeners on root entry completion
    if (!parentName) {
      this.activeSession = null;
      this.notifyListeners();
    }
  }

  /**
   * Merge entry into root
   */
  private mergeIntoRoot(entry: TimingEntry): void {
    this.rootEntry.lastMs = entry.lastMs;
    this.rootEntry.count++;

    // Update EMA
    if (this.rootEntry.count === 1) {
      this.rootEntry.avgMs = entry.lastMs;
    } else {
      this.rootEntry.avgMs = EMA_ALPHA * entry.lastMs + (1 - EMA_ALPHA) * this.rootEntry.avgMs;
    }

    this.rootEntry.overBudget = entry.lastMs > FRAME_BUDGET_MS;
    this.rootEntry.metadata = entry.metadata;

    // Merge children
    for (const child of entry.children) {
      this.mergeChildEntry(this.rootEntry.children, child);
    }
  }

  /**
   * Find parent entry and merge child into it
   */
  private mergeChild(current: TimingEntry, child: TimingEntry, parentName: string): boolean {
    if (current.name === parentName) {
      this.mergeChildEntry(current.children, child);
      return true;
    }

    for (const c of current.children) {
      if (this.mergeChild(c, child, parentName)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Merge a child entry into a children array
   */
  private mergeChildEntry(children: TimingEntry[], entry: TimingEntry): void {
    // Find existing entry with same name
    const existing = children.find((c) => c.name === entry.name);

    if (existing) {
      // Update existing entry
      existing.lastMs = entry.lastMs;
      existing.count++;

      // Update EMA
      if (existing.count === 1) {
        existing.avgMs = entry.lastMs;
      } else {
        existing.avgMs = EMA_ALPHA * entry.lastMs + (1 - EMA_ALPHA) * existing.avgMs;
      }

      existing.overBudget = entry.lastMs > FRAME_BUDGET_MS;
      existing.metadata = entry.metadata;

      // Merge children recursively
      for (const child of entry.children) {
        this.mergeChildEntry(existing.children, child);
      }
    } else {
      // Add new entry (first time seeing this path)
      // Deep clone to avoid reference issues
      children.push(this.cloneEntry(entry));
    }
  }

  /**
   * Deep clone a timing entry
   */
  private cloneEntry(entry: TimingEntry): TimingEntry {
    return {
      name: entry.name,
      lastMs: entry.lastMs,
      avgMs: entry.avgMs,
      count: entry.count,
      children: entry.children.map((c) => this.cloneEntry(c)),
      metadata: entry.metadata ? { ...entry.metadata } : undefined,
      overBudget: entry.overBudget,
    };
  }

  /**
   * Notify all listeners of timing update
   */
  private notifyListeners(): void {
    for (const listener of this.listeners) {
      try {
        listener();
      } catch (e) {
        log.error(Modules.PERFORMANCE, 'Listener error', e);
      }
    }
  }
}

/**
 * Format milliseconds for display
 */
export function formatMs(ms: number): string {
  if (ms < 0.1) return '<0.1ms';
  if (ms < 10) return `${ms.toFixed(1)}ms`;
  return `${Math.round(ms)}ms`;
}

/**
 * Check if an entry or any of its children are over budget
 */
export function hasOverBudget(entry: TimingEntry): boolean {
  if (entry.overBudget) return true;
  return entry.children.some((c) => hasOverBudget(c));
}
