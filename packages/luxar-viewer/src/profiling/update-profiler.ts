/**
 * Update Profiler - Hierarchical timing for scene updates
 *
 * Provides low-overhead profiling of the scene update pipeline with:
 * - Hierarchical timing breakdown (parent/child relationships)
 * - TWO persistent timing trees: 'Total Update' (per-frame demand updates)
 *   and 'LOD Refinement' (background passes that load LODs 1..N after
 *   first paint, see data/scene-loader/progressive/refinement.ts)
 * - Per-update sequence accounting: multiple sessions with the same name
 *   within ONE update SUM into a single row (a progressive loader opens one
 *   'Load Arrays' per LOD level); rows not touched by the latest update are
 *   flagged `stale` so the UI can grey them out instead of showing a stale
 *   lastMs next to fresh parent rows
 * - Exponential moving average (alpha=0.1), one sample per update
 * - Metadata tracking (chunks, cache hits, point counts) — numeric fields
 *   sum across same-update merges
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
 *
 * // Background refinement pass (its own tree, own pass counter)
 * const pass = profiler.beginPass();
 * const node = pass.begin('GSplats (/path)');
 * // ... load + commit ...
 * node.end();
 * pass.end();
 * ```
 */

import { log, Modules } from '../utils/log';

/** Name of the persistent root for per-frame demand updates. */
export const TOTAL_UPDATE_ROOT = 'Total Update';

/** Name of the persistent root for background LOD-refinement passes. */
export const REFINEMENT_ROOT = 'LOD Refinement';

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

/** Numeric metadata fields that SUM when same-name sessions merge within one update. */
const SUMMED_METADATA_KEYS = [
  'chunks',
  'cacheHits',
  'cacheMisses',
  'points',
  'segments',
  'splats',
] as const;

/**
 * A single timing entry in the hierarchy
 */
export interface TimingEntry {
  /** Name of this timing entry */
  name: string;
  /** Last measured duration in ms (summed across same-update merges) */
  lastMs: number;
  /** Exponential moving average in ms (one sample per update) */
  avgMs: number;
  /** Number of updates this operation ran in */
  count: number;
  /** Child timing entries */
  children: TimingEntry[];
  /** Optional metadata */
  metadata?: TimingMetadata;
  /** Whether this entry exceeds 60fps budget (>16ms) */
  overBudget?: boolean;
  /** Sequence number of the update/pass this entry last recorded in */
  lastSeq?: number;
  /**
   * The avgMs value BEFORE the current update's samples — lets a second
   * same-name session in the same update recompute the EMA against the
   * pre-update base instead of double-applying alpha.
   */
  emaBase?: number;
  /**
   * True when this entry did NOT run in its tree's latest update/pass.
   * The UI greys stale rows and excludes their lastMs from aggregation
   * sums (a stale 94ms child under a fresh 29ms parent is the exact
   * artifact this prevents).
   */
  stale?: boolean;
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
 * Merge metadata from a same-update sibling session: numeric fields sum,
 * `skipped` survives only if BOTH were skipped, string fields last-wins.
 */
function mergeMetadata(
  a: TimingMetadata | undefined,
  b: TimingMetadata | undefined
): TimingMetadata | undefined {
  if (!a) return b;
  if (!b) return a;
  const out: TimingMetadata = { ...a, ...b };
  for (const key of SUMMED_METADATA_KEYS) {
    if (a[key] !== undefined || b[key] !== undefined) {
      out[key] = (a[key] ?? 0) + (b[key] ?? 0);
    }
  }
  if (a.skipped === true && b.skipped === true) {
    out.skipped = true;
  } else {
    delete out.skipped;
    delete out.skipReason;
  }
  return out;
}

/**
 * Internal session implementation
 */
class SessionImpl implements UpdateSession {
  private readonly entry: TimingEntry;
  private readonly startTime: number;
  private ended = false;
  private readonly parent: SessionImpl | null;
  private readonly profiler: UpdateProfiler;
  /** Which persistent root tree this session's subtree merges into. */
  private readonly rootName: string;
  /**
   * Update/pass sequence this session belongs to. Root sessions capture it
   * from the profiler at construction; children INHERIT their parent's seq
   * so a whole session tree always accounts to one update, even if a child
   * is constructed after a newer update began.
   */
  private readonly seq: number;
  // Set by markSkipped() so end() bypasses duration measurement + EMA.
  // Without this flag, skipped entries still record the begin→markSkipped→end
  // overhead because markSkipped() zeroes lastMs/avgMs BEFORE end() runs them.
  private skipBypassesEMA = false;
  // Generation captured at construction. If the profiler's generation
  // advances (because reset() was called) before this session ends, the
  // session is considered "abandoned": its end() becomes a no-op merge,
  // so an in-flight timeTopLevel() that resolves after reset() does NOT
  // pollute the freshly-rebuilt root tree.
  private readonly generation: number;

  constructor(
    name: string,
    parent: SessionImpl | null,
    profiler: UpdateProfiler,
    rootName?: string,
    seq?: number
  ) {
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
    this.rootName = parent ? parent.rootName : (rootName ?? TOTAL_UPDATE_ROOT);
    this.seq = parent ? parent.seq : (seq ?? 0);
    this.generation = profiler._currentGeneration();

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

    // Generation gate: a reset() between begin() and end() bumps the
    // profiler's generation, abandoning every session that was in flight
    // at the time. Their end() walks must NOT merge into the new root
    // tree (or any other freshly-rebuilt state). Drop silently.
    if (this.generation !== this.profiler._currentGeneration()) {
      return;
    }

    if (this.skipBypassesEMA) {
      // markSkipped() already set lastMs/avgMs to 0; do NOT measure duration
      // or apply the EMA, otherwise the begin→markSkipped→end overhead would
      // leak into the persistent state, contradicting the JSDoc claim that
      // skipped entries are zero-duration.
      this.entry.count++;
      this.entry.overBudget = false;
    } else {
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
    }

    // Stamp the owning update so the persistent-tree merge can decide
    // between sum-within-update, rollover, and stale-drop.
    this.entry.lastSeq = this.seq;

    // Merge into profiler's persistent state
    this.profiler._mergeEntry(this.entry, this.parent?.entry.name, this.rootName, this.seq);
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
    // Skipped entries have 0 duration. Flag set so end() bypasses the EMA
    // entirely — otherwise end() would overwrite lastMs/avgMs with the
    // begin→markSkipped→end overhead.
    this.entry.lastMs = 0;
    this.entry.avgMs = 0;
    this.skipBypassesEMA = true;
  }

  getEntry(): TimingEntry {
    return this.entry;
  }
}

/**
 * Root session that tracks the entire update
 */
export class RootSession extends SessionImpl {
  constructor(profiler: UpdateProfiler, seq: number) {
    super(TOTAL_UPDATE_ROOT, null, profiler, TOTAL_UPDATE_ROOT, seq);
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
  // Persistent timing state (survives across updates), one tree per root.
  private roots = new Map<string, TimingEntry>([
    [TOTAL_UPDATE_ROOT, UpdateProfiler.makeRoot(TOTAL_UPDATE_ROOT)],
    [REFINEMENT_ROOT, UpdateProfiler.makeRoot(REFINEMENT_ROOT)],
  ]);

  // Current active session (null when not profiling)
  private activeSession: RootSession | null = null;

  // Track active sessions by ID for concurrent async support
  // Each async operation gets its own context that doesn't interfere with others
  private activeSessions = new Map<number, UpdateSession>();
  private nextSessionId = 0;

  // Track the current session context for nested operations
  // Uses AsyncLocalStorage-like pattern: each sync execution path has its own context
  private currentSessionContext: UpdateSession | null = null;

  // Per-update sequence for the 'Total Update' tree (bumped in beginUpdate)
  // and per-pass sequence for the 'LOD Refinement' tree (bumped in beginPass).
  // Sessions capture their seq at (root) construction; the merge uses it to
  // sum same-update siblings, roll over on a new update, and DROP merges
  // that arrive late from a superseded update.
  private updateSeq = 0;
  private passSeq = 0;

  // Generation counter, bumped on every reset(). Sessions capture the
  // generation at construction; their end() is a no-op if the profiler's
  // generation has advanced past theirs (the session was "abandoned").
  // Internal: only the SessionImpl reads this — exposed via the package-
  // private `_currentGeneration()` accessor below.
  private generation = 0;

  private static makeRoot(name: string): TimingEntry {
    return {
      name,
      lastMs: 0,
      avgMs: 0,
      count: 0,
      children: [],
    };
  }

  /**
   * Internal: current generation counter. Read by `SessionImpl` to gate
   * its `end()` merge against being abandoned by a reset() that landed
   * mid-flight. NOT a public API.
   */
  _currentGeneration(): number {
    return this.generation;
  }

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

    this.updateSeq++;
    this.activeSession = new RootSession(this, this.updateSeq);
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
   * Begin a background LOD-refinement pass. Returns a detached root session
   * that merges into the 'LOD Refinement' persistent tree and does NOT
   * touch the active update session or the ambient context — a refinement
   * pass ending mid-update must never disable the update's own profiling.
   *
   * Callers pass the returned session explicitly (pass.begin('GSplats (/p)'))
   * down the load → process → commit chain, mirroring the main update's
   * per-node top-level sessions.
   */
  beginPass(): UpdateSession {
    this.passSeq++;
    return new SessionImpl(REFINEMENT_ROOT, null, this, REFINEMENT_ROOT, this.passSeq);
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
    const prev = this.currentSessionContext;
    const session = this.begin(name);
    // Set this session as the current context so nested time()/begin() calls
    // become children of it (the context/ambient pattern advertised in the
    // class docstring).
    this.currentSessionContext = session;
    try {
      const result = fn();
      // Handle promises
      if (result instanceof Promise) {
        // Restore context immediately so subsequent synchronous code at the
        // caller's level does not see this session as the active parent.
        this.currentSessionContext = prev;
        return result.finally(() => session.end()) as T;
      }
      session.end();
      this.currentSessionContext = prev;
      return result;
    } catch (e) {
      session.end();
      this.currentSessionContext = prev;
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
   * Get the 'Total Update' timing hierarchy (for UI display)
   */
  getTimings(): TimingEntry {
    return this.roots.get(TOTAL_UPDATE_ROOT)!;
  }

  /**
   * Get the 'LOD Refinement' timing hierarchy (background passes).
   * `count` on this root is the number of refinement passes recorded.
   */
  getRefinementTimings(): TimingEntry {
    return this.roots.get(REFINEMENT_ROOT)!;
  }

  /**
   * Reset all timing data
   *
   * Clears active-session state too: if reset() is called mid-update, any
   * stale RootSession / currentSessionContext / per-id sessions are dropped
   * so that subsequent endUpdate() / timeTopLevel() calls don't try to merge
   * into the freshly rebuilt rootEntry under a name they no longer own.
   */
  reset(): void {
    // Bump generation FIRST so any session whose end() runs *during*
    // notifyListeners() (synchronous listener callbacks could trigger
    // it) sees the new generation and bails out.
    this.generation++;
    this.activeSession = null;
    this.currentSessionContext = null;
    this.activeSessions.clear();
    this.updateSeq = 0;
    this.passSeq = 0;
    this.roots = new Map<string, TimingEntry>([
      [TOTAL_UPDATE_ROOT, UpdateProfiler.makeRoot(TOTAL_UPDATE_ROOT)],
      [REFINEMENT_ROOT, UpdateProfiler.makeRoot(REFINEMENT_ROOT)],
    ]);
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
  _mergeEntry(
    entry: TimingEntry,
    parentName: string | undefined,
    rootName: string,
    seq: number
  ): void {
    const root = this.roots.get(rootName);
    if (!root) return;

    if (!parentName) {
      // This is a root entry (update root or refinement pass root)
      this.mergeEntryValues(root, entry, seq);
      // Sweep: anything this update/pass did NOT touch is now stale. The
      // root itself just merged with `seq`, so it stays fresh.
      this.markStaleTree(root, seq);

      // Only the ACTIVE UPDATE root may clear the active session — a
      // refinement pass root ending while a demand update is in flight
      // must not disable that update's profiling (beginTopLevel would
      // start returning NOOP sessions).
      if (rootName === TOTAL_UPDATE_ROOT) {
        this.activeSession = null;
      }
      this.notifyListeners();
    } else {
      // Find parent within THIS root's tree and merge child
      this.mergeChild(root, entry, parentName, seq);
    }
  }

  /**
   * Merge a completed session entry's values into a persistent entry.
   *
   * - seq OLDER than the persistent entry's → drop (a late merge from a
   *   superseded update must not overwrite newer data)
   * - seq EQUAL → same update: SUM lastMs, sum numeric metadata, recompute
   *   the EMA against the pre-update base (one EMA sample per update)
   * - seq NEWER → rollover: snapshot avg as emaBase, start a fresh lastMs,
   *   count++ (count = number of updates the op ran in)
   */
  private mergeEntryValues(existing: TimingEntry, entry: TimingEntry, seq: number): void {
    if (existing.lastSeq !== undefined && seq < existing.lastSeq) {
      return;
    }

    if (existing.lastSeq === seq) {
      existing.lastMs += entry.lastMs;
      existing.metadata = mergeMetadata(existing.metadata, entry.metadata);
    } else {
      existing.emaBase = existing.count > 0 ? existing.avgMs : undefined;
      existing.lastMs = entry.lastMs;
      existing.count++;
      existing.lastSeq = seq;
      existing.stale = false;
      existing.metadata = entry.metadata;
    }

    if (existing.count === 1) {
      existing.avgMs = existing.lastMs;
    } else {
      existing.avgMs =
        EMA_ALPHA * existing.lastMs + (1 - EMA_ALPHA) * (existing.emaBase ?? existing.avgMs);
    }

    existing.overBudget = existing.metadata?.skipped !== true && existing.lastMs > FRAME_BUDGET_MS;
  }

  /**
   * Flag every entry the update/pass `seq` did not touch as stale.
   */
  private markStaleTree(entry: TimingEntry, seq: number): void {
    if (entry.lastSeq !== undefined && entry.lastSeq < seq) {
      entry.stale = true;
    }
    for (const child of entry.children) {
      this.markStaleTree(child, seq);
    }
  }

  /**
   * Find parent entry and merge child into it
   */
  private mergeChild(
    current: TimingEntry,
    child: TimingEntry,
    parentName: string,
    seq: number
  ): boolean {
    if (current.name === parentName) {
      this.mergeChildEntry(current.children, child, seq);
      return true;
    }

    for (const c of current.children) {
      if (this.mergeChild(c, child, parentName, seq)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Merge a child entry into a children array
   */
  private mergeChildEntry(children: TimingEntry[], entry: TimingEntry, seq: number): void {
    // Find existing entry with same name
    const existing = children.find((c) => c.name === entry.name);

    if (existing) {
      this.mergeEntryValues(existing, entry, seq);

      // Note: children are NOT re-merged here. Each child session merges itself
      // via its own SessionImpl.end() → _mergeEntry path (which walks the
      // persistent tree to find its parent). Re-merging here would
      // double-increment count and double-apply the EMA for every descendant.
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
      lastSeq: entry.lastSeq,
      emaBase: entry.emaBase,
      stale: entry.stale,
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
 * Check if an entry or any of its children are over budget.
 * Stale entries (not touched by the latest update) are ignored — a stale
 * over-budget child must not paint a fresh parent red.
 */
export function hasOverBudget(entry: TimingEntry): boolean {
  if (entry.stale) return false;
  if (entry.overBudget) return true;
  return entry.children.some((c) => hasOverBudget(c));
}
