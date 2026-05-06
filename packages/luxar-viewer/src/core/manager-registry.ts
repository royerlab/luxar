/**
 * Central lifecycle coordinator for the viewer's singleton managers.
 *
 * The viewer has six+ singletons: SceneLoaderManager,
 * DataMonitorManager, ThemeManager, sceneDimsManager, materialManager,
 * consoleInterceptor, the WorkerPool, etc. Each owns its own
 * `getInstance()` / `disposeInstance()` plumbing, but disposal order
 * is currently order-by-luck — `LuxarApp.dispose()` calls each
 * manager's `dispose` individually, and the order is whatever the
 * code happens to write.
 *
 * `ManagerRegistry` makes the order explicit. Each manager registers
 * itself on first instantiation; `disposeAll()` walks them in the
 * reverse of registration order (LIFO), which mirrors typical
 * teardown semantics: the last thing constructed is the first
 * thing torn down. Idempotent — repeated `disposeAll()` calls are
 * safe.
 *
 * Tests can call `disposeAll()` between cases to start clean. The
 * registry is itself a process-level singleton (lazy, zero-config).
 *
 * @module core/manager-registry
 */

import { log, Modules } from '../utils/log';

/** Minimum surface a registered manager must expose. */
export interface DisposableManager {
  /** Tear down all owned resources. Must be idempotent. */
  dispose(): void;
}

/** Snapshot of registry contents for `__luxarDebug.managers`. */
export interface ManagerStatus {
  name: string;
  disposed: boolean;
}

export class ManagerRegistry {
  private readonly managers = new Map<string, DisposableManager>();
  private readonly disposed = new Set<string>();
  /**
   * Names in registration order. `disposeAll()` walks this in
   * reverse so the last-registered manager tears down first.
   */
  private readonly order: string[] = [];

  /**
   * Register a manager. If a manager with this name is already
   * registered, log a warning and keep the original — duplicate
   * registration is almost always a bug.
   */
  register(name: string, manager: DisposableManager): void {
    if (this.managers.has(name)) {
      log.warning(
        Modules.LUXAR,
        `ManagerRegistry: '${name}' is already registered; skipping duplicate`
      );
      return;
    }
    this.managers.set(name, manager);
    this.order.push(name);
  }

  /**
   * Look up a registered manager. Throws if the manager has been
   * disposed — production code that needs a manager should hold
   * its own reference; the registry is for lifecycle, not lookup.
   */
  get<T extends DisposableManager = DisposableManager>(name: string): T | undefined {
    if (this.disposed.has(name)) {
      throw new Error(`ManagerRegistry: '${name}' has already been disposed`);
    }
    return this.managers.get(name) as T | undefined;
  }

  /** True if a manager with this name is registered and not yet disposed. */
  has(name: string): boolean {
    return this.managers.has(name) && !this.disposed.has(name);
  }

  /**
   * Dispose every registered manager in reverse registration order.
   * Each manager's `dispose()` is wrapped in try/catch so one bad
   * teardown doesn't block the rest. Idempotent across repeat calls.
   */
  disposeAll(): void {
    // Walk in reverse so most-recently-registered tears down first.
    for (let i = this.order.length - 1; i >= 0; i--) {
      const name = this.order[i];
      if (this.disposed.has(name)) continue;
      const manager = this.managers.get(name);
      if (!manager) continue;
      try {
        manager.dispose();
      } catch (err) {
        log.warning(
          Modules.LUXAR,
          `ManagerRegistry: error disposing '${name}': ${err}`
        );
      }
      this.disposed.add(name);
    }
  }

  /** Snapshot for debug interfaces. */
  getStatus(): ManagerStatus[] {
    return this.order.map((name) => ({
      name,
      disposed: this.disposed.has(name),
    }));
  }

  /**
   * Reset the registry entirely (clear maps + disposed set + order).
   * Tests use this to start fresh between cases. Production code
   * should not call this — `disposeAll()` is the right teardown.
   */
  reset(): void {
    this.managers.clear();
    this.disposed.clear();
    this.order.length = 0;
  }
}

// Process-level singleton. Lazy, zero-config.
let _instance: ManagerRegistry | undefined;

export function getManagerRegistry(): ManagerRegistry {
  if (!_instance) _instance = new ManagerRegistry();
  return _instance;
}

/**
 * Drop the singleton (for tests). Production code should call
 * `disposeAll()` instead — it preserves the registry shell so
 * subsequent `register()` calls go to the same instance.
 */
export function __resetManagerRegistryForTests(): void {
  _instance = undefined;
}
