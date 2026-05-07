/**
 * SceneLoaderManager - Manages SceneLoader instances without global state
 *
 * This manager provides a clean way to access SceneLoader instances without
 * polluting the global window object. It uses a singleton pattern with
 * explicit instance management.
 *
 * Also owns the UpdateProfiler singleton for timing scene updates.
 */

import { SceneLoader } from './scene-loader';
import { LoaderConfig } from './data-loader-types';
import { UpdateProfiler } from '../profiling/update-profiler';
import type { SceneLoaderMonitorFactory } from './scene-loader-monitor-port';

/**
 * Manager for SceneLoader instances.
 * Provides centralized access to loader instances without global variables.
 */
export class SceneLoaderManager {
  private static instance: SceneLoaderManager | null = null;
  private loaders = new Map<string, SceneLoader>();
  private defaultLoaderId: string | null = null;

  /**
   * Update profiler for timing scene updates
   * Singleton owned by the manager, shared with all loaders
   */
  private readonly profiler: UpdateProfiler;

  /**
   * Optional monitor factory injected by `core/app.ts` so each
   * `SceneLoader` we create can resolve a UI monitor without the
   * `data/` layer importing `ui/`. Null means "no UI monitor wired
   * up" (tests / embedders) and SceneLoader treats every monitor
   * call as a no-op.
   */
  private monitorFactory: SceneLoaderMonitorFactory | null = null;

  /**
   * Private constructor to enforce singleton pattern
   */
  private constructor() {
    this.profiler = new UpdateProfiler();
  }

  /**
   * Provide the monitor factory. Called once at app boot from
   * `core/app.ts` (which holds the `DataMonitorManager` reference).
   * Subsequent `createLoader` calls forward the factory to each
   * `SceneLoader` instance.
   */
  setMonitorFactory(factory: SceneLoaderMonitorFactory | null): void {
    this.monitorFactory = factory;
  }

  /**
   * Get the update profiler instance
   * Used by DataLoadingMonitor to display timing data
   */
  getProfiler(): UpdateProfiler {
    return this.profiler;
  }

  /**
   * Get the singleton instance of SceneLoaderManager
   */
  static getInstance(): SceneLoaderManager {
    if (!SceneLoaderManager.instance) {
      SceneLoaderManager.instance = new SceneLoaderManager();
    }
    return SceneLoaderManager.instance;
  }

  /**
   * Create a new SceneLoader instance
   *
   * @param id - Unique identifier for this loader
   * @param config - Optional loader configuration
   * @param setAsDefault - Whether to set this as the default loader
   * @returns The created SceneLoader instance
   */
  createLoader(
    id: string = 'default',
    config?: LoaderConfig,
    setAsDefault: boolean = true
  ): SceneLoader {
    // Dispose existing loader with same ID if it exists
    if (this.loaders.has(id)) {
      this.destroyLoader(id);
    }

    // Pass the profiler + monitor factory to the loader. The factory
    // is the dependency-inversion handle that lets `SceneLoader` reach
    // the UI monitor without importing `ui/` directly.
    const loader = new SceneLoader(config, id, this.profiler, this.monitorFactory);
    this.loaders.set(id, loader);

    if (setAsDefault || !this.defaultLoaderId) {
      this.defaultLoaderId = id;
    }

    return loader;
  }

  /**
   * Get a SceneLoader by ID
   *
   * @param id - The loader ID
   * @returns The SceneLoader instance or null if not found
   */
  getLoader(id: string): SceneLoader | null {
    return this.loaders.get(id) || null;
  }

  /**
   * Get the default SceneLoader
   *
   * @returns The default SceneLoader instance or null
   */
  getDefaultLoader(): SceneLoader | null {
    if (!this.defaultLoaderId) {
      return null;
    }
    return this.loaders.get(this.defaultLoaderId) || null;
  }

  /**
   * Get all active loaders
   *
   * @returns Map of all active loaders
   */
  getAllLoaders(): Map<string, SceneLoader> {
    return new Map(this.loaders);
  }

  /**
   * Destroy a specific loader
   *
   * @param id - The loader ID to destroy
   */
  destroyLoader(id: string): void {
    const loader = this.loaders.get(id);
    if (loader) {
      loader.dispose();
      this.loaders.delete(id);

      // Update default if needed
      if (this.defaultLoaderId === id) {
        this.defaultLoaderId =
          this.loaders.size > 0 ? (this.loaders.keys().next().value ?? null) : null;
      }
    }
  }

  /**
   * Destroy all loaders and reset the manager
   */
  destroyAll(): void {
    for (const loader of this.loaders.values()) {
      loader.dispose();
    }
    this.loaders.clear();
    this.defaultLoaderId = null;
  }

  /**
   * Check if a loader exists
   *
   * @param id - The loader ID to check
   * @returns True if the loader exists
   */
  hasLoader(id: string): boolean {
    return this.loaders.has(id);
  }

  /**
   * Get the number of active loaders
   *
   * @returns The number of active loaders
   */
  getLoaderCount(): number {
    return this.loaders.size;
  }

  /**
   * Dispose the current instance and clear the singleton slot.
   *
   * Used at app shutdown and between tests. The next `getInstance()` call
   * lazily constructs a fresh manager.
   */
  static disposeInstance(): void {
    if (SceneLoaderManager.instance) {
      SceneLoaderManager.instance.destroyAll();
      SceneLoaderManager.instance = null;
    }
  }
}

// Export a convenient accessor function
export function getSceneLoader(id?: string): SceneLoader | null {
  const manager = SceneLoaderManager.getInstance();
  return id ? manager.getLoader(id) : manager.getDefaultLoader();
}
