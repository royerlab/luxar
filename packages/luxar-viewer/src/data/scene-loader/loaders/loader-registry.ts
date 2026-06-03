/**
 * LoaderRegistry - Manages lifecycle of geometry loaders (Points, Lines, GSplats).
 *
 * Extracted from SceneLoader to reduce God Object complexity.
 * Holds the 4 loader Maps and provides registration, disposal, iteration,
 * and error tracking methods.
 *
 * @module data/loader-registry
 */

import type { DataLoader } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import { log, Modules } from '../../../utils/log';

/**
 * Error information tracked for failed loaders.
 */
export interface FailedLoaderInfo {
  error: Error;
  timestamp: number;
  retryCount: number;
}

/**
 * Registry that manages all geometry loaders (Points, Lines, GSplats)
 * and tracks loading failures for retry/recovery.
 */
export class LoaderRegistry {
  /** Points loaders indexed by scene path */
  readonly loaders = new Map<string, DataLoader>();

  /** Lines loaders indexed by scene path */
  readonly linesLoaders = new Map<string, LinesDataLoader>();

  /** GSplats loaders indexed by scene path */
  readonly gsplatLoaders = new Map<string, GSplatsDataLoader>();

  /** Error tracking for failed loaders */
  readonly failedLoaders = new Map<string, FailedLoaderInfo>();

  // ---------------------------------------------------------------------------
  // Registration
  // ---------------------------------------------------------------------------

  /**
   * Register a points loader for a given path.
   */
  registerPointsLoader(path: string, loader: DataLoader): void {
    this.loaders.set(path, loader);
  }

  /**
   * Register a lines loader for a given path.
   */
  registerLinesLoader(path: string, loader: LinesDataLoader): void {
    this.linesLoaders.set(path, loader);
  }

  /**
   * Register a gsplats loader for a given path.
   */
  registerGSplatsLoader(path: string, loader: GSplatsDataLoader): void {
    this.gsplatLoaders.set(path, loader);
  }

  /**
   * Drop a single gsplats loader so it no longer participates in
   * scene-wide ``updateView`` sweeps. Used when a lazily-loaded
   * substitutive LOD level is released back to the buffer pool — leaving
   * it registered would reload its geometry on the next view update,
   * defeating the release. The loader object itself is kept alive by the
   * lod_group's ``ensureLoaded`` closure and re-registered on reload.
   */
  unregisterGSplatsLoader(path: string): void {
    this.gsplatLoaders.delete(path);
  }

  // ---------------------------------------------------------------------------
  // Lookup
  // ---------------------------------------------------------------------------

  /** Total number of loaders across all geometry types. */
  get totalLoaderCount(): number {
    return this.loaders.size + this.linesLoaders.size + this.gsplatLoaders.size;
  }

  /** Whether there are any registered loaders. */
  get hasLoaders(): boolean {
    return this.loaders.size > 0 || this.linesLoaders.size > 0 || this.gsplatLoaders.size > 0;
  }

  /**
   * Find which loader type owns a given path.
   * Returns 'points', 'lines', 'gsplats', or null.
   */
  getLoaderType(path: string): 'points' | 'lines' | 'gsplats' | null {
    if (this.loaders.has(path)) return 'points';
    if (this.linesLoaders.has(path)) return 'lines';
    if (this.gsplatLoaders.has(path)) return 'gsplats';
    return null;
  }

  // ---------------------------------------------------------------------------
  // Error tracking
  // ---------------------------------------------------------------------------

  /**
   * Record a loader failure.
   */
  recordFailure(path: string, error: Error): void {
    const existing = this.failedLoaders.get(path);
    const retryCount = existing ? existing.retryCount + 1 : 0;
    this.failedLoaders.set(path, {
      error,
      timestamp: Date.now(),
      retryCount,
    });
  }

  /**
   * Clear failure tracking for a specific path.
   */
  clearFailure(path: string): void {
    this.failedLoaders.delete(path);
  }

  /**
   * Get information about failed loaders (read-only view).
   */
  getFailedLoaders(): ReadonlyMap<string, FailedLoaderInfo> {
    return this.failedLoaders;
  }

  /** Whether there are any failed loaders. */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
  }

  /**
   * Clear all failure tracking.
   */
  clearAllFailures(): void {
    const count = this.failedLoaders.size;
    this.failedLoaders.clear();
    if (count > 0) {
      log.info(Modules.SCENE_LOADER, `Cleared ${count} failed loader(s) from tracking`);
    }
  }

  // ---------------------------------------------------------------------------
  // Disposal
  // ---------------------------------------------------------------------------

  /**
   * Dispose all loaders and clear all maps.
   */
  disposeAll(): void {
    for (const loader of this.loaders.values()) {
      loader.dispose();
    }
    this.loaders.clear();

    for (const loader of this.linesLoaders.values()) {
      loader.dispose();
    }
    this.linesLoaders.clear();

    for (const loader of this.gsplatLoaders.values()) {
      loader.dispose();
    }
    this.gsplatLoaders.clear();
  }
}
