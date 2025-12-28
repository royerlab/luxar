/**
 * LoaderOrchestrator - Manages creation, storage, and lifecycle of data loaders.
 *
 * This module is responsible for:
 * - Creating appropriate loaders for Points, Lines, and GSplats nodes
 * - Storing and managing loader instances
 * - Tracking failed loaders for error recovery
 * - Aggregating accumulator statistics
 * - Disposing loaders on cleanup
 *
 * Extracted from SceneLoader to reduce its complexity.
 *
 * @module data/loader-orchestrator
 */

import * as zarr from 'zarrita';
import { PointSpatialIndexLoader } from './point-spatial-index-loader';
import { LinesSpatialIndexLoader } from './lines-spatial-index-loader';
import { GSplatsSpatialIndexLoader } from './gsplats-spatial-index-loader';
import { DataLoader, SceneNode, LoaderConfig } from './data-loader-types';
import type { LinesDataLoader } from '../types/lines';
import type { GSplatsDataLoader } from '../types/gsplats';
import { ArrayRefRegistry } from './array-decoder';
import { DataMonitorManager } from './data-monitor-manager';
import { log, Modules } from '../utils/log';
import type { UpdateProfiler } from '../profiling/update-profiler';
import type { AccumulatorStats } from './data-accumulator';

/**
 * Information about a failed loader for error tracking.
 */
export interface FailedLoaderInfo {
  /** The error that caused the failure */
  error: Error;
  /** Timestamp when the failure occurred */
  timestamp: number;
  /** Number of retry attempts */
  retryCount: number;
}

/**
 * Configuration for LoaderOrchestrator.
 */
export interface OrchestratorConfig {
  /** Loader configuration passed to individual loaders */
  loaderConfig: LoaderConfig;
  /** Array reference registry for encoding resolution */
  arrayRefRegistry: ArrayRefRegistry;
  /** Zarr store for data access */
  store: zarr.Readable;
  /** Optional profiler for timing */
  profiler?: UpdateProfiler;
  /** Optional monitor ID for connecting loaders */
  monitorId?: string | null;
}

/**
 * Orchestrates the creation and management of data loaders.
 *
 * This class extracts loader management responsibilities from SceneLoader,
 * providing a focused interface for loader lifecycle management.
 *
 * Key features:
 * - Factory methods for creating type-appropriate loaders
 * - Centralized loader storage with path-based lookup
 * - Failed loader tracking for error recovery
 * - Aggregated statistics across all loaders
 *
 * @example
 * ```typescript
 * const orchestrator = new LoaderOrchestrator({
 *   loaderConfig: config,
 *   arrayRefRegistry: registry,
 *   store: zarrStore,
 *   profiler: updateProfiler,
 *   monitorId: 'default-monitor',
 * });
 *
 * // Create and register a points loader
 * const loader = orchestrator.createPointsLoader(node, location);
 * orchestrator.registerPointsLoader(node.path, loader);
 *
 * // Check for failures
 * if (orchestrator.hasFailures()) {
 *   await orchestrator.retryAllFailedLoaders(viewState, geometryManager, rootGroup);
 * }
 * ```
 */
export class LoaderOrchestrator {
  private loaders = new Map<string, DataLoader>();
  private linesLoaders = new Map<string, LinesDataLoader>();
  private gsplatLoaders = new Map<string, GSplatsDataLoader>();
  private failedLoaders = new Map<string, FailedLoaderInfo>();

  private config: OrchestratorConfig;

  constructor(config: OrchestratorConfig) {
    this.config = config;
  }

  // ============================================================================
  // Loader Factory Methods
  // ============================================================================

  /**
   * Create a points loader for a node.
   *
   * Uses PointSpatialIndexLoader which handles both spatial-indexed
   * and non-indexed datasets.
   *
   * @param node - Scene node to create loader for
   * @param loc - Zarr location for the node
   * @returns Configured DataLoader instance
   */
  createPointsLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): DataLoader {
    const nodeLoc =
      node.path === '/' ? loc : zarr.root(this.config.store).resolve(node.path.slice(1));

    log.query(Modules.SCENE_LOADER, `Using PointSpatialIndexLoader for ${node.path}`);

    const loader = new PointSpatialIndexLoader(
      nodeLoc,
      node,
      this.config.loaderConfig,
      this.config.arrayRefRegistry,
      this.config.store,
      this.config.profiler
    );

    // Connect to monitor if available
    if (this.config.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.config.monitorId);
      if (monitor) {
        monitor.connectLoader(node.path, loader);
      }
    }

    return loader;
  }

  /**
   * Create a lines loader for a node.
   *
   * @param node - Scene node to create loader for
   * @param loc - Zarr location for the node
   * @returns Configured LinesDataLoader instance
   */
  createLinesLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): LinesDataLoader {
    const nodeLoc =
      node.path === '/' ? loc : zarr.root(this.config.store).resolve(node.path.slice(1));

    log.query(Modules.SCENE_LOADER, `Using LinesSpatialIndexLoader for ${node.path}`);

    const loader = new LinesSpatialIndexLoader(
      nodeLoc,
      node,
      this.config.arrayRefRegistry,
      this.config.store,
      this.config.profiler
    );

    return loader;
  }

  /**
   * Create a gsplats loader for a node.
   *
   * @param node - Scene node to create loader for
   * @param loc - Zarr location for the node
   * @returns Configured GSplatsDataLoader instance
   */
  createGSplatsLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): GSplatsDataLoader {
    const nodeLoc =
      node.path === '/' ? loc : zarr.root(this.config.store).resolve(node.path.slice(1));

    log.query(Modules.SCENE_LOADER, `Using GSplatsSpatialIndexLoader for ${node.path}`);

    const loader = new GSplatsSpatialIndexLoader(
      nodeLoc,
      node,
      this.config.arrayRefRegistry,
      this.config.store,
      this.config.profiler
    );

    return loader;
  }

  // ============================================================================
  // Loader Registration
  // ============================================================================

  /**
   * Register a points loader for a path.
   */
  registerPointsLoader(path: string, loader: DataLoader): void {
    this.loaders.set(path, loader);
  }

  /**
   * Register a lines loader for a path.
   */
  registerLinesLoader(path: string, loader: LinesDataLoader): void {
    this.linesLoaders.set(path, loader);
  }

  /**
   * Register a gsplats loader for a path.
   */
  registerGSplatsLoader(path: string, loader: GSplatsDataLoader): void {
    this.gsplatLoaders.set(path, loader);
  }

  // ============================================================================
  // Loader Access
  // ============================================================================

  /**
   * Get all registered points loaders.
   */
  getPointsLoaders(): ReadonlyMap<string, DataLoader> {
    return this.loaders;
  }

  /**
   * Get all registered lines loaders.
   */
  getLinesLoaders(): ReadonlyMap<string, LinesDataLoader> {
    return this.linesLoaders;
  }

  /**
   * Get all registered gsplats loaders.
   */
  getGSplatsLoaders(): ReadonlyMap<string, GSplatsDataLoader> {
    return this.gsplatLoaders;
  }

  /**
   * Get a specific points loader by path.
   */
  getPointsLoader(path: string): DataLoader | undefined {
    return this.loaders.get(path);
  }

  /**
   * Get a specific lines loader by path.
   */
  getLinesLoader(path: string): LinesDataLoader | undefined {
    return this.linesLoaders.get(path);
  }

  /**
   * Get a specific gsplats loader by path.
   */
  getGSplatsLoader(path: string): GSplatsDataLoader | undefined {
    return this.gsplatLoaders.get(path);
  }

  // ============================================================================
  // Error Tracking
  // ============================================================================

  /**
   * Record a loader failure.
   *
   * @param path - Path of the failed loader
   * @param error - The error that occurred
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
   * Clear a failure record after successful recovery.
   *
   * @param path - Path of the recovered loader
   */
  clearFailure(path: string): void {
    this.failedLoaders.delete(path);
  }

  /**
   * Get information about failed loaders.
   */
  getFailedLoaders(): ReadonlyMap<string, FailedLoaderInfo> {
    return this.failedLoaders;
  }

  /**
   * Check if there are any failed loaders.
   */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
  }

  /**
   * Clear all failed loader tracking.
   */
  clearAllFailures(): void {
    const count = this.failedLoaders.size;
    this.failedLoaders.clear();
    if (count > 0) {
      log.info(Modules.SCENE_LOADER, `Cleared ${count} failed loader(s) from tracking`);
    }
  }

  /**
   * Determine which type of loader a path belongs to.
   *
   * @param path - The loader path to check
   * @returns The loader type or null if not found
   */
  getLoaderType(path: string): 'points' | 'lines' | 'gsplats' | null {
    if (this.loaders.has(path)) return 'points';
    if (this.linesLoaders.has(path)) return 'lines';
    if (this.gsplatLoaders.has(path)) return 'gsplats';
    return null;
  }

  // ============================================================================
  // Statistics
  // ============================================================================

  /**
   * Get aggregated accumulator stats for all points loaders.
   */
  getAggregatedPointsAccumulatorStats(): AccumulatorStats {
    let totalCapacity = 0;
    let totalAllocations = 0;
    let totalGrowthEvents = 0;
    let totalMemoryMB = 0;

    for (const loader of this.loaders.values()) {
      const stats = (loader as PointSpatialIndexLoader).getAccumulatorStats?.();
      if (stats) {
        totalCapacity += stats.capacity;
        totalAllocations += stats.allocations;
        totalGrowthEvents += stats.growthEvents;
        totalMemoryMB += stats.memoryMB;
      }
    }

    return {
      capacity: totalCapacity,
      allocations: totalAllocations,
      growthEvents: totalGrowthEvents,
      memoryMB: totalMemoryMB,
    };
  }

  /**
   * Get aggregated accumulator stats for all lines loaders.
   */
  getAggregatedLinesAccumulatorStats(): AccumulatorStats {
    let totalCapacity = 0;
    let totalAllocations = 0;
    let totalGrowthEvents = 0;
    let totalMemoryMB = 0;

    for (const loader of this.linesLoaders.values()) {
      const stats = (loader as LinesSpatialIndexLoader).getAccumulatorStats?.();
      if (stats) {
        totalCapacity += stats.capacity;
        totalAllocations += stats.allocations;
        totalGrowthEvents += stats.growthEvents;
        totalMemoryMB += stats.memoryMB;
      }
    }

    return {
      capacity: totalCapacity,
      allocations: totalAllocations,
      growthEvents: totalGrowthEvents,
      memoryMB: totalMemoryMB,
    };
  }

  /**
   * Get aggregated accumulator stats for all gsplats loaders.
   */
  getAggregatedGSplatsAccumulatorStats(): AccumulatorStats {
    let totalCapacity = 0;
    let totalAllocations = 0;
    let totalGrowthEvents = 0;
    let totalMemoryMB = 0;

    for (const loader of this.gsplatLoaders.values()) {
      const stats = (loader as GSplatsSpatialIndexLoader).getAccumulatorStats?.();
      if (stats) {
        totalCapacity += stats.capacity;
        totalAllocations += stats.allocations;
        totalGrowthEvents += stats.growthEvents;
        totalMemoryMB += stats.memoryMB;
      }
    }

    return {
      capacity: totalCapacity,
      allocations: totalAllocations,
      growthEvents: totalGrowthEvents,
      memoryMB: totalMemoryMB,
    };
  }

  // ============================================================================
  // Lifecycle
  // ============================================================================

  /**
   * Dispose all loaders and clear storage.
   */
  dispose(): void {
    // Dispose points loaders
    for (const loader of this.loaders.values()) {
      loader.dispose();
    }
    this.loaders.clear();

    // Dispose lines loaders
    for (const loader of this.linesLoaders.values()) {
      loader.dispose();
    }
    this.linesLoaders.clear();

    // Dispose gsplat loaders
    for (const loader of this.gsplatLoaders.values()) {
      loader.dispose();
    }
    this.gsplatLoaders.clear();

    // Clear failure tracking
    this.failedLoaders.clear();

    log.info(Modules.SCENE_LOADER, 'LoaderOrchestrator disposed');
  }
}
