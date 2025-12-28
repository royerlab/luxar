/**
 * Unified scene loader that orchestrates the loading of complete Luxar scenes.
 *
 * This loader handles the entire scene graph, using spatial index-based
 * loading for all points nodes and managing the THREE.js scene construction.
 */

import * as zarr from 'zarrita';
import type { Readable } from '@zarrita/storage';
import * as THREE from 'three';
import { PointSpatialIndexLoader } from './point-spatial-index-loader';
import { LinesSpatialIndexLoader, buildInstanceBuffers } from './lines-spatial-index-loader';
import { computeLinesTolerance } from './lines-chunk-spatial-index';
import { DataLoader, ViewState, SceneNode, LoaderConfig, PointsData } from './data-loader-types';
import type { SceneGraphNode } from '../ui/data-monitor-types';
import { ZarrSceneAttrs, ZarrNodeAttrs, hasContentsMethod } from '../types/zarr';
import { materialManager, BlendingMode } from '../rendering/material-manager';
import { createInstancedLinesMesh, LineMaterial } from '../rendering/line-material';
import { DataMonitorManager } from './data-monitor-manager';
import { ArrayRefRegistry } from './array-decoder';
import { ViewStateManager, type SceneDimensions } from './view-state-manager';
import { log, Modules, LogEmoji } from '../utils/log';
import { config as appConfig } from '../config';
import { TwoLevelCachingStore, ChunkPrefetcher } from '../cache';
import type {
  LinesMetadata,
  LinesDataLoader,
  LinesUserData,
  LoadedLinesData,
} from '../types/lines';
import { isLinesUserData } from '../types/lines';
import type {
  GSplatsMetadata,
  GSplatsDataLoader,
  GSplatsUserData,
  GSplatsViewState,
  LoadedGSplatsData,
} from '../types/gsplats';
import { GSplatsSpatialIndexLoader } from './gsplats-spatial-index-loader';
import { processGSplats } from './gsplats-processor';
import {
  createInstancedGSplatsMesh,
  updateInstancedGSplatsMesh,
  packCholeskyForShader,
} from '../rendering/gsplat-material';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { UpdateProfiler } from '../profiling/update-profiler';

/**
 * Main scene loader that handles the complete loading pipeline.
 *
 * Features:
 * - Spatial index-based loading for efficient nD queries
 * - Hierarchical scene graph construction
 * - Transform and rendering attribute inheritance
 * - Dimension metadata management
 * - Memory-efficient loading with proper caching
 */
export class SceneLoader {
  private store: any | null = null;
  private cachingStore: TwoLevelCachingStore | null = null;
  private loaders = new Map<string, DataLoader>();
  private linesLoaders = new Map<string, LinesDataLoader>();
  private gsplatLoaders = new Map<string, GSplatsDataLoader>();
  private viewState: ViewState;
  private config: LoaderConfig;
  private rootGroup: THREE.Group | null = null;
  private monitorId: string | null = null;
  private arrayRefRegistry: ArrayRefRegistry;

  // Performance profiling for hierarchical timing display
  private profiler: UpdateProfiler | null = null;

  // Phase 4: GPU buffer pool for geometry reuse ✅ INTEGRATED
  // Integrated into updatePointsGeometry/updateLinesGeometry/updateGSplatsGeometry
  // Enabled via config.dataLoading.performance.useGPUBufferPool
  private _gpuBufferPool: GPUBufferPool | null = null;

  // Error recovery tracking
  private failedLoaders = new Map<
    string,
    { error: Error; timestamp: number; retryCount: number }
  >();

  constructor(config: LoaderConfig = {}, id?: string) {
    this.config = config;
    this.viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [],
      tolerance: [],
    };
    this.arrayRefRegistry = new ArrayRefRegistry();

    // Phase 4: Initialize GPU buffer pool if enabled
    // Fully integrated into updatePointsGeometry/updateLinesGeometry/updateGSplatsGeometry
    // Note: Requires Float32Array data; falls back to standard path for Uint8/Uint16
    if (appConfig.dataLoading.performance.useGPUBufferPool) {
      this._gpuBufferPool = new GPUBufferPool(
        appConfig.dataLoading.performance.gpuPoolMaxSize,
        appConfig.dataLoading.performance.gpuPoolEvictionFrames
      );
      log.info(
        Modules.GPU_BUFFER_POOL,
        `GPU buffer pool enabled (max size: ${appConfig.dataLoading.performance.gpuPoolMaxSize}, ` +
          `eviction: ${appConfig.dataLoading.performance.gpuPoolEvictionFrames} frames)`
      );
    }

    // Initialize profiler for performance timing
    if (appConfig.dataLoading.performance.enablePerformanceMonitoring) {
      this.profiler = new UpdateProfiler();
      log.info(Modules.SCENE_LOADER, 'Performance profiler initialized');
    }

    // Use the DataMonitorManager to get or create a monitor
    if (typeof document !== 'undefined' && config.enableMonitor !== false) {
      const monitorManager = DataMonitorManager.getInstance();
      const monitorId = id ? `${id}-monitor` : 'default';

      // Only create if it doesn't exist
      if (!monitorManager.hasMonitor(monitorId)) {
        monitorManager.createMonitor(monitorId, document.body);
      }
      this.monitorId = monitorId;

      // Connect profiler to monitor for Performance tab
      if (this.profiler) {
        const monitor = monitorManager.getMonitor(monitorId);
        if (monitor) {
          monitor.setProfiler(this.profiler);
          log.info(Modules.SCENE_LOADER, 'Profiler connected to monitor');
        }
      }
    }
  }

  /**
   * Load a complete scene from a Zarr store using chunk-based spatial indexing.
   *
   * Orchestrates the loading of hierarchical scene graphs, managing spatial indices,
   * attribute inheritance, and dimension metadata. Supports both points and lines
   * with automatic fallback for datasets without spatial ordering.
   *
   * The loading process:
   * 1. Opens Zarr store with optional two-level caching (L1 memory + L2 OPFS)
   * 2. Loads scene metadata and initializes dimensions
   * 3. Recursively constructs THREE.js scene graph from Zarr group hierarchy
   * 4. Creates spatial index loaders for efficient nD queries
   * 5. Connects loaders to data monitor for debugging
   *
   * @param url - Complete URL to the Zarr store. Can be:
   *              - HTTP URL: 'https://example.com/data.zarr'
   *              - Local path: '/path/to/data.zarr'
   *              - With query params: 'https://example.com/data.zarr?no-cache'
   *
   * @returns Promise resolving to a THREE.Group containing the complete scene graph.
   *          The group's userData contains:
   *          - sceneDimensions: Dimension metadata if available
   *          - bounds: AABB of all points
   *          - nodeCount: Total number of leaf nodes
   *
   * @throws {Error} If the Zarr store cannot be opened or is invalid
   * @throws {Error} If consolidated metadata (.zmetadata) is malformed
   * @throws {Error} If required arrays (positions) are missing from point nodes
   *
   * @example
   * ```typescript
   * // Load a scene from HTTP URL
   * const scene = await sceneLoader.loadScene('https://example.com/data.zarr');
   * threeScene.add(scene);
   * console.log(`Loaded ${scene.children.length} top-level nodes`);
   * ```
   *
   * @example
   * ```typescript
   * // Load with error handling
   * try {
   *   const scene = await sceneLoader.loadScene(url);
   *   if (scene.children.length === 0) {
   *     console.warn('Scene is empty');
   *   }
   * } catch (error) {
   *   console.error('Failed to load scene:', error);
   *   // Fallback to default visualization
   * }
   * ```
   *
   * @example
   * ```typescript
   * // Access scene metadata after loading
   * const scene = await sceneLoader.loadScene(url);
   * const dims = scene.userData.sceneDimensions;
   * if (dims) {
   *   console.log(`${dims.length}D dataset:`, dims.map(d => d.name).join(', '));
   * }
   * ```
   *
   * @see {@link ../cache/two-level-caching-store.ts} for caching implementation
   * @see {@link SPECIFICATIONS.md} Section 4 for complete scene loading protocol
   */
  async loadScene(url: string): Promise<THREE.Group> {
    log.custom(LogEmoji.SCENE, Modules.SCENE_LOADER, `Loading scene from ${url}`);

    // Clear any existing loaders from monitor before loading new scene
    if (this.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
      if (monitor) {
        monitor.disconnectAllLoaders();
      }
    }

    // Dispose of any existing loaders
    if (this.loaders.size > 0) {
      this.dispose();
    }

    // Open zarr store with caching
    let rawStore: Readable;
    if (appConfig.cache.enabled) {
      const cachingStore = new TwoLevelCachingStore(this.normalizeURL(url), {
        l1MaxSize: appConfig.cache.l1MaxSizeMB * 1024 * 1024,
        l2MaxSize: appConfig.cache.l2MaxSizeMB * 1024 * 1024,
        debug: appConfig.cache.debug,
      });
      await cachingStore.init();

      // Attach prefetcher to enable transparent adjacent chunk prefetching
      // (Respects ?no-prefetch URL parameter for debugging)
      const prefetcher = new ChunkPrefetcher(cachingStore, {
        maxConcurrent: 4,
        enabled: true,
      });
      cachingStore.setPrefetcher(prefetcher);

      rawStore = cachingStore;
      this.cachingStore = cachingStore;
    } else {
      rawStore = new zarr.FetchStore(this.normalizeURL(url));
    }
    this.store = await zarr.tryWithConsolidated(rawStore);

    // Create root THREE.js group
    this.rootGroup = new THREE.Group();
    this.rootGroup.name = 'LuxarScene';

    // Load scene metadata
    const rootLoc = zarr.root(this.store);
    const rootZarrGroup = await zarr.open(rootLoc, { kind: 'group' });
    const sceneAttrs = rootZarrGroup.attrs as ZarrSceneAttrs;

    // Initialize scene dimensions - CRITICAL for extend_to_all feature
    if (sceneAttrs?.scene_dimensions) {
      this.initializeSceneDimensions(sceneAttrs.scene_dimensions);
      this.rootGroup.userData.sceneDimensions = sceneAttrs.scene_dimensions;

      // Log dimension initialization status for debugging
      if (this.viewState.dimensions?.metadata) {
        log.success(
          Modules.SCENE_LOADER,
          `Scene dimensions initialized: ${this.viewState.dimensions.metadata.length} dimensions, ` +
            `displayed=[${this.viewState.displayDims.join(', ')}]`
        );
      }
    } else {
      log.warning(
        Modules.SCENE_LOADER,
        'No scene_dimensions found in scene metadata. extend_to_all features will not work.'
      );
    }

    // Store scene-level position bounds (from Python compiler)
    // These bounds represent the full dataset extent, available immediately without loading points
    if (sceneAttrs?.position_bounds) {
      this.rootGroup.userData.positionBounds = sceneAttrs.position_bounds;
      log.info(
        Modules.SCENE_LOADER,
        `Scene bounds loaded: min=[${sceneAttrs.position_bounds.min.join(', ')}], ` +
          `max=[${sceneAttrs.position_bounds.max.join(', ')}]`
      );
    }

    // Build scene graph
    const sceneGraph = await this.buildSceneGraph(rootLoc, sceneAttrs);

    // Load points
    await this.loadSceneNodes(sceneGraph, this.rootGroup, rootLoc);

    // Force update the monitor UI after all loaders are connected
    // This ensures the UI shows the correct state even if no events have fired yet
    if (this.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
      if (monitor) {
        // Connect cache stats provider for L1/L2 cache monitoring
        if (this.cachingStore) {
          monitor.setCacheStatsProvider(this.cachingStore);
        }

        // Send scene graph to monitor for display
        const sceneGraphRoot = this.convertToSceneGraphNode(sceneGraph);
        monitor.setSceneGraph(sceneGraphRoot);

        // Update visible segments count (initial load)
        this.updateVisibleCountsInMonitor();

        monitor.forceUpdate();
      }
    }

    log.success(Modules.SCENE_LOADER, 'Scene loaded successfully');
    return this.rootGroup;
  }

  /**
   * Update all points and lines for a new view state
   */
  async updateView(viewState: Partial<ViewState>): Promise<void> {
    this.viewState = { ...this.viewState, ...viewState };

    const totalLoaders = this.loaders.size + this.linesLoaders.size + this.gsplatLoaders.size;
    log.update(Modules.SCENE_LOADER, `Updating view for ${totalLoaders} loaders`);

    // Begin profiling cycle
    this.profiler?.beginUpdate();

    try {
      // Update points loaders
      const pointsUpdates = Array.from(this.loaders.entries()).map(async ([path, loader]) => {
        try {
          if (this.profiler) {
            await this.profiler.timeTopLevel(`Points (${path})`, async (session) => {
              const points = await loader.updateView(this.viewState);
              if (points) {
                this.updatePointsGeometry(path, points);
                session.setMetadata({ points: points.metadata.loadedPoints });
              }
            });
          } else {
            const points = await loader.updateView(this.viewState);
            if (points) {
              this.updatePointsGeometry(path, points);
            }
          }
          this.failedLoaders.delete(path);
        } catch (error) {
          const errorInfo = this.failedLoaders.get(path);
          const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
          this.failedLoaders.set(path, {
            error: error as Error,
            timestamp: Date.now(),
            retryCount,
          });
          log.error(
            Modules.SCENE_LOADER,
            `Failed to update ${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
          );
        }
      });

      // Update lines loaders
      const linesUpdates = Array.from(this.linesLoaders.entries()).map(async ([path, loader]) => {
        try {
          const updateFn = async (session: any) => {
            // Get mesh to check extend_to_all attribute
            const mesh = this.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
            const attrs = mesh?.userData?.attrs as { extend_to_all?: string[] } | undefined;
            const extendDims: string[] = attrs?.extend_to_all || [];

            // Check if we can skip this update (extend_to_all optimization)
            if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
              const dims = this.viewState.dimensions.metadata;
              const nonDisplayedDims = dims
                .filter(
                  (_: { name?: string }, idx: number) => !this.viewState.displayDims.includes(idx)
                )
                .map((d: { name?: string }) => d.name)
                .filter((name: string | undefined): name is string => !!name);

              const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
                extendDims.includes(dimName)
              );

              if (isFullyExtended) {
                // All non-displayed dimensions are extended - geometry is unchanged
                log.info(
                  Modules.SCENE_LOADER,
                  `Skipping update for ${path} - all non-displayed dims are extended`
                );
                session.markSkipped('extend_to_all');
                return;
              }
            }

            const linesViewState = {
              displayDims: this.viewState.displayDims,
              slicePosition: this.viewState.slicePosition,
              tolerance: this.viewState.tolerance,
              dimensions: this.viewState.dimensions?.metadata,
            };

            const data = await loader.updateView(linesViewState);
            if (data) {
              this.updateLinesGeometry(path, data, linesViewState);
              session.setMetadata({ segments: data.segments ? data.segments.length / 2 : 0 });
            }
          };

          if (this.profiler) {
            await this.profiler.timeTopLevel(`Lines (${path})`, updateFn);
          } else {
            await updateFn({ markSkipped: () => {}, setMetadata: () => {} });
          }
          this.failedLoaders.delete(path);
      } catch (error) {
        const errorInfo = this.failedLoaders.get(path);
        const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
        this.failedLoaders.set(path, {
          error: error as Error,
          timestamp: Date.now(),
          retryCount,
        });
        log.error(
          Modules.SCENE_LOADER,
          `Failed to update lines ${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
        );
      }
    });

      // Update gsplat loaders
      const gsplatsUpdates = Array.from(this.gsplatLoaders.entries()).map(async ([path, loader]) => {
        try {
          const updateFn = async (session: any) => {
            // Get mesh to check extend_to_all attribute
            const mesh = this.rootGroup?.getObjectByName(path) as THREE.Mesh | undefined;
            const attrs = mesh?.userData?.attrs as GSplatsMetadata | undefined;
            const extendDims: string[] = attrs?.extend_to_all || [];

            // Check if we can skip this update (extend_to_all optimization)
            if (extendDims.length > 0 && this.viewState.dimensions?.metadata) {
              const dims = this.viewState.dimensions.metadata;
              const nonDisplayedDims = dims
                .filter(
                  (_: { name?: string }, idx: number) => !this.viewState.displayDims.includes(idx)
                )
                .map((d: { name?: string }) => d.name)
                .filter((name: string | undefined): name is string => !!name);

              const isFullyExtended = nonDisplayedDims.every((dimName: string) =>
                extendDims.includes(dimName)
              );

              if (isFullyExtended) {
                // All non-displayed dimensions are extended - geometry is unchanged
                log.info(
                  Modules.SCENE_LOADER,
                  `Skipping gsplats update for ${path} - all non-displayed dims are extended`
                );
                session.markSkipped('extend_to_all');
                return;
              }
            }

            const gsplatsViewState: GSplatsViewState = {
              displayDims: this.viewState.displayDims,
              slicePosition: this.viewState.slicePosition,
              tolerance: this.viewState.tolerance,
              dimensions: this.viewState.dimensions?.metadata,
            };

            const data = await loader.updateView(gsplatsViewState);
            if (data) {
              this.updateGSplatsGeometry(path, data, gsplatsViewState);
              session.setMetadata({ splats: data.splatCount });
            }
          };

          if (this.profiler) {
            await this.profiler.timeTopLevel(`GSplats (${path})`, updateFn);
          } else {
            await updateFn({ markSkipped: () => {}, setMetadata: () => {} });
          }
          this.failedLoaders.delete(path);
      } catch (error) {
        const errorInfo = this.failedLoaders.get(path);
        const retryCount = errorInfo ? errorInfo.retryCount + 1 : 0;
        this.failedLoaders.set(path, {
          error: error as Error,
          timestamp: Date.now(),
          retryCount,
        });
        log.error(
          Modules.SCENE_LOADER,
          `Failed to update gsplats ${path} (attempt ${retryCount + 1}): ${(error as Error).message}`
        );
      }
    });

      await Promise.all([...pointsUpdates, ...linesUpdates, ...gsplatsUpdates]);

      // Update monitor with total visible segments across all lines nodes
      this.updateVisibleCountsInMonitor();

      // Warn user if any loaders failed
      if (this.failedLoaders.size > 0) {
        const failedPaths = Array.from(this.failedLoaders.keys()).join(', ');
        log.warning(
          Modules.SCENE_LOADER,
          `⚠️ ${this.failedLoaders.size} loader(s) failed: ${failedPaths}`
        );
        console.warn(
          `[SceneLoader] Some data could not be loaded. Failed loaders: ${failedPaths}. ` +
            'Check browser console for details. Data may be incomplete.'
        );
      }
    } finally {
      // End profiling cycle (always, even if errors)
      this.profiler?.endUpdate();
    }
  }

  /**
   * Aggregate visible counts from all lines and gsplats meshes and update monitor.
   * This should be called after view updates to report accurate visible counts.
   */
  private updateVisibleCountsInMonitor(): void {
    if (!this.rootGroup || !this.monitorId) return;

    let totalVisibleSegments = 0;
    let totalVisibleSplats = 0;

    // Traverse all objects in the scene graph
    this.rootGroup.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        if (isLinesUserData(object.userData)) {
          totalVisibleSegments += object.userData.visibleSegmentCount ?? 0;
        } else if (object.userData?.nodeType === 'gsplats') {
          totalVisibleSplats += (object.userData as GSplatsUserData).visibleSplatCount ?? 0;
        }
      }
    });

    // Update the monitor
    const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
    if (monitor) {
      monitor.updateVisibleSegments(totalVisibleSegments);
      monitor.updateVisibleSplats(totalVisibleSplats);
    }
  }

  /**
   * Update lines geometry for a specific path
   */
  private updateLinesGeometry(
    path: string,
    data: LoadedLinesData,
    viewState: { displayDims: number[]; slicePosition: number[]; dimensions?: any[] }
  ): void {
    if (!this.rootGroup) return;

    const mesh = this.rootGroup.getObjectByName(path) as THREE.Mesh;
    if (!mesh || !isLinesUserData(mesh.userData)) return;

    // Build new instance buffers
    const ndim = data.ndim;
    let tolerance = viewState.dimensions
      ? computeLinesTolerance(viewState.dimensions, viewState.displayDims)
      : new Array(ndim).fill(0).map((_, i) => (viewState.displayDims.includes(i) ? 1e10 : 0));

    // CRITICAL: For extend_to_all dimensions, set tolerance to infinity
    const attrs = mesh.userData.attrs as { extend_to_all?: string[] };
    const extendDims: string[] = attrs.extend_to_all || [];
    if (extendDims.length > 0 && viewState.dimensions) {
      tolerance = [...tolerance]; // Make a copy to avoid mutating shared array
      for (const dimName of extendDims) {
        const dimIndex = viewState.dimensions.findIndex(
          (d: { name?: string }) => d.name === dimName
        );
        if (dimIndex >= 0 && dimIndex < tolerance.length) {
          tolerance[dimIndex] = 1e10; // Effectively infinite tolerance
        }
      }
    }

    const processed = buildInstanceBuffers(
      data,
      viewState.slicePosition,
      tolerance,
      viewState.displayDims
    );

    // Phase 4: Use GPU buffer pool if enabled
    if (this._gpuBufferPool) {
      // Acquire geometry from pool (may reuse existing)
      const geometry = this._gpuBufferPool.acquireLinesGeometry(path, processed.segmentCount);

      // Update attributes in place (zero GPU allocations on reuse)
      this._gpuBufferPool.updateLinesGeometry(geometry, processed, processed.segmentCount);

      // Assign to mesh (might be same geometry, reused)
      mesh.geometry = geometry;
      mesh.count = processed.segmentCount;
    } else {
      // Fallback: Original path (dispose + create new)
      const oldGeometry = mesh.geometry;
      if (oldGeometry) {
        oldGeometry.dispose();
      }

      // Create new geometry with updated data
      const newMesh = createInstancedLinesMesh(processed, mesh.material as LineMaterial);

      // Copy geometry to existing mesh
      mesh.geometry = newMesh.geometry;
      mesh.count = processed.segmentCount;

      // Clean up temporary mesh (but not its geometry, which is now on the original mesh)
      newMesh.geometry = new THREE.BufferGeometry(); // Replace to avoid double disposal
      newMesh.geometry.dispose();
    }

    // Track visible segment count in mesh userData for monitor reporting
    if (isLinesUserData(mesh.userData)) {
      mesh.userData.visibleSegmentCount = processed.segmentCount;
    }

    if (data.segmentCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing lines for ${path} (no visible segments at current slice)`
      );
    }
  }

  /**
   * Update gsplats geometry for a specific path
   */
  private updateGSplatsGeometry(
    path: string,
    data: LoadedGSplatsData,
    viewState: GSplatsViewState
  ): void {
    if (!this.rootGroup) return;

    const mesh = this.rootGroup.getObjectByName(path) as THREE.Mesh;
    if (!mesh || mesh.userData?.nodeType !== 'gsplats') return;

    // Process nD data to 3D for rendering
    const processed = processGSplats(data, viewState);

    // Pack Cholesky factors for shader
    const { cholesky01, cholesky23, cholesky45 } = packCholeskyForShader(
      processed.choleskyFactors3D,
      processed.splatCount
    );

    // Phase 4: Use GPU buffer pool if enabled
    if (this._gpuBufferPool) {
      // Acquire geometry from pool (may reuse existing)
      const geometry = this._gpuBufferPool.acquireGSplatsGeometry(path, processed.splatCount);

      // Update attributes in place (zero GPU allocations on reuse)
      this._gpuBufferPool.updateGSplatsGeometry(
        geometry,
        {
          centers3D: processed.centers3D,
          amplitudes: processed.amplitudes,
          cholesky01,
          cholesky23,
          cholesky45,
          colors: processed.colors,
          sharpness: processed.sharpness,
          splatCount: processed.splatCount,
        },
        processed.splatCount
      );

      // Assign to mesh (might be same geometry, reused)
      mesh.geometry = geometry;
    } else {
      // Fallback: Original path (updateInstancedGSplatsMesh)
      updateInstancedGSplatsMesh(mesh, {
        centers: processed.centers3D,
        cholesky01,
        cholesky23,
        cholesky45,
        amplitudes: processed.amplitudes,
        sharpness: processed.sharpness,
        colors: processed.colors,
        splatCount: processed.splatCount,
      });
    }

    // Track visible splat count in mesh userData
    if (mesh.userData) {
      (mesh.userData as GSplatsUserData).visibleSplatCount = processed.splatCount;
    }

    if (processed.splatCount === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing gsplats for ${path} (no visible splats at current slice)`
      );
    }
  }

  /**
   * Build the scene graph structure
   */
  private async buildSceneGraph(
    rootLoc: zarr.Location<zarr.Readable>,
    rootAttrs: any
  ): Promise<SceneNode> {
    // Enumerate all groups in the store
    const listing = await this.enumerateStore();

    // Build hierarchical structure
    const root: SceneNode = {
      path: '/',
      type: 'scene',
      attrs: rootAttrs,
      hasSpatialIndex: false,
      children: [],
    };

    // Build node map
    const nodeMap = new Map<string, SceneNode>();
    nodeMap.set('/', root);

    // Sort by path depth to ensure parents are created before children
    const sortedPaths = listing
      .filter((e) => e.kind === 'group' && e.path !== '/')
      .sort((a, b) => a.path.split('/').length - b.path.split('/').length);

    for (const entry of sortedPaths) {
      const loc = rootLoc.resolve(entry.path.slice(1)); // Remove leading /
      const group = await zarr.open(loc, { kind: 'group' });
      const attrs = group.attrs as ZarrNodeAttrs;

      // We no longer check for spatial index here - PointSpatialIndexLoader handles it
      const node: SceneNode = {
        path: entry.path,
        type: attrs?.type || 'group',
        attrs: attrs || {},
        hasSpatialIndex: false, // Will be determined by the loader
        children: [],
      };

      // Log if extend_to_all is present
      if (attrs?.extend_to_all) {
        log.data(
          Modules.SCENE_LOADER,
          `Node ${entry.path} has extend_to_all: ${attrs.extend_to_all.join(', ')}`
        );
      }

      // Find parent and add as child
      const parentPath = entry.path.substring(0, entry.path.lastIndexOf('/')) || '/';
      const parent = nodeMap.get(parentPath);
      if (parent) {
        parent.children = parent.children || [];
        parent.children.push(node);
      }

      nodeMap.set(entry.path, node);
    }

    return root;
  }

  /**
   * Load all nodes in the scene graph
   */
  private async loadSceneNodes(
    node: SceneNode,
    parentThree: THREE.Object3D,
    parentLoc: zarr.Location<zarr.Readable>
  ): Promise<void> {
    if (node.type === 'points') {
      // Load points
      const points = await this.loadPoints(node, parentLoc);
      if (points) {
        parentThree.add(points);
      }
    } else if (node.type === 'lines') {
      // Load lines
      const lines = await this.loadLines(node, parentLoc);
      if (lines) {
        parentThree.add(lines);
      }
    } else if (node.type === 'gsplats') {
      // Load gsplats
      const gsplats = await this.loadGSplats(node, parentLoc);
      if (gsplats) {
        parentThree.add(gsplats);
      }
    } else if (node.children) {
      // Create group and recurse
      const group = new THREE.Group();
      group.name = node.path;

      // Apply transform if present
      if (node.attrs.transform) {
        this.applyTransform(group, node.attrs.transform);
      }

      parentThree.add(group);

      // Load children
      for (const child of node.children) {
        const childLoc = parentLoc.resolve(child.path.slice(1));
        await this.loadSceneNodes(child, group, childLoc);
      }
    }
  }

  /**
   * Load a single points node
   */
  private async loadPoints(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Points | null> {
    log.custom('📍', Modules.SCENE_LOADER, `Loading points: ${node.path}`);
    log.info(Modules.SCENE_LOADER, `  Has spatial index: ${node.hasSpatialIndex}`);
    log.info(Modules.SCENE_LOADER, `  Total points: ${node.attrs.n_points || 'unknown'}`);

    // Create appropriate loader
    const loader = this.createLoader(node, loc);

    // Store loader for updates
    this.loaders.set(node.path, loader);

    try {
      // Load points data
      log.info(Modules.SCENE_LOADER, 'Initial ViewState for loading:');
      log.info(Modules.SCENE_LOADER, `  displayDims: [${this.viewState.displayDims.join(', ')}]`);
      log.info(
        Modules.SCENE_LOADER,
        `  slicePosition: [${this.viewState.slicePosition.join(', ')}]`
      );
      log.info(Modules.SCENE_LOADER, `  tolerance: [${this.viewState.tolerance.join(', ')}]`);
      const data = await loader.loadPoints(this.viewState);

      // Create THREE.js geometry even if empty (for future updates)
      // Pass max_radius and max_sharpness from node attributes for proper scaling
      const maxRadius = (node.attrs.max_radius as number | undefined) ?? 1.0;
      const maxSharpness = (node.attrs.max_sharpness as number | undefined) ?? 31.0;
      const geometry = this.createGeometry(data, maxRadius, maxSharpness);

      // Log if no initial points are visible (this is normal for nD slicing)
      if (data.metadata.loadedPoints === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible points for ${node.path} - object created for future updates`
        );
      }

      // Create material with radius and sharpness scales from geometry userData
      const radiusScale = geometry.userData.radiusScale ?? 1.0;
      const sharpnessScale = geometry.userData.sharpnessScale ?? 1.0;
      const material = this.createMaterial(node.attrs, radiusScale, sharpnessScale);

      // Create points object
      const points = new THREE.Points(geometry, material);
      points.name = node.path;
      points.userData.loader = loader;
      points.userData.node = node;

      // Apply transform
      if (node.attrs.transform) {
        this.applyTransform(points, node.attrs.transform);
      }

      log.success(
        Modules.SCENE_LOADER,
        `Loaded ${data.metadata.loadedPoints} points for ${node.path}`
      );

      return points;
    } catch (error) {
      // Improved error logging - extract message from error object
      const errorMessage =
        error instanceof Error
          ? error.message
          : typeof error === 'string'
            ? error
            : JSON.stringify(error);
      log.error(Modules.SCENE_LOADER, `Failed to load ${node.path}: ${errorMessage}`);
      // Also log stack trace for debugging
      if (error instanceof Error && error.stack) {
        console.error(`[SceneLoader] Stack trace for ${node.path}:`, error.stack);
      }
      return null;
    }
  }

  /**
   * Load a single lines node
   */
  private async loadLines(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Mesh | null> {
    log.custom('📐', Modules.SCENE_LOADER, `Loading lines: ${node.path}`);

    const attrs = node.attrs as unknown as LinesMetadata;
    log.info(Modules.SCENE_LOADER, `  Segments: ${attrs.n_segments || 'unknown'}`);
    log.info(Modules.SCENE_LOADER, `  Vertices: ${attrs.n_vertices || 'unknown'}`);

    // Create lines loader
    const loader = this.createLinesLoader(node, loc);

    // Store loader for updates
    this.linesLoaders.set(node.path, loader);

    try {
      // Load lines data
      const linesViewState = {
        displayDims: this.viewState.displayDims,
        slicePosition: this.viewState.slicePosition,
        tolerance: this.viewState.tolerance,
        dimensions: this.viewState.dimensions?.metadata,
      };

      const data = await loader.loadLines(linesViewState);

      if (data.segmentCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible segments for ${node.path} - object created for future updates`
        );
      }

      // Build instance buffers with nD clipping
      let tolerance = linesViewState.dimensions
        ? computeLinesTolerance(linesViewState.dimensions, linesViewState.displayDims)
        : new Array(attrs.ndim || 3)
            .fill(0)
            .map((_, i) => (linesViewState.displayDims.includes(i) ? 1e10 : 0));

      // CRITICAL: For extend_to_all dimensions, set tolerance to infinity
      // This ensures segments aren't clipped when navigating through extended dimensions
      const extendDims: string[] = attrs.extend_to_all || [];
      if (extendDims.length > 0 && linesViewState.dimensions) {
        tolerance = [...tolerance]; // Make a copy to avoid mutating shared array
        for (const dimName of extendDims) {
          const dimIndex = linesViewState.dimensions.findIndex(
            (d: { name?: string }) => d.name === dimName
          );
          if (dimIndex >= 0 && dimIndex < tolerance.length) {
            tolerance[dimIndex] = 1e10; // Effectively infinite tolerance
          }
        }
      }

      const processed = buildInstanceBuffers(
        data,
        linesViewState.slicePosition,
        tolerance,
        linesViewState.displayDims
      );

      // Create material
      const material = materialManager.getLineMaterial({
        opacity: attrs.opacity ?? 1.0,
        blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
      });

      // Create instanced mesh
      const mesh = createInstancedLinesMesh(processed, material);
      mesh.name = node.path;

      // Store user data for identification (including visible segment count for monitor)
      mesh.userData = {
        nodeType: 'lines',
        loader,
        attrs,
        maxWidth: attrs.max_width ?? 1.0,
        visibleSegmentCount: processed.segmentCount,
      } as LinesUserData;

      // Apply transform
      if (attrs.transform) {
        this.applyTransform(mesh, attrs.transform);
      }

      log.success(
        Modules.SCENE_LOADER,
        `Loaded ${processed.segmentCount} segments for ${node.path}`
      );

      return mesh;
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : typeof error === 'string'
            ? error
            : JSON.stringify(error);
      log.error(Modules.SCENE_LOADER, `Failed to load lines ${node.path}: ${errorMessage}`);
      if (error instanceof Error && error.stack) {
        console.error(`[SceneLoader] Stack trace for ${node.path}:`, error.stack);
      }
      return null;
    }
  }

  /**
   * Create a lines loader for a node
   */
  private createLinesLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): LinesDataLoader {
    const nodeLoc = node.path === '/' ? loc : zarr.root(this.store!).resolve(node.path.slice(1));

    log.query(Modules.SCENE_LOADER, `Using LinesSpatialIndexLoader for ${node.path}`);
    const loader = new LinesSpatialIndexLoader(nodeLoc, node, this.arrayRefRegistry, this.store!);

    return loader;
  }

  /**
   * Load a single gsplats node
   */
  private async loadGSplats(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): Promise<THREE.Mesh | null> {
    const attrs = node.attrs as unknown as GSplatsMetadata;
    log.custom('🔮', Modules.SCENE_LOADER, `Loading gsplats: ${node.path}`);
    log.info(Modules.SCENE_LOADER, `  Splats: ${attrs.n_splats?.toLocaleString() || 'unknown'}`);
    log.info(Modules.SCENE_LOADER, `  Dimensions: ${attrs.ndim || 'unknown'}D`);

    // Create gsplats loader
    const loader = this.createGSplatsLoader(node, loc);

    // Store loader for updates
    this.gsplatLoaders.set(node.path, loader);

    try {
      // Build gsplats view state
      const gsplatsViewState: GSplatsViewState = {
        displayDims: this.viewState.displayDims,
        slicePosition: this.viewState.slicePosition,
        tolerance: this.viewState.tolerance,
        dimensions: this.viewState.dimensions?.metadata,
      };

      // Load gsplats data
      const data = await loader.loadGSplats(gsplatsViewState);

      if (data.splatCount === 0) {
        log.info(
          Modules.SCENE_LOADER,
          `No initially visible gsplats for ${node.path} - object created for future updates`
        );
      }

      // Process nD data to 3D for rendering
      const processed = processGSplats(data, gsplatsViewState);

      // Pack Cholesky factors for shader
      const { cholesky01, cholesky23, cholesky45 } = packCholeskyForShader(
        processed.choleskyFactors3D,
        processed.splatCount
      );

      // Create material
      const material = materialManager.getGSplatMaterial({
        opacity: attrs.opacity ?? 1.0,
        blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
      });

      // Create instanced mesh
      const mesh = createInstancedGSplatsMesh(
        {
          centers: processed.centers3D,
          cholesky01,
          cholesky23,
          cholesky45,
          amplitudes: processed.amplitudes,
          sharpness: processed.sharpness,
          colors: processed.colors,
          splatCount: processed.splatCount,
        },
        material
      );
      mesh.name = node.path;

      // Store user data for identification
      mesh.userData = {
        nodeType: 'gsplats',
        loader,
        attrs,
        visibleSplatCount: processed.splatCount,
      } as GSplatsUserData;

      // Apply transform
      if (attrs.transform) {
        this.applyTransform(mesh, attrs.transform);
      }

      log.success(
        Modules.SCENE_LOADER,
        `Loaded ${processed.splatCount.toLocaleString()} gsplats for ${node.path}`
      );

      return mesh;
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : typeof error === 'string'
            ? error
            : JSON.stringify(error);
      log.error(Modules.SCENE_LOADER, `Failed to load gsplats ${node.path}: ${errorMessage}`);
      if (error instanceof Error && error.stack) {
        console.error(`[SceneLoader] Stack trace for ${node.path}:`, error.stack);
      }
      return null;
    }
  }

  /**
   * Create a gsplats loader for a node
   */
  private createGSplatsLoader(
    node: SceneNode,
    loc: zarr.Location<zarr.Readable>
  ): GSplatsDataLoader {
    const nodeLoc = node.path === '/' ? loc : zarr.root(this.store!).resolve(node.path.slice(1));

    log.query(Modules.SCENE_LOADER, `Using GSplatsSpatialIndexLoader for ${node.path}`);
    const loader = new GSplatsSpatialIndexLoader(nodeLoc, node, this.arrayRefRegistry, this.store!);

    return loader;
  }

  /**
   * Create the spatial index loader for a node
   */
  private createLoader(node: SceneNode, loc: zarr.Location<zarr.Readable>): DataLoader {
    // Resolve the correct location for this node
    const nodeLoc = node.path === '/' ? loc : zarr.root(this.store!).resolve(node.path.slice(1));

    // Use PointSpatialIndexLoader for all nodes (it will handle 3D datasets without indices)
    log.query(Modules.SCENE_LOADER, `Using PointSpatialIndexLoader for ${node.path}`);
    // Pass the store reference for array_ref resolution (needed by ArrayDecoder)
    const loader = new PointSpatialIndexLoader(
      nodeLoc,
      node,
      this.config,
      this.arrayRefRegistry,
      this.store!
    );

    // Connect to monitor if available
    if (this.monitorId) {
      const monitor = DataMonitorManager.getInstance().getMonitor(this.monitorId);
      if (monitor) {
        monitor.connectLoader(node.path, loader);
      }
    }

    return loader;
  }

  /**
   * Validate points data for edge cases and malformed data
   *
   * Logs detailed diagnostics to browser console for debugging
   */
  private validatePointsData(data: PointsData): void {
    const pointCount = data.positions.length / 3;

    // Log data summary for debugging
    log.info(Modules.SCENE_LOADER, 'Points Data Validation:', {
      pointCount,
      positionsLength: data.positions.length,
      positionsType: data.positions.constructor.name,
      hasColors: !!data.colors,
      colorsType: data.colors?.constructor.name,
      colorsLength: data.colors?.length,
      hasRadii: !!data.radii,
      radiiType: data.radii?.constructor.name,
      radiiLength: data.radii?.length,
      hasSharpness: !!data.sharpness,
      sharpnessType: data.sharpness?.constructor.name,
      sharpnessLength: data.sharpness?.length,
    });

    // EDGE CASE: Empty dataset
    if (pointCount === 0) {
      log.warning(Modules.SCENE_LOADER, 'Empty dataset detected - no points to render');
      return;
    }

    // EDGE CASE: Malformed positions (not multiple of 3)
    if (data.positions.length % 3 !== 0) {
      const error = `Malformed positions array: length ${data.positions.length} is not divisible by 3`;
      log.error(Modules.SCENE_LOADER, error);
      throw new Error(error);
    }

    // VALIDATION: Colors length consistency
    if (data.colors && data.colors.length !== data.positions.length) {
      const expected = data.positions.length;
      const actual = data.colors.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Colors length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    // VALIDATION: Radii length consistency
    if (data.radii && data.radii.length !== pointCount) {
      const expected = pointCount;
      const actual = data.radii.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Radii length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    // VALIDATION: Sharpness length consistency
    if (data.sharpness && data.sharpness.length !== pointCount) {
      const expected = pointCount;
      const actual = data.sharpness.length;
      log.warning(
        Modules.SCENE_LOADER,
        `Sharpness length mismatch: expected ${expected}, got ${actual}`,
        { expected, actual }
      );
    }

    // Log successful validation
    log.success(Modules.SCENE_LOADER, `Points data validated: ${pointCount} points`);
  }

  /**
   * Create THREE.js geometry from points data
   * @param data - Points data with positions, colors, radii, sharpness
   * @param maxRadius - Maximum radius from node attributes for scaling uint8 radii
   * @param maxSharpness - Maximum sharpness from node attributes for scaling uint8 sharpness
   */
  private createGeometry(
    data: PointsData,
    maxRadius: number = 1.0,
    maxSharpness: number = 31.0
  ): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();

    // VALIDATION: Check for edge cases and log detailed diagnostics
    this.validatePointsData(data);

    // Set positions (handle Float16Array conversion if needed)
    if (
      typeof (globalThis as any).Float16Array !== 'undefined' &&
      data.positions instanceof (globalThis as any).Float16Array
    ) {
      // Convert Float16Array to Float32Array for THREE.js compatibility
      const float32Positions = new Float32Array(data.positions);
      geometry.setAttribute('position', new THREE.BufferAttribute(float32Positions, 3));
    } else {
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(data.positions as Float32Array, 3)
      );
    }

    // Set colors if available
    if (data.colors) {
      // Validate color mode consistency
      this.validateColorMode(data.colors, data.metadata as any);

      // Check if colors need normalization (for uint8/uint16 arrays)
      const needsNormalization =
        data.colors instanceof Uint8Array || data.colors instanceof Uint16Array;

      geometry.setAttribute('color', new THREE.BufferAttribute(data.colors, 3, needsNormalization));
    }

    // Set radii if available, or use default
    let radiusScale = 1.0; // Default scale for float32 radii

    if (data.radii) {
      // Check if radii need normalization or conversion
      if (
        typeof (globalThis as any).Float16Array !== 'undefined' &&
        data.radii instanceof (globalThis as any).Float16Array
      ) {
        // Convert Float16Array to Float32Array for THREE.js
        const float32Radii = new Float32Array(data.radii);
        geometry.setAttribute('radius', new THREE.BufferAttribute(float32Radii, 1));
        // Float16 values are already in world units, no scaling needed
        radiusScale = 1.0;
      } else if (data.radii instanceof Uint8Array) {
        // Uint8 radii need scaling from 0-255 to 0-1 (or world units)
        // Use the normalization flag for proper GPU upload
        geometry.setAttribute(
          'radius',
          new THREE.BufferAttribute(data.radii, 1, true) // true = normalize on GPU
        );
        // GPU normalizes uint8 [0, 255] to [0, 1]
        // Python encodes radii with bounded_scalar_uint8: value in [0, max_radius]
        // After GPU normalization we get normalized values in [0, 1]
        // Multiply by maxRadius to get world-space radius
        radiusScale = maxRadius;
      } else {
        // Float32 radii - no normalization or scaling needed
        geometry.setAttribute(
          'radius',
          new THREE.BufferAttribute(data.radii as Float32Array, 1, false)
        );
        radiusScale = 1.0;
      }
    } else {
      // Create default radius array with value 0.5 for all points
      const numPoints = data.positions.length / 3;
      const defaultRadii = new Float32Array(numPoints).fill(0.5);
      geometry.setAttribute('radius', new THREE.BufferAttribute(defaultRadii, 1));
      radiusScale = 1.0;
    }

    // Set sharpness if available, or use default
    let sharpnessScale = 1.0; // Default scale for float32 sharpness

    if (data.sharpness) {
      // Check if sharpness needs normalization or conversion
      if (
        typeof (globalThis as any).Float16Array !== 'undefined' &&
        data.sharpness instanceof (globalThis as any).Float16Array
      ) {
        // Convert Float16Array to Float32Array for THREE.js
        const float32Sharpness = new Float32Array(data.sharpness);
        geometry.setAttribute('sharpness', new THREE.BufferAttribute(float32Sharpness, 1));
        // Float16 values are already in world units, no scaling needed
        sharpnessScale = 1.0;
      } else if (data.sharpness instanceof Uint8Array) {
        // Uint8 sharpness needs scaling - check metadata for range
        // Use the normalization flag for proper GPU upload
        geometry.setAttribute(
          'sharpness',
          new THREE.BufferAttribute(data.sharpness, 1, true) // true = normalize on GPU
        );

        // GPU normalizes uint8 to [0,1], then scale to sharpness range
        // Use max_sharpness passed from node attributes
        sharpnessScale = maxSharpness;
      } else {
        // Float32 sharpness - no normalization or scaling needed
        geometry.setAttribute(
          'sharpness',
          new THREE.BufferAttribute(data.sharpness as Float32Array, 1, false)
        );
        sharpnessScale = 1.0;
      }
    } else {
      // Create default sharpness array with value 2.0 for all points
      const numPoints = data.positions.length / 3;
      const defaultSharpness = new Float32Array(numPoints).fill(2.0);
      geometry.setAttribute('sharpness', new THREE.BufferAttribute(defaultSharpness, 1));
      sharpnessScale = 1.0;
    }

    // Compute bounding box
    geometry.boundingBox = data.metadata.bounds.clone();

    // Store radius and sharpness scales as user data for material creation
    if (!geometry.userData) {
      geometry.userData = {};
    }
    geometry.userData.radiusScale = radiusScale;
    geometry.userData.sharpnessScale = sharpnessScale;

    return geometry;
  }

  /**
   * Create material for points
   */
  private createMaterial(
    attrs: any,
    radiusScale: number = 1.0,
    sharpnessScale: number = 1.0
  ): THREE.ShaderMaterial {
    return materialManager.getPointMaterial({
      opacity: attrs.opacity ?? 1.0,
      gamma: attrs.gamma ?? 1.0,
      blendingMode: (attrs.blending_mode as BlendingMode) ?? 'additive',
      radiusScale: radiusScale,
      sharpnessScale: sharpnessScale,
    });
  }

  /**
   * Validate transform matrix format (detect row-major vs column-major)
   *
   * THREE.js expects column-major (OpenGL-style) where translation is at indices [12, 13, 14]
   * NumPy uses row-major (C-style) where translation is at indices [3, 7, 11]
   *
   * Python should transpose before writing: matrix.T.ravel().tolist()
   */
  private validateTransformFormat(transform: number[]): boolean {
    // Check if translation components look suspicious
    // In column-major (correct for THREE.js): [12]=tx, [13]=ty, [14]=tz
    // In row-major (wrong for THREE.js): [3]=tx, [7]=ty, [11]=tz

    const colMajorTranslation = [transform[12], transform[13], transform[14]];
    const rowMajorTranslation = [transform[3], transform[7], transform[11]];

    const colMajorNonZero = colMajorTranslation.some((v) => Math.abs(v) > 0.001);
    const rowMajorNonZero = rowMajorTranslation.some((v) => Math.abs(v) > 0.001);

    // If row-major positions are non-zero but column-major are zero, likely wrong format
    if (rowMajorNonZero && !colMajorNonZero) {
      log.warning(
        Modules.SCENE_LOADER,
        'Transform matrix appears to be in row-major (NumPy) format instead of column-major (THREE.js). ' +
          'Translation detected at wrong indices [3,7,11] instead of [12,13,14]. ' +
          'Python should transpose before storing: matrix.T.ravel().tolist()'
      );
      return false;
    }

    return true;
  }

  /**
   * Apply transformation matrix to object
   */
  private applyTransform(object: THREE.Object3D, transform: number[]): void {
    if (transform.length !== 16) {
      log.warning(Modules.SCENE_LOADER, `Invalid transform length: ${transform.length}`);
      return;
    }

    // Validate transform format (detect common mistakes)
    this.validateTransformFormat(transform);

    const matrix = new THREE.Matrix4().fromArray(transform);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();

    matrix.decompose(position, quaternion, scale);

    object.position.copy(position);
    object.quaternion.copy(quaternion);
    object.scale.copy(scale);
  }

  /**
   * Update geometry for a specific points
   */
  private updatePointsGeometry(path: string, data: PointsData): void {
    if (!this.rootGroup) return;

    // Find the points object
    const points = this.rootGroup.getObjectByName(path) as THREE.Points;
    if (!points) return;

    // Log if updating to empty geometry (clearing points)
    if (data.metadata.loadedPoints === 0) {
      log.info(
        Modules.SCENE_LOADER,
        `Clearing points for ${path} (no visible points at current slice)`
      );
    }

    // Phase 4: Use GPU buffer pool if enabled (now supports all TypedArray types!)
    if (this._gpuBufferPool) {
      // Acquire geometry from pool (type-aware: matches capacity AND attribute types)
      const geometry = this._gpuBufferPool.acquirePointsGeometry(
        path,
        data,
        data.metadata.loadedPoints
      );

      // Update attributes in place (zero GPU allocations on reuse)
      this._gpuBufferPool.updatePointsGeometry(geometry, data, data.metadata.loadedPoints);

      // Update bounding box
      if (data.metadata.bounds) {
        geometry.boundingBox = data.metadata.bounds.clone();
      }

      // Assign to mesh (might be same geometry, reused)
      points.geometry = geometry;
    } else {
      // Fallback: GPU buffer pool disabled, use standard path
      const oldGeometry = points.geometry;

      // Save bounding box/sphere BEFORE disposal to preserve them
      const savedBoundingBox = oldGeometry?.boundingBox?.clone() || null;
      const savedBoundingSphere = oldGeometry?.boundingSphere?.clone() || null;

      // Dispose old geometry FIRST to free GPU memory immediately
      if (oldGeometry) {
        oldGeometry.dispose();
      }

      // Create new geometry AFTER disposal
      const newGeometry = this.createGeometry(data);

      // Restore bounding box/sphere if available
      if (savedBoundingBox) {
        newGeometry.boundingBox = savedBoundingBox;
      }
      if (savedBoundingSphere) {
        newGeometry.boundingSphere = savedBoundingSphere;
      }

      // Assign the new geometry
      points.geometry = newGeometry;
    }
  }

  /**
   * Validate color mode consistency
   *
   * Ensures color array type matches expected encoding:
   * - Float32Array for HDR colors (values > 1.0)
   * - Uint8Array for SDR colors (values [0, 1])
   * - Warns about potential issues
   */
  private validateColorMode(
    colors: Uint8Array | Uint16Array | Float32Array,
    nodeMetadata: any
  ): void {
    const isHDR = colors instanceof Float32Array;
    const isSDR = colors instanceof Uint8Array || colors instanceof Uint16Array;

    // Check for suspicious patterns
    if (isSDR && nodeMetadata?.color_mode === 'hdr') {
      log.warning(
        Modules.SCENE_LOADER,
        `Node metadata indicates HDR colors but array is ${colors.constructor.name}. ` +
          'HDR colors should use Float32Array. This may indicate incorrect encoding.'
      );
    }

    if (isHDR) {
      // For float32 colors, check if any values exceed 1.0 (HDR range)
      const hasHDRValues = Array.from(colors).some((v) => v > 1.0);
      if (!hasHDRValues && nodeMetadata?.color_mode === 'hdr') {
        log.info(
          Modules.SCENE_LOADER,
          'HDR color mode specified but all values in [0, 1] range. Consider using SDR mode for better compression.'
        );
      }
    }

    // Log color mode for debugging
    const colorType = colors.constructor.name;
    const colorMode = isHDR ? 'HDR (float32)' : 'SDR (normalized integer)';
    log.info(Modules.SCENE_LOADER, `Colors: ${colorType} - ${colorMode}`);
  }

  /**
   * Initialize scene dimensions from metadata using ViewStateManager
   */
  private initializeSceneDimensions(sceneDims: any): void {
    // Validate sceneDims structure
    if (!sceneDims || typeof sceneDims !== 'object' || !Array.isArray(sceneDims.dimensions)) {
      log.warning(Modules.SCENE_LOADER, 'Invalid scene_dimensions format, skipping');
      return;
    }

    // Validate dimensions using ViewStateManager
    const validation = ViewStateManager.validateDimensions(sceneDims.dimensions);

    // Log validation results
    const displayedCount = sceneDims.dimensions.filter((d: any) => d.display === true).length;
    ViewStateManager.logValidationResults(validation, sceneDims.dimensions.length, displayedCount);

    // Stop if validation failed with errors
    if (!validation.isValid) {
      log.error(Modules.SCENE_LOADER, 'Scene dimensions validation failed, cannot initialize');
      return;
    }

    // Initialize ViewState using ViewStateManager
    this.viewState = ViewStateManager.initializeFromDimensions(sceneDims as SceneDimensions);
  }

  /**
   * Convert internal SceneNode to SceneGraphNode for monitor display
   */
  private convertToSceneGraphNode(node: SceneNode): SceneGraphNode {
    // Get display name from path
    const name =
      node.path === '/' ? 'Scene' : node.path.split('/').filter(Boolean).pop() || node.path;

    // Determine node type for display
    const type = node.type as 'scene' | 'group' | 'points' | 'lines' | 'gsplats' | 'mesh';

    // Build the graph node
    const graphNode: SceneGraphNode = {
      path: node.path,
      name,
      type: type === 'scene' || !type ? 'scene' : type,
      children: [],
      hasSpatialIndex: node.hasSpatialIndex,
    };

    // Add type-specific stats
    if (node.type === 'points') {
      graphNode.pointCount = node.attrs.n_points;
    } else if (node.type === 'lines') {
      graphNode.segmentCount = node.attrs.n_segments as number | undefined;
      graphNode.vertexCount = node.attrs.n_vertices as number | undefined;
    } else if (node.type === 'gsplats') {
      graphNode.splatCount = node.attrs.n_splats as number | undefined;
    }

    // Convert children recursively
    if (node.children) {
      graphNode.children = node.children.map((child) => this.convertToSceneGraphNode(child));
    }

    return graphNode;
  }

  /**
   * Enumerate all groups and arrays in the store
   */
  private async enumerateStore(): Promise<Array<{ path: string; kind: string }>> {
    if (!this.store) return [];

    // Try to use consolidated metadata
    if (hasContentsMethod(this.store)) {
      const contents = await this.store.contents();
      log.custom('📋', Modules.SCENE_LOADER, `Found ${contents.length} items in store`);
      return contents;
    }

    // Fallback enumeration
    log.warning(Modules.SCENE_LOADER, 'Store does not support contents(), using fallback');
    return [{ path: '/', kind: 'group' }];
  }

  /**
   * Normalize URL for zarr store access
   */
  private normalizeURL(url: string): string {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return url.endsWith('/') ? url : url + '/';
    }
    const baseUrl = window.location.origin;
    const cleanPath = url.startsWith('/') ? url : '/' + url;
    return baseUrl + cleanPath + (cleanPath.endsWith('/') ? '' : '/');
  }

  /**
   * Show the monitor UI
   */
  showMonitor(): void {
    if (this.monitorId) {
      DataMonitorManager.getInstance().showMonitor(this.monitorId);
    }
  }

  /**
   * Hide the monitor UI
   */
  hideMonitor(): void {
    if (this.monitorId) {
      DataMonitorManager.getInstance().hideMonitor(this.monitorId);
    }
  }

  /**
   * Toggle the monitor UI
   */
  toggleMonitor(): void {
    if (this.monitorId) {
      DataMonitorManager.getInstance().toggleMonitor(this.monitorId);
    }
  }

  /**
   * Get information about failed loaders
   * @returns Map of loader paths to error information
   */
  getFailedLoaders(): ReadonlyMap<string, { error: Error; timestamp: number; retryCount: number }> {
    return this.failedLoaders;
  }

  /**
   * Check if there are any failed loaders
   */
  hasFailures(): boolean {
    return this.failedLoaders.size > 0;
  }

  /**
   * Clear failed loader tracking
   * Useful for retry operations or after user acknowledges errors
   */
  clearFailures(): void {
    const count = this.failedLoaders.size;
    this.failedLoaders.clear();
    if (count > 0) {
      log.info(Modules.SCENE_LOADER, `Cleared ${count} failed loader(s) from tracking`);
    }
  }

  /**
   * Dispose of all resources
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

    // Dispose caching store (flushes L2 metadata, clears L1)
    if (this.cachingStore) {
      this.cachingStore.dispose().catch(() => {});
      this.cachingStore = null;
    }

    this.store = null;
    this.rootGroup = null;

    // Note: We don't dispose the monitor here as it's managed by DataMonitorManager
    // The monitor can be reused by other SceneLoader instances
    this.monitorId = null;
  }
}
