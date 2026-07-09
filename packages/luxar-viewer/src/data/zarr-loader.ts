/**
 * Zarr-based nD scene loader.
 *
 * Handles spatial indices and keeps all attribute loads aligned to the
 * same selected ranges.
 *
 * Key properties:
 * - Spatial index integration with proper attribute alignment
 * - No cache key collisions
 * - Clean separation of loading strategies
 * - Proper memory management
 */

import * as THREE from 'three';
import { SceneLoaderManager } from './scene-loader-manager';
import { ViewState, LoaderConfig } from './data-loader-types';
import { SimpleDims } from '../types/dims';
import { log, Modules, LogEmoji } from '../utils/log';
import { config } from '../config';
import { simpleDimsToViewState } from './dims-to-view-state';
import { computeSceneStats } from './stats/scene-stats';

/**
 * Load a complete scene from a Zarr store.
 *
 * This is the main entry point for SceneLoader-backed loading.
 *
 * @param src - URL or path to the Zarr store
 * @param config - Optional loader configuration
 * @param loaderId - Optional ID to register the created SceneLoader under (default: 'default')
 * @returns Promise resolving to a THREE.Group containing the scene
 */
export async function loadScene(
  src: string,
  config?: LoaderConfig,
  loaderId: string = 'default'
): Promise<THREE.Group> {
  log.custom(LogEmoji.START, Modules.LUXAR, 'Loading scene');

  const manager = SceneLoaderManager.getInstance();

  // Always create a fresh scene loader for each load to ensure clean state.
  // This disposes any existing loader and its connections first.
  const sceneLoader = manager.createLoader(loaderId, config);

  try {
    // Load the scene
    const scene = await sceneLoader.loadScene(src);

    // Log success
    log.success(Modules.LUXAR, 'Scene loaded successfully');
    logSceneStats(scene);

    return scene;
  } catch (error) {
    log.error(Modules.LUXAR, 'Failed to load scene:', error);
    throw error;
  }
}

/**
 * Update the view state for all loaded points.
 *
 * This function updates all points when the user navigates
 * through nD space or changes display dimensions.
 *
 * @param viewState - New view state to apply
 * @param loaderId - Optional loader ID, defaults to default loader
 */
export async function updateView(viewState: Partial<ViewState>, loaderId?: string): Promise<void> {
  const manager = SceneLoaderManager.getInstance();
  const sceneLoader = loaderId ? manager.getLoader(loaderId) : manager.getDefaultLoader();

  if (!sceneLoader) {
    log.warning(Modules.LUXAR, 'No scene loaded, cannot update view');
    return;
  }

  log.update(Modules.LUXAR, 'Updating view state');
  try {
    await sceneLoader.updateView(viewState);
  } catch (error) {
    log.error(Modules.LUXAR, 'Failed to update view:', error);
    // Don't throw - allow system to continue functioning
  }
}

/**
 * Update the scene when navigating through dimensions.
 *
 * This is a convenience function for dimension navigation that
 * automatically converts dimension state to view state.
 *
 * @param dims - Current dimension state
 * @param scene - THREE.Group containing the scene
 * @param loaderId - Optional loader ID, defaults to default loader
 */
export async function updateSceneForDimensions(
  dims: SimpleDims,
  scene: THREE.Group,
  loaderId?: string,
  opts?: {
    /**
     * Per-tick LOD time budget during dimension-animation playback (see
     * `ViewState.frameBudgetMs`). A per-pass directive — attached to this
     * one update call, never persisted.
     */
    frameBudgetMs?: number;
  }
): Promise<void> {
  const maxRadius = scene.userData.maxRadius || config.dataLoading.spatial.defaultMaxRadius;
  const viewState = simpleDimsToViewState(dims, {
    maxRadius,
    defaultTolerance: config.dataLoading.spatial.defaultTolerance,
  });
  if (opts?.frameBudgetMs !== undefined) {
    viewState.frameBudgetMs = opts.frameBudgetMs;
  }

  await updateView(viewState, loaderId);
}

/**
 * Dispose of all resources and clean up.
 *
 * @param loaderId - Optional loader ID to dispose, or dispose all if not specified
 */
export function dispose(loaderId?: string): void {
  const manager = SceneLoaderManager.getInstance();

  if (loaderId) {
    manager.destroyLoader(loaderId);
    log.custom(LogEmoji.CLEAN, Modules.LUXAR, `Scene loader '${loaderId}' disposed`);
  } else {
    manager.destroyAll();
    log.custom(LogEmoji.CLEAN, Modules.LUXAR, 'All scene loaders disposed');
  }
}

/**
 * Log statistics about the loaded scene. The traversal/counting logic is
 * `data/stats/scene-stats.ts::computeSceneStats`; this function is only
 * the logging layer.
 */
function logSceneStats(scene: THREE.Group): void {
  const stats = computeSceneStats(scene);
  if (!stats) {
    log.warning(Modules.LUXAR, 'Scene does not have traverse method, skipping stats');
    return;
  }

  log.info(Modules.LUXAR, 'Scene statistics:');
  log.info(Modules.LUXAR, `  - Points objects: ${stats.pointsObjects}`);
  log.info(Modules.LUXAR, `  - Total points loaded: ${stats.totalPoints.toLocaleString()}`);
  log.info(Modules.LUXAR, `  - Lines objects: ${stats.linesObjects}`);
  log.info(Modules.LUXAR, `  - Total segments loaded: ${stats.totalSegments.toLocaleString()}`);
  log.info(Modules.LUXAR, `  - GSplats objects: ${stats.gsplatsObjects}`);
  log.info(Modules.LUXAR, `  - Total gsplats loaded: ${stats.totalGSplats.toLocaleString()}`);
  log.info(
    Modules.LUXAR,
    `  - Using spatial index: ${stats.spatialIndexed}/${stats.pointsObjects + stats.linesObjects + stats.gsplatsObjects}`
  );
}
