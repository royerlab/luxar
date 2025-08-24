/**
 * Zarr-based nD point cloud data loader - Clean Architecture v2
 *
 * This is the new implementation that properly handles spatial indices
 * and ensures all attributes are loaded with aligned ranges.
 *
 * Key fixes:
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

/**
 * Load a complete scene from a Zarr store using the new architecture.
 *
 * This is the main entry point that replaces the old loadScene function.
 * It uses the new SceneLoader which uses SpatialIndexLoader for all point clouds.
 *
 * @param src - URL or path to the Zarr store
 * @param config - Optional loader configuration
 * @returns Promise resolving to a THREE.Group containing the scene
 */
export async function loadScene(
  src: string,
  config?: LoaderConfig,
  loaderId: string = 'default'
): Promise<THREE.Group> {
  log.custom(LogEmoji.START, Modules.LUXAR, 'Loading scene with clean architecture');

  const manager = SceneLoaderManager.getInstance();

  // Create or reuse scene loader
  let sceneLoader = manager.getLoader(loaderId);
  if (!sceneLoader) {
    sceneLoader = manager.createLoader(loaderId, config);
  }

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
 * Update the view state for all loaded point clouds.
 *
 * This function updates all point clouds when the user navigates
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
  loaderId?: string
): Promise<void> {
  const viewState: ViewState = {
    displayDims: dims.displayed,
    slicePosition: dims.currentStep,
    tolerance: new Array(dims.ndim).fill(0.1), // Default tolerance
    dimensions: dims,
  };

  // Update max radius from scene metadata if available
  const maxRadius = scene.userData.maxRadius || 0.1;

  // For non-displayed dimensions, use the max radius as tolerance for slicing
  // Displayed dimensions are not indexed, so no tolerance is needed for them
  viewState.tolerance = viewState.tolerance.map(
    (_, i) => (dims.displayed.includes(i) ? 0 : maxRadius) // 0 for displayed dims (not used in queries)
  );

  await updateView(viewState, loaderId);
}

/**
 * Get cache statistics for monitoring.
 *
 * @param loaderId - Optional loader ID, defaults to default loader
 */
export function getCacheStats(loaderId?: string): Map<string, any> | null {
  const manager = SceneLoaderManager.getInstance();
  const sceneLoader = loaderId ? manager.getLoader(loaderId) : manager.getDefaultLoader();

  if (!sceneLoader) {
    return null;
  }
  return sceneLoader.getCacheStats();
}

/**
 * Clear all caches to free memory.
 *
 * @param loaderId - Optional loader ID, defaults to default loader
 */
export function clearCaches(loaderId?: string): void {
  const manager = SceneLoaderManager.getInstance();
  const sceneLoader = loaderId ? manager.getLoader(loaderId) : manager.getDefaultLoader();

  if (sceneLoader) {
    sceneLoader.clearCaches();
    log.custom(LogEmoji.CLEAN, Modules.LUXAR, 'Caches cleared');
  }
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
 * Log statistics about the loaded scene.
 */
function logSceneStats(scene: THREE.Group): void {
  let totalPoints = 0;
  let totalPointClouds = 0;
  let usedSpatialIndex = 0;

  // Check if scene has traverse method (it might be a mock in tests)
  if (!scene || typeof scene.traverse !== 'function') {
    log.warning(Modules.LUXAR, 'Scene does not have traverse method, skipping stats');
    return;
  }

  scene.traverse((obj) => {
    if (obj instanceof THREE.Points) {
      totalPointClouds++;
      const geometry = obj.geometry;
      const positions = geometry.getAttribute('position');
      if (positions) {
        totalPoints += positions.count;
      }
      if (obj.userData.node?.hasSpatialIndex) {
        usedSpatialIndex++;
      }
    }
  });

  log.info(Modules.LUXAR, 'Scene statistics:');
  log.info(Modules.LUXAR, `  - Point clouds: ${totalPointClouds}`);
  log.info(Modules.LUXAR, `  - Total points loaded: ${totalPoints.toLocaleString()}`);
  log.info(Modules.LUXAR, `  - Using spatial index: ${usedSpatialIndex}/${totalPointClouds}`);
}
