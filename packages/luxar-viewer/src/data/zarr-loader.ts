/**
 * Zarr-based nD points data loader - Clean Architecture v2
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
import { config } from '../config';

/**
 * Load a complete scene from a Zarr store using the new architecture.
 *
 * This is the main entry point that replaces the old loadScene function.
 * It uses the new SceneLoader which uses PointsSpatialIndexLoader for all points.
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
  log.custom(LogEmoji.START, Modules.LUXAR, 'Loading scene with clean architecture');

  const manager = SceneLoaderManager.getInstance();

  // Always create a fresh scene loader for each load to ensure clean state
  // This properly disposes the old loader and its connections if it exists
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
  loaderId?: string
): Promise<void> {
  const viewState: ViewState = {
    displayDims: [...dims.displayed], // Copy to avoid reference mutation
    slicePosition: [...dims.currentStep], // Copy to avoid reference mutation
    tolerance: new Array(dims.ndim).fill(config.dataLoading.spatial.defaultTolerance), // Default tolerance
    dimensions: dims.metadata,
  };

  // Update max radius from scene metadata if available
  const maxRadius = scene.userData.maxRadius || config.dataLoading.spatial.defaultMaxRadius;

  // Set tolerance per dimension based on type:
  // - Displayed dimensions: 0 (they're in the viewing plane, not queried)
  // - Discrete non-displayed: 0.5 (exact match with float tolerance)
  // - Spatial/continuous non-displayed: maxRadius (points extend through these)
  viewState.tolerance = viewState.tolerance.map((_, i) => {
    if (dims.displayed.includes(i)) {
      return 0;
    }
    const meta = dims.metadata?.[i];
    if (meta?.discrete && !meta?.spatial) {
      return 0.5; // Discrete dimensions need near-exact match
    }
    return maxRadius;
  });

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
 * Log statistics about the loaded scene.
 */
function logSceneStats(scene: THREE.Group): void {
  let totalPoints = 0;
  let totalPointsObjects = 0;
  let totalGSplats = 0;
  let totalGSplatsObjects = 0;
  let usedSpatialIndex = 0;

  // Check if scene has traverse method (it might be a mock in tests)
  if (!scene || typeof scene.traverse !== 'function') {
    log.warning(Modules.LUXAR, 'Scene does not have traverse method, skipping stats');
    return;
  }

  scene.traverse((obj) => {
    if (obj instanceof THREE.Points) {
      totalPointsObjects++;
      const geometry = obj.geometry;
      const positions = geometry.getAttribute('position');
      if (positions) {
        totalPoints += positions.count;
      }
      if (obj.userData.attrs?.has_spatial_index) {
        usedSpatialIndex++;
      }
    } else if (obj instanceof THREE.Mesh && obj.userData?.nodeType === 'gsplats') {
      totalGSplatsObjects++;
      totalGSplats += obj.userData.visibleSplatCount ?? 0;
      if (obj.userData.attrs?.has_spatial_index) {
        usedSpatialIndex++;
      }
    }
  });

  log.info(Modules.LUXAR, 'Scene statistics:');
  log.info(Modules.LUXAR, `  - Points objects: ${totalPointsObjects}`);
  log.info(Modules.LUXAR, `  - Total points loaded: ${totalPoints.toLocaleString()}`);
  log.info(Modules.LUXAR, `  - GSplats objects: ${totalGSplatsObjects}`);
  log.info(Modules.LUXAR, `  - Total gsplats loaded: ${totalGSplats.toLocaleString()}`);
  log.info(
    Modules.LUXAR,
    `  - Using spatial index: ${usedSpatialIndex}/${totalPointsObjects + totalGSplatsObjects}`
  );
}
