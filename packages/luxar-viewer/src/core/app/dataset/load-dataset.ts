import { getSceneLoader } from '../../../data/scene-loader-manager';
import type * as THREE from 'three';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { InputHandler } from '../../../input/input-handler';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { LayersPanel } from '../../../ui/layers';
import type { LoaderConfig } from '../../../data/data-loader-types';
import type { ZarrViewerConfig } from '../../../types/zarr';

/**
 * Load a dataset and initialize the scene-dependent UI in the order
 * the orchestrator originally used. The sequence is part of observable
 * behaviour (e.g. setSceneId must precede loadSceneData so persisted
 * settings reach materials at construction time), so this helper keeps
 * the steps strictly in order.
 *
 * Each per-overlay / per-system init step is supplied as a callback so
 * the helper composes the orchestrator's other delegate methods without
 * the helper needing to know how they're implemented internally.
 */
export interface LoadDatasetPorts {
  inputHandler: InputHandler;
  renderingControls: RenderingControls;
  sceneManager: SceneManager;
  animationController: AnimationController;
  layersPanel: LayersPanel | undefined;
  loaderConfig: LoaderConfig | undefined;
  openCacheStats: boolean;
  disposeOverlays: () => void;
  initScaleBar: () => void;
  initColormapLegend: () => void;
  initOverlays: () => Promise<void>;
  initPicking: () => Promise<void>;
  applyViewerConfigState: (config: ZarrViewerConfig | undefined) => void;
  openCacheStatsView: () => void;
}

export async function loadDataset(src: string, ports: LoadDatasetPorts): Promise<void> {
  // Clear any existing dimension UI
  ports.inputHandler.clearDimensionUI();

  // Dispose previous-scene overlays upfront so they are cleared in lockstep
  // with clearSceneContent() — otherwise a failing scene load leaves the old
  // overlay DOM elements visible on top of an empty canvas.
  ports.disposeOverlays();

  // Note: Monitor cleanup is handled by SceneLoader.loadScene() which calls
  // monitor.disconnectAllLoaders() when loading a new scene

  // Set scene ID for rendering controls persistence BEFORE loading scene
  // This ensures saved settings (like HDR intensity) are applied before materials are created
  ports.renderingControls.setSceneId(src);

  // Load scene data (animation loop will continue even if this fails)
  await ports.sceneManager.loadSceneData(src, ports.loaderConfig);

  // Pass zarr viewer_config to rendering controls (available after scene loads).
  // If no localStorage settings exist for this scene, apply zarr defaults.
  const viewerConfig = ports.sceneManager.getSceneViewerConfig();
  ports.renderingControls.setZarrViewerConfig(viewerConfig);
  if (!ports.renderingControls.hasStoredSettings() && viewerConfig) {
    ports.renderingControls.applyZarrDefaults();
  }

  // Update fly speed slider range and value based on scene scale
  ports.renderingControls.updateSceneScale();

  // Initialize UI components that depend on loaded scene data
  ports.inputHandler.initDimensionSliders();
  ports.initScaleBar();

  const sceneLoader = getSceneLoader('default');
  if (sceneLoader?.sceneGraph && ports.layersPanel) {
    const root = ports.sceneManager.scene.children.find((c) => c.name === 'LuxarScene');
    if (root) {
      ports.layersPanel.initFromScene(root as THREE.Group, sceneLoader.sceneGraph);
    }
  }
  // Notify on-screen affordances (the control rail's Layers button gates its
  // disabled state on the layer count) that layers may have changed. Best-effort
  // — guarded so a stubbed/absent window (unit tests) can't fail the load.
  if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
    window.dispatchEvent(new CustomEvent('luxar-layers-changed'));
  }

  // Initialize colormap legend after layers panel (needs layer state)
  if (ports.layersPanel) {
    ports.initColormapLegend();
  }

  // Initialize overlays (screen-space annotations from zarr)
  await ports.initOverlays();

  // Initialize GPU picking system (if any node has labels)
  await ports.initPicking();

  // Apply zarr viewer_config: UI visibility, theme, dimension state, animation
  ports.applyViewerConfigState(viewerConfig);

  // ?cache-stats: open the data-loading monitor on the Cache tab. The
  // monitor was created during sceneManager.loadSceneData() above, so
  // it's safe to look it up via DataMonitorManager now.
  if (ports.openCacheStats) {
    ports.openCacheStatsView();
  }

  // Trigger animation to ensure scene is rendered immediately
  ports.animationController.startAnimation();
}
