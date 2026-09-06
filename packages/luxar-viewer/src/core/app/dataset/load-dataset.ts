import { getSceneLoader } from '../../../data/scene-loader-manager';
import type * as THREE from 'three';
import type { SceneManager } from '../../../scene/scene-manager';
import type { AnimationController } from '../../../scene/animation/animation-controller';
import type { InputHandler } from '../../../input';
import type { RenderingControls } from '../../../ui/rendering-controls';
import type { LayersPanel } from '../../../ui/layers';
import type { LoaderConfig } from '../../../data/data-loader-types';
import type { ZarrViewerConfig } from '../../../types/zarr';
import { extractEnvironmentConfig } from '../../../config/zarr-bridge/viewer-config-utils';

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
  disposePicking: () => void;
  initScaleBar: () => void;
  initColormapLegend: () => void;
  initOverlays: () => Promise<void>;
  initPicking: () => Promise<void>;
  applyViewerConfigState: (config: ZarrViewerConfig | undefined) => void;
  openCacheStatsView: () => void;
}

export async function loadDataset(src: string, ports: LoadDatasetPorts): Promise<void> {
  ports.sceneManager.environment?.resetForDataset();

  // Clear any existing dimension UI
  ports.inputHandler.clearDimensionUI();

  // Dispose previous-scene overlays upfront so they are cleared in lockstep
  // with clearSceneContent() — otherwise a failing scene load leaves the old
  // overlay DOM elements visible on top of an empty canvas.
  ports.disposeOverlays();

  // Dispose the previous picking session upfront for the same reason:
  // its nodeMap references geometries that clearSceneContent() is about
  // to dispose. If loadSceneData below throws mid-load, a still-alive
  // session (with a live embedder-selection consumer) would keep firing
  // picks against disposed geometries until the next successful load.
  // The end-of-load initPicking remains the (re)creation point.
  ports.disposePicking();

  // Note: Monitor cleanup is handled by SceneLoader.loadScene() which calls
  // monitor.disconnectAllLoaders() when loading a new scene

  // Set scene ID for rendering controls persistence BEFORE loading scene
  // This ensures saved settings (like HDR intensity) are applied before materials are created
  ports.renderingControls.setSceneId(src);
  const applyViewerConfigDefaults = !ports.renderingControls.hasStoredSettings();

  // Load scene data (animation loop will continue even if this fails)
  await ports.sceneManager.loadSceneData(src, ports.loaderConfig, {
    applyViewerConfigFov: applyViewerConfigDefaults,
  });

  // Pass zarr viewer_config to rendering controls (available after scene loads).
  // If no localStorage settings exist for this scene, apply zarr defaults. An
  // authored camera position already carried its resolved FOV during loading.
  const viewerConfig = ports.sceneManager.getSceneViewerConfig();
  ports.renderingControls.setZarrViewerConfig(viewerConfig);
  // The scene environment: authored source (room | scene | hdri) plus any baked map
  // the loader found. Both are inert until a physical material asks for light.
  ports.sceneManager.environment?.configure(
    viewerConfig ? (extractEnvironmentConfig(viewerConfig) ?? null) : null
  );
  ports.sceneManager.environment?.setBaked(ports.sceneManager.getSceneBakedEnvironment());
  if (applyViewerConfigDefaults && viewerConfig) {
    ports.renderingControls.applyZarrDefaults();
  } else if (!applyViewerConfigDefaults) {
    // The scene may have replaced the stored FOV to keep an authored position
    // paired with its lens. Keep panel state and Ctrl+Shift+S export aligned
    // with the live camera before any later settings application can reuse it.
    ports.renderingControls.syncCameraFovState();
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
      // Hand the layers panel an equivalent failed-loads provider over the same
      // live failure set the data monitor reads (each getFailedLoadsProvider()
      // call returns a new object, but all close over the loader's one
      // failedLoaders map), so a node whose loader threw shows a per-row error
      // badge in the always-open panel instead of only in the console /
      // collapsed monitor. After initFromScene: its clear() resets any prior
      // provider first.
      ports.layersPanel.setFailedLoadsProvider(sceneLoader.getFailedLoadsProvider());
      // Per-layer "Frame camera" context-menu action (same late-binding
      // pattern as the provider above).
      ports.layersPanel.setCameraFramer((obj) => ports.sceneManager.fitCameraToObject(obj));
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

  // Request WebGL blend-program warm-up only after the complete dataset setup.
  // Links remain cooperative, but dataset readiness waits until all reachable
  // variants are pinned so later interactions cannot race the initial queue.
  await ports.sceneManager.warmBlendModePrograms();
}
