// Main application class for the Luxar scene player

import * as THREE from 'three';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { InputHandler } from '../input/input-handler';
import { RenderingControls } from '../ui/rendering-controls';
import { cleanupUI, clearError, showHelpOverlay } from '../ui/helpers';
import { config } from '../config';
import { DatasetBrowser } from '../ui/dataset-browser';
import { log, Modules } from '../utils/log';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { AdaptiveDPRManager } from '../rendering/adaptive-dpr-manager';
import { ResolutionIndicator } from '../ui/components/resolution-indicator';
import { SceneLoaderManager, getSceneLoader } from '../data/scene-loader-manager';
import { ScaleBar } from '../ui/components/scale-bar';
import { ColormapLegend } from '../ui/components/colormap-legend';
import { RecordingPanel } from '../ui/recording-panel';
import { LayersPanel } from '../ui/layers';
import { ThemeManager } from '../themes/theme-manager';
import type { ZarrViewerConfig } from '../types/zarr';
import { OverlayManager } from '../ui/overlay-manager';

export class LuxarApp {
  private sceneManager!: SceneManager;
  private animationController!: AnimationController;
  private inputHandler!: InputHandler;
  private renderingControls!: RenderingControls;
  private adaptiveDPRManager!: AdaptiveDPRManager;
  private resolutionIndicator!: ResolutionIndicator;
  private datasetBrowser?: DatasetBrowser;
  private scaleBar?: ScaleBar;
  private colormapLegend?: ColormapLegend;
  private recordingPanel?: RecordingPanel;
  private layersPanel?: LayersPanel;
  private overlayManager?: OverlayManager;
  private isInitialized = false;
  private boundCleanup: (() => void) | null = null;
  private boundFocusHandler: (() => void) | null = null;
  private boundVisibilityHandler: (() => void) | null = null;
  private boundDatasetBrowserHandler: (() => void) | null = null;

  /**
   * Initialize the complete Luxar application.
   *
   * Sets up the complete visualization pipeline including:
   * - WebGL renderer and scene manager
   * - Animation loop with post-processing
   * - Input handling (keyboard/mouse)
   * - UI controls (dimension sliders, rendering settings)
   * - Data loading with spatial indexing
   *
   * The initialization sequence is carefully ordered to ensure the
   * animation loop starts BEFORE data loading, providing visual feedback
   * even during long load operations.
   *
   * @param src - URL or path to the Zarr dataset. Can be:
   *              - HTTP URL: 'https://example.com/data.zarr'
   *              - Directory path ending with '/': Shows dataset browser
   *              - Omitted: Uses config.defaultZarrPath
   *              - Query params supported: '?no-cache', '?cache-debug', '?clear-cache',
   *                '?debug', '?no-prefetch'
   *
   * @returns Promise that resolves when initialization is complete and
   *          dataset loading has started (may still be loading in background).
   *          Does NOT wait for all chunks to load.
   *
   * @throws {Error} If WebGL is not supported by browser
   * @throws {Error} If scene manager initialization fails
   * @throws {Error} Dataset loading errors are caught and displayed to user
   *
   * @example
   * ```typescript
   * // Basic initialization with URL
   * const app = new LuxarApp();
   * await app.init('https://example.com/cells.zarr');
   * // App is now running, data loading in background
   * ```
   *
   * @example
   * ```typescript
   * // Show dataset browser
   * const app = new LuxarApp();
   * await app.init('https://example.com/datasets/');
   * // User can browse and select datasets
   * ```
   *
   * @example
   * ```typescript
   * // With error handling
   * const app = new LuxarApp();
   * try {
   *   await app.init(datasetUrl);
   *   console.log('✅ Luxar initialized successfully');
   * } catch (error) {
   *   console.error('❌ Initialization failed:', error);
   *   // Fallback or retry logic
   * }
   * ```
   *
   * @see {@link SceneManager} for rendering pipeline setup
   * @see README.md - initialization sequence section for detailed init flow
   */
  async init(src?: string): Promise<void> {
    if (this.isInitialized) {
      log.info(Modules.APP, 'Re-initializing app (cleaning up previous state)');
      this.cleanup();
    }

    try {
      // Inform users about expected console messages
      log.info(
        Modules.LUXAR,
        'Note: You may see 404 errors for optional features like spatial indexes and array attributes.'
      );
      log.info(
        Modules.LUXAR,
        'These are expected and do not indicate a problem - the app checks for optional features that may not exist.'
      );

      // Use provided source or default
      const sceneSrc = src ?? config.defaultZarrPath;

      // Initialize scene manager first
      this.sceneManager = new SceneManager();
      await this.sceneManager.init();

      // Initialize animation controller with HDR post-processing
      this.animationController = new AnimationController(
        this.sceneManager.controls,
        this.sceneManager.postProcessing
      );

      // Set up per-frame callback for dynamic clipping plane updates
      // Uses unique ID so it won't conflict with other per-frame callbacks (e.g., dimension animation)
      this.animationController.addPerFrameCallback('dynamic-clipping', () => {
        this.sceneManager.updateDynamicClippingPlanes();
      });

      // Initialize adaptive DPR manager for dynamic resolution scaling
      this.adaptiveDPRManager = new AdaptiveDPRManager();
      this.adaptiveDPRManager.setRenderer(this.sceneManager);
      this.animationController.setAdaptiveDPRManager(this.adaptiveDPRManager);

      // Initialize resolution indicator and connect to DPR manager
      this.resolutionIndicator = new ResolutionIndicator();
      // Display target FPS rounded up from maxFPS (58 → 60) since targetFPS (55) is a hysteresis threshold
      const displayTargetFPS = Math.ceil(config.adaptiveDPR.maxFPS / 5) * 5;
      this.resolutionIndicator.setTargetFPS(displayTargetFPS);
      this.adaptiveDPRManager.setOnDPRChangeCallback((dpr, isReducedResolution) => {
        if (isReducedResolution) {
          this.resolutionIndicator.show(dpr);
        } else {
          // Reset the indicator so it can show again on next reduced resolution mode activation
          this.resolutionIndicator.reset();
        }
      });

      // Initialize input handler
      this.inputHandler = new InputHandler(this.sceneManager, this.animationController);
      this.inputHandler.init();

      // Initialize rendering controls
      this.renderingControls = new RenderingControls(
        this.sceneManager.postProcessing,
        this.sceneManager
      );

      // Connect rendering controls to animation controller
      this.renderingControls.setAnimationController(this.animationController);

      // Connect rendering controls to adaptive DPR manager for performance UI
      this.renderingControls.setAdaptiveDPRManager(this.adaptiveDPRManager);

      // Connect rendering controls to input handler
      this.inputHandler.setRenderingControls(this.renderingControls);

      // Initialize recording panel (screenshot/video capture)
      this.recordingPanel = new RecordingPanel(this.sceneManager, this.animationController);
      this.recordingPanel.setPanelStateCallbacks(
        () => this.getPanelVisibilityStates(),
        (states) => this.restorePanelVisibilityStates(states)
      );
      this.recordingPanel.setAdaptiveDPRManager(this.adaptiveDPRManager);
      this.inputHandler.setRecordingPanel(this.recordingPanel);

      // Initialize layers panel (per-node controls)
      this.layersPanel = new LayersPanel(document.body, this.animationController);
      this.inputHandler.setLayersPanel(this.layersPanel);

      // Start animation loop first to ensure background is rendered
      this.animationController.startAnimation();

      // Check if source might be a directory (for navigation)
      if (await this.shouldShowBrowser(sceneSrc)) {
        // Show dataset browser for directory navigation
        try {
          this.showDatasetBrowser();
        } catch (error) {
          log.warning(
            Modules.APP,
            'Dataset browser initialization had issues, but browser is shown:',
            error
          );
          // Browser is shown even if navigation fails - user can use manual entry
        }
      } else {
        // Load scene data directly
        await this.loadDataset(sceneSrc);
      }

      // Setup cleanup on page unload
      this.setupCleanup();

      // Setup dataset browser keyboard shortcut
      this.setupDatasetBrowserShortcut();

      // Setup window focus handling to trigger render on focus
      this.setupFocusHandling();

      // Expose debug interface for testing and AI-assisted development
      this.setupDebugInterface();

      this.isInitialized = true;
    } catch (error) {
      log.error(Modules.APP, 'Failed to initialize Luxar app:', error);
      // Don't call cleanup() here as it removes error messages that were just displayed
      // The error UI should remain visible to inform the user
      throw error;
    }
  }

  /**
   * Check if we should show the dataset browser
   */
  private async shouldShowBrowser(src: string): Promise<boolean> {
    // If no source or empty string, show browser immediately
    if (!src || src.trim() === '') {
      return true;
    }

    // If it's a directory URL (ends with /), show browser
    if (src.endsWith('/')) {
      return true;
    }

    // Check if it's a Zarr dataset by looking for zarr metadata files
    // Try both v2 (.zgroup) and v3 (zarr.json) formats
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const zarrChecks = [
      fetch(src + '/.zgroup', { method: 'HEAD', signal: controller.signal }),
      fetch(src + '/.zattrs', { method: 'HEAD', signal: controller.signal }),
      fetch(src + '/zarr.json', { method: 'HEAD', signal: controller.signal }),
    ];

    try {
      // Short-circuit: return as soon as any probe confirms zarr metadata exists.
      // This avoids waiting for the zarr.json 404 on v2 stores (~200ms on slow networks).
      await Promise.any(
        zarrChecks.map((p) =>
          p.then((r) => {
            if (!r.ok) throw new Error('not ok');
            return r;
          })
        )
      );
      return false; // At least one zarr metadata file exists — load directly
    } catch {
      // All probes failed or errored — likely a directory, show browser
    } finally {
      clearTimeout(timeoutId);
    }

    return true;
  }

  /**
   * Show the dataset browser UI
   */
  private showDatasetBrowser(): void {
    // Close existing browser if any
    if (this.datasetBrowser) {
      return; // Browser already open
    }

    // Clear any existing error messages when opening the browser
    clearError();

    this.datasetBrowser = new DatasetBrowser({
      container: document.body,
      onDatasetSelect: async (fullUrl: string) => {
        // The browser now passes full URLs directly, preserving directory context
        // Strip any trailing slashes to ensure consistent URL format
        const cleanUrl = fullUrl.replace(/\/+$/, '');

        // Update URL parameter
        const params = new URLSearchParams(window.location.search);
        params.set('src', cleanUrl);
        window.history.replaceState({}, '', `${window.location.pathname}?${params}`);

        // Load the dataset
        await this.loadDataset(cleanUrl);
      },
      onClose: () => {
        this.datasetBrowser = undefined;
      },
    });
  }

  /**
   * Load a dataset and initialize UI
   */
  private async loadDataset(src: string): Promise<void> {
    // Clear any existing dimension UI
    this.inputHandler.clearDimensionUI();

    // Note: Monitor cleanup is handled by SceneLoader.loadScene() which calls
    // monitor.disconnectAllLoaders() when loading a new scene

    // Set scene ID for rendering controls persistence BEFORE loading scene
    // This ensures saved settings (like HDR intensity) are applied before materials are created
    this.renderingControls.setSceneId(src);

    // Load scene data (animation loop will continue even if this fails)
    await this.sceneManager.loadSceneData(src);

    // Pass zarr viewer_config to rendering controls (available after scene loads).
    // If no localStorage settings exist for this scene, apply zarr defaults.
    const viewerConfig = this.sceneManager.getSceneViewerConfig();
    this.renderingControls.setZarrViewerConfig(viewerConfig);
    if (!this.renderingControls.hasStoredSettings() && viewerConfig) {
      this.renderingControls.applyZarrDefaults();
    }

    // Update fly speed slider range and value based on scene scale
    this.renderingControls.updateSceneScale();

    // Initialize UI components that depend on loaded scene data
    this.inputHandler.initDimensionSliders();
    this.initScaleBar();

    const sceneLoader = getSceneLoader('default');
    if (sceneLoader?.sceneGraph && this.layersPanel) {
      const root = this.sceneManager.scene.children.find((c) => c.name === 'LuxarScene');
      if (root) {
        this.layersPanel.initFromScene(root as THREE.Group, sceneLoader.sceneGraph);
      }
    }

    // Initialize colormap legend after layers panel (needs layer state)
    if (this.layersPanel) {
      this.initColormapLegend();
    }

    // Initialize overlays (screen-space annotations from zarr)
    await this.initOverlays();

    // Apply zarr viewer_config: UI visibility, theme, dimension state, animation
    this.applyViewerConfigState(viewerConfig);

    // Trigger animation to ensure scene is rendered immediately
    this.animationController.startAnimation();
  }

  /**
   * Apply zarr viewer_config state that isn't handled by RenderingControls.
   *
   * RenderingControls handles the 47 rendering settings (bloom, AO, AA, etc.).
   * This method handles everything else: UI panel visibility, theme,
   * dimension navigation state, and animation state.
   *
   * Only explicitly set fields (not undefined) are applied — unset fields
   * preserve the viewer's built-in defaults.
   */
  private applyViewerConfigState(viewerConfig: ZarrViewerConfig | undefined): void {
    if (!viewerConfig) return;

    // --- UI panel visibility ---
    const ui = viewerConfig.ui;
    if (ui) {
      if (ui.show_help === true) showHelpOverlay();
      if (ui.show_rendering_controls === true) this.renderingControls.show();
      if (ui.show_rendering_controls === false) this.renderingControls.hide();
      if (ui.show_performance_monitor === true) {
        this.animationController.performanceStats?.show();
      }
      if (ui.show_dimensions === true) {
        this.inputHandler.showDimensionSliders();
      }
      if (ui.show_scale_bar === true && this.scaleBar) this.scaleBar.show();
      if (ui.show_scale_bar === false && this.scaleBar) this.scaleBar.hide();
      if (ui.show_layers === true && this.layersPanel) this.layersPanel.show();
      if (ui.show_layers === false && this.layersPanel) this.layersPanel.hide();
      if (ui.show_overlays === true && this.overlayManager) this.overlayManager.show();
      if (ui.show_overlays === false && this.overlayManager) this.overlayManager.hide();
    }

    // --- Theme ---
    if (viewerConfig.theme) {
      ThemeManager.getInstance().setTheme(viewerConfig.theme);
    }

    // --- Dimension navigation state ---
    if (viewerConfig.dimensions?.current_step) {
      for (let i = 0; i < viewerConfig.dimensions.current_step.length; i++) {
        sceneDimsManager.setDimensionValue(i, viewerConfig.dimensions.current_step[i]);
      }
    }
  }

  /**
   * Initialize or recreate the scale bar overlay.
   * Creates a ScaleBar component and registers a per-frame callback
   * to update it as the camera moves.
   */
  private initScaleBar(): void {
    // Dispose previous instance if reloading
    if (this.scaleBar) {
      this.animationController.removePerFrameCallback('scale-bar');
      this.scaleBar.dispose();
    }

    this.scaleBar = new ScaleBar({
      camera: this.sceneManager.camera,
      controls: this.sceneManager.controls,
      canvas: this.sceneManager.renderer.domElement,
      targetWidthPx: config.ui.scaleBar.targetWidthPx,
      position: config.ui.scaleBar.position,
    });

    // Register per-frame update for live camera tracking
    this.animationController.addPerFrameCallback('scale-bar', () => {
      this.scaleBar?.update();
    });

    // Wire to input handler for keyboard toggle
    this.inputHandler.setScaleBar(this.scaleBar);
  }

  /**
   * Initialize the colormap legend overlay.
   * Shows per-layer colormap gradients with names and data ranges.
   */
  private initColormapLegend(): void {
    if (this.colormapLegend) {
      this.colormapLegend.dispose();
    }

    if (!this.layersPanel) return;

    try {
      this.colormapLegend = new ColormapLegend({
        layerState: this.layersPanel.layerState,
      });

      // Wire to input handler for keyboard toggle
      this.inputHandler.setColormapLegend(this.colormapLegend);
    } catch {
      // ColormapLegend requires DOM; may fail in test environments
    }
  }

  /**
   * Initialize screen-space overlays from zarr metadata.
   * Creates an OverlayManager if the loaded scene contains overlays.
   */
  private async initOverlays(): Promise<void> {
    // Dispose previous overlay manager if reloading
    if (this.overlayManager) {
      this.overlayManager.dispose();
      this.overlayManager = undefined;
    }

    const root = this.sceneManager.scene?.children?.find(
      (c) => c.name === 'LuxarScene'
    ) as THREE.Group | undefined;

    const overlayConfigs = root?.userData?.overlayConfigs;
    const zarrBaseUrl = root?.userData?.zarrBaseUrl;

    if (overlayConfigs?.length > 0 && zarrBaseUrl) {
      this.overlayManager = new OverlayManager();
      await this.overlayManager.loadOverlays(overlayConfigs, zarrBaseUrl);
      this.inputHandler.setOverlayManager(this.overlayManager);
    }
  }

  /**
   * Setup cleanup on page unload
   */
  private setupCleanup(): void {
    this.boundCleanup = this.cleanup.bind(this);
    window.addEventListener('beforeunload', this.boundCleanup);
  }

  /**
   * Setup keyboard shortcut for opening dataset browser
   */
  private setupDatasetBrowserShortcut(): void {
    this.boundDatasetBrowserHandler = () => {
      if (!this.datasetBrowser) {
        this.showDatasetBrowser();
      }
    };
    window.addEventListener('open-dataset-browser', this.boundDatasetBrowserHandler);
  }

  /**
   * Setup window focus handling to trigger render on focus
   * This prevents stale renders when switching between windows/tabs
   */
  private setupFocusHandling(): void {
    this.boundFocusHandler = () => {
      // Suppress focus-triggered renders during recording — they can interfere
      // with the deterministic capture loop or cause resize side effects
      if (this.recordingPanel?.isCurrentlyRecording()) return;
      this.animationController.startAnimation();
      log.info(Modules.LUXAR, 'Window focused - triggering render refresh');
    };
    this.boundVisibilityHandler = () => {
      // Don't stop animation during recording (offline capture needs the loop alive)
      if (this.recordingPanel?.isCurrentlyRecording()) return;
      if (document.hidden) {
        this.animationController.stopAnimation();
        log.info(Modules.LUXAR, 'Document hidden - stopping animation to save resources');
      } else {
        this.animationController.startAnimation();
        log.info(Modules.LUXAR, 'Document became visible - resuming animation');
      }
    };
    window.addEventListener('focus', this.boundFocusHandler);
    document.addEventListener('visibilitychange', this.boundVisibilityHandler);
  }

  /**
   * Setup debug interface for testing and AI-assisted development
   *
   * This extends the existing debug interface (created in main.ts) with
   * runtime components that are only available after initialization:
   * - Three.js scene, camera, renderer
   * - Controls and animation state
   * - Helper functions for testing
   *
   * Preserves existing properties (app, consoleInterceptor, version) from main.ts
   *
   * Only enabled when ?debug URL parameter is present
   */
  private setupDebugInterface(): void {
    // Check if debug mode is enabled via URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const debugEnabled = urlParams.has('debug');

    if (!debugEnabled) {
      return;
    }

    log.info(Modules.LUXAR, 'Extending debug interface with runtime components');

    // Extend existing debug interface (preserve app, consoleInterceptor, version from main.ts)
    const existing = (window as any).__luxarDebug || {};

    (window as any).__luxarDebug = {
      // Preserve existing properties from main.ts
      ...existing,

      // Add runtime components (only available after initialization)
      scene: this.sceneManager.scene,
      camera: this.sceneManager.camera,
      renderer: this.sceneManager.renderer,
      controls: this.sceneManager.controls,
      postProcessing: this.sceneManager.postProcessing,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      recordingPanel: this.recordingPanel,
      sceneDimsManager: sceneDimsManager,
      app: this,

      // Helper function to get current state snapshot
      getState: () => {
        const scene = this.sceneManager.scene;

        // Count points across all point clouds
        let totalPoints = 0;
        const pointClouds: any[] = [];

        // Count gsplats across all gsplat meshes
        let totalGSplats = 0;
        const gsplatMeshes: any[] = [];

        scene.traverse((object) => {
          if (object.type === 'Points') {
            const geometry = (object as any).geometry;
            // Use drawRange.count if set (GPU buffer pool uses drawRange to limit rendering)
            // Fall back to position.count for geometries without drawRange
            const drawRangeCount = geometry?.drawRange?.count;
            const bufferCount = geometry?.attributes?.position?.count || 0;
            // Infinity means "draw all", so use buffer count in that case
            const pointCount =
              drawRangeCount !== undefined && drawRangeCount !== Infinity
                ? Math.min(drawRangeCount, bufferCount)
                : bufferCount;
            totalPoints += pointCount;

            pointClouds.push({
              name: object.name || 'unnamed',
              pointCount,
              visible: object.visible,
              hasColors: !!geometry?.attributes?.color,
              hasRadii: !!geometry?.attributes?.radius,
              hasSharpness: !!geometry?.attributes?.sharpness,
            });
          }

          // Count gsplat instances (Mesh with InstancedBufferGeometry and nodeType 'gsplats')
          if (
            object instanceof THREE.Mesh &&
            (object as any).userData?.nodeType === 'gsplats' &&
            object.geometry instanceof THREE.InstancedBufferGeometry
          ) {
            const splatCount = (object.geometry as THREE.InstancedBufferGeometry).instanceCount;
            totalGSplats += splatCount;
            gsplatMeshes.push({
              name: object.name || 'unnamed',
              splatCount,
              visible: object.visible,
            });
          }
        });

        // Get dimensions from sceneDimsManager
        const dims = sceneDimsManager.getDims();
        const dimensionsInfo = dims
          ? { ndim: dims.ndim, displayed: dims.displayed, currentStep: dims.currentStep }
          : null;

        return {
          totalPoints,
          totalGSplats,
          totalElements: totalPoints + totalGSplats,
          pointClouds,
          gsplatMeshes,
          dimensions: dimensionsInfo,
          camera: {
            position: {
              x: this.sceneManager.camera.position.x,
              y: this.sceneManager.camera.position.y,
              z: this.sceneManager.camera.position.z,
            },
            fov: this.sceneManager.currentFov,
          },
          // Keep legacy cameraPosition for backward compatibility
          cameraPosition: {
            x: this.sceneManager.camera.position.x,
            y: this.sceneManager.camera.position.y,
            z: this.sceneManager.camera.position.z,
          },
          cameraFov: this.sceneManager.currentFov,
          isAnimating: this.animationController.isActive,
          initialized: this.isInitialized,
        };
      },

      // Helper to trigger a single frame render (for stable screenshots)
      renderOnce: () => {
        this.animationController.startAnimation();
      },

      // Helper to get scene loader manager (for cache inspection)
      getSceneLoader: () => {
        return SceneLoaderManager.getInstance();
      },

      // Cache-specific helpers
      cache: {
        // Get current cache statistics (L0, L1, L2)
        getStats: () => {
          const manager = SceneLoaderManager.getInstance();
          const loader = manager.getDefaultLoader();
          if (!loader) {
            return { error: 'No active loader found' };
          }

          // Get L1/L2 stats from caching store
          const l1l2Stats = (loader as any).cachingStore
            ? (loader as any).cachingStore.getStats()
            : { l1: null, l2: null };

          // Get L0 stats from decompressed chunk cache
          const l0Cache = (loader as any).l0Cache;
          const l0Stats = l0Cache ? l0Cache.getStats() : null;

          return {
            l0: l0Stats,
            l1: l1l2Stats.l1,
            l2: l1l2Stats.l2,
          };
        },

        // List all cached datasets
        listDatasets: () => {
          const manager = SceneLoaderManager.getInstance();
          const loader = manager.getDefaultLoader();
          if (!loader || !(loader as any).cachingStore) {
            return { error: 'No active cache found' };
          }
          return (loader as any).cachingStore.listDatasets();
        },

        // Clear L0 decompressed chunk cache only
        clearL0: () => {
          const manager = SceneLoaderManager.getInstance();
          const loader = manager.getDefaultLoader();
          const l0Cache = loader ? (loader as any).l0Cache : null;
          if (!l0Cache) {
            log.warning(Modules.CACHE, 'No L0 cache found');
            return;
          }
          l0Cache.clear();
          log.info(Modules.CACHE, 'L0 cache cleared');
        },

        // Clear L1 cache only
        clearL1: () => {
          const manager = SceneLoaderManager.getInstance();
          const loader = manager.getDefaultLoader();
          if (!loader || !(loader as any).cachingStore) {
            log.warning(Modules.CACHE, 'No active cache found');
            return;
          }
          (loader as any).cachingStore.clearL1();
          log.info(Modules.CACHE, 'L1 cache cleared');
        },

        // Clear L2 cache only
        clearL2: async () => {
          const manager = SceneLoaderManager.getInstance();
          const loader = manager.getDefaultLoader();
          if (!loader || !(loader as any).cachingStore) {
            log.warning(Modules.CACHE, 'No active cache found');
            return;
          }
          await (loader as any).cachingStore.clearL2();
          log.info(Modules.CACHE, 'L2 cache cleared');
        },

        // Clear all caches (L0, L1, L2)
        clearAll: async () => {
          const manager = SceneLoaderManager.getInstance();
          const loader = manager.getDefaultLoader();
          if (!loader) {
            log.warning(Modules.CACHE, 'No active loader found');
            return;
          }

          // Clear L0 decompressed chunk cache
          const l0Cache = (loader as any).l0Cache;
          if (l0Cache) {
            l0Cache.clear();
          }

          // Clear L1/L2 caching store
          if ((loader as any).cachingStore) {
            await (loader as any).cachingStore.clearAll();
          }

          log.info(Modules.CACHE, 'All caches cleared (L0, L1, L2)');
        },
      },

      // Mark that runtime components are now available
      runtimeReady: true,
    };

    // Log available debug commands
    log.info(Modules.LUXAR, 'Debug interface ready:');
    log.info(Modules.LUXAR, '  __luxarDebug.getState() - Get current state snapshot');
    log.info(Modules.LUXAR, '  __luxarDebug.renderOnce() - Trigger single frame render');
    log.info(Modules.LUXAR, '  __luxarDebug.scene - Access THREE.js scene');
    log.info(Modules.LUXAR, '  __luxarDebug.camera - Access camera');
    log.info(Modules.LUXAR, '  __luxarDebug.app - Access LuxarApp instance');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.getStats() - Get cache statistics (L0, L1, L2)');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL0() - Clear L0 decompressed chunk cache');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL1() - Clear L1 memory cache');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearL2() - Clear L2 OPFS cache');
    log.info(Modules.LUXAR, '  __luxarDebug.cache.clearAll() - Clear all caches (L0, L1, L2)');
  }

  /**
   * Get application components (for testing or advanced usage)
   */
  get components() {
    return {
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
      renderingControls: this.renderingControls,
      adaptiveDPRManager: this.adaptiveDPRManager,
    };
  }

  /**
   * Get initialization state
   */
  get initialized(): boolean {
    return this.isInitialized;
  }

  /**
   * Clean up all application resources
   */
  cleanup(): void {
    try {
      // Stop animation first
      if (this.animationController) {
        this.animationController.dispose();
      }

      // Clean up adaptive DPR manager
      if (this.adaptiveDPRManager) {
        this.adaptiveDPRManager.dispose();
      }

      // Clean up resolution indicator
      if (this.resolutionIndicator) {
        this.resolutionIndicator.dispose();
      }

      // Clean up scale bar
      if (this.scaleBar) {
        this.scaleBar.dispose();
        this.scaleBar = undefined;
      }

      // Clean up colormap legend
      if (this.colormapLegend) {
        this.colormapLegend.dispose();
        this.colormapLegend = undefined;
      }

      // Clean up recording panel
      if (this.recordingPanel) {
        this.recordingPanel.dispose();
        this.recordingPanel = undefined;
      }

      // Clean up layers panel
      if (this.layersPanel) {
        this.layersPanel.dispose();
        this.layersPanel = undefined;
      }

      // Clean up overlay manager
      if (this.overlayManager) {
        this.overlayManager.dispose();
        this.overlayManager = undefined;
      }

      // Clean up input handlers
      if (this.inputHandler) {
        this.inputHandler.dispose();
      }

      // Clean up rendering controls
      if (this.renderingControls) {
        this.renderingControls.dispose();
      }

      // Clean up scene resources
      if (this.sceneManager) {
        this.sceneManager.dispose();
      }

      // Clean up UI resources
      cleanupUI();

      // Remove focus and visibility listeners
      if (this.boundFocusHandler) {
        window.removeEventListener('focus', this.boundFocusHandler);
        this.boundFocusHandler = null;
      }
      if (this.boundVisibilityHandler) {
        document.removeEventListener('visibilitychange', this.boundVisibilityHandler);
        this.boundVisibilityHandler = null;
      }
      if (this.boundDatasetBrowserHandler) {
        window.removeEventListener('open-dataset-browser', this.boundDatasetBrowserHandler);
        this.boundDatasetBrowserHandler = null;
      }

      // Remove beforeunload listener with stored reference
      if (this.boundCleanup) {
        window.removeEventListener('beforeunload', this.boundCleanup);
        this.boundCleanup = null;
      }

      this.isInitialized = false;
    } catch (error) {
      log.error(Modules.LUXAR, 'Error during cleanup:', error);
    }
  }

  /**
   * Get visibility states of all UI panels for save/restore during recording.
   */
  private getPanelVisibilityStates(): Map<string, boolean> {
    const states = new Map<string, boolean>();
    states.set('renderingControls', this.renderingControls?.isVisible() ?? false);
    states.set('recordingPanel', this.recordingPanel?.isVisible() ?? false);
    return states;
  }

  /**
   * Restore UI panel visibility from a saved state map.
   */
  private restorePanelVisibilityStates(states: Map<string, boolean>): void {
    if (states.get('renderingControls')) {
      this.renderingControls?.show();
    } else {
      if (this.renderingControls?.isVisible()) this.renderingControls.hide();
    }
    if (states.get('recordingPanel')) {
      this.recordingPanel?.show();
    } else {
      if (this.recordingPanel?.isVisible()) this.recordingPanel.hide();
    }
  }
}
