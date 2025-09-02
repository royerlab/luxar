// Main application class for the Luxar scene player

import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { InputHandler } from '../input/input-handler';
import { RenderingControls } from '../ui/rendering-controls';
import { cleanupUI, clearError } from '../ui/helpers';
import { config } from '../config';
import { DatasetBrowser } from '../ui/dataset-browser';
import { log, Modules, LogEmoji } from '../utils/log';

export class LuxarApp {
  private sceneManager!: SceneManager;
  private animationController!: AnimationController;
  private inputHandler!: InputHandler;
  private renderingControls!: RenderingControls;
  private datasetBrowser?: DatasetBrowser;
  private isInitialized = false;

  /**
   * Initialize the complete application
   */
  async init(src?: string): Promise<void> {
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
        this.sceneManager.renderer,
        this.sceneManager.scene,
        this.sceneManager.camera,
        this.sceneManager.controls,
        this.sceneManager.postProcessing
      );

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

      // Connect rendering controls to input handler
      this.inputHandler.setRenderingControls(this.renderingControls);

      // Start animation loop first to ensure background is rendered
      this.animationController.startAnimation();

      // Check if source might be a directory (for navigation)
      if (await this.shouldShowBrowser(sceneSrc)) {
        // Show dataset browser for directory navigation
        this.showDatasetBrowser();
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
    // If no source or it's a directory URL (ends with /), show browser
    if (!src || src.endsWith('/')) {
      return true;
    }

    // Check if it's a Zarr dataset
    try {
      const response = await fetch(src + '/.zgroup', { method: 'HEAD' });
      if (response.ok) {
        return false; // It's a Zarr dataset, load directly
      }
    } catch {
      // Ignore errors, proceed with check
    }

    // Check if it looks like a directory (no file extension)
    const hasExtension = src.split('/').pop()?.includes('.');
    if (!hasExtension) {
      return true; // Likely a directory
    }

    return false;
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
      onDatasetSelect: async (path: string) => {
        // Construct full URL
        const params = new URLSearchParams(window.location.search);
        const currentSrc = params.get('src') || '';

        let baseUrl: string;
        try {
          const url = new URL(currentSrc);
          baseUrl = url.origin;
        } catch {
          baseUrl = window.location.origin;
        }

        // Ensure proper path joining without double slashes
        const cleanPath = path.startsWith('/') ? path : '/' + path;
        const fullUrl = baseUrl + cleanPath;

        // Update URL parameter
        params.set('src', fullUrl);
        window.history.replaceState({}, '', `${window.location.pathname}?${params}`);

        // Load the dataset
        await this.loadDataset(fullUrl);
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

    // Clear cache when loading new scene (new architecture)
    import('../data/scene-loader-manager').then(({ SceneLoaderManager }) => {
      const sceneLoader = SceneLoaderManager.getInstance().getDefaultLoader();
      if (sceneLoader) {
        sceneLoader.clearCaches();
        log.custom(LogEmoji.DELETE, Modules.LUXAR, 'Cleared data cache for new scene');
      }
    });

    // Note: Monitor cleanup is handled by SceneLoader.loadScene() which calls
    // monitor.disconnectAllLoaders() when loading a new scene

    // Set scene ID for rendering controls persistence BEFORE loading scene
    // This ensures saved settings (like HDR intensity) are applied before materials are created
    this.renderingControls.setSceneId(src);

    // Load scene data (animation loop will continue even if this fails)
    await this.sceneManager.loadSceneData(src);

    // Initialize dimension sliders for nD data
    this.inputHandler.initDimensionSliders();

    // Trigger animation to ensure scene is rendered immediately
    this.animationController.startAnimation();
  }

  /**
   * Setup cleanup on page unload
   */
  private setupCleanup(): void {
    window.addEventListener('beforeunload', this.cleanup.bind(this));
  }

  /**
   * Setup keyboard shortcut for opening dataset browser
   */
  private setupDatasetBrowserShortcut(): void {
    window.addEventListener('open-dataset-browser', () => {
      if (!this.datasetBrowser) {
        this.showDatasetBrowser();
      }
    });
  }

  /**
   * Setup window focus handling to trigger render on focus
   * This prevents stale renders when switching between windows/tabs
   */
  private setupFocusHandling(): void {
    // Trigger a render when window gains focus
    window.addEventListener('focus', () => {
      // Start animation briefly to ensure fresh render
      this.animationController.startAnimation();
      log.info(Modules.LUXAR, 'Window focused - triggering render refresh');
    });

    // Also handle visibility change (tab switching)
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) {
        // Document became visible, trigger render
        this.animationController.startAnimation();
        log.info(Modules.LUXAR, 'Document became visible - triggering render refresh');
      }
    });
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

      // Remove beforeunload listener
      window.removeEventListener('beforeunload', this.cleanup.bind(this));

      this.isInitialized = false;
    } catch (error) {
      log.error(Modules.LUXAR, 'Error during cleanup:', error);
    }
  }
}
