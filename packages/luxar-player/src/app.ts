// Main application class for the Luxar scene player

import { SceneManager } from "./scene-manager";
import { AnimationController } from "./animation-controller";
import { InputHandler } from "./input-handler";
import { cleanupUI } from "./ui";
import { CONFIG } from "./config";

export class LuxarApp {
  private sceneManager!: SceneManager;
  private animationController!: AnimationController;
  private inputHandler!: InputHandler;
  private isInitialized = false;

  /**
   * Initialize the complete application
   */
  async init(src?: string): Promise<void> {
    try {
      // Use provided source or default
      const sceneSrc = src ?? CONFIG.DEFAULT_ZARR_PATH;

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
      this.inputHandler = new InputHandler(
        this.sceneManager,
        this.animationController
      );
      this.inputHandler.init();

      // Start animation loop first to ensure background is rendered
      this.animationController.startAnimation();

      // Load scene data (animation loop will continue even if this fails)
      await this.sceneManager.loadSceneData(sceneSrc);

      // Setup cleanup on page unload
      this.setupCleanup();

      this.isInitialized = true;
      
    } catch (error) {
      console.error('Failed to initialize Luxar app:', error);
      // Don't call cleanup() here as it removes error messages that were just displayed
      // The error UI should remain visible to inform the user
      throw error;
    }
  }

  /**
   * Setup cleanup on page unload
   */
  private setupCleanup(): void {
    window.addEventListener('beforeunload', this.cleanup.bind(this));
  }

  /**
   * Get application components (for testing or advanced usage)
   */
  get components() {
    return {
      sceneManager: this.sceneManager,
      animationController: this.animationController,
      inputHandler: this.inputHandler,
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
      console.error('Error during cleanup:', error);
    }
  }
}