// Input event handling for the Luxar scene player

import { SceneManager } from "./scene-manager";
import { AnimationController } from "./animation-controller";
import { showHelpOverlay, hideHelpOverlay } from "./ui";

export class InputHandler {
  private eventListeners: (() => void)[] = [];

  constructor(
    private sceneManager: SceneManager,
    private animationController: AnimationController
  ) {}

  /**
   * Initialize all event listeners
   */
  init(): void {
    this.setupWindowEvents();
    this.setupControlEvents();
    this.setupUserInteractionEvents();
  }

  /**
   * Setup window-level event listeners
   */
  private setupWindowEvents(): void {
    const onResize = this.onWindowResize.bind(this);
    const onWheel = this.onWheel.bind(this);
    const onKeyDown = this.onKeyDown.bind(this);
    const onKeyUp = this.onKeyUp.bind(this);
    const onFullscreenChange = this.onFullscreenChange.bind(this);

    window.addEventListener('resize', onResize);
    window.addEventListener('wheel', onWheel);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    document.addEventListener('fullscreenchange', onFullscreenChange);

    // Store cleanup functions
    this.eventListeners.push(
      () => window.removeEventListener('resize', onResize),
      () => window.removeEventListener('wheel', onWheel),
      () => window.removeEventListener('keydown', onKeyDown),
      () => window.removeEventListener('keyup', onKeyUp),
      () => document.removeEventListener('fullscreenchange', onFullscreenChange)
    );
  }

  /**
   * Setup control-specific event listeners
   */
  private setupControlEvents(): void {
    const startAnimation = this.animationController.startAnimation;
    
    this.sceneManager.controls.addEventListener('start', startAnimation);
    this.sceneManager.controls.addEventListener('change', startAnimation);

    this.eventListeners.push(
      () => this.sceneManager.controls.removeEventListener('start', startAnimation),
      () => this.sceneManager.controls.removeEventListener('change', startAnimation)
    );
  }

  /**
   * Setup user interaction event listeners
   */
  private setupUserInteractionEvents(): void {
    const startAnimation = this.animationController.startAnimation;
    const canvas = this.sceneManager.renderer.domElement;

    canvas.addEventListener('mousedown', startAnimation);
    canvas.addEventListener('touchstart', startAnimation);

    this.eventListeners.push(
      () => canvas.removeEventListener('mousedown', startAnimation),
      () => canvas.removeEventListener('touchstart', startAnimation)
    );
  }

  /**
   * Handle window resize events
   */
  private onWindowResize(): void {
    this.sceneManager.updateSize();
  }

  /**
   * Handle fullscreen change events
   */
  private onFullscreenChange(): void {
    const canvas = this.sceneManager.renderer.domElement;
    
    if (document.fullscreenElement) {
      // Entering fullscreen - ensure canvas fills the entire screen
      canvas.style.width = '100vw';
      canvas.style.height = '100vh';
      canvas.style.position = 'fixed';
      canvas.style.top = '0';
      canvas.style.left = '0';
      console.log('✓ Entering fullscreen mode');
    } else {
      // Exiting fullscreen - restore normal canvas styling
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.position = '';
      canvas.style.top = '';
      canvas.style.left = '';
      console.log('✓ Exiting fullscreen mode');
    }
    
    // Add small delay to ensure canvas dimensions are updated by browser
    setTimeout(() => {
      // Update canvas size when entering/exiting fullscreen
      this.sceneManager.updateSize();
      
      // Restart animation to ensure smooth transition
      this.animationController.startAnimation();
    }, 100); // 100ms delay to avoid race conditions
  }

  /**
   * Handle mouse wheel events (zoom and FOV control)
   */
  private onWheel(event: WheelEvent): void {
    this.animationController.startAnimation();
    
    if (event.shiftKey) {
      event.preventDefault();
      this.sceneManager.updateFOV(event.deltaY);
    }
  }

  /**
   * Handle key down events
   */
  private onKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case 'Shift':
        this.sceneManager.controls.enableZoom = false;
        break;
        
      case 'h':
      case 'H':
        event.preventDefault();
        this.toggleHelp();
        break;
        
      case 'p':
      case 'P':
        // Shift+P to toggle performance metrics
        if (event.shiftKey) {
          event.preventDefault();
          this.togglePerformanceStats();
        }
        break;
        
      case ' ':
        // Only toggle fullscreen if not focused on a UI element
        if (this.shouldHandleSpaceKey()) {
          event.preventDefault();
          this.toggleFullscreen();
        }
        break;
    }
  }

  /**
   * Handle key up events
   */
  private onKeyUp(event: KeyboardEvent): void {
    if (event.key === 'Shift') {
      this.sceneManager.controls.enableZoom = true;
    }
  }

  /**
   * Toggle help overlay visibility
   */
  private toggleHelp(): void {
    const helpOverlay = document.getElementById('help-overlay');
    if (helpOverlay) {
      hideHelpOverlay();
    } else {
      showHelpOverlay();
    }
  }

  /**
   * Toggle performance statistics display
   */
  private togglePerformanceStats(): void {
    this.animationController.performanceStats.toggle();
  }

  /**
   * Toggle fullscreen mode
   */
  private toggleFullscreen(): void {
    if (!document.fullscreenElement) {
      // Enter fullscreen - target the document element for true fullscreen
      document.documentElement.requestFullscreen().catch(err => {
        console.error('Error attempting to enable fullscreen:', err);
        // Fallback: try the canvas element
        this.sceneManager.renderer.domElement.requestFullscreen().catch(fallbackErr => {
          console.error('Fallback fullscreen also failed:', fallbackErr);
        });
      });
    } else {
      // Exit fullscreen
      document.exitFullscreen().catch(err => {
        console.error('Error attempting to exit fullscreen:', err);
      });
    }
  }

  /**
   * Check if space key should trigger fullscreen
   */
  private shouldHandleSpaceKey(): boolean {
    const activeElement = document.activeElement;
    return activeElement === document.body || 
           activeElement === this.sceneManager.renderer.domElement;
  }

  /**
   * Clean up all event listeners
   */
  dispose(): void {
    this.eventListeners.forEach(cleanup => cleanup());
    this.eventListeners = [];
  }
}