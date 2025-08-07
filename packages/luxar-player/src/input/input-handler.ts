/**
 * Comprehensive input handling system for nD navigation and scene interaction.
 * 
 * This class provides the complete user interface layer for Luxar, handling:
 * - Keyboard navigation through nD dimensions
 * - Mouse and touch interaction coordination
 * - Fullscreen and FOV control
 * - UI element state management
 * - Dimension slider integration
 * - Performance monitoring controls
 * 
 * The input handler bridges user interactions with the underlying nD visualization
 * system, translating keyboard/mouse events into dimension changes that trigger
 * coordinated updates across all scene objects.
 * 
 * Key interaction patterns:
 * - [ ] keys navigate through dimensions with adaptive step sizes
 * - Number keys (1-9) select which dimension to control
 * - Space bar toggles fullscreen mode
 * - Shift+wheel adjusts field of view
 * - Ctrl+P toggles performance statistics
 * - Ctrl+A toggles advanced rendering controls
 * - H key shows/hides help overlay
 * 
 * The system maintains careful separation between:
 * - Camera controls (handled by THREE.js OrbitControls)
 * - Dimension navigation (handled by scene dimension manager)
 * - UI state management (handled by various UI components)
 * 
 * All navigation events are coordinated through the scene dimension manager
 * to ensure consistent state across all nD objects in the scene.
 */

import * as THREE from 'three';
import { SceneManager } from '../scene/scene-manager';
import { AnimationController } from '../scene/animation-controller';
import { RenderingControls } from '../ui/rendering-controls';
import { showHelpOverlay, hideHelpOverlay } from '../ui/helpers';
import { updatePointCloudSlice } from '../utils/dims-navigation';
import { SimpleDims } from '../types/dims';
import { DimensionSliders } from '../ui/dimension-sliders';
import { sceneDimsManager } from '../scene/scene-dims-manager';

/**
 * Central coordinator for all user input events and nD navigation.
 * 
 * @class InputHandler
 */
export class InputHandler {
  /** Cleanup functions for all registered event listeners */
  private eventListeners: (() => void)[] = [];
  
  /** Optional reference to advanced rendering controls */
  private renderingControls?: RenderingControls;
  
  /** Index of currently selected dimension for keyboard navigation */
  private selectedDimension: number = 0;
  
  /** UI component for interactive dimension sliders */
  private dimensionSliders?: DimensionSliders;

  /**
   * Constructs the input handler with required system dependencies.
   * 
   * @param sceneManager - Scene management system
   * @param animationController - Animation and rendering coordination
   */
  constructor(
    private sceneManager: SceneManager,
    private animationController: AnimationController
  ) {}

  /**
   * Associates rendering controls for advanced UI interactions.
   * 
   * @param controls - Rendering controls interface
   */
  setRenderingControls(controls: RenderingControls): void {
    this.renderingControls = controls;
  }

  /**
   * Initializes all event listeners for user interaction.
   * 
   * This sets up the complete input handling system including keyboard,
   * mouse, touch, and window events. Should be called once during
   * application initialization.
   */
  init(): void {
    this.setupWindowEvents();
    this.setupControlEvents();
    this.setupUserInteractionEvents();
  }

  /**
   * Clear dimension UI and reset dimension manager
   */
  clearDimensionUI(): void {
    // Dispose of existing dimension sliders
    if (this.dimensionSliders) {
      this.dimensionSliders.dispose();
      this.dimensionSliders = undefined;
    }
    
    // Reset the scene dimension manager
    sceneDimsManager.reset();
    
    // Reset selected dimension
    this.selectedDimension = 0;
  }

  /**
   * Initializes dimension navigation UI after scene loading completes.
   * 
   * This method is called after the scene is fully loaded and dimension
   * metadata is available. It sets up:
   * - Scene dimension manager integration
   * - Interactive dimension sliders
   * - Reactive updates for all nD objects
   * - Keyboard navigation targets
   * 
   * The initialization process ensures all nD objects share the same
   * dimensional coordinate system and respond consistently to navigation.
   */
  initDimensionSliders(): void {
    // Initialize scene dims manager
    if (!sceneDimsManager.initFromScene(this.sceneManager.scene)) {
      return; // No nD objects found
    }

    const dims = sceneDimsManager.getDims();
    const dimensionRanges = sceneDimsManager.getDimensionRanges();

    if (!dims || !dimensionRanges) {
      return;
    }

    // Clean up existing sliders if any
    if (this.dimensionSliders) {
      this.dimensionSliders.dispose();
    }

    // Create new dimension sliders
    const dimensionNames = sceneDimsManager.getDimensionNames();
    const dimensionUnits = sceneDimsManager.getDimensionUnits();
    
    this.dimensionSliders = new DimensionSliders({
      container: document.body,
      dims,
      dimensionRanges,
      dimensionNames,
      dimensionUnits,
    });

    // Show sliders only if we have non-displayed dimensions
    this.dimensionSliders.setVisible(sceneDimsManager.hasNonDisplayedDimensions());

    // Listen for dimension changes
    sceneDimsManager.addListener(() => {
      this.updateAllNDPointClouds();
      if (this.dimensionSliders) {
        this.dimensionSliders.update();
      }
      // Trigger animation to render the changes
      this.animationController.startAnimation();
    });
  }


  /**
   * Update all nD point clouds with current dimension values
   */
  private updateAllNDPointClouds(): void {
    const dims = sceneDimsManager.getDims();
    if (!dims) return;

    const nDPointClouds = this.findNDPointClouds();

    nDPointClouds.forEach((points: THREE.Points) => {
      const {
        originalNumPoints,
        originalPositions,
        originalColors,
        originalRadii,
        originalSharpness,
      } = points.userData;

      updatePointCloudSlice(
        points,
        originalPositions,
        originalColors,
        originalRadii,
        originalSharpness,
        dims, // Use shared dims from manager
        originalNumPoints
      );
    });
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
      // Ensure the canvas has full opacity and no filters
      canvas.style.opacity = '1';
      canvas.style.filter = 'none';
      // Ensure document background doesn't interfere
      document.documentElement.style.backgroundColor = '#111111';
      console.log('✓ Entering fullscreen mode');
    } else {
      // Exiting fullscreen - restore normal canvas styling
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.position = '';
      canvas.style.top = '';
      canvas.style.left = '';
      canvas.style.opacity = '';
      canvas.style.filter = '';
      document.documentElement.style.backgroundColor = '';
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

      case 'd':
      case 'D':
        event.preventDefault();
        this.toggleDimensionSliders();
        break;

      case 'o':
      case 'O':
        // O key to open dataset browser
        event.preventDefault();
        window.dispatchEvent(new CustomEvent('open-dataset-browser'));
        break;

      case 'p':
      case 'P':
        // Ctrl+P to toggle performance metrics
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.togglePerformanceStats();
        }
        break;

      case 'a':
      case 'A':
        // Ctrl+A to toggle advanced rendering controls
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.toggleRenderingControls();
        }
        break;

      case ' ':
        // Only toggle fullscreen if not focused on a UI element
        if (this.shouldHandleSpaceKey()) {
          event.preventDefault();
          this.toggleFullscreen();
        }
        break;

      // Dimension navigation
      case '[':
      case ']':
        event.preventDefault();
        this.handleDimensionNavigation(event.key === '[' ? -1 : 1);
        break;

      // Number keys for dimension selection
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        // Only handle if not using modifiers
        if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
          event.preventDefault();
          this.selectDimension(parseInt(event.key) - 1); // Convert to 0-based index
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
   * Toggle dimension sliders visibility
   */
  private toggleDimensionSliders(): void {
    if (this.dimensionSliders) {
      this.dimensionSliders.toggle();
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
      document.documentElement.requestFullscreen().catch((err) => {
        console.error('Error attempting to enable fullscreen:', err);
        // Fallback: try the canvas element
        this.sceneManager.renderer.domElement.requestFullscreen().catch((fallbackErr) => {
          console.error('Fallback fullscreen also failed:', fallbackErr);
        });
      });
    } else {
      // Exit fullscreen
      document.exitFullscreen().catch((err) => {
        console.error('Error attempting to exit fullscreen:', err);
      });
    }
  }

  /**
   * Toggle advanced rendering controls
   */
  private toggleRenderingControls(): void {
    this.renderingControls?.toggle();
  }

  /**
   * Check if space key should trigger fullscreen
   */
  private shouldHandleSpaceKey(): boolean {
    const activeElement = document.activeElement;
    return (
      activeElement === document.body || activeElement === this.sceneManager.renderer.domElement
    );
  }

  /**
   * Handles keyboard navigation through nD dimensions using [ and ] keys.
   * 
   * This implements intelligent dimension navigation with adaptive step sizes:
   * - Discrete dimensions step by their defined increment
   * - Continuous dimensions step by 1% of their total range
   * - Steps are clamped to dimension bounds
   * - Only updates if the value actually changes
   * 
   * The navigation respects the currently selected dimension (set by number keys)
   * and provides smooth, predictable movement through nD space.
   * 
   * @param direction - Direction to navigate: -1 for backward, 1 for forward
   * @private
   */
  private handleDimensionNavigation(direction: -1 | 1): void {
    const dims = sceneDimsManager.getDims();
    const dimensionRanges = sceneDimsManager.getDimensionRanges();
    if (!dims || !dimensionRanges) return;

    const navigableDims = this.getNavigableDimensionsList(dims);
    if (navigableDims.length === 0) return;

    // Target the currently selected dimension (bounded by available dimensions)
    const dimIndex = Math.min(this.selectedDimension, navigableDims.length - 1);
    const targetDim = navigableDims[dimIndex];

    // Gather dimension properties for step calculation
    const currentValue = dims.currentStep[targetDim];
    const dimMeta = dims.metadata?.[targetDim];
    const [min, max] = dimensionRanges[targetDim];

    let newValue: number;
    if (dimMeta?.discrete) {
      // Discrete dimensions: step by defined increment (e.g., time frames)
      const step = dimMeta.step || 1.0;
      newValue = currentValue + direction * step;
      newValue = Math.round(newValue / step) * step; // Ensure step boundary alignment
    } else {
      // Continuous dimensions: step by 1% of range for smooth navigation
      const range = max - min;
      const step = range * 0.01;
      newValue = currentValue + direction * step;
    }

    // Apply bounds constraints
    newValue = Math.max(min, Math.min(max, newValue));

    // Update dimension state if value actually changed
    if (Math.abs(newValue - currentValue) > 1e-6) {
      sceneDimsManager.setDimensionValue(targetDim, newValue);
      
      // Trigger visual update
      this.animationController.startAnimation();
    }
  }

  /**
   * Selects which dimension to control with keyboard navigation.
   * 
   * Number keys (1-9) map to navigable dimensions, allowing users to
   * switch between controlling different non-displayed dimensions with
   * the [ and ] navigation keys.
   * 
   * @param index - Zero-based dimension index to select
   * @private
   */
  private selectDimension(index: number): void {
    const dims = sceneDimsManager.getDims();
    if (!dims) return;

    const navigableDims = this.getNavigableDimensionsList(dims);

    if (index < navigableDims.length) {
      this.selectedDimension = index;
      // Dimension is now selected for [ ] navigation
    } else {
      console.log(
        `Dimension ${index + 1} not available (only ${navigableDims.length} non-displayed dimensions)`
      );
    }
  }

  /**
   * Get list of navigable (non-displayed) dimensions
   */
  private getNavigableDimensionsList(dims: SimpleDims): number[] {
    const navigable: number[] = [];
    for (let i = 0; i < dims.ndim; i++) {
      if (!dims.displayed.includes(i)) {
        navigable.push(i);
      }
    }
    return navigable;
  }

  /**
   * Find all nD point clouds in the scene
   */
  private findNDPointClouds(): THREE.Points[] {
    const nDPoints: THREE.Points[] = [];

    this.sceneManager.scene.traverse((object) => {
      if (object instanceof THREE.Points && object.userData.originalPositions) {
        nDPoints.push(object);
      }
    });

    return nDPoints;
  }

  /**
   * Clean up all event listeners
   */
  dispose(): void {
    // Dispose dimension sliders
    if (this.dimensionSliders) {
      this.dimensionSliders.dispose();
      this.dimensionSliders = undefined;
    }

    // Clean up event listeners
    this.eventListeners.forEach((cleanup) => cleanup());
    this.eventListeners = [];
  }
}
