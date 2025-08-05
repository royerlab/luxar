// Input event handling for the Luxar scene player

import * as THREE from 'three';
import { SceneManager } from './scene-manager';
import { AnimationController } from './animation-controller';
import { RenderingControls } from './rendering-controls';
import { showHelpOverlay, hideHelpOverlay } from './ui';
import { stepDimension, updatePointCloudSlice } from './utils/dims-navigation';
import { SimpleDims } from './types/dims';

export class InputHandler {
  private eventListeners: (() => void)[] = [];
  private renderingControls?: RenderingControls;
  private selectedDimension: number = 0; // Currently selected non-displayed dimension

  constructor(
    private sceneManager: SceneManager,
    private animationController: AnimationController
  ) {}

  /**
   * Set the rendering controls instance
   */
  setRenderingControls(controls: RenderingControls): void {
    this.renderingControls = controls;
  }

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
   * Handle dimension navigation with [ and ] keys
   */
  private handleDimensionNavigation(direction: -1 | 1): void {
    // Find all nD point clouds in the scene
    const nDPointClouds = this.findNDPointClouds();
    if (nDPointClouds.length === 0) return;

    // Get scene dimensions if available
    const sceneDimensions = this.sceneManager.scene.userData.sceneDimensions;

    // Update each nD point cloud
    nDPointClouds.forEach((points) => {
      const { dims, dimensionRanges, originalNumPoints } = points.userData;
      const navigableDims = this.getNavigableDimensionsList(dims);

      // Check if we have any non-displayed dimensions
      if (navigableDims.length === 0) return;

      // Use the selected dimension, bounded by available dimensions
      const dimIndex = Math.min(this.selectedDimension, navigableDims.length - 1);
      const targetDim = navigableDims[dimIndex];

      // Get step size from scene dimensions if available
      let navigationOptions: any = {};
      if (sceneDimensions?.dimensions?.[targetDim]) {
        const dimDef = sceneDimensions.dimensions[targetDim];
        if (dimDef.step !== null && dimDef.step !== undefined) {
          navigationOptions.absoluteStep = dimDef.step;
          console.log(`Using scene-defined step size: ${dimDef.step} for ${dimDef.name}`);
        }
      }

      // Step through the dimension
      const changed = stepDimension(dims, targetDim, direction, dimensionRanges, navigationOptions);

      if (changed) {
        // Update the geometry
        updatePointCloudSlice(
          points,
          points.userData.originalPositions,
          points.userData.originalColors,
          points.userData.originalRadii,
          points.userData.originalSharpness,
          dims,
          originalNumPoints
        );

        // Show feedback about which dimension is being navigated
        const dimName = dims.metadata?.[targetDim]?.name || `Dimension ${targetDim}`;
        console.log(
          `Navigating ${dimName} (dim ${targetDim}): ${dims.currentStep[targetDim].toFixed(2)}`
        );

        // Trigger animation
        this.animationController.startAnimation();
      }
    });
  }

  /**
   * Select which dimension to control with number keys
   */
  private selectDimension(index: number): void {
    // Find nD point clouds to validate dimension exists
    const nDPointClouds = this.findNDPointClouds();
    if (nDPointClouds.length === 0) return;

    // Check if this dimension index is valid
    const firstCloud = nDPointClouds[0];
    const { dims } = firstCloud.userData;
    const navigableDims = this.getNavigableDimensionsList(dims);

    if (index < navigableDims.length) {
      this.selectedDimension = index;
      const targetDim = navigableDims[index];
      const dimName = dims.metadata?.[targetDim]?.name || `Dimension ${targetDim}`;
      console.log(
        `Selected ${dimName} (dim ${targetDim}) for navigation. Current value: ${dims.currentStep[targetDim].toFixed(2)}`
      );
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
      if (object instanceof THREE.Points && object.userData.dims) {
        nDPoints.push(object);
      }
    });

    return nDPoints;
  }

  /**
   * Clean up all event listeners
   */
  dispose(): void {
    this.eventListeners.forEach((cleanup) => cleanup());
    this.eventListeners = [];
  }
}
