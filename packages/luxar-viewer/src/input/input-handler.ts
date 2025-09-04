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
 * - P key toggles performance statistics
 * - M key cycles data loading monitor
 * - R key toggles rendering controls
 * - H key shows/hides help overlay
 * - Ctrl+L toggles debug console
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
import { SimpleDims } from '../types/dims';
import { DimensionSliders } from '../ui/dimension-sliders';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { DebugConsole } from '../ui/debug-console';
import { InputContextManager, InputContext } from './input-context-manager';
import { config } from '../config';
import {
  getNonDisplayedDimensions,
  calculateStepSize,
  calculateNextPosition,
  mapKeyToDimension,
} from './input-handler-utils';
import { log, Modules, LogEmoji } from '../utils/log';

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

  /** Debug console for capturing browser console output */
  private debugConsole: DebugConsole;

  /** Input context manager for handling keyboard conflicts */
  private contextManager: InputContextManager;

  /**
   * Constructs the input handler with required system dependencies.
   *
   * @param sceneManager - Scene management system
   * @param animationController - Animation and rendering coordination
   */
  constructor(
    private sceneManager: SceneManager,
    private animationController: AnimationController
  ) {
    // Initialize debug console
    this.debugConsole = new DebugConsole();

    // Initialize input context manager
    this.contextManager = new InputContextManager();
  }

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
      this.updateAllNDPoints();
      if (this.dimensionSliders) {
        this.dimensionSliders.update();
      }
      // Trigger animation to render the changes
      this.animationController.startAnimation();
    });
  }

  /**
   * Update all nD points with current dimension values
   */
  private async updateAllNDPoints(): Promise<void> {
    const dims = sceneDimsManager.getDims();
    if (!dims) {
      return;
    }

    // Use the new loader architecture's update mechanism
    const { updateSceneForDimensions } = await import('../data');
    await updateSceneForDimensions(dims, this.sceneManager.scene as unknown as THREE.Group);

    // Trigger re-render after update
    this.animationController.startAnimation();
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

    // Register context-specific key bindings
    this.registerKeyBindings();

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
    // Trigger animation to render the resized scene
    this.animationController.startAnimation();
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
    } else {
      // Exiting fullscreen - completely clear all inline styles
      canvas.removeAttribute('style');
      document.documentElement.style.backgroundColor = '';
    }

    // Wait for fullscreen transition to complete before updating
    // This prevents intermediate size updates that can confuse the renderer
    setTimeout(() => {
      // Update canvas size after fullscreen transition
      this.sceneManager.updateSize();

      // Restart animation to ensure smooth transition
      this.animationController.startAnimation();

      // One more update to catch any final adjustments
      setTimeout(() => {
        this.sceneManager.updateSize();
        // Ensure animation continues for the final update
        this.animationController.startAnimation();
      }, 100);
    }, 200); // Wait 200ms for transition to complete
  }

  /**
   * Handle mouse wheel events (zoom and FOV control)
   */
  private onWheel(event: WheelEvent): void {
    this.animationController.startAnimation();

    if (event.shiftKey) {
      event.preventDefault();
      this.sceneManager.updateFOV(event.deltaY);

      // Update rendering controls display if available
      if (this.renderingControls) {
        // Shift+wheel FOV change should switch to Custom preset
        (this.renderingControls as any).settings.fovPreset = 'Custom';
        this.renderingControls.syncCurrentState();
      }
    }
  }

  /**
   * Register all key bindings with the context manager
   */
  private registerKeyBindings(): void {
    // No need to register fly control bindings here
    // They will be handled directly when fly mode is active
  }

  /**
   * Handle key down events
   */
  private onKeyDown(event: KeyboardEvent): void {
    // Check if fly controls are active and should handle this key
    const flyControls = this.sceneManager.controls.getFlyControls();
    if (flyControls && flyControls.enabled) {
      const flyKeys = [
        ...config.input.keyboard.flyModeKeys,
        'ArrowUp',
        'ArrowDown',
        'ArrowLeft',
        'ArrowRight',
      ];
      const keyLower = event.key.toLowerCase();

      // Check if this is a fly control key (including Alt/Option modifier for W/S)
      const isFlyKey =
        flyKeys.some((k) => k.toLowerCase() === keyLower) || event.key.startsWith('Arrow');

      if (isFlyKey) {
        // Don't prevent default if typing in input
        if (!this.isTypingInInput()) {
          flyControls.handleKeyDown(event);
          return;
        }
      }
    }

    // Handle remaining keys that aren't context-specific
    switch (event.key) {
      case 'Shift':
        this.sceneManager.controls.setEnableZoom(false);
        break;

      case 'h':
      case 'H':
        event.preventDefault();
        this.toggleHelp();
        break;

      case 'c':
      case 'C':
        // C key for cinematic mode toggle
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !this.isTypingInInput()) {
          event.preventDefault();
          this.toggleCinematicMode();
        }
        break;

      case 'n':
      case 'N':
        // N key for nD dimension sliders
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
        // P key to toggle performance metrics
        event.preventDefault();
        this.togglePerformanceStats();
        break;

      case 'r':
      case 'R':
        // R key to toggle rendering controls (only when pressed alone)
        // Ignore if Cmd/Ctrl or Shift are held to avoid conflicts with browser shortcuts
        if (!event.metaKey && !event.ctrlKey && !event.shiftKey) {
          event.preventDefault();
          this.toggleRenderingControls();
        }
        break;

      // 'C' key removed - use 'F' to recenter on bounding box instead

      case 'l':
      case 'L':
        // Ctrl+L to toggle debug console
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          this.debugConsole.toggle();
          log.info(
            Modules.DEBUG_CONSOLE,
            `Debug console ${this.debugConsole.getIsVisible() ? 'opened' : 'closed'}`
          );
        }
        break;

      case 'm':
      case 'M':
        // M key to cycle data loading monitor (hidden → mini → expanded → hidden)
        // Only handle if not using modifiers and not typing in input
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !this.isTypingInInput()) {
          event.preventDefault();
          import('../data').then(({ cycleDataMonitor }) => {
            cycleDataMonitor();
            log.info(Modules.DATA_MONITOR, 'Data loading monitor cycled');
          });
        }
        break;

      case 'f':
      case 'F':
        // F key to recenter/focus camera on scene bounding box center
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !this.isTypingInInput()) {
          event.preventDefault();
          this.recenterCamera();
        }
        break;

      case 'v':
      case 'V':
        // V key to cycle through Orbit, Arcball, and Fly control modes
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !this.isTypingInInput()) {
          event.preventDefault();
          this.toggleControlMode();
        }
        break;

      case 'i':
      case 'I':
        // I key to toggle inertial mode (when in fly mode)
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey && !this.isTypingInInput()) {
          event.preventDefault();
          this.toggleInertialMode();
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

      case 'Escape':
        // Handle ESC with priority: fullscreen first, then panels
        event.preventDefault();
        this.handleEscapeKey();
        break;
    }
  }

  /**
   * Handle key up events
   */
  private onKeyUp(event: KeyboardEvent): void {
    // Check fly controls first
    const flyControls = this.sceneManager.controls.getFlyControls();
    if (flyControls && flyControls.enabled) {
      const flyKeys = [
        ...config.input.keyboard.flyModeKeys,
        'ArrowUp',
        'ArrowDown',
        'ArrowLeft',
        'ArrowRight',
      ];
      const keyLower = event.key.toLowerCase();

      // Check if this is a fly control key
      const isFlyKey =
        flyKeys.some((k) => k.toLowerCase() === keyLower) || event.key.startsWith('Arrow');

      if (isFlyKey) {
        flyControls.handleKeyUp(event);
        return;
      }
    }

    if (event.key === 'Shift') {
      this.sceneManager.controls.setEnableZoom(true);
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
        log.error(Modules.INPUT, 'Error attempting to enable fullscreen:', err);
        // Fallback: try the canvas element
        this.sceneManager.renderer.domElement.requestFullscreen().catch((fallbackErr) => {
          log.error(Modules.INPUT, 'Fallback fullscreen also failed:', fallbackErr);
        });
      });
    } else {
      // Exit fullscreen
      document.exitFullscreen().catch((err) => {
        log.error(Modules.INPUT, 'Error attempting to exit fullscreen:', err);
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
   * Toggle cinematic mode - enables/disables noise, vignette, chromatic aberration, and lens distortion
   */
  private toggleCinematicMode(): void {
    this.renderingControls?.toggleCinematicMode();
  }

  /**
   * Cycle through Orbit, Arcball, and Fly control modes
   */
  private toggleControlMode(): void {
    const currentType = this.sceneManager.controls.getControlType();
    let newType: 'orbit' | 'arcball' | 'fly';

    // Cycle through: orbit -> arcball -> fly -> orbit
    switch (currentType) {
      case 'orbit':
        newType = 'arcball';
        break;
      case 'arcball':
        newType = 'fly';
        break;
      case 'fly':
        newType = 'orbit';
        break;
      default:
        newType = 'orbit';
    }

    this.sceneManager.controls.setControlType(newType);

    // Update input context based on control mode
    if (newType === 'fly') {
      this.contextManager.setContext(InputContext.FLY_CONTROLS);
    } else {
      this.contextManager.setContext(InputContext.NAVIGATION);
    }

    // Sync rendering controls if they exist
    if (this.renderingControls) {
      this.renderingControls.syncCurrentState();
    }

    log.custom(
      LogEmoji.CONTROLS,
      Modules.CONTROLS,
      `Switched to ${newType} controls (press V to toggle)`
    );
  }

  /**
   * Toggle inertial mode for fly controls
   */
  private toggleInertialMode(): void {
    const controls = this.sceneManager.controls.getControls();
    if (controls && 'setInertialMode' in controls) {
      const flyControls = controls as any; // Type assertion for fly controls
      const currentInertial = flyControls.inertialMode;
      flyControls.setInertialMode(!currentInertial);

      // Sync rendering controls if they exist
      if (this.renderingControls) {
        this.renderingControls.syncCurrentState();
      }

      log.custom(
        LogEmoji.ROCKET,
        Modules.CONTROLS,
        `Fly controls inertial mode: ${!currentInertial ? 'ON' : 'OFF'}`
      );
    } else {
      log.info(
        Modules.INPUT,
        'Inertial mode is only available in fly control mode (press V to switch)'
      );
    }
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
   * Check if user is typing in an input field
   */
  private isTypingInInput(): boolean {
    const activeElement = document.activeElement;
    if (!activeElement) return false;

    const tagName = activeElement.tagName.toLowerCase();
    // Check if it's an input field or contenteditable element
    return (
      tagName === 'input' ||
      tagName === 'textarea' ||
      tagName === 'select' ||
      activeElement.getAttribute('contenteditable') === 'true'
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

    // Use utility functions for step calculation and navigation
    const stepSize = calculateStepSize(targetDim, dims);
    const newValue = calculateNextPosition(
      currentValue,
      direction,
      stepSize,
      [min, max],
      dimMeta?.discrete,
      false // no wrap-around
    );

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

    // Use utility to map key to actual dimension index
    const dimIndex = mapKeyToDimension((index + 1).toString(), dims);

    if (dimIndex >= 0) {
      // Find which position this is in the non-displayed list
      const navigableDims = this.getNavigableDimensionsList(dims);
      const position = navigableDims.indexOf(dimIndex);
      if (position >= 0) {
        this.selectedDimension = position;
        // Dimension is now selected for [ ] navigation
      }
    } else {
      const navigableDims = this.getNavigableDimensionsList(dims);
      log.info(
        Modules.INPUT,
        `Dimension ${index + 1} not available (only ${navigableDims.length} non-displayed dimensions)`
      );
    }
  }

  /**
   * Get list of navigable (non-displayed) dimensions.
   * Delegates to the extracted utility function.
   */
  private getNavigableDimensionsList(dims: SimpleDims): number[] {
    return getNonDisplayedDimensions(dims);
  }

  /**
   * Handle ESC key:
   * - If in fullscreen: do nothing (browser handles fullscreen exit)
   * - If not in fullscreen: close all panels
   */
  private handleEscapeKey(): void {
    // Only close panels if we're NOT in fullscreen
    // When in fullscreen, the browser handles ESC to exit fullscreen
    if (!document.fullscreenElement) {
      this.closeAllPanels();
    }
  }

  /**
   * Close all open panels (helper for ESC key handling)
   */
  private closeAllPanels(): void {
    // Close all open panels (starting with topmost)
    // Close help overlay (usually topmost)
    const helpOverlay = document.getElementById('help-overlay');
    if (helpOverlay) {
      helpOverlay.remove();
    }

    // Close dataset browser
    const datasetBrowser = document.getElementById('dataset-browser');
    if (datasetBrowser) {
      datasetBrowser.remove();
    }

    // Close rendering controls
    if (this.renderingControls?.isVisible()) {
      this.renderingControls.hide();
    }

    // Close data loading monitor
    import('../data').then(({ hideDataMonitor }) => {
      hideDataMonitor();
    });

    // Close dimension sliders
    if (this.dimensionSliders?.getIsVisible()) {
      this.dimensionSliders.hide();
    }

    // Close debug console
    if (this.debugConsole.getIsVisible()) {
      this.debugConsole.hide();
    }

    // Close performance stats
    const statsElement = document.querySelector('.stats') as HTMLElement;
    if (statsElement && statsElement.style.display !== 'none') {
      this.animationController.performanceStats.hide();
    }
  }

  /**
   * Recenter/focus camera on the scene's bounding box center
   * Uses smooth animation for fly controls, immediate for orbit/arcball controls
   */
  private recenterCamera(): void {
    // Compute bounding box center of all visible objects
    const box = new THREE.Box3();

    this.sceneManager.scene.traverse((object) => {
      if (object instanceof THREE.Points && object.visible) {
        const geometry = object.geometry;

        // For points, compute bounding box from position attribute
        const positions = geometry.attributes.position;
        if (positions && positions.count > 0) {
          // Compute the bounding box if needed
          if (!geometry.boundingBox) {
            geometry.computeBoundingBox();
          }

          if (geometry.boundingBox) {
            const tempBox = geometry.boundingBox.clone();
            // Apply object's world transform
            tempBox.applyMatrix4(object.matrixWorld);
            // Expand our overall box
            box.expandByObject(object);
          }
        }
      }
    });

    // Get the center of the bounding box
    const center = new THREE.Vector3();

    // Check if box is valid (has content)
    if (!box.isEmpty()) {
      box.getCenter(center);
    } else {
      // Fallback to origin if no objects found
      center.set(0, 0, 0);
    }

    // Get the controls manager if it exists
    const controlsManager = this.sceneManager.getControlsManager();

    if (controlsManager) {
      // Start animation for smooth transition
      this.animationController.startAnimation();

      // For fly controls, we need to call this repeatedly for smooth animation
      if (controlsManager.getControlType() === 'fly') {
        let iterations = 0;
        const maxIterations = 60; // About 1 second at 60fps

        const smoothRecenter = () => {
          if (iterations < maxIterations) {
            controlsManager.lookAt(center, true);
            iterations++;
            requestAnimationFrame(smoothRecenter);
          }
        };

        smoothRecenter();
        log.custom(LogEmoji.TARGET, Modules.CONTROLS, 'Recentering camera on scene (smooth)');
      } else {
        // For orbit/arcball controls, just update the target
        controlsManager.lookAt(center, false);
        log.custom(LogEmoji.TARGET, Modules.CONTROLS, 'Recentered camera on scene');
      }
    }
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

    // Dispose debug console
    this.debugConsole.dispose();

    // Clean up event listeners
    this.eventListeners.forEach((cleanup) => cleanup());
    this.eventListeners = [];
  }
}
