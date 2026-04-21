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
 * - Ctrl+wheel adjusts field of view, Shift+wheel rolls the view axis
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
import { DimensionAnimationManager } from '../scene/dimension-animation-manager';
import { RenderingControls } from '../ui/rendering-controls';
import type { RecordingPanel } from '../ui/recording-panel';
import type { LayersPanel } from '../ui/layers';
import type { ScaleBar } from '../ui/components/scale-bar';
import type { ColormapLegend } from '../ui/components/colormap-legend';
import type { OverlayManager } from '../ui/overlay-manager';
import { showHelpOverlay, hideHelpOverlay, clearError, showToast } from '../ui/helpers';
import { config } from '../config';
import { captureViewerState } from '../config/viewer-state-capture';
import { SimpleDims } from '../types/dims';
import { DimensionSliders } from '../ui/dimension-sliders';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { DebugConsole } from '../ui/debug-console';
import { InputContextManager, InputContext } from './input-context-manager';
import {
  getNonDisplayedDimensions,
  calculateStepSize,
  calculateNextPosition,
  mapKeyToDimension,
} from './input-handler-utils';
import { log, Modules, LogEmoji } from '../utils/log';
import { updateSceneForDimensions, cycleDataMonitor, hideDataMonitor } from '../data';

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

  /** Optional reference to scale bar overlay */
  private scaleBar?: ScaleBar;

  /** Optional reference to colormap legend overlay */
  private colormapLegend?: ColormapLegend;

  /** Optional reference to recording panel */
  private recordingPanel?: RecordingPanel;

  /** Optional reference to layers panel */
  private layersPanel?: LayersPanel;

  /** Optional reference to overlay manager */
  private overlayManager?: OverlayManager;

  /** Index of currently selected dimension for keyboard navigation */
  private selectedDimension: number = 0;

  /** UI component for interactive dimension sliders */
  private dimensionSliders?: DimensionSliders;

  /** Animation manager for dimension playback */
  private animationManager?: DimensionAnimationManager;

  /** Debug console for capturing browser console output */
  private debugConsole: DebugConsole;

  /** Input context manager for handling keyboard conflicts */
  private contextManager: InputContextManager;

  /**
   * Create a new input handler for nD visualization interaction.
   *
   * Sets up the complete input handling infrastructure including context
   * management and debug console. Does not register event listeners until
   * init() is called.
   *
   * The input handler coordinates keyboard, mouse, and touch input across
   * the entire application, managing conflicts between different UI contexts
   * (navigation, typing, fly controls, etc.).
   *
   * @param sceneManager - Scene management system providing access to THREE.js
   *                       scene, camera, controls, and renderer
   * @param animationController - Animation loop coordinator for triggering
   *                              re-renders after input events
   *
   * @example
   * ```typescript
   * const sceneManager = new SceneManager(canvas);
   * const animController = new AnimationController(sceneManager);
   * const inputHandler = new InputHandler(sceneManager, animController);
   *
   * // Initialize event listeners
   * inputHandler.init();
   *
   * // Later, when scene loads, initialize dimension navigation
   * await sceneManager.loadSceneData(url);
   * inputHandler.initDimensionSliders();
   * ```
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
   * Associate rendering controls for post-processing and visual effects.
   *
   * Enables keyboard shortcuts (R, C) to toggle rendering controls panel
   * and cinematic mode. Should be called after rendering controls are
   * created. Optional - rendering controls integration is not required
   * for basic functionality.
   *
   * @param controls - Rendering controls UI component providing access to
   *                   bloom, HDR, noise, and other post-processing effects
   *
   * @example
   * ```typescript
   * const renderingControls = new RenderingControls(sceneManager);
   * inputHandler.setRenderingControls(renderingControls);
   *
   * // Now 'R' key toggles rendering controls panel
   * // Now 'C' key toggles cinematic mode
   * ```
   */
  setRenderingControls(controls: RenderingControls): void {
    this.renderingControls = controls;
  }

  setScaleBar(scaleBar: ScaleBar): void {
    this.scaleBar = scaleBar;
  }

  setColormapLegend(legend: ColormapLegend): void {
    this.colormapLegend = legend;
  }

  setRecordingPanel(panel: RecordingPanel): void {
    this.recordingPanel = panel;
  }

  setOverlayManager(manager: OverlayManager): void {
    this.overlayManager = manager;
  }

  setLayersPanel(panel: LayersPanel): void {
    this.layersPanel = panel;
  }

  /**
   * Initialize all event listeners for user interaction.
   *
   * Sets up the complete input handling system including:
   * - Window events (resize, wheel, keyboard, fullscreen)
   * - Control events (orbit/fly control integration)
   * - User interaction events (mousedown, touchstart)
   * - Context-specific key bindings
   *
   * Must be called once during application initialization, after scene manager
   * is created but before scene loading. Event listeners are automatically
   * cleaned up when dispose() is called.
   *
   * @example
   * ```typescript
   * const app = new LuxarApp();
   * const inputHandler = new InputHandler(sceneManager, animController);
   *
   * // Initialize input system
   * inputHandler.init();
   *
   * // Input handlers are now active
   * // User can press H for help, P for performance, etc.
   * ```
   */
  init(): void {
    this.setupWindowEvents();
    this.setupControlEvents();
    this.setupUserInteractionEvents();
  }

  /**
   * Clear dimension UI and reset dimension manager to initial state.
   *
   * Disposes of dimension sliders and resets the scene dimension manager.
   * Used when loading a new scene to ensure clean state. The dimension
   * manager is reset to allow it to be reinitialized with new scene metadata.
   *
   * This is called automatically before loading a new scene. You typically
   * don't need to call this manually unless implementing custom scene
   * switching logic.
   *
   * @example
   * ```typescript
   * // Before loading new scene
   * inputHandler.clearDimensionUI();
   * await sceneManager.loadSceneData(newUrl);
   * inputHandler.initDimensionSliders();
   * ```
   */
  clearDimensionUI(): void {
    // Dispose of existing dimension sliders
    if (this.dimensionSliders) {
      this.dimensionSliders.dispose();
      this.dimensionSliders = undefined;
    }

    // Dispose of animation manager
    if (this.animationManager) {
      this.animationManager.dispose();
      this.animationManager = undefined;
    }

    // Reset the scene dimension manager
    sceneDimsManager.reset();

    // Reset selected dimension
    this.selectedDimension = 0;
  }

  /**
   * Initialize dimension navigation UI after scene loading completes.
   *
   * This method must be called after the scene is fully loaded and dimension
   * metadata is available. It sets up:
   * - Scene dimension manager integration with scene metadata
   * - Interactive dimension sliders UI for non-displayed dimensions
   * - Reactive update system for all nD objects (points, lines, splats)
   * - Keyboard navigation targets ([/] keys and number keys 1-9)
   *
   * The initialization process ensures all nD objects share the same
   * dimensional coordinate system and respond consistently to navigation.
   * If no nD objects are found in the scene, initialization is skipped
   * gracefully (3D-only scene).
   *
   * @example
   * ```typescript
   * // After scene loads
   * await sceneManager.loadSceneData(url);
   *
   * // Initialize dimension navigation
   * inputHandler.initDimensionSliders();
   *
   * // Now users can:
   * // - Press 1-9 to select dimension
   * // - Press [ ] to navigate selected dimension
   * // - Use sliders to navigate visually
   * ```
   *
   * @example
   * ```typescript
   * // Check if dimension sliders were created
   * inputHandler.initDimensionSliders();
   *
   * const dims = sceneDimsManager.getDims();
   * if (!dims) {
   *   console.log('No nD objects in scene (3D only)');
   * } else {
   *   console.log(`Navigating ${dims.ndim}D dataset`);
   * }
   * ```
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

    // Initialize animation manager and register keyboard shortcuts
    this.initAnimationManager();

    // Pass animation manager to dimension sliders and recording panel
    if (this.animationManager) {
      this.dimensionSliders?.setAnimationManager(this.animationManager);
      this.recordingPanel?.setAnimationManager(this.animationManager);
    }

    // Listen for dimension changes (returns Promise for animation synchronization)
    sceneDimsManager.addListener(async () => {
      // Update sliders immediately (sync UI feedback)
      if (this.dimensionSliders) {
        this.dimensionSliders.update();
      }
      // Trigger animation to render the changes
      this.animationController.startAnimation();
      // Await data loading - this allows animation to synchronize
      await this.updateAllNDNodes();
    });

    // Trigger initial update now that listener is registered
    // This ensures data loads at the correct initial slice position
    this.updateAllNDNodes();
    this.animationController.startAnimation();
  }

  /**
   * Show the dimension sliders panel (if it exists).
   * Called from viewer_config application.
   */
  showDimensionSliders(): void {
    this.dimensionSliders?.setVisible(true);
  }

  /**
   * Initialize animation manager and register keyboard shortcuts
   * Called from initDimensionSliders() after scene loads
   * @private
   */
  private initAnimationManager(): void {
    if (!this.animationManager) {
      this.animationManager = new DimensionAnimationManager(
        sceneDimsManager,
        this.animationController
      );

      // Register animation shortcuts
      this.registerAnimationShortcuts();
    }
  }

  /**
   * Get the actual dimension index from the selected position.
   * Converts from position in navigable dimensions list to actual dimension index.
   *
   * @returns Dimension index, or -1 if no dimension selected
   * @private
   */
  private getSelectedDimensionIndex(): number {
    if (this.selectedDimension < 0) {
      return -1;
    }
    const dims = sceneDimsManager.getDims();
    if (!dims) {
      return -1;
    }
    const navigableDims = this.getNavigableDimensionsList(dims);
    if (this.selectedDimension >= navigableDims.length) return -1;
    return navigableDims[this.selectedDimension];
  }

  /**
   * Register dimension animation keyboard shortcuts
   * Uses InputContextManager for proper context handling
   * @private
   */
  private registerAnimationShortcuts(): void {
    // K - Toggle play/pause
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'k',
      handler: () => {
        const dimIndex = this.getSelectedDimensionIndex();
        if (dimIndex >= 0 && this.animationManager) {
          const isPlaying = this.animationManager.togglePlay(dimIndex);
          log.info(Modules.ANIMATION, `Dimension ${dimIndex} ${isPlaying ? 'playing' : 'paused'}`);
        }
      },
      preventDefault: true,
      description: 'Toggle dimension animation (K)',
    });

    // Home - Jump to start
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'Home',
      handler: () => {
        const dimIndex = this.getSelectedDimensionIndex();
        if (dimIndex >= 0) {
          const ranges = sceneDimsManager.getDimensionRanges();
          if (ranges) {
            sceneDimsManager.setDimensionValue(dimIndex, ranges[dimIndex][0]);
            log.info(Modules.ANIMATION, `Jumped to start of dimension ${dimIndex}`);
          }
        }
      },
      preventDefault: true,
      description: 'Jump to dimension start (Home)',
    });

    // End - Jump to end
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'End',
      handler: () => {
        const dimIndex = this.getSelectedDimensionIndex();
        if (dimIndex >= 0) {
          const ranges = sceneDimsManager.getDimensionRanges();
          if (ranges) {
            sceneDimsManager.setDimensionValue(dimIndex, ranges[dimIndex][1]);
            log.info(Modules.ANIMATION, `Jumped to end of dimension ${dimIndex}`);
          }
        }
      },
      preventDefault: true,
      description: 'Jump to dimension end (End)',
    });

    // Shift+Up - Increase speed
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'ArrowUp',
      modifiers: { shift: true },
      handler: () => {
        const dimIndex = this.getSelectedDimensionIndex();
        if (dimIndex >= 0 && this.animationManager) {
          this.animationManager.increaseSpeed(dimIndex);
          const fps = this.animationManager.getState(dimIndex)?.targetFPS;
          log.info(Modules.ANIMATION, `Increased speed to ${fps} FPS`);
        }
      },
      preventDefault: true,
      description: 'Increase animation speed (Shift+↑)',
    });

    // Shift+Down - Decrease speed
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'ArrowDown',
      modifiers: { shift: true },
      handler: () => {
        const dimIndex = this.getSelectedDimensionIndex();
        if (dimIndex >= 0 && this.animationManager) {
          this.animationManager.decreaseSpeed(dimIndex);
          const fps = this.animationManager.getState(dimIndex)?.targetFPS;
          log.info(Modules.ANIMATION, `Decreased speed to ${fps} FPS`);
        }
      },
      preventDefault: true,
      description: 'Decrease animation speed (Shift+↓)',
    });

    log.success(Modules.ANIMATION, 'Animation keyboard shortcuts registered');
  }

  /**
   * Update all nD nodes (points, lines, splats) with current dimension values.
   *
   * Called automatically when dimension slice positions change. Updates ALL
   * nD-aware data nodes in the scene by:
   * - Querying spatial indices for visible chunks in current slice
   * - Loading necessary data chunks from cache/HTTP
   * - Updating point positions, colors, and other attributes
   * - Triggering re-render to display new data
   *
   * This is the core of nD navigation - it translates dimension changes into
   * data updates. The update is asynchronous because it may need to fetch
   * data over the network.
   *
   * @private
   * @returns Promise that resolves when all nD nodes have been updated and
   *          data loading is complete (or in progress)
   *
   * @example
   * ```typescript
   * // Called automatically by dimension change listener:
   * sceneDimsManager.addListener(() => {
   *   this.updateAllNDNodes();  // Update data for new slice
   *   this.animationController.startAnimation();  // Re-render
   * });
   * ```
   */
  private async updateAllNDNodes(): Promise<void> {
    const dims = sceneDimsManager.getDims();
    if (!dims) {
      return;
    }

    // Use the new loader architecture's update mechanism
    await updateSceneForDimensions(dims, this.sceneManager.scene as unknown as THREE.Group);

    // Trigger re-render after update
    this.animationController.startAnimation();
  }

  /**
   * Set up window-level event listeners for global input handling.
   *
   * Registers listeners for:
   * - Window resize: Updates canvas size and camera aspect ratio
   * - Mouse wheel: Zoom and FOV control (Ctrl+wheel for FOV, Shift+wheel for roll)
   * - Keyboard: All keyboard shortcuts and navigation
   * - Fullscreen changes: Adjusts canvas styling for fullscreen mode
   *
   * All listeners are bound to class instance and stored for cleanup.
   * Called once during init().
   *
   * @private
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

    // Register all key bindings with context manager
    this.registerAllKeyBindings();

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
   * Set up event listeners for THREE.js orbit controls.
   *
   * Registers listeners on the controls object to trigger animation
   * when user interacts with camera controls (orbit, pan, zoom).
   * Ensures smooth rendering during camera manipulation.
   *
   * @private
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
   * Set up user interaction event listeners for canvas.
   *
   * Registers listeners for mousedown and touchstart on the canvas
   * to trigger animation when user begins interaction. Provides
   * visual feedback that system is responding to input.
   *
   * @private
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
   * Handle window resize events.
   *
   * Updates canvas size, camera aspect ratio, and renderer dimensions
   * when browser window is resized. Triggers re-render to display
   * resized view without distortion.
   *
   * @private
   */
  private onWindowResize(): void {
    this.sceneManager.updateSize();
    // Trigger animation to render the resized scene
    this.animationController.startAnimation();
  }

  /**
   * Handle fullscreen mode enter/exit events.
   *
   * Adjusts canvas styling when entering or exiting fullscreen to ensure
   * proper display. Adds multiple size update passes to handle browser
   * transition timing issues. Sets background color for aesthetic fullscreen
   * experience.
   *
   * Fullscreen is triggered by Space key (when not focused on UI element).
   *
   * @private
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

    // Single resize after browser has applied fullscreen layout.
    // Modern browsers fire fullscreenchange after the transition completes,
    // so one rAF is sufficient to capture final dimensions.
    requestAnimationFrame(() => {
      this.sceneManager.updateSize();
      this.animationController.startAnimation();
    });
  }

  /**
   * Handle mouse wheel events for zoom and FOV control.
   *
   * Normal wheel: Zoom in/out via orbit controls
   * Ctrl+wheel: Adjust field of view (wide angle vs telephoto)
   * (Shift+wheel is used for view-axis rotation in orbit/ortho modes)
   *
   * FOV changes update rendering controls display if active, switching
   * preset to "Custom" since FOV was manually adjusted.
   *
   * @param event - Wheel event with deltaY for scroll direction/amount
   * @private
   */
  private onWheel(event: WheelEvent): void {
    this.animationController.startAnimation();

    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      this.sceneManager.updateFOV(event.deltaY);

      // Update rendering controls display if available
      if (this.renderingControls) {
        // Ctrl+wheel FOV change should switch to Custom preset
        this.renderingControls.settings.fovPreset = 'Custom';
        this.renderingControls.syncCurrentState();
      }
    }
  }

  /**
   * Handle cycling the data loading monitor (hidden → mini → expanded → hidden).
   *
   * Extracted to a method to support binding registration.
   *
   * @private
   */
  private handleDataMonitorCycle(): void {
    cycleDataMonitor();
    log.info(Modules.DATA_MONITOR, 'Data loading monitor cycled');
  }

  /**
   * Register all key bindings with the context manager.
   *
   * This method registers all keyboard shortcuts using the binding registration
   * system. Bindings are organized by input context:
   * - NAVIGATION: Default orbit mode shortcuts
   * - FLY_CONTROLS: WASD movement keys for fly mode
   * - All contexts: Passthrough allows global shortcuts to work everywhere
   *
   * Called during init() to set up the complete keyboard interface.
   *
   * @private
   */
  private registerAllKeyBindings(): void {
    // ===== NAVIGATION CONTEXT BINDINGS =====
    // These work in the default orbit navigation mode

    // Ctrl/Cmd key - disable zoom while held so Ctrl+scroll only adjusts FOV.
    // Use a counter so releasing one key while the other is held doesn't re-enable zoom.
    let fovKeyHeldCount = 0;
    for (const key of ['Control', 'Meta']) {
      this.contextManager.registerBinding(InputContext.NAVIGATION, {
        key,
        handler: () => {
          fovKeyHeldCount++;
          this.sceneManager.controls.setEnableZoom(false);
        },
        keyupHandler: () => {
          fovKeyHeldCount = Math.max(0, fovKeyHeldCount - 1);
          if (fovKeyHeldCount === 0) {
            this.sceneManager.controls.setEnableZoom(true);
          }
        },
        description: 'FOV control (hold Ctrl/Cmd + scroll to adjust field of view)',
      });
    }

    // Dimension navigation
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: '[',
      handler: () => this.handleDimensionNavigation(-1),
      preventDefault: true,
      description: 'Navigate dimension backward',
    });

    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: ']',
      handler: () => this.handleDimensionNavigation(1),
      preventDefault: true,
      description: 'Navigate dimension forward',
    });

    // Dimension selection (keys 1-9, only without modifiers)
    for (let i = 1; i <= 9; i++) {
      this.contextManager.registerBinding(InputContext.NAVIGATION, {
        key: String(i),
        handler: (event) => {
          if (!event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey) {
            event.preventDefault();
            this.selectDimension(i - 1);
          }
        },
        preventDefault: false,
        description: `Select dimension ${i}`,
      });
    }

    // Help overlay
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'h',
      handler: () => this.toggleHelp(),
      preventDefault: true,
      description: 'Toggle help overlay',
    });

    // Dimension sliders
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'n',
      handler: () => this.toggleDimensionSliders(),
      preventDefault: true,
      description: 'Toggle dimension sliders',
    });

    // Dataset browser
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'o',
      handler: () => window.dispatchEvent(new CustomEvent('open-dataset-browser')),
      preventDefault: true,
      description: 'Open dataset browser',
    });

    // Performance stats
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'p',
      handler: () => this.togglePerformanceStats(),
      preventDefault: true,
      description: 'Toggle performance stats',
    });

    // Rendering controls (only without modifiers)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'r',
      handler: (event) => {
        if (!event.metaKey && !event.ctrlKey && !event.shiftKey) {
          event.preventDefault();
          this.toggleRenderingControls();
        }
      },
      preventDefault: false,
      description: 'Toggle rendering controls',
    });

    // Scale bar overlay
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'b',
      handler: () => this.scaleBar?.toggle(),
      preventDefault: true,
      description: 'Toggle scale bar',
    });

    // Colormap legend overlay
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: config.input.keyboard.shortcuts.toggleColormapLegend,
      handler: () => this.colormapLegend?.toggle(),
      preventDefault: true,
      description: 'Toggle colormap legend',
    });

    // Screen-space overlays toggle
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: config.input.keyboard.shortcuts.toggleOverlays,
      handler: () => this.overlayManager?.toggle(),
      preventDefault: true,
      description: 'Toggle overlays',
    });

    // Recording panel toggle
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 't',
      handler: () => this.recordingPanel?.toggle(),
      preventDefault: true,
      description: 'Toggle recording panel',
    });

    // Quick screenshot
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'g',
      handler: () => this.recordingPanel?.captureScreenshot(),
      preventDefault: true,
      description: 'Quick screenshot',
    });

    // Layers panel (L key without modifiers).
    // If the focus is already inside the panel (e.g. on a range slider,
    // select, or bound-edit text input), swallow L so dragging sliders
    // doesn't accidentally close the panel.
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: config.input.keyboard.shortcuts.toggleLayers,
      handler: (event) => {
        if (event.metaKey || event.ctrlKey || event.shiftKey) return;
        const active = document.activeElement as HTMLElement | null;
        if (active && active.closest('.luxar-layers-panel')) return;
        event.preventDefault();
        this.layersPanel?.toggle();
      },
      preventDefault: false,
      description: 'Toggle layers panel',
    });

    // Debug console (Ctrl+L)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'l',
      modifiers: { ctrl: true },
      handler: () => {
        this.debugConsole.toggle();
        log.info(
          Modules.DEBUG_CONSOLE,
          `Debug console ${this.debugConsole.getIsVisible() ? 'opened' : 'closed'}`
        );
      },
      preventDefault: true,
      description: 'Toggle debug console',
    });

    // Data loading monitor (M key, no modifiers)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'm',
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
          event.preventDefault();
          this.handleDataMonitorCycle();
        }
      },
      preventDefault: false,
      description: 'Cycle data loading monitor',
    });

    // Recenter camera (F key, no modifiers)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'f',
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
          event.preventDefault();
          this.recenterCamera();
        }
      },
      preventDefault: false,
      description: 'Recenter camera on scene',
    });

    // Toggle control mode (V key, no modifiers)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'v',
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
          event.preventDefault();
          this.toggleControlMode();
        }
      },
      preventDefault: false,
      description: 'Cycle control mode (orbit/fly/ortho)',
    });

    // Toggle inertial mode (I key, no modifiers)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'i',
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
          event.preventDefault();
          this.toggleInertialMode();
        }
      },
      preventDefault: false,
      description: 'Toggle inertial mode (fly controls)',
    });

    // Toggle cinematic mode (C key, no modifiers)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'c',
      handler: (event) => {
        if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
          event.preventDefault();
          this.toggleCinematicMode();
        }
      },
      preventDefault: false,
      description: 'Toggle cinematic mode',
    });

    // Fullscreen toggle (Space, context-aware)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: ' ',
      handler: (event) => {
        if (this.shouldHandleSpaceKey()) {
          event.preventDefault();
          this.toggleFullscreen();
        }
      },
      preventDefault: false,
      description: 'Toggle fullscreen',
    });

    // Escape key - context-aware panel closing
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 'Escape',
      handler: () => this.handleEscapeKey(),
      preventDefault: true,
      description: 'Close panels / Exit fullscreen',
    });

    // Export viewer state (Ctrl+Shift+S)
    this.contextManager.registerBinding(InputContext.NAVIGATION, {
      key: 's',
      modifiers: { ctrl: true, shift: true },
      handler: (event) => {
        event.preventDefault();
        this.exportViewerState();
      },
      preventDefault: true,
      description: 'Export viewer state to clipboard',
    });

    // ===== FLY CONTROLS CONTEXT BINDINGS =====
    // These are active when in fly mode (WASD movement)
    // Fly controls must work with ANY modifiers:
    // - Shift: Speed boost
    // - Alt: Vertical movement (W/S only)
    // - Shift+Alt: Fast vertical movement

    // Get fly controls reference once
    const getFlyControls = () => this.sceneManager.controls.getFlyControls();

    // WASD movement keys - register with all relevant modifier combinations
    // Need both keydown (start movement) and keyup (stop movement) handlers
    const flyMovementKeys = ['w', 'a', 's', 'd', 'q', 'e'];
    const modifierCombinations = [
      {}, // No modifiers
      { shift: true }, // Shift only (speed boost)
      { alt: true }, // Alt only (vertical for W/S)
      { shift: true, alt: true }, // Shift+Alt (fast vertical)
    ];

    for (const key of flyMovementKeys) {
      for (const modifiers of modifierCombinations) {
        this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
          key,
          modifiers: Object.keys(modifiers).length > 0 ? modifiers : undefined,
          handler: (event) => getFlyControls()?.handleKeyDown(event),
          keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
          description: `Fly: ${key.toUpperCase()}${
            modifiers.shift ? '+Shift' : ''
          }${modifiers.alt ? '+Alt' : ''}`,
        });
      }
    }

    // Arrow keys for look direction (with and without Shift)
    const arrowKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
    for (const key of arrowKeys) {
      // Base arrow key
      this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
        key,
        handler: (event) => getFlyControls()?.handleKeyDown(event),
        keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
        description: `Fly look: ${key}`,
      });

      // Arrow + Shift (potentially faster look)
      this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
        key,
        modifiers: { shift: true },
        handler: (event) => getFlyControls()?.handleKeyDown(event),
        keyupHandler: (event) => getFlyControls()?.handleKeyUp(event),
        description: `Fly look: ${key}+Shift`,
      });
    }

    // Shift key in fly mode - also used for speed boost
    this.contextManager.registerBinding(InputContext.FLY_CONTROLS, {
      key: 'Shift',
      handler: () => this.sceneManager.controls.setEnableZoom(false),
      keyupHandler: () => this.sceneManager.controls.setEnableZoom(true),
      description: 'Speed boost + zoom control',
    });
  }

  /**
   * Handle key down events.
   *
   * Routes ALL keyboard events through the context manager's binding system.
   * No special cases - everything uses the unified binding system.
   */
  private onKeyDown(event: KeyboardEvent): void {
    // Check if typing in input field (belt-and-suspenders with context manager)
    if (this.isTypingInInput()) {
      return;
    }

    // Route ALL keys through context manager (including Shift, fly controls, etc.)
    this.contextManager.handleKeyEvent(event, 'down');
  }

  /**
   * Handle key up events.
   *
   * Routes through context manager for keys that have keyupHandler registered.
   * Keys without keyupHandler (most toggle actions) are ignored on keyup.
   */
  private onKeyUp(event: KeyboardEvent): void {
    // Route through context manager
    // Only bindings with keyupHandler will execute (Shift, fly controls)
    // Toggle actions (C, R, etc.) don't have keyupHandler so they're ignored
    this.contextManager.handleKeyEvent(event, 'up');
  }

  /**
   * Toggle help overlay visibility on/off.
   *
   * Shows or hides the keyboard shortcuts help overlay. Triggered by H key.
   * The help overlay displays all available keyboard shortcuts organized
   * by category (navigation, view controls, panels, etc.).
   *
   * @private
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
   * Toggle dimension sliders panel visibility.
   *
   * Shows or hides the nD dimension navigation sliders. Triggered by N key.
   * Only functional if dimension sliders have been initialized (nD dataset loaded).
   * Has no effect for 3D-only datasets.
   *
   * @private
   */
  private toggleDimensionSliders(): void {
    if (this.dimensionSliders) {
      this.dimensionSliders.toggle();
    }
  }

  /**
   * Toggle performance statistics (FPS, memory) display.
   *
   * Shows or hides the stats.js performance monitor in top-left corner.
   * Triggered by P key. Displays:
   * - FPS (frames per second)
   * - Frame time in milliseconds
   * - Memory usage (if available)
   *
   * @private
   */
  private togglePerformanceStats(): void {
    this.animationController.performanceStats.toggle();
  }

  /**
   * Toggle fullscreen mode on/off.
   *
   * Requests fullscreen for the document element (true fullscreen including
   * browser chrome). Falls back to canvas-only fullscreen if document
   * fullscreen fails. Triggered by Space key (when not focused on UI element).
   *
   * Fullscreen exit is also possible via browser's native ESC key handling.
   *
   * @private
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
   * Toggle advanced rendering controls panel visibility.
   *
   * Shows or hides the rendering controls UI providing access to:
   * - Post-processing effects (bloom, HDR, vignette, chromatic aberration)
   * - Camera settings (FOV presets)
   * - Control mode selection (orbit, fly, ortho)
   * - Point rendering parameters
   *
   * Triggered by R key. Only functional if rendering controls have been
   * associated via setRenderingControls().
   *
   * @private
   */
  private toggleRenderingControls(): void {
    this.renderingControls?.toggle();
  }

  /**
   * Toggle cinematic mode (film-like visual effects).
   *
   * Enables or disables a preset combination of effects:
   * - Film grain noise
   * - Vignette (darkened corners)
   * - Chromatic aberration (color fringing)
   * - Lens distortion
   *
   * Triggered by C key. Provides quick access to cinematic aesthetics without
   * manually adjusting individual effects. Only functional if rendering controls
   * have been associated.
   *
   * @private
   */
  private toggleCinematicMode(): void {
    this.renderingControls?.toggleCinematicMode();
  }

  /**
   * Export the complete viewer state as JSON to the clipboard.
   *
   * Triggered by Ctrl+Shift+S. Captures all rendering settings, camera state,
   * dimensions, theme, etc. and copies the JSON to the clipboard. The JSON
   * can be loaded in Python with `luxar.ViewerConfig.from_json()`.
   *
   * @private
   */
  private exportViewerState(): void {
    if (!this.renderingControls) {
      log.warning(Modules.INPUT, 'Cannot export state: rendering controls not available');
      return;
    }

    const state = captureViewerState(
      this.sceneManager,
      this.renderingControls,
      sceneDimsManager,
      this.animationManager
    );

    const json = JSON.stringify(state, null, 2);

    // Copy to clipboard
    navigator.clipboard
      .writeText(json)
      .then(() => {
        showToast('Viewer state copied to clipboard');
        log.info(Modules.INPUT, 'Viewer state exported to clipboard');
      })
      .catch((err) => {
        log.error(Modules.INPUT, 'Failed to copy state to clipboard:', err);
        showToast('Failed to copy state to clipboard');
      });

    // Also store on debug interface for programmatic access
    if ((window as any).__luxarDebug) {
      (window as any).__luxarDebug.lastExportedState = state;
    }
  }

  /**
   * Cycle through camera control modes: Orbit → Fly → Ortho → Orbit.
   *
   * Triggered by V key. Control modes provide different camera interaction styles:
   * - Orbit: Quaternion-based rotation with no gimbal lock (drag to rotate around target)
   * - Fly: First-person WASD movement (for exploring inside datasets)
   * - Ortho: Orthographic pan + zoom (for 2D viewing)
   *
   * Updates input context when switching to fly mode to enable WASD keys.
   * Syncs rendering controls display if active.
   *
   * @private
   */
  private toggleControlMode(): void {
    const currentType = this.sceneManager.controls.getControlType();
    let newType: 'orbit' | 'fly' | 'ortho';

    log.custom(LogEmoji.CONTROLS, Modules.INPUT, `toggleControlMode called: ${currentType} → ?`);

    // Cycle through: orbit -> fly -> ortho -> orbit
    switch (currentType) {
      case 'orbit':
        newType = 'fly';
        break;
      case 'fly':
        newType = 'ortho';
        break;
      case 'ortho':
        newType = 'orbit';
        break;
      default:
        newType = 'orbit';
    }

    // Use sceneManager.setControlType for ortho (handles camera swap)
    this.sceneManager.setControlType(newType);

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
   * Toggle inertial mode for fly controls (momentum-based movement).
   *
   * Triggered by I key. Only functional when in fly control mode.
   *
   * Inertial mode adds physics-based momentum:
   * - ON: Movement continues after releasing keys (space-like float)
   * - OFF: Movement stops immediately when keys released (FPS-like control)
   *
   * Syncs rendering controls display if active. Logs info message if called
   * while not in fly mode.
   *
   * @private
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
   * Check if Space key should trigger fullscreen toggle.
   *
   * Returns true only if focus is on document body or canvas, preventing
   * fullscreen toggle when user is interacting with UI elements (buttons,
   * inputs, etc.) where Space might have other meanings (submit, type space).
   *
   * @returns true if Space key should toggle fullscreen, false otherwise
   * @private
   */
  private shouldHandleSpaceKey(): boolean {
    const activeElement = document.activeElement;
    return (
      activeElement === document.body || activeElement === this.sceneManager.renderer.domElement
    );
  }

  /**
   * Check if user is currently typing in a text input field.
   *
   * Checks if focus is in an input, textarea, select, or contenteditable
   * element. Used to prevent navigation shortcuts from interfering with
   * text entry. For example, prevents [ ] keys from navigating dimensions
   * when user is typing in a search box.
   *
   * @returns true if user is typing in text field, false otherwise
   * @private
   */
  private isTypingInInput(): boolean {
    const activeElement = document.activeElement;
    if (!activeElement) return false;

    const tagName = activeElement.tagName.toLowerCase();
    // Check if it's an input field or contenteditable element
    // Exclude non-text input types (range sliders, checkboxes, radios) that don't capture typing
    if (tagName === 'input') {
      const inputType = (activeElement as HTMLInputElement).type?.toLowerCase();
      if (inputType === 'range' || inputType === 'checkbox' || inputType === 'radio') {
        return false;
      }
      return true;
    }
    return (
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
    const isCyclic = dimMeta?.cyclic || false; // Respect cyclic flag from metadata
    const newValue = calculateNextPosition(
      currentValue,
      direction,
      stepSize,
      [min, max],
      dimMeta?.discrete,
      isCyclic // Enable wrap-around for cyclic dimensions
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
    if (!dims) {
      return;
    }

    // mapKeyToDimension maps key (index+1) to the N-th navigable dimension
    const dimIndex = mapKeyToDimension((index + 1).toString(), dims);

    if (dimIndex >= 0) {
      // index is already the 0-based navigable position (key 1 → index 0, etc.)
      this.selectedDimension = index;
    } else {
      const navigableDims = this.getNavigableDimensionsList(dims);
      log.info(
        Modules.INPUT,
        `Dimension ${index + 1} not available (only ${navigableDims.length} non-displayed dimensions)`
      );
    }
  }

  /**
   * Get list of navigable (non-displayed) dimension indices.
   *
   * Delegates to the extracted utility function getNonDisplayedDimensions.
   * Returns dimensions that are not part of the 3D spatial view and can be
   * controlled with keyboard navigation.
   *
   * @param dims - Dimension configuration
   * @returns Array of non-displayed dimension indices
   * @private
   */
  private getNavigableDimensionsList(dims: SimpleDims): number[] {
    return getNonDisplayedDimensions(dims);
  }

  /**
   * Handle Escape key with context-aware behavior.
   *
   * Behavior depends on fullscreen state:
   * - If IN fullscreen: Does nothing (browser handles fullscreen exit natively)
   * - If NOT in fullscreen: Closes all open panels (help, controls, monitor, etc.)
   *
   * This ensures Escape behaves predictably - fullscreen exit takes priority,
   * then panel closing.
   *
   * @private
   */
  private handleEscapeKey(): void {
    // If recording video, stop recording first (takes priority)
    if (this.recordingPanel?.isCurrentlyRecording()) {
      this.recordingPanel.stopVideoRecording();
      return;
    }

    // Only close panels if we're NOT in fullscreen
    // When in fullscreen, the browser handles ESC to exit fullscreen
    if (!document.fullscreenElement) {
      this.closeAllPanels();
    }
  }

  /**
   * Close all open UI panels and overlays.
   *
   * Closes in priority order (topmost first):
   * 1. Help overlay
   * 2. Dataset browser
   * 3. Rendering controls
   * 4. Data loading monitor
   * 5. Dimension sliders
   * 6. Debug console
   * 7. Performance stats
   *
   * Used by Escape key handling to provide clean "exit all UI" behavior.
   *
   * @private
   */
  private closeAllPanels(): void {
    // Close all open panels (starting with topmost)
    // Close help overlay (usually topmost) - use hideHelpOverlay to clean up click listener
    hideHelpOverlay();

    // Close error messages
    clearError();

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
    hideDataMonitor();

    // Close dimension sliders
    if (this.dimensionSliders?.getIsVisible()) {
      this.dimensionSliders.hide();
    }

    // Close debug console
    if (this.debugConsole.getIsVisible()) {
      this.debugConsole.hide();
    }

    // Close recording panel
    if (this.recordingPanel?.isVisible()) {
      this.recordingPanel.hide();
    }

    // Close performance stats
    const statsElement = document.querySelector('.stats') as HTMLElement;
    if (statsElement && statsElement.style.display !== 'none') {
      this.animationController.performanceStats.hide();
    }
  }

  /**
   * Frame camera to fit the entire scene.
   *
   * Triggered by F key. Computes bounding box of all visible geometry
   * and repositions camera at the optimal distance to see everything.
   * Works for both perspective (distance) and orthographic (zoom) cameras.
   *
   * @private
   */
  private recenterCamera(): void {
    // Use the scene manager's centerCameraOnScene() which properly computes
    // optimal camera distance AND orbit target (not just the pivot point).
    // This ensures F key actually zooms to fit the whole scene, not just
    // re-centers the orbit pivot at the same distance.
    this.sceneManager.centerCameraOnScene();

    // Ensure a render happens after reframing
    this.animationController.startAnimation();
  }

  /**
   * Clean up all event listeners and dispose of managed resources.
   *
   * Removes all registered event listeners from window, document, and canvas
   * to prevent memory leaks. Disposes of dimension sliders and debug console.
   * Should be called when the input handler is no longer needed (e.g., when
   * destroying the application).
   *
   * After calling dispose(), the input handler cannot be reused - create a
   * new instance if needed.
   *
   * @example
   * ```typescript
   * // During application teardown
   * inputHandler.dispose();
   * sceneManager.dispose();
   * animationController.dispose();
   * ```
   */
  dispose(): void {
    // Dispose dimension sliders
    if (this.dimensionSliders) {
      this.dimensionSliders.dispose();
      this.dimensionSliders = undefined;
    }

    // Dispose animation manager
    if (this.animationManager) {
      this.animationManager.dispose();
      this.animationManager = undefined;
    }

    // Dispose debug console
    this.debugConsole.dispose();

    // Clean up event listeners
    this.eventListeners.forEach((cleanup) => cleanup());
    this.eventListeners = [];
  }
}
