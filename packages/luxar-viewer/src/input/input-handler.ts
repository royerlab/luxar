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
import type { OverlayManager } from '../ui/helpers/overlay-manager';
import { notifier } from '../utils/notifier';
import { captureViewerState } from '../config/viewer-state-capture';
import { DimensionSliders } from '../ui/panels/dimension-sliders';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import type { DebugConsole } from '../ui/panels/debug-console';
import type { PerformanceMonitor } from '../ui/monitors/performance-monitor';
import { InputContextManager, InputContext } from './input-context-manager';
import { computeDimensionStep, resolveSelectedDimension } from './handlers/dimension-navigation';
import { PanelCoordinator } from './handlers/panel-coordinator';
import { WindowEventHandler } from './handlers/window-event-handler';
import { AnimationShortcuts } from './handlers/animation-shortcuts';
import { registerAllKeyBindings } from './handlers/key-bindings';
import { log, Modules, LogEmoji } from '../utils/log';
import { updateSceneForDimensions } from '../data';
import { eventBus } from '../utils/event-bus';

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
   * Panel-coordination concern: owns the priority-ordered "close all
   * panels" flow used by Escape. Constructed in the InputHandler ctor
   * once `debugConsole` and `animationController` are available.
   */
  private panelCoordinator: PanelCoordinator;

  /**
   * Window-event concern: owns resize / wheel / fullscreenchange.
   * Keyboard listeners stay in InputHandler — they're a separate
   * concern coordinating with InputContextManager.
   */
  private windowEvents: WindowEventHandler;

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
    private animationController: AnimationController,
    private performanceMonitor: PerformanceMonitor,
    debugConsole: DebugConsole
  ) {
    // DebugConsole is now constructed at the app level (Phase 8.6.c)
    // and passed in here, so InputHandler doesn't need to import the
    // class — keeps the input → ui layer-cruiser rule clean.
    this.debugConsole = debugConsole;

    // Initialize input context manager
    this.contextManager = new InputContextManager();

    // Wire the panel coordinator with the always-present panels.
    // Optional panels (renderingControls, dimensionSliders, recordingPanel)
    // are pushed in via setRenderingControls / initDimensionSliders /
    // setRecordingPanel as they're created.
    this.panelCoordinator = new PanelCoordinator({
      debugConsole: this.debugConsole,
      performanceStats: this.performanceMonitor,
    });

    this.windowEvents = new WindowEventHandler(
      this.sceneManager,
      this.animationController
    );
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
    this.panelCoordinator.setRenderingControls(controls);
    this.windowEvents.setRenderingControls(controls);
  }

  setScaleBar(scaleBar: ScaleBar): void {
    this.scaleBar = scaleBar;
  }

  setColormapLegend(legend: ColormapLegend): void {
    this.colormapLegend = legend;
  }

  setRecordingPanel(panel: RecordingPanel): void {
    this.recordingPanel = panel;
    this.panelCoordinator.setRecordingPanel(panel);
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
      this.panelCoordinator.setDimensionSliders(undefined);
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
    this.panelCoordinator.setDimensionSliders(this.dimensionSliders);

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

      // Register animation shortcuts via the dedicated AnimationShortcuts
      // concern. The context callbacks read instance state at dispatch
      // time so subsequent dim selections / animation-manager swaps are
      // picked up automatically.
      const shortcuts = new AnimationShortcuts(this.contextManager, {
        getSelectedDimension: () => this.selectedDimension,
        getAnimationManager: () => this.animationManager,
      });
      shortcuts.register();
    }
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
    // Window-level resize / wheel / fullscreenchange — owned by
    // WindowEventHandler. Keyboard stays here because it has to
    // coordinate with InputContextManager and the registered
    // key-binding table.
    this.windowEvents.attach(this.eventListeners);

    const onKeyDown = this.onKeyDown.bind(this);
    const onKeyUp = this.onKeyUp.bind(this);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // Register all key bindings with context manager
    this.registerAllKeyBindings();

    this.eventListeners.push(
      () => window.removeEventListener('keydown', onKeyDown),
      () => window.removeEventListener('keyup', onKeyUp)
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
   * Handle cycling the data loading monitor (hidden → mini → expanded → hidden).
   *
   * Extracted to a method to support binding registration.
   *
   * @private
   */
  private handleDataMonitorCycle(): void {
    eventBus.emit('panel-cycle', { panelId: 'data-monitor' });
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
    registerAllKeyBindings({
      contextManager: this.contextManager,
      sceneManager: this.sceneManager,
      debugConsole: this.debugConsole,
      cleanups: this.eventListeners,
      panels: {
        getScaleBar: () => this.scaleBar,
        getColormapLegend: () => this.colormapLegend,
        getOverlayManager: () => this.overlayManager,
        getRecordingPanel: () => this.recordingPanel,
        getLayersPanel: () => this.layersPanel,
      },
      commands: {
        navigateDimension: (direction) => this.handleDimensionNavigation(direction),
        selectDimension: (index) => this.selectDimension(index),
        toggleHelp: () => this.toggleHelp(),
        toggleDimensionSliders: () => this.toggleDimensionSliders(),
        togglePerformanceStats: () => this.togglePerformanceStats(),
        toggleRenderingControls: () => this.toggleRenderingControls(),
        toggleControlMode: () => this.toggleControlMode(),
        toggleInertialMode: () => this.toggleInertialMode(),
        toggleCinematicMode: () => this.toggleCinematicMode(),
        toggleFullscreen: () => this.toggleFullscreen(),
        cycleDataMonitor: () => this.handleDataMonitorCycle(),
        recenterCamera: () => this.recenterCamera(),
        exportViewerState: () => this.exportViewerState(),
        handleEscape: () => this.handleEscapeKey(),
        shouldHandleSpaceKey: () => this.shouldHandleSpaceKey(),
      },
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
    const helpOverlay = document.getElementById('luxar-help-overlay');
    if (helpOverlay) {
      notifier.hideHelp();
    } else {
      notifier.showHelp();
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
    this.performanceMonitor.toggle();
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
        notifier.toast('Viewer state copied to clipboard');
        log.info(Modules.INPUT, 'Viewer state exported to clipboard');
      })
      .catch((err) => {
        log.error(Modules.INPUT, 'Failed to copy state to clipboard:', err);
        notifier.toast('Failed to copy state to clipboard');
      });

    // Also store on debug interface for programmatic access
    if (window.__luxarDebug) {
      window.__luxarDebug.lastExportedState = state;
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
    const flyControls = this.sceneManager.controls.getFlyControls();
    if (flyControls) {
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
    const step = computeDimensionStep(
      direction,
      this.selectedDimension,
      sceneDimsManager.getDims(),
      sceneDimsManager.getDimensionRanges()
    );
    if (!step || !step.changed) return;

    sceneDimsManager.setDimensionValue(step.targetDim, step.newValue);
    this.animationController.startAnimation();
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
    const result = resolveSelectedDimension(index, sceneDimsManager.getDims());
    if (result.selectedDimension !== null) {
      this.selectedDimension = result.selectedDimension;
    } else if ('navigableCount' in result) {
      log.info(
        Modules.INPUT,
        `Dimension ${index + 1} not available (only ${result.navigableCount} non-displayed dimensions)`
      );
    }
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
  /**
   * Escape-key dispatch. Delegates to PanelCoordinator which owns the
   * recording-priority and fullscreen-defer rules.
   */
  private handleEscapeKey(): void {
    this.panelCoordinator.handleEscape();
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
